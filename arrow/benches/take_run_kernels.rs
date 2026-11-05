// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Benchmarks `take` on run-end encoded arrays.
//!
//! Each case takes rows of a 32,000-row run array with one shape of indices: one contiguous
//! range, dense (1 in 2) or sparse (1 in 32) ascending rows, 16,000 random rows, 16,000
//! sorted random rows (ascending with duplicates), or 16,000 random rows with 10% nulls.
//! Runs have length 1 or random lengths with mean 16 or 1,024; their values are `UInt8`
//! or `Dictionary<Int32, Utf8>`.
//!
//! Next to `take`, two reference baselines run on the same case. They are not lower bounds:
//! each skips work that a `take` must do, and `take` can beat them.
//!
//! - `baseline_stream` sums the indices once and copies the run ends and values of the
//!   expected output. It does not find the run of any index, check bounds, handle null
//!   indices or merge runs.
//! - `baseline_lookup` looks up each index's value in a logical-to-physical table built
//!   before timing, then run-encodes the values. It does not build the table, check bounds,
//!   handle null indices (it reads their backing values) or take the output values, and it
//!   allocates its output at the number of indices, so it can be slower than a `take` that
//!   copies whole runs.

use std::hint;
use std::sync::Arc;

use arrow::array::*;
use arrow::compute::take;
use arrow::datatypes::*;
use arrow::util::test_util::seedable_rng;
use arrow_buffer::ArrowNativeType;
use criterion::{Criterion, criterion_group, criterion_main};
use rand::RngExt;
use rand::rngs::StdRng;

/// Fits the largest `Int16` run end.
const LEN: usize = 32_000;

/// Runs of length 1, or of random length with mean `mean_run_len`.
fn run_ends(rng: &mut StdRng, mean_run_len: usize) -> Vec<usize> {
    let mut ends = Vec::new();
    let mut end = 0;
    while end < LEN {
        end += match mean_run_len {
            1 => 1,
            _ => rng.random_range(1..2 * mean_run_len),
        };
        ends.push(end.min(LEN));
    }
    ends
}

/// A run array whose adjacent runs hold different values.
fn run_array<R: RunEndIndexType>(ends: &[usize], values: ArrayRef) -> ArrayRef {
    let ends = ends.iter().map(|&e| R::Native::usize_as(e));
    let run_ends = PrimitiveArray::<R>::from_iter_values(ends);
    Arc::new(RunArray::<R>::try_new(&run_ends, &values).unwrap())
}

/// `UInt8` values, as in a low-cardinality attribute column.
fn u8_values(runs: usize) -> UInt8Array {
    UInt8Array::from_iter_values((0..runs).map(|i| (i % 251) as u8))
}

/// `Dictionary<Int32, Utf8>` values, as in a string attribute column. The dictionary holds
/// each string once, so the baselines compare keys to compare values.
fn dict_values(runs: usize) -> DictionaryArray<Int32Type> {
    let strings = StringArray::from_iter_values((0..64).map(|i| format!("attribute-value-{i}")));
    let keys = Int32Array::from_iter_values((0..runs).map(|i| (i % 64) as i32));
    DictionaryArray::new(keys, Arc::new(strings))
}

fn indices(rng: &mut StdRng, shape: &str) -> UInt32Array {
    let ascending = |rng: &mut StdRng, p: f64| -> Vec<u32> {
        (0..LEN as u32).filter(|_| rng.random_bool(p)).collect()
    };
    let random = |rng: &mut StdRng| -> Vec<u32> {
        (0..LEN / 2)
            .map(|_| rng.random_range(0..LEN as u32))
            .collect()
    };
    match shape {
        "contiguous" => UInt32Array::from_iter_values(LEN as u32 / 4..3 * LEN as u32 / 4),
        "dense" => ascending(rng, 0.5).into(),
        "sparse" => ascending(rng, 1.0 / 32.0).into(),
        "random" => random(rng).into(),
        "duplicates" => {
            let mut indices = random(rng);
            indices.sort_unstable();
            indices.into()
        }
        "nulls" => random(rng)
            .into_iter()
            .map(|i| rng.random_bool(0.9).then_some(i))
            .collect(),
        _ => unreachable!("{shape}"),
    }
}

/// Reads every index once and writes the output runs.
fn baseline_stream<R: Copy, V: Copy>(
    indices: &[u32],
    out_ends: &[R],
    out_values: &[V],
) -> (u32, Vec<R>, Vec<V>) {
    let sum = indices.iter().fold(0_u32, |sum, &i| sum.wrapping_add(i));
    (sum, out_ends.to_vec(), out_values.to_vec())
}

/// Looks up each index's value through `physical` and run-encodes the output without branches
/// or bounds checks.
///
/// # Safety
///
/// Every index must be in bounds for `physical`, and every physical index for `values`.
unsafe fn baseline_lookup<R: ArrowNativeType, V: Copy + PartialEq>(
    indices: &[u32],
    physical: &[u32],
    values: &[V],
) -> (Vec<R>, Vec<V>) {
    let len = indices.len();
    // SAFETY: the caller bounds every lookup.
    let value = |index: u32| unsafe {
        *values.get_unchecked(*physical.get_unchecked(index as usize) as usize)
    };
    let mut out_ends = vec![R::usize_as(len); len];
    let mut current = value(indices[0]);
    let mut out_values = vec![current; len];
    let mut run = 0;
    for (i, &index) in indices.iter().enumerate().skip(1) {
        let value = value(index);
        // SAFETY: `run <= i < len`.
        unsafe {
            *out_ends.get_unchecked_mut(run) = R::usize_as(i);
            run += usize::from(value != current);
            *out_values.get_unchecked_mut(run) = value;
        }
        current = value;
    }
    out_ends[run] = R::usize_as(len);
    out_ends.truncate(run + 1);
    out_values.truncate(run + 1);
    (out_ends, out_values)
}

/// Benchmarks `take` and its baselines. `natives` returns the values a baseline reads for an array
/// of run values: the values of a primitive array, the keys of a dictionary.
fn bench_case<R: RunEndIndexType, V: Copy + PartialEq>(
    c: &mut Criterion,
    name: &str,
    ends: &[usize],
    values: ArrayRef,
    natives: impl Fn(&dyn Array) -> Vec<V>,
    indices: &UInt32Array,
) {
    let source_values = natives(values.as_ref());
    let array = run_array::<R>(ends, values);
    let taken = take(&array, indices, None).unwrap();
    let taken = taken.as_run::<R>();
    let out_ends = taken.run_ends().values().to_vec();
    let out_values = natives(taken.values().as_ref());
    let mut physical = Vec::with_capacity(LEN);
    for (p, &end) in ends.iter().enumerate() {
        physical.resize(end, p as u32);
    }
    let indices_values = indices.values();

    c.bench_function(&format!("{name}/take"), |b| {
        b.iter(|| take(&array, indices, None).unwrap())
    });
    c.bench_function(&format!("{name}/baseline_stream"), |b| {
        b.iter(|| baseline_stream(hint::black_box(indices_values), &out_ends, &out_values))
    });
    c.bench_function(&format!("{name}/baseline_lookup"), |b| {
        assert!(
            indices_values
                .iter()
                .all(|&i| (i as usize) < physical.len())
        );
        assert!(physical.iter().all(|&p| (p as usize) < source_values.len()));
        // SAFETY: the asserts above bound every lookup.
        b.iter(|| unsafe {
            baseline_lookup::<R::Native, V>(
                hint::black_box(indices_values),
                &physical,
                &source_values,
            )
        })
    });
}

fn u8_natives(array: &dyn Array) -> Vec<u8> {
    array.as_primitive::<UInt8Type>().values().to_vec()
}

fn dict_natives(array: &dyn Array) -> Vec<i32> {
    array.as_dictionary::<Int32Type>().keys().values().to_vec()
}

fn add_benchmark(c: &mut Criterion) {
    let shapes = [
        "contiguous",
        "dense",
        "sparse",
        "random",
        "duplicates",
        "nulls",
    ];
    for mean_run_len in [1, 16, 1024] {
        let mut rng = seedable_rng();
        let ends = run_ends(&mut rng, mean_run_len);
        let u8s: ArrayRef = Arc::new(u8_values(ends.len()));
        let dict: ArrayRef = Arc::new(dict_values(ends.len()));
        for shape in shapes {
            let indices = indices(&mut rng, shape);
            let name = |values: &str, run_end: &str| {
                format!("take_run {values} {run_end} run_len={mean_run_len} {shape}")
            };
            let (u8s, dict) = (u8s.clone(), dict.clone());
            if shape == "nulls" {
                bench_case::<Int32Type, _>(c, &name("u8", "i32"), &ends, u8s, u8_natives, &indices);
                continue;
            }
            bench_case::<Int16Type, _>(
                c,
                &name("u8", "i16"),
                &ends,
                u8s.clone(),
                u8_natives,
                &indices,
            );
            bench_case::<Int32Type, _>(
                c,
                &name("u8", "i32"),
                &ends,
                u8s.clone(),
                u8_natives,
                &indices,
            );
            bench_case::<Int64Type, _>(c, &name("u8", "i64"), &ends, u8s, u8_natives, &indices);
            bench_case::<Int32Type, _>(
                c,
                &name("dict", "i32"),
                &ends,
                dict,
                dict_natives,
                &indices,
            );
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(std::time::Duration::from_millis(300))
        .measurement_time(std::time::Duration::from_secs(1));
    targets = add_benchmark
}
criterion_main!(benches);

//! Bench-hash-batched: per-message digest throughput across batched APIs.
//!
//! Workloads are sized to exercise the three regimes spelled out in
//! `docs/v0.2_planning/12_HASH_BATCHED.md`:
//!
//! - Canonical Merkle-leaf workload: 200_000 messages × 1 KiB each (~200 MiB
//!   total), measuring how the batched API distributes per-message work.
//! - Small-message regime: 1_000_000 messages × 64 B each, where per-message
//!   call overhead and rayon scheduling dominate.
//! - Single-large-message regime: 1 message × 1 GiB, where the batched form
//!   should be indistinguishable from a single-shot call to the underlying
//!   hash.
//!
//! Run all: `cargo bench -p tokenfs-algos --bench hash_batched`
//! Quick-only: `cargo bench -p tokenfs-algos --bench hash_batched -- --quick`

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::hash::sha256_batch_st;

#[cfg(feature = "parallel")]
use tokenfs_algos::hash::sha256_batch_par;

#[cfg(feature = "blake3")]
use tokenfs_algos::hash::blake3_batch_st_32;

#[cfg(all(feature = "blake3", feature = "parallel"))]
use tokenfs_algos::hash::blake3_batch_par_32;

/// Generate `n_messages` random byte strings, each `msg_size` bytes long.
fn make_messages(n_messages: usize, msg_size: usize, seed: u64) -> Vec<Vec<u8>> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..n_messages)
        .map(|_| {
            let mut buf = vec![0_u8; msg_size];
            for byte in &mut buf {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *byte = (state >> 33) as u8;
            }
            buf
        })
        .collect()
}

/// Workload-set for the bench matrix.
#[derive(Clone, Copy)]
struct Workload {
    label: &'static str,
    n_messages: usize,
    msg_size: usize,
}

const WORKLOADS: &[Workload] = &[
    Workload {
        label: "merkle_200k_x_1kib",
        n_messages: 200_000,
        msg_size: 1024,
    },
    Workload {
        label: "small_1m_x_64b",
        n_messages: 1_000_000,
        msg_size: 64,
    },
    Workload {
        label: "single_1_x_1gib",
        n_messages: 1,
        msg_size: 1 << 30,
    },
];

fn bench_sha256_st(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_batched/sha256_st");
    for wl in WORKLOADS {
        let owned = make_messages(wl.n_messages, wl.msg_size, 0xA5A5_A5A5);
        let messages: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let total_bytes = wl.n_messages as u64 * wl.msg_size as u64;
        group.throughput(Throughput::Bytes(total_bytes));
        let id = BenchmarkId::from_parameter(wl.label);
        group.bench_with_input(id, &(), |bencher, _| {
            let mut out = vec![[0_u8; 32]; messages.len()];
            bencher.iter(|| {
                sha256_batch_st(black_box(&messages), black_box(&mut out));
            });
        });
    }
    group.finish();
}

#[cfg(feature = "parallel")]
fn bench_sha256_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_batched/sha256_par");
    for wl in WORKLOADS {
        let owned = make_messages(wl.n_messages, wl.msg_size, 0xA5A5_A5A5);
        let messages: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let total_bytes = wl.n_messages as u64 * wl.msg_size as u64;
        group.throughput(Throughput::Bytes(total_bytes));
        let id = BenchmarkId::from_parameter(wl.label);
        group.bench_with_input(id, &(), |bencher, _| {
            let mut out = vec![[0_u8; 32]; messages.len()];
            bencher.iter(|| {
                sha256_batch_par(black_box(&messages), black_box(&mut out));
            });
        });
    }
    group.finish();
}

#[cfg(feature = "blake3")]
fn bench_blake3_st(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_batched/blake3_st");
    for wl in WORKLOADS {
        let owned = make_messages(wl.n_messages, wl.msg_size, 0xA5A5_A5A5);
        let messages: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let total_bytes = wl.n_messages as u64 * wl.msg_size as u64;
        group.throughput(Throughput::Bytes(total_bytes));
        let id = BenchmarkId::from_parameter(wl.label);
        group.bench_with_input(id, &(), |bencher, _| {
            let mut out = vec![[0_u8; 32]; messages.len()];
            bencher.iter(|| {
                blake3_batch_st_32(black_box(&messages), black_box(&mut out));
            });
        });
    }
    group.finish();
}

#[cfg(all(feature = "blake3", feature = "parallel"))]
fn bench_blake3_par(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_batched/blake3_par");
    for wl in WORKLOADS {
        let owned = make_messages(wl.n_messages, wl.msg_size, 0xA5A5_A5A5);
        let messages: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
        let total_bytes = wl.n_messages as u64 * wl.msg_size as u64;
        group.throughput(Throughput::Bytes(total_bytes));
        let id = BenchmarkId::from_parameter(wl.label);
        group.bench_with_input(id, &(), |bencher, _| {
            let mut out = vec![[0_u8; 32]; messages.len()];
            bencher.iter(|| {
                blake3_batch_par_32(black_box(&messages), black_box(&mut out));
            });
        });
    }
    group.finish();
}

// ----------------------------------------------------------------------------
// Group registration with conditional compilation.
// ----------------------------------------------------------------------------

#[cfg(all(feature = "parallel", feature = "blake3"))]
criterion_group!(
    benches,
    bench_sha256_st,
    bench_sha256_par,
    bench_blake3_st,
    bench_blake3_par,
);

#[cfg(all(feature = "parallel", not(feature = "blake3")))]
criterion_group!(benches, bench_sha256_st, bench_sha256_par,);

#[cfg(all(not(feature = "parallel"), feature = "blake3"))]
criterion_group!(benches, bench_sha256_st, bench_blake3_st,);

#[cfg(all(not(feature = "parallel"), not(feature = "blake3")))]
criterion_group!(benches, bench_sha256_st);

criterion_main!(benches);

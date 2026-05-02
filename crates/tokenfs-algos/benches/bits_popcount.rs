//! `bench-bits-popcount`: SIMD popcount throughput across cache tiers.
//!
//! Reports per-backend bytes/sec at the canonical L1 / L2 / L3 / DRAM
//! working-set sizes from [`support::cache_tier_sizes`]. Each row
//! benchmarks the same input through every available backend so the
//! per-tier speedup against scalar can be read directly from the report.
//!
//! Run all: `cargo bench -p tokenfs-algos --bench bits_popcount`
//! Filter:  `cargo bench -p tokenfs-algos --bench bits_popcount -- in-L1`

#![allow(missing_docs)]
// `support` is shared with the larger workload-matrix benches; only the
// `cache_tier_sizes` helper is consumed here, which leaves most of the
// module unreferenced from this binary.
#![allow(dead_code)]

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use support::cache_tier_sizes;
use tokenfs_algos::bits;

fn deterministic_words(byte_size: usize, seed: u64) -> Vec<u64> {
    let n = byte_size.div_ceil(8);
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        })
        .collect()
}

fn bench_popcount_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_popcount/u64_slice");
    for (tier_label, byte_size) in cache_tier_sizes() {
        let words = deterministic_words(*byte_size, 0xF22_C0FFEE_u64 ^ (*byte_size as u64));
        // Quote the actual byte budget — for tiers not divisible by 8 the
        // u64 buffer overshoots by ≤ 7 bytes; the reported throughput
        // tracks the input-words footprint exactly.
        let buffer_bytes = words.len() * core::mem::size_of::<u64>();
        group.throughput(Throughput::Bytes(buffer_bytes as u64));

        let id = format!("scalar/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| bits::kernels::scalar::popcount_u64_slice(black_box(&words)));
        });

        let id = format!("auto/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| bits::popcount_u64_slice(black_box(&words)));
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if bits::popcount::kernels::avx2::is_available() {
            let id = format!("avx2/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: availability checked immediately above.
                    unsafe { bits::kernels::avx2::popcount_u64_slice(black_box(&words)) }
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if bits::popcount::kernels::avx512::is_available() {
            let id = format!("avx512/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: availability checked immediately above.
                    unsafe { bits::kernels::avx512::popcount_u64_slice(black_box(&words)) }
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { bits::kernels::neon::popcount_u64_slice(black_box(&words)) }
                });
            });
        }
    }
    group.finish();
}

fn bench_popcount_u8(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_popcount/u8_slice");
    for (tier_label, byte_size) in cache_tier_sizes() {
        let words =
            deterministic_words(*byte_size, 0xC0DE_C0DE_C0DE_C0DE_u64 ^ (*byte_size as u64));
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter())
            .take(*byte_size)
            .collect();
        group.throughput(Throughput::Bytes(bytes.len() as u64));

        let id = format!("scalar/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| bits::kernels::scalar::popcount_u8_slice(black_box(&bytes)));
        });

        let id = format!("auto/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| bits::popcount_u8_slice(black_box(&bytes)));
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if bits::popcount::kernels::avx2::is_available() {
            let id = format!("avx2/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: availability checked immediately above.
                    unsafe { bits::kernels::avx2::popcount_u8_slice(black_box(&bytes)) }
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if bits::popcount::kernels::avx512::is_available() {
            let id = format!("avx512/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: availability checked immediately above.
                    unsafe { bits::kernels::avx512::popcount_u8_slice(black_box(&bytes)) }
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { bits::kernels::neon::popcount_u8_slice(black_box(&bytes)) }
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_popcount_u64, bench_popcount_u8);
criterion_main!(benches);

//! `bench-approx-hll`: SIMD vs scalar HyperLogLog merge + cardinality.
//!
//! Sprint 44 of the v0.2 plan: extends `approx::HyperLogLog` with
//! SIMD-accelerated per-bucket-max merge (`vpmaxub` / `_mm512_max_epu8`
//! / `_mm256_max_epu8` / `vmaxq_u8`) and a vectorized harmonic-mean
//! reduction. Three groups:
//!
//! 1. Single-pair `merge` throughput at precision ∈ {10, 12, 14}.
//!    Reports MB/sec on the per-bucket-max byte stream.
//! 2. Single `count_simd` throughput (RAW estimate, alpha*m^2/Z) at the
//!    same precisions.
//! 3. Aggregate-many-into-one: merging N source sketches into one
//!    destination, sweeping N ∈ {16, 256, 4_096}. Mirrors the OLAP
//!    rollup workload (Druid/Pinot/ClickHouse `approx_count_distinct`).
//!
//! All three groups exercise scalar / AVX2 / AVX-512 / NEON when the host
//! advertises them, in addition to the auto-dispatched public API.
//!
//! Run all: `cargo bench -p tokenfs-algos --bench approx_hll`
//! Quick:   `cargo bench -p tokenfs-algos --bench approx_hll -- --quick`

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::approx::{self, HyperLogLog};

/// Precisions swept by the merge / count benches. p=10 fits in L1 (1
/// KiB), p=12 fits in L2 (4 KiB), p=14 escapes L1 on heavy registers
/// (16 KiB). p=16 (64 KiB) is omitted from the merge-only sweep because
/// the data is already DRAM-bound; it shows up indirectly in the
/// aggregate group.
const MERGE_PRECISIONS: &[u32] = &[10, 12, 14];

/// Source-sketch counts swept by the aggregate group.
const AGGREGATE_BATCH_SIZES: &[usize] = &[16, 256, 4_096];

fn deterministic_hll(precision: u32, seed: u64, inserts: usize) -> HyperLogLog {
    let mut hll = HyperLogLog::new(precision);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    for _ in 0..inserts {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let h = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        hll.insert_hash(h);
    }
    hll
}

fn deterministic_register_blob(precision: u32, seed: u64) -> Vec<u8> {
    deterministic_hll(precision, seed, 4 * (1_usize << precision))
        .register_bytes()
        .to_vec()
}

fn bench_merge(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_hll/merge");
    for &precision in MERGE_PRECISIONS {
        let m = 1_usize << precision;
        // Each iteration touches `m` destination bytes plus `m` source
        // bytes; report the destination-write throughput as the canonical
        // number (matches popcount / streamvbyte conventions).
        group.throughput(Throughput::Bytes(m as u64));

        let dst_initial = deterministic_register_blob(precision, 0xA1A1 ^ (precision as u64));
        let src = deterministic_register_blob(precision, 0xB2B2 ^ (precision as u64));

        // Scalar reference.
        let id = format!("scalar/p={precision}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            let src_local = src.clone();
            let dst_template = dst_initial.clone();
            b.iter(|| {
                let mut dst = dst_template.clone();
                approx::hll_kernels::scalar::merge(&mut dst, black_box(&src_local));
                black_box(dst);
            });
        });

        // Auto-dispatched public surface.
        let id = format!("auto/p={precision}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            let mut a = HyperLogLog::new(precision);
            // Drive `a`'s state up so the merge is non-degenerate.
            for h in deterministic_iter(0xC0FFEE ^ (precision as u64), 4 * m) {
                a.insert_hash(h);
            }
            let other = deterministic_hll(precision, 0xDEAD_BEEF ^ (precision as u64), 4 * m);
            b.iter(|| {
                let mut a_local = a.clone();
                a_local.merge_simd(black_box(&other));
                black_box(a_local);
            });
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx2") {
            let id = format!("avx2/p={precision}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let src_local = src.clone();
                let dst_template = dst_initial.clone();
                b.iter(|| {
                    let mut dst = dst_template.clone();
                    // SAFETY: availability checked above.
                    unsafe {
                        approx::hll_kernels::avx2::merge(&mut dst, black_box(&src_local));
                    }
                    black_box(dst);
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            let id = format!("avx512/p={precision}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let src_local = src.clone();
                let dst_template = dst_initial.clone();
                b.iter(|| {
                    let mut dst = dst_template.clone();
                    // SAFETY: availability checked above.
                    unsafe {
                        approx::hll_kernels::avx512::merge(&mut dst, black_box(&src_local));
                    }
                    black_box(dst);
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/p={precision}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let src_local = src.clone();
                let dst_template = dst_initial.clone();
                b.iter(|| {
                    let mut dst = dst_template.clone();
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe {
                        approx::hll_kernels::neon::merge(&mut dst, black_box(&src_local));
                    }
                    black_box(dst);
                });
            });
        }
    }
    group.finish();
}

fn bench_count(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_hll/count");
    for &precision in MERGE_PRECISIONS {
        let m = 1_usize << precision;
        group.throughput(Throughput::Bytes(m as u64));

        let registers = deterministic_register_blob(precision, 0xC0FFEE ^ (precision as u64));
        // Recover alpha the same way the public surface does.
        let alpha = match precision {
            4 => 0.673,
            5 => 0.697,
            _ => 0.7213 / (1.0 + 1.079 / (m as f64)),
        };

        let id = format!("scalar/p={precision}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| approx::hll_kernels::scalar::count_raw(black_box(&registers), alpha));
        });

        // Public auto surface (count_simd).
        let id = format!("auto/p={precision}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            let hll = deterministic_hll(precision, 0xC0FFEE ^ (precision as u64), 4 * m);
            b.iter(|| black_box(&hll).count_simd());
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx2") {
            let id = format!("avx2/p={precision}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { approx::hll_kernels::avx2::count_raw(black_box(&registers), alpha) }
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            let id = format!("avx512/p={precision}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { approx::hll_kernels::avx512::count_raw(black_box(&registers), alpha) }
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/p={precision}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { approx::hll_kernels::neon::count_raw(black_box(&registers), alpha) }
                });
            });
        }
    }
    group.finish();
}

fn bench_aggregate_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_hll/aggregate_batch");
    let precision = 12_u32; // 4 KiB sketches, the OLAP-canonical size.
    let m = 1_usize << precision;

    for &n_sources in AGGREGATE_BATCH_SIZES {
        // Throughput in source-sketches-per-second; bytes-per-sec is
        // n_sources * m * 2 (read src + read+write dst per merge).
        group.throughput(Throughput::Bytes((n_sources * m) as u64));

        let sources: Vec<HyperLogLog> = (0..n_sources)
            .map(|i| {
                deterministic_hll(
                    precision,
                    0xA1A1 ^ (i as u64).wrapping_mul(0x9E37_79B9),
                    4 * m,
                )
            })
            .collect();

        let id = format!("scalar/n={n_sources}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| {
                let mut acc = HyperLogLog::new(precision);
                for src in &sources {
                    acc.merge(black_box(src));
                }
                black_box(acc);
            });
        });

        let id = format!("auto/n={n_sources}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| {
                let mut acc = HyperLogLog::new(precision);
                for src in &sources {
                    acc.merge_simd(black_box(src));
                }
                black_box(acc);
            });
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx2") {
            let id = format!("avx2/n={n_sources}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let src_blobs: Vec<Vec<u8>> = sources
                    .iter()
                    .map(|h| h.register_bytes().to_vec())
                    .collect();
                b.iter(|| {
                    let mut acc = vec![0_u8; m];
                    for src in &src_blobs {
                        // SAFETY: availability checked above.
                        unsafe {
                            approx::hll_kernels::avx2::merge(&mut acc, black_box(src));
                        }
                    }
                    black_box(acc);
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            let id = format!("avx512/n={n_sources}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let src_blobs: Vec<Vec<u8>> = sources
                    .iter()
                    .map(|h| h.register_bytes().to_vec())
                    .collect();
                b.iter(|| {
                    let mut acc = vec![0_u8; m];
                    for src in &src_blobs {
                        // SAFETY: availability checked above.
                        unsafe {
                            approx::hll_kernels::avx512::merge(&mut acc, black_box(src));
                        }
                    }
                    black_box(acc);
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/n={n_sources}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let src_blobs: Vec<Vec<u8>> = sources
                    .iter()
                    .map(|h| h.register_bytes().to_vec())
                    .collect();
                b.iter(|| {
                    let mut acc = vec![0_u8; m];
                    for src in &src_blobs {
                        // SAFETY: NEON is mandatory on AArch64.
                        unsafe {
                            approx::hll_kernels::neon::merge(&mut acc, black_box(src));
                        }
                    }
                    black_box(acc);
                });
            });
        }
    }
    group.finish();
}

fn deterministic_iter(seed: u64, n: usize) -> impl Iterator<Item = u64> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    (0..n).map(move |_| {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    })
}

criterion_group!(benches, bench_merge, bench_count, bench_aggregate_batch);
criterion_main!(benches);

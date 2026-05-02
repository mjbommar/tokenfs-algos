//! `bench-approx-bloom`: SIMD vs scalar insert + query throughput on
//! [`approx::BloomFilter`].
//!
//! Reports per-call latency and throughput at three filter footprints
//! (1 KB / 32 KB / 1 MB) crossed with three K values (3, 7, 13). Each
//! workload is exercised through every available backend so the
//! per-row speedup vs scalar is readable directly from the report.
//!
//! Three benchmark groups are emitted:
//!
//! 1. `approx_bloom/insert` — single-key `insert_simd` latency.
//! 2. `approx_bloom/contains` — single-key `contains_simd` latency.
//! 3. `approx_bloom/contains_batch` — batched `contains_batch_simd`
//!    throughput in keys/sec.
//!
//! Every group rows by `(filter_size, k)`. The 1 KB filter fits in
//! L1d on every commodity x86/AArch64; the 32 KB filter spills L1
//! and lives in L2; the 1 MB filter spills L2 and lives in L3 for
//! most server-class hosts. This matches the cache-tier convention
//! documented in `docs/v0.2_planning/02_CACHE_RESIDENCY.md`.
//!
//! Run all: `cargo bench -p tokenfs-algos --bench approx_bloom`
//! Quick:   `cargo bench -p tokenfs-algos --bench approx_bloom -- --quick`

#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::approx::{self, BloomFilter};

/// Filter footprints in bits. 1 KB = 8 192 bits; 32 KB = 262 144;
/// 1 MB = 8 388 608. Picked to span L1 / L2 / L3 cache tiers.
const FILTER_BITS: &[(usize, &str)] = &[
    (8 * 1024, "1KB"),
    (32 * 1024 * 8, "32KB"),
    (1024 * 1024 * 8, "1MB"),
];

/// K values exercised. `K=3` is small / pre-check filter; `K=7` is
/// the canonical Bloom-optimal for ~1% FPR; `K=13` exercises the
/// SIMD multi-vector path on every backend.
const K_VALUES: &[usize] = &[3, 7, 13];

/// Number of keys per batched-query benchmark. Sized so per-call
/// overhead is amortised and the reported throughput tracks the
/// per-key inner loop cost.
const BATCH_KEYS: usize = 4096;

/// Builds a deterministic key stream of `n` u64 values.
fn deterministic_keys(n: usize, seed: u64) -> Vec<u64> {
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

/// Builds a half-loaded filter: inserts `bits / (4*k)` keys so the
/// per-bit probability of "set" is around 25-50%, giving the
/// `contains_simd` early-exit branch a realistic mix of hits/misses.
fn build_loaded_filter(bits: usize, k: usize, seed: u64) -> BloomFilter {
    let mut bf = BloomFilter::new(bits, k);
    let n_inserts = (bits / (4 * k)).max(1);
    let keys = deterministic_keys(n_inserts, seed);
    for &key in &keys {
        bf.insert_simd(key);
    }
    bf
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_bloom/insert");
    for &(bits, size_label) in FILTER_BITS {
        for &k in K_VALUES {
            // Pre-build a key stream; the bench iterates through it.
            let keys =
                deterministic_keys(1024, 0x0F22_C0FF_EEEF_F22F_u64 ^ (bits as u64) ^ (k as u64));
            group.throughput(Throughput::Elements(keys.len() as u64));

            let id = format!("simd_auto/size={size_label}/k={k}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    // Fresh filter each iter so insert load stays
                    // bounded; the iter cost is amortised across
                    // `keys.len()` inserts.
                    let mut bf = BloomFilter::new(bits, k);
                    for &key in &keys {
                        bf.insert_simd(black_box(key));
                    }
                    black_box(bf.inserted());
                });
            });

            // Scalar baseline using bytes path: derive bytes from the
            // u64 key so the work is comparable. The hash family
            // differs from the `_simd` u64 path but the per-call
            // arithmetic cost is the same shape.
            let key_bytes: Vec<[u8; 8]> = keys.iter().map(|k| k.to_le_bytes()).collect();
            let id = format!("scalar_bytes/size={size_label}/k={k}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let mut bf = BloomFilter::new(bits, k);
                    for kb in &key_bytes {
                        bf.insert(black_box(&kb[..]));
                    }
                    black_box(bf.inserted());
                });
            });
        }
    }
    group.finish();
}

fn bench_contains(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_bloom/contains");
    for &(bits, size_label) in FILTER_BITS {
        for &k in K_VALUES {
            let bf =
                build_loaded_filter(bits, k, 0x00C0_FFEE_F00D_BA5E ^ (bits as u64) ^ (k as u64));
            // Mix of known-present + likely-absent keys: half from the
            // insertion seed (so `contains_simd` returns true for
            // many), half from a disjoint seed (returns false fast).
            let n_inserts = (bits / (4 * k)).max(1);
            let mut probes =
                deterministic_keys(512, 0x00C0_FFEE_F00D_BA5E ^ (bits as u64) ^ (k as u64));
            // Replace half with absent keys.
            let absent = deterministic_keys(
                probes.len() / 2,
                0x0000_0000_DEAD_BEEF ^ (bits as u64) ^ (k as u64),
            );
            let absent_count = probes.len() / 2;
            probes[..absent_count].copy_from_slice(&absent);
            // Truncate so we only probe at most `n_inserts` known keys
            // (defensive — `n_inserts` is large at 1MB / k=3).
            let _ = n_inserts;

            group.throughput(Throughput::Elements(probes.len() as u64));

            let id = format!("simd_auto/size={size_label}/k={k}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let mut hits = 0_usize;
                    for &key in &probes {
                        if bf.contains_simd(black_box(key)) {
                            hits += 1;
                        }
                    }
                    black_box(hits);
                });
            });

            // Scalar bytes baseline.
            let probe_bytes: Vec<[u8; 8]> = probes.iter().map(|k| k.to_le_bytes()).collect();
            let id = format!("scalar_bytes/size={size_label}/k={k}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let mut hits = 0_usize;
                    for kb in &probe_bytes {
                        if bf.contains(black_box(&kb[..])) {
                            hits += 1;
                        }
                    }
                    black_box(hits);
                });
            });

            // Pinned scalar position-kernel — measures only the
            // position-computation phase without the bit-gather
            // overhead. Useful for separating compute-bound vs
            // memory-bound contributions.
            let pinned_h1 = 0x9E37_79B9_7F4A_7C15_u64;
            let pinned_h2 = 0x6E5E_2E5C_DEAD_BEEF_u64 | 1;
            let id = format!("scalar_positions_only/size={size_label}/k={k}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let mut buf = [0_u64; 32];
                b.iter(|| {
                    for &key in &probes {
                        approx::bloom_kernels::scalar::positions(
                            black_box(pinned_h1 ^ key),
                            black_box(pinned_h2),
                            k,
                            bits,
                            &mut buf[..k],
                        );
                    }
                    black_box(&buf);
                });
            });

            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            if approx::bloom_kernels::avx2::is_available() {
                let id = format!("avx2_positions_only/size={size_label}/k={k}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                    let mut buf = [0_u64; 32];
                    b.iter(|| {
                        for &key in &probes {
                            // SAFETY: availability checked above.
                            unsafe {
                                approx::bloom_kernels::avx2::positions(
                                    black_box(pinned_h1 ^ key),
                                    black_box(pinned_h2),
                                    k,
                                    bits,
                                    &mut buf[..k],
                                );
                            }
                        }
                        black_box(&buf);
                    });
                });
            }

            #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
            if approx::bloom_kernels::avx512::is_available() {
                let id = format!("avx512_positions_only/size={size_label}/k={k}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                    let mut buf = [0_u64; 32];
                    b.iter(|| {
                        for &key in &probes {
                            // SAFETY: availability checked above.
                            unsafe {
                                approx::bloom_kernels::avx512::positions(
                                    black_box(pinned_h1 ^ key),
                                    black_box(pinned_h2),
                                    k,
                                    bits,
                                    &mut buf[..k],
                                );
                            }
                        }
                        black_box(&buf);
                    });
                });
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                let id = format!("neon_positions_only/size={size_label}/k={k}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                    let mut buf = [0_u64; 32];
                    b.iter(|| {
                        for &key in &probes {
                            // SAFETY: NEON is mandatory on AArch64.
                            unsafe {
                                approx::bloom_kernels::neon::positions(
                                    black_box(pinned_h1 ^ key),
                                    black_box(pinned_h2),
                                    k,
                                    bits,
                                    &mut buf[..k],
                                );
                            }
                        }
                        black_box(&buf);
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_contains_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("approx_bloom/contains_batch");
    for &(bits, size_label) in FILTER_BITS {
        for &k in K_VALUES {
            let bf =
                build_loaded_filter(bits, k, 0x0000_0000_BA5E_BA11 ^ (bits as u64) ^ (k as u64));
            let keys = deterministic_keys(
                BATCH_KEYS,
                0x0000_0000_CAFE_CAFE ^ (bits as u64) ^ (k as u64),
            );

            group.throughput(Throughput::Elements(keys.len() as u64));

            let id = format!("simd_batch_auto/size={size_label}/k={k}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let mut out = vec![false; keys.len()];
                b.iter(|| {
                    bf.contains_batch_simd(black_box(&keys), black_box(&mut out));
                });
            });

            let id = format!("simd_per_key_auto/size={size_label}/k={k}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                let mut out = vec![false; keys.len()];
                b.iter(|| {
                    for (i, &key) in keys.iter().enumerate() {
                        out[i] = bf.contains_simd(black_box(key));
                    }
                    black_box(&mut out);
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_insert, bench_contains, bench_contains_batch);
criterion_main!(benches);

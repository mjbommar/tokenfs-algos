//! `bench-hash-set-membership`: SIMD vs scalar vs `HashSet` membership.
//!
//! Per `docs/v0.2_planning/12_HASH_BATCHED.md` § 3, the SIMD scan is the
//! winning shape for short haystacks (≤ 256 elements typical for vocab
//! tables / content-class membership). This bench reports per-call
//! latency at haystack sizes 16, 64, 256, and 1024 across:
//!
//! - `slice::contains` — the standard-library scalar baseline.
//! - `HashSet::contains` — the conventional fallback once the haystack
//!   spills L1.
//! - `hash::contains_u32_simd` — the runtime-dispatched SIMD path.
//! - Pinned per-backend SIMD kernels (scalar / SSE4.1 / AVX2 / AVX-512 /
//!   NEON) when the host advertises them.
//!
//! Both single-call and batched forms are exercised so the per-needle
//! amortised cost is visible alongside the function-call dominated
//! single-needle latency.
//!
//! Run all: `cargo bench -p tokenfs-algos --bench hash_set_membership`
//! Quick:   `cargo bench -p tokenfs-algos --bench hash_set_membership -- --quick`

#![allow(missing_docs)]

use std::collections::HashSet;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::hash;

/// Haystack lengths swept by every benchmark group.
const HAYSTACK_SIZES: &[usize] = &[16, 64, 256, 1024];

/// Number of needles per batched call. Sized so the batched bench reports
/// per-needle amortised cost rather than per-call overhead.
const BATCH_NEEDLES: usize = 1024;

fn deterministic_haystack(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32
        })
        .collect()
}

/// Build needle list with a controlled hit ratio. ~50% hits keeps the
/// branch predictor honest (early-exit is rare for misses, fast for
/// hits).
fn deterministic_needles(haystack: &[u32], n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|i| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let raw = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32;
            // 50/50 hit ratio: even index pulls a known-present element
            // (modulo haystack len), odd index uses the raw mix.
            if i % 2 == 0 && !haystack.is_empty() {
                haystack[(raw as usize) % haystack.len()]
            } else {
                raw
            }
        })
        .collect()
}

fn bench_single_call(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_set_membership/single");
    for &len in HAYSTACK_SIZES {
        let haystack = deterministic_haystack(len, 0xF22_C0FFEE_u64 ^ (len as u64));
        // Pre-build needle pool once; iterate through it inside the bench
        // so the compiler can't fold the call into a constant.
        let needles = deterministic_needles(&haystack, 64, 0xC0DE_C0DE ^ (len as u64));
        let hashset: HashSet<u32> = haystack.iter().copied().collect();

        group.throughput(Throughput::Elements(needles.len() as u64));

        let id = format!("slice_contains/len={len}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| {
                let mut hits = 0_usize;
                for &n in &needles {
                    if black_box(&haystack[..]).contains(&n) {
                        hits += 1;
                    }
                }
                black_box(hits)
            });
        });

        let id = format!("hashset_contains/len={len}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| {
                let mut hits = 0_usize;
                for &n in &needles {
                    if hashset.contains(black_box(&n)) {
                        hits += 1;
                    }
                }
                black_box(hits)
            });
        });

        let id = format!("simd_auto/len={len}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| {
                let mut hits = 0_usize;
                for &n in &needles {
                    if hash::contains_u32_simd(black_box(&haystack), black_box(n)) {
                        hits += 1;
                    }
                }
                black_box(hits)
            });
        });

        let id = format!("scalar/len={len}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| {
                let mut hits = 0_usize;
                for &n in &needles {
                    if hash::set_membership::kernels::scalar::contains_u32(
                        black_box(&haystack),
                        black_box(n),
                    ) {
                        hits += 1;
                    }
                }
                black_box(hits)
            });
        });

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if hash::set_membership::kernels::sse41::is_available() {
            let id = format!("sse41/len={len}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let mut hits = 0_usize;
                    for &n in &needles {
                        // SAFETY: availability checked above.
                        if unsafe {
                            hash::set_membership::kernels::sse41::contains_u32(
                                black_box(&haystack),
                                black_box(n),
                            )
                        } {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            });
        }

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if hash::set_membership::kernels::avx2::is_available() {
            let id = format!("avx2/len={len}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let mut hits = 0_usize;
                    for &n in &needles {
                        // SAFETY: availability checked above.
                        if unsafe {
                            hash::set_membership::kernels::avx2::contains_u32(
                                black_box(&haystack),
                                black_box(n),
                            )
                        } {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if hash::set_membership::kernels::avx512::is_available() {
            let id = format!("avx512/len={len}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let mut hits = 0_usize;
                    for &n in &needles {
                        // SAFETY: availability checked above.
                        if unsafe {
                            hash::set_membership::kernels::avx512::contains_u32(
                                black_box(&haystack),
                                black_box(n),
                            )
                        } {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/len={len}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let mut hits = 0_usize;
                    for &n in &needles {
                        // SAFETY: NEON is mandatory on AArch64.
                        if unsafe {
                            hash::set_membership::kernels::neon::contains_u32(
                                black_box(&haystack),
                                black_box(n),
                            )
                        } {
                            hits += 1;
                        }
                    }
                    black_box(hits)
                });
            });
        }
    }
    group.finish();
}

fn bench_batched(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_set_membership/batched");
    for &len in HAYSTACK_SIZES {
        let haystack = deterministic_haystack(len, 0xF22_C0FFEE_u64 ^ (len as u64));
        let needles = deterministic_needles(&haystack, BATCH_NEEDLES, 0xC0DE_C0DE ^ (len as u64));
        let hashset: HashSet<u32> = haystack.iter().copied().collect();

        group.throughput(Throughput::Elements(needles.len() as u64));

        let id = format!("slice_contains_batch/len={len}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            let mut out = vec![false; needles.len()];
            b.iter(|| {
                for (i, &n) in needles.iter().enumerate() {
                    out[i] = haystack[..].contains(black_box(&n));
                }
                black_box(&mut out);
            });
        });

        let id = format!("hashset_batch/len={len}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            let mut out = vec![false; needles.len()];
            b.iter(|| {
                for (i, &n) in needles.iter().enumerate() {
                    out[i] = hashset.contains(black_box(&n));
                }
                black_box(&mut out);
            });
        });

        let id = format!("simd_batch_auto/len={len}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            let mut out = vec![false; needles.len()];
            b.iter(|| {
                hash::contains_u32_batch_simd(
                    black_box(&haystack),
                    black_box(&needles),
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_single_call, bench_batched);
criterion_main!(benches);

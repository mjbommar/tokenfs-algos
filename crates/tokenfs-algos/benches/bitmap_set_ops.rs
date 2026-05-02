//! `bench-bitmap-set-ops`: Roaring container set-algebra throughput.
//!
//! Bench every primary (pair, op) combination from
//! `docs/v0.2_planning/11_BITMAP.md` § 11 across the four canonical
//! posting-list sizes (100 / 10K / 100K / 1M elements), comparing the
//! scalar oracle against the SIMD kernels.
//!
//! Each "size" parameter selects the dense element count of the inputs;
//! the actual representation (array vs bitmap container) is determined
//! by the standard 4096-element threshold, but the benches address the
//! pinned scalar / AVX2 / SSE4.2 / AVX-512 / NEON backends directly so
//! the per-backend numbers are stable across hosts.
//!
//! Run all: `cargo bench --bench bitmap_set_ops`
//! Filter:  `cargo bench --bench bitmap_set_ops -- bitmap_x_bitmap`

#![allow(missing_docs)]
#![allow(dead_code)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::bitmap;

const BITMAP_WORDS: usize = 1024;

/// Sizes (in u32 elements) that approximate dense container fill levels.
/// 100 → array container (sparse). 10K, 100K, 1M → bitmap container.
const POSTING_SIZES: &[(&str, usize)] = &[
    ("n=100", 100),
    ("n=10K", 10_000),
    ("n=100K", 100_000),
    ("n=1M", 1_000_000),
];

fn deterministic_bitmap(seed: u64) -> [u64; BITMAP_WORDS] {
    let mut bm = [0_u64; BITMAP_WORDS];
    let mut state = seed;
    for word in &mut bm {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *word = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    }
    bm
}

fn deterministic_sorted_u16(n: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    let mut last: u32 = 0;
    let mut out = Vec::with_capacity(n);
    // Step density chosen so an `n`-element list distributes across the
    // 65 536-value space.
    let step_max = (65_535_u32 / (n as u32).max(1)).max(1);
    for _ in 0..n {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let step = ((state as u32) % step_max).max(1);
        last += step;
        if last >= 65_536 {
            break;
        }
        out.push(last as u16);
    }
    out
}

fn bench_bitmap_x_bitmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitmap/bitmap_x_bitmap");
    group.throughput(Throughput::Bytes((BITMAP_WORDS * 8) as u64));

    let a = deterministic_bitmap(0xC0FF_EE00);
    let b = deterministic_bitmap(0xDEAD_BEEF);
    let mut out = [0_u64; BITMAP_WORDS];

    // Scalar paths.
    group.bench_function("scalar/and_into", |bench| {
        bench.iter(|| {
            bitmap::kernels::bitmap_x_bitmap_scalar::and_into(
                black_box(&a),
                black_box(&b),
                &mut out,
            )
        });
    });
    group.bench_function("scalar/and_cardinality", |bench| {
        bench.iter(|| {
            bitmap::kernels::bitmap_x_bitmap_scalar::and_cardinality(black_box(&a), black_box(&b))
        });
    });
    group.bench_function("scalar/or_into", |bench| {
        bench.iter(|| {
            bitmap::kernels::bitmap_x_bitmap_scalar::or_into(black_box(&a), black_box(&b), &mut out)
        });
    });
    group.bench_function("scalar/xor_into", |bench| {
        bench.iter(|| {
            bitmap::kernels::bitmap_x_bitmap_scalar::xor_into(
                black_box(&a),
                black_box(&b),
                &mut out,
            )
        });
    });
    group.bench_function("scalar/andnot_into", |bench| {
        bench.iter(|| {
            bitmap::kernels::bitmap_x_bitmap_scalar::andnot_into(
                black_box(&a),
                black_box(&b),
                &mut out,
            )
        });
    });

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    if bitmap::kernels::bitmap_x_bitmap_avx2::is_available() {
        group.bench_function("avx2/and_into", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx2::and_into(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
        group.bench_function("avx2/and_cardinality", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx2::and_cardinality(
                        black_box(&a),
                        black_box(&b),
                    )
                }
            });
        });
        group.bench_function("avx2/and_into_nocard", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx2::and_into_nocard(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
        group.bench_function("avx2/or_into", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx2::or_into(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
        group.bench_function("avx2/xor_into", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx2::xor_into(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
        group.bench_function("avx2/andnot_into", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx2::andnot_into(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
    }

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    if bitmap::kernels::bitmap_x_bitmap_avx512::is_available() {
        group.bench_function("avx512/and_into", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx512::and_into(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
        group.bench_function("avx512/and_cardinality", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx512::and_cardinality(
                        black_box(&a),
                        black_box(&b),
                    )
                }
            });
        });
        group.bench_function("avx512/and_into_nocard", |bench| {
            bench.iter(|| {
                // SAFETY: availability checked above.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_avx512::and_into_nocard(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        group.bench_function("neon/and_into", |bench| {
            bench.iter(|| {
                // SAFETY: NEON is mandatory on AArch64.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_neon::and_into(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    )
                }
            });
        });
        group.bench_function("neon/and_cardinality", |bench| {
            bench.iter(|| {
                // SAFETY: NEON is mandatory on AArch64.
                unsafe {
                    bitmap::kernels::bitmap_x_bitmap_neon::and_cardinality(
                        black_box(&a),
                        black_box(&b),
                    )
                }
            });
        });
    }

    group.finish();
}

fn bench_array_x_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitmap/array_x_array");

    for &(label, n) in &POSTING_SIZES[..2] {
        // Array containers cap at 4096 elements; benches at 100 and 10K
        // exercise the array×array kernel directly. Anything larger
        // would either truncate or auto-promote to a bitmap container.
        let n = n.min(4096);
        let a = deterministic_sorted_u16(n, 0xC0FF_EE00);
        let b = deterministic_sorted_u16(n, 0xDEAD_BEEF);
        group.throughput(Throughput::Elements((a.len() + b.len()) as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar/intersect", label),
            &(),
            |bench, _| {
                let mut out = Vec::with_capacity(a.len().min(b.len()));
                bench.iter(|| {
                    bitmap::kernels::array_x_array_scalar::intersect(
                        black_box(&a),
                        black_box(&b),
                        &mut out,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar/intersect_cardinality", label),
            &(),
            |bench, _| {
                bench.iter(|| {
                    bitmap::kernels::array_x_array_scalar::intersect_cardinality(
                        black_box(&a),
                        black_box(&b),
                    )
                });
            },
        );

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if bitmap::kernels::array_x_array_sse42::is_available() {
            group.bench_with_input(
                BenchmarkId::new("sse42/intersect", label),
                &(),
                |bench, _| {
                    let mut out = Vec::with_capacity(a.len().min(b.len()));
                    bench.iter(|| {
                        // SAFETY: availability checked above.
                        unsafe {
                            bitmap::kernels::array_x_array_sse42::intersect(
                                black_box(&a),
                                black_box(&b),
                                &mut out,
                            );
                        }
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new("sse42/intersect_cardinality", label),
                &(),
                |bench, _| {
                    bench.iter(|| {
                        // SAFETY: availability checked above.
                        unsafe {
                            bitmap::kernels::array_x_array_sse42::intersect_cardinality(
                                black_box(&a),
                                black_box(&b),
                            )
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

fn bench_array_x_bitmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("bitmap/array_x_bitmap");

    for &(label, n) in &POSTING_SIZES[..2] {
        let n = n.min(4096);
        let array = deterministic_sorted_u16(n, 0xC0FF_EE00);
        let bm = deterministic_bitmap(0xDEAD_BEEF);
        group.throughput(Throughput::Elements(array.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar/intersect", label),
            &(),
            |bench, _| {
                let mut out = Vec::with_capacity(array.len());
                bench.iter(|| {
                    bitmap::kernels::array_x_bitmap_scalar::intersect_array_bitmap(
                        black_box(&array),
                        black_box(&bm),
                        &mut out,
                    );
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("scalar/intersect_cardinality", label),
            &(),
            |bench, _| {
                bench.iter(|| {
                    bitmap::kernels::array_x_bitmap_scalar::intersect_cardinality_array_bitmap(
                        black_box(&array),
                        black_box(&bm),
                    )
                });
            },
        );

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if bitmap::kernels::array_x_bitmap_avx2::is_available() {
            group.bench_with_input(
                BenchmarkId::new("avx2/intersect", label),
                &(),
                |bench, _| {
                    let mut out = Vec::with_capacity(array.len());
                    bench.iter(|| {
                        // SAFETY: availability checked above.
                        unsafe {
                            bitmap::kernels::array_x_bitmap_avx2::intersect_array_bitmap(
                                black_box(&array),
                                black_box(&bm),
                                &mut out,
                            );
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bitmap_x_bitmap,
    bench_array_x_array,
    bench_array_x_bitmap
);
criterion_main!(benches);

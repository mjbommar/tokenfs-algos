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
//!
//! Real-data path: set `TOKENFS_ALGOS_REAL_FILES=<path1>:<path2>` to
//! synthesise additional posting lists from each file. Two-byte windows
//! become candidate doc-ids: their low 12 bits, sorted and deduplicated,
//! drive the array-container kernels; their low 16 bits set bits in a
//! 65 536-bit Roaring bitmap container. Each listed file produces an
//! extra bench row labelled `real/<file-stem>`.

#![allow(missing_docs)]
// `support` is shared with the larger workload-matrix benches; only
// `real_files_as_bytes` is consumed here.
#![allow(dead_code)]

mod support;

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

/// Real-data input shape: paired posting-list halves derived from the
/// first / second half of a file. Each 2-byte window contributes a
/// candidate doc-id; we keep the low 12 bits (sorted + deduplicated) for
/// array-container benches and the low 16 bits as set bits for bitmap-
/// container benches. Splitting on file halves keeps `a` and `b`
/// distinct enough to exercise non-trivial set-algebra paths.
struct RealPostingInput {
    label: String,
    array_a: Vec<u16>,
    array_b: Vec<u16>,
    bitmap_a: [u64; BITMAP_WORDS],
    bitmap_b: [u64; BITMAP_WORDS],
}

fn real_data_inputs() -> Vec<RealPostingInput> {
    support::real_files_as_bytes()
        .into_iter()
        .filter_map(|(label, bytes)| {
            // Need at least four bytes to form one window per half.
            if bytes.len() < 4 {
                return None;
            }
            let mid = bytes.len() / 2;
            let (head, tail) = bytes.split_at(mid);

            let array_a = sorted_unique_low12(head);
            let array_b = sorted_unique_low12(tail);
            if array_a.is_empty() || array_b.is_empty() {
                return None;
            }
            let bitmap_a = bitmap_from_u16_windows(head);
            let bitmap_b = bitmap_from_u16_windows(tail);

            Some(RealPostingInput {
                label,
                array_a,
                array_b,
                bitmap_a,
                bitmap_b,
            })
        })
        .collect()
}

/// Build a sorted, deduplicated `Vec<u16>` from the low 12 bits of every
/// 2-byte LE window in `bytes`. Caps at 4096 entries to keep the result
/// inside an array container (the 4096-element promotion threshold).
fn sorted_unique_low12(bytes: &[u8]) -> Vec<u16> {
    let mut seen = [false; 4096];
    for chunk in bytes.chunks_exact(2) {
        let value = u16::from_le_bytes([chunk[0], chunk[1]]) & 0x0FFF;
        seen[value as usize] = true;
    }
    seen.iter()
        .enumerate()
        .filter_map(|(idx, &set)| if set { Some(idx as u16) } else { None })
        .collect()
}

/// Build a `[u64; BITMAP_WORDS]` (a 65 536-bit Roaring bitmap container)
/// where each 2-byte LE window's low 16 bits sets the corresponding bit.
fn bitmap_from_u16_windows(bytes: &[u8]) -> [u64; BITMAP_WORDS] {
    let mut bm = [0_u64; BITMAP_WORDS];
    for chunk in bytes.chunks_exact(2) {
        let value = u16::from_le_bytes([chunk[0], chunk[1]]) as usize;
        let word = value / 64;
        let bit = value % 64;
        bm[word] |= 1_u64 << bit;
    }
    bm
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

    for input in &real_data_inputs() {
        let label_prefix = format!("real/{}", input.label);
        let a = input.bitmap_a;
        let b = input.bitmap_b;
        group.bench_function(format!("scalar/and_into/{label_prefix}"), |bench| {
            bench.iter(|| {
                bitmap::kernels::bitmap_x_bitmap_scalar::and_into(
                    black_box(&a),
                    black_box(&b),
                    &mut out,
                )
            });
        });
        group.bench_function(format!("scalar/and_cardinality/{label_prefix}"), |bench| {
            bench.iter(|| {
                bitmap::kernels::bitmap_x_bitmap_scalar::and_cardinality(
                    black_box(&a),
                    black_box(&b),
                )
            });
        });
        group.bench_function(format!("scalar/or_into/{label_prefix}"), |bench| {
            bench.iter(|| {
                bitmap::kernels::bitmap_x_bitmap_scalar::or_into(
                    black_box(&a),
                    black_box(&b),
                    &mut out,
                )
            });
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if bitmap::kernels::bitmap_x_bitmap_avx2::is_available() {
            group.bench_function(format!("avx2/and_into/{label_prefix}"), |bench| {
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
            group.bench_function(format!("avx2/and_cardinality/{label_prefix}"), |bench| {
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
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if bitmap::kernels::bitmap_x_bitmap_avx512::is_available() {
            group.bench_function(format!("avx512/and_into/{label_prefix}"), |bench| {
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
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            group.bench_function(format!("neon/and_into/{label_prefix}"), |bench| {
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
        }
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

    for input in &real_data_inputs() {
        let label = format!("real/{}", input.label);
        let a = input.array_a.clone();
        let b = input.array_b.clone();
        group.throughput(Throughput::Elements((a.len() + b.len()) as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar/intersect", &label),
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
            BenchmarkId::new("scalar/intersect_cardinality", &label),
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
                BenchmarkId::new("sse42/intersect", &label),
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

    for input in &real_data_inputs() {
        let label = format!("real/{}", input.label);
        let array = input.array_a.clone();
        let bm = input.bitmap_b;
        group.throughput(Throughput::Elements(array.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("scalar/intersect", &label),
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
            BenchmarkId::new("scalar/intersect_cardinality", &label),
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
                BenchmarkId::new("avx2/intersect", &label),
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

//! Bench-similarity-distance: dense distance kernels across types,
//! backends, sizes, alignment offsets, and batched query shapes.
//!
//! Per `docs/v0.2_planning/13_VECTOR.md` § 7. The matrix is:
//! - metrics: dot, l2_squared, cosine_similarity (f32/u32),
//!   hamming, jaccard (u64 packed bitvectors)
//! - element types: u32, f32, u64
//! - backends: scalar, dispatched (auto), avx2, avx512, neon
//! - sizes: 256 (byte histogram), 1024 (compact sketch), 4096 (richer
//!   sketch), 16_384 (stress)
//! - alignment offsets: 0, 1, 3, 7, 31
//! - batched: query=1, db ∈ {16, 256, 4_096, 65_536} rows of stride=1024
//! - hamming/jaccard signature widths: 256, 1024, 4096 bits
//!
//! Run all: `cargo bench -p tokenfs-algos --bench similarity`
//! Filter: `cargo bench -p tokenfs-algos --bench similarity -- 'dot_u32/n=4096'`

#![allow(missing_docs)]
// `support` is shared with the larger workload-matrix benches; only the
// `cache_tier_sizes` helper is consumed here, which leaves most of the
// module unreferenced from this binary.
#![allow(dead_code)]

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::vector::{self, kernels};

const SIZES: [usize; 4] = [256, 1024, 4096, 16_384];
const ALIGNMENTS: [usize; 5] = [0, 1, 3, 7, 31];

/// Hamming/Jaccard signature widths, in bits.
const SIG_BITS: [usize; 3] = [256, 1024, 4096];

/// Many-vs-one batched DB sizes (number of rows of length [`BATCH_STRIDE`]).
const BATCH_DB_ROWS: [usize; 4] = [16, 256, 4_096, 65_536];
/// Stride per row for the batched many-vs-one tests.
const BATCH_STRIDE: usize = 1024;

fn make_u32(n: usize, seed: u32) -> Vec<u32> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state & 0x000F_FFFF // keep magnitudes bounded so dot fits comfortably in u64
        })
        .collect()
}

fn make_f32(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Map to [-1.0, 1.0).
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn make_u64(n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed.wrapping_add(1);
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        })
        .collect()
}

/// Bytes processed per call: a + b are both touched, so the working set is
/// `2 * n * size_of::<T>()`.
fn working_bytes_u32(n: usize) -> u64 {
    (n * core::mem::size_of::<u32>() * 2) as u64
}

fn working_bytes_f32(n: usize) -> u64 {
    (n * core::mem::size_of::<f32>() * 2) as u64
}

fn working_bytes_u64(n: usize) -> u64 {
    (n * core::mem::size_of::<u64>() * 2) as u64
}

// ---------- Single-pair benches: u32 dot/L2/cosine ----------

fn bench_dot_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/dot_u32");
    for n in SIZES {
        for offset in ALIGNMENTS {
            // Allocate enough headroom that an offset-of-N start still has
            // room for n elements.
            let raw_a = make_u32(n + offset, 0x9E37_79B9);
            let raw_b = make_u32(n + offset, 0x6E5E_2E5C);
            let a = &raw_a[offset..offset + n];
            let b = &raw_b[offset..offset + n];
            group.throughput(Throughput::Bytes(working_bytes_u32(n)));
            let id = format!("scalar/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| kernels::scalar::dot_u32(black_box(a), black_box(b)).unwrap_or(0));
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| vector::dot_u32(black_box(a), black_box(b)));
            });
        }
    }
    group.finish();
}

fn bench_l2_squared_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/l2_squared_u32");
    for n in SIZES {
        for offset in ALIGNMENTS {
            let raw_a = make_u32(n + offset, 0x1234_5678);
            let raw_b = make_u32(n + offset, 0x8765_4321);
            let a = &raw_a[offset..offset + n];
            let b = &raw_b[offset..offset + n];
            group.throughput(Throughput::Bytes(working_bytes_u32(n)));
            let id = format!("scalar/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    kernels::scalar::l2_squared_u32(black_box(a), black_box(b)).unwrap_or(0)
                });
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| vector::l2_squared_u32(black_box(a), black_box(b)));
            });
        }
    }
    group.finish();
}

fn bench_cosine_u32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/cosine_u32");
    for n in SIZES {
        // Cosine touches a, b twice each (dot, norm_a, norm_b).
        let raw_a = make_u32(n, 0xDEAD_BEEF);
        let raw_b = make_u32(n, 0xCAFE_F00D);
        group.throughput(Throughput::Bytes(working_bytes_u32(n) * 3 / 2));
        let id = format!("scalar/n={n}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| {
                kernels::scalar::cosine_similarity_u32(black_box(&raw_a), black_box(&raw_b))
                    .unwrap_or(0.0)
            });
        });
        let id = format!("auto/n={n}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| {
                vector::cosine_similarity_u32(black_box(&raw_a), black_box(&raw_b)).unwrap_or(0.0)
            });
        });
    }
    group.finish();
}

// ---------- Single-pair benches: f32 dot/L2 across all backends ----------

fn bench_dot_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/dot_f32");
    for n in SIZES {
        for offset in ALIGNMENTS {
            let raw_a = make_f32(n + offset, 0x1111_1111);
            let raw_b = make_f32(n + offset, 0x2222_2222);
            let a = &raw_a[offset..offset + n];
            let b = &raw_b[offset..offset + n];
            group.throughput(Throughput::Bytes(working_bytes_f32(n)));
            let id = format!("scalar/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher
                    .iter(|| kernels::scalar::dot_f32(black_box(a), black_box(b)).unwrap_or(0.0));
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| vector::dot_f32(black_box(a), black_box(b)).unwrap_or(0.0));
            });

            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            if kernels::avx2::is_available() {
                let id = format!("avx2/n={n}/offset={offset}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                    bencher.iter(|| {
                        // SAFETY: availability checked above.
                        unsafe { kernels::avx2::dot_f32(black_box(a), black_box(b)) }
                    });
                });
            }

            #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
            if kernels::avx512::is_available() {
                let id = format!("avx512/n={n}/offset={offset}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                    bencher.iter(|| {
                        // SAFETY: availability checked above.
                        unsafe { kernels::avx512::dot_f32(black_box(a), black_box(b)) }
                    });
                });
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                let id = format!("neon/n={n}/offset={offset}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                    bencher.iter(|| {
                        // SAFETY: NEON is mandatory on AArch64.
                        unsafe { kernels::neon::dot_f32(black_box(a), black_box(b)) }
                    });
                });
            }
        }
    }
    group.finish();
}

fn bench_l2_squared_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/l2_squared_f32");
    for n in SIZES {
        for offset in ALIGNMENTS {
            let raw_a = make_f32(n + offset, 0x3333_3333);
            let raw_b = make_f32(n + offset, 0x4444_4444);
            let a = &raw_a[offset..offset + n];
            let b = &raw_b[offset..offset + n];
            group.throughput(Throughput::Bytes(working_bytes_f32(n)));
            let id = format!("scalar/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    kernels::scalar::l2_squared_f32(black_box(a), black_box(b)).unwrap_or(0.0)
                });
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| vector::l2_squared_f32(black_box(a), black_box(b)).unwrap_or(0.0));
            });

            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            if kernels::avx2::is_available() {
                let id = format!("avx2/n={n}/offset={offset}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                    bencher.iter(|| {
                        // SAFETY: availability checked above.
                        unsafe { kernels::avx2::l2_squared_f32(black_box(a), black_box(b)) }
                    });
                });
            }

            #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
            if kernels::avx512::is_available() {
                let id = format!("avx512/n={n}/offset={offset}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                    bencher.iter(|| {
                        // SAFETY: availability checked above.
                        unsafe { kernels::avx512::l2_squared_f32(black_box(a), black_box(b)) }
                    });
                });
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                let id = format!("neon/n={n}/offset={offset}");
                group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                    bencher.iter(|| {
                        // SAFETY: NEON is mandatory on AArch64.
                        unsafe { kernels::neon::l2_squared_f32(black_box(a), black_box(b)) }
                    });
                });
            }
        }
    }
    group.finish();
}

// ---------- Hamming / Jaccard u64 ----------

fn bench_hamming_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/hamming_u64");
    for &bits in &SIG_BITS {
        let n_words = bits / 64;
        let a = make_u64(n_words, 0xA1A1_B2B2_C3C3_D4D4);
        let b = make_u64(n_words, 0x5151_5eed_f22c_0ffe);
        group.throughput(Throughput::Bytes(working_bytes_u64(n_words)));

        let id = format!("scalar/bits={bits}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher
                .iter(|| kernels::scalar::hamming_u64(black_box(&a), black_box(&b)).unwrap_or(0));
        });
        let id = format!("auto/bits={bits}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| vector::hamming_u64(black_box(&a), black_box(&b)).unwrap_or(0));
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels::avx2::is_available() {
            let id = format!("avx2/bits={bits}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { kernels::avx2::hamming_u64(black_box(&a), black_box(&b)) }
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels::avx512::is_popcnt_available() {
            let id = format!("avx512/bits={bits}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { kernels::avx512::hamming_u64(black_box(&a), black_box(&b)) }
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/bits={bits}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { kernels::neon::hamming_u64(black_box(&a), black_box(&b)) }
                });
            });
        }
    }
    group.finish();
}

fn bench_jaccard_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/jaccard_u64");
    for &bits in &SIG_BITS {
        let n_words = bits / 64;
        let a = make_u64(n_words, 0x0C0F_FEEF_22F2_2F22);
        let b = make_u64(n_words, 0xDEAD_BEEF_DEAD_BEEF);
        group.throughput(Throughput::Bytes(working_bytes_u64(n_words)));

        let id = format!("scalar/bits={bits}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher
                .iter(|| kernels::scalar::jaccard_u64(black_box(&a), black_box(&b)).unwrap_or(0.0));
        });
        let id = format!("auto/bits={bits}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| vector::jaccard_u64(black_box(&a), black_box(&b)).unwrap_or(0.0));
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels::avx2::is_available() {
            let id = format!("avx2/bits={bits}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { kernels::avx2::jaccard_u64(black_box(&a), black_box(&b)) }
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels::avx512::is_popcnt_available() {
            let id = format!("avx512/bits={bits}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { kernels::avx512::jaccard_u64(black_box(&a), black_box(&b)) }
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/bits={bits}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { kernels::neon::jaccard_u64(black_box(&a), black_box(&b)) }
                });
            });
        }
    }
    group.finish();
}

// ---------- Batched many-vs-one ----------

fn bench_batched_dot_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/dot_f32_one_to_many");
    for &n_rows in &BATCH_DB_ROWS {
        let stride = BATCH_STRIDE;
        let query = make_f32(stride, 0xDEAD_BEEF);
        let db = make_f32(n_rows * stride, 0xCAFE_F00D);
        let mut out = vec![0.0_f32; n_rows];
        // Throughput: query + db read once each.
        group.throughput(Throughput::Bytes(
            ((stride + n_rows * stride) * core::mem::size_of::<f32>()) as u64,
        ));

        let id = format!("auto/db={n_rows}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| {
                vector::dot_f32_one_to_many(
                    black_box(&query),
                    black_box(&db),
                    stride,
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

fn bench_batched_l2_squared_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/l2_squared_f32_one_to_many");
    for &n_rows in &BATCH_DB_ROWS {
        let stride = BATCH_STRIDE;
        let query = make_f32(stride, 0xFADE_BABE);
        let db = make_f32(n_rows * stride, 0xC0FE_FACE);
        let mut out = vec![0.0_f32; n_rows];
        group.throughput(Throughput::Bytes(
            ((stride + n_rows * stride) * core::mem::size_of::<f32>()) as u64,
        ));

        let id = format!("auto/db={n_rows}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| {
                vector::l2_squared_f32_one_to_many(
                    black_box(&query),
                    black_box(&db),
                    stride,
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

fn bench_batched_cosine_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/cosine_similarity_f32_one_to_many");
    for &n_rows in &BATCH_DB_ROWS {
        let stride = BATCH_STRIDE;
        let query = make_f32(stride, 0xF000_0000);
        let db = make_f32(n_rows * stride, 0xBA5E_BA11);
        let mut out = vec![0.0_f32; n_rows];
        group.throughput(Throughput::Bytes(
            ((stride + n_rows * stride) * core::mem::size_of::<f32>()) as u64,
        ));

        let id = format!("auto/db={n_rows}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| {
                vector::cosine_similarity_f32_one_to_many(
                    black_box(&query),
                    black_box(&db),
                    stride,
                    black_box(&mut out),
                );
            });
        });
    }
    group.finish();
}

fn bench_batched_hamming_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/hamming_u64_one_to_many");
    // For hamming/jaccard a "vector" is a 256-bit MinHash signature
    // (4 u64 words). We sweep over typical N and signature widths.
    for &bits in &SIG_BITS {
        let stride = bits / 64;
        for &n_rows in &BATCH_DB_ROWS {
            let query = make_u64(stride, 0xF0F0_F0F0_F0F0_F0F0);
            let db = make_u64(n_rows * stride, 0xCC11_BB22_AA33_9944);
            let mut out = vec![0_u32; n_rows];
            group.throughput(Throughput::Bytes(
                ((stride + n_rows * stride) * core::mem::size_of::<u64>()) as u64,
            ));

            let id = format!("auto/bits={bits}/db={n_rows}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    vector::hamming_u64_one_to_many(
                        black_box(&query),
                        black_box(&db),
                        stride,
                        black_box(&mut out),
                    );
                });
            });
        }
    }
    group.finish();
}

fn bench_batched_jaccard_u64(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/jaccard_u64_one_to_many");
    for &bits in &SIG_BITS {
        let stride = bits / 64;
        for &n_rows in &BATCH_DB_ROWS {
            let query = make_u64(stride, 0xACAC_ACAC_ACAC_ACAC);
            let db = make_u64(n_rows * stride, 0xBDBD_BDBD_BDBD_BDBD);
            let mut out = vec![0.0_f64; n_rows];
            group.throughput(Throughput::Bytes(
                ((stride + n_rows * stride) * core::mem::size_of::<u64>()) as u64,
            ));

            let id = format!("auto/bits={bits}/db={n_rows}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    vector::jaccard_u64_one_to_many(
                        black_box(&query),
                        black_box(&db),
                        stride,
                        black_box(&mut out),
                    );
                });
            });
        }
    }
    group.finish();
}

// ---------- Cache-tier sweep on dot_f32 ----------

fn bench_dot_f32_cache_tiers(c: &mut Criterion) {
    let mut group = c.benchmark_group("similarity_distance/dot_f32_cache_tiers");
    for (tier_label, byte_size) in support::cache_tier_sizes() {
        // Half of the working set in each input vector.
        let n_floats = (byte_size / 2).div_ceil(core::mem::size_of::<f32>());
        let a = make_f32(n_floats, 0xC0DE_F00D);
        let b = make_f32(n_floats, 0xBEEF_BABE);
        group.throughput(Throughput::Bytes(working_bytes_f32(n_floats)));

        let id = format!("scalar/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| kernels::scalar::dot_f32(black_box(&a), black_box(&b)).unwrap_or(0.0));
        });
        let id = format!("auto/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| vector::dot_f32(black_box(&a), black_box(&b)).unwrap_or(0.0));
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels::avx2::is_available() {
            let id = format!("avx2/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { kernels::avx2::dot_f32(black_box(&a), black_box(&b)) }
                });
            });
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels::avx512::is_available() {
            let id = format!("avx512/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe { kernels::avx512::dot_f32(black_box(&a), black_box(&b)) }
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = format!("neon/{tier_label}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { kernels::neon::dot_f32(black_box(&a), black_box(&b)) }
                });
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dot_u32,
    bench_l2_squared_u32,
    bench_cosine_u32,
    bench_dot_f32,
    bench_l2_squared_f32,
    bench_hamming_u64,
    bench_jaccard_u64,
    bench_batched_dot_f32,
    bench_batched_l2_squared_f32,
    bench_batched_cosine_f32,
    bench_batched_hamming_u64,
    bench_batched_jaccard_u64,
    bench_dot_f32_cache_tiers,
);
criterion_main!(benches);

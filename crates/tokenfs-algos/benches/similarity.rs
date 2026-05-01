//! Bench-similarity-distance: dense distance kernels across types, backends,
//! sizes, and alignment offsets.
//!
//! Per docs/SIMILARITY_APPROXIMATION_ROADMAP.md step 7. The matrix is:
//! - metrics: dot, l2_squared, cosine_similarity
//! - element types: u32, f32
//! - backends: scalar, dispatched (auto -> AVX2/NEON when available)
//! - sizes: 256 (byte histogram), 1024 (compact sketch), 4096 (richer sketch),
//!   16_384 (stress)
//! - alignment offsets: 0, 1, 3, 7, 31
//!
//! Run all: `cargo bench -p tokenfs-algos --bench similarity`
//! Filter: `cargo bench -p tokenfs-algos --bench similarity -- 'dot_u32/n=4096'`

#![allow(missing_docs)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tokenfs_algos::similarity::{distance, kernels::scalar};

const SIZES: [usize; 4] = [256, 1024, 4096, 16_384];
const ALIGNMENTS: [usize; 5] = [0, 1, 3, 7, 31];

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

/// Bytes processed per call: a + b are both touched, so the working set is
/// `2 * n * size_of::<T>()`.
fn working_bytes_u32(n: usize) -> u64 {
    (n * core::mem::size_of::<u32>() * 2) as u64
}

fn working_bytes_f32(n: usize) -> u64 {
    (n * core::mem::size_of::<f32>() * 2) as u64
}

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
                bencher.iter(|| scalar::dot_u32(black_box(a), black_box(b)).unwrap_or(0));
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| distance::dot_u32(black_box(a), black_box(b)).unwrap_or(0));
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
                bencher.iter(|| scalar::l2_squared_u32(black_box(a), black_box(b)).unwrap_or(0));
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| distance::l2_squared_u32(black_box(a), black_box(b)).unwrap_or(0));
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
                scalar::cosine_similarity_u32(black_box(&raw_a), black_box(&raw_b)).unwrap_or(0.0)
            });
        });
        let id = format!("auto/n={n}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
            bencher.iter(|| {
                distance::cosine_similarity_u32(black_box(&raw_a), black_box(&raw_b)).unwrap_or(0.0)
            });
        });
    }
    group.finish();
}

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
                bencher.iter(|| scalar::dot_f32(black_box(a), black_box(b)).unwrap_or(0.0));
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher.iter(|| distance::dot_f32(black_box(a), black_box(b)).unwrap_or(0.0));
            });
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
                bencher.iter(|| scalar::l2_squared_f32(black_box(a), black_box(b)).unwrap_or(0.0));
            });
            let id = format!("auto/n={n}/offset={offset}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |bencher, _| {
                bencher
                    .iter(|| distance::l2_squared_f32(black_box(a), black_box(b)).unwrap_or(0.0));
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
);
criterion_main!(benches);

//! `bench-bits-streamvbyte`: Stream-VByte encode + decode throughput.
//!
//! Reports per-backend throughput at the canonical sizes called out in
//! `docs/v0.2_planning/10_BITS.md` § 3 — 256, 1K, 16K, 256K, 4M elements.
//! Throughput is reported as `Bytes(n * 4)`, i.e. raw `u32` bytes through
//! the pipeline; readers can divide by 4 to get elements/sec.
//!
//! Run all: `cargo bench -p tokenfs-algos --bench bits_streamvbyte`
//! Filter:  `cargo bench -p tokenfs-algos --bench bits_streamvbyte -- decode/avx2`
//!
//! Real-data path: set `TOKENFS_ALGOS_REAL_FILES=<path1>:<path2>` to
//! synthesise additional `Vec<u32>` streams from each file (4-byte LE
//! windows masked to 24 bits, matching the synthetic posting-list-delta
//! distribution). Each listed file produces an extra row in every
//! sub-bench, labelled `real/<file-stem>/n=<element-count>`.

#![allow(missing_docs)]
// `support` is shared with the larger workload-matrix benches; only
// `real_files_as_bytes` is consumed here.
#![allow(dead_code)]

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::bits;

/// Element counts called out in the spec: 256, 1K, 16K, 256K, 4M. Covers
/// L1 (≈1 KB-shaped working set), L2 (≈64 KB), L3 (≈1 MB), and DRAM
/// (≈16 MB) tiers when measuring the encoded/decoded `u32` byte
/// footprint (`n * 4` bytes per side).
const SIZES: &[(usize, &str)] = &[
    (256, "n=256"),
    (1024, "n=1K"),
    (16 * 1024, "n=16K"),
    (256 * 1024, "n=256K"),
    (4 * 1024 * 1024, "n=4M"),
];

fn deterministic_values(n: usize) -> Vec<u32> {
    // Three-byte mask captures a realistic posting-list-delta-style mix
    // (most values fit in 1-3 bytes; nothing degenerates to a 1-byte
    // run that hides the dispatcher and shuffle-table costs).
    let mut state = 0xF22_C0FFEE_u64;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32) & 0x00ff_ffff
        })
        .collect()
}

/// Real-data input shape: a `Vec<u32>` materialised from 4-byte LE
/// windows of the file, masked to the low 24 bits (matching the
/// posting-list-delta-style distribution used by the synthetic path so
/// the same shuffle-table and dispatcher costs are exercised).
struct RealU32Input {
    label: String,
    values: Vec<u32>,
}

fn real_data_inputs() -> Vec<RealU32Input> {
    support::real_files_as_bytes()
        .into_iter()
        .filter_map(|(label, bytes)| {
            if bytes.len() < 4 {
                return None;
            }
            let values: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|chunk| {
                    u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) & 0x00ff_ffff
                })
                .collect();
            if values.is_empty() {
                return None;
            }
            Some(RealU32Input { label, values })
        })
        .collect()
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_streamvbyte/decode");
    let real_inputs = real_data_inputs();
    let synthetic_iter = SIZES
        .iter()
        .map(|&(n, label)| (label.to_string(), deterministic_values(n)));
    let real_iter = real_inputs.iter().map(|input| {
        (
            format!("real/{}/n={}", input.label, input.values.len()),
            input.values.clone(),
        )
    });

    for (label, values) in synthetic_iter.chain(real_iter) {
        let n = values.len();
        if n == 0 {
            continue;
        }
        let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(n)];
        let mut data = vec![0_u8; bits::streamvbyte_data_max_len(n)];
        let written = bits::streamvbyte_encode_u32(&values, &mut ctrl, &mut data);
        // Pad data with 16 bytes of slack so SIMD kernels can overshoot
        // at the tail without falling out of the loop too early.
        data.resize(written + 16, 0);
        let mut out = vec![0_u32; n];

        // Throughput is computed against the decoded `u32` bytes (n * 4):
        // that's the dominant cost for callers and matches the convention
        // used by the other `bits_*` benches.
        group.throughput(Throughput::Bytes((n * 4) as u64));

        let id = BenchmarkId::new(format!("scalar/{label}"), n);
        group.bench_function(id, |b| {
            b.iter(|| {
                bits::streamvbyte::kernels::scalar::decode_u32(
                    black_box(&ctrl),
                    black_box(&data[..written]),
                    black_box(n),
                    black_box(&mut out),
                );
            });
        });

        let id = BenchmarkId::new(format!("auto/{label}"), n);
        group.bench_function(id, |b| {
            b.iter(|| {
                bits::streamvbyte_decode_u32(
                    black_box(&ctrl),
                    black_box(&data[..written + 16]),
                    black_box(n),
                    black_box(&mut out),
                );
            });
        });

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if bits::streamvbyte::kernels::ssse3::is_available() {
            let id = BenchmarkId::new(format!("ssse3/{label}"), n);
            group.bench_function(id, |b| {
                b.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe {
                        bits::streamvbyte::kernels::ssse3::decode_u32(
                            black_box(&ctrl),
                            black_box(&data[..written + 16]),
                            black_box(n),
                            black_box(&mut out),
                        )
                    }
                });
            });
        }

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if bits::streamvbyte::kernels::avx2::is_available() {
            let id = BenchmarkId::new(format!("avx2/{label}"), n);
            group.bench_function(id, |b| {
                b.iter(|| {
                    // SAFETY: availability checked above.
                    unsafe {
                        bits::streamvbyte::kernels::avx2::decode_u32(
                            black_box(&ctrl),
                            black_box(&data[..written + 16]),
                            black_box(n),
                            black_box(&mut out),
                        )
                    }
                });
            });
        }

        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            let id = BenchmarkId::new(format!("neon/{label}"), n);
            group.bench_function(id, |b| {
                b.iter(|| {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe {
                        bits::streamvbyte::kernels::neon::decode_u32(
                            black_box(&ctrl),
                            black_box(&data[..written + 16]),
                            black_box(n),
                            black_box(&mut out),
                        )
                    }
                });
            });
        }
    }
    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_streamvbyte/encode");
    let real_inputs = real_data_inputs();
    let synthetic_iter = SIZES
        .iter()
        .map(|&(n, label)| (label.to_string(), deterministic_values(n)));
    let real_iter = real_inputs.iter().map(|input| {
        (
            format!("real/{}/n={}", input.label, input.values.len()),
            input.values.clone(),
        )
    });

    for (label, values) in synthetic_iter.chain(real_iter) {
        let n = values.len();
        if n == 0 {
            continue;
        }
        let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(n)];
        let mut data = vec![0_u8; bits::streamvbyte_data_max_len(n)];
        group.throughput(Throughput::Bytes((n * 4) as u64));

        let id = BenchmarkId::new(format!("scalar/{label}"), n);
        group.bench_function(id, |b| {
            b.iter(|| {
                bits::streamvbyte::kernels::scalar::encode_u32(
                    black_box(&values),
                    black_box(&mut ctrl),
                    black_box(&mut data),
                );
            });
        });

        let id = BenchmarkId::new(format!("auto/{label}"), n);
        group.bench_function(id, |b| {
            b.iter(|| {
                bits::streamvbyte_encode_u32(
                    black_box(&values),
                    black_box(&mut ctrl),
                    black_box(&mut data),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);

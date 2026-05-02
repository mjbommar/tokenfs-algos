//! Cross-backend, cross-architecture micro-benchmark.
//!
//! For each primitive that has a scalar reference plus one or more
//! hardware-accelerated backends, this example measures median ns/op on
//! a fixed payload, then prints a TSV table to stdout. Designed to be
//! run inside CI on every native runner and the rows aggregated across
//! architectures for a same-commit speedup comparison.
//!
//! Output schema (TSV, one row per measurement):
//!   primitive\tbackend\tpayload_bytes\tmedian_ns\tthroughput_GBps
//!
//! Two header lines are also printed (lines starting with `# `): one
//! with environment metadata (runner label, target triple, rustc
//! version, timestamp) and one with the column names.
//!
//! No external test framework. Each measurement: 3 warmup calls,
//! adaptive inner-iteration calibration to land near `MEASURE_TARGET_MS`
//! per sample, 11 samples, median reported. Outliers are not trimmed —
//! median is robust enough.

#![allow(missing_docs)]
#![allow(clippy::cast_precision_loss)]

use std::env;
use std::hint::black_box;
use std::time::Instant;

use tokenfs_algos::histogram::summary::byte_value_moments;
use tokenfs_algos::histogram::topk::MisraGries;
use tokenfs_algos::{
    byteclass, fingerprint, histogram, runlength,
    similarity::{self, kernels as sim_kernels},
    sketch,
};

const MEASURE_TARGET_MS: u64 = 30;
const SAMPLES: usize = 11;

const PAYLOAD_SIZES_BYTES: &[usize] = &[1024, 64 * 1024, 1024 * 1024];

fn main() {
    print_header();

    bench_histogram_block();
    bench_histogram_block_rle();
    bench_topk_misra_gries();
    bench_byte_value_moments();
    bench_fingerprint_block();
    bench_byteclass_classify();
    bench_byteclass_validate_utf8();
    bench_runlength_transitions();
    bench_sketch_crc32_hash4();
    bench_similarity_dot_f32();
    bench_similarity_l2_squared_f32();
}

// ---------- environment header ----------

fn print_header() {
    let runner = env::var("RUNNER_LABEL").unwrap_or_else(|_| "local".to_string());
    println!("# runner={runner}");
    println!("# target_arch={}", env::consts::ARCH);
    println!("# target_os={}", env::consts::OS);
    println!("# target_pointer_width=64");
    println!("# rustc_channel=nightly");
    println!("# samples={SAMPLES} target_ms={MEASURE_TARGET_MS}");
    println!("primitive\tbackend\tpayload_bytes\tmedian_ns\tthroughput_GBps");
}

// ---------- per-primitive benches ----------

fn bench_histogram_block() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "histogram-block",
            "scalar-direct-u64",
            n,
            measure(|| {
                black_box(histogram::kernels::direct_u64::block(black_box(&bytes)));
            }),
        );
        emit(
            "histogram-block",
            "scalar-stripe4-u32",
            n,
            measure(|| {
                black_box(histogram::kernels::stripe4_u32::block(black_box(&bytes)));
            }),
        );
        emit(
            "histogram-block",
            "scalar-stripe8-u32",
            n,
            measure(|| {
                black_box(histogram::kernels::stripe8_u32::block(black_box(&bytes)));
            }),
        );
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        emit(
            "histogram-block",
            "avx2-stripe4-u32",
            n,
            measure(|| {
                black_box(histogram::kernels::avx2_stripe4_u32::block(black_box(
                    &bytes,
                )));
            }),
        );
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        emit(
            "histogram-block",
            "avx2-palette-u32",
            n,
            measure(|| {
                black_box(histogram::kernels::avx2_palette_u32::block(black_box(
                    &bytes,
                )));
            }),
        );
    }
}

fn bench_histogram_block_rle() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "histogram-block-rle",
            "scalar-stripe4-u32",
            n,
            measure(|| {
                black_box(histogram::kernels::stripe4_u32::block(black_box(&bytes)));
            }),
        );
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        emit(
            "histogram-block-rle",
            "avx2-rle-stripe4-u32",
            n,
            measure(|| {
                black_box(histogram::kernels::avx2_rle_stripe4_u32::block(black_box(
                    &bytes,
                )));
            }),
        );
    }
}

fn bench_topk_misra_gries() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "topk-misra-gries-k16",
            "scalar",
            n,
            measure(|| {
                let mut sk = MisraGries::<16>::new();
                sk.update_slice(black_box(&bytes));
                black_box(&sk);
            }),
        );
    }
}

fn bench_byte_value_moments() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "byte-value-moments",
            "scalar",
            n,
            measure(|| {
                black_box(byte_value_moments(black_box(&bytes)));
            }),
        );
    }
}

fn bench_fingerprint_block() {
    let block = make_random_bytes(fingerprint::BLOCK_SIZE);
    let arr: &[u8; fingerprint::BLOCK_SIZE] = block.as_slice().try_into().expect("block size");
    emit(
        "fingerprint-block",
        "scalar",
        fingerprint::BLOCK_SIZE,
        measure(|| {
            black_box(fingerprint::kernels::scalar::block(black_box(arr)));
        }),
    );
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    if fingerprint::kernels::avx2::is_available() {
        emit(
            "fingerprint-block",
            "avx2",
            fingerprint::BLOCK_SIZE,
            measure(|| {
                black_box(fingerprint::kernels::avx2::block(black_box(arr)));
            }),
        );
    }
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    if fingerprint::kernels::neon::is_available() {
        emit(
            "fingerprint-block",
            "neon",
            fingerprint::BLOCK_SIZE,
            measure(|| {
                black_box(fingerprint::kernels::neon::block(black_box(arr)));
            }),
        );
    }
}

fn bench_byteclass_classify() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_text_bytes(n);
        emit(
            "byteclass-classify",
            "scalar",
            n,
            measure(|| {
                black_box(byteclass::kernels::scalar::classify(black_box(&bytes)));
            }),
        );
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if byteclass::kernels::avx2::is_available() {
            emit(
                "byteclass-classify",
                "avx2",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { byteclass::kernels::avx2::classify(black_box(&bytes)) });
                }),
            );
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        emit(
            "byteclass-classify",
            "neon",
            n,
            measure(|| {
                // SAFETY: NEON is mandatory in the AArch64 ABI.
                black_box(unsafe { byteclass::kernels::neon::classify(black_box(&bytes)) });
            }),
        );
    }
}

fn bench_byteclass_validate_utf8() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_text_bytes(n);
        emit(
            "byteclass-validate-utf8",
            "scalar",
            n,
            measure(|| {
                black_box(byteclass::kernels::scalar::validate_utf8(black_box(&bytes)));
            }),
        );
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if byteclass::kernels::avx2::is_available() {
            emit(
                "byteclass-validate-utf8",
                "avx2",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        byteclass::kernels::avx2::validate_utf8(black_box(&bytes))
                    });
                }),
            );
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        emit(
            "byteclass-validate-utf8",
            "neon",
            n,
            measure(|| {
                // SAFETY: NEON is mandatory in the AArch64 ABI.
                black_box(unsafe { byteclass::kernels::neon::validate_utf8(black_box(&bytes)) });
            }),
        );
    }
}

fn bench_runlength_transitions() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_runlike_bytes(n);
        emit(
            "runlength-transitions",
            "scalar",
            n,
            measure(|| {
                black_box(runlength::kernels::scalar::transitions(black_box(&bytes)));
            }),
        );
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if runlength::kernels::avx2::is_available() {
            emit(
                "runlength-transitions",
                "avx2",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { runlength::kernels::avx2::transitions(black_box(&bytes)) });
                }),
            );
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        emit(
            "runlength-transitions",
            "neon",
            n,
            measure(|| {
                // SAFETY: NEON is mandatory in the AArch64 ABI.
                black_box(unsafe { runlength::kernels::neon::transitions(black_box(&bytes)) });
            }),
        );
    }
}

fn bench_sketch_crc32_hash4() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        let mut bins_scalar = [0_u32; 4096];
        emit(
            "sketch-crc32-hash4-bins",
            "scalar",
            n,
            measure(|| {
                bins_scalar.fill(0);
                sketch::kernels::scalar::crc32_hash4_bins(black_box(&bytes), &mut bins_scalar);
                black_box(&bins_scalar);
            }),
        );
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if sketch::kernels::sse42::is_available() {
            let mut bins_sse = [0_u32; 4096];
            emit(
                "sketch-crc32-hash4-bins",
                "sse42",
                n,
                measure(|| {
                    bins_sse.fill(0);
                    // SAFETY: availability checked immediately above.
                    unsafe {
                        sketch::kernels::sse42::crc32_hash4_bins(black_box(&bytes), &mut bins_sse);
                    }
                    black_box(&bins_sse);
                }),
            );
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if sketch::kernels::neon::is_available() {
            let mut bins_neon = [0_u32; 4096];
            emit(
                "sketch-crc32-hash4-bins",
                "neon",
                n,
                measure(|| {
                    bins_neon.fill(0);
                    // SAFETY: availability checked immediately above.
                    unsafe {
                        sketch::kernels::neon::crc32_hash4_bins(black_box(&bytes), &mut bins_neon);
                    }
                    black_box(&bins_neon);
                }),
            );
        }
    }
}

fn bench_similarity_dot_f32() {
    // 256, 4096, 65536 floats == 1KiB, 16KiB, 256KiB
    for &n_floats in &[256_usize, 4096, 65_536] {
        let payload_bytes = n_floats * 4;
        let a = make_random_floats(n_floats, 0xa1b2_c3d4);
        let b = make_random_floats(n_floats, 0xdead_beef);
        emit(
            "similarity-dot-f32",
            "scalar",
            payload_bytes,
            measure(|| {
                black_box(sim_kernels::scalar::dot_f32(black_box(&a), black_box(&b)));
            }),
        );
        emit(
            "similarity-dot-f32",
            "auto",
            payload_bytes,
            measure(|| {
                black_box(similarity::distance::dot_f32(black_box(&a), black_box(&b)));
            }),
        );
    }
}

fn bench_similarity_l2_squared_f32() {
    for &n_floats in &[256_usize, 4096, 65_536] {
        let payload_bytes = n_floats * 4;
        let a = make_random_floats(n_floats, 0xfeed_face);
        let b = make_random_floats(n_floats, 0xcafe_d00d);
        emit(
            "similarity-l2-squared-f32",
            "scalar",
            payload_bytes,
            measure(|| {
                black_box(sim_kernels::scalar::l2_squared_f32(
                    black_box(&a),
                    black_box(&b),
                ));
            }),
        );
        emit(
            "similarity-l2-squared-f32",
            "auto",
            payload_bytes,
            measure(|| {
                black_box(similarity::distance::l2_squared_f32(
                    black_box(&a),
                    black_box(&b),
                ));
            }),
        );
    }
}

// ---------- measurement core ----------

fn measure<F: FnMut()>(mut f: F) -> u64 {
    // Warmup.
    for _ in 0..3 {
        f();
    }
    // Calibrate inner_iters so each sample takes ~MEASURE_TARGET_MS.
    let calibration_iters = 10_u128;
    let cal_start = Instant::now();
    for _ in 0..calibration_iters {
        f();
    }
    let cal_ns = cal_start.elapsed().as_nanos().max(1);
    let per_call_ns = cal_ns / calibration_iters;
    let target_ns = u128::from(MEASURE_TARGET_MS) * 1_000_000;
    let inner_iters = (target_ns / per_call_ns.max(1)).clamp(10, 5_000_000) as u64;

    let mut samples = [0_u64; SAMPLES];
    for sample in &mut samples {
        let t0 = Instant::now();
        for _ in 0..inner_iters {
            f();
        }
        let ns = t0.elapsed().as_nanos();
        *sample = (ns / u128::from(inner_iters)) as u64;
    }
    samples.sort_unstable();
    samples[SAMPLES / 2]
}

fn emit(primitive: &str, backend: &str, payload_bytes: usize, median_ns: u64) {
    let gbps = throughput_gbps(payload_bytes, median_ns);
    println!("{primitive}\t{backend}\t{payload_bytes}\t{median_ns}\t{gbps:.3}");
}

fn throughput_gbps(payload_bytes: usize, median_ns: u64) -> f64 {
    if median_ns == 0 {
        return f64::INFINITY;
    }
    // bytes/ns == GB/s (1e9 bytes/sec divided by 1e9 ns/sec).
    payload_bytes as f64 / median_ns as f64
}

// ---------- payload generators ----------

fn make_random_bytes(n: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(n);
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    while out.len() < n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(n);
    out
}

fn make_text_bytes(n: usize) -> Vec<u8> {
    // Mostly-ASCII Lorem-style filler, repeated. Includes whitespace,
    // punctuation, and a few multibyte UTF-8 sequences so the validator
    // exercises both the fast path and the multi-byte branch.
    const FILLER: &[u8] = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
        Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
        Caf\xc3\xa9 na\xc3\xafve r\xc3\xa9sum\xc3\xa9 \xe2\x80\x94 a few multibyte glyphs. ";
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let take = (n - out.len()).min(FILLER.len());
        out.extend_from_slice(&FILLER[..take]);
    }
    out
}

fn make_runlike_bytes(n: usize) -> Vec<u8> {
    // Mix of short runs and noise so transitions() finds work to do.
    let mut out = Vec::with_capacity(n);
    let mut state = 0xC0FF_EE12_3456_7890_u64;
    let mut byte = 0_u8;
    while out.len() < n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let run = ((state >> 24) & 0x07) as usize + 1; // 1..8
        let take = run.min(n - out.len());
        for _ in 0..take {
            out.push(byte);
        }
        byte = byte.wrapping_add(((state >> 16) & 0xFF) as u8 | 1);
    }
    out
}

fn make_random_floats(n: usize, seed: u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Map upper 24 bits to a signed value in [-1, 1).
        let bits = (state >> 40) as u32;
        let v = (bits as f32) / (1_u32 << 23) as f32 - 1.0;
        out.push(v);
    }
    out
}

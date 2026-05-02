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

#[cfg(feature = "blake3")]
use tokenfs_algos::hash::blake3 as b3;
use tokenfs_algos::hash::sha256;
use tokenfs_algos::histogram::summary::byte_value_moments;
use tokenfs_algos::histogram::topk::MisraGries;
use tokenfs_algos::sketch::Crc32cHasher;
use tokenfs_algos::{
    byteclass, fingerprint, histogram, runlength,
    search::{
        bitap::Bitap16, bitap::Bitap64, packed_dfa::PackedDfa, packed_pair::PackedPair,
        rabin_karp::RabinKarp, shift_or::ShiftOr, two_way::TwoWay,
    },
    similarity, sketch,
    vector::kernels as vec_kernels,
};
// SVE backend lives in `similarity::kernels::sve` (the v0.2 vector module
// only enumerates scalar/avx2/avx512/neon); access via the deprecated
// shim until SVE work moves over.
#[cfg(all(feature = "sve", target_arch = "aarch64"))]
#[allow(deprecated)]
use tokenfs_algos::similarity::kernels::sve as sve_kernels;

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
    bench_byteclass_classify_lut();
    bench_byteclass_validate_utf8();
    bench_runlength_transitions();
    bench_sketch_crc32_hash4();
    bench_similarity_dot_f32();
    bench_similarity_l2_squared_f32();
    bench_search();
    bench_fuzzy_digest();
    bench_histogram_bit_marginals();
    bench_minhash_update();
    bench_simhash_update();
    bench_hash_sha256();
    bench_hash_blake3();
    bench_incremental_hash();
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
        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if byteclass::kernels::avx512::is_available() {
            emit(
                "byteclass-classify",
                "avx512",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { byteclass::kernels::avx512::classify(black_box(&bytes)) });
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
        #[cfg(all(feature = "sve2", target_arch = "aarch64"))]
        if byteclass::kernels::sve2::is_available() {
            emit(
                "byteclass-classify",
                "sve2",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { byteclass::kernels::sve2::classify(black_box(&bytes)) });
                }),
            );
        }
    }
}

fn bench_byteclass_classify_lut() {
    // 4-class table: same partition as the named `ByteClassCounts`
    // (printable / whitespace / control / high-bit), so this row is
    // directly comparable to `byteclass-classify` above.
    let table_4 = byteclass::printable_control_whitespace_high_bit_table();
    // 16-class table: ASCII letters split by case, digits, and 13 fine
    // partitions over the rest of the byte range, exercising every
    // class slot in the kernel's fixed `MAX_CLASSES`-wide popcount loop.
    let table_16 = byteclass::class_table_from_fn(|b| match b {
        b'A'..=b'Z' => 0,
        b'a'..=b'z' => 1,
        b'0'..=b'9' => 2,
        b' ' | b'\t' => 3,
        b'\n' | b'\r' => 4,
        b'.' | b',' | b';' | b':' | b'?' | b'!' => 5,
        b'(' | b')' | b'[' | b']' | b'{' | b'}' => 6,
        b'<' | b'>' | b'/' | b'\\' | b'|' => 7,
        b'+' | b'-' | b'*' | b'%' | b'^' | b'&' => 8,
        b'\'' | b'"' | b'`' => 9,
        b'_' | b'$' | b'#' | b'@' => 10,
        b'~' | b'=' => 11,
        0x00..=0x1f | 0x7f => 12,
        0x80..=0xbf => 13,
        0xc0..=0xdf => 14,
        0xe0..=0xff => 15,
    });

    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_text_bytes(n);

        emit(
            "byteclass-classify-lut-c4",
            "scalar",
            n,
            measure(|| {
                black_box(byteclass::classify_with_table(black_box(&bytes), &table_4));
            }),
        );
        emit(
            "byteclass-classify-lut-c16",
            "scalar",
            n,
            measure(|| {
                black_box(byteclass::classify_with_table(black_box(&bytes), &table_16));
            }),
        );

        // Existing AVX-512BW kernel for direct comparison against the new
        // permute kernel on the 4-class table.
        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if byteclass::kernels::avx512::is_available() {
            emit(
                "byteclass-classify-lut-c4",
                "avx512bw-cmp-popcnt",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { byteclass::kernels::avx512::classify(black_box(&bytes)) });
                }),
            );
        }

        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if byteclass::kernels::avx512_vbmi::is_available() {
            emit(
                "byteclass-classify-lut-c4",
                "avx512vbmi-permute",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        byteclass::kernels::avx512_vbmi::classify_with_lut(
                            black_box(&bytes),
                            &table_4,
                        )
                    });
                }),
            );
            emit(
                "byteclass-classify-lut-c16",
                "avx512vbmi-permute",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        byteclass::kernels::avx512_vbmi::classify_with_lut(
                            black_box(&bytes),
                            &table_16,
                        )
                    });
                }),
            );
        }
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
        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if byteclass::kernels::avx512::is_available() {
            emit(
                "byteclass-validate-utf8",
                "avx512",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        byteclass::kernels::avx512::validate_utf8(black_box(&bytes))
                    });
                }),
            );
        }
        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if byteclass::kernels::avx512_vbmi::is_available() {
            emit(
                "byteclass-validate-utf8",
                "avx512-vbmi",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        byteclass::kernels::avx512_vbmi::validate_utf8(black_box(&bytes))
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
        #[cfg(all(feature = "sve2", target_arch = "aarch64"))]
        if byteclass::kernels::sve2::is_available() {
            emit(
                "byteclass-validate-utf8",
                "sve2",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        byteclass::kernels::sve2::validate_utf8(black_box(&bytes))
                    });
                }),
            );
        }
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
        #[cfg(all(feature = "sve2", target_arch = "aarch64"))]
        if runlength::kernels::sve2::is_available() {
            emit(
                "runlength-transitions",
                "sve2",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { runlength::kernels::sve2::transitions(black_box(&bytes)) });
                }),
            );
        }
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
                black_box(vec_kernels::scalar::dot_f32(black_box(&a), black_box(&b)));
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
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        emit(
            "similarity-dot-f32",
            "neon",
            payload_bytes,
            measure(|| {
                // SAFETY: NEON is mandatory in the AArch64 ABI.
                black_box(unsafe { vec_kernels::neon::dot_f32(black_box(&a), black_box(&b)) });
            }),
        );
        #[cfg(all(feature = "sve", target_arch = "aarch64"))]
        if sve_kernels::is_available() {
            emit(
                "similarity-dot-f32",
                "sve",
                payload_bytes,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { sve_kernels::dot_f32(black_box(&a), black_box(&b)) });
                }),
            );
        }
    }
}

fn bench_minhash_update() {
    use tokenfs_algos::similarity::kernels_gather;
    let seeds: [u64; 8] = core::array::from_fn(|i| 0xCAFE_BABE_u64 ^ (i as u64));
    let table = kernels_gather::build_table_from_seeds(&seeds);

    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "minhash-update-k8",
            "scalar",
            n,
            measure(|| {
                let mut sig = [u64::MAX; 8];
                kernels_gather::update_minhash_scalar::<8>(black_box(&bytes), &table, &mut sig);
                black_box(&sig);
            }),
        );

        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels_gather::avx2::is_available() {
            emit(
                "minhash-update-k8",
                "avx2-gather",
                n,
                measure(|| {
                    let mut sig = [u64::MAX; 8];
                    // SAFETY: availability checked above.
                    unsafe {
                        kernels_gather::avx2::update_minhash_8way(
                            black_box(&bytes),
                            &table,
                            &mut sig,
                        );
                    }
                    black_box(&sig);
                }),
            );
        }
        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels_gather::avx512::is_available() {
            emit(
                "minhash-update-k8",
                "avx512-gather",
                n,
                measure(|| {
                    let mut sig = [u64::MAX; 8];
                    // SAFETY: availability checked above.
                    unsafe {
                        kernels_gather::avx512::update_minhash_8way(
                            black_box(&bytes),
                            &table,
                            &mut sig,
                        );
                    }
                    black_box(&sig);
                }),
            );
        }
    }
}

fn bench_simhash_update() {
    use tokenfs_algos::similarity::kernels_gather;
    let seeds: [u64; kernels_gather::simhash::BITS] =
        core::array::from_fn(|i| 0x9E37_79B9_7F4A_7C15_u64.wrapping_mul((i as u64) + 1));
    let table = kernels_gather::simhash::build_table_from_seeds(&seeds);

    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "simhash-update",
            "scalar",
            n,
            measure(|| {
                let mut acc = [0_i32; kernels_gather::simhash::BITS];
                kernels_gather::simhash::update_accumulator_scalar(
                    black_box(&bytes),
                    &table,
                    &mut acc,
                );
                black_box(&acc);
            }),
        );
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        if kernels_gather::simhash::avx2::is_available() {
            emit(
                "simhash-update",
                "avx2-gather",
                n,
                measure(|| {
                    let mut acc = [0_i32; kernels_gather::simhash::BITS];
                    // SAFETY: availability checked above.
                    unsafe {
                        kernels_gather::simhash::avx2::update_accumulator(
                            black_box(&bytes),
                            &table,
                            &mut acc,
                        );
                    }
                    black_box(&acc);
                }),
            );
        }
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
                black_box(vec_kernels::scalar::l2_squared_f32(
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
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        emit(
            "similarity-l2-squared-f32",
            "neon",
            payload_bytes,
            measure(|| {
                // SAFETY: NEON is mandatory in the AArch64 ABI.
                black_box(unsafe {
                    vec_kernels::neon::l2_squared_f32(black_box(&a), black_box(&b))
                });
            }),
        );
        #[cfg(all(feature = "sve", target_arch = "aarch64"))]
        if sve_kernels::is_available() {
            emit(
                "similarity-l2-squared-f32",
                "sve",
                payload_bytes,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe { sve_kernels::l2_squared_f32(black_box(&a), black_box(&b)) });
                }),
            );
        }
    }
}

fn bench_search() {
    // Three needle lengths: 4 / 16 / 64 bytes. The needle is taken from
    // a deterministic offset inside each haystack so every algorithm
    // observes the same hit position.
    const NEEDLE_LENS: &[usize] = &[4_usize, 16, 64];
    const HIT_OFFSET_FRAC: usize = 4; // hit at 1/4 into the haystack

    for &n in PAYLOAD_SIZES_BYTES {
        let haystack = make_random_bytes(n);
        for &nlen in NEEDLE_LENS {
            if haystack.len() < nlen {
                continue;
            }
            // Carve a deterministic needle out of the haystack so each
            // algorithm finds it at the same position.
            let hit_at = (haystack.len() / HIT_OFFSET_FRAC).min(haystack.len() - nlen);
            let needle: Vec<u8> = haystack[hit_at..hit_at + nlen].to_vec();
            let primitive = match nlen {
                4 => "search-needle4",
                16 => "search-needle16",
                64 => "search-needle64",
                _ => "search-needle",
            };

            // Two-Way (general).
            let tw = TwoWay::new(&needle);
            emit(
                primitive,
                "two_way",
                n,
                measure(|| {
                    black_box(tw.find(black_box(&haystack)));
                }),
            );

            // Rabin-Karp (rolling hash).
            let rk = RabinKarp::new(&needle);
            emit(
                primitive,
                "rabin_karp",
                n,
                measure(|| {
                    black_box(rk.find(black_box(&haystack)));
                }),
            );

            // Bitap16 (only for ≤16 byte needles).
            if let Some(bp) = Bitap16::new(&needle) {
                emit(
                    primitive,
                    "bitap16",
                    n,
                    measure(|| {
                        black_box(bp.find(black_box(&haystack)));
                    }),
                );
            }

            // Bitap64 (≤64 byte needles).
            if let Some(bp) = Bitap64::new(&needle) {
                emit(
                    primitive,
                    "bitap64",
                    n,
                    measure(|| {
                        black_box(bp.find(black_box(&haystack)));
                    }),
                );
            }

            // Shift-Or (≤64 byte needles).
            if let Some(so) = ShiftOr::new(&needle) {
                emit(
                    primitive,
                    "shift_or",
                    n,
                    measure(|| {
                        black_box(so.find(black_box(&haystack)));
                    }),
                );
            }

            // PackedPair (needle ≥ 2 bytes).
            if let Some(pp) = PackedPair::new(&needle) {
                emit(
                    primitive,
                    "packed_pair",
                    n,
                    measure(|| {
                        black_box(pp.find(black_box(&haystack)));
                    }),
                );
            }

            // PackedDfa (single-pattern, single match).
            let dfa = PackedDfa::new(&[needle.as_slice()]);
            emit(
                primitive,
                "packed_dfa",
                n,
                measure(|| {
                    black_box(dfa.find(black_box(&haystack)));
                }),
            );
        }
    }
}
fn bench_hash_sha256() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "hash-sha256",
            "scalar",
            n,
            measure(|| {
                black_box(sha256::kernels::scalar::sha256(black_box(&bytes)));
            }),
        );
        emit(
            "hash-sha256",
            "auto",
            n,
            measure(|| {
                black_box(sha256::sha256(black_box(&bytes)));
            }),
        );
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if sha256::kernels::x86_shani::is_available() {
            emit(
                "hash-sha256",
                "x86-shani",
                n,
                measure(|| {
                    // SAFETY: availability checked above.
                    black_box(unsafe { sha256::kernels::x86_shani::sha256(black_box(&bytes)) });
                }),
            );
        }
        #[cfg(target_arch = "aarch64")]
        if sha256::kernels::aarch64_sha2::is_available() {
            emit(
                "hash-sha256",
                "aarch64-sha2",
                n,
                measure(|| {
                    // SAFETY: availability checked above.
                    black_box(unsafe { sha256::kernels::aarch64_sha2::sha256(black_box(&bytes)) });
                }),
            );
        }
    }
}

#[cfg(feature = "blake3")]
fn bench_hash_blake3() {
    // Streaming chunk size paired with the upstream `Hasher`. 4 KiB is
    // a typical filesystem block size; comparing against the one-shot
    // path tells us the overhead of the chunked feed.
    const STREAM_CHUNK: usize = 4 * 1024;

    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);

        // Wrapper one-shot vs SHA-256 (already emitted by `bench_hash_sha256`).
        emit(
            "hash-blake3",
            "wrapper",
            n,
            measure(|| {
                black_box(b3::blake3(black_box(&bytes)));
            }),
        );

        // Streaming `Hasher` fed in 4 KiB chunks. Output is bit-exact
        // with the one-shot row above; the timing difference is purely
        // the chunked-feed overhead.
        if n >= STREAM_CHUNK {
            emit(
                "hash-blake3-stream4k",
                "wrapper",
                n,
                measure(|| {
                    let mut h = b3::Hasher::new();
                    for window in black_box(&bytes).chunks(STREAM_CHUNK) {
                        h.update(window);
                    }
                    black_box(h.finalize());
                }),
            );
        }
    }
}

#[cfg(not(feature = "blake3"))]
fn bench_hash_blake3() {
    // No-op stub when the `blake3` feature is not enabled. The rest of
    // the benchmark suite still runs.
}

/// Compare one-shot vs streaming hashers, fed in 4 KiB chunks. The per-update
/// overhead should land within ~5% of the one-shot path; the streaming
/// hashers do detect their backend at construction (constant cost) and
/// dispatch through a `match` on every chunk (a single predicted branch).
///
/// Output rows for SHA-256 (`hash-sha256-stream`) and CRC32C
/// (`hash-crc32c-stream`) include both the one-shot baseline and the
/// streaming path so the speedup ratio can be derived directly.
fn bench_incremental_hash() {
    const CHUNK: usize = 4 * 1024;

    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);

        // SHA-256 one-shot baseline (auto-dispatched).
        emit(
            "hash-sha256-stream",
            "auto-one-shot",
            n,
            measure(|| {
                black_box(sha256::sha256(black_box(&bytes)));
            }),
        );

        // SHA-256 streaming via Hasher fed in 4 KiB chunks.
        emit(
            "hash-sha256-stream",
            "stream-4kib",
            n,
            measure(|| {
                let mut h = sha256::Hasher::new();
                for chunk in bytes.chunks(CHUNK) {
                    h.update(black_box(chunk));
                }
                black_box(h.finalize());
            }),
        );

        // CRC32C one-shot baseline (auto-dispatched).
        emit(
            "hash-crc32c-stream",
            "auto-one-shot",
            n,
            measure(|| {
                black_box(tokenfs_algos::sketch::crc32c_bytes(0, black_box(&bytes)));
            }),
        );

        // CRC32C streaming via Crc32cHasher fed in 4 KiB chunks.
        emit(
            "hash-crc32c-stream",
            "stream-4kib",
            n,
            measure(|| {
                let mut h = Crc32cHasher::new();
                for chunk in bytes.chunks(CHUNK) {
                    h.update(black_box(chunk));
                }
                black_box(h.finalize());
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

fn bench_fuzzy_digest() {
    use tokenfs_algos::similarity::fuzzy::{ctph, tlsh_like};
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_text_bytes(n);
        let bytes_b = make_random_bytes(n);

        // CTPH digest build.
        emit(
            "fuzzy-ctph-digest",
            "scalar",
            n,
            measure(|| {
                black_box(ctph::Digest::from_bytes(black_box(&bytes)));
            }),
        );
        // CTPH similarity (digest vs digest — measures comparison cost only).
        let dig_a = ctph::Digest::from_bytes(&bytes);
        let dig_b = ctph::Digest::from_bytes(&bytes_b);
        emit(
            "fuzzy-ctph-similarity",
            "scalar",
            n,
            measure(|| {
                black_box(ctph::similarity(black_box(&dig_a), black_box(&dig_b)));
            }),
        );
        // TLSH-like digest build (requires ≥ 256 bytes for a stable
        // digest; below that the upstream implementation refuses).
        if n >= 256 {
            emit(
                "fuzzy-tlsh-digest",
                "scalar",
                n,
                measure(|| {
                    black_box(tlsh_like::digest(black_box(&bytes)));
                }),
            );
            let t_a = tlsh_like::digest(&bytes);
            let t_b = tlsh_like::digest(&bytes_b);
            emit(
                "fuzzy-tlsh-distance",
                "scalar",
                n,
                measure(|| {
                    black_box(tlsh_like::distance(black_box(&t_a), black_box(&t_b)));
                }),
            );
        }
    }
}

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
fn bench_histogram_bit_marginals() {
    for &n in PAYLOAD_SIZES_BYTES {
        let bytes = make_random_bytes(n);
        emit(
            "histogram-bit-marginals",
            "scalar",
            n,
            measure(|| {
                black_box(histogram::kernels::avx512_bitalg_bitsliced::block_scalar(
                    black_box(&bytes),
                ));
            }),
        );
        if histogram::kernels::avx512_bitalg_bitsliced::is_available() {
            emit(
                "histogram-bit-marginals",
                "avx512-bitalg",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        histogram::kernels::avx512_bitalg_bitsliced::block_unchecked(black_box(
                            &bytes,
                        ))
                    });
                }),
            );
        }
        if histogram::kernels::avx512_gfni_bitsliced::is_available() {
            emit(
                "histogram-bit-marginals",
                "avx512-gfni",
                n,
                measure(|| {
                    // SAFETY: availability checked immediately above.
                    black_box(unsafe {
                        histogram::kernels::avx512_gfni_bitsliced::block_unchecked(black_box(
                            &bytes,
                        ))
                    });
                }),
            );
        }
    }
}

/// No-op stub when AVX-512 features are not enabled or the target
/// architecture is not x86. Kept so the `main` call site does not need
/// `cfg` gating around it.
#[cfg(not(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64"))))]
fn bench_histogram_bit_marginals() {}

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

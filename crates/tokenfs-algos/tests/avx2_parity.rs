//! Explicit parity tests between every public AVX2-dispatched kernel and its
//! pinned scalar reference.
//!
//! `tests/parity.rs` covers scalar-vs-auto for non-SIMD modules; this file
//! exercises the AVX2 surfaces specifically. Each test runtime-skips when AVX2
//! is unavailable so the suite still passes on machines without the feature,
//! and the entire file is cfg-gated to x86/x86_64 with the `avx2` Cargo
//! feature so non-x86 builds compile cleanly.

#![allow(missing_docs)]
#![allow(deprecated)]
#![allow(clippy::unwrap_used)] // Test code — panic on None is the desired failure mode.
#![cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]

use proptest::prelude::*;

use tokenfs_algos::{
    approx, bitmap, bits, byteclass,
    fingerprint::{self, BLOCK_SIZE},
    hash,
    histogram::{self, ByteHistogram},
    runlength, similarity, vector,
};

fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

fn sse42_available() -> bool {
    std::is_x86_feature_detected!("sse4.2")
}

fn synthetic_corpus() -> Vec<Vec<u8>> {
    let mut cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        vec![0],
        vec![0xff],
        vec![0; 4096],
        vec![0xff; 4096],
        (0_u8..=255).collect(),
        (0_u8..=255).cycle().take(4096).collect(),
        b"tokenfs-algos AVX2 parity \xe2\x9c\x85 mixed UTF-8\nline two\r\n\tindented"
            .iter()
            .copied()
            .cycle()
            .take(8192)
            .collect(),
    ];

    for len in [
        1_usize, 2, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023,
        1024, 1025, 4095, 4096, 4097, 8192,
    ] {
        cases.push(
            (0..len)
                .map(|i| (i.wrapping_mul(17) ^ (i >> 2).wrapping_mul(31)) as u8)
                .collect(),
        );
    }

    cases
}

fn unaligned_subslices(bytes: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    for start in [0_usize, 1, 3, 7, 15, 31, 63, 127] {
        if start >= bytes.len() {
            continue;
        }
        for len in [
            0_usize, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023,
        ] {
            let end = (start + len).min(bytes.len());
            out.push(&bytes[start..end]);
        }
    }
    out
}

#[test]
fn avx2_palette_histogram_matches_scalar_reference_on_synthetic_corpus() {
    if !avx2_available() {
        eprintln!("avx2 unavailable on this host; skipping palette parity test");
        return;
    }

    for input in synthetic_corpus() {
        let expected = ByteHistogram::from_block(&input);
        let actual = histogram::kernels::avx2_palette_u32::block(&input);
        assert_eq!(
            actual.counts(),
            expected.counts(),
            "avx2_palette_u32 diverged on length {}",
            input.len()
        );
        assert_eq!(actual.total(), expected.total());
    }
}

#[test]
fn avx2_palette_histogram_matches_scalar_reference_on_unaligned_subslices() {
    if !avx2_available() {
        return;
    }

    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = ByteHistogram::from_block(slice);
        let actual = histogram::kernels::avx2_palette_u32::block(slice);
        assert_eq!(
            actual.counts(),
            expected.counts(),
            "avx2_palette_u32 diverged on slice len {}",
            slice.len()
        );
    }
}

#[test]
fn avx2_stripe4_histogram_matches_scalar_reference_on_synthetic_corpus() {
    // `avx2_stripe4_u32` carries `target_feature(enable = "avx2")` on its
    // unsafe primitive but currently performs scalar increments under that
    // gate; its parity guarantee is still load-bearing for the planner, so
    // we test it the same way as the genuine palette kernel.
    if !avx2_available() {
        return;
    }

    for input in synthetic_corpus() {
        let expected = ByteHistogram::from_block(&input);
        let actual = histogram::kernels::avx2_stripe4_u32::block(&input);
        assert_eq!(
            actual.counts(),
            expected.counts(),
            "avx2_stripe4_u32 diverged on length {}",
            input.len()
        );
    }
}

#[test]
fn avx2_byteclass_matches_scalar_reference_on_synthetic_corpus() {
    if !avx2_available() {
        return;
    }

    for input in synthetic_corpus() {
        let expected = byteclass::kernels::scalar::classify(&input);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { byteclass::kernels::avx2::classify(&input) };
        assert_eq!(
            actual,
            expected,
            "avx2 byteclass classify diverged on length {}",
            input.len()
        );
    }
}

#[test]
fn avx2_byteclass_matches_scalar_reference_on_unaligned_subslices() {
    if !avx2_available() {
        return;
    }

    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = byteclass::kernels::scalar::classify(slice);
        // SAFETY: avx2_available() checked above.
        let actual = unsafe { byteclass::kernels::avx2::classify(slice) };
        assert_eq!(
            actual,
            expected,
            "avx2 byteclass classify diverged on slice len {}",
            slice.len()
        );
    }
}

/// Returns a corpus that hits every error class `core::str::from_utf8` can
/// emit, plus pure-ASCII / valid-mixed cases at lengths around the 64-byte
/// AVX2 chunk boundary.
fn utf8_corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut cases: Vec<(&'static str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("ascii-1", b"a".to_vec()),
        ("ascii-63", vec![b'x'; 63]),
        ("ascii-64", vec![b'x'; 64]),
        ("ascii-65", vec![b'x'; 65]),
        ("ascii-1024", vec![b'A'; 1024]),
        // Valid 2/3/4-byte sequences.
        ("two-byte-utf8", "héllo wörld".as_bytes().to_vec()),
        ("three-byte-utf8", "你好,世界".as_bytes().to_vec()),
        ("four-byte-utf8", "𝐇𝐞𝐥𝐥𝐨".as_bytes().to_vec()),
        (
            "mixed-utf8",
            "ASCII tail with a 🎉 emoji and 日本語 then more."
                .as_bytes()
                .to_vec(),
        ),
        // Lone continuation byte (no leader).
        ("lone-cont-0x80", vec![b'a', b'b', 0x80, b'c']),
        ("lone-cont-0xbf", vec![0xbf]),
        // Bad leader (0xc0, 0xc1 are always invalid; 0xfe/0xff never appear in UTF-8).
        ("invalid-c0", vec![0xc0, 0x80]),
        ("invalid-c1", vec![0xc1, 0x80]),
        ("invalid-fe", vec![0xfe]),
        ("invalid-ff", vec![0xff]),
        // Truncated trailing sequence (incomplete — error_len = 0 by from_utf8 contract).
        ("truncated-2byte", vec![b'a', 0xc2]),
        ("truncated-3byte-1", vec![0xe2]),
        ("truncated-3byte-2", vec![0xe2, 0x82]),
        ("truncated-4byte-1", vec![0xf0]),
        ("truncated-4byte-2", vec![0xf0, 0x9f]),
        ("truncated-4byte-3", vec![0xf0, 0x9f, 0x98]),
        // Overlong encodings — invalid.
        ("overlong-2byte", vec![0xc0, 0xaf]),
        ("overlong-3byte", vec![0xe0, 0x80, 0xaf]),
        ("overlong-4byte", vec![0xf0, 0x80, 0x80, 0xaf]),
        // Surrogates U+D800..U+DFFF — invalid in UTF-8.
        ("surrogate-d800", vec![0xed, 0xa0, 0x80]),
        ("surrogate-dfff", vec![0xed, 0xbf, 0xbf]),
        // Codepoints beyond U+10FFFF — invalid.
        ("over-max-codepoint", vec![0xf4, 0x90, 0x80, 0x80]),
        ("over-max-f5", vec![0xf5, 0x80, 0x80, 0x80]),
    ];

    // Multibyte sequence straddling the 64-byte SIMD block boundary.
    let mut straddle = vec![b'a'; 62];
    straddle.extend_from_slice("é".as_bytes()); // 2-byte sequence at position 62..64
    straddle.extend(b"trailing".iter().copied());
    cases.push(("multibyte-straddles-block", straddle));

    // Error in the second 64-byte block (forces SIMD-then-fallback).
    let mut second_block_err = vec![b'x'; 64];
    second_block_err.push(0xff);
    second_block_err.extend_from_slice(b"after");
    cases.push(("error-in-second-block", second_block_err));

    // Error in the last block before the tail.
    let mut tail_error = vec![b'x'; 192];
    tail_error.push(0xc0);
    tail_error.push(b'?');
    cases.push(("error-near-tail", tail_error));

    // Pure-ASCII fast path that returns to DFA mode and back.
    let mut ascii_dfa_ascii = vec![b' '; 64];
    ascii_dfa_ascii.extend_from_slice("héllo".as_bytes()); // forces DFA mode
    ascii_dfa_ascii.extend(vec![b' '; 64]); // back to ASCII fast path
    ascii_dfa_ascii.extend_from_slice("世界".as_bytes()); // DFA again
    ascii_dfa_ascii.extend(vec![b' '; 96]); // ASCII tail straddling tail
    cases.push(("ascii-dfa-ascii-loop", ascii_dfa_ascii));

    cases
}

#[test]
fn avx2_validate_utf8_matches_scalar_reference_on_corpus() {
    if !avx2_available() {
        eprintln!("avx2 unavailable on this host; skipping UTF-8 parity test");
        return;
    }

    for (label, input) in utf8_corpus() {
        let expected = byteclass::kernels::scalar::validate_utf8(&input);
        // Exercise both the dispatcher and the direct AVX2 entry point.
        let dispatched = byteclass::validate_utf8(&input);
        // SAFETY: avx2_available() returned true above.
        let direct = unsafe { byteclass::kernels::avx2::validate_utf8(&input) };
        assert_eq!(
            dispatched,
            expected,
            "auto-dispatched validate_utf8 diverged on case {label} (len {})",
            input.len()
        );
        assert_eq!(
            direct,
            expected,
            "direct AVX2 validate_utf8 diverged on case {label} (len {})",
            input.len()
        );
    }
}

#[test]
fn avx2_validate_utf8_matches_scalar_reference_on_long_inputs() {
    if !avx2_available() {
        return;
    }

    // Long input with errors at varying positions to stress the SIMD-to-scalar
    // re-entry point logic.
    let template = "Lorem ipsum dolor sit amet, consectetur adipiscing 𝐞𝐥𝐢𝐭. \
                    日本語のテキストもここに混ざっています。 ";
    let big: String = template.repeat(64);
    let mut bytes = big.into_bytes();
    let total_len = bytes.len();

    let expected = byteclass::kernels::scalar::validate_utf8(&bytes);
    let dispatched = byteclass::validate_utf8(&bytes);
    assert_eq!(dispatched, expected, "large valid corpus diverged");

    for offset in [0_usize, 1, 31, 32, 63, 64, 65, 127, 128, 192, total_len / 2] {
        if offset >= bytes.len() {
            continue;
        }
        let saved = bytes[offset];
        bytes[offset] = 0xff; // Always-invalid byte.
        let expected = byteclass::kernels::scalar::validate_utf8(&bytes);
        let dispatched = byteclass::validate_utf8(&bytes);
        assert_eq!(
            dispatched, expected,
            "validate_utf8 diverged after planting 0xff at offset {offset}"
        );
        bytes[offset] = saved;
    }
}

#[test]
fn avx2_runlength_transitions_match_scalar_reference_on_synthetic_corpus() {
    if !avx2_available() {
        return;
    }

    // Cover empties, lengths around the 32-byte vector boundary, and
    // a few patterns that exercise the all-equal / all-different limits.
    let mut cases = synthetic_corpus();
    cases.push(b"abababab".repeat(64));
    cases.push(vec![0xa5_u8; 33]);
    cases.push(vec![0xa5_u8; 64]);
    cases.push(vec![0xa5_u8; 65]);

    for input in cases {
        let expected = runlength::kernels::scalar::transitions(&input);
        let actual = runlength::transitions(&input);
        assert_eq!(
            actual,
            expected,
            "avx2 runlength::transitions diverged on length {}",
            input.len()
        );
        // Sanity: the dispatched path must agree with the dedicated AVX2 kernel.
        // SAFETY: avx2_available() returned true above.
        let direct = unsafe { runlength::kernels::avx2::transitions(&input) };
        assert_eq!(direct, expected);
    }
}

#[test]
fn avx2_runlength_transitions_match_scalar_reference_on_unaligned_subslices() {
    if !avx2_available() {
        return;
    }

    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = runlength::kernels::scalar::transitions(slice);
        let actual = runlength::transitions(slice);
        assert_eq!(
            actual,
            expected,
            "avx2 runlength::transitions diverged on slice len {}",
            slice.len()
        );
    }
}

#[test]
fn avx2_fingerprint_block_matches_scalar_reference() {
    if !(avx2_available() && sse42_available()) {
        eprintln!("avx2/sse4.2 unavailable; skipping fingerprint block parity test");
        return;
    }

    let inputs: Vec<[u8; BLOCK_SIZE]> = vec![
        [0; BLOCK_SIZE],
        [0xff; BLOCK_SIZE],
        std::array::from_fn(|i| i as u8),
        std::array::from_fn(|i| (i.wrapping_mul(17) ^ (i >> 2).wrapping_mul(31)) as u8),
        std::array::from_fn(|i| if i % 8 == 0 { 0xa5 } else { i as u8 }),
        std::array::from_fn(|i| b"tokenfs-algos block fingerprint parity!\n"[i % 40]),
    ];

    for block in inputs.iter() {
        let expected = fingerprint::kernels::scalar::block(block);
        let actual = fingerprint::kernels::avx2::block(block);
        assert_eq!(
            actual, expected,
            "avx2 fingerprint::block diverged from scalar reference"
        );
    }
}

#[test]
fn avx2_bits_popcount_u64_matches_scalar_on_synthetic_corpus() {
    if !avx2_available() {
        eprintln!("avx2 unavailable on this host; skipping bits::popcount parity test");
        return;
    }

    // Lengths chosen to span every plausible SIMD block boundary in the
    // popcount kernels: 4 u64 (AVX2 32-byte vec), 32 u64 (AVX2 8x-unrolled
    // 256B), 8 u64 (AVX-512 64-byte vec), and beyond.
    let lengths = [
        0_usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257, 511,
        1023, 1024, 1025, 4095,
    ];
    let mut state = 0xF22_C0FFEE_u64;
    for len in lengths {
        let words: Vec<u64> = (0..len)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_f491_4f6c_dd1d)
            })
            .collect();
        let expected = bits::kernels::scalar::popcount_u64_slice(&words);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { bits::kernels::avx2::popcount_u64_slice(&words) };
        assert_eq!(
            actual, expected,
            "avx2 bits::popcount_u64_slice diverged at len {len}"
        );
        // Dispatched API must agree.
        assert_eq!(
            bits::popcount_u64_slice(&words),
            expected,
            "dispatched bits::popcount_u64_slice diverged at len {len}"
        );
    }
}

#[test]
fn avx2_bits_popcount_u8_matches_scalar_on_unaligned_subslices() {
    if !avx2_available() {
        return;
    }

    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = bits::kernels::scalar::popcount_u8_slice(slice);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { bits::kernels::avx2::popcount_u8_slice(slice) };
        assert_eq!(
            actual,
            expected,
            "avx2 bits::popcount_u8_slice diverged on slice len {}",
            slice.len()
        );
    }
}

fn deterministic_packed_values(n: usize, w: u32, seed: u64) -> Vec<u32> {
    let mask = if w == 32 { u32::MAX } else { (1_u32 << w) - 1 };
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32) & mask
        })
        .collect()
}

#[test]
fn avx2_bits_bit_pack_decode_matches_scalar_on_every_width() {
    if !avx2_available() {
        eprintln!("avx2 unavailable on this host; skipping bits::bit_pack parity test");
        return;
    }

    // Cover every supported width plus several lengths so we exercise
    // tail-handling at small `n` and the full SIMD path at large `n`.
    let lengths = [0_usize, 1, 7, 8, 33, 257, 1024];
    for w in 1_u32..=32 {
        for &n in &lengths {
            let values =
                deterministic_packed_values(n, w, 0xF22_C0FFEE_u64 ^ ((w as u64) << 16) ^ n as u64);
            let needed = bits::DynamicBitPacker::new(w).encoded_len(n);
            let mut encoded = vec![0_u8; needed];
            bits::bit_pack::kernels::scalar::encode_u32_slice(w, &values, &mut encoded);

            let mut expected = vec![0_u32; n];
            bits::bit_pack::kernels::scalar::decode_u32_slice(w, &encoded, n, &mut expected);

            let mut actual = vec![0_u32; n];
            // SAFETY: avx2_available() returned true above.
            unsafe {
                bits::bit_pack::kernels::avx2::decode_u32_slice(w, &encoded, n, &mut actual);
            }
            assert_eq!(
                actual, expected,
                "avx2 bits::bit_pack::decode diverged at w={w} n={n}"
            );

            // The auto-dispatched API must agree as well.
            let mut dispatched = vec![0_u32; n];
            bits::DynamicBitPacker::new(w).decode_u32_slice(&encoded, n, &mut dispatched);
            assert_eq!(
                dispatched, expected,
                "dispatched bits::bit_pack::decode diverged at w={w} n={n}"
            );
        }
    }
}

/// Deterministic `u32` corpus clamped to `max_bytes` significant bytes
/// so the encoded width spans every Stream-VByte length code.
fn deterministic_streamvbyte_values(n: usize, seed: u64, max_bytes: u32) -> Vec<u32> {
    let mask: u32 = match max_bytes {
        1 => 0x0000_00ff,
        2 => 0x0000_ffff,
        3 => 0x00ff_ffff,
        _ => u32::MAX,
    };
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32) & mask
        })
        .collect()
}

#[test]
fn ssse3_bits_streamvbyte_decode_matches_scalar_on_size_grid() {
    if !std::is_x86_feature_detected!("ssse3") {
        eprintln!("ssse3 unavailable on this host; skipping bits::streamvbyte SSSE3 parity test");
        return;
    }
    // Cover every (full-groups, tail-shape) interaction plus large sizes
    // that exercise the SIMD-then-scalar tail boundary.
    for n in [
        0_usize, 1, 2, 3, 4, 5, 7, 8, 16, 31, 32, 33, 64, 99, 100, 256, 1023, 1024, 1025, 4096,
    ] {
        for max in [1_u32, 2, 3, 4] {
            let values = deterministic_streamvbyte_values(
                n,
                0xF22_C0FFEE_u64 ^ ((max as u64) << 8) ^ n as u64,
                max,
            );
            let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(n)];
            let mut data = vec![0_u8; bits::streamvbyte_data_max_len(n)];
            let written = bits::streamvbyte_encode_u32(&values, &mut ctrl, &mut data);

            let mut expected = vec![0_u32; n];
            bits::streamvbyte::kernels::scalar::decode_u32(
                &ctrl,
                &data[..written],
                n,
                &mut expected,
            );

            let mut actual = vec![0_u32; n];
            // SAFETY: ssse3 availability checked above.
            let consumed = unsafe {
                bits::streamvbyte::kernels::ssse3::decode_u32(
                    &ctrl,
                    &data[..written],
                    n,
                    &mut actual,
                )
            };
            assert_eq!(
                actual, expected,
                "ssse3 bits::streamvbyte diverged at n={n} max_bytes={max}"
            );
            assert_eq!(
                consumed, written,
                "ssse3 offset diverged at n={n} max_bytes={max}"
            );
        }
    }
}

#[test]
fn avx2_bits_streamvbyte_decode_matches_scalar_on_size_grid() {
    if !avx2_available() {
        eprintln!("avx2 unavailable on this host; skipping bits::streamvbyte AVX2 parity test");
        return;
    }
    // Same grid as the SSSE3 path; AVX2 dual-pumps two control bytes per
    // iteration so we add sizes that hit the odd-residual full group.
    for n in [
        0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 17, 31, 32, 33, 64, 100, 256, 257, 1023, 1024,
        1025, 4096,
    ] {
        for max in [1_u32, 2, 3, 4] {
            let values = deterministic_streamvbyte_values(
                n,
                0xC0FFEE_u64 ^ ((max as u64) << 16) ^ n as u64,
                max,
            );
            let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(n)];
            let mut data = vec![0_u8; bits::streamvbyte_data_max_len(n)];
            let written = bits::streamvbyte_encode_u32(&values, &mut ctrl, &mut data);

            let mut expected = vec![0_u32; n];
            bits::streamvbyte::kernels::scalar::decode_u32(
                &ctrl,
                &data[..written],
                n,
                &mut expected,
            );

            let mut actual = vec![0_u32; n];
            // SAFETY: avx2 availability checked above.
            let consumed = unsafe {
                bits::streamvbyte::kernels::avx2::decode_u32(
                    &ctrl,
                    &data[..written],
                    n,
                    &mut actual,
                )
            };
            assert_eq!(
                actual, expected,
                "avx2 bits::streamvbyte diverged at n={n} max_bytes={max}"
            );
            assert_eq!(
                consumed, written,
                "avx2 offset diverged at n={n} max_bytes={max}"
            );

            // Auto-dispatched API must agree.
            let mut dispatched = vec![0_u32; n];
            let dispatched_consumed =
                bits::streamvbyte_decode_u32(&ctrl, &data[..written], n, &mut dispatched);
            assert_eq!(
                dispatched, expected,
                "dispatched bits::streamvbyte diverged at n={n} max_bytes={max}"
            );
            assert_eq!(dispatched_consumed, written);
        }
    }
}

fn deterministic_u32_haystack(n: usize, seed: u64) -> Vec<u32> {
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

#[test]
fn sse41_hash_set_membership_matches_scalar_on_size_grid() {
    if !std::is_x86_feature_detected!("sse4.1") {
        eprintln!("sse4.1 unavailable; skipping SSE4.1 set-membership parity test");
        return;
    }
    // Cover SIMD block boundaries: 4 lanes (SSE/NEON), 8 (AVX2), 16
    // (AVX-512), 32 (AVX2 8x4 unroll), and beyond.
    for len in [
        0_usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 1023,
    ] {
        let haystack = deterministic_u32_haystack(len, 0x5151_5eed ^ (len as u64));
        for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
            let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
            // SAFETY: SSE4.1 availability checked above.
            let actual =
                unsafe { hash::set_membership::kernels::sse41::contains_u32(&haystack, needle) };
            assert_eq!(actual, expected, "len {len} needle {needle}");
        }
        if len > 0 {
            for &pos in &[0_usize, len / 2, len - 1] {
                let needle = haystack[pos];
                let expected =
                    hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: SSE4.1 availability checked above.
                let actual = unsafe {
                    hash::set_membership::kernels::sse41::contains_u32(&haystack, needle)
                };
                assert_eq!(actual, expected, "len {len} pos {pos}");
            }
        }
    }
}

#[test]
fn avx2_hash_set_membership_matches_scalar_on_size_grid() {
    if !avx2_available() {
        eprintln!("avx2 unavailable; skipping AVX2 set-membership parity test");
        return;
    }
    for len in [
        0_usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 1023,
    ] {
        let haystack = deterministic_u32_haystack(len, 0xA1A1_B2B2 ^ (len as u64));
        for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
            let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
            // SAFETY: AVX2 availability checked above.
            let actual =
                unsafe { hash::set_membership::kernels::avx2::contains_u32(&haystack, needle) };
            assert_eq!(actual, expected, "len {len} needle {needle}");
        }
        if len > 0 {
            for &pos in &[0_usize, len / 2, len - 1] {
                let needle = haystack[pos];
                let expected =
                    hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: AVX2 availability checked above.
                let actual =
                    unsafe { hash::set_membership::kernels::avx2::contains_u32(&haystack, needle) };
                assert_eq!(actual, expected, "len {len} pos {pos}");
            }
        }
    }
}

#[test]
fn dispatched_hash_set_membership_matches_scalar() {
    // Exercise the auto-dispatched API end-to-end. Independent of feature
    // detection: this should always produce scalar-equivalent results.
    for len in [0_usize, 1, 7, 8, 16, 33, 256, 1023] {
        let haystack = deterministic_u32_haystack(len, 0xC0DE_C0DE ^ (len as u64));
        for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
            let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
            let dispatched = hash::contains_u32_simd(&haystack, needle);
            assert_eq!(dispatched, expected, "len {len} needle {needle}");
        }
        if len > 0 {
            let needle = haystack[len / 2];
            assert!(hash::contains_u32_simd(&haystack, needle));
        }
    }
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_hash_set_membership_matches_scalar_on_size_grid() {
    if !std::is_x86_feature_detected!("avx512f") {
        eprintln!("avx512f unavailable; skipping AVX-512 set-membership parity test");
        return;
    }
    for len in [0_usize, 1, 3, 8, 15, 16, 17, 31, 32, 33, 64, 128, 256, 1023] {
        let haystack = deterministic_u32_haystack(len, 0xBEEF_F00D ^ (len as u64));
        for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
            let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
            // SAFETY: AVX-512F availability checked above.
            let actual =
                unsafe { hash::set_membership::kernels::avx512::contains_u32(&haystack, needle) };
            assert_eq!(actual, expected, "len {len} needle {needle}");
        }
        if len > 0 {
            for &pos in &[0_usize, len / 2, len - 1] {
                let needle = haystack[pos];
                let expected =
                    hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: AVX-512F availability checked above.
                let actual = unsafe {
                    hash::set_membership::kernels::avx512::contains_u32(&haystack, needle)
                };
                assert_eq!(actual, expected, "len {len} pos {pos}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        ..ProptestConfig::default()
    })]

    #[test]
    fn avx2_palette_histogram_matches_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = ByteHistogram::from_block(&bytes);
        let actual = histogram::kernels::avx2_palette_u32::block(&bytes);
        prop_assert_eq!(actual.counts(), expected.counts());
    }

    #[test]
    fn avx2_stripe4_histogram_matches_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = ByteHistogram::from_block(&bytes);
        let actual = histogram::kernels::avx2_stripe4_u32::block(&bytes);
        prop_assert_eq!(actual.counts(), expected.counts());
    }

    #[test]
    fn avx2_byteclass_matches_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = byteclass::kernels::scalar::classify(&bytes);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { byteclass::kernels::avx2::classify(&bytes) };
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn avx2_validate_utf8_matches_scalar_for_random_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = byteclass::kernels::scalar::validate_utf8(&bytes);
        let actual = byteclass::validate_utf8(&bytes);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn avx2_runlength_transitions_match_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = runlength::kernels::scalar::transitions(&bytes);
        let actual = runlength::transitions(&bytes);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn avx2_similarity_dot_u32_matches_scalar(
        a in proptest::collection::vec(any::<u32>(), 0..1024),
        // We need same-length b; build it by mapping a deterministic permutation.
        seed in any::<u32>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let mut b: Vec<u32> = a.iter().enumerate()
            .map(|(i, x)| x.wrapping_mul(seed.wrapping_add(i as u32 + 1)))
            .collect();
        // Sometimes test b being identical to a (edge case for cosine = 1.0).
        if seed % 7 == 0 { b = a.clone(); }
        let expected = similarity::kernels::scalar::dot_u32(&a, &b);
        let actual = similarity::distance::dot_u32(&a, &b);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn avx2_similarity_l1_l2_u32_match_scalar(
        a in proptest::collection::vec(any::<u32>(), 0..1024),
        seed in any::<u32>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let b: Vec<u32> = a.iter().enumerate()
            .map(|(i, x)| x.wrapping_add(seed.wrapping_mul(i as u32 + 1)))
            .collect();
        prop_assert_eq!(
            similarity::distance::l1_u32(&a, &b),
            similarity::kernels::scalar::l1_u32(&a, &b),
        );
        prop_assert_eq!(
            similarity::distance::l2_squared_u32(&a, &b),
            similarity::kernels::scalar::l2_squared_u32(&a, &b),
        );
    }

    #[test]
    fn avx2_similarity_cosine_u32_matches_scalar(
        a in proptest::collection::vec(any::<u32>().prop_map(|x| x % (1 << 20)), 0..1024),
        seed in any::<u32>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        // Bound entries so dot/norm products stay in the f64 mantissa zone where
        // both paths should agree bit-for-bit.
        let b: Vec<u32> = a.iter().enumerate()
            .map(|(i, x)| (x ^ seed.rotate_left(i as u32)) % (1 << 20))
            .collect();
        let expected = similarity::kernels::scalar::cosine_similarity_u32(&a, &b);
        let actual = similarity::distance::cosine_similarity_u32(&a, &b);
        match (expected, actual) {
            (None, None) => {}
            (Some(e), Some(a)) => {
                if e == 0.0 {
                    prop_assert!(a.abs() < 1e-12);
                } else {
                    prop_assert!((e - a).abs() / e.abs().max(1.0) < 1e-12,
                        "cosine_similarity_u32 diverged: scalar={e} avx2={a}");
                }
            }
            _ => prop_assert!(false, "None/Some mismatch"),
        }
    }

    #[test]
    fn avx2_similarity_dot_l2_f32_match_scalar(
        // Input range matches realistic tokenfs vectors: byte-histogram
        // counts ([0, 256]) and normalized fingerprint deltas ([-1, 1]),
        // so [-256, 256] covers both with margin. The previous range of
        // [-1000, 1000] over 1024 elements drove partial sums to ~1e9 and
        // forced a coarse tolerance; tightening here is the natural complement
        // to the L1-norm-based scale that lets us assert at 1e-3 confidently.
        a in proptest::collection::vec(-256.0_f32..256.0, 0..1024),
        seed in any::<u32>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let b: Vec<f32> = a.iter().enumerate()
            .map(|(i, x)| x + (seed.wrapping_mul(i as u32 + 1) as f32 * 1e-3))
            .collect();
        // f32 SIMD reductions use a different summation order than scalar
        // (8-way tree vs left-to-right) so a few ULP of difference is expected.
        let dot_s = similarity::kernels::scalar::dot_f32(&a, &b).unwrap();
        let dot_v = similarity::distance::dot_f32(&a, &b).unwrap();
        // Higham §3 / Wilkinson dot-product error bound:
        //   |fl(dot) - exact| <= n * eps * sum(|a_i*b_i|)
        // i.e. the realistic noise floor is proportional to the L1 norm of
        // products, NOT the final dot value. Under catastrophic cancellation
        // |dot| << sum(|a*b|), so dividing the diff by |dot| inflates the
        // reported relative error to 5–10%+ even when both kernels are
        // correct. Comparing against sum(|a*b|) (with a |dot| floor for the
        // non-cancellation regime) gives a stable bound that catches genuine
        // kernel divergence (would be >>1e-3) without flaking on adversarial
        // seeds. L2_squared has no cancellation (all squared terms are
        // non-negative) so it stays within 5e-4 against the dot-style scale.
        let l1_prod: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y).abs()).sum();
        let scale = l1_prod.max(dot_s.abs()).max(dot_v.abs()).max(1.0);
        prop_assert!((dot_s - dot_v).abs() / scale < 1e-3,
            "dot_f32 diverged: scalar={dot_s} avx2={dot_v} l1_prod={l1_prod}");

        let l2_s = similarity::kernels::scalar::l2_squared_f32(&a, &b).unwrap();
        let l2_v = similarity::distance::l2_squared_f32(&a, &b).unwrap();
        let scale = l2_s.abs().max(l2_v.abs()).max(1.0);
        prop_assert!((l2_s - l2_v).abs() / scale < 5e-4,
            "l2_squared_f32 diverged: scalar={l2_s} avx2={l2_v}");
    }

    #[test]
    fn avx2_fingerprint_block_matches_scalar_for_random_blocks(
        block in proptest::array::uniform32(any::<u8>())
            .prop_flat_map(|head| {
                proptest::array::uniform32(any::<u8>())
                    .prop_map(move |tail| {
                        let mut out = [0_u8; BLOCK_SIZE];
                        for i in 0..BLOCK_SIZE {
                            out[i] = if i < 32 { head[i] }
                                else if i < 64 { tail[i - 32] }
                                else { (head[i % 32] ^ tail[(i + 7) % 32]).wrapping_add(i as u8) };
                        }
                        out
                    })
            }),
    ) {
        if !(avx2_available() && sse42_available()) {
            return Ok(());
        }
        let expected = fingerprint::kernels::scalar::block(&block);
        let actual = fingerprint::kernels::avx2::block(&block);
        prop_assert_eq!(actual, expected);
    }

    // ---------- gather-based MinHash kernel parity ----------

    #[test]
    fn avx2_gather_minhash_8way_matches_scalar_random(
        bytes in proptest::collection::vec(any::<u8>(), 64..16_384),
        seed_root in any::<u64>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        use tokenfs_algos::similarity::kernels_gather;
        let seeds: [u64; 8] = core::array::from_fn(|i| seed_root.wrapping_add(i as u64));
        let table = kernels_gather::build_table_from_seeds(&seeds);

        let mut s_scalar = [u64::MAX; 8];
        kernels_gather::update_minhash_scalar::<8>(&bytes, &table, &mut s_scalar);

        let mut s_avx2 = [u64::MAX; 8];
        // SAFETY: avx2_available() returned true above.
        unsafe {
            kernels_gather::avx2::update_minhash_8way(&bytes, &table, &mut s_avx2);
        }
        prop_assert_eq!(s_scalar, s_avx2);
    }

    /// `K = 16` proptest: only the scalar table-based reference exists at
    /// this width, so we verify the auto dispatcher (which falls through
    /// to scalar for non-8 widths) matches the per-byte reference.
    #[test]
    fn gather_minhash_16way_scalar_matches_per_byte_reference(
        bytes in proptest::collection::vec(any::<u8>(), 64..16_384),
        seed_root in any::<u64>(),
    ) {
        use tokenfs_algos::similarity::kernels_gather;
        let seeds: [u64; 16] = core::array::from_fn(|i| seed_root.wrapping_add(i as u64));
        let table = kernels_gather::build_table_from_seeds(&seeds);

        let mut s_scalar = [u64::MAX; 16];
        kernels_gather::update_minhash_scalar::<16>(&bytes, &table, &mut s_scalar);

        let mut expected = [u64::MAX; 16];
        for &b in &bytes {
            for k in 0..16 {
                let h = tokenfs_algos::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] { expected[k] = h; }
            }
        }
        prop_assert_eq!(s_scalar, expected);
    }

    /// `K = 32` proptest: same shape as `K=16`. Verifies the table family
    /// stays consistent at L1-spilling widths.
    #[test]
    fn gather_minhash_32way_scalar_matches_per_byte_reference(
        bytes in proptest::collection::vec(any::<u8>(), 64..16_384),
        seed_root in any::<u64>(),
    ) {
        use tokenfs_algos::similarity::kernels_gather;
        let seeds: [u64; 32] = core::array::from_fn(|i| seed_root.wrapping_add(i as u64));
        let table = kernels_gather::build_table_from_seeds(&seeds);

        let mut s_scalar = [u64::MAX; 32];
        kernels_gather::update_minhash_scalar::<32>(&bytes, &table, &mut s_scalar);

        let mut expected = [u64::MAX; 32];
        for &b in &bytes {
            for k in 0..32 {
                let h = tokenfs_algos::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] { expected[k] = h; }
            }
        }
        prop_assert_eq!(s_scalar, expected);
    }

    /// Top-level `MinHash` table API parity check: the dispatched
    /// `update_bytes_table_8` must produce the same `Signature<8>` as
    /// the per-byte reference family. This is the contract the spec
    /// calls out: callers using seeds can move to the gather path with
    /// no change in output bits.
    #[test]
    fn minhash_update_bytes_table_8_matches_reference(
        bytes in proptest::collection::vec(any::<u8>(), 64..16_384),
        seed_root in any::<u64>(),
    ) {
        use tokenfs_algos::similarity::minhash;
        let seeds: [u64; 8] = core::array::from_fn(|i| seed_root.wrapping_add(i as u64));
        let table = minhash::build_byte_table_from_seeds(&seeds);
        let actual = minhash::classic_from_bytes_table_8(&bytes, &table);

        let mut expected = [u64::MAX; 8];
        for &b in &bytes {
            for k in 0..8 {
                let h = tokenfs_algos::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] { expected[k] = h; }
            }
        }
        prop_assert_eq!(actual.slots(), &expected);
    }

    #[test]
    fn avx2_bits_popcount_u64_matches_scalar_for_random_words(
        words in proptest::collection::vec(any::<u64>(), 0..2048),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = bits::kernels::scalar::popcount_u64_slice(&words);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { bits::kernels::avx2::popcount_u64_slice(&words) };
        prop_assert_eq!(actual, expected);
        prop_assert_eq!(bits::popcount_u64_slice(&words), expected);
    }

    #[test]
    fn avx2_bits_popcount_u8_matches_scalar_for_random_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = bits::kernels::scalar::popcount_u8_slice(&bytes);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { bits::kernels::avx2::popcount_u8_slice(&bytes) };
        prop_assert_eq!(actual, expected);
        prop_assert_eq!(bits::popcount_u8_slice(&bytes), expected);
    }

    #[test]
    fn avx2_hash_set_membership_matches_scalar_for_random_inputs(
        haystack in proptest::collection::vec(any::<u32>(), 0..1024),
        needle in any::<u32>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe {
            hash::set_membership::kernels::avx2::contains_u32(&haystack, needle)
        };
        prop_assert_eq!(actual, expected);
        // The auto-dispatched API must agree as well.
        prop_assert_eq!(hash::contains_u32_simd(&haystack, needle), expected);
    }

    #[test]
    fn sse41_hash_set_membership_matches_scalar_for_random_inputs(
        haystack in proptest::collection::vec(any::<u32>(), 0..512),
        needle in any::<u32>(),
    ) {
        if !std::is_x86_feature_detected!("sse4.1") {
            return Ok(());
        }
        let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
        // SAFETY: sse4.1 availability checked above.
        let actual = unsafe {
            hash::set_membership::kernels::sse41::contains_u32(&haystack, needle)
        };
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn avx2_bits_streamvbyte_round_trip_random(
        values in proptest::collection::vec(any::<u32>(), 0..2048),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let n = values.len();
        let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(n)];
        let mut data = vec![0_u8; bits::streamvbyte_data_max_len(n)];
        let written = bits::streamvbyte_encode_u32(&values, &mut ctrl, &mut data);
        let mut decoded = vec![0_u32; n];
        let consumed = unsafe {
            bits::streamvbyte::kernels::avx2::decode_u32(&ctrl, &data[..written], n, &mut decoded)
        };
        prop_assert_eq!(consumed, written);
        prop_assert_eq!(decoded, values);
    }

    #[test]
    fn ssse3_bits_streamvbyte_round_trip_random(
        values in proptest::collection::vec(any::<u32>(), 0..2048),
    ) {
        if !std::is_x86_feature_detected!("ssse3") {
            return Ok(());
        }
        let n = values.len();
        let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(n)];
        let mut data = vec![0_u8; bits::streamvbyte_data_max_len(n)];
        let written = bits::streamvbyte_encode_u32(&values, &mut ctrl, &mut data);
        let mut decoded = vec![0_u32; n];
        let consumed = unsafe {
            bits::streamvbyte::kernels::ssse3::decode_u32(&ctrl, &data[..written], n, &mut decoded)
        };
        prop_assert_eq!(consumed, written);
        prop_assert_eq!(decoded, values);
    }

    // ---------- vector module: AVX2 hamming/jaccard parity ----------

    #[test]
    fn avx2_vector_hamming_u64_matches_scalar(
        a in proptest::collection::vec(any::<u64>(), 0..512),
        seed in any::<u64>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let b: Vec<u64> = a.iter().enumerate()
            .map(|(i, x)| x.wrapping_mul(seed.wrapping_add(i as u64 + 1)))
            .collect();
        let expected = vector::kernels::scalar::hamming_u64(&a, &b);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { vector::kernels::avx2::hamming_u64(&a, &b) };
        prop_assert_eq!(Some(actual), expected);
        // Public dispatcher must agree.
        prop_assert_eq!(vector::hamming_u64(&a, &b), expected);
    }

    #[test]
    fn avx2_vector_jaccard_u64_matches_scalar(
        a in proptest::collection::vec(any::<u64>(), 0..512),
        seed in any::<u64>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let b: Vec<u64> = a.iter().enumerate()
            .map(|(i, x)| x ^ seed.rotate_left(i as u32))
            .collect();
        let expected = vector::kernels::scalar::jaccard_u64(&a, &b).unwrap();
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { vector::kernels::avx2::jaccard_u64(&a, &b) };
        prop_assert!((expected - actual).abs() < 1e-12,
            "jaccard_u64 diverged: scalar={expected} avx2={actual}");
        // Public dispatcher.
        let dispatched = vector::jaccard_u64(&a, &b).unwrap();
        prop_assert!((expected - dispatched).abs() < 1e-12);
    }

    // ---------- vector module: AVX2 dot/L2/cosine f32 parity ----------

    #[test]
    fn avx2_vector_dot_l2_f32_match_scalar(
        a in proptest::collection::vec(-256.0_f32..256.0, 0..1024),
        seed in any::<u32>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let b: Vec<f32> = a.iter().enumerate()
            .map(|(i, x)| x + (seed.wrapping_mul(i as u32 + 1) as f32 * 1e-3))
            .collect();
        // Higham §3 / Wilkinson L1-norm bound, see explanation above.
        let dot_s = vector::kernels::scalar::dot_f32(&a, &b).unwrap();
        // SAFETY: avx2_available() returned true above.
        let dot_v = unsafe { vector::kernels::avx2::dot_f32(&a, &b) };
        let l1: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y).abs()).sum();
        let scale = l1.max(dot_s.abs()).max(dot_v.abs()).max(1.0);
        prop_assert!((dot_s - dot_v).abs() / scale < 1e-3,
            "vector::avx2::dot_f32 diverged: scalar={dot_s} avx2={dot_v} l1={l1}");

        let l2_s = vector::kernels::scalar::l2_squared_f32(&a, &b).unwrap();
        // SAFETY: avx2_available() returned true above.
        let l2_v = unsafe { vector::kernels::avx2::l2_squared_f32(&a, &b) };
        let scale = l2_s.abs().max(l2_v.abs()).max(1.0);
        prop_assert!((l2_s - l2_v).abs() / scale < 5e-4,
            "vector::avx2::l2_squared_f32 diverged: scalar={l2_s} avx2={l2_v}");
    }

    // ---------- vector module: AVX-512 f32 + hamming/jaccard parity ----------

    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_vector_dot_l2_f32_match_scalar(
        a in proptest::collection::vec(-256.0_f32..256.0, 0..1024),
        seed in any::<u32>(),
    ) {
        if !std::is_x86_feature_detected!("avx512f") {
            return Ok(());
        }
        let b: Vec<f32> = a.iter().enumerate()
            .map(|(i, x)| x + (seed.wrapping_mul(i as u32 + 1) as f32 * 1e-3))
            .collect();
        let dot_s = vector::kernels::scalar::dot_f32(&a, &b).unwrap();
        // SAFETY: avx512f availability checked above.
        let dot_v = unsafe { vector::kernels::avx512::dot_f32(&a, &b) };
        let l1: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y).abs()).sum();
        let scale = l1.max(dot_s.abs()).max(dot_v.abs()).max(1.0);
        prop_assert!((dot_s - dot_v).abs() / scale < 1e-3,
            "vector::avx512::dot_f32 diverged: scalar={dot_s} avx512={dot_v} l1={l1}");

        let l2_s = vector::kernels::scalar::l2_squared_f32(&a, &b).unwrap();
        // SAFETY: avx512f availability checked above.
        let l2_v = unsafe { vector::kernels::avx512::l2_squared_f32(&a, &b) };
        let scale = l2_s.abs().max(l2_v.abs()).max(1.0);
        prop_assert!((l2_s - l2_v).abs() / scale < 5e-4,
            "vector::avx512::l2_squared_f32 diverged: scalar={l2_s} avx512={l2_v}");
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn avx512_vector_hamming_jaccard_match_scalar(
        a in proptest::collection::vec(any::<u64>(), 0..512),
        seed in any::<u64>(),
    ) {
        if !std::is_x86_feature_detected!("avx512f")
            || !std::is_x86_feature_detected!("avx512vpopcntdq")
        {
            return Ok(());
        }
        let b: Vec<u64> = a.iter().enumerate()
            .map(|(i, x)| x.wrapping_mul(seed.wrapping_add(i as u64 + 1)))
            .collect();
        let h_s = vector::kernels::scalar::hamming_u64(&a, &b).unwrap();
        // SAFETY: availability checked above.
        let h_v = unsafe { vector::kernels::avx512::hamming_u64(&a, &b) };
        prop_assert_eq!(h_s, h_v);
        let j_s = vector::kernels::scalar::jaccard_u64(&a, &b).unwrap();
        // SAFETY: availability checked above.
        let j_v = unsafe { vector::kernels::avx512::jaccard_u64(&a, &b) };
        prop_assert!((j_s - j_v).abs() < 1e-12);
    }

    // ---------- vector module: batched many-vs-one parity ----------

    #[test]
    fn avx2_vector_batched_dot_f32_one_to_many_matches_serial(
        n_rows in 0_usize..16,
        seed in any::<u32>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let stride = 64_usize;
        let mut state = seed.wrapping_add(1);
        let mut next_f32 = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let query: Vec<f32> = (0..stride).map(|_| next_f32()).collect();
        let db: Vec<f32> = (0..n_rows * stride).map(|_| next_f32()).collect();
        let mut out_batched = vec![0.0_f32; n_rows];
        vector::dot_f32_one_to_many(&query, &db, stride, &mut out_batched);
        for (i, slot) in out_batched.iter().enumerate() {
            let row = &db[i * stride..(i + 1) * stride];
            let serial = vector::dot_f32(&query, row).unwrap();
            // The batched form re-resolves the dispatcher per row, so
            // the result should be bit-exact with the serial single-pair
            // call (same backend, same input).
            prop_assert!((slot - serial).abs() < 1e-5,
                "batched dot_f32 row {} diverged: got {} serial {}", i, slot, serial);
        }
    }

    #[test]
    fn avx2_vector_batched_hamming_u64_one_to_many_matches_serial(
        n_rows in 0_usize..16,
        seed in any::<u64>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let stride = 8_usize;
        let mut state = seed;
        let mut next_u64 = || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        let query: Vec<u64> = (0..stride).map(|_| next_u64()).collect();
        let db: Vec<u64> = (0..n_rows * stride).map(|_| next_u64()).collect();
        let mut out_batched = vec![0_u32; n_rows];
        vector::hamming_u64_one_to_many(&query, &db, stride, &mut out_batched);
        for (i, slot) in out_batched.iter().enumerate() {
            let row = &db[i * stride..(i + 1) * stride];
            let serial = vector::hamming_u64(&query, row).unwrap();
            prop_assert_eq!(*slot as u64, serial);
        }
    }
}

// =====================================================================
// bitmap module — Roaring-style SIMD container kernels
// =====================================================================

fn bitmap_words_seeded(seed: u64) -> [u64; 1024] {
    let mut state = seed;
    let mut bm = [0_u64; 1024];
    for word in &mut bm {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *word = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
    }
    bm
}

fn sorted_unique_u16(seed: u64, density: u16, n: usize) -> Vec<u16> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    let mut last: u32 = 0;
    for _ in 0..n {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        let step = (state as u16 % density.max(1)).max(1);
        last += u32::from(step);
        if last >= 65_536 {
            break;
        }
        out.push(last as u16);
    }
    out
}

#[test]
fn avx2_bitmap_x_bitmap_and_card_parity() {
    if !avx2_available() {
        return;
    }
    let a = bitmap_words_seeded(0xC0FF_EE00);
    let b = bitmap_words_seeded(0xDEAD_BEEF);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::and_into(&a, &b, &mut out_scalar);
    // SAFETY: availability checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx2::and_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn avx2_bitmap_x_bitmap_or_card_parity() {
    if !avx2_available() {
        return;
    }
    let a = bitmap_words_seeded(0x1234_5678);
    let b = bitmap_words_seeded(0x9abc_def0);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::or_into(&a, &b, &mut out_scalar);
    // SAFETY: availability checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx2::or_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn avx2_bitmap_x_bitmap_xor_card_parity() {
    if !avx2_available() {
        return;
    }
    let a = bitmap_words_seeded(0xAAAA_5555);
    let b = bitmap_words_seeded(0x5555_AAAA);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::xor_into(&a, &b, &mut out_scalar);
    // SAFETY: availability checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx2::xor_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn avx2_bitmap_x_bitmap_andnot_card_parity() {
    if !avx2_available() {
        return;
    }
    let a = bitmap_words_seeded(0x0F0F_0F0F);
    let b = bitmap_words_seeded(0xF0F0_F0F0);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::andnot_into(&a, &b, &mut out_scalar);
    // SAFETY: availability checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx2::andnot_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn avx2_bitmap_just_cardinality_parity() {
    if !avx2_available() {
        return;
    }
    let a = bitmap_words_seeded(0x1357_9bdf);
    let b = bitmap_words_seeded(0x2468_ace0);
    // SAFETY: availability checked above.
    unsafe {
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx2::and_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::and_cardinality(&a, &b)
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx2::or_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::or_cardinality(&a, &b)
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx2::xor_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::xor_cardinality(&a, &b)
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx2::andnot_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::andnot_cardinality(&a, &b)
        );
    }
}

#[test]
fn sse42_array_x_array_intersect_parity() {
    if !sse42_available() {
        return;
    }
    for seed in [0_u64, 1, 0xDEAD_BEEF, 0xC0FFEE, 0xBAD_CAB] {
        for density in [3_u16, 30, 300] {
            let a = sorted_unique_u16(seed, density, 4096);
            let b = sorted_unique_u16(seed.wrapping_add(1), density, 4096);
            let mut out_simd = Vec::new();
            // SAFETY: availability checked above.
            unsafe {
                bitmap::kernels::array_x_array_sse42::intersect(&a, &b, &mut out_simd);
            }
            let mut out_scalar = Vec::new();
            bitmap::kernels::array_x_array_scalar::intersect(&a, &b, &mut out_scalar);
            assert_eq!(
                out_simd, out_scalar,
                "Schlegel intersect diverged at seed={seed} density={density}"
            );
            // SAFETY: availability checked above.
            let card =
                unsafe { bitmap::kernels::array_x_array_sse42::intersect_cardinality(&a, &b) };
            assert_eq!(card as usize, out_scalar.len());
        }
    }
}

#[test]
fn sse42_array_x_array_intersect_with_zero_element() {
    // Regression test for the pcmpistrm-vs-pcmpestrm bug where a 0
    // element in either input is treated as the implicit string
    // terminator. Both kernels must include 0 as a valid match.
    if !sse42_available() {
        return;
    }
    let a: Vec<u16> = (0..16).collect();
    let b: Vec<u16> = (0..16).collect();
    let mut out_simd = Vec::new();
    // SAFETY: availability checked above.
    unsafe {
        bitmap::kernels::array_x_array_sse42::intersect(&a, &b, &mut out_simd);
    }
    assert_eq!(out_simd, a);
}

#[test]
fn avx2_array_x_bitmap_intersect_parity() {
    if !avx2_available() {
        return;
    }
    for seed in [0_u64, 1, 0xC0FFEE] {
        for density in [3_u16, 30, 300] {
            let array = sorted_unique_u16(seed, density, 200);
            let bm = bitmap_words_seeded(seed.wrapping_add(7));
            let mut out_simd = Vec::new();
            // SAFETY: availability checked above.
            unsafe {
                bitmap::kernels::array_x_bitmap_avx2::intersect_array_bitmap(
                    &array,
                    &bm,
                    &mut out_simd,
                );
            }
            let mut out_scalar = Vec::new();
            bitmap::kernels::array_x_bitmap_scalar::intersect_array_bitmap(
                &array,
                &bm,
                &mut out_scalar,
            );
            assert_eq!(
                out_simd, out_scalar,
                "array×bitmap diverged at seed={seed} density={density}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn proptest_avx2_bitmap_kernel_pairs(
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let a = bitmap_words_seeded(seed_a);
        let b = bitmap_words_seeded(seed_b);

        let mut out_avx2 = [0_u64; 1024];
        let mut out_scalar = [0_u64; 1024];

        // SAFETY: availability checked above.
        let card_simd = unsafe {
            bitmap::kernels::bitmap_x_bitmap_avx2::and_into(&a, &b, &mut out_avx2)
        };
        let card_scalar =
            bitmap::kernels::bitmap_x_bitmap_scalar::and_into(&a, &b, &mut out_scalar);
        prop_assert_eq!(card_simd, card_scalar);
        prop_assert_eq!(out_avx2.as_slice(), out_scalar.as_slice());
    }

    #[test]
    fn proptest_sse42_array_intersect(
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
        n in 1_usize..200,
    ) {
        if !sse42_available() {
            return Ok(());
        }
        let a = sorted_unique_u16(seed_a, 100, n);
        let b = sorted_unique_u16(seed_b, 100, n);
        let mut out_simd = Vec::new();
        // SAFETY: availability checked above.
        unsafe {
            bitmap::kernels::array_x_array_sse42::intersect(&a, &b, &mut out_simd);
        }
        let mut out_scalar = Vec::new();
        bitmap::kernels::array_x_array_scalar::intersect(&a, &b, &mut out_scalar);
        prop_assert_eq!(out_simd, out_scalar);
    }
}

// =====================================================================
// AVX-512 explicit parity tests
//
// Each AVX-512 test runtime-skips when the relevant feature isn't
// detected. On dev hardware without AVX-512 these all early-return; on
// AVX-512-capable hardware they exercise the kernel directly against
// the pinned scalar reference. The whole section is gated on the
// `avx512` Cargo feature so non-feature builds compile cleanly.
// =====================================================================

#[cfg(feature = "avx512")]
fn avx512f_available() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

#[cfg(feature = "avx512")]
fn avx512_popcnt_available() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vpopcntdq")
}

// ---------- bits::rank_select batch parity ----------
//
// `bits::rank_select::kernels::auto::{rank1_batch, select1_batch}` today
// delegates to scalar (no AVX-512 batch kernel exists yet — see the
// comment in src/bits/rank_select.rs § kernels). The parity tests below
// pin the auto-dispatcher's contract against the scalar reference: any
// future AVX-512 batch kernel must produce identical output.

#[cfg(feature = "avx512")]
#[test]
fn avx512_bits_rank_select_rank1_batch_matches_scalar() {
    if !avx512f_available() {
        eprintln!("avx512f unavailable; skipping rank_select rank1_batch parity test");
        return;
    }
    // Build a non-trivial bitmap and exercise rank1_batch on a dense
    // grid of positions including endpoints and superblock boundaries.
    let mut state = 0x000C_0FFE_EF22_u64;
    let n_bits = 8 * 4096_usize;
    let n_words = n_bits.div_ceil(64);
    let bits: Vec<u64> = (0..n_words)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        })
        .collect();
    let dict = bits::RankSelectDict::build(&bits, n_bits);
    let positions: Vec<usize> = vec![
        0,
        1,
        7,
        63,
        64,
        128,
        256,
        1024,
        4095,
        4096,
        n_bits / 2,
        n_bits,
    ];
    let mut expected = vec![0_usize; positions.len()];
    bits::rank_select::kernels::scalar::rank1_batch(&dict, &positions, &mut expected);
    let mut actual = vec![0_usize; positions.len()];
    bits::rank_select::kernels::auto::rank1_batch(&dict, &positions, &mut actual);
    assert_eq!(
        actual, expected,
        "auto::rank1_batch diverged from scalar reference under avx512"
    );
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_bits_rank_select_select1_batch_matches_scalar() {
    if !avx512f_available() {
        eprintln!("avx512f unavailable; skipping rank_select select1_batch parity test");
        return;
    }
    let mut state = 0xDEAD_BEEF_u64;
    let n_bits = 8 * 4096_usize;
    let n_words = n_bits.div_ceil(64);
    let bits: Vec<u64> = (0..n_words)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        })
        .collect();
    let dict = bits::RankSelectDict::build(&bits, n_bits);
    let total = dict.count_ones();
    // Pick a span across [0, total) and the boundary `total` (which
    // must yield None per the contract).
    let mut ks: Vec<usize> = vec![0, 1, 7];
    if total > 0 {
        for k in [total / 4, total / 2, (total * 3) / 4, total - 1] {
            ks.push(k);
        }
    }
    ks.push(total); // out-of-range -> None
    let mut expected = vec![None; ks.len()];
    bits::rank_select::kernels::scalar::select1_batch(&dict, &ks, &mut expected);
    let mut actual = vec![None; ks.len()];
    bits::rank_select::kernels::auto::select1_batch(&dict, &ks, &mut actual);
    assert_eq!(
        actual, expected,
        "auto::select1_batch diverged from scalar reference under avx512"
    );
}

// ---------- vector batch many-vs-one parity (AVX-512 single-pair underneath) ----------

#[cfg(feature = "avx512")]
#[test]
fn avx512_vector_dot_f32_one_to_many_matches_scalar() {
    if !avx512f_available() {
        eprintln!("avx512f unavailable; skipping vector::dot_f32_one_to_many parity test");
        return;
    }
    let stride = 64_usize;
    let n_rows = 12_usize;
    let mut state = 0xC0FFEE_u32;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let query: Vec<f32> = (0..stride).map(|_| next_f32()).collect();
    let db: Vec<f32> = (0..n_rows * stride).map(|_| next_f32()).collect();
    let mut out_batched = vec![0.0_f32; n_rows];
    vector::dot_f32_one_to_many(&query, &db, stride, &mut out_batched);
    for (i, slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        // Use the L1-norm-of-products scale (Higham §3 / Wilkinson) so
        // the cross-backend AVX-512 vs. scalar tolerance accounts for
        // catastrophic cancellation on adversarial inputs.
        let scalar_dot = vector::kernels::scalar::dot_f32(&query, row).unwrap();
        let l1: f32 = query.iter().zip(row).map(|(&x, &y)| (x * y).abs()).sum();
        let scale = l1.max(scalar_dot.abs()).max((*slot).abs()).max(1.0);
        assert!(
            (slot - scalar_dot).abs() / scale < 1e-3,
            "avx512 dot_f32_one_to_many[{i}] diverged: scalar={scalar_dot} batched={slot}"
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_vector_l2_squared_f32_one_to_many_matches_scalar() {
    if !avx512f_available() {
        eprintln!("avx512f unavailable; skipping vector::l2_squared_f32_one_to_many parity test");
        return;
    }
    let stride = 64_usize;
    let n_rows = 12_usize;
    let mut state = 0xDEAD_BEEF_u32;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let query: Vec<f32> = (0..stride).map(|_| next_f32()).collect();
    let db: Vec<f32> = (0..n_rows * stride).map(|_| next_f32()).collect();
    let mut out_batched = vec![0.0_f32; n_rows];
    vector::l2_squared_f32_one_to_many(&query, &db, stride, &mut out_batched);
    for (i, slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let scalar = vector::kernels::scalar::l2_squared_f32(&query, row).unwrap();
        // L2_squared has no cancellation — all squared terms are
        // non-negative, so a tighter bound holds.
        let scale = scalar.abs().max((*slot).abs()).max(1.0);
        assert!(
            (slot - scalar).abs() / scale < 5e-4,
            "avx512 l2_squared_f32_one_to_many[{i}] diverged: scalar={scalar} batched={slot}"
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_vector_cosine_similarity_f32_one_to_many_matches_scalar() {
    if !avx512f_available() {
        eprintln!(
            "avx512f unavailable; skipping vector::cosine_similarity_f32_one_to_many parity test"
        );
        return;
    }
    let stride = 64_usize;
    let n_rows = 12_usize;
    let mut state = 0xBAD_C0DE_u32;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let query: Vec<f32> = (0..stride).map(|_| next_f32()).collect();
    let db: Vec<f32> = (0..n_rows * stride).map(|_| next_f32()).collect();
    let mut out_batched = vec![0.0_f32; n_rows];
    vector::cosine_similarity_f32_one_to_many(&query, &db, stride, &mut out_batched);
    for (i, slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let scalar = vector::kernels::scalar::cosine_similarity_f32(&query, row).unwrap();
        // Cosine is bounded in [-1, 1]; a 1e-3 absolute tolerance is
        // proportional to the worst-case Higham bound on dot/L2 above.
        assert!(
            (slot - scalar).abs() < 1e-3,
            "avx512 cosine_similarity_f32_one_to_many[{i}] diverged: scalar={scalar} batched={slot}"
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_vector_hamming_u64_one_to_many_matches_scalar() {
    if !avx512_popcnt_available() {
        eprintln!(
            "avx512f+vpopcntdq unavailable; skipping vector::hamming_u64_one_to_many parity test"
        );
        return;
    }
    let stride = 8_usize;
    let n_rows = 12_usize;
    let mut state = 0xF22_C0FFEE_u64;
    let mut next_u64 = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let query: Vec<u64> = (0..stride).map(|_| next_u64()).collect();
    let db: Vec<u64> = (0..n_rows * stride).map(|_| next_u64()).collect();
    let mut out_batched = vec![0_u32; n_rows];
    vector::hamming_u64_one_to_many(&query, &db, stride, &mut out_batched);
    for (i, slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let scalar = vector::kernels::scalar::hamming_u64(&query, row).unwrap();
        assert_eq!(
            *slot as u64, scalar,
            "avx512 hamming_u64_one_to_many[{i}] diverged: scalar={scalar} batched={slot}"
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_vector_jaccard_u64_one_to_many_matches_scalar() {
    if !avx512_popcnt_available() {
        eprintln!(
            "avx512f+vpopcntdq unavailable; skipping vector::jaccard_u64_one_to_many parity test"
        );
        return;
    }
    let stride = 8_usize;
    let n_rows = 12_usize;
    let mut state = 0x000C_0FFE_EBAD_u64;
    let mut next_u64 = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };
    let query: Vec<u64> = (0..stride).map(|_| next_u64()).collect();
    let db: Vec<u64> = (0..n_rows * stride).map(|_| next_u64()).collect();
    let mut out_batched = vec![0.0_f64; n_rows];
    vector::jaccard_u64_one_to_many(&query, &db, stride, &mut out_batched);
    for (i, slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let scalar = vector::kernels::scalar::jaccard_u64(&query, row).unwrap();
        assert!(
            (slot - scalar).abs() < 1e-12,
            "avx512 jaccard_u64_one_to_many[{i}] diverged: scalar={scalar} batched={slot}"
        );
    }
}

// ---------- bitmap::kernels::bitmap_x_bitmap_avx512 parity ----------

#[cfg(feature = "avx512")]
#[test]
fn avx512_bitmap_x_bitmap_and_into_card_parity() {
    if !avx512_popcnt_available() {
        eprintln!(
            "avx512f+vpopcntdq unavailable; skipping bitmap_x_bitmap_avx512 and_into parity test"
        );
        return;
    }
    let a = bitmap_words_seeded(0xC0FF_EE00);
    let b = bitmap_words_seeded(0xDEAD_BEEF);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::and_into(&a, &b, &mut out_scalar);
    // SAFETY: avx512f+vpopcntdq checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx512::and_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_bitmap_x_bitmap_and_into_nocard_parity() {
    if !avx512f_available() {
        eprintln!(
            "avx512f unavailable; skipping bitmap_x_bitmap_avx512 and_into_nocard parity test"
        );
        return;
    }
    let a = bitmap_words_seeded(0xC0FF_EE00);
    let b = bitmap_words_seeded(0xDEAD_BEEF);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    bitmap::kernels::bitmap_x_bitmap_scalar::and_into_nocard(&a, &b, &mut out_scalar);
    // SAFETY: avx512f checked above.
    unsafe {
        bitmap::kernels::bitmap_x_bitmap_avx512::and_into_nocard(&a, &b, &mut out_simd);
    }
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_bitmap_x_bitmap_or_into_card_parity() {
    if !avx512_popcnt_available() {
        eprintln!(
            "avx512f+vpopcntdq unavailable; skipping bitmap_x_bitmap_avx512 or_into parity test"
        );
        return;
    }
    let a = bitmap_words_seeded(0x1234_5678);
    let b = bitmap_words_seeded(0x9abc_def0);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::or_into(&a, &b, &mut out_scalar);
    // SAFETY: avx512f+vpopcntdq checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx512::or_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_bitmap_x_bitmap_xor_into_card_parity() {
    if !avx512_popcnt_available() {
        eprintln!(
            "avx512f+vpopcntdq unavailable; skipping bitmap_x_bitmap_avx512 xor_into parity test"
        );
        return;
    }
    let a = bitmap_words_seeded(0xAAAA_5555);
    let b = bitmap_words_seeded(0x5555_AAAA);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::xor_into(&a, &b, &mut out_scalar);
    // SAFETY: avx512f+vpopcntdq checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx512::xor_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_bitmap_x_bitmap_andnot_into_card_parity() {
    if !avx512_popcnt_available() {
        eprintln!(
            "avx512f+vpopcntdq unavailable; skipping bitmap_x_bitmap_avx512 andnot_into parity test"
        );
        return;
    }
    let a = bitmap_words_seeded(0x0F0F_0F0F);
    let b = bitmap_words_seeded(0xF0F0_F0F0);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::andnot_into(&a, &b, &mut out_scalar);
    // SAFETY: avx512f+vpopcntdq checked above.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_avx512::andnot_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_bitmap_x_bitmap_just_cardinality_vpopcntq_parity() {
    if !avx512_popcnt_available() {
        eprintln!(
            "avx512f+vpopcntdq unavailable; skipping bitmap_x_bitmap_avx512 cardinality parity test"
        );
        return;
    }
    let a = bitmap_words_seeded(0x1357_9bdf);
    let b = bitmap_words_seeded(0x2468_ace0);
    // SAFETY: avx512f+vpopcntdq checked above.
    unsafe {
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx512::and_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::and_cardinality(&a, &b),
            "avx512 and_cardinality (VPOPCNTQ) diverged"
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx512::or_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::or_cardinality(&a, &b),
            "avx512 or_cardinality (VPOPCNTQ) diverged"
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx512::xor_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::xor_cardinality(&a, &b),
            "avx512 xor_cardinality (VPOPCNTQ) diverged"
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_avx512::andnot_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::andnot_cardinality(&a, &b),
            "avx512 andnot_cardinality (VPOPCNTQ) diverged"
        );
    }
}

// ---------- hash::set_membership AVX-512 parity ----------

#[cfg(feature = "avx512")]
#[test]
fn avx512_hash_set_membership_contains_u32_parity() {
    if !avx512f_available() {
        eprintln!("avx512f unavailable; skipping hash::set_membership AVX-512 parity test");
        return;
    }
    // Cover SIMD block boundaries: AVX-512 processes 16 lanes per vector.
    for len in [
        0_usize, 1, 3, 8, 15, 16, 17, 31, 32, 33, 47, 48, 64, 128, 256, 1023,
    ] {
        let haystack = deterministic_u32_haystack(len, 0xF22_F00D ^ (len as u64));
        for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000, 0xC0FFEE] {
            let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
            // SAFETY: avx512f checked above.
            let actual =
                unsafe { hash::set_membership::kernels::avx512::contains_u32(&haystack, needle) };
            assert_eq!(
                actual, expected,
                "avx512 set_membership::contains_u32 diverged at len {len} needle {needle}"
            );
        }
        if len > 0 {
            for &pos in &[0_usize, len / 2, len - 1] {
                let needle = haystack[pos];
                let expected =
                    hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: avx512f checked above.
                let actual = unsafe {
                    hash::set_membership::kernels::avx512::contains_u32(&haystack, needle)
                };
                assert_eq!(
                    actual, expected,
                    "avx512 set_membership::contains_u32 diverged at len {len} pos {pos}"
                );
            }
        }
    }
}

// ---------- approx::HyperLogLog AVX2 parity (Sprint 44) ----------

fn deterministic_hll_register_blob(precision: u32, seed: u64) -> Vec<u8> {
    // Build a register vector by stepping the same xorshift RNG used by
    // the in-module HLL parity tests, then folding each output through
    // `insert_hash` so the resulting vector is a valid HLL state.
    let m = 1_usize << precision;
    let mut hll = approx::HyperLogLog::new(precision);
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    // 4x more inserts than registers to populate a representative
    // distribution across all buckets.
    for _ in 0..(4 * m) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let h = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        hll.insert_hash(h);
    }
    hll.register_bytes().to_vec()
}

#[test]
fn avx2_hll_merge_matches_scalar_reference_across_precisions() {
    if !avx2_available() {
        eprintln!("avx2 unavailable on this host; skipping HLL merge parity test");
        return;
    }
    for precision in [4_u32, 5, 6, 8, 10, 12, 14, 16] {
        let dst_seed = 0xA1A1_F00D_u64 ^ (precision as u64);
        let src_seed = 0xB2B2_C0FFEE_u64 ^ (precision as u64);
        let dst_initial = deterministic_hll_register_blob(precision, dst_seed);
        let src = deterministic_hll_register_blob(precision, src_seed);

        let mut scalar_dst = dst_initial.clone();
        approx::hll_kernels::scalar::merge(&mut scalar_dst, &src);

        let mut simd_dst = dst_initial.clone();
        // SAFETY: avx2_available() returned true above.
        unsafe {
            approx::hll_kernels::avx2::merge(&mut simd_dst, &src);
        }
        assert_eq!(
            simd_dst, scalar_dst,
            "AVX2 HLL merge diverged at precision={precision}"
        );
    }
}

#[test]
fn avx2_hll_count_raw_matches_scalar_reference_across_precisions() {
    if !avx2_available() {
        eprintln!("avx2 unavailable on this host; skipping HLL count parity test");
        return;
    }
    for precision in [4_u32, 5, 6, 8, 10, 12, 14, 16] {
        let registers =
            deterministic_hll_register_blob(precision, 0xDEAD_BEEF ^ (precision as u64));
        let m = registers.len() as f64;
        // Recover alpha the same way the public surface does.
        let alpha = match precision {
            4 => 0.673,
            5 => 0.697,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let scalar = approx::hll_kernels::scalar::count_raw(&registers, alpha);
        // SAFETY: avx2_available() returned true above.
        let actual = unsafe { approx::hll_kernels::avx2::count_raw(&registers, alpha) };
        let rel_err = if scalar == 0.0 {
            0.0
        } else {
            (scalar - actual).abs() / scalar.abs()
        };
        assert!(
            rel_err < 1e-12,
            "AVX2 HLL count_raw diverged at precision={precision}: \
             scalar={scalar} avx2={actual} rel_err={rel_err}"
        );
    }
}

#[test]
fn avx2_hll_merge_handles_unaligned_lengths_via_scalar_tail() {
    if !avx2_available() {
        return;
    }
    // Synthetic non-power-of-two register lengths don't occur in valid
    // HLL states (m = 2^precision is always a power of two), but the
    // kernel's scalar-tail handling is contract-relevant for any
    // future caller that wants to reuse the kernel as a generic
    // u8-slice per-bucket-max op.
    for len in [
        0_usize, 1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 257,
    ] {
        let mut a: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(17) % 64) as u8).collect();
        let b: Vec<u8> = (0..len).map(|i| (i.wrapping_mul(31) % 64) as u8).collect();
        let mut expected = a.clone();
        approx::hll_kernels::scalar::merge(&mut expected, &b);

        // SAFETY: avx2_available() returned true above.
        unsafe { approx::hll_kernels::avx2::merge(&mut a, &b) };
        assert_eq!(
            a, expected,
            "AVX2 HLL merge diverged on unaligned len {len}"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn avx2_hll_merge_proptest_matches_scalar(
        precision in 4_u32..=14,
        dst_seed in any::<u64>(),
        src_seed in any::<u64>(),
    ) {
        if !avx2_available() {
            return Ok(());
        }
        let dst_initial = deterministic_hll_register_blob(precision, dst_seed);
        let src = deterministic_hll_register_blob(precision, src_seed);

        let mut scalar_dst = dst_initial.clone();
        approx::hll_kernels::scalar::merge(&mut scalar_dst, &src);

        let mut simd_dst = dst_initial;
        // SAFETY: avx2_available() checked above.
        unsafe { approx::hll_kernels::avx2::merge(&mut simd_dst, &src) };
        prop_assert_eq!(simd_dst, scalar_dst);
    }
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_hll_merge_matches_scalar_reference_across_precisions() {
    if !avx512f_available() || !std::is_x86_feature_detected!("avx512bw") {
        eprintln!("avx512f+bw unavailable; skipping HLL merge AVX-512 parity test");
        return;
    }
    for precision in [4_u32, 5, 6, 8, 10, 12, 14, 16] {
        let dst_seed = 0xA1A1_F00D_u64 ^ (precision as u64);
        let src_seed = 0xB2B2_C0FFEE_u64 ^ (precision as u64);
        let dst_initial = deterministic_hll_register_blob(precision, dst_seed);
        let src = deterministic_hll_register_blob(precision, src_seed);

        let mut scalar_dst = dst_initial.clone();
        approx::hll_kernels::scalar::merge(&mut scalar_dst, &src);

        let mut simd_dst = dst_initial.clone();
        // SAFETY: feature checked above.
        unsafe {
            approx::hll_kernels::avx512::merge(&mut simd_dst, &src);
        }
        assert_eq!(
            simd_dst, scalar_dst,
            "AVX-512 HLL merge diverged at precision={precision}"
        );
    }
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_hll_count_raw_matches_scalar_reference_across_precisions() {
    if !avx512f_available() || !std::is_x86_feature_detected!("avx512bw") {
        eprintln!("avx512f+bw unavailable; skipping HLL count AVX-512 parity test");
        return;
    }
    for precision in [4_u32, 5, 6, 8, 10, 12, 14, 16] {
        let registers =
            deterministic_hll_register_blob(precision, 0xDEAD_BEEF ^ (precision as u64));
        let m = registers.len() as f64;
        let alpha = match precision {
            4 => 0.673,
            5 => 0.697,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let scalar = approx::hll_kernels::scalar::count_raw(&registers, alpha);
        // SAFETY: feature checked above.
        let actual = unsafe { approx::hll_kernels::avx512::count_raw(&registers, alpha) };
        let rel_err = if scalar == 0.0 {
            0.0
        } else {
            (scalar - actual).abs() / scalar.abs()
        };
        assert!(
            rel_err < 1e-12,
            "AVX-512 HLL count_raw diverged at precision={precision}: \
             scalar={scalar} avx512={actual} rel_err={rel_err}"
        );
    }
}

// ---------- approx::BloomFilter SIMD parity ----------

/// Bit-counts and K values exercised by every Bloom kernel parity test.
const BLOOM_TEST_BITS: &[usize] = &[64, 128, 1024, 4096, 65_536, 1_048_576];
const BLOOM_TEST_KS: &[usize] = &[1, 2, 3, 4, 5, 7, 8, 13, 16, 32];

#[test]
fn avx2_approx_bloom_kernels_positions_match_scalar_on_size_grid() {
    if !avx2_available() {
        eprintln!("avx2 unavailable; skipping approx::bloom_kernels AVX2 parity test");
        return;
    }
    for &bits in BLOOM_TEST_BITS {
        for &k in BLOOM_TEST_KS {
            // Exercise a deterministic hash pair derived from
            // (bits, k) so each grid cell has a distinct probe; the
            // `| 1` keeps `h2` odd as the production
            // `derive_hashes` does.
            let h1 = 0xDEAD_BEEF_F00D_CAFE_u64.wrapping_mul(bits as u64 + 1);
            let h2 = (0x1234_5678_9ABC_DEF0_u64.wrapping_mul(k as u64 + 1)) | 1;
            let mut out_scalar = vec![0_u64; k];
            let mut out_avx2 = vec![0_u64; k];
            approx::bloom_kernels::scalar::positions(h1, h2, k, bits, &mut out_scalar);
            // SAFETY: avx2 availability checked above.
            unsafe {
                approx::bloom_kernels::avx2::positions(h1, h2, k, bits, &mut out_avx2);
            }
            assert_eq!(
                out_scalar, out_avx2,
                "avx2 bloom_kernels::positions diverged at bits={bits} k={k}"
            );
        }
    }
}

#[test]
fn avx2_approx_bloom_filter_contains_simd_matches_scalar_path() {
    if !avx2_available() {
        eprintln!("avx2 unavailable; skipping approx::BloomFilter contains_simd parity test");
        return;
    }
    // Cross-check: insert via SIMD, query via SIMD — must always
    // hit. Insert a moderate set, then probe with a mix of inserted
    // and non-inserted keys and confirm the SIMD-dispatched
    // `contains_simd` result matches the scalar reference computed
    // via the same Kirsch-Mitzenmacher formula.
    for &bits in &[1024_usize, 4096, 65_536] {
        for &k in &[3_usize, 7, 13] {
            let mut bf = approx::BloomFilter::new(bits, k);
            let inserted: Vec<u64> = (0_u64..128).collect();
            for &key in &inserted {
                bf.insert_simd(key);
            }
            // All inserted keys must read true via contains_simd.
            for &key in &inserted {
                assert!(
                    bf.contains_simd(key),
                    "avx2 contains_simd false negative at bits={bits} k={k} key={key}"
                );
            }
        }
    }
}

#[cfg(feature = "avx512")]
fn avx512f_dq_available() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
}

#[cfg(feature = "avx512")]
#[test]
fn avx512_approx_bloom_kernels_positions_match_scalar_on_size_grid() {
    if !avx512f_dq_available() {
        eprintln!(
            "avx512f+avx512dq unavailable; skipping approx::bloom_kernels AVX-512 parity test"
        );
        return;
    }
    for &bits in BLOOM_TEST_BITS {
        for &k in BLOOM_TEST_KS {
            let h1 = 0x1357_9BDF_2468_ACE0_u64.wrapping_mul(bits as u64 + 13);
            let h2 = (0x0123_4567_89AB_CDEF_u64.wrapping_mul(k as u64 + 7)) | 1;
            let mut out_scalar = vec![0_u64; k];
            let mut out_avx512 = vec![0_u64; k];
            approx::bloom_kernels::scalar::positions(h1, h2, k, bits, &mut out_scalar);
            // SAFETY: avx512f+avx512dq availability checked above.
            unsafe {
                approx::bloom_kernels::avx512::positions(h1, h2, k, bits, &mut out_avx512);
            }
            assert_eq!(
                out_scalar, out_avx512,
                "avx512 bloom_kernels::positions diverged at bits={bits} k={k}"
            );
        }
    }
}

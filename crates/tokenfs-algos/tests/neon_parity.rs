//! Parity tests for the AArch64 NEON byte-class kernel.
//!
//! Same shape as `tests/avx2_parity.rs`: cfg-gated to the relevant ISA so
//! non-aarch64 builds compile cleanly, and runtime-tolerant on the aarch64
//! side (NEON is mandatory in the AArch64 ABI, so no runtime-skip path is
//! needed).
//!
//! On x86 hosts this file is empty after preprocessing — `cargo test` will
//! report "0 tests" for the suite. When run on real aarch64 hardware (or a
//! properly configured cross-test environment with sysroot + QEMU) it
#![allow(deprecated)]
//! exercises the NEON kernel against the pinned scalar reference.

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)] // Test code — panic on None/Err is the desired failure mode.
#![cfg(all(feature = "neon", target_arch = "aarch64"))]

use proptest::prelude::*;
use tokenfs_algos::{
    bitmap, bits, byteclass, fingerprint, hash, runlength, similarity, sketch, vector,
};

fn synthetic_corpus() -> Vec<Vec<u8>> {
    let mut cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        vec![0],
        vec![0xff],
        vec![0; 4096],
        vec![0xff; 4096],
        (0_u8..=255).collect(),
        (0_u8..=255).cycle().take(4096).collect(),
        b"tokenfs-algos NEON parity \xe2\x9c\x85 mixed UTF-8\nline two\r\n\tindented"
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
fn neon_byteclass_matches_scalar_reference_on_synthetic_corpus() {
    for input in synthetic_corpus() {
        let expected = byteclass::kernels::scalar::classify(&input);
        // SAFETY: NEON is mandatory on AArch64 builds.
        let actual = unsafe { byteclass::kernels::neon::classify(&input) };
        assert_eq!(
            actual,
            expected,
            "neon byteclass classify diverged on length {}",
            input.len()
        );
    }
}

#[test]
fn neon_byteclass_matches_scalar_reference_on_unaligned_subslices() {
    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = byteclass::kernels::scalar::classify(slice);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { byteclass::kernels::neon::classify(slice) };
        assert_eq!(
            actual,
            expected,
            "neon byteclass classify diverged on slice len {}",
            slice.len()
        );
    }
}

#[test]
fn neon_dispatched_path_matches_scalar_reference() {
    for input in synthetic_corpus() {
        let expected = byteclass::kernels::scalar::classify(&input);
        let dispatched = byteclass::classify(&input);
        assert_eq!(
            dispatched,
            expected,
            "auto-dispatched byteclass diverged on length {}",
            input.len()
        );
    }
}

/// Mirror of `tests/avx2_parity.rs::utf8_corpus()` so the two SIMD paths are
/// exercised against an identical corpus.
fn utf8_corpus() -> Vec<(&'static str, Vec<u8>)> {
    let mut cases: Vec<(&'static str, Vec<u8>)> = vec![
        ("empty", Vec::new()),
        ("ascii-1", b"a".to_vec()),
        ("ascii-63", vec![b'x'; 63]),
        ("ascii-64", vec![b'x'; 64]),
        ("ascii-65", vec![b'x'; 65]),
        ("ascii-1024", vec![b'A'; 1024]),
        ("two-byte-utf8", "héllo wörld".as_bytes().to_vec()),
        ("three-byte-utf8", "你好,世界".as_bytes().to_vec()),
        ("four-byte-utf8", "𝐇𝐞𝐥𝐥𝐨".as_bytes().to_vec()),
        (
            "mixed-utf8",
            "ASCII tail with a 🎉 emoji and 日本語 then more."
                .as_bytes()
                .to_vec(),
        ),
        ("lone-cont-0x80", vec![b'a', b'b', 0x80, b'c']),
        ("lone-cont-0xbf", vec![0xbf]),
        ("invalid-c0", vec![0xc0, 0x80]),
        ("invalid-c1", vec![0xc1, 0x80]),
        ("invalid-fe", vec![0xfe]),
        ("invalid-ff", vec![0xff]),
        ("truncated-2byte", vec![b'a', 0xc2]),
        ("truncated-3byte-1", vec![0xe2]),
        ("truncated-3byte-2", vec![0xe2, 0x82]),
        ("truncated-4byte-1", vec![0xf0]),
        ("truncated-4byte-2", vec![0xf0, 0x9f]),
        ("truncated-4byte-3", vec![0xf0, 0x9f, 0x98]),
        ("overlong-2byte", vec![0xc0, 0xaf]),
        ("overlong-3byte", vec![0xe0, 0x80, 0xaf]),
        ("overlong-4byte", vec![0xf0, 0x80, 0x80, 0xaf]),
        ("surrogate-d800", vec![0xed, 0xa0, 0x80]),
        ("surrogate-dfff", vec![0xed, 0xbf, 0xbf]),
        ("over-max-codepoint", vec![0xf4, 0x90, 0x80, 0x80]),
        ("over-max-f5", vec![0xf5, 0x80, 0x80, 0x80]),
    ];

    let mut straddle = vec![b'a'; 62];
    straddle.extend_from_slice("é".as_bytes());
    straddle.extend(b"trailing".iter().copied());
    cases.push(("multibyte-straddles-block", straddle));

    let mut second_block_err = vec![b'x'; 64];
    second_block_err.push(0xff);
    second_block_err.extend_from_slice(b"after");
    cases.push(("error-in-second-block", second_block_err));

    let mut tail_error = vec![b'x'; 192];
    tail_error.push(0xc0);
    tail_error.push(b'?');
    cases.push(("error-near-tail", tail_error));

    let mut ascii_dfa_ascii = vec![b' '; 64];
    ascii_dfa_ascii.extend_from_slice("héllo".as_bytes());
    ascii_dfa_ascii.extend(vec![b' '; 64]);
    ascii_dfa_ascii.extend_from_slice("世界".as_bytes());
    ascii_dfa_ascii.extend(vec![b' '; 96]);
    cases.push(("ascii-dfa-ascii-loop", ascii_dfa_ascii));

    cases
}

#[test]
fn neon_validate_utf8_matches_scalar_reference_on_corpus() {
    for (label, input) in utf8_corpus() {
        let expected = byteclass::kernels::scalar::validate_utf8(&input);
        let dispatched = byteclass::validate_utf8(&input);
        // SAFETY: NEON is mandatory on AArch64.
        let direct = unsafe { byteclass::kernels::neon::validate_utf8(&input) };
        assert_eq!(
            dispatched,
            expected,
            "auto-dispatched validate_utf8 diverged on case {label} (len {})",
            input.len()
        );
        assert_eq!(
            direct,
            expected,
            "direct NEON validate_utf8 diverged on case {label} (len {})",
            input.len()
        );
    }
}

#[test]
fn neon_validate_utf8_matches_scalar_reference_on_long_inputs() {
    let template = "Lorem ipsum dolor sit amet, consectetur adipiscing 𝐞𝐥𝐢𝐭. \
                    日本語のテキストもここに混ざっています。 ";
    let mut bytes = template.repeat(64).into_bytes();
    let total_len = bytes.len();

    assert_eq!(
        byteclass::validate_utf8(&bytes),
        byteclass::kernels::scalar::validate_utf8(&bytes),
        "large valid corpus diverged"
    );

    for offset in [0_usize, 1, 31, 32, 63, 64, 65, 127, 128, 192, total_len / 2] {
        if offset >= bytes.len() {
            continue;
        }
        let saved = bytes[offset];
        bytes[offset] = 0xff;
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
fn neon_runlength_transitions_match_scalar_reference_on_synthetic_corpus() {
    let mut cases = synthetic_corpus();
    cases.push(b"abababab".repeat(64));
    cases.push(vec![0xa5_u8; 33]);
    cases.push(vec![0xa5_u8; 64]);
    cases.push(vec![0xa5_u8; 65]);

    for input in cases {
        let expected = runlength::kernels::scalar::transitions(&input);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { runlength::kernels::neon::transitions(&input) };
        assert_eq!(
            actual,
            expected,
            "neon runlength::transitions diverged on length {}",
            input.len()
        );
        // Dispatched API must agree as well.
        assert_eq!(runlength::transitions(&input), expected);
    }
}

#[test]
fn neon_runlength_transitions_match_scalar_reference_on_unaligned_subslices() {
    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = runlength::kernels::scalar::transitions(slice);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { runlength::kernels::neon::transitions(slice) };
        assert_eq!(
            actual,
            expected,
            "neon runlength::transitions diverged on slice len {}",
            slice.len()
        );
    }
}

#[test]
fn neon_bits_popcount_u64_matches_scalar_on_synthetic_corpus() {
    // Lengths span every plausible SIMD block boundary in the popcount
    // kernels: 2 u64 (NEON 16-byte vec), 16 u64 (NEON 8x-unrolled
    // 128B), 4 u64 (AVX2 32-byte vec), 8 u64 (AVX-512 64-byte vec).
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
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { bits::kernels::neon::popcount_u64_slice(&words) };
        assert_eq!(
            actual, expected,
            "neon bits::popcount_u64_slice diverged at len {len}"
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
fn neon_bits_popcount_u8_matches_scalar_on_unaligned_subslices() {
    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = bits::kernels::scalar::popcount_u8_slice(slice);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { bits::kernels::neon::popcount_u8_slice(slice) };
        assert_eq!(
            actual,
            expected,
            "neon bits::popcount_u8_slice diverged on slice len {}",
            slice.len()
        );
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
fn neon_bits_bit_pack_decode_matches_scalar_on_every_width() {
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
            // SAFETY: NEON is mandatory on AArch64.
            unsafe {
                bits::bit_pack::kernels::neon::decode_u32_slice(w, &encoded, n, &mut actual);
            }
            assert_eq!(
                actual, expected,
                "neon bits::bit_pack::decode diverged at w={w} n={n}"
            );

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
fn neon_bits_streamvbyte_decode_matches_scalar_on_size_grid() {
    // Cover every (full-groups, tail-shape) interaction plus large sizes
    // that exercise the SIMD-then-scalar tail boundary.
    for n in [
        0_usize, 1, 2, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 64, 99, 100, 256, 1023, 1024, 1025, 4096,
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
            // SAFETY: NEON is mandatory on AArch64.
            let consumed = unsafe {
                bits::streamvbyte::kernels::neon::decode_u32(
                    &ctrl,
                    &data[..written],
                    n,
                    &mut actual,
                )
            };
            assert_eq!(
                actual, expected,
                "neon bits::streamvbyte diverged at n={n} max_bytes={max}"
            );
            assert_eq!(
                consumed, written,
                "neon offset diverged at n={n} max_bytes={max}"
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

#[test]
fn neon_hash_set_membership_matches_scalar_on_size_grid() {
    // Cover SIMD block boundaries for the 4-lane NEON kernel plus the
    // x86-side shapes (8/16/32) so corpora are interchangeable across
    // ISAs.
    for len in [
        0_usize, 1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 256, 1023,
    ] {
        let haystack = deterministic_u32_haystack(len, 0xBEEF_F00D ^ (len as u64));
        for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
            let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
            // SAFETY: NEON is mandatory on AArch64.
            let actual =
                unsafe { hash::set_membership::kernels::neon::contains_u32(&haystack, needle) };
            assert_eq!(actual, expected, "len {len} needle {needle}");
        }
        if len > 0 {
            for &pos in &[0_usize, len / 2, len - 1] {
                let needle = haystack[pos];
                let expected =
                    hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: NEON is mandatory on AArch64.
                let actual =
                    unsafe { hash::set_membership::kernels::neon::contains_u32(&haystack, needle) };
                assert_eq!(actual, expected, "len {len} pos {pos}");
            }
        }
        // Dispatched API must agree as well.
        for &needle in &[0_u32, 1, u32::MAX] {
            assert_eq!(
                hash::contains_u32_simd(&haystack, needle),
                hash::set_membership::kernels::scalar::contains_u32(&haystack, needle),
                "dispatched neon set-membership diverged at len {len}"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        ..ProptestConfig::default()
    })]

    #[test]
    fn neon_byteclass_matches_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        let expected = byteclass::kernels::scalar::classify(&bytes);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { byteclass::kernels::neon::classify(&bytes) };
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn neon_runlength_transitions_match_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        let expected = runlength::kernels::scalar::transitions(&bytes);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { runlength::kernels::neon::transitions(&bytes) };
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn neon_validate_utf8_matches_scalar_for_random_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        let expected = byteclass::kernels::scalar::validate_utf8(&bytes);
        let actual = byteclass::validate_utf8(&bytes);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn neon_similarity_dot_l1_l2_u32_match_scalar(
        a in proptest::collection::vec(any::<u32>(), 0..1024),
        seed in any::<u32>(),
    ) {
        let b: Vec<u32> = a.iter().enumerate()
            .map(|(i, x)| x.wrapping_mul(seed.wrapping_add(i as u32 + 1)))
            .collect();
        prop_assert_eq!(
            similarity::distance::dot_u32(&a, &b),
            similarity::kernels::scalar::dot_u32(&a, &b),
        );
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
    fn neon_similarity_cosine_u32_matches_scalar(
        a in proptest::collection::vec(any::<u32>().prop_map(|x| x % (1 << 20)), 0..1024),
        seed in any::<u32>(),
    ) {
        let b: Vec<u32> = a.iter().enumerate()
            .map(|(i, x)| (x ^ seed.rotate_left(i as u32)) % (1 << 20))
            .collect();
        let expected = similarity::kernels::scalar::cosine_similarity_u32(&a, &b);
        let actual = similarity::distance::cosine_similarity_u32(&a, &b);
        match (expected, actual) {
            (None, None) => {}
            (Some(e), Some(act)) => {
                if e == 0.0 {
                    prop_assert!(act.abs() < 1e-12);
                } else {
                    prop_assert!((e - act).abs() / e.abs().max(1.0) < 1e-12,
                        "cosine_similarity_u32 diverged: scalar={e} neon={act}");
                }
            }
            _ => prop_assert!(false, "None/Some mismatch"),
        }
    }

    #[test]
    fn neon_similarity_dot_l2_f32_match_scalar(
        // See avx2_parity.rs for the rationale: range matches realistic
        // tokenfs vectors (byte-histogram counts [0, 256] and normalized
        // fingerprint deltas [-1, 1]).
        a in proptest::collection::vec(-256.0_f32..256.0, 0..1024),
        seed in any::<u32>(),
    ) {
        let b: Vec<f32> = a.iter().enumerate()
            .map(|(i, x)| x + (seed.wrapping_mul(i as u32 + 1) as f32 * 1e-3))
            .collect();
        let dot_s = similarity::kernels::scalar::dot_f32(&a, &b).unwrap();
        let dot_v = similarity::distance::dot_f32(&a, &b).unwrap();
        // Higham dot-product noise scale: see avx2_parity.rs for the full
        // explanation. Comparing against sum(|a*b|) (with a |dot| floor) is
        // the right scale because cancellation makes |dot| << sum(|a*b|) on
        // adversarial proptest seeds; relative error against |dot| can hit
        // 5–10% even with correct kernels.
        let l1_prod: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y).abs()).sum();
        let scale = l1_prod.max(dot_s.abs()).max(dot_v.abs()).max(1.0);
        prop_assert!((dot_s - dot_v).abs() / scale < 1e-3,
            "dot_f32 diverged: scalar={dot_s} neon={dot_v} l1_prod={l1_prod}");

        let l2_s = similarity::kernels::scalar::l2_squared_f32(&a, &b).unwrap();
        let l2_v = similarity::distance::l2_squared_f32(&a, &b).unwrap();
        let scale = l2_s.abs().max(l2_v.abs()).max(1.0);
        prop_assert!((l2_s - l2_v).abs() / scale < 5e-4,
            "l2_squared_f32 diverged: scalar={l2_s} neon={l2_v}");
    }

    #[test]
    fn neon_fingerprint_block_matches_scalar_for_random_blocks(
        block in proptest::collection::vec(any::<u8>(), fingerprint::BLOCK_SIZE..=fingerprint::BLOCK_SIZE),
    ) {
        let arr: &[u8; fingerprint::BLOCK_SIZE] = block.as_slice().try_into().unwrap();
        let scalar = fingerprint::kernels::scalar::block(arr);
        let neon_dispatched = fingerprint::block(arr);
        let neon_direct = fingerprint::kernels::neon::block(arr);
        prop_assert_eq!(neon_dispatched, scalar, "auto-dispatched neon fingerprint diverged from scalar");
        prop_assert_eq!(neon_direct, scalar, "direct neon fingerprint diverged from scalar");
    }

    #[test]
    fn neon_sketch_crc32_hash4_bins_matches_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        // Power-of-two BINS picks the pipelined 4-stream code path.
        let mut scalar_bins = [0_u32; 1024];
        sketch::kernels::scalar::crc32_hash4_bins(&bytes, &mut scalar_bins);

        let mut neon_bins = [0_u32; 1024];
        // SAFETY: NEON is mandatory on AArch64 and the FEAT_CRC32 ext
        // is universally present on every aarch64-linux production core.
        // The cfg gates this whole proptest block to that environment.
        unsafe {
            sketch::kernels::neon::crc32_hash4_bins(&bytes, &mut neon_bins);
        }
        prop_assert_eq!(neon_bins, scalar_bins);

        // The auto-dispatched public entry must agree as well.
        let mut dispatched = [0_u32; 1024];
        sketch::crc32_hash4_bins(&bytes, &mut dispatched);
        prop_assert_eq!(dispatched, scalar_bins);
    }

    #[test]
    fn neon_sketch_crc32_hash4_bins_non_power_of_two_matches_scalar(
        bytes in proptest::collection::vec(any::<u8>(), 0..4_096),
    ) {
        // Non-power-of-two BINS forces the scalar fallback path inside
        // the NEON kernel — verify that fallback still agrees with the
        // pure-scalar reference.
        let mut scalar_bins = [0_u32; 257];
        sketch::kernels::scalar::crc32_hash4_bins(&bytes, &mut scalar_bins);

        let mut neon_bins = [0_u32; 257];
        // SAFETY: NEON is mandatory on AArch64; FEAT_CRC32 is universal
        // on every aarch64-linux production core.
        unsafe {
            sketch::kernels::neon::crc32_hash4_bins(&bytes, &mut neon_bins);
        }
        prop_assert_eq!(neon_bins, scalar_bins);
    }

    #[test]
    fn neon_bits_popcount_u64_matches_scalar_for_random_words(
        words in proptest::collection::vec(any::<u64>(), 0..2048),
    ) {
        let expected = bits::kernels::scalar::popcount_u64_slice(&words);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { bits::kernels::neon::popcount_u64_slice(&words) };
        prop_assert_eq!(actual, expected);
        prop_assert_eq!(bits::popcount_u64_slice(&words), expected);
    }

    #[test]
    fn neon_bits_popcount_u8_matches_scalar_for_random_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        let expected = bits::kernels::scalar::popcount_u8_slice(&bytes);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { bits::kernels::neon::popcount_u8_slice(&bytes) };
        prop_assert_eq!(actual, expected);
        prop_assert_eq!(bits::popcount_u8_slice(&bytes), expected);
    }

    #[test]
    fn neon_hash_set_membership_matches_scalar_for_random_inputs(
        haystack in proptest::collection::vec(any::<u32>(), 0..1024),
        needle in any::<u32>(),
    ) {
        let expected = hash::set_membership::kernels::scalar::contains_u32(&haystack, needle);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe {
            hash::set_membership::kernels::neon::contains_u32(&haystack, needle)
        };
        prop_assert_eq!(actual, expected);
        // The auto-dispatched API must agree as well.
        prop_assert_eq!(hash::contains_u32_simd(&haystack, needle), expected);
    }

    #[test]
    fn neon_bits_streamvbyte_round_trip_random(
        values in proptest::collection::vec(any::<u32>(), 0..2048),
    ) {
        let n = values.len();
        let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(n)];
        let mut data = vec![0_u8; bits::streamvbyte_data_max_len(n)];
        let written = bits::streamvbyte_encode_u32(&values, &mut ctrl, &mut data);
        let mut decoded = vec![0_u32; n];
        // SAFETY: NEON is mandatory on AArch64.
        let consumed = unsafe {
            bits::streamvbyte::kernels::neon::decode_u32(&ctrl, &data[..written], n, &mut decoded)
        };
        prop_assert_eq!(consumed, written);
        prop_assert_eq!(decoded, values);
    }

    // ---------- vector module: NEON hamming/jaccard parity ----------

    #[test]
    fn neon_vector_hamming_u64_matches_scalar(
        a in proptest::collection::vec(any::<u64>(), 0..512),
        seed in any::<u64>(),
    ) {
        let b: Vec<u64> = a.iter().enumerate()
            .map(|(i, x)| x.wrapping_mul(seed.wrapping_add(i as u64 + 1)))
            .collect();
        let expected = vector::kernels::scalar::hamming_u64(&a, &b);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { vector::kernels::neon::hamming_u64(&a, &b) };
        prop_assert_eq!(Some(actual), expected);
        prop_assert_eq!(vector::hamming_u64(&a, &b), expected);
    }

    #[test]
    fn neon_vector_jaccard_u64_matches_scalar(
        a in proptest::collection::vec(any::<u64>(), 0..512),
        seed in any::<u64>(),
    ) {
        let b: Vec<u64> = a.iter().enumerate()
            .map(|(i, x)| x ^ seed.rotate_left(i as u32))
            .collect();
        let expected = vector::kernels::scalar::jaccard_u64(&a, &b).unwrap();
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { vector::kernels::neon::jaccard_u64(&a, &b) };
        prop_assert!((expected - actual).abs() < 1e-12,
            "jaccard_u64 diverged: scalar={expected} neon={actual}");
        let dispatched = vector::jaccard_u64(&a, &b).unwrap();
        prop_assert!((expected - dispatched).abs() < 1e-12);
    }

    // ---------- vector module: NEON dot/L2 f32 parity ----------

    #[test]
    fn neon_vector_dot_l2_f32_match_scalar(
        a in proptest::collection::vec(-256.0_f32..256.0, 0..1024),
        seed in any::<u32>(),
    ) {
        let b: Vec<f32> = a.iter().enumerate()
            .map(|(i, x)| x + (seed.wrapping_mul(i as u32 + 1) as f32 * 1e-3))
            .collect();
        let dot_s = vector::kernels::scalar::dot_f32(&a, &b).unwrap();
        // SAFETY: NEON is mandatory on AArch64.
        let dot_v = unsafe { vector::kernels::neon::dot_f32(&a, &b) };
        let l1: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y).abs()).sum();
        let scale = l1.max(dot_s.abs()).max(dot_v.abs()).max(1.0);
        prop_assert!((dot_s - dot_v).abs() / scale < 1e-3,
            "vector::neon::dot_f32 diverged: scalar={dot_s} neon={dot_v} l1={l1}");

        let l2_s = vector::kernels::scalar::l2_squared_f32(&a, &b).unwrap();
        // SAFETY: NEON is mandatory on AArch64.
        let l2_v = unsafe { vector::kernels::neon::l2_squared_f32(&a, &b) };
        let scale = l2_s.abs().max(l2_v.abs()).max(1.0);
        prop_assert!((l2_s - l2_v).abs() / scale < 5e-4,
            "vector::neon::l2_squared_f32 diverged: scalar={l2_s} neon={l2_v}");
    }

    // ---------- vector module: batched many-vs-one parity ----------

    #[test]
    fn neon_vector_batched_dot_f32_one_to_many_matches_serial(
        n_rows in 0_usize..16,
        seed in any::<u32>(),
    ) {
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
            prop_assert!((slot - serial).abs() < 1e-5,
                "batched dot_f32 row {} diverged: got {} serial {}", i, slot, serial);
        }
    }

    #[test]
    fn neon_vector_batched_hamming_u64_one_to_many_matches_serial(
        n_rows in 0_usize..16,
        seed in any::<u64>(),
    ) {
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

#[test]
fn neon_bitmap_x_bitmap_and_card_parity() {
    let a = bitmap_words_seeded(0xC0FF_EE00);
    let b = bitmap_words_seeded(0xDEAD_BEEF);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::and_into(&a, &b, &mut out_scalar);
    // SAFETY: NEON is mandatory on AArch64.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_neon::and_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn neon_bitmap_x_bitmap_or_card_parity() {
    let a = bitmap_words_seeded(0x1234_5678);
    let b = bitmap_words_seeded(0x9abc_def0);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::or_into(&a, &b, &mut out_scalar);
    // SAFETY: NEON is mandatory on AArch64.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_neon::or_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn neon_bitmap_x_bitmap_xor_card_parity() {
    let a = bitmap_words_seeded(0xAAAA_5555);
    let b = bitmap_words_seeded(0x5555_AAAA);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::xor_into(&a, &b, &mut out_scalar);
    // SAFETY: NEON is mandatory on AArch64.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_neon::xor_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn neon_bitmap_x_bitmap_andnot_card_parity() {
    let a = bitmap_words_seeded(0x0F0F_0F0F);
    let b = bitmap_words_seeded(0xF0F0_F0F0);
    let mut out_simd = [0_u64; 1024];
    let mut out_scalar = [0_u64; 1024];
    let card_scalar = bitmap::kernels::bitmap_x_bitmap_scalar::andnot_into(&a, &b, &mut out_scalar);
    // SAFETY: NEON is mandatory on AArch64.
    let card_simd =
        unsafe { bitmap::kernels::bitmap_x_bitmap_neon::andnot_into(&a, &b, &mut out_simd) };
    assert_eq!(card_simd, card_scalar);
    assert_eq!(out_simd[..], out_scalar[..]);
}

#[test]
fn neon_bitmap_just_cardinality_parity() {
    let a = bitmap_words_seeded(0x1357_9bdf);
    let b = bitmap_words_seeded(0x2468_ace0);
    // SAFETY: NEON is mandatory on AArch64.
    unsafe {
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_neon::and_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::and_cardinality(&a, &b)
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_neon::or_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::or_cardinality(&a, &b)
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_neon::xor_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::xor_cardinality(&a, &b)
        );
        assert_eq!(
            bitmap::kernels::bitmap_x_bitmap_neon::andnot_cardinality(&a, &b),
            bitmap::kernels::bitmap_x_bitmap_scalar::andnot_cardinality(&a, &b)
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    #[test]
    fn proptest_neon_bitmap_kernel_pairs(
        seed_a in any::<u64>(),
        seed_b in any::<u64>(),
    ) {
        let a = bitmap_words_seeded(seed_a);
        let b = bitmap_words_seeded(seed_b);

        let mut out_neon = [0_u64; 1024];
        let mut out_scalar = [0_u64; 1024];

        // SAFETY: NEON is mandatory on AArch64.
        let card_simd = unsafe {
            bitmap::kernels::bitmap_x_bitmap_neon::and_into(&a, &b, &mut out_neon)
        };
        let card_scalar =
            bitmap::kernels::bitmap_x_bitmap_scalar::and_into(&a, &b, &mut out_scalar);
        prop_assert_eq!(card_simd, card_scalar);
        prop_assert_eq!(out_neon.as_slice(), out_scalar.as_slice());
    }
}

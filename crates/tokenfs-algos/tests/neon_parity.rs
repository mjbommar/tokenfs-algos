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
//! exercises the NEON kernel against the pinned scalar reference.

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)] // Test code — panic on None/Err is the desired failure mode.
#![cfg(all(feature = "neon", target_arch = "aarch64"))]

use proptest::prelude::*;
use tokenfs_algos::{byteclass, fingerprint, runlength, similarity};

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
        a in proptest::collection::vec(-1000.0_f32..1000.0, 0..1024),
        seed in any::<u32>(),
    ) {
        let b: Vec<f32> = a.iter().enumerate()
            .map(|(i, x)| x + (seed.wrapping_mul(i as u32 + 1) as f32 * 1e-3))
            .collect();
        let dot_s = similarity::kernels::scalar::dot_f32(&a, &b).unwrap();
        let dot_v = similarity::distance::dot_f32(&a, &b).unwrap();
        let scale = dot_s.abs().max(dot_v.abs()).max(1.0);
        // f32 reduction-order tolerance: see avx2_parity.rs for the full
        // explanation. dot_f32 over 1024 elements with cancellation can
        // drift by ~1% (partial sums are O(10x) the final sum); L2_squared
        // sums squared terms only so cancellation can't blow it up.
        prop_assert!((dot_s - dot_v).abs() / scale < 1e-2,
            "dot_f32 diverged: scalar={dot_s} neon={dot_v}");

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
}

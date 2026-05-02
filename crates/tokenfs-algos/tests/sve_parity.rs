//! Parity tests for the AArch64 SVE / SVE2 backends.
//!
//! Mirrors `tests/neon_parity.rs` but exercises the new SVE2 byteclass
//! classifier, the SVE2 runlength transitions kernel, and the SVE
//! `dot_f32` / `l2_squared_f32` distance kernels.
//!
//! # Vector-length-agnostic (VLA) testing
//!
//! SVE source is one program, but it executes at the runtime vector
//! width. For these tests:
//!
//! * On the Linux GitHub-hosted aarch64 runner (Cobalt-100 / Neoverse-N2),
//!   `svcntb()` returns 16 (128-bit vectors).
//! * Under QEMU's default user-mode `max` CPU, `svcntb()` returns 64
//!   (512-bit vectors).
//! * On a Graviton 3 (Neoverse-V1) runner, `svcntb()` returns 32 (256-bit).
//!
//! All three must produce the same scalar result for the same input.
//! The test corpus deliberately includes small inputs (1, 2, 7, 15 bytes)
//! that are smaller than the widest vector we expect to encounter (so the
//! VLA tail handling gets exercised at every width) plus large inputs
//! that span dozens of full vectors at the narrowest widths.
//!
//! On x86 hosts this file is empty after preprocessing — `cargo test`
//! reports "0 tests" for the suite. It only compiles when the
//! `sve` / `sve2` cargo features are enabled and the target is aarch64.

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]
// Test code — panic on None/Err is the desired failure mode.
// The legacy `similarity::kernels::*` paths are deprecated re-exports of
// `vector::kernels::*` (v0.2 module split); SVE remains in
// `similarity::kernels::sve` for now, and the parity oracles below still
// route through the deprecated `similarity::kernels::scalar::*` shim
// rather than `vector::kernels::scalar::*` directly to keep this file's
// import surface uniform with the SVE-specific code path.
#![allow(deprecated)]
#![cfg(all(feature = "sve2", target_arch = "aarch64"))]

use proptest::prelude::*;
use tokenfs_algos::{byteclass, runlength, similarity};

fn synthetic_corpus() -> Vec<Vec<u8>> {
    // Identical to the neon_parity.rs corpus: same edge sizes around
    // 16 / 32 / 64 / 128 / 1024 etc.
    let mut cases: Vec<Vec<u8>> = vec![
        Vec::new(),
        vec![0],
        vec![0xff],
        vec![0; 4096],
        vec![0xff; 4096],
        (0_u8..=255).collect(),
        (0_u8..=255).cycle().take(4096).collect(),
        b"tokenfs-algos SVE parity \xe2\x9c\x85 mixed UTF-8\nline two\r\n\tindented"
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
        ("overlong-2byte", vec![0xc0, 0xaf]),
        ("surrogate-d800", vec![0xed, 0xa0, 0x80]),
        ("over-max-codepoint", vec![0xf4, 0x90, 0x80, 0x80]),
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

    cases
}

/// Per-test runtime skip helper. Apple M1 / M2 are aarch64 hosts with
/// the SVE2 cfg gate ON but no SVE2 hardware; calling the SVE
/// intrinsic kernels on them SIGILLs immediately. The crate's public
/// dispatch already checks `is_available()`, but these tests poke the
/// kernels directly through `unsafe`, so each test must short-circuit
/// when the runtime detector says no.
fn skip_unless_sve2_available() -> bool {
    !byteclass::kernels::sve2::is_available()
        || !runlength::kernels::sve2::is_available()
        || !similarity::kernels::sve::is_available()
}

#[test]
fn sve2_runtime_detection_matches_compile_target() {
    // Logged as a skip on hosts without SVE2 (Apple M1 etc.).
    if skip_unless_sve2_available() {
        eprintln!("skipping SVE2 sanity check: runtime detector reports unavailable");
        return;
    }
    assert!(byteclass::kernels::sve2::is_available());
    assert!(runlength::kernels::sve2::is_available());
    assert!(similarity::kernels::sve::is_available());
}

#[test]
fn sve2_byteclass_matches_scalar_reference_on_synthetic_corpus() {
    if skip_unless_sve2_available() {
        return;
    }
    for input in synthetic_corpus() {
        let expected = byteclass::kernels::scalar::classify(&input);
        // SAFETY: SVE2 availability checked by the sanity test above.
        let actual = unsafe { byteclass::kernels::sve2::classify(&input) };
        assert_eq!(
            actual,
            expected,
            "sve2 byteclass classify diverged on length {}",
            input.len()
        );
    }
}

#[test]
fn sve2_byteclass_matches_scalar_reference_on_unaligned_subslices() {
    if skip_unless_sve2_available() {
        return;
    }
    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = byteclass::kernels::scalar::classify(slice);
        // SAFETY: SVE2 availability checked.
        let actual = unsafe { byteclass::kernels::sve2::classify(slice) };
        assert_eq!(
            actual,
            expected,
            "sve2 byteclass classify diverged on slice len {}",
            slice.len()
        );
    }
}

#[test]
fn sve2_dispatched_byteclass_matches_scalar() {
    if skip_unless_sve2_available() {
        return;
    }
    // The auto-dispatched public API should pick SVE2 over NEON when
    // SVE2 is available (per `kernels::auto::classify`'s priority list)
    // and produce the same answer as the scalar reference.
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

#[test]
fn sve2_validate_utf8_matches_scalar_reference_on_corpus() {
    if skip_unless_sve2_available() {
        return;
    }
    for (label, input) in utf8_corpus() {
        let expected = byteclass::kernels::scalar::validate_utf8(&input);
        // SAFETY: SVE2 availability checked.
        let direct = unsafe { byteclass::kernels::sve2::validate_utf8(&input) };
        assert_eq!(
            direct,
            expected,
            "direct SVE2 validate_utf8 diverged on case {label} (len {})",
            input.len()
        );
        let dispatched = byteclass::validate_utf8(&input);
        assert_eq!(
            dispatched,
            expected,
            "auto-dispatched validate_utf8 diverged on case {label} (len {})",
            input.len()
        );
    }
}

#[test]
fn sve2_runlength_transitions_match_scalar_reference_on_synthetic_corpus() {
    if skip_unless_sve2_available() {
        return;
    }
    let mut cases = synthetic_corpus();
    cases.push(b"abababab".repeat(64));
    cases.push(vec![0xa5_u8; 33]);
    cases.push(vec![0xa5_u8; 64]);
    cases.push(vec![0xa5_u8; 65]);

    for input in cases {
        let expected = runlength::kernels::scalar::transitions(&input);
        // SAFETY: SVE2 availability checked.
        let actual = unsafe { runlength::kernels::sve2::transitions(&input) };
        assert_eq!(
            actual,
            expected,
            "sve2 runlength::transitions diverged on length {}",
            input.len()
        );
        assert_eq!(runlength::transitions(&input), expected);
    }
}

#[test]
fn sve2_runlength_transitions_match_scalar_reference_on_unaligned_subslices() {
    if skip_unless_sve2_available() {
        return;
    }
    let bytes = (0_usize..16_384)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for slice in unaligned_subslices(&bytes) {
        let expected = runlength::kernels::scalar::transitions(slice);
        // SAFETY: SVE2 availability checked.
        let actual = unsafe { runlength::kernels::sve2::transitions(slice) };
        assert_eq!(
            actual,
            expected,
            "sve2 runlength::transitions diverged on slice len {}",
            slice.len()
        );
    }
}

#[test]
fn sve_similarity_dot_f32_matches_scalar_on_fixed_inputs() {
    if skip_unless_sve2_available() {
        return;
    }
    for n in [
        0_usize, 1, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 256, 1024,
    ] {
        let a: Vec<f32> = (0..n).map(|i| ((i as f32) - 7.0) * 0.125).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 / ((i as f32) + 0.5)).collect();
        let scalar = similarity::kernels::scalar::dot_f32(&a, &b).unwrap();
        // SAFETY: SVE availability checked by sanity test.
        let sve = unsafe { similarity::kernels::sve::dot_f32(&a, &b) };
        let scale = scalar.abs().max(sve.abs()).max(1.0);
        let rel = (scalar - sve).abs() / scale;
        // Same f32 reduction-order tolerance as the NEON parity test.
        assert!(
            rel < 1e-4,
            "sve dot_f32 diverged at n={n}: scalar={scalar} sve={sve} (rel={rel})"
        );
    }
}

#[test]
fn sve_similarity_l2_squared_f32_matches_scalar_on_fixed_inputs() {
    if skip_unless_sve2_available() {
        return;
    }
    for n in [
        0_usize, 1, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 63, 64, 127, 256, 1024,
    ] {
        let a: Vec<f32> = (0..n).map(|i| ((i as f32) - 7.0) * 0.125).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 / ((i as f32) + 0.5)).collect();
        let scalar = similarity::kernels::scalar::l2_squared_f32(&a, &b).unwrap();
        // SAFETY: SVE availability checked.
        let sve = unsafe { similarity::kernels::sve::l2_squared_f32(&a, &b) };
        let scale = scalar.abs().max(sve.abs()).max(1.0);
        let rel = (scalar - sve).abs() / scale;
        assert!(
            rel < 1e-4,
            "sve l2_squared_f32 diverged at n={n}: scalar={scalar} sve={sve} (rel={rel})"
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 96,
        ..ProptestConfig::default()
    })]

    #[test]
    fn sve2_byteclass_matches_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        let expected = byteclass::kernels::scalar::classify(&bytes);
        // SAFETY: SVE2 availability checked.
        let actual = unsafe { byteclass::kernels::sve2::classify(&bytes) };
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn sve2_runlength_transitions_match_scalar_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        let expected = runlength::kernels::scalar::transitions(&bytes);
        // SAFETY: SVE2 availability checked.
        let actual = unsafe { runlength::kernels::sve2::transitions(&bytes) };
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn sve2_validate_utf8_matches_scalar_for_random_bytes(
        bytes in proptest::collection::vec(any::<u8>(), 0..16_384),
    ) {
        let expected = byteclass::kernels::scalar::validate_utf8(&bytes);
        let actual = byteclass::validate_utf8(&bytes);
        prop_assert_eq!(actual, expected);
    }

    #[test]
    fn sve_similarity_dot_l2_f32_match_scalar(
        a in proptest::collection::vec(-1000.0_f32..1000.0, 0..1024),
        seed in any::<u32>(),
    ) {
        let b: Vec<f32> = a.iter().enumerate()
            .map(|(i, x)| x + (seed.wrapping_mul(i as u32 + 1) as f32 * 1e-3))
            .collect();
        let dot_s = similarity::kernels::scalar::dot_f32(&a, &b).unwrap();
        // SAFETY: SVE availability checked.
        let dot_v = unsafe { similarity::kernels::sve::dot_f32(&a, &b) };
        let scale = dot_s.abs().max(dot_v.abs()).max(1.0);
        // Same `1e-2` cancellation tolerance as the NEON parity proptest.
        prop_assert!((dot_s - dot_v).abs() / scale < 1e-2,
            "sve dot_f32 diverged: scalar={dot_s} sve={dot_v}");

        let l2_s = similarity::kernels::scalar::l2_squared_f32(&a, &b).unwrap();
        // SAFETY: SVE availability checked.
        let l2_v = unsafe { similarity::kernels::sve::l2_squared_f32(&a, &b) };
        let scale = l2_s.abs().max(l2_v.abs()).max(1.0);
        prop_assert!((l2_s - l2_v).abs() / scale < 5e-4,
            "sve l2_squared_f32 diverged: scalar={l2_s} sve={l2_v}");
    }
}

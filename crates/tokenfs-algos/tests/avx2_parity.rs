//! Explicit parity tests between every public AVX2-dispatched kernel and its
//! pinned scalar reference.
//!
//! `tests/parity.rs` covers scalar-vs-auto for non-SIMD modules; this file
//! exercises the AVX2 surfaces specifically. Each test runtime-skips when AVX2
//! is unavailable so the suite still passes on machines without the feature,
//! and the entire file is cfg-gated to x86/x86_64 with the `avx2` Cargo
//! feature so non-x86 builds compile cleanly.

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)] // Test code — panic on None is the desired failure mode.
#![cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]

use proptest::prelude::*;

use tokenfs_algos::{
    byteclass,
    fingerprint::{self, BLOCK_SIZE},
    histogram::{self, ByteHistogram},
    runlength, similarity,
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
}

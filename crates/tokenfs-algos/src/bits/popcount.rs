//! SIMD-accelerated bit-count over `&[u64]` and `&[u8]` slices.
//!
//! Foundation kernel for `bits::rank_select` and for `bitmap` cardinality.
//! See `docs/v0.2_planning/10_BITS.md` § 4 for the spec and per-backend
//! throughput targets.
//!
//! ## API surface
//!
//! * [`popcount_u64_slice`] — runtime-dispatched bit-count over `&[u64]`.
//! * [`popcount_u8_slice`] — runtime-dispatched bit-count over `&[u8]`.
//! * [`kernels::scalar`] — portable reference path.
//! * `kernels::avx2` — x86 Mula nibble-LUT (`feature = "avx2"`).
//! * `kernels::avx512` — x86 VPOPCNTQ (`feature = "avx512"`).
//! * `kernels::neon` — AArch64 VCNT + horizontal add (`feature = "neon"`).

/// Returns the total number of set bits across `words`.
///
/// Uses the best available kernel detected at runtime (AVX-512 VPOPCNTQ
/// > AVX2 Mula nibble-LUT > scalar on x86; NEON VCNT on AArch64).
///
/// Bit-exact with the scalar reference [`kernels::scalar::popcount_u64_slice`]
/// on every backend.
#[must_use]
pub fn popcount_u64_slice(words: &[u64]) -> u64 {
    kernels::auto::popcount_u64_slice(words)
}

/// Returns the total number of set bits across `bytes`.
///
/// Uses the best available kernel detected at runtime; bit-exact with
/// [`kernels::scalar::popcount_u8_slice`].
#[must_use]
pub fn popcount_u8_slice(bytes: &[u8]) -> u64 {
    kernels::auto::popcount_u8_slice(bytes)
}

/// Pinned popcount kernels.
pub mod kernels {
    /// Runtime-dispatched popcount kernels.
    pub mod auto {
        /// Runtime-dispatched popcount over `&[u64]`.
        #[must_use]
        pub fn popcount_u64_slice(words: &[u64]) -> u64 {
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512::popcount_u64_slice(words) };
                }
            }

            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::popcount_u64_slice(words) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { super::neon::popcount_u64_slice(words) };
                }
            }

            super::scalar::popcount_u64_slice(words)
        }

        /// Runtime-dispatched popcount over `&[u8]`.
        #[must_use]
        pub fn popcount_u8_slice(bytes: &[u8]) -> u64 {
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512::popcount_u8_slice(bytes) };
                }
            }

            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::popcount_u8_slice(bytes) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { super::neon::popcount_u8_slice(bytes) };
                }
            }

            super::scalar::popcount_u8_slice(bytes)
        }
    }

    /// Portable scalar popcount.
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    /// x86 AVX2 popcount via the Mula nibble-LUT method.
    ///
    /// Reference: Mula, Kurz, Lemire, "Faster population counts using AVX2
    /// instructions," The Computer Journal, 2018 (arXiv:1611.07612).
    ///
    /// AVX2 has no native 64-bit popcount instruction. The kernel uses a
    /// 16-entry lookup table holding the popcount of each 4-bit nibble,
    /// applies it via `_mm256_shuffle_epi8` (per-128-bit-lane PSHUFB) on
    /// the low and high nibbles of each byte, sums the two lookups into a
    /// per-byte popcount vector, and folds that into 64-bit lane sums via
    /// `_mm256_sad_epu8` (sum-of-absolute-differences against zero).
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx2;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx2;

    /// x86 AVX-512 popcount via VPOPCNTQ.
    ///
    /// `_mm512_popcnt_epi64` (one cycle per 64-bit lane on Ice Lake and
    /// later) is the native AVX-512 popcount. Requires the
    /// `AVX512VPOPCNTDQ` CPU feature; gated on the crate's `avx512`
    /// cargo feature (which itself implies `nightly` for the unstable
    /// AVX-512 intrinsic surface).
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx512;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx512;

    /// AArch64 NEON popcount via VCNT + horizontal add.
    ///
    /// `vcntq_u8` produces a per-byte popcount in 0..=8 lanes; multiple
    /// vectors are accumulated in a u8 vector with `vaddq_u8` and folded
    /// into u16 lanes via `vpaddlq_u8` to avoid u8 saturation. The final
    /// horizontal sum uses `vaddvq_u16`.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "neon",
        target_arch = "aarch64"
    ))]
    pub mod neon;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "neon",
        target_arch = "aarch64"
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod neon;
}

#[cfg(test)]
mod tests {
    use super::{kernels, popcount_u8_slice, popcount_u64_slice};

    fn deterministic_u64_words(n: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_f491_4f6c_dd1d)
            })
            .collect()
    }

    fn naive_popcount_u64(words: &[u64]) -> u64 {
        words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    fn naive_popcount_u8(bytes: &[u8]) -> u64 {
        bytes.iter().map(|b| u64::from(b.count_ones())).sum()
    }

    #[test]
    fn empty_slices_yield_zero() {
        assert_eq!(popcount_u64_slice(&[]), 0);
        assert_eq!(popcount_u8_slice(&[]), 0);
    }

    #[test]
    fn single_all_zeros_word() {
        assert_eq!(popcount_u64_slice(&[0_u64]), 0);
        assert_eq!(popcount_u8_slice(&[0_u8; 8]), 0);
    }

    #[test]
    fn single_all_ones_word() {
        assert_eq!(popcount_u64_slice(&[u64::MAX]), 64);
        assert_eq!(popcount_u8_slice(&[0xff_u8; 8]), 64);
    }

    #[test]
    fn matches_scalar_for_small_inputs() {
        let cases: &[&[u64]] = &[
            &[],
            &[0],
            &[u64::MAX],
            &[0, u64::MAX, 0xa5a5_a5a5_a5a5_a5a5],
            &[0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80],
        ];
        for case in cases {
            assert_eq!(
                popcount_u64_slice(case),
                kernels::scalar::popcount_u64_slice(case),
                "default vs scalar diverged on len {}",
                case.len()
            );
        }
    }

    #[test]
    fn matches_scalar_at_sub_block_lengths() {
        // Cover lengths around every plausible SIMD block boundary:
        // 4 u64 (AVX2 32-byte vec), 8 u64 (AVX-512 64-byte vec),
        // 32 u64 (AVX2 8x-unrolled 256B), and beyond.
        let seed = 0xF22_C0FFEE_u64;
        for len in [
            0_usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257,
            511, 1023,
        ] {
            let words = deterministic_u64_words(len, seed.wrapping_add(len as u64));
            let expected = naive_popcount_u64(&words);
            assert_eq!(
                popcount_u64_slice(&words),
                expected,
                "popcount_u64_slice diverged at len {len}"
            );
            assert_eq!(
                kernels::scalar::popcount_u64_slice(&words),
                expected,
                "scalar popcount_u64_slice diverged at len {len}"
            );
        }
    }

    #[test]
    fn u8_path_matches_scalar_at_byte_lengths() {
        let seed = 0xC0DE_C0DE_C0DE_C0DE_u64;
        for len in [
            0_usize, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            1024, 1025,
        ] {
            let words = deterministic_u64_words(len.div_ceil(8) + 1, seed);
            let bytes_full: Vec<u8> = words
                .iter()
                .flat_map(|w| w.to_le_bytes().into_iter())
                .collect();
            let bytes = &bytes_full[..len];
            let expected = naive_popcount_u8(bytes);
            assert_eq!(
                popcount_u8_slice(bytes),
                expected,
                "popcount_u8_slice diverged at len {len}"
            );
            assert_eq!(
                kernels::scalar::popcount_u8_slice(bytes),
                expected,
                "scalar popcount_u8_slice diverged at len {len}"
            );
        }
    }

    #[test]
    fn u64_and_u8_paths_agree_on_same_bits() {
        let seed = 0x1234_5678_9abc_def0_u64;
        let words = deterministic_u64_words(64, seed);
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter())
            .collect();
        assert_eq!(popcount_u64_slice(&words), popcount_u8_slice(&bytes));
    }

    #[test]
    fn long_random_slice_matches_scalar() {
        let words = deterministic_u64_words(8192, 0xA1A1_B2B2_C3C3_D4D4);
        let expected = naive_popcount_u64(&words);
        assert_eq!(popcount_u64_slice(&words), expected);
        assert_eq!(kernels::scalar::popcount_u64_slice(&words), expected);
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_kernel_matches_scalar_when_available() {
        if !kernels::avx2::is_available() {
            eprintln!("avx2 unavailable on this host; skipping inline AVX2 parity test");
            return;
        }
        let words = deterministic_u64_words(1024, 0x5151_5eed);
        let expected = kernels::scalar::popcount_u64_slice(&words);
        // SAFETY: availability checked above.
        let actual = unsafe { kernels::avx2::popcount_u64_slice(&words) };
        assert_eq!(actual, expected);

        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter())
            .collect();
        let expected_b = kernels::scalar::popcount_u8_slice(&bytes);
        // SAFETY: availability checked above.
        let actual_b = unsafe { kernels::avx2::popcount_u8_slice(&bytes) };
        assert_eq!(actual_b, expected_b);
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_kernel_matches_scalar_when_available() {
        let words = deterministic_u64_words(1024, 0x5151_5eed);
        let expected = kernels::scalar::popcount_u64_slice(&words);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { kernels::neon::popcount_u64_slice(&words) };
        assert_eq!(actual, expected);

        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter())
            .collect();
        let expected_b = kernels::scalar::popcount_u8_slice(&bytes);
        // SAFETY: NEON is mandatory on AArch64.
        let actual_b = unsafe { kernels::neon::popcount_u8_slice(&bytes) };
        assert_eq!(actual_b, expected_b);
    }
}

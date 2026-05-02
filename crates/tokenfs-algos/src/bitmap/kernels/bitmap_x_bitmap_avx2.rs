//! AVX2 bitmap × bitmap kernels.
//!
//! Each AND/OR/XOR/ANDNOT kernel processes 8 × 256-bit vectors per outer
//! iteration over a 1024 × `u64` bitmap container. The cardinality
//! variant accumulates a 64-bit popcount per outer iteration via Mula's
//! per-byte nibble-LUT method (mirroring `crate::bits::popcount`'s AVX2
//! kernel) so the result is identical to scalar `count_ones` summation.
//!
//! Reference: Lemire et al., "Roaring Bitmaps: Implementation of an
//! Optimized Software Library", SPE 48(4), 2018 (arXiv 1709.07821), § 6.

#![allow(clippy::cast_lossless)]

use crate::bitmap::containers::BITMAP_WORDS;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_extract_epi64, _mm256_loadu_si256, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi8,
    _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_xor_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_andnot_si256,
    _mm256_extract_epi64, _mm256_loadu_si256, _mm256_or_si256, _mm256_sad_epu8, _mm256_set1_epi8,
    _mm256_setr_epi8, _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16,
    _mm256_storeu_si256, _mm256_xor_si256,
};

/// 256-bit vector covers 32 bytes = 4 u64 words.
const VEC_WORDS: usize = 4;

/// Bitmap container has 1024 / 4 = 256 AVX2 vectors.
const TOTAL_VECTORS: usize = BITMAP_WORDS / VEC_WORDS;

/// Returns true when AVX2 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Returns true when AVX2 is available at runtime (no_std stub).
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX2 256-bit bitwise op + popcount accumulator (`a OP b` into `out`).
///
/// # Safety
///
/// AVX2 must be available; the caller asserts via runtime check.
macro_rules! bitwise_into {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// AVX2 must be available; the caller asserts via runtime check.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn $name(
            a: &[u64; BITMAP_WORDS],
            b: &[u64; BITMAP_WORDS],
            out: &mut [u64; BITMAP_WORDS],
        ) -> u32 {
            // Mula nibble-LUT (same as bits::popcount AVX2 kernel).
            let lookup: __m256i = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                2, 3, 3, 4,
            );
            let low_mask = _mm256_set1_epi8(0x0F);

            let mut acc_u64 = _mm256_setzero_si256();

            // 256 vectors / 8-vec unroll = 32 outer iterations. Each
            // per-byte popcount fits in 0..=8; eight unrolled per-byte
            // sums fit in 0..=64 < 255, so the u8 accumulator does not
            // saturate before the per-iteration SAD reduction.
            let mut i = 0_usize;
            while i + 8 <= TOTAL_VECTORS {
                let mut acc_u8 = _mm256_setzero_si256();
                for k in 0..8 {
                    // SAFETY: i + 7 < TOTAL_VECTORS, so loading 32 bytes
                    // at offset (i + k) * 32 fits in the 8 KiB bitmap.
                    let av = unsafe {
                        _mm256_loadu_si256(a.as_ptr().add((i + k) * VEC_WORDS).cast::<__m256i>())
                    };
                    let bv = unsafe {
                        _mm256_loadu_si256(b.as_ptr().add((i + k) * VEC_WORDS).cast::<__m256i>())
                    };
                    let r: __m256i = $op(av, bv);
                    // SAFETY: as above; the corresponding output range is
                    // also within 8 KiB.
                    unsafe {
                        _mm256_storeu_si256(
                            out.as_mut_ptr().add((i + k) * VEC_WORDS).cast::<__m256i>(),
                            r,
                        );
                    }
                    acc_u8 =
                        _mm256_add_epi8(acc_u8, unsafe { popcnt_per_byte(r, lookup, low_mask) });
                }
                let zero = _mm256_setzero_si256();
                acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(acc_u8, zero));
                i += 8;
            }

            // Tail: any vectors not covered by the unrolled body. With
            // TOTAL_VECTORS = 256 the unroll covers everything, but
            // we keep the loop for robustness.
            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS so the offset is in bounds.
                let av =
                    unsafe { _mm256_loadu_si256(a.as_ptr().add(i * VEC_WORDS).cast::<__m256i>()) };
                let bv =
                    unsafe { _mm256_loadu_si256(b.as_ptr().add(i * VEC_WORDS).cast::<__m256i>()) };
                let r: __m256i = $op(av, bv);
                unsafe {
                    _mm256_storeu_si256(out.as_mut_ptr().add(i * VEC_WORDS).cast::<__m256i>(), r);
                }
                let per_byte = unsafe { popcnt_per_byte(r, lookup, low_mask) };
                let zero = _mm256_setzero_si256();
                acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(per_byte, zero));
                i += 1;
            }

            let s0 = _mm256_extract_epi64::<0>(acc_u64) as u64;
            let s1 = _mm256_extract_epi64::<1>(acc_u64) as u64;
            let s2 = _mm256_extract_epi64::<2>(acc_u64) as u64;
            let s3 = _mm256_extract_epi64::<3>(acc_u64) as u64;
            (s0 + s1 + s2 + s3) as u32
        }
    };
}

/// AVX2 256-bit bitwise op `_nocard` (`a OP b` into `out`, no popcount).
///
/// # Safety
///
/// AVX2 must be available.
macro_rules! bitwise_into_nocard {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// AVX2 must be available; the caller asserts via runtime check.
        #[target_feature(enable = "avx2")]
        pub unsafe fn $name(
            a: &[u64; BITMAP_WORDS],
            b: &[u64; BITMAP_WORDS],
            out: &mut [u64; BITMAP_WORDS],
        ) {
            let mut i = 0_usize;
            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load/store.
                let av =
                    unsafe { _mm256_loadu_si256(a.as_ptr().add(i * VEC_WORDS).cast::<__m256i>()) };
                let bv =
                    unsafe { _mm256_loadu_si256(b.as_ptr().add(i * VEC_WORDS).cast::<__m256i>()) };
                let r: __m256i = $op(av, bv);
                unsafe {
                    _mm256_storeu_si256(out.as_mut_ptr().add(i * VEC_WORDS).cast::<__m256i>(), r);
                }
                i += 1;
            }
        }
    };
}

/// AVX2 256-bit bitwise op `_justcard` (`a OP b` cardinality only, no store).
///
/// # Safety
///
/// AVX2 must be available.
macro_rules! bitwise_cardinality {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// AVX2 must be available; the caller asserts via runtime check.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn $name(a: &[u64; BITMAP_WORDS], b: &[u64; BITMAP_WORDS]) -> u32 {
            let lookup: __m256i = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                2, 3, 3, 4,
            );
            let low_mask = _mm256_set1_epi8(0x0F);

            let mut acc_u64 = _mm256_setzero_si256();

            let mut i = 0_usize;
            while i + 8 <= TOTAL_VECTORS {
                let mut acc_u8 = _mm256_setzero_si256();
                for k in 0..8 {
                    // SAFETY: same bounds reasoning as `bitwise_into`.
                    let av = unsafe {
                        _mm256_loadu_si256(a.as_ptr().add((i + k) * VEC_WORDS).cast::<__m256i>())
                    };
                    let bv = unsafe {
                        _mm256_loadu_si256(b.as_ptr().add((i + k) * VEC_WORDS).cast::<__m256i>())
                    };
                    let r: __m256i = $op(av, bv);
                    acc_u8 =
                        _mm256_add_epi8(acc_u8, unsafe { popcnt_per_byte(r, lookup, low_mask) });
                }
                let zero = _mm256_setzero_si256();
                acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(acc_u8, zero));
                i += 8;
            }

            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load.
                let av =
                    unsafe { _mm256_loadu_si256(a.as_ptr().add(i * VEC_WORDS).cast::<__m256i>()) };
                let bv =
                    unsafe { _mm256_loadu_si256(b.as_ptr().add(i * VEC_WORDS).cast::<__m256i>()) };
                let r: __m256i = $op(av, bv);
                let per_byte = unsafe { popcnt_per_byte(r, lookup, low_mask) };
                let zero = _mm256_setzero_si256();
                acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(per_byte, zero));
                i += 1;
            }

            let s0 = _mm256_extract_epi64::<0>(acc_u64) as u64;
            let s1 = _mm256_extract_epi64::<1>(acc_u64) as u64;
            let s2 = _mm256_extract_epi64::<2>(acc_u64) as u64;
            let s3 = _mm256_extract_epi64::<3>(acc_u64) as u64;
            (s0 + s1 + s2 + s3) as u32
        }
    };
}

bitwise_into!(
    and_into,
    "AVX2 `_card` AND of two 1024-word bitmaps. Stores `a & b` into `out` and returns the result cardinality.",
    |a, b| _mm256_and_si256(a, b)
);
bitwise_into!(
    or_into,
    "AVX2 `_card` OR of two 1024-word bitmaps. Stores `a | b` into `out` and returns the result cardinality.",
    |a, b| _mm256_or_si256(a, b)
);
bitwise_into!(
    xor_into,
    "AVX2 `_card` XOR of two 1024-word bitmaps. Stores `a ^ b` into `out` and returns the result cardinality.",
    |a, b| _mm256_xor_si256(a, b)
);
// `_mm256_andnot_si256(a, b)` computes `(~a) & b`; we want `a & ~b`,
// which is `_mm256_andnot_si256(b, a)` — note the swapped operand order.
bitwise_into!(
    andnot_into,
    "AVX2 `_card` AND-NOT of two 1024-word bitmaps. Stores `a & !b` into `out` and returns the result cardinality.",
    |a, b| _mm256_andnot_si256(b, a)
);

bitwise_into_nocard!(
    and_into_nocard,
    "AVX2 `_nocard` AND of two 1024-word bitmaps. Stores `a & b` into `out` without computing cardinality.",
    |a, b| _mm256_and_si256(a, b)
);
bitwise_into_nocard!(
    or_into_nocard,
    "AVX2 `_nocard` OR of two 1024-word bitmaps. Stores `a | b` into `out` without computing cardinality.",
    |a, b| _mm256_or_si256(a, b)
);
bitwise_into_nocard!(
    xor_into_nocard,
    "AVX2 `_nocard` XOR of two 1024-word bitmaps. Stores `a ^ b` into `out` without computing cardinality.",
    |a, b| _mm256_xor_si256(a, b)
);
bitwise_into_nocard!(
    andnot_into_nocard,
    "AVX2 `_nocard` AND-NOT of two 1024-word bitmaps. Stores `a & !b` into `out` without computing cardinality.",
    |a, b| _mm256_andnot_si256(b, a)
);

bitwise_cardinality!(
    and_cardinality,
    "AVX2 `_justcard` AND of two 1024-word bitmaps. Returns `popcount(a & b)` without materialising the result.",
    |a, b| _mm256_and_si256(a, b)
);
bitwise_cardinality!(
    or_cardinality,
    "AVX2 `_justcard` OR of two 1024-word bitmaps. Returns `popcount(a | b)` without materialising the result.",
    |a, b| _mm256_or_si256(a, b)
);
bitwise_cardinality!(
    xor_cardinality,
    "AVX2 `_justcard` XOR of two 1024-word bitmaps. Returns `popcount(a ^ b)` without materialising the result.",
    |a, b| _mm256_xor_si256(a, b)
);
bitwise_cardinality!(
    andnot_cardinality,
    "AVX2 `_justcard` AND-NOT of two 1024-word bitmaps. Returns `popcount(a & !b)` without materialising the result.",
    |a, b| _mm256_andnot_si256(b, a)
);

/// One Mula step on a 32-byte input vector — per-byte popcount.
///
/// # Safety
///
/// AVX2 must be available; caller asserts via `target_feature`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn popcnt_per_byte(v: __m256i, lookup: __m256i, low_mask: __m256i) -> __m256i {
    let lo = _mm256_and_si256(v, low_mask);
    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), low_mask);
    let lo_pc = _mm256_shuffle_epi8(lookup, lo);
    let hi_pc = _mm256_shuffle_epi8(lookup, hi);
    _mm256_add_epi8(lo_pc, hi_pc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::kernels::bitmap_x_bitmap_scalar as scalar;

    fn distinct_bitmaps() -> ([u64; BITMAP_WORDS], [u64; BITMAP_WORDS]) {
        let mut a = [0_u64; BITMAP_WORDS];
        let mut b = [0_u64; BITMAP_WORDS];
        let mut state = 0xC0FFEE_u64;
        for w in 0..BITMAP_WORDS {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            a[w] = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            b[w] = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        }
        (a, b)
    }

    macro_rules! assert_avx2_matches_scalar {
        ($scalar:path, $simd:path, $a:expr, $b:expr) => {{
            let mut out_scalar = [0_u64; BITMAP_WORDS];
            let card_scalar = $scalar(&$a, &$b, &mut out_scalar);
            if !is_available() {
                eprintln!(
                    "avx2 unavailable; skipping {} parity test",
                    stringify!($simd)
                );
                return;
            }
            let mut out_simd = [0_u64; BITMAP_WORDS];
            // SAFETY: availability checked above.
            let card_simd = unsafe { $simd(&$a, &$b, &mut out_simd) };
            assert_eq!(card_simd, card_scalar);
            assert_eq!(out_simd[..], out_scalar[..]);
        }};
    }

    #[test]
    fn avx2_and_into_matches_scalar() {
        let (a, b) = distinct_bitmaps();
        assert_avx2_matches_scalar!(scalar::and_into, and_into, a, b);
    }

    #[test]
    fn avx2_or_into_matches_scalar() {
        let (a, b) = distinct_bitmaps();
        assert_avx2_matches_scalar!(scalar::or_into, or_into, a, b);
    }

    #[test]
    fn avx2_xor_into_matches_scalar() {
        let (a, b) = distinct_bitmaps();
        assert_avx2_matches_scalar!(scalar::xor_into, xor_into, a, b);
    }

    #[test]
    fn avx2_andnot_into_matches_scalar() {
        let (a, b) = distinct_bitmaps();
        assert_avx2_matches_scalar!(scalar::andnot_into, andnot_into, a, b);
    }

    #[test]
    fn avx2_just_cardinality_matches_scalar() {
        if !is_available() {
            return;
        }
        let (a, b) = distinct_bitmaps();
        // SAFETY: availability checked above.
        unsafe {
            assert_eq!(and_cardinality(&a, &b), scalar::and_cardinality(&a, &b));
            assert_eq!(or_cardinality(&a, &b), scalar::or_cardinality(&a, &b));
            assert_eq!(xor_cardinality(&a, &b), scalar::xor_cardinality(&a, &b));
            assert_eq!(
                andnot_cardinality(&a, &b),
                scalar::andnot_cardinality(&a, &b)
            );
        }
    }

    #[test]
    fn avx2_nocard_matches_scalar_store() {
        if !is_available() {
            return;
        }
        let (a, b) = distinct_bitmaps();
        let mut out_simd = [0_u64; BITMAP_WORDS];
        let mut out_scalar = [0_u64; BITMAP_WORDS];
        // SAFETY: availability checked above.
        unsafe {
            and_into_nocard(&a, &b, &mut out_simd);
        }
        scalar::and_into_nocard(&a, &b, &mut out_scalar);
        assert_eq!(out_simd[..], out_scalar[..]);
    }
}

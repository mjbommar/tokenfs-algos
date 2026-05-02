//! NEON bitmap × bitmap kernels.
//!
//! Each AND/OR/XOR/ANDNOT kernel processes 4 × 128-bit vectors per outer
//! iteration over a 1024 × `u64` bitmap container. The cardinality
//! variants accumulate per-byte popcounts via `vcntq_u8` and widen via
//! pairwise add (`vpaddlq_u8`) to avoid u8 saturation, mirroring the
//! AArch64 popcount kernel in `crate::bits::popcount`.

#![allow(clippy::cast_lossless)]

use crate::bitmap::containers::BITMAP_WORDS;

use core::arch::aarch64::{
    uint8x16_t, vaddq_u8, vaddvq_u16, vandq_u8, vbicq_u8, vcntq_u8, vdupq_n_u8, veorq_u8, vld1q_u8,
    vorrq_u8, vpaddlq_u8, vst1q_u8,
};

/// 16 bytes (2 u64) per NEON vector.
const VEC_BYTES: usize = 16;

/// Bitmap container has 8192 / 16 = 512 NEON vectors.
const TOTAL_VECTORS: usize = (BITMAP_WORDS * 8) / VEC_BYTES;

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this exists for API symmetry.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// NEON bitwise op + per-byte popcount accumulator (with output store).
///
/// # Safety
///
/// NEON must be available; mandatory on AArch64.
macro_rules! bitwise_into {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// NEON must be available; mandatory on AArch64.
        #[target_feature(enable = "neon")]
        #[must_use]
        pub unsafe fn $name(
            a: &[u64; BITMAP_WORDS],
            b: &[u64; BITMAP_WORDS],
            out: &mut [u64; BITMAP_WORDS],
        ) -> u32 {
            let mut total: u64 = 0;
            let a_bytes = a.as_ptr().cast::<u8>();
            let b_bytes = b.as_ptr().cast::<u8>();
            let out_bytes = out.as_mut_ptr().cast::<u8>();

            // 4-vec unroll = 64 bytes per outer iteration; max per-byte
            // popcount across the unroll is 4 * 8 = 32, so the u8
            // accumulator stays under saturation (max 255).
            let mut i = 0_usize;
            while i + 4 <= TOTAL_VECTORS {
                let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
                for k in 0..4 {
                    // SAFETY: i + 3 < TOTAL_VECTORS bounds the load/store.
                    let av = unsafe { vld1q_u8(a_bytes.add((i + k) * VEC_BYTES)) };
                    let bv = unsafe { vld1q_u8(b_bytes.add((i + k) * VEC_BYTES)) };
                    let r: uint8x16_t = $op(av, bv);
                    unsafe {
                        vst1q_u8(out_bytes.add((i + k) * VEC_BYTES), r);
                    }
                    acc_u8 = vaddq_u8(acc_u8, vcntq_u8(r));
                }
                let widened = vpaddlq_u8(acc_u8);
                total += u64::from(vaddvq_u16(widened));
                i += 4;
            }

            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load/store.
                let av = unsafe { vld1q_u8(a_bytes.add(i * VEC_BYTES)) };
                let bv = unsafe { vld1q_u8(b_bytes.add(i * VEC_BYTES)) };
                let r: uint8x16_t = $op(av, bv);
                unsafe {
                    vst1q_u8(out_bytes.add(i * VEC_BYTES), r);
                }
                let widened = vpaddlq_u8(vcntq_u8(r));
                total += u64::from(vaddvq_u16(widened));
                i += 1;
            }

            total as u32
        }
    };
}

/// NEON bitwise op `_nocard` (no popcount).
///
/// # Safety
///
/// NEON must be available.
macro_rules! bitwise_into_nocard {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// NEON must be available; mandatory on AArch64.
        #[target_feature(enable = "neon")]
        pub unsafe fn $name(
            a: &[u64; BITMAP_WORDS],
            b: &[u64; BITMAP_WORDS],
            out: &mut [u64; BITMAP_WORDS],
        ) {
            let a_bytes = a.as_ptr().cast::<u8>();
            let b_bytes = b.as_ptr().cast::<u8>();
            let out_bytes = out.as_mut_ptr().cast::<u8>();

            let mut i = 0_usize;
            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load/store.
                let av = unsafe { vld1q_u8(a_bytes.add(i * VEC_BYTES)) };
                let bv = unsafe { vld1q_u8(b_bytes.add(i * VEC_BYTES)) };
                let r: uint8x16_t = $op(av, bv);
                unsafe {
                    vst1q_u8(out_bytes.add(i * VEC_BYTES), r);
                }
                i += 1;
            }
        }
    };
}

/// NEON bitwise op `_justcard` (cardinality only, no store).
///
/// # Safety
///
/// NEON must be available.
macro_rules! bitwise_cardinality {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// NEON must be available; mandatory on AArch64.
        #[target_feature(enable = "neon")]
        #[must_use]
        pub unsafe fn $name(a: &[u64; BITMAP_WORDS], b: &[u64; BITMAP_WORDS]) -> u32 {
            let mut total: u64 = 0;
            let a_bytes = a.as_ptr().cast::<u8>();
            let b_bytes = b.as_ptr().cast::<u8>();

            let mut i = 0_usize;
            while i + 4 <= TOTAL_VECTORS {
                let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
                for k in 0..4 {
                    // SAFETY: same bounds reasoning.
                    let av = unsafe { vld1q_u8(a_bytes.add((i + k) * VEC_BYTES)) };
                    let bv = unsafe { vld1q_u8(b_bytes.add((i + k) * VEC_BYTES)) };
                    let r: uint8x16_t = $op(av, bv);
                    acc_u8 = vaddq_u8(acc_u8, vcntq_u8(r));
                }
                let widened = vpaddlq_u8(acc_u8);
                total += u64::from(vaddvq_u16(widened));
                i += 4;
            }

            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load.
                let av = unsafe { vld1q_u8(a_bytes.add(i * VEC_BYTES)) };
                let bv = unsafe { vld1q_u8(b_bytes.add(i * VEC_BYTES)) };
                let r: uint8x16_t = $op(av, bv);
                let widened = vpaddlq_u8(vcntq_u8(r));
                total += u64::from(vaddvq_u16(widened));
                i += 1;
            }

            total as u32
        }
    };
}

bitwise_into!(
    and_into,
    "NEON `_card` AND of two 1024-word bitmaps. Stores `a & b` into `out` and returns the result cardinality.",
    |a, b| vandq_u8(a, b)
);
bitwise_into!(
    or_into,
    "NEON `_card` OR of two 1024-word bitmaps. Stores `a | b` into `out` and returns the result cardinality.",
    |a, b| vorrq_u8(a, b)
);
bitwise_into!(
    xor_into,
    "NEON `_card` XOR of two 1024-word bitmaps. Stores `a ^ b` into `out` and returns the result cardinality.",
    |a, b| veorq_u8(a, b)
);
// `vbicq_u8(a, b)` computes `a AND NOT b` — the BIC (bit clear) instruction.
bitwise_into!(
    andnot_into,
    "NEON `_card` AND-NOT of two 1024-word bitmaps. Stores `a & !b` into `out` and returns the result cardinality.",
    |a, b| vbicq_u8(a, b)
);

bitwise_into_nocard!(
    and_into_nocard,
    "NEON `_nocard` AND of two 1024-word bitmaps. Stores `a & b` into `out` without computing cardinality.",
    |a, b| vandq_u8(a, b)
);
bitwise_into_nocard!(
    or_into_nocard,
    "NEON `_nocard` OR of two 1024-word bitmaps. Stores `a | b` into `out` without computing cardinality.",
    |a, b| vorrq_u8(a, b)
);
bitwise_into_nocard!(
    xor_into_nocard,
    "NEON `_nocard` XOR of two 1024-word bitmaps. Stores `a ^ b` into `out` without computing cardinality.",
    |a, b| veorq_u8(a, b)
);
bitwise_into_nocard!(
    andnot_into_nocard,
    "NEON `_nocard` AND-NOT of two 1024-word bitmaps. Stores `a & !b` into `out` without computing cardinality.",
    |a, b| vbicq_u8(a, b)
);

bitwise_cardinality!(
    and_cardinality,
    "NEON `_justcard` AND of two 1024-word bitmaps. Returns `popcount(a & b)`.",
    |a, b| vandq_u8(a, b)
);
bitwise_cardinality!(
    or_cardinality,
    "NEON `_justcard` OR of two 1024-word bitmaps. Returns `popcount(a | b)`.",
    |a, b| vorrq_u8(a, b)
);
bitwise_cardinality!(
    xor_cardinality,
    "NEON `_justcard` XOR of two 1024-word bitmaps. Returns `popcount(a ^ b)`.",
    |a, b| veorq_u8(a, b)
);
bitwise_cardinality!(
    andnot_cardinality,
    "NEON `_justcard` AND-NOT of two 1024-word bitmaps. Returns `popcount(a & !b)`.",
    |a, b| vbicq_u8(a, b)
);

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

    #[test]
    fn neon_bitmap_kernels_match_scalar() {
        let (a, b) = distinct_bitmaps();
        // SAFETY: NEON is mandatory on AArch64.
        unsafe {
            let mut out_simd = [0_u64; BITMAP_WORDS];
            let mut out_scalar = [0_u64; BITMAP_WORDS];

            let card_simd = and_into(&a, &b, &mut out_simd);
            let card_scalar = scalar::and_into(&a, &b, &mut out_scalar);
            assert_eq!(card_simd, card_scalar);
            assert_eq!(out_simd[..], out_scalar[..]);
            assert_eq!(and_cardinality(&a, &b), card_scalar);

            let card_simd = or_into(&a, &b, &mut out_simd);
            let card_scalar = scalar::or_into(&a, &b, &mut out_scalar);
            assert_eq!(card_simd, card_scalar);
            assert_eq!(out_simd[..], out_scalar[..]);
            assert_eq!(or_cardinality(&a, &b), card_scalar);

            let card_simd = xor_into(&a, &b, &mut out_simd);
            let card_scalar = scalar::xor_into(&a, &b, &mut out_scalar);
            assert_eq!(card_simd, card_scalar);
            assert_eq!(out_simd[..], out_scalar[..]);
            assert_eq!(xor_cardinality(&a, &b), card_scalar);

            let card_simd = andnot_into(&a, &b, &mut out_simd);
            let card_scalar = scalar::andnot_into(&a, &b, &mut out_scalar);
            assert_eq!(card_simd, card_scalar);
            assert_eq!(out_simd[..], out_scalar[..]);
            assert_eq!(andnot_cardinality(&a, &b), card_scalar);

            xor_into_nocard(&a, &b, &mut out_simd);
            scalar::xor_into_nocard(&a, &b, &mut out_scalar);
            assert_eq!(out_simd[..], out_scalar[..]);
        }
    }
}

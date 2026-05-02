//! AVX-512 bitmap × bitmap kernels.
//!
//! Each AND/OR/XOR/ANDNOT kernel processes 4 × 512-bit vectors per outer
//! iteration over a 1024 × `u64` bitmap container. The cardinality
//! variants use `_mm512_popcnt_epi64` (the AVX-512 VPOPCNTDQ instruction)
//! which is one cycle per 64-bit lane on Ice Lake / Zen 4 and later.
//!
//! Reference: Lemire et al., "Roaring Bitmaps: Implementation of an
//! Optimized Software Library", SPE 48(4), 2018 (arXiv 1709.07821).

#![allow(clippy::cast_lossless)]

use crate::bitmap::containers::BITMAP_WORDS;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_andnot_si512, _mm512_loadu_si512,
    _mm512_or_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64, _mm512_setzero_si512,
    _mm512_storeu_si512, _mm512_xor_si512,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_and_si512, _mm512_andnot_si512, _mm512_loadu_si512,
    _mm512_or_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64, _mm512_setzero_si512,
    _mm512_storeu_si512, _mm512_xor_si512,
};

/// 512-bit vector covers 64 bytes = 8 u64 words.
const VEC_WORDS: usize = 8;

/// Bitmap container has 1024 / 8 = 128 AVX-512 vectors.
const TOTAL_VECTORS: usize = BITMAP_WORDS / VEC_WORDS;

/// Returns true when AVX-512F + AVX-512VPOPCNTDQ are available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vpopcntdq")
}

/// Returns true when AVX-512F + AVX-512VPOPCNTDQ are available (no_std stub).
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX-512 bitwise op + VPOPCNTQ accumulator.
///
/// # Safety
///
/// AVX-512F + AVX-512VPOPCNTDQ must be available.
macro_rules! bitwise_into {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// AVX-512F + AVX-512VPOPCNTDQ must be available; the caller asserts via runtime check.
        #[target_feature(enable = "avx512f,avx512vpopcntdq")]
        #[must_use]
        pub unsafe fn $name(
            a: &[u64; BITMAP_WORDS],
            b: &[u64; BITMAP_WORDS],
            out: &mut [u64; BITMAP_WORDS],
        ) -> u32 {
            let mut acc0 = _mm512_setzero_si512();
            let mut acc1 = _mm512_setzero_si512();
            let mut acc2 = _mm512_setzero_si512();
            let mut acc3 = _mm512_setzero_si512();

            // 128 vectors / 4-vec unroll = 32 outer iterations.
            let mut i = 0_usize;
            while i + 4 <= TOTAL_VECTORS {
                // SAFETY: i + 3 < TOTAL_VECTORS bounds every load and store.
                let a0 =
                    unsafe { _mm512_loadu_si512(a.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let a1 = unsafe {
                    _mm512_loadu_si512(a.as_ptr().add((i + 1) * VEC_WORDS).cast::<__m512i>())
                };
                let a2 = unsafe {
                    _mm512_loadu_si512(a.as_ptr().add((i + 2) * VEC_WORDS).cast::<__m512i>())
                };
                let a3 = unsafe {
                    _mm512_loadu_si512(a.as_ptr().add((i + 3) * VEC_WORDS).cast::<__m512i>())
                };
                let b0 =
                    unsafe { _mm512_loadu_si512(b.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let b1 = unsafe {
                    _mm512_loadu_si512(b.as_ptr().add((i + 1) * VEC_WORDS).cast::<__m512i>())
                };
                let b2 = unsafe {
                    _mm512_loadu_si512(b.as_ptr().add((i + 2) * VEC_WORDS).cast::<__m512i>())
                };
                let b3 = unsafe {
                    _mm512_loadu_si512(b.as_ptr().add((i + 3) * VEC_WORDS).cast::<__m512i>())
                };
                let r0: __m512i = $op(a0, b0);
                let r1: __m512i = $op(a1, b1);
                let r2: __m512i = $op(a2, b2);
                let r3: __m512i = $op(a3, b3);

                unsafe {
                    _mm512_storeu_si512(out.as_mut_ptr().add(i * VEC_WORDS).cast::<__m512i>(), r0);
                    _mm512_storeu_si512(
                        out.as_mut_ptr().add((i + 1) * VEC_WORDS).cast::<__m512i>(),
                        r1,
                    );
                    _mm512_storeu_si512(
                        out.as_mut_ptr().add((i + 2) * VEC_WORDS).cast::<__m512i>(),
                        r2,
                    );
                    _mm512_storeu_si512(
                        out.as_mut_ptr().add((i + 3) * VEC_WORDS).cast::<__m512i>(),
                        r3,
                    );
                }

                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(r0));
                acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(r1));
                acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(r2));
                acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(r3));

                i += 4;
            }

            // Single-vector tail.
            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load/store.
                let av =
                    unsafe { _mm512_loadu_si512(a.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let bv =
                    unsafe { _mm512_loadu_si512(b.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let r: __m512i = $op(av, bv);
                unsafe {
                    _mm512_storeu_si512(out.as_mut_ptr().add(i * VEC_WORDS).cast::<__m512i>(), r);
                }
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(r));
                i += 1;
            }

            let sum01 = _mm512_add_epi64(acc0, acc1);
            let sum23 = _mm512_add_epi64(acc2, acc3);
            let sum = _mm512_add_epi64(sum01, sum23);
            _mm512_reduce_add_epi64(sum) as u32
        }
    };
}

/// AVX-512 bitwise op `_nocard` (no popcount).
///
/// # Safety
///
/// AVX-512F must be available.
macro_rules! bitwise_into_nocard {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// AVX-512F must be available; the caller asserts via runtime check.
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(
            a: &[u64; BITMAP_WORDS],
            b: &[u64; BITMAP_WORDS],
            out: &mut [u64; BITMAP_WORDS],
        ) {
            let mut i = 0_usize;
            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load/store.
                let av =
                    unsafe { _mm512_loadu_si512(a.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let bv =
                    unsafe { _mm512_loadu_si512(b.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let r: __m512i = $op(av, bv);
                unsafe {
                    _mm512_storeu_si512(out.as_mut_ptr().add(i * VEC_WORDS).cast::<__m512i>(), r);
                }
                i += 1;
            }
        }
    };
}

/// AVX-512 bitwise op `_justcard` (cardinality only, no store).
///
/// # Safety
///
/// AVX-512F + AVX-512VPOPCNTDQ must be available.
macro_rules! bitwise_cardinality {
    ($name:ident, $doc:literal, $op:expr) => {
        #[doc = $doc]
        ///
        /// # Safety
        ///
        /// AVX-512F + AVX-512VPOPCNTDQ must be available; the caller asserts via runtime check.
        #[target_feature(enable = "avx512f,avx512vpopcntdq")]
        #[must_use]
        pub unsafe fn $name(a: &[u64; BITMAP_WORDS], b: &[u64; BITMAP_WORDS]) -> u32 {
            let mut acc0 = _mm512_setzero_si512();
            let mut acc1 = _mm512_setzero_si512();
            let mut acc2 = _mm512_setzero_si512();
            let mut acc3 = _mm512_setzero_si512();

            let mut i = 0_usize;
            while i + 4 <= TOTAL_VECTORS {
                // SAFETY: bounds reasoning identical to `bitwise_into`.
                let a0 =
                    unsafe { _mm512_loadu_si512(a.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let a1 = unsafe {
                    _mm512_loadu_si512(a.as_ptr().add((i + 1) * VEC_WORDS).cast::<__m512i>())
                };
                let a2 = unsafe {
                    _mm512_loadu_si512(a.as_ptr().add((i + 2) * VEC_WORDS).cast::<__m512i>())
                };
                let a3 = unsafe {
                    _mm512_loadu_si512(a.as_ptr().add((i + 3) * VEC_WORDS).cast::<__m512i>())
                };
                let b0 =
                    unsafe { _mm512_loadu_si512(b.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let b1 = unsafe {
                    _mm512_loadu_si512(b.as_ptr().add((i + 1) * VEC_WORDS).cast::<__m512i>())
                };
                let b2 = unsafe {
                    _mm512_loadu_si512(b.as_ptr().add((i + 2) * VEC_WORDS).cast::<__m512i>())
                };
                let b3 = unsafe {
                    _mm512_loadu_si512(b.as_ptr().add((i + 3) * VEC_WORDS).cast::<__m512i>())
                };
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64($op(a0, b0)));
                acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64($op(a1, b1)));
                acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64($op(a2, b2)));
                acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64($op(a3, b3)));
                i += 4;
            }

            while i < TOTAL_VECTORS {
                // SAFETY: i < TOTAL_VECTORS bounds the load.
                let av =
                    unsafe { _mm512_loadu_si512(a.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                let bv =
                    unsafe { _mm512_loadu_si512(b.as_ptr().add(i * VEC_WORDS).cast::<__m512i>()) };
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64($op(av, bv)));
                i += 1;
            }

            let sum01 = _mm512_add_epi64(acc0, acc1);
            let sum23 = _mm512_add_epi64(acc2, acc3);
            let sum = _mm512_add_epi64(sum01, sum23);
            _mm512_reduce_add_epi64(sum) as u32
        }
    };
}

bitwise_into!(
    and_into,
    "AVX-512 `_card` AND of two 1024-word bitmaps. Stores `a & b` into `out` and returns the result cardinality via VPOPCNTQ.",
    |a, b| _mm512_and_si512(a, b)
);
bitwise_into!(
    or_into,
    "AVX-512 `_card` OR of two 1024-word bitmaps. Stores `a | b` into `out` and returns the result cardinality via VPOPCNTQ.",
    |a, b| _mm512_or_si512(a, b)
);
bitwise_into!(
    xor_into,
    "AVX-512 `_card` XOR of two 1024-word bitmaps. Stores `a ^ b` into `out` and returns the result cardinality via VPOPCNTQ.",
    |a, b| _mm512_xor_si512(a, b)
);
// `_mm512_andnot_si512(a, b)` computes `(~a) & b`; we want `a & ~b`.
bitwise_into!(
    andnot_into,
    "AVX-512 `_card` AND-NOT of two 1024-word bitmaps. Stores `a & !b` into `out` and returns the result cardinality via VPOPCNTQ.",
    |a, b| _mm512_andnot_si512(b, a)
);

bitwise_into_nocard!(
    and_into_nocard,
    "AVX-512 `_nocard` AND of two 1024-word bitmaps. Stores `a & b` into `out` without computing cardinality.",
    |a, b| _mm512_and_si512(a, b)
);
bitwise_into_nocard!(
    or_into_nocard,
    "AVX-512 `_nocard` OR of two 1024-word bitmaps. Stores `a | b` into `out` without computing cardinality.",
    |a, b| _mm512_or_si512(a, b)
);
bitwise_into_nocard!(
    xor_into_nocard,
    "AVX-512 `_nocard` XOR of two 1024-word bitmaps. Stores `a ^ b` into `out` without computing cardinality.",
    |a, b| _mm512_xor_si512(a, b)
);
bitwise_into_nocard!(
    andnot_into_nocard,
    "AVX-512 `_nocard` AND-NOT of two 1024-word bitmaps. Stores `a & !b` into `out` without computing cardinality.",
    |a, b| _mm512_andnot_si512(b, a)
);

bitwise_cardinality!(
    and_cardinality,
    "AVX-512 `_justcard` AND of two 1024-word bitmaps. Returns `popcount(a & b)` via VPOPCNTQ.",
    |a, b| _mm512_and_si512(a, b)
);
bitwise_cardinality!(
    or_cardinality,
    "AVX-512 `_justcard` OR of two 1024-word bitmaps. Returns `popcount(a | b)` via VPOPCNTQ.",
    |a, b| _mm512_or_si512(a, b)
);
bitwise_cardinality!(
    xor_cardinality,
    "AVX-512 `_justcard` XOR of two 1024-word bitmaps. Returns `popcount(a ^ b)` via VPOPCNTQ.",
    |a, b| _mm512_xor_si512(a, b)
);
bitwise_cardinality!(
    andnot_cardinality,
    "AVX-512 `_justcard` AND-NOT of two 1024-word bitmaps. Returns `popcount(a & !b)` via VPOPCNTQ.",
    |a, b| _mm512_andnot_si512(b, a)
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
    fn avx512_bitmap_kernels_match_scalar_when_available() {
        if !is_available() {
            eprintln!("avx512 unavailable; skipping bitmap×bitmap parity test");
            return;
        }
        let (a, b) = distinct_bitmaps();
        // SAFETY: availability checked above.
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

            and_into_nocard(&a, &b, &mut out_simd);
            scalar::and_into_nocard(&a, &b, &mut out_scalar);
            assert_eq!(out_simd[..], out_scalar[..]);
        }
    }
}

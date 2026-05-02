//! AVX2 array × bitmap intersect kernel.
//!
//! For each chunk of 8 sorted u16 values, build an 8-bit "is-set" mask
//! from the bitmap (one bit per value), then use the same Schlegel
//! shuffle table to compact the surviving u16 values into the front of
//! the output register.
//!
//! Reference: Lemire et al., "Roaring Bitmaps: Implementation of an
//! Optimized Software Library", SPE 48(4), 2018 (arXiv 1709.07821), §
//! 5 ("array-bitset intersection"). On AVX2 the SIMD win is mostly in
//! the output materialisation; the bitmap reads themselves are scalar
//! per-lane bit-tests because AVX2 lacks a native bit-gather.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bitmap::containers::BITMAP_WORDS;

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128i, _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128};

use std::sync::OnceLock;

/// Number of u16 lanes per SSE register.
const LANES: usize = 8;

/// Returns true when AVX2 is available at runtime.
///
/// The kernel itself only needs SSSE3 (PSHUFB) for the compress step;
/// it is gated on AVX2 to align with the rest of the bitmap kernels'
/// runtime-dispatch story.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Returns true when AVX2 is available (no_std stub).
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Same shuffle table as the Schlegel SSE4.2 intersect kernel.
static SHUFFLE: OnceLock<[[u8; 16]; 256]> = OnceLock::new();

fn shuffle_table() -> &'static [[u8; 16]; 256] {
    SHUFFLE.get_or_init(build_shuffle_table)
}

fn build_shuffle_table() -> [[u8; 16]; 256] {
    let mut table = [[0xff_u8; 16]; 256];
    for (m, entry) in table.iter_mut().enumerate() {
        let mut shuffle = [0xff_u8; 16];
        let mut dst = 0_usize;
        for i in 0..LANES {
            if (m >> i) & 1 == 1 {
                shuffle[dst] = (2 * i) as u8;
                shuffle[dst + 1] = (2 * i + 1) as u8;
                dst += 2;
            }
        }
        *entry = shuffle;
    }
    table
}

/// AVX2-flavoured array × bitmap intersect (PSHUFB compress).
///
/// # Safety
///
/// AVX2 (and thus SSSE3 for PSHUFB) must be available.
#[target_feature(enable = "avx2")]
pub unsafe fn intersect_array_bitmap(
    array: &[u16],
    bitmap: &[u64; BITMAP_WORDS],
    out: &mut Vec<u16>,
) {
    out.clear();

    if array.is_empty() {
        return;
    }

    let shuf = shuffle_table();

    let mut idx = 0_usize;
    while idx + LANES <= array.len() {
        // Compute the 8-bit mask scalar — AVX2 has no native bit-gather.
        let mut mask = 0_u8;
        for i in 0..LANES {
            let v = array[idx + i] as usize;
            if (bitmap[v >> 6] >> (v & 63)) & 1 == 1 {
                mask |= 1 << i;
            }
        }

        // SAFETY: AVX2 enabled by `target_feature`; PSHUFB available.
        let va = unsafe { _mm_loadu_si128(array.as_ptr().add(idx).cast::<__m128i>()) };
        let shuf_v = unsafe { _mm_loadu_si128(shuf[mask as usize].as_ptr().cast::<__m128i>()) };
        let result = _mm_shuffle_epi8(va, shuf_v);

        let mut scratch = [0_u16; LANES];
        // SAFETY: scratch is 16 bytes.
        unsafe {
            _mm_storeu_si128(scratch.as_mut_ptr().cast::<__m128i>(), result);
        }
        let count = mask.count_ones() as usize;
        for &v in &scratch[..count] {
            out.push(v);
        }

        idx += LANES;
    }

    // Scalar tail.
    for &v in &array[idx..] {
        let vi = v as usize;
        if (bitmap[vi >> 6] >> (vi & 63)) & 1 == 1 {
            out.push(v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::kernels::array_x_bitmap_scalar as scalar;

    fn deterministic_bitmap(seed: u64) -> [u64; BITMAP_WORDS] {
        let mut bm = [0_u64; BITMAP_WORDS];
        let mut state = seed;
        for word in &mut bm {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            *word = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        }
        bm
    }

    fn deterministic_array(seed: u64, density: u16, n: usize) -> Vec<u16> {
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
    fn array_bitmap_intersect_matches_scalar() {
        if !is_available() {
            return;
        }
        for seed in [0_u64, 1, 0xDEAD_BEEF, 0xC0FFEE] {
            let array = deterministic_array(seed, 30, 200);
            let bm = deterministic_bitmap(seed.wrapping_add(1));
            let mut out_simd = Vec::new();
            // SAFETY: availability checked above.
            unsafe {
                intersect_array_bitmap(&array, &bm, &mut out_simd);
            }
            let mut out_scalar = Vec::new();
            scalar::intersect_array_bitmap(&array, &bm, &mut out_scalar);
            assert_eq!(out_simd, out_scalar, "seed {seed}");
        }
    }

    #[test]
    fn array_bitmap_handles_empty_array() {
        if !is_available() {
            return;
        }
        let bm = deterministic_bitmap(0);
        let mut out = Vec::new();
        // SAFETY: availability checked above.
        unsafe {
            intersect_array_bitmap(&[], &bm, &mut out);
        }
        assert!(out.is_empty());
    }

    #[test]
    fn array_bitmap_handles_block_boundaries() {
        if !is_available() {
            return;
        }
        let bm = deterministic_bitmap(0xCAFE);
        for len in [1_usize, 7, 8, 9, 15, 16, 17, 33, 65] {
            let array: Vec<u16> = (0..len as u16).collect();
            let mut out_simd = Vec::new();
            // SAFETY.
            unsafe {
                intersect_array_bitmap(&array, &bm, &mut out_simd);
            }
            let mut out_scalar = Vec::new();
            scalar::intersect_array_bitmap(&array, &bm, &mut out_scalar);
            assert_eq!(out_simd, out_scalar, "len={len}");
        }
    }
}

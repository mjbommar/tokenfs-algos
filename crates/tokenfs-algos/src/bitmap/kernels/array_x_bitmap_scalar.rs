//! Portable scalar array × bitmap kernels.
//!
//! For an array of sorted `u16` values and a 65 536-bit bitmap, the four
//! Boolean ops reduce to bit-tests against the bitmap word `bm[v >> 6]
//! & (1 << (v & 63))`. These scalar versions are the parity oracles for
//! the SIMD backends and are also production-grade — bit-tests are
//! already cache-friendly so the SIMD win comes from output
//! materialisation (PSHUFB / VPCOMPRESSD), not the test itself.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bitmap::containers::BITMAP_WORDS;

/// Returns whether `bitmap` contains the value `v`.
#[inline]
pub fn bitmap_contains(bitmap: &[u64; BITMAP_WORDS], v: u16) -> bool {
    let v = v as usize;
    (bitmap[v >> 6] >> (v & 63)) & 1 == 1
}

/// Intersect a sorted `u16` array with a 65 536-bit bitmap.
///
/// Emits ascending values from `array` that are also set in `bitmap`.
pub fn intersect_array_bitmap(array: &[u16], bitmap: &[u64; BITMAP_WORDS], out: &mut Vec<u16>) {
    out.clear();
    for &v in array {
        if bitmap_contains(bitmap, v) {
            out.push(v);
        }
    }
}

/// Returns the cardinality of `array INTERSECT bitmap`.
#[must_use]
pub fn intersect_cardinality_array_bitmap(array: &[u16], bitmap: &[u64; BITMAP_WORDS]) -> u32 {
    let mut card = 0_u32;
    for &v in array {
        if bitmap_contains(bitmap, v) {
            card += 1;
        }
    }
    card
}

/// Difference `array \ bitmap`.
///
/// Emits ascending values from `array` that are NOT set in `bitmap`.
pub fn difference_array_bitmap(array: &[u16], bitmap: &[u64; BITMAP_WORDS], out: &mut Vec<u16>) {
    out.clear();
    for &v in array {
        if !bitmap_contains(bitmap, v) {
            out.push(v);
        }
    }
}

/// Difference `bitmap \ array` — bitmap output.
///
/// Copies `bitmap` into `out_bitmap`, then clears each bit in
/// `out_bitmap` that corresponds to an element of `array`.
pub fn difference_bitmap_array(
    bitmap: &[u64; BITMAP_WORDS],
    array: &[u16],
    out_bitmap: &mut [u64; BITMAP_WORDS],
) {
    *out_bitmap = *bitmap;
    for &v in array {
        let v = v as usize;
        out_bitmap[v >> 6] &= !(1_u64 << (v & 63));
    }
}

/// Union `array UNION bitmap` — bitmap output.
///
/// Copies `bitmap` into `out_bitmap`, then sets each bit corresponding
/// to an element of `array`.
pub fn union_array_bitmap(
    array: &[u16],
    bitmap: &[u64; BITMAP_WORDS],
    out_bitmap: &mut [u64; BITMAP_WORDS],
) {
    *out_bitmap = *bitmap;
    for &v in array {
        let v = v as usize;
        out_bitmap[v >> 6] |= 1_u64 << (v & 63);
    }
}

/// XOR `array XOR bitmap` — bitmap output.
///
/// Copies `bitmap` into `out_bitmap`, then toggles each bit
/// corresponding to an element of `array`.
pub fn xor_array_bitmap(
    array: &[u16],
    bitmap: &[u64; BITMAP_WORDS],
    out_bitmap: &mut [u64; BITMAP_WORDS],
) {
    *out_bitmap = *bitmap;
    for &v in array {
        let v = v as usize;
        out_bitmap[v >> 6] ^= 1_u64 << (v & 63);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn intersect_basic() {
        let bm = deterministic_bitmap(0xC0FFEE);
        let array: Vec<u16> = (0..1000).map(|i| i as u16).collect();
        let mut out = Vec::new();
        intersect_array_bitmap(&array, &bm, &mut out);
        let card = intersect_cardinality_array_bitmap(&array, &bm);
        assert_eq!(out.len() as u32, card);
        // Each element of `out` must be present in both inputs.
        for &v in &out {
            assert!(bitmap_contains(&bm, v));
            assert!(array.binary_search(&v).is_ok());
        }
    }

    #[test]
    fn difference_array_minus_bitmap() {
        let mut bm = [0_u64; BITMAP_WORDS];
        bm[0] = 0b10110;
        // bm has values {1, 2, 4}.
        let array = vec![0_u16, 1, 2, 3];
        let mut out = Vec::new();
        difference_array_bitmap(&array, &bm, &mut out);
        // Values in array not in bitmap: {0, 3}.
        assert_eq!(out, vec![0, 3]);
    }

    #[test]
    fn union_array_with_bitmap() {
        let mut bm = [0_u64; BITMAP_WORDS];
        bm[0] = 0b10;
        let array = vec![0_u16, 2];
        let mut out_bm = [0_u64; BITMAP_WORDS];
        union_array_bitmap(&array, &bm, &mut out_bm);
        assert_eq!(out_bm[0], 0b111);
    }

    #[test]
    fn xor_array_with_bitmap() {
        let mut bm = [0_u64; BITMAP_WORDS];
        bm[0] = 0b110;
        let array = vec![1_u16, 2];
        let mut out_bm = [0_u64; BITMAP_WORDS];
        xor_array_bitmap(&array, &bm, &mut out_bm);
        // Toggling bits 1 and 2 of {1, 2} = {} → 0.
        assert_eq!(out_bm[0], 0);
    }
}

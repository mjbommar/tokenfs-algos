//! SSE4.2 array × array intersect via Schlegel's pcmpestrm kernel.
//!
//! Reference: Schlegel, Willhalm, Lehner, "Fast Sorted-Set Intersection
//! using SIMD Instructions", ADMS @ VLDB 2011. CRoaring's `intersect_
//! vector16` is the canonical production implementation; this kernel is
//! the same algorithm but written in safe-ish Rust over the
//! `core::arch::x86_64` intrinsics.
//!
//! ## Algorithm
//!
//! For each pair of 8 × `u16` chunks `Va`, `Vb`:
//!
//! 1. `mask = _mm_cmpestrm::<_SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY |
//!    _SIDD_BIT_MASK>(Vb, 8, Va, 8)` — produces an 8-bit mask in the
//!    low byte of `xmm0`. Bit `i` is set iff `Va[i]` matches any
//!    element of `Vb`. We use `pcmpestrm` (explicit lengths) rather
//!    than `pcmpistrm` (implicit-length) because the latter treats
//!    `0` as a string-terminator — a `0` element in `Va` or `Vb`
//!    would silently truncate the comparison range and miss matches.
//! 2. `shuf = SHUFFLE_TABLE[mask]` — 16-byte permutation that compacts
//!    the matching u16 lanes to the front. Indices `0xFF` zero the
//!    destination lane on PSHUFB.
//! 3. `out = _mm_shuffle_epi8(Va, shuf)` — gathers survivors.
//! 4. Store the low `popcount(mask) * 2` bytes (or all 16 bytes — the
//!    trailing lanes are zeroed and we advance the output pointer by
//!    `popcount(mask)`).
//! 5. Advance `Va` or `Vb` based on which had the smaller maximum
//!    element so that the merge invariant holds.
//!
//! The final partial chunks (where one side has < 8 remaining) are
//! finished by the scalar merge oracle.
//!
//! ## Shuffle table format
//!
//! 256 entries × `[u8; 16]`. Index `m` corresponds to mask `m`. For
//! every set bit `i` in `m`, two consecutive bytes of the shuffle entry
//! are filled with the indices `2 * i` and `2 * i + 1` (the two bytes of
//! `Va[i]`). The destination position is incremented by 2 bytes per set
//! bit. Trailing destination bytes hold `0xFF` so PSHUFB zeros them.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, _SIDD_BIT_MASK, _SIDD_CMP_EQUAL_ANY, _SIDD_UWORD_OPS, _mm_cmpestrm, _mm_extract_epi32,
    _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _SIDD_BIT_MASK, _SIDD_CMP_EQUAL_ANY, _SIDD_UWORD_OPS, _mm_cmpestrm, _mm_extract_epi32,
    _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128,
};

use std::sync::OnceLock;

/// Returns true when SSE4.2 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("sse4.2")
}

/// Returns true when SSE4.2 is available (no_std stub).
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Number of u16 lanes per SSE register.
const LANES: usize = 8;

/// Shuffle table: 256 entries of `[u8; 16]`. 4 KiB total.
static SHUFFLE: OnceLock<[[u8; 16]; 256]> = OnceLock::new();

/// Returns the (potentially first-use-initialised) shuffle table.
fn shuffle_table() -> &'static [[u8; 16]; 256] {
    SHUFFLE.get_or_init(build_shuffle_table)
}

/// Builds the Schlegel shuffle table at first use.
///
/// For each 8-bit mask `m`, write the byte indices `2 * i` and `2 * i +
/// 1` into successive destination bytes for every set bit `i ∈ 0..8`.
/// Trailing destination bytes hold `0xFF` so PSHUFB zeros them.
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

/// Schlegel pcmpistrm-based intersect of two sorted `u16` slices.
///
/// Pushes matching elements onto `out` in ascending order. Falls back to
/// the scalar merge oracle for the trailing partial chunks (< 8 u16 on
/// either side) and for empty inputs.
///
/// # Safety
///
/// The caller must ensure SSE4.2 is available.
#[target_feature(enable = "sse4.2")]
pub unsafe fn intersect(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    out.clear();

    if a.is_empty() || b.is_empty() {
        return;
    }

    let shuf = shuffle_table();

    let mut ia = 0_usize;
    let mut ib = 0_usize;

    // Outer loop while both sides have a full 8-lane chunk available.
    while ia + LANES <= a.len() && ib + LANES <= b.len() {
        // Load 8 u16 from each side.
        // SAFETY: 8 lanes (16 bytes) fit at offsets `ia` and `ib` per
        // the loop condition; SSE4.2 enabled by enclosing `target_feature`.
        let va = unsafe { _mm_loadu_si128(a.as_ptr().add(ia).cast::<__m128i>()) };
        let vb = unsafe { _mm_loadu_si128(b.as_ptr().add(ib).cast::<__m128i>()) };

        // pcmpestrm with EQUAL_ANY + UWORD_OPS + BIT_MASK returns an 8-bit
        // mask in the low byte of xmm0; bit `i` of the result is set iff
        // `Va[i]` is found in `Vb`. We pass explicit length 8 for both
        // operands so a `0` element does not silently truncate the
        // comparison range (the Schlegel paper's `pcmpistrm` form has
        // that footgun on inputs containing 0).
        // (Pure-register intrinsic; no memory; safe under `target_feature`.)
        let mask_vec = _mm_cmpestrm::<{ _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK }>(
            vb,
            LANES as i32,
            va,
            LANES as i32,
        );
        // Extract the low 32 bits — only the low 8 bits hold the mask.
        let mask = (_mm_extract_epi32::<0>(mask_vec) as u32 & 0xFF) as usize;

        // Shuffle Va to compact the surviving lanes to the front of the
        // register. The shuffle table holds 0xFF in trailing positions
        // so PSHUFB zeros them.
        // SAFETY: shuffle table entries are 16 bytes, aligned for unaligned
        // load.
        let shuf_v = unsafe { _mm_loadu_si128(shuf[mask].as_ptr().cast::<__m128i>()) };
        let result = _mm_shuffle_epi8(va, shuf_v);

        // Store 16 bytes (8 u16) into a stack scratch buffer, then push
        // the first `popcount(mask)` u16s onto `out`. We can't store
        // directly into `out`'s backing buffer because `out` is `Vec`
        // and may need to grow.
        let mut scratch = [0_u16; LANES];
        // SAFETY: scratch is 16 bytes; SSE4.2 implies SSSE3 storeu.
        unsafe {
            _mm_storeu_si128(scratch.as_mut_ptr().cast::<__m128i>(), result);
        }
        let count = (mask as u8).count_ones() as usize;
        // Push only the survivors; the rest of `scratch` holds zeros.
        for &v in &scratch[..count] {
            out.push(v);
        }

        // Advance the side whose current 8th lane has the smaller
        // maximum so that the merge invariant holds. (PCMPISTRM compared
        // each Va[i] against all of Vb; advancing the side with the
        // smaller max guarantees no missed matches in the next pass.)
        let max_a = a[ia + LANES - 1];
        let max_b = b[ib + LANES - 1];
        match max_a.cmp(&max_b) {
            core::cmp::Ordering::Less => ia += LANES,
            core::cmp::Ordering::Greater => ib += LANES,
            core::cmp::Ordering::Equal => {
                ia += LANES;
                ib += LANES;
            }
        }
    }

    // Scalar merge tail.
    super::array_x_array_scalar::intersect_into_remainder(&a[ia..], &b[ib..], out);
}

/// Schlegel pcmpistrm-based intersect cardinality (no output materialisation).
///
/// Replaces the `pshufb` + store with `popcount(mask)` per chunk.
///
/// # Safety
///
/// The caller must ensure SSE4.2 is available.
#[target_feature(enable = "sse4.2")]
#[must_use]
pub unsafe fn intersect_cardinality(a: &[u16], b: &[u16]) -> u32 {
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    let mut card: u32 = 0;
    let mut ia = 0_usize;
    let mut ib = 0_usize;

    while ia + LANES <= a.len() && ib + LANES <= b.len() {
        // SAFETY: same bounds reasoning as `intersect`.
        let va = unsafe { _mm_loadu_si128(a.as_ptr().add(ia).cast::<__m128i>()) };
        let vb = unsafe { _mm_loadu_si128(b.as_ptr().add(ib).cast::<__m128i>()) };
        // (Pure-register intrinsic; no memory; safe under `target_feature`.)
        let mask_vec = _mm_cmpestrm::<{ _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK }>(
            vb,
            LANES as i32,
            va,
            LANES as i32,
        );
        let mask = (_mm_extract_epi32::<0>(mask_vec) as u32 & 0xFF) as u8;
        card += mask.count_ones();

        let max_a = a[ia + LANES - 1];
        let max_b = b[ib + LANES - 1];
        match max_a.cmp(&max_b) {
            core::cmp::Ordering::Less => ia += LANES,
            core::cmp::Ordering::Greater => ib += LANES,
            core::cmp::Ordering::Equal => {
                ia += LANES;
                ib += LANES;
            }
        }
    }

    card + super::array_x_array_scalar::intersect_cardinality(&a[ia..], &b[ib..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::kernels::array_x_array_scalar as scalar;

    fn deterministic_sorted_u16(n: usize, seed: u64, density: u16) -> Vec<u16> {
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
    fn shuffle_table_is_consistent() {
        let table = shuffle_table();
        // Mask 0b00000000 → all 0xFF (zero output).
        assert_eq!(table[0], [0xFF; 16]);
        // Mask 0b00000001 → first two bytes [0, 1], rest 0xFF.
        let m1 = table[0b0000_0001];
        assert_eq!(m1[0], 0);
        assert_eq!(m1[1], 1);
        assert_eq!(m1[2], 0xFF);
        // Mask 0b00000010 → first two bytes [2, 3], rest 0xFF.
        let m2 = table[0b0000_0010];
        assert_eq!(m2[0], 2);
        assert_eq!(m2[1], 3);
        assert_eq!(m2[2], 0xFF);
        // Mask 0b11111111 → identity-shuffle (all 8 lanes survive in order).
        let m_all = table[0xFF];
        for (i, &byte) in m_all.iter().enumerate() {
            assert_eq!(byte, i as u8);
        }
    }

    #[test]
    fn intersect_matches_scalar_random() {
        if !is_available() {
            return;
        }
        for seed in [0_u64, 1, 0xDEAD_BEEF, 0xC0FFEE] {
            let a = deterministic_sorted_u16(200, seed, 50);
            let b = deterministic_sorted_u16(200, seed.wrapping_add(1), 50);
            let mut out_simd = Vec::new();
            // SAFETY: availability checked above.
            unsafe {
                intersect(&a, &b, &mut out_simd);
            }
            let mut out_scalar = Vec::new();
            scalar::intersect(&a, &b, &mut out_scalar);
            assert_eq!(out_simd, out_scalar, "seed {seed}");
            // SAFETY: availability checked above.
            let card = unsafe { intersect_cardinality(&a, &b) };
            assert_eq!(card as usize, out_scalar.len());
        }
    }

    #[test]
    fn intersect_handles_empty_and_short_inputs() {
        if !is_available() {
            return;
        }
        let mut out = Vec::new();
        // SAFETY: availability checked above.
        unsafe {
            intersect(&[], &[1, 2, 3], &mut out);
        }
        assert!(out.is_empty());
        // SAFETY.
        unsafe {
            intersect(&[1, 2, 3], &[], &mut out);
        }
        assert!(out.is_empty());
        // SAFETY.
        unsafe {
            intersect(&[1, 2, 3], &[2, 3, 4], &mut out);
        }
        assert_eq!(out, vec![2, 3]);
    }

    #[test]
    fn intersect_handles_identical_inputs() {
        if !is_available() {
            return;
        }
        let a: Vec<u16> = (0..1000).map(|i| i as u16).collect();
        let mut out = Vec::new();
        // SAFETY: availability checked above.
        unsafe {
            intersect(&a, &a, &mut out);
        }
        assert_eq!(out, a);
    }

    #[test]
    fn intersect_handles_disjoint_inputs() {
        if !is_available() {
            return;
        }
        let a: Vec<u16> = (0..500).map(|i| i as u16).collect();
        let b: Vec<u16> = (500..1000).map(|i| i as u16).collect();
        let mut out = Vec::new();
        // SAFETY: availability checked above.
        unsafe {
            intersect(&a, &b, &mut out);
        }
        assert!(out.is_empty());
    }

    #[test]
    fn intersect_handles_at_block_boundaries() {
        if !is_available() {
            return;
        }
        // Lengths around the 8-lane block boundary.
        for len_a in [1_usize, 7, 8, 9, 15, 16, 17] {
            for len_b in [1_usize, 7, 8, 9, 15, 16, 17] {
                let a: Vec<u16> = (0..len_a as u16).collect();
                let b: Vec<u16> = (0..len_b as u16).collect();
                let mut out_simd = Vec::new();
                // SAFETY: availability checked above.
                unsafe {
                    intersect(&a, &b, &mut out_simd);
                }
                let mut out_scalar = Vec::new();
                scalar::intersect(&a, &b, &mut out_scalar);
                assert_eq!(out_simd, out_scalar, "len_a={len_a} len_b={len_b}");
            }
        }
    }
}

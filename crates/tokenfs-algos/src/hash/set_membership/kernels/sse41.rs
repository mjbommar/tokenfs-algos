//! SSE4.1 set-membership kernel.
//!
//! Extracted from set_membership.rs as part of audit-R10 T1.3
//! follow-up to v0.4.2's #180 file-split — sse41 was missed in
//! the original split. Now properly file-gated behind
//! `arch-pinned-kernels`.

use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, _mm_cmpeq_epi32, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi32, _mm_testz_si128,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_cmpeq_epi32, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi32, _mm_testz_si128,
};

/// 4 lanes (16 bytes) per SSE vector.
const LANES: usize = 4;

/// 4x unrolled = 16 lanes (64 B) per outer iteration. The OR of
/// four independent compares amortises the testz cost across the
/// unrolled block; hits inside still branch out on the next
/// iteration's testz.
const UNROLL_VECTORS: usize = 4;

/// Returns true when SSE4.1 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("sse4.1")
}

/// Returns true when SSE4.1 is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// SSE4.1 single-needle membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.1.
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
    let broadcast = _mm_set1_epi32(needle as i32);
    let mut index = 0_usize;
    let unroll_lanes = LANES * UNROLL_VECTORS;

    while index + unroll_lanes <= haystack.len() {
        // SAFETY: each load reads 16 bytes (4 u32s); the loop
        // bound enforces `index + UNROLL_VECTORS * LANES <=
        // haystack.len()`. SSE4.1 is enabled by the enclosing
        // target_feature.
        let v0 = unsafe { _mm_loadu_si128(haystack.as_ptr().add(index).cast::<__m128i>()) };
        let v1 = unsafe { _mm_loadu_si128(haystack.as_ptr().add(index + LANES).cast::<__m128i>()) };
        let v2 =
            unsafe { _mm_loadu_si128(haystack.as_ptr().add(index + 2 * LANES).cast::<__m128i>()) };
        let v3 =
            unsafe { _mm_loadu_si128(haystack.as_ptr().add(index + 3 * LANES).cast::<__m128i>()) };
        let c0 = _mm_cmpeq_epi32(v0, broadcast);
        let c1 = _mm_cmpeq_epi32(v1, broadcast);
        let c2 = _mm_cmpeq_epi32(v2, broadcast);
        let c3 = _mm_cmpeq_epi32(v3, broadcast);
        let or01 = _mm_or_si128(c0, c1);
        let or23 = _mm_or_si128(c2, c3);
        let any = _mm_or_si128(or01, or23);
        if _mm_testz_si128(any, any) == 0 {
            return true;
        }
        index += unroll_lanes;
    }

    // Single-vector loop for the leftover lanes after the unrolled
    // block.
    while index + LANES <= haystack.len() {
        // SAFETY: index + LANES <= haystack.len() bounds the load.
        let chunk = unsafe { _mm_loadu_si128(haystack.as_ptr().add(index).cast::<__m128i>()) };
        let cmp = _mm_cmpeq_epi32(chunk, broadcast);
        if _mm_testz_si128(cmp, cmp) == 0 {
            return true;
        }
        index += LANES;
    }

    scalar::contains_u32(&haystack[index..], needle)
}

/// SSE4.1 batched membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSE4.1.
///
/// # Panics
///
/// Panics if `needles.len() != out.len()`. Available only with
/// `feature = "userspace"` (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    assert_eq!(needles.len(), out.len());
    // SAFETY: SSE4.1 supplied; precondition checked above.
    unsafe { contains_u32_batch_unchecked(haystack, needles, out) }
}

/// Unchecked SSE4.1 batched membership.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and
/// `needles.len() == out.len()`.
#[target_feature(enable = "sse4.1")]
pub unsafe fn contains_u32_batch_unchecked(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    for (needle, slot) in needles.iter().zip(out.iter_mut()) {
        // SAFETY: target_feature(enable = "sse4.1") on this fn
        // forwards the SSE4.1 precondition to the inner kernel.
        *slot = unsafe { contains_u32(haystack, *needle) };
    }
}

use super::scalar;

use core::arch::aarch64::{vceqq_u32, vdupq_n_u32, vld1q_u32, vmaxvq_u32};

/// 4 lanes (16 bytes) per NEON vector.
const LANES: usize = 4;

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this exists for API symmetry
/// with the x86 `is_available` helpers.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// NEON single-needle membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
    let broadcast = vdupq_n_u32(needle);
    let mut index = 0_usize;

    while index + LANES <= haystack.len() {
        // SAFETY: index + LANES <= haystack.len() bounds the load
        // and the enclosing target_feature supplies NEON.
        let v = unsafe { vld1q_u32(haystack.as_ptr().add(index)) };
        let cmp = vceqq_u32(v, broadcast);
        if vmaxvq_u32(cmp) != 0 {
            return true;
        }
        index += LANES;
    }

    scalar::contains_u32(&haystack[index..], needle)
}

/// NEON batched membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
///
/// # Panics
///
/// Panics if `needles.len() != out.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    assert_eq!(needles.len(), out.len());
    for (needle, slot) in needles.iter().zip(out.iter_mut()) {
        // SAFETY: target_feature(enable = "neon") on this fn
        // forwards the NEON precondition to the inner kernel.
        *slot = unsafe { contains_u32(haystack, *needle) };
    }
}

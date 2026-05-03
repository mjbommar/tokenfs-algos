use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m512i, _mm512_cmpeq_epi32_mask, _mm512_loadu_si512, _mm512_set1_epi32};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m512i, _mm512_cmpeq_epi32_mask, _mm512_loadu_si512, _mm512_set1_epi32};

/// 16 lanes (64 bytes) per AVX-512 vector.
const LANES: usize = 16;

/// Returns true when AVX-512F is available at runtime.
///
/// `_mm512_cmpeq_epi32_mask` is part of AVX-512F.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

/// Returns true when AVX-512F is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX-512 single-needle membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
    let broadcast = _mm512_set1_epi32(needle as i32);
    let mut index = 0_usize;

    while index + LANES <= haystack.len() {
        // SAFETY: index + LANES <= haystack.len() bounds the load
        // and the enclosing target_feature supplies AVX-512F.
        let v = unsafe { _mm512_loadu_si512(haystack.as_ptr().add(index).cast::<__m512i>()) };
        let mask = _mm512_cmpeq_epi32_mask(v, broadcast);
        if mask != 0 {
            return true;
        }
        index += LANES;
    }

    scalar::contains_u32(&haystack[index..], needle)
}

/// AVX-512 batched membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F.
///
/// # Panics
///
/// Panics if `needles.len() != out.len()`.
#[target_feature(enable = "avx512f")]
pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    assert_eq!(needles.len(), out.len());
    for (needle, slot) in needles.iter().zip(out.iter_mut()) {
        // SAFETY: target_feature(enable = "avx512f") on this fn
        // forwards the AVX-512F precondition to the inner kernel.
        *slot = unsafe { contains_u32(haystack, *needle) };
    }
}

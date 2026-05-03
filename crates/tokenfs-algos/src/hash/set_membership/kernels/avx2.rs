use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_testz_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_testz_si256,
};

/// 8 lanes (32 bytes) per AVX2 vector.
const LANES: usize = 8;

/// 4x unrolled = 32 lanes (128 B) per outer iteration. The OR of
/// four independent compares short-circuits the early-exit cost
/// for haystacks that miss; for haystacks that hit, the early
/// exit triggers within the 32-lane window so the unroll cost is
/// at most one extra compare.
const UNROLL_VECTORS: usize = 4;

/// Returns true when AVX2 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Returns true when AVX2 is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX2 single-needle membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
    let broadcast = _mm256_set1_epi32(needle as i32);
    let mut index = 0_usize;
    let unroll_lanes = LANES * UNROLL_VECTORS;

    // Outer loop: process UNROLL_VECTORS vectors per iteration; OR
    // their compare results together so the testz amortises across
    // the unrolled block. Hits inside the block still branch out
    // on the next iteration's testz.
    while index + unroll_lanes <= haystack.len() {
        // SAFETY: each load reads 32 bytes (8 u32s) and `index +
        // UNROLL_VECTORS * LANES <= haystack.len()` is enforced by
        // the loop condition. AVX2 is enabled by the enclosing
        // target_feature.
        let v0 = unsafe { _mm256_loadu_si256(haystack.as_ptr().add(index).cast::<__m256i>()) };
        let v1 =
            unsafe { _mm256_loadu_si256(haystack.as_ptr().add(index + LANES).cast::<__m256i>()) };
        let v2 = unsafe {
            _mm256_loadu_si256(haystack.as_ptr().add(index + 2 * LANES).cast::<__m256i>())
        };
        let v3 = unsafe {
            _mm256_loadu_si256(haystack.as_ptr().add(index + 3 * LANES).cast::<__m256i>())
        };
        let c0 = _mm256_cmpeq_epi32(v0, broadcast);
        let c1 = _mm256_cmpeq_epi32(v1, broadcast);
        let c2 = _mm256_cmpeq_epi32(v2, broadcast);
        let c3 = _mm256_cmpeq_epi32(v3, broadcast);
        // _mm256_or_si256 isn't strictly needed: testz accepts two
        // operands and returns nonzero iff their AND is zero. We
        // can fold pairs without an extra OR by passing distinct
        // operands, but readability is better with explicit OR.
        use core::arch::x86_64::_mm256_or_si256;
        let or01 = _mm256_or_si256(c0, c1);
        let or23 = _mm256_or_si256(c2, c3);
        let any = _mm256_or_si256(or01, or23);
        if _mm256_testz_si256(any, any) == 0 {
            return true;
        }
        index += unroll_lanes;
    }

    // Single-vector loop for the leftover lanes after the
    // unrolled block.
    while index + LANES <= haystack.len() {
        // SAFETY: index + LANES <= haystack.len() bounds the load.
        let v = unsafe { _mm256_loadu_si256(haystack.as_ptr().add(index).cast::<__m256i>()) };
        let cmp = _mm256_cmpeq_epi32(v, broadcast);
        if _mm256_testz_si256(cmp, cmp) == 0 {
            return true;
        }
        index += LANES;
    }

    scalar::contains_u32(&haystack[index..], needle)
}

/// AVX2 batched membership.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
///
/// # Panics
///
/// Panics if `needles.len() != out.len()`. Available only with
/// `feature = "userspace"`; kernel-safe callers must use
/// [`contains_u32_batch_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    assert_eq!(needles.len(), out.len());
    // SAFETY: AVX2 supplied by target_feature; precondition checked above.
    unsafe { contains_u32_batch_unchecked(haystack, needles, out) }
}

/// Unchecked AVX2 batched membership.
///
/// # Safety
///
/// Caller must ensure the current CPU supports AVX2 and
/// `needles.len() == out.len()`.
#[target_feature(enable = "avx2")]
pub unsafe fn contains_u32_batch_unchecked(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    for (needle, slot) in needles.iter().zip(out.iter_mut()) {
        // SAFETY: target_feature(enable = "avx2") on this fn
        // forwards the AVX2 precondition to the inner kernel.
        *slot = unsafe { contains_u32(haystack, *needle) };
    }
}

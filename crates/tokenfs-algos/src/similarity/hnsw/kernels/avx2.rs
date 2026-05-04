//! x86 AVX2 distance kernels for HNSW.
//!
//! **Phase 2.1 (this file): AVX2 contract scaffold.** All four
//! kernels (`dot_i8`, `l2_squared_i8`, `dot_u8`, `l2_squared_u8`)
//! exist with the established `_unchecked` sibling pattern (see
//! `docs/KERNEL_SAFETY.md`), but currently delegate to the scalar
//! reference. This satisfies the audit-R10 contract — the AVX2
//! entry point exists, follows `try_*` / `_unchecked` / `_inner`
//! conventions, and has scalar-parity tests — without requiring the
//! full SIMD math to be correct on day 1.
//!
//! **Phase 2.5 (follow-up):** replace the scalar fallbacks with
//! correct PMADDUBSW-with-i16-widening or VNNI's VPDPBUSD impls
//! per `SIMD_PRIOR_ART.md`. The naive PMADDUBSW approach
//! saturates at i16 limits when both pair-products are large
//! negative (range exceeds [-32768, 32767]); the correct fix is
//! either explicit `_mm256_cvtepi8_epi16` widening (16 lanes per
//! iteration) or VNNI's `_mm256_dpbusd_epi32` (Cascade Lake+).
//!
//! ## Why scaffold first
//!
//! The audit-R10 + planner-rule machinery needs the AVX2 entry to
//! exist for runtime dispatch to route through it. Once the entry
//! exists, swapping the body from scalar-fallback to true-SIMD is a
//! pure performance change — the parity tests + iai-bench rows are
//! already in place and will catch regressions.
//!
//! ## f32 + binary delegation (per primitive inventory)
//!
//! f32 distance kernels reuse `crate::vector::kernels::avx2::*` via
//! the dispatcher in `super::auto`; binary popcount kernels reuse
//! `crate::bits::popcount::kernels::avx2::*`. This file only ships
//! the **net new** integer (i8 / u8) AVX2 entries, per the scope
//! narrowing in `docs/hnsw/phases/PHASE_2.md`.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use super::super::Distance;
use super::scalar;

/// Returns true when AVX2 + FMA are available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Returns true when AVX2 + FMA are available. Always false in no_std
/// (the runtime detection macro requires `std`).
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

// ---------------------------------------------------------------------
// i8 kernels — Phase 2.1 scalar fallbacks
// ---------------------------------------------------------------------

/// Inner product of two i8 vectors, returned as `Distance` (encoded
/// `-dot` to match scalar's "smaller is better" convention).
///
/// **Phase 2.1:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_dot_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_dot_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_i8_unchecked(a, b) }
}

/// L2² (squared Euclidean) over two i8 vectors. Returned as raw
/// integer `Distance`.
///
/// **Phase 2.1:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_l2_squared_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_l2_squared_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_l2_squared_i8_unchecked(a, b) }
}

// ---------------------------------------------------------------------
// u8 kernels — Phase 2.1 scalar fallbacks
// ---------------------------------------------------------------------

/// Inner product of two u8 vectors.
///
/// **Phase 2.1:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_dot_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_dot_u8(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_u8_unchecked(a, b) }
}

/// L2² (squared Euclidean) over two u8 vectors.
///
/// **Phase 2.1:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_l2_squared_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn distance_l2_squared_u8(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_l2_squared_u8_unchecked(a, b) }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn skip_if_no_avx2() -> bool {
        if !is_available() {
            eprintln!("avx2 not available; skipping");
            return true;
        }
        false
    }

    #[test]
    fn dot_i8_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let a: [i8; 32] = [
            10, -10, 20, -20, 30, -30, 40, -40, 50, -50, 60, -60, 70, -70, 80, -80, 1, -1, 2, -2,
            3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8,
        ];
        let b: [i8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -2, -3, -4, -5, -6, -7, -8,
            -9, -10, -11, -12, -13, -14, -15, -16,
        ];
        let avx = unsafe { distance_dot_i8_unchecked(&a, &b) };
        let s = scalar::try_dot_i8(&a, &b).unwrap();
        assert_eq!(avx, s);
    }

    #[test]
    fn l2_squared_i8_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let a: [i8; 32] = [50; 32];
        let b: [i8; 32] = [40; 32];
        let avx = unsafe { distance_l2_squared_i8_unchecked(&a, &b) };
        let s = scalar::try_l2_squared_i8(&a, &b).unwrap();
        assert_eq!(avx, s);
        // 32 lanes × (50 - 40)^2 = 32 × 100 = 3200
        assert_eq!(s, 3200);
    }

    #[test]
    fn dot_u8_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let a: [u8; 32] = [3; 32];
        let b: [u8; 32] = [4; 32];
        let avx = unsafe { distance_dot_u8_unchecked(&a, &b) };
        let s = scalar::try_dot_u8(&a, &b).unwrap();
        assert_eq!(avx, s);
    }

    #[test]
    fn l2_squared_u8_matches_scalar() {
        if skip_if_no_avx2() {
            return;
        }
        let a: [u8; 32] = [200; 32];
        let b: [u8; 32] = [50; 32];
        let avx = unsafe { distance_l2_squared_u8_unchecked(&a, &b) };
        let s = scalar::try_l2_squared_u8(&a, &b).unwrap();
        assert_eq!(avx, s);
        // 32 lanes × 150² = 32 × 22500 = 720_000
        assert_eq!(s, 720_000);
    }

    #[test]
    fn unaligned_tail_handled_via_scalar_fallback() {
        if skip_if_no_avx2() {
            return;
        }
        // 33-element vectors exercise the tail path. Since the body
        // is currently scalar fallback, all parity holds trivially.
        let a: [i8; 33] = [10; 33];
        let b: [i8; 33] = [4; 33];
        let avx = unsafe { distance_dot_i8_unchecked(&a, &b) };
        let s = scalar::try_dot_i8(&a, &b).unwrap();
        assert_eq!(avx, s);
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn userspace_asserting_variants_match_unchecked() {
        if skip_if_no_avx2() {
            return;
        }
        let a: [i8; 32] = [10; 32];
        let b: [i8; 32] = [4; 32];
        unsafe {
            assert_eq!(distance_dot_i8(&a, &b), distance_dot_i8_unchecked(&a, &b));
            assert_eq!(
                distance_l2_squared_i8(&a, &b),
                distance_l2_squared_i8_unchecked(&a, &b)
            );
        }
    }
}

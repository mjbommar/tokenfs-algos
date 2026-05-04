//! AArch64 NEON distance kernels for HNSW.
//!
//! **Phase 2.2 (this file): NEON contract scaffold.** Same shape as
//! [`super::avx2`] — all four kernels exist with the established
//! `_unchecked` sibling pattern, currently delegating to the scalar
//! reference. Phase 2.5 will replace the bodies with correct NEON
//! impls (per `SIMD_PRIOR_ART.md`).
//!
//! ## NEON specifics for HNSW i8/u8
//!
//! AArch64 NEON's natural register width is 128 bits (16×i8 lanes).
//! Two relevant primitives for Phase 2.5:
//!
//! - **VMULL_S8** (`vmull_s8`) — i8×i8 → i16 widening multiply, lane-wise.
//!   Half-vector input; produces 8 i16 lanes. No saturation issue
//!   (each product fits in i16).
//! - **VMLAL_S8** (`vmlal_s8`) — multiply-accumulate variant.
//! - **SDOT** (FEAT_DotProd, A65+) — `vdotq_s32` 4-byte i8 dot per
//!   lane with i32 accumulation. Single-cycle on supporting cores.
//!
//! For the Phase 2.5 follow-up, default to VMULL/VMLAL pairs (works
//! everywhere AArch64 + NEON), promoting to SDOT when
//! `is_aarch64_feature_detected!("dotprod")` returns true.
//!
//! ## f32 + binary delegation (per primitive inventory)
//!
//! f32 distance kernels reuse `crate::vector::kernels::neon::*`;
//! binary popcount reuses `crate::bits::popcount::kernels::neon::*`.
//! This file ships only the i8/u8 entries.

#![cfg(target_arch = "aarch64")]

use super::super::Distance;
use super::scalar;

/// Returns true when NEON is available. NEON is mandatory on AArch64,
/// so this is always true (matches `vector::kernels::neon`).
#[must_use]
pub const fn is_available() -> bool {
    true
}

// ---------------------------------------------------------------------
// i8 kernels — Phase 2.2 scalar fallbacks
// ---------------------------------------------------------------------

/// Inner product of two i8 vectors, returned as `Distance` (encoded
/// `-dot` to match scalar's "smaller is better" convention).
///
/// **Phase 2.2:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn distance_dot_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn distance_dot_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_i8_unchecked(a, b) }
}

/// L2² (squared Euclidean) over two i8 vectors.
///
/// **Phase 2.2:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn distance_l2_squared_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn distance_l2_squared_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_l2_squared_i8_unchecked(a, b) }
}

// ---------------------------------------------------------------------
// u8 kernels — Phase 2.2 scalar fallbacks
// ---------------------------------------------------------------------

/// Inner product of two u8 vectors.
///
/// **Phase 2.2:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn distance_dot_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn distance_dot_u8(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_u8_unchecked(a, b) }
}

/// L2² (squared Euclidean) over two u8 vectors.
///
/// **Phase 2.2:** scalar fallback. See module-level docs.
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn distance_l2_squared_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "neon")]
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

    #[test]
    fn dot_i8_matches_scalar() {
        let a: [i8; 32] = [
            10, -10, 20, -20, 30, -30, 40, -40, 50, -50, 60, -60, 70, -70, 80, -80, 1, -1, 2, -2,
            3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8,
        ];
        let b: [i8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, -1, -2, -3, -4, -5, -6, -7, -8,
            -9, -10, -11, -12, -13, -14, -15, -16,
        ];
        let neon = unsafe { distance_dot_i8_unchecked(&a, &b) };
        let s = scalar::try_dot_i8(&a, &b).unwrap();
        assert_eq!(neon, s);
    }

    #[test]
    fn l2_squared_i8_matches_scalar() {
        let a: [i8; 32] = [50; 32];
        let b: [i8; 32] = [40; 32];
        let neon = unsafe { distance_l2_squared_i8_unchecked(&a, &b) };
        let s = scalar::try_l2_squared_i8(&a, &b).unwrap();
        assert_eq!(neon, s);
        assert_eq!(s, 3200);
    }

    #[test]
    fn dot_u8_matches_scalar() {
        let a: [u8; 32] = [3; 32];
        let b: [u8; 32] = [4; 32];
        let neon = unsafe { distance_dot_u8_unchecked(&a, &b) };
        let s = scalar::try_dot_u8(&a, &b).unwrap();
        assert_eq!(neon, s);
    }

    #[test]
    fn l2_squared_u8_matches_scalar() {
        let a: [u8; 32] = [200; 32];
        let b: [u8; 32] = [50; 32];
        let neon = unsafe { distance_l2_squared_u8_unchecked(&a, &b) };
        let s = scalar::try_l2_squared_u8(&a, &b).unwrap();
        assert_eq!(neon, s);
        assert_eq!(s, 720_000);
    }

    #[test]
    fn unaligned_tail_handled() {
        let a: [i8; 33] = [10; 33];
        let b: [i8; 33] = [4; 33];
        let neon = unsafe { distance_dot_i8_unchecked(&a, &b) };
        let s = scalar::try_dot_i8(&a, &b).unwrap();
        assert_eq!(neon, s);
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn userspace_asserting_variants_match_unchecked() {
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

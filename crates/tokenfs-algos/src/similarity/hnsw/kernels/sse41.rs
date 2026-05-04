//! x86 SSE4.1 distance kernel scaffold (HNSW Phase 2.3).
//!
//! Fallback for x86 hosts without AVX2. SSE4.1 has 128-bit registers
//! (16×i8 lanes). Same `_unchecked` sibling pattern as
//! [`super::avx2`] and [`super::neon`].
//!
//! **Phase 2.3 (this file):** scalar fallbacks. Phase 2.5 will land
//! correct SSE4.1 i8/u8 SIMD bodies (PMADDUBSW + PMADDWD with
//! widening; native POPCNT for SSE4.2 hosts via separate detection).
//!
//! Skipped: f32 distance kernels (AVX2 reuses `vector::kernels::avx2`;
//! SSE4.1-only x86 hosts that need f32 distance fall back to scalar
//! through `super::auto`).

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]

use super::super::Distance;
use super::scalar;

/// Returns true when SSE4.1 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("sse4.1")
}

#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// i8 dot product. **Phase 2.3:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn distance_dot_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn distance_dot_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_i8_unchecked(a, b) }
}

/// i8 L2². **Phase 2.3:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn distance_l2_squared_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn distance_l2_squared_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_l2_squared_i8_unchecked(a, b) }
}

/// u8 dot product. **Phase 2.3:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn distance_dot_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn distance_dot_u8(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_u8_unchecked(a, b) }
}

/// u8 L2². **Phase 2.3:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[target_feature(enable = "sse4.1")]
#[must_use]
pub unsafe fn distance_l2_squared_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure SSE4.1 is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "sse4.1")]
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

    fn skip_if_no_sse41() -> bool {
        if !is_available() {
            eprintln!("sse4.1 not available; skipping");
            return true;
        }
        false
    }

    #[test]
    fn dot_i8_matches_scalar() {
        if skip_if_no_sse41() {
            return;
        }
        let a: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8];
        let b: [i8; 16] = [1; 16];
        let r = unsafe { distance_dot_i8_unchecked(&a, &b) };
        assert_eq!(r, scalar::try_dot_i8(&a, &b).unwrap());
    }

    #[test]
    fn l2_squared_u8_matches_scalar() {
        if skip_if_no_sse41() {
            return;
        }
        let a: [u8; 16] = [200; 16];
        let b: [u8; 16] = [50; 16];
        let r = unsafe { distance_l2_squared_u8_unchecked(&a, &b) };
        assert_eq!(r, scalar::try_l2_squared_u8(&a, &b).unwrap());
    }

    #[test]
    fn dot_u8_matches_scalar() {
        if skip_if_no_sse41() {
            return;
        }
        let a: [u8; 16] = [3; 16];
        let b: [u8; 16] = [4; 16];
        let r = unsafe { distance_dot_u8_unchecked(&a, &b) };
        assert_eq!(r, scalar::try_dot_u8(&a, &b).unwrap());
    }

    #[test]
    fn l2_squared_i8_matches_scalar() {
        if skip_if_no_sse41() {
            return;
        }
        let a: [i8; 16] = [50; 16];
        let b: [i8; 16] = [40; 16];
        let r = unsafe { distance_l2_squared_i8_unchecked(&a, &b) };
        assert_eq!(r, scalar::try_l2_squared_i8(&a, &b).unwrap());
    }
}

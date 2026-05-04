//! x86 AVX-512 distance kernel scaffold (HNSW Phase 3.2).
//!
//! AVX-512 intrinsics stabilized in Rust 1.89 (2025-08-07) — no
//! nightly required. Same `_unchecked` sibling pattern as
//! [`super::avx2`] / [`super::neon`] / [`super::sse41`].
//!
//! **Phase 3.2 (this file):** scalar fallbacks. Phase 2.5 (queued)
//! will land correct AVX-512 bodies:
//!
//! - **VPDPBUSD** (AVX-512_VNNI, Cascade Lake+): u8 × i8 → i32 with
//!   horizontal sum-of-products. Replaces the AVX2 PMADDUBSW +
//!   PMADDWD chain with a single 512-bit instruction.
//! - **VPOPCNTQ** (AVX-512_BITALG): native 64-bit popcount per lane;
//!   would replace the AVX2 PSHUFB-based Mula popcount for binary
//!   metrics (when those land).
//! - **VFMADD231PS / FMA**: f32 dot / cosine over 16-lane registers.
//!
//! ## f32 + binary delegation
//!
//! f32 distance kernels reuse `crate::vector::kernels::avx512::*`;
//! binary popcount reuses `crate::bits::popcount::kernels::avx512::*`.
//! This file ships only the i8/u8 entries (the net new cells per
//! the Phase 2/3 scope narrowing).
//!
//! Module gated on `cfg(feature = "avx512")` per the existing crate
//! convention (see `src/lib.rs`'s feature table).

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![cfg(feature = "avx512")]

use super::super::Distance;
use super::scalar;

/// Returns true when AVX-512F is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// i8 dot product. **Phase 3.2:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn distance_dot_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn distance_dot_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_i8_unchecked(a, b) }
}

/// i8 L2². **Phase 3.2:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn distance_l2_squared_i8_unchecked(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_i8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_i8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn distance_l2_squared_i8(a: &[i8], b: &[i8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_l2_squared_i8_unchecked(a, b) }
}

/// u8 dot product. **Phase 3.2:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn distance_dot_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_dot_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_dot_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn distance_dot_u8(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(is_available());
    unsafe { distance_dot_u8_unchecked(a, b) }
}

/// u8 L2². **Phase 3.2:** scalar fallback.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn distance_l2_squared_u8_unchecked(a: &[u8], b: &[u8]) -> Distance {
    debug_assert_eq!(a.len(), b.len());
    scalar::try_l2_squared_u8(a, b).unwrap_or(Distance::MAX)
}

/// Userspace-asserting variant of [`distance_l2_squared_u8_unchecked`].
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx512f")]
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

    fn skip_if_no_avx512() -> bool {
        if !is_available() {
            eprintln!("avx512f not available; skipping");
            return true;
        }
        false
    }

    #[test]
    fn dot_i8_matches_scalar() {
        if skip_if_no_avx512() {
            return;
        }
        let a: [i8; 64] = {
            let mut v = [0_i8; 64];
            for (i, lane) in v.iter_mut().enumerate() {
                *lane = (i as i32 - 32) as i8;
            }
            v
        };
        let b: [i8; 64] = [1; 64];
        let r = unsafe { distance_dot_i8_unchecked(&a, &b) };
        assert_eq!(r, scalar::try_dot_i8(&a, &b).unwrap());
    }

    #[test]
    fn l2_squared_u8_matches_scalar() {
        if skip_if_no_avx512() {
            return;
        }
        let a: [u8; 64] = [200; 64];
        let b: [u8; 64] = [50; 64];
        let r = unsafe { distance_l2_squared_u8_unchecked(&a, &b) };
        assert_eq!(r, scalar::try_l2_squared_u8(&a, &b).unwrap());
    }
}

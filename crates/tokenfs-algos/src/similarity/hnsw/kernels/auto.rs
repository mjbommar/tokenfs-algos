//! Runtime-dispatched distance kernels.
//!
//! Picks the best available backend per host capabilities:
//!
//! ```text
//! x86_64:    AVX2 (preferred) → SSE4.1 → scalar
//! aarch64:   NEON             → scalar
//! other:     scalar
//! ```
//!
//! Selection happens at the *outer* boundary (once per `try_search`
//! call, not per-candidate) so the per-candidate hot path is a direct
//! call into the chosen backend's `_unchecked` kernel — no further
//! feature detection inside the inner loop.
//!
//! # Audit posture
//!
//! - All `auto::*` entries are kernel-safe (`no_std + alloc`).
//! - Routes through `_unchecked` kernels after upfront validation
//!   (length match + CPU feature detection). Per `KERNEL_SAFETY.md`
//!   §"Pattern B: `_unchecked` sibling".
//! - Returns `Result<Distance, HnswKernelError>` from the scalar
//!   reference; SIMD backends return raw `Distance` (no error path
//!   for length-matched inputs since the `_unchecked` contract
//!   guarantees them).

use super::super::Distance;
use super::HnswKernelError;
use super::scalar;

/// Compute i8 dot product. Routes to the fastest available backend.
pub fn distance_dot_i8(a: &[i8], b: &[i8]) -> Result<Distance, HnswKernelError> {
    if a.len() != b.len() {
        return Err(HnswKernelError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "arch-pinned-kernels",
    ))]
    {
        if super::avx2::is_available() {
            // SAFETY: AVX2 detected at runtime; lengths checked above.
            return Ok(unsafe { super::avx2::distance_dot_i8_unchecked(a, b) });
        }
        if super::sse41::is_available() {
            // SAFETY: SSE4.1 detected at runtime; lengths checked above.
            return Ok(unsafe { super::sse41::distance_dot_i8_unchecked(a, b) });
        }
    }
    #[cfg(all(
        target_arch = "aarch64",
        feature = "arch-pinned-kernels",
        feature = "neon",
    ))]
    {
        if super::neon::is_available() {
            // SAFETY: NEON is mandatory on AArch64; lengths checked above.
            return Ok(unsafe { super::neon::distance_dot_i8_unchecked(a, b) });
        }
    }
    scalar::try_dot_i8(a, b)
}

/// Compute i8 L2² distance. Routes to the fastest available backend.
pub fn distance_l2_squared_i8(a: &[i8], b: &[i8]) -> Result<Distance, HnswKernelError> {
    if a.len() != b.len() {
        return Err(HnswKernelError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "arch-pinned-kernels",
    ))]
    {
        if super::avx2::is_available() {
            return Ok(unsafe { super::avx2::distance_l2_squared_i8_unchecked(a, b) });
        }
        if super::sse41::is_available() {
            return Ok(unsafe { super::sse41::distance_l2_squared_i8_unchecked(a, b) });
        }
    }
    #[cfg(all(
        target_arch = "aarch64",
        feature = "arch-pinned-kernels",
        feature = "neon",
    ))]
    {
        if super::neon::is_available() {
            return Ok(unsafe { super::neon::distance_l2_squared_i8_unchecked(a, b) });
        }
    }
    scalar::try_l2_squared_i8(a, b)
}

/// Compute u8 dot product. Routes to the fastest available backend.
pub fn distance_dot_u8(a: &[u8], b: &[u8]) -> Result<Distance, HnswKernelError> {
    if a.len() != b.len() {
        return Err(HnswKernelError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "arch-pinned-kernels",
    ))]
    {
        if super::avx2::is_available() {
            return Ok(unsafe { super::avx2::distance_dot_u8_unchecked(a, b) });
        }
        if super::sse41::is_available() {
            return Ok(unsafe { super::sse41::distance_dot_u8_unchecked(a, b) });
        }
    }
    #[cfg(all(
        target_arch = "aarch64",
        feature = "arch-pinned-kernels",
        feature = "neon",
    ))]
    {
        if super::neon::is_available() {
            return Ok(unsafe { super::neon::distance_dot_u8_unchecked(a, b) });
        }
    }
    scalar::try_dot_u8(a, b)
}

/// Compute u8 L2² distance. Routes to the fastest available backend.
pub fn distance_l2_squared_u8(a: &[u8], b: &[u8]) -> Result<Distance, HnswKernelError> {
    if a.len() != b.len() {
        return Err(HnswKernelError::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        });
    }
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "arch-pinned-kernels",
    ))]
    {
        if super::avx2::is_available() {
            return Ok(unsafe { super::avx2::distance_l2_squared_u8_unchecked(a, b) });
        }
        if super::sse41::is_available() {
            return Ok(unsafe { super::sse41::distance_l2_squared_u8_unchecked(a, b) });
        }
    }
    #[cfg(all(
        target_arch = "aarch64",
        feature = "arch-pinned-kernels",
        feature = "neon",
    ))]
    {
        if super::neon::is_available() {
            return Ok(unsafe { super::neon::distance_l2_squared_u8_unchecked(a, b) });
        }
    }
    scalar::try_l2_squared_u8(a, b)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn auto_dot_i8_matches_scalar() {
        let a: [i8; 16] = [
            10, -10, 20, -20, 30, -30, 40, -40, 5, -5, 6, -6, 7, -7, 8, -8,
        ];
        let b: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8];
        let auto_r = distance_dot_i8(&a, &b).unwrap();
        let scalar_r = scalar::try_dot_i8(&a, &b).unwrap();
        assert_eq!(auto_r, scalar_r);
    }

    #[test]
    fn auto_l2_squared_u8_matches_scalar() {
        let a: [u8; 16] = [200; 16];
        let b: [u8; 16] = [50; 16];
        let auto_r = distance_l2_squared_u8(&a, &b).unwrap();
        let scalar_r = scalar::try_l2_squared_u8(&a, &b).unwrap();
        assert_eq!(auto_r, scalar_r);
    }

    #[test]
    fn auto_length_mismatch_returns_error() {
        let a = [0_i8; 4];
        let b = [0_i8; 8];
        let err = distance_dot_i8(&a, &b).unwrap_err();
        assert_eq!(err, HnswKernelError::LengthMismatch { len_a: 4, len_b: 8 });
    }

    #[test]
    fn auto_dot_u8_matches_scalar() {
        let a: [u8; 32] = [3; 32];
        let b: [u8; 32] = [4; 32];
        let auto_r = distance_dot_u8(&a, &b).unwrap();
        let scalar_r = scalar::try_dot_u8(&a, &b).unwrap();
        assert_eq!(auto_r, scalar_r);
    }

    #[test]
    fn auto_l2_squared_i8_matches_scalar() {
        let a: [i8; 32] = [50; 32];
        let b: [i8; 32] = [40; 32];
        let auto_r = distance_l2_squared_i8(&a, &b).unwrap();
        let scalar_r = scalar::try_l2_squared_i8(&a, &b).unwrap();
        assert_eq!(auto_r, scalar_r);
        assert_eq!(scalar_r, 3200);
    }
}

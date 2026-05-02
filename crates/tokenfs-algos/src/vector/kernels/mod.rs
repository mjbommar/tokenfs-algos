//! Pinned dense-distance kernels for the [`crate::vector`] module.
//!
//! Each backend submodule (`scalar`, `avx2`, `avx512`, `neon`) exposes
//! the same shape of functions so the dispatcher in [`auto`] can pick
//! the best one at runtime without changing caller code. Every backend
//! is bit-exact with [`scalar`] for integer kernels and stays within
//! the documented Higham §3 / Wilkinson tolerance for `f32` kernels
//! (see `docs/v0.2_planning/13_VECTOR.md` § 5).
//!
//! ## Reduction-order convention (`f32` kernels)
//!
//! - `scalar`: strictly left-to-right.
//! - `avx2`: 8-way pairwise tree.
//! - `avx512`: 16-way pairwise tree.
//! - `neon`: 4-way pairwise tree (via `vaddvq_f32`).
//!
//! Cross-backend results differ within `1e-3` of `sum(|a_i*b_i|)` (the
//! Higham bound). Same-backend results are bit-exact across runs.

pub mod scalar;

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2;

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub mod neon;

/// Runtime-dispatched dense-distance kernels.
pub mod auto {
    /// Inner product of two `u32` vectors using the best available kernel.
    ///
    /// AVX-512 is not used for u32 kernels — the AVX2 path already
    /// saturates the integer multiply-accumulate units; widening to
    /// 16-lane VPMULDQ would add complexity without changing the
    /// memory-bandwidth-bound reality at 1024-element vectors.
    #[must_use]
    pub fn dot_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::dot_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked immediately above; lengths match.
                return Some(unsafe { super::avx2::dot_u32(a, b) });
            }
        }
        super::scalar::dot_u32(a, b)
    }

    /// Overflow-checked counterpart of [`dot_u32`].
    ///
    /// Always uses the scalar checked path because the SIMD kernels
    /// share the wrapping semantics of [`super::scalar::dot_u32`]; the
    /// checked variant has no SIMD analog.
    #[must_use]
    pub fn try_dot_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
        super::scalar::try_dot_u32(a, b)
    }

    /// L1 distance of two `u32` vectors using the best available kernel.
    #[must_use]
    pub fn l1_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::l1_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::l1_u32(a, b) });
            }
        }
        super::scalar::l1_u32(a, b)
    }

    /// Squared L2 distance of two `u32` vectors.
    #[must_use]
    pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::l2_squared_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::l2_squared_u32(a, b) });
            }
        }
        super::scalar::l2_squared_u32(a, b)
    }

    /// L2 distance of two `u32` vectors as `f64`.
    #[must_use]
    pub fn l2_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        l2_squared_u32(a, b).map(|s| crate::math::sqrt_f64(s as f64))
    }

    /// Cosine similarity of two `u32` vectors as `f64`.
    #[must_use]
    pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::cosine_similarity_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::cosine_similarity_u32(a, b) });
            }
        }
        super::scalar::cosine_similarity_u32(a, b)
    }

    /// Inner product of two `f32` vectors.
    #[must_use]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(
            feature = "std",
            feature = "avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx512::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx512::dot_f32(a, b) });
            }
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::dot_f32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::dot_f32(a, b) });
            }
        }
        super::scalar::dot_f32(a, b)
    }

    /// Squared L2 distance of two `f32` vectors.
    #[must_use]
    pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(
            feature = "std",
            feature = "avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx512::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx512::l2_squared_f32(a, b) });
            }
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::l2_squared_f32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::l2_squared_f32(a, b) });
            }
        }
        super::scalar::l2_squared_f32(a, b)
    }

    /// Cosine similarity of two `f32` vectors.
    #[must_use]
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(
            feature = "std",
            feature = "avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx512::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx512::cosine_similarity_f32(a, b) });
            }
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::cosine_similarity_f32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::cosine_similarity_f32(a, b) });
            }
        }
        super::scalar::cosine_similarity_f32(a, b)
    }

    /// Hamming distance of two packed `u64` bitvector slices.
    #[must_use]
    pub fn hamming_u64(a: &[u64], b: &[u64]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(
            feature = "std",
            feature = "avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx512::is_popcnt_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx512::hamming_u64(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::hamming_u64(a, b) });
            }
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::hamming_u64(a, b) });
            }
        }
        super::scalar::hamming_u64(a, b)
    }

    /// Jaccard similarity of two packed `u64` bitvector slices.
    #[must_use]
    pub fn jaccard_u64(a: &[u64], b: &[u64]) -> Option<f64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(
            feature = "std",
            feature = "avx512",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx512::is_popcnt_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx512::jaccard_u64(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::jaccard_u64(a, b) });
            }
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::jaccard_u64(a, b) });
            }
        }
        super::scalar::jaccard_u64(a, b)
    }
}

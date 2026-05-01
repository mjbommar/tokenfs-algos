//! Dense distance and similarity primitives.
//!
//! This module is the foundation of the similarity layer described in
//! `docs/SIMILARITY_APPROXIMATION_ROADMAP.md`. The public contract follows
//! the same ladder as histogram and fingerprint:
//!
//! ```text
//! similarity::distance::cosine_u32(a, b)            // public, runtime-dispatched
//! similarity::kernels::scalar::cosine_u32(a, b)     // pinned scalar reference
//! similarity::kernels::avx2::cosine_u32(a, b)       // pinned AVX2 (when feature gate active)
//! similarity::kernels::neon::cosine_u32(a, b)       // pinned NEON (when target is aarch64)
//! ```
//!
//! At this stage only the scalar kernels exist. AVX2 and NEON kernels land in
//! follow-up tasks; the public dispatcher already has the slot wired so adding
//! a backend does not change the API.
//!
//! Existing count-vector distance functions in [`crate::divergence`] remain in
//! place and are re-exported below under [`distance::counts`]; the unified
//! `similarity` namespace is the recommended import path going forward.

#[cfg(test)]
mod tests;

pub mod fuzzy;
pub mod kernels;
#[cfg(feature = "std")]
pub mod lsh;
pub mod minhash;
pub mod simhash;

/// Distance functions for raw vectors and count vectors.
pub mod distance {
    /// Re-exports of the count-vector distance functions in
    /// [`crate::divergence`].
    pub mod counts {
        pub use crate::divergence::{
            cosine_distance_counts_u32, hellinger_distance_counts, jensen_shannon_distance_counts,
            jensen_shannon_distance_counts_u32, kl_divergence_counts, ks_statistic_counts,
            l2_distance_counts_u32, normalized_l2_distance_counts_u32, total_variation_counts,
            triangular_discrimination_counts,
        };
    }

    use super::kernels;

    /// Inner product of two `u32` vectors using the runtime-dispatched kernel.
    #[must_use]
    pub fn dot_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        kernels::auto::dot_u32(a, b)
    }

    /// Manhattan / L1 distance of two `u32` vectors.
    #[must_use]
    pub fn l1_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        kernels::auto::l1_u32(a, b)
    }

    /// Squared L2 distance of two `u32` vectors.
    ///
    /// Avoids the `sqrt` and is the right primitive for ranking. Use
    /// [`l2_u32`] when an actual Euclidean distance is required.
    #[must_use]
    pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        kernels::auto::l2_squared_u32(a, b)
    }

    /// Euclidean / L2 distance of two `u32` vectors as `f64`.
    #[must_use]
    pub fn l2_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        kernels::auto::l2_u32(a, b)
    }

    /// Cosine similarity of two `u32` vectors as `f64` in `[-1, 1]`.
    ///
    /// Returns `None` for length mismatch and `Some(0.0)` when either vector
    /// has zero norm.
    #[must_use]
    pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        kernels::auto::cosine_similarity_u32(a, b)
    }

    /// Cosine distance = `1 - cosine_similarity`.
    #[must_use]
    pub fn cosine_distance_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        kernels::auto::cosine_similarity_u32(a, b).map(|s| 1.0 - s)
    }

    /// Inner product of two `f32` vectors using the runtime-dispatched kernel.
    #[must_use]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        kernels::auto::dot_f32(a, b)
    }

    /// Squared L2 distance of two `f32` vectors.
    #[must_use]
    pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        kernels::auto::l2_squared_f32(a, b)
    }

    /// Euclidean / L2 distance of two `f32` vectors.
    #[must_use]
    pub fn l2_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        kernels::auto::l2_squared_f32(a, b).map(|s| s.sqrt())
    }

    /// Cosine similarity of two `f32` vectors in `[-1, 1]`.
    #[must_use]
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        kernels::auto::cosine_similarity_f32(a, b)
    }

    /// Cosine distance = `1 - cosine_similarity` for `f32`.
    #[must_use]
    pub fn cosine_distance_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        kernels::auto::cosine_similarity_f32(a, b).map(|s| 1.0 - s)
    }
}

/// Nearest-reference helpers for candidate selection.
pub mod nearest {
    /// Re-exports of the count-vector nearest-reference functions in
    /// [`crate::distribution`].
    pub use crate::distribution::{nearest_byte_distribution, nearest_reference};
}

/// Fixed-size dense distance kernels for the canonical bin counts.
///
/// These wrap the slice-based [`distance`] functions but take `&[T; N]`
/// arrays so the compiler can fully unroll the inner loop and elide bounds
/// checks. The roadmap calls out 256 (byte histogram), 1024 (compact n-gram
/// sketch), and 4096 (richer n-gram sketch) as the load-bearing widths.
///
/// All three sizes go through the same runtime-dispatched kernel. The
/// performance win comes from compile-time-known length and contiguous,
/// cache-friendly `&[T; N]` access — not from a separate kernel per size.
pub mod fixed {
    use super::distance;

    /// Inner product of two 256-bin `u32` count vectors.
    #[must_use]
    pub fn dot_u32_256(a: &[u32; 256], b: &[u32; 256]) -> u64 {
        // unwrap is safe: lengths are equal by construction.
        distance::dot_u32(a, b).unwrap_or(0)
    }

    /// L1 distance of two 256-bin `u32` count vectors.
    #[must_use]
    pub fn l1_u32_256(a: &[u32; 256], b: &[u32; 256]) -> u64 {
        distance::l1_u32(a, b).unwrap_or(0)
    }

    /// Squared L2 distance of two 256-bin `u32` count vectors.
    #[must_use]
    pub fn l2_squared_u32_256(a: &[u32; 256], b: &[u32; 256]) -> u64 {
        distance::l2_squared_u32(a, b).unwrap_or(0)
    }

    /// Cosine similarity of two 256-bin `u32` count vectors.
    #[must_use]
    pub fn cosine_similarity_u32_256(a: &[u32; 256], b: &[u32; 256]) -> f64 {
        distance::cosine_similarity_u32(a, b).unwrap_or(0.0)
    }

    /// Inner product of two 1024-bin `u32` sketch vectors.
    #[must_use]
    pub fn dot_u32_1024(a: &[u32; 1024], b: &[u32; 1024]) -> u64 {
        distance::dot_u32(a, b).unwrap_or(0)
    }

    /// L1 distance of two 1024-bin `u32` sketch vectors.
    #[must_use]
    pub fn l1_u32_1024(a: &[u32; 1024], b: &[u32; 1024]) -> u64 {
        distance::l1_u32(a, b).unwrap_or(0)
    }

    /// Squared L2 distance of two 1024-bin `u32` sketch vectors.
    #[must_use]
    pub fn l2_squared_u32_1024(a: &[u32; 1024], b: &[u32; 1024]) -> u64 {
        distance::l2_squared_u32(a, b).unwrap_or(0)
    }

    /// Cosine similarity of two 1024-bin `u32` sketch vectors.
    #[must_use]
    pub fn cosine_similarity_u32_1024(a: &[u32; 1024], b: &[u32; 1024]) -> f64 {
        distance::cosine_similarity_u32(a, b).unwrap_or(0.0)
    }

    /// Inner product of two 4096-bin `u32` sketch vectors.
    #[must_use]
    pub fn dot_u32_4096(a: &[u32; 4096], b: &[u32; 4096]) -> u64 {
        distance::dot_u32(a, b).unwrap_or(0)
    }

    /// L1 distance of two 4096-bin `u32` sketch vectors.
    #[must_use]
    pub fn l1_u32_4096(a: &[u32; 4096], b: &[u32; 4096]) -> u64 {
        distance::l1_u32(a, b).unwrap_or(0)
    }

    /// Squared L2 distance of two 4096-bin `u32` sketch vectors.
    #[must_use]
    pub fn l2_squared_u32_4096(a: &[u32; 4096], b: &[u32; 4096]) -> u64 {
        distance::l2_squared_u32(a, b).unwrap_or(0)
    }

    /// Cosine similarity of two 4096-bin `u32` sketch vectors.
    #[must_use]
    pub fn cosine_similarity_u32_4096(a: &[u32; 4096], b: &[u32; 4096]) -> f64 {
        distance::cosine_similarity_u32(a, b).unwrap_or(0.0)
    }

    /// Generic const-generic version. Pick this for non-canonical sizes.
    /// Takes `&[u32; N]` so the compiler can specialize per call site.
    #[must_use]
    pub fn dot_u32_n<const N: usize>(a: &[u32; N], b: &[u32; N]) -> u64 {
        distance::dot_u32(a, b).unwrap_or(0)
    }

    /// Generic const-generic L2-squared. See [`dot_u32_n`].
    #[must_use]
    pub fn l2_squared_u32_n<const N: usize>(a: &[u32; N], b: &[u32; N]) -> u64 {
        distance::l2_squared_u32(a, b).unwrap_or(0)
    }

    /// Generic const-generic cosine similarity. See [`dot_u32_n`].
    #[must_use]
    pub fn cosine_similarity_u32_n<const N: usize>(a: &[u32; N], b: &[u32; N]) -> f64 {
        distance::cosine_similarity_u32(a, b).unwrap_or(0.0)
    }
}

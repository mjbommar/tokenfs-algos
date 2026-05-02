//! Single-pair distance APIs.
//!
//! The functions here are thin runtime-dispatched shims over
//! [`super::kernels::auto`]. They are the canonical "two vectors in,
//! one number out" entry points for callers that don't want to think
//! about backend selection.
//!
//! See the [crate-level module documentation](super) for the full API
//! surface and the `docs/v0.2_planning/13_VECTOR.md` spec for the
//! algorithm and tolerance contracts.

use super::kernels;

/// Inner product of two `f32` vectors using the runtime-dispatched kernel.
///
/// Returns `None` on length mismatch. SIMD reduction is bounded by the
/// Higham §3 / Wilkinson model: the result is within `n * eps *
/// sum(|a_i*b_i|)` of the exact dot product. Cross-backend agreement is
/// asserted at `1e-3` of `sum(|a*b|)` in
/// `tests/avx2_parity.rs` / `tests/neon_parity.rs`.
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    kernels::auto::dot_f32(a, b)
}

/// Inner product of two `u32` vectors using the runtime-dispatched
/// kernel.
///
/// Returns `0` on length mismatch (matches the
/// `Option<u64>`-flattening convention of the corresponding fixed-size
/// kernels). For length validation prefer [`try_dot_u32`].
///
/// # Overflow regime
///
/// Wraps on overflow — see [`super::kernels::scalar::dot_u32`] for the
/// safe regime and [`try_dot_u32`] for an overflow-checked variant.
#[must_use]
pub fn dot_u32(a: &[u32], b: &[u32]) -> u64 {
    kernels::auto::dot_u32(a, b).unwrap_or(0)
}

/// Overflow-checked counterpart of [`dot_u32`].
///
/// Returns `None` on length mismatch, `Some(None)` on accumulator
/// overflow, and `Some(Some(sum))` on success.
#[must_use]
pub fn try_dot_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
    kernels::auto::try_dot_u32(a, b)
}

/// Squared L2 distance of two `f32` vectors.
#[must_use]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    kernels::auto::l2_squared_f32(a, b)
}

/// Squared L2 distance of two `u32` vectors.
///
/// Returns `0` on length mismatch.
#[must_use]
pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64 {
    kernels::auto::l2_squared_u32(a, b).unwrap_or(0)
}

/// Cosine similarity of two `f32` vectors.
///
/// Returns `None` on length mismatch and `Some(0.0)` when either vector
/// has zero norm.
#[must_use]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    kernels::auto::cosine_similarity_f32(a, b)
}

/// Cosine similarity of two `u32` vectors as `f64` in `[-1, 1]` (in
/// practice `[0, 1]` for non-negative count vectors).
///
/// Returns `None` on length mismatch and `Some(0.0)` when either vector
/// has zero norm.
#[must_use]
pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
    kernels::auto::cosine_similarity_u32(a, b)
}

/// Hamming distance of two packed `u64` bitvector slices.
///
/// Returns `None` on length mismatch.
///
/// Reuses the [`crate::bits::popcount`] kernels for the per-word
/// popcount; the wrapper XORs the words pairwise. AVX-512 VPOPCNTQ is
/// the single largest SIMD win here (~10x over AVX2 software popcount)
/// when available.
#[must_use]
pub fn hamming_u64(a: &[u64], b: &[u64]) -> Option<u64> {
    kernels::auto::hamming_u64(a, b)
}

/// Jaccard similarity of two packed `u64` bitvector slices.
///
/// Returns `None` on length mismatch. Returns `Some(1.0)` when both
/// inputs are all-zeros (the convention for "two empty sets are
/// identical"). The result is in `[0.0, 1.0]`.
#[must_use]
pub fn jaccard_u64(a: &[u64], b: &[u64]) -> Option<f64> {
    kernels::auto::jaccard_u64(a, b)
}

//! Dense vector distance kernels.
//!
//! Five distance metrics × two query shapes × four backends. The
//! single source of truth for vector inner-loop primitives across the
//! crate; the [`crate::similarity`] module re-exports these kernels
//! under deprecated aliases for backward compatibility (see
//! `docs/v0.2_planning/13_VECTOR.md` § 1).
//!
//! ## API surface
//!
//! Single-pair APIs live in [`distance`]; many-vs-one batched APIs
//! live in [`batch`]. Both are re-exported at the module root for
//! convenience.
//!
//! Pinned per-backend kernels live in [`kernels`]:
//!
//! - [`kernels::scalar`] — portable reference path (always present).
//! - `kernels::avx2` — x86 AVX2 (`feature = "avx2"`).
//! - `kernels::avx512` — x86 AVX-512F + VPOPCNTQ (`feature = "avx512"`).
//! - `kernels::neon` — AArch64 NEON (`feature = "neon"`).
//!
//! ## Reduction-order numerics (`f32` kernels)
//!
//! The pairwise-tree reduction order is part of the public contract:
//!
//! - `scalar`: strictly left-to-right.
//! - `avx2`: 8-way pairwise tree.
//! - `avx512`: 16-way pairwise tree.
//! - `neon`: 4-way pairwise tree.
//!
//! Cross-backend results differ within the Higham §3 / Wilkinson bound
//! `1e-3 * sum(|a_i*b_i|)`. Same-backend results are bit-exact across
//! runs.
//!
//! ## Allocation policy
//!
//! - All single-pair APIs are stateless and operate on borrowed
//!   slices.
//! - The `_one_to_many` batched APIs take a caller-provided output
//!   buffer (`out: &mut [_]`).
//! - No internal allocation. No rayon. No `std`-only dependencies in
//!   any hot kernel. **Kernel-safe across the board.**
//!
//! See `docs/v0.2_planning/13_VECTOR.md` for the full spec.

pub mod batch;
pub mod distance;
pub mod kernels;

pub use batch::{
    cosine_similarity_f32_one_to_many, dot_f32_one_to_many, hamming_u64_one_to_many,
    jaccard_u64_one_to_many, l2_squared_f32_one_to_many,
};
pub use distance::{
    cosine_similarity_f32, cosine_similarity_u32, dot_f32, dot_u32, hamming_u64, jaccard_u64,
    l2_squared_f32, l2_squared_u32, try_dot_u32,
};

#[cfg(test)]
mod tests;

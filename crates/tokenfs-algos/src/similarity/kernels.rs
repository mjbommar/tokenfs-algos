//! Deprecated re-exports of the dense-distance kernels.
//!
//! The kernels formerly defined here have moved to [`crate::vector::kernels`]
//! per the v0.2 module split (`docs/v0.2_planning/13_VECTOR.md` § 1). This
//! module forwards every public symbol to the new location and is marked
//! `#[deprecated]` for callers — new code should import from `vector::*`
//! directly.
//!
//! The SVE backend (AArch64 only, gated on the `sve` cargo feature) remains
//! here because the `vector` module spec only enumerates `scalar / avx2 /
//! avx512 / neon`. SVE will be added there when the SVE inner-loop work
//! lands as a separate v0.2 sprint.

/// Runtime-dispatched dense-distance kernels.
///
/// Re-export of [`crate::vector::kernels::auto`] for backward
/// compatibility. New code should call [`crate::vector::kernels::auto`]
/// or the higher-level helpers in [`crate::vector::distance`] directly.
pub mod auto {
    /// Inner product of two `u32` vectors using the best available kernel.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::dot_u32` (or `vector::kernels::auto::dot_u32`) directly"
    )]
    #[must_use]
    pub fn dot_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        crate::vector::kernels::auto::dot_u32(a, b)
    }

    /// L1 distance of two `u32` vectors using the best available kernel.
    #[deprecated(since = "0.2.0", note = "use `vector::kernels::auto::l1_u32` directly")]
    #[must_use]
    pub fn l1_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        crate::vector::kernels::auto::l1_u32(a, b)
    }

    /// Squared L2 distance of two `u32` vectors.
    #[deprecated(since = "0.2.0", note = "use `vector::l2_squared_u32` directly")]
    #[must_use]
    pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        crate::vector::kernels::auto::l2_squared_u32(a, b)
    }

    /// L2 distance of two `u32` vectors as `f64`.
    #[deprecated(since = "0.2.0", note = "use `vector::kernels::auto::l2_u32` directly")]
    #[must_use]
    pub fn l2_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        crate::vector::kernels::auto::l2_u32(a, b)
    }

    /// Cosine similarity of two `u32` vectors as `f64`.
    #[deprecated(since = "0.2.0", note = "use `vector::cosine_similarity_u32` directly")]
    #[must_use]
    pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        crate::vector::kernels::auto::cosine_similarity_u32(a, b)
    }

    /// Inner product of two `f32` vectors.
    #[deprecated(since = "0.2.0", note = "use `vector::dot_f32` directly")]
    #[must_use]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        crate::vector::kernels::auto::dot_f32(a, b)
    }

    /// Squared L2 distance of two `f32` vectors.
    #[deprecated(since = "0.2.0", note = "use `vector::l2_squared_f32` directly")]
    #[must_use]
    pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        crate::vector::kernels::auto::l2_squared_f32(a, b)
    }

    /// Cosine similarity of two `f32` vectors.
    #[deprecated(since = "0.2.0", note = "use `vector::cosine_similarity_f32` directly")]
    #[must_use]
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        crate::vector::kernels::auto::cosine_similarity_f32(a, b)
    }
}

/// Portable scalar dense-distance kernels.
///
/// Deprecated re-exports of [`crate::vector::kernels::scalar`].
#[cfg(feature = "arch-pinned-kernels")]
pub mod scalar;
#[cfg(not(feature = "arch-pinned-kernels"))]
#[allow(dead_code, unreachable_pub)]
pub(crate) mod scalar;

/// AVX2 dense-distance kernels.
///
/// Deprecated re-exports of [`crate::vector::kernels::avx2`].
#[cfg(all(
    feature = "arch-pinned-kernels",
    feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub mod avx2;
#[cfg(all(
    not(feature = "arch-pinned-kernels"),
    feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
#[allow(dead_code, unreachable_pub)]
pub(crate) mod avx2;

/// AArch64 NEON dense-distance kernels.
///
/// Deprecated re-exports of [`crate::vector::kernels::neon`].
#[cfg(all(
    feature = "arch-pinned-kernels",
    feature = "neon",
    target_arch = "aarch64"
))]
pub mod neon;
#[cfg(all(
    not(feature = "arch-pinned-kernels"),
    feature = "neon",
    target_arch = "aarch64"
))]
#[allow(dead_code, unreachable_pub)]
pub(crate) mod neon;

/// AArch64 SVE / SVE2 dense-distance kernels.
///
/// # Why SVE here
///
/// SVE's predicated FMA (`svmla_f32_m`) plus the active-lane predicate
/// (`svwhilelt_b32`) is the textbook win over fixed-128-bit NEON for
/// floating-point reductions: one vector-length-agnostic loop
/// processes the whole input, the tail iteration zeros inactive lanes
/// (so the FMA accumulator is unchanged for those positions), and a
/// final `svaddv_f32` reduces the partial sums to a scalar.
///
/// On Neoverse-N2 (128-bit SVE), the per-iteration work is identical
/// to the NEON kernel; the savings come from eliminating the scalar
/// tail loop. On Neoverse-V1 (256-bit) or A64FX (512-bit) the same
/// source compiles to a 2× / 4× wider vector with no source change —
/// that is the VLA promise.
///
/// Note: this module compiles under the `sve` cargo feature, not
/// `sve2`. The intrinsics used here (`svmla_f32_m`, `svwhilelt_b32`,
/// `svaddv_f32`) all live in the SVE base ISA. SVE2 is a strict
/// superset; everything compiled here also runs on SVE2 hardware.
///
/// # Numeric tolerance
///
/// SVE reductions use a different summation order than the scalar
/// reference (lane-parallel partial sums + final `svaddv`), so f32
/// results can differ by a few ULP on long vectors. The cross-parity
/// proptest in `tests/sve_parity.rs` asserts the same `1e-2` (dot)
/// and `5e-4` (l2) tolerances that `tests/neon_parity.rs` uses.
#[cfg(all(
    feature = "arch-pinned-kernels",
    feature = "sve",
    target_arch = "aarch64"
))]
pub mod sve;
#[cfg(all(
    not(feature = "arch-pinned-kernels"),
    feature = "sve",
    target_arch = "aarch64"
))]
#[allow(dead_code, unreachable_pub)]
pub(crate) mod sve;

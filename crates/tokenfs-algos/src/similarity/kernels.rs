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
pub mod scalar {
    /// Inner product of two `u32` vectors. Returns `None` on length mismatch.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::dot_u32` directly"
    )]
    #[must_use]
    pub fn dot_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        crate::vector::kernels::scalar::dot_u32(a, b)
    }

    /// Overflow-checked counterpart of [`dot_u32`].
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::try_dot_u32` directly"
    )]
    #[must_use]
    pub fn try_dot_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
        crate::vector::kernels::scalar::try_dot_u32(a, b)
    }

    /// Manhattan / L1 distance of two `u32` vectors.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::l1_u32` directly"
    )]
    #[must_use]
    pub fn l1_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        crate::vector::kernels::scalar::l1_u32(a, b)
    }

    /// Overflow-checked counterpart of [`l1_u32`].
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::try_l1_u32` directly"
    )]
    #[must_use]
    pub fn try_l1_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
        crate::vector::kernels::scalar::try_l1_u32(a, b)
    }

    /// Squared L2 distance of two `u32` vectors.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::l2_squared_u32` directly"
    )]
    #[must_use]
    pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        crate::vector::kernels::scalar::l2_squared_u32(a, b)
    }

    /// Overflow-checked counterpart of [`l2_squared_u32`].
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::try_l2_squared_u32` directly"
    )]
    #[must_use]
    pub fn try_l2_squared_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
        crate::vector::kernels::scalar::try_l2_squared_u32(a, b)
    }

    /// Cosine similarity of two `u32` vectors as `f64`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::cosine_similarity_u32` directly"
    )]
    #[must_use]
    pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        crate::vector::kernels::scalar::cosine_similarity_u32(a, b)
    }

    /// Inner product of two `f32` vectors.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::dot_f32` directly"
    )]
    #[must_use]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        crate::vector::kernels::scalar::dot_f32(a, b)
    }

    /// Squared L2 distance of two `f32` vectors.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::l2_squared_f32` directly"
    )]
    #[must_use]
    pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        crate::vector::kernels::scalar::l2_squared_f32(a, b)
    }

    /// Cosine similarity of two `f32` vectors.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::scalar::cosine_similarity_f32` directly"
    )]
    #[must_use]
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        crate::vector::kernels::scalar::cosine_similarity_f32(a, b)
    }
}

/// AVX2 dense-distance kernels.
///
/// Deprecated re-exports of [`crate::vector::kernels::avx2`].
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2 {
    /// Returns true when AVX2 is available at runtime.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::avx2::is_available` directly"
    )]
    #[must_use]
    pub fn is_available() -> bool {
        crate::vector::kernels::avx2::is_available()
    }

    /// Inner product of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::avx2::dot_u32` directly"
    )]
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn dot_u32(a: &[u32], b: &[u32]) -> u64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::avx2::dot_u32(a, b) }
    }

    /// L1 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[deprecated(since = "0.2.0", note = "use `vector::kernels::avx2::l1_u32` directly")]
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn l1_u32(a: &[u32], b: &[u32]) -> u64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::avx2::l1_u32(a, b) }
    }

    /// Squared L2 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::avx2::l2_squared_u32` directly"
    )]
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::avx2::l2_squared_u32(a, b) }
    }

    /// Cosine similarity of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::avx2::cosine_similarity_u32` directly"
    )]
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> f64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::avx2::cosine_similarity_u32(a, b) }
    }

    /// Inner product of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::avx2::dot_f32` directly"
    )]
    #[target_feature(enable = "avx2,fma")]
    #[must_use]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::avx2::dot_f32(a, b) }
    }

    /// Squared L2 distance of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::avx2::l2_squared_f32` directly"
    )]
    #[target_feature(enable = "avx2,fma")]
    #[must_use]
    pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::avx2::l2_squared_f32(a, b) }
    }

    /// Cosine similarity of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::avx2::cosine_similarity_f32` directly"
    )]
    #[target_feature(enable = "avx2,fma")]
    #[must_use]
    pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::avx2::cosine_similarity_f32(a, b) }
    }
}

/// AArch64 NEON dense-distance kernels.
///
/// Deprecated re-exports of [`crate::vector::kernels::neon`].
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub mod neon {
    /// Returns true when NEON is available at runtime.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::neon::is_available` directly"
    )]
    #[must_use]
    pub const fn is_available() -> bool {
        crate::vector::kernels::neon::is_available()
    }

    /// Inner product of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::neon::dot_u32` directly"
    )]
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn dot_u32(a: &[u32], b: &[u32]) -> u64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::neon::dot_u32(a, b) }
    }

    /// L1 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[deprecated(since = "0.2.0", note = "use `vector::kernels::neon::l1_u32` directly")]
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn l1_u32(a: &[u32], b: &[u32]) -> u64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::neon::l1_u32(a, b) }
    }

    /// Squared L2 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::neon::l2_squared_u32` directly"
    )]
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::neon::l2_squared_u32(a, b) }
    }

    /// Cosine similarity of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::neon::cosine_similarity_u32` directly"
    )]
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> f64 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::neon::cosine_similarity_u32(a, b) }
    }

    /// Inner product of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::neon::dot_f32` directly"
    )]
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::neon::dot_f32(a, b) }
    }

    /// Squared L2 distance of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::neon::l2_squared_f32` directly"
    )]
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::neon::l2_squared_f32(a, b) }
    }

    /// Cosine similarity of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[deprecated(
        since = "0.2.0",
        note = "use `vector::kernels::neon::cosine_similarity_f32` directly"
    )]
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        // SAFETY: forwarded; same precondition.
        unsafe { crate::vector::kernels::neon::cosine_similarity_f32(a, b) }
    }
}

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
#[cfg(all(feature = "sve", target_arch = "aarch64"))]
pub mod sve {
    use core::arch::aarch64::{
        svaddv_f32, svcntw, svdup_n_f32, svld1_f32, svmla_f32_m, svptest_any, svptrue_b32,
        svsub_f32_x, svwhilelt_b32_u64,
    };

    /// Returns true when SVE is available at runtime.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::arch::is_aarch64_feature_detected!("sve")
    }

    /// Returns true when SVE is available at runtime.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Inner product of two `f32` vectors using SVE.
    ///
    /// # Safety
    ///
    /// Caller must ensure SVE is available and `a.len() == b.len()`.
    #[target_feature(enable = "sve")]
    #[must_use]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len() as u64;
        let mut i: u64 = 0;
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let all_lanes = svptrue_b32();
        let step = svcntw();
        let mut acc = svdup_n_f32(0.0);

        loop {
            let pg = svwhilelt_b32_u64(i, n);
            if !svptest_any(all_lanes, pg) {
                break;
            }
            // SAFETY: `pg` zeros inactive lanes; the predicated load
            // does not fault past valid memory.
            let va = unsafe { svld1_f32(pg, pa.add(i as usize)) };
            let vb = unsafe { svld1_f32(pg, pb.add(i as usize)) };
            // svmla_f32_m: acc = acc + va * vb under predicate `pg`,
            // with inactive lanes preserved from `acc`. The merge
            // ("_m") variant is required here — the don't-care ("_x")
            // variant leaves inactive lanes implementation-defined,
            // and the subsequent svaddv_f32 sums every lane, so any
            // garbage in inactive lanes would poison the reduction on
            // tail iterations. Real ARM cores are allowed to leave
            // inactive lanes as previous register state; QEMU happens
            // to zero them, masking the bug locally.
            acc = svmla_f32_m(pg, acc, va, vb);
            i += step;
        }

        svaddv_f32(all_lanes, acc)
    }

    /// Squared L2 distance of two `f32` vectors using SVE.
    ///
    /// # Safety
    ///
    /// Caller must ensure SVE is available and `a.len() == b.len()`.
    #[target_feature(enable = "sve")]
    #[must_use]
    pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let n = a.len() as u64;
        let mut i: u64 = 0;
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        let all_lanes = svptrue_b32();
        let step = svcntw();
        let mut acc = svdup_n_f32(0.0);

        loop {
            let pg = svwhilelt_b32_u64(i, n);
            if !svptest_any(all_lanes, pg) {
                break;
            }
            // SAFETY: as `dot_f32`.
            let va = unsafe { svld1_f32(pg, pa.add(i as usize)) };
            let vb = unsafe { svld1_f32(pg, pb.add(i as usize)) };
            let d = svsub_f32_x(pg, va, vb);
            // svmla_f32_m: see dot_f32 — _m preserves `acc` in inactive
            // lanes so the trailing svaddv_f32 reduction stays correct.
            // Garbage in inactive lanes of `d` is harmless because the
            // _m variant ignores the new computation entirely there.
            acc = svmla_f32_m(pg, acc, d, d);
            i += step;
        }

        svaddv_f32(all_lanes, acc)
    }

    /// Cosine similarity of two `f32` vectors using SVE.
    ///
    /// # Safety
    ///
    /// Caller must ensure SVE is available and `a.len() == b.len()`.
    #[target_feature(enable = "sve")]
    #[must_use]
    pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: SVE enabled, lengths match.
        let dot = unsafe { dot_f32(a, b) };
        let norm_a = unsafe { dot_f32(a, a) };
        let norm_b = unsafe { dot_f32(b, b) };
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / crate::math::sqrt_f32(norm_a * norm_b)
    }
}

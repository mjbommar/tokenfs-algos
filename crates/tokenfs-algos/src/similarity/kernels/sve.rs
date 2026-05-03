use core::arch::aarch64::{
    svaddv_f32, svcntw, svdup_n_f32, svld1_f32, svmla_f32_m, svptest_any, svptrue_b32, svsub_f32_x,
    svwhilelt_b32_u64,
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

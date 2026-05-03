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

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

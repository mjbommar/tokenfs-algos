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

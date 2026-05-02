//! Many-vs-one batched dense-distance APIs.
//!
//! `query` is one vector of length `stride`; `db` is a flat array of `N`
//! database vectors each of length `stride`, contiguous (row-major). The
//! output buffer has length `N`.
//!
//! ## Why batched
//!
//! For K-nearest-neighbor scans the inner loop is "one query × N
//! database vectors". Repeated single-pair calls re-load the query into
//! SIMD registers each iteration — wasteful when the query stays
//! resident across all N comparisons. The batched form keeps the query
//! pinned and sweeps the database past it, beating naive iteration on
//! L1-resident workloads.
//!
//! The implementation here is the "amortize the query once per batch"
//! shape: the dispatcher resolves the kernel once for the whole batch,
//! then calls the per-pair SIMD primitive in a tight loop. This is a
//! 5-10% win over re-dispatching per pair on small `N` and trivial on
//! large `N` where the per-pair work dominates.
//!
//! ## Allocation policy
//!
//! Caller-provided `out: &mut [_]` — never internally allocates. This
//! keeps the API kernel-safe per the contract in
//! `docs/v0.2_planning/13_VECTOR.md` § 9.
//!
//! ## Panics
//!
//! Each function panics if the output buffer length does not equal
//! `db.len() / stride`, or if `db.len()` is not a multiple of `stride`,
//! or if `stride == 0`. These conditions indicate a caller bug, not a
//! recoverable runtime error.

use super::kernels;

/// Many-vs-one inner product over `f32` vectors.
///
/// Writes `out[i] = dot_f32(query, db[i*stride .. (i+1)*stride])` for
/// each `i ∈ 0..out.len()`.
///
/// # Panics
///
/// See module docs.
pub fn dot_f32_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        *slot = kernels::auto::dot_f32(query, row).unwrap_or(0.0);
    }
}

/// Many-vs-one squared L2 distance over `f32` vectors.
///
/// Writes `out[i] = l2_squared_f32(query, db[i*stride .. (i+1)*stride])`
/// for each `i`.
///
/// # Panics
///
/// See module docs.
pub fn l2_squared_f32_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        *slot = kernels::auto::l2_squared_f32(query, row).unwrap_or(0.0);
    }
}

/// Many-vs-one cosine similarity over `f32` vectors.
///
/// Writes `out[i] = cosine_similarity_f32(query, db[i*stride .. (i+1)*stride])`
/// for each `i`.
///
/// **Optimization**: the query's L2 norm is computed once and reused
/// across the batch — the database row's L2 norm and the dot product
/// are computed fresh per pair.
///
/// # Panics
///
/// See module docs.
pub fn cosine_similarity_f32_one_to_many(
    query: &[f32],
    db: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    if out.is_empty() {
        return;
    }
    let query_norm_sq = kernels::auto::dot_f32(query, query).unwrap_or(0.0);
    if query_norm_sq == 0.0 {
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        return;
    }
    let query_norm = crate::math::sqrt_f32(query_norm_sq);
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let dot = kernels::auto::dot_f32(query, row).unwrap_or(0.0);
        let row_norm_sq = kernels::auto::dot_f32(row, row).unwrap_or(0.0);
        if row_norm_sq == 0.0 {
            *slot = 0.0;
        } else {
            *slot = dot / (query_norm * crate::math::sqrt_f32(row_norm_sq));
        }
    }
}

/// Many-vs-one Hamming distance over packed `u64` bitvector queries.
///
/// Writes `out[i] = hamming_u64(query, db[i*stride .. (i+1)*stride]) as u32`
/// for each `i`. The narrowing to `u32` is safe because each row of
/// `stride` u64 words holds at most `64 * stride` bits, and any
/// `stride` < `2^26` keeps the count well below `u32::MAX`.
///
/// # Panics
///
/// See module docs. Additionally panics if `stride > u32::MAX / 64` —
/// outside the regime where the narrowed `u32` cannot represent the
/// per-pair bit count.
pub fn hamming_u64_one_to_many(query: &[u64], db: &[u64], stride: usize, out: &mut [u32]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    assert!(
        stride <= (u32::MAX as usize) / 64,
        "hamming_u64_one_to_many: stride={stride} would overflow u32 output"
    );
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        // unwrap is safe: shapes asserted above.
        *slot = kernels::auto::hamming_u64(query, row).unwrap_or(0) as u32;
    }
}

/// Many-vs-one Jaccard similarity over packed `u64` bitvector queries.
///
/// Writes `out[i] = jaccard_u64(query, db[i*stride .. (i+1)*stride])` for
/// each `i`. Result is `f64` ∈ `[0.0, 1.0]`; `1.0` when both query and
/// row are all-zeros.
///
/// # Panics
///
/// See module docs.
pub fn jaccard_u64_one_to_many(query: &[u64], db: &[u64], stride: usize, out: &mut [f64]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        *slot = kernels::auto::jaccard_u64(query, row).unwrap_or(0.0);
    }
}

/// Validates the batched-API shape invariants and panics with a clear
/// message on mismatch.
#[inline]
fn assert_batch_shape(query_len: usize, db_len: usize, stride: usize, out_len: usize) {
    assert!(stride > 0, "batched API: stride must be > 0");
    assert_eq!(
        query_len, stride,
        "batched API: query.len()={query_len} but stride={stride}"
    );
    assert!(
        db_len.is_multiple_of(stride),
        "batched API: db.len()={db_len} not a multiple of stride={stride}"
    );
    let expected_out = db_len / stride;
    assert_eq!(
        out_len, expected_out,
        "batched API: out.len()={out_len} but db has {expected_out} rows of stride={stride}"
    );
}

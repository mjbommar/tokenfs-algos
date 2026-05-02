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

/// Failure modes for the fallible many-vs-one batched APIs in this
/// module ([`try_dot_f32_one_to_many`], [`try_l2_squared_f32_one_to_many`],
/// [`try_cosine_similarity_f32_one_to_many`], [`try_hamming_u64_one_to_many`],
/// and [`try_jaccard_u64_one_to_many`]).
///
/// Returned instead of panicking when the caller-supplied buffer shapes
/// are inconsistent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchShapeError {
    /// `out.len() != db.len() / stride`. Either the output buffer is
    /// over- or under-sized for the implied row count.
    OutLenMismatch {
        /// Expected output length (`db.len() / stride`).
        expected: usize,
        /// Caller-supplied output length.
        actual: usize,
    },
    /// `db.len()` is not divisible by `stride`.
    DbStrideMismatch {
        /// Caller-supplied `db.len()`.
        db_len: usize,
        /// Caller-supplied `stride`.
        stride: usize,
    },
    /// `stride` was zero — implies a zero-dimensional query.
    StrideZero,
    /// `query.len() != stride`.
    QueryStrideMismatch {
        /// Caller-supplied `query.len()`.
        query_len: usize,
        /// Caller-supplied `stride`.
        stride: usize,
    },
}

impl core::fmt::Display for BatchShapeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutLenMismatch { expected, actual } => write!(
                f,
                "batched API: out.len()={actual} but db has {expected} rows for the requested stride"
            ),
            Self::DbStrideMismatch { db_len, stride } => write!(
                f,
                "batched API: db.len()={db_len} not a multiple of stride={stride}"
            ),
            Self::StrideZero => write!(f, "batched API: stride must be > 0"),
            Self::QueryStrideMismatch { query_len, stride } => write!(
                f,
                "batched API: query.len()={query_len} but stride={stride}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BatchShapeError {}

/// Many-vs-one inner product over `f32` vectors.
///
/// Writes `out[i] = dot_f32(query, db[i*stride .. (i+1)*stride])` for
/// each `i ∈ 0..out.len()`.
///
/// # Panics
///
/// See module docs. Use [`try_dot_f32_one_to_many`] for a fallible
/// variant that returns [`BatchShapeError`] instead.
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_dot_f32_one_to_many`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn dot_f32_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    dot_f32_one_to_many_inner(query, db, stride, out);
}

/// Inner kernel for [`dot_f32_one_to_many`] / [`try_dot_f32_one_to_many`].
/// Assumes shape invariants have already been checked by the caller.
#[inline]
fn dot_f32_one_to_many_inner(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]) {
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        *slot = kernels::auto::dot_f32(query, row).unwrap_or(0.0);
    }
}

/// Fallible variant of [`dot_f32_one_to_many`] that returns
/// [`BatchShapeError`] when the caller-supplied buffer shapes are
/// inconsistent, instead of panicking.
pub fn try_dot_f32_one_to_many(
    query: &[f32],
    db: &[f32],
    stride: usize,
    out: &mut [f32],
) -> Result<(), BatchShapeError> {
    check_batch_shape(query.len(), db.len(), stride, out.len())?;
    dot_f32_one_to_many_inner(query, db, stride, out);
    Ok(())
}

/// Many-vs-one squared L2 distance over `f32` vectors.
///
/// Writes `out[i] = l2_squared_f32(query, db[i*stride .. (i+1)*stride])`
/// for each `i`.
///
/// # Panics
///
/// See module docs. Use [`try_l2_squared_f32_one_to_many`] for a
/// fallible variant that returns [`BatchShapeError`] instead.
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_l2_squared_f32_one_to_many`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn l2_squared_f32_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    l2_squared_f32_one_to_many_inner(query, db, stride, out);
}

/// Inner kernel for [`l2_squared_f32_one_to_many`] /
/// [`try_l2_squared_f32_one_to_many`]. Assumes shape invariants have
/// already been checked by the caller.
#[inline]
fn l2_squared_f32_one_to_many_inner(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]) {
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        *slot = kernels::auto::l2_squared_f32(query, row).unwrap_or(0.0);
    }
}

/// Fallible variant of [`l2_squared_f32_one_to_many`] that returns
/// [`BatchShapeError`] when the caller-supplied buffer shapes are
/// inconsistent, instead of panicking.
pub fn try_l2_squared_f32_one_to_many(
    query: &[f32],
    db: &[f32],
    stride: usize,
    out: &mut [f32],
) -> Result<(), BatchShapeError> {
    check_batch_shape(query.len(), db.len(), stride, out.len())?;
    l2_squared_f32_one_to_many_inner(query, db, stride, out);
    Ok(())
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
/// See module docs. Use [`try_cosine_similarity_f32_one_to_many`] for
/// a fallible variant that returns [`BatchShapeError`] instead.
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_cosine_similarity_f32_one_to_many`] (audit-R5
/// #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn cosine_similarity_f32_one_to_many(
    query: &[f32],
    db: &[f32],
    stride: usize,
    out: &mut [f32],
) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    cosine_similarity_f32_one_to_many_inner(query, db, stride, out);
}

/// Inner kernel for [`cosine_similarity_f32_one_to_many`] /
/// [`try_cosine_similarity_f32_one_to_many`]. Assumes shape invariants
/// have already been checked by the caller.
#[inline]
fn cosine_similarity_f32_one_to_many_inner(
    query: &[f32],
    db: &[f32],
    stride: usize,
    out: &mut [f32],
) {
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

/// Fallible variant of [`cosine_similarity_f32_one_to_many`] that
/// returns [`BatchShapeError`] when the caller-supplied buffer shapes
/// are inconsistent, instead of panicking.
pub fn try_cosine_similarity_f32_one_to_many(
    query: &[f32],
    db: &[f32],
    stride: usize,
    out: &mut [f32],
) -> Result<(), BatchShapeError> {
    check_batch_shape(query.len(), db.len(), stride, out.len())?;
    cosine_similarity_f32_one_to_many_inner(query, db, stride, out);
    Ok(())
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
/// per-pair bit count. Use [`try_hamming_u64_one_to_many`] for a
/// fallible variant that returns [`BatchShapeError`] instead.
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_hamming_u64_one_to_many`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn hamming_u64_one_to_many(query: &[u64], db: &[u64], stride: usize, out: &mut [u32]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    assert!(
        stride <= (u32::MAX as usize) / 64,
        "hamming_u64_one_to_many: stride={stride} would overflow u32 output"
    );
    hamming_u64_one_to_many_inner(query, db, stride, out);
}

/// Inner kernel for [`hamming_u64_one_to_many`] /
/// [`try_hamming_u64_one_to_many`]. Assumes shape invariants and the
/// `stride <= u32::MAX / 64` bound have already been checked by the
/// caller.
#[inline]
fn hamming_u64_one_to_many_inner(query: &[u64], db: &[u64], stride: usize, out: &mut [u32]) {
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        // unwrap is safe: shapes asserted above.
        *slot = kernels::auto::hamming_u64(query, row).unwrap_or(0) as u32;
    }
}

/// Fallible variant of [`hamming_u64_one_to_many`] that returns
/// [`BatchShapeError`] when the caller-supplied buffer shapes are
/// inconsistent, instead of panicking.
///
/// The `stride > u32::MAX / 64` overflow check is **not** mapped to a
/// [`BatchShapeError`] variant — that condition still panics, since it
/// reflects an output-encoding overflow rather than a buffer-shape
/// issue. Even with `panicking-shape-apis` disabled, this stride bound
/// remains a panicking precondition because no `try_*` mapping exists.
pub fn try_hamming_u64_one_to_many(
    query: &[u64],
    db: &[u64],
    stride: usize,
    out: &mut [u32],
) -> Result<(), BatchShapeError> {
    check_batch_shape(query.len(), db.len(), stride, out.len())?;
    assert!(
        stride <= (u32::MAX as usize) / 64,
        "try_hamming_u64_one_to_many: stride={stride} would overflow u32 output"
    );
    hamming_u64_one_to_many_inner(query, db, stride, out);
    Ok(())
}

/// Many-vs-one Jaccard similarity over packed `u64` bitvector queries.
///
/// Writes `out[i] = jaccard_u64(query, db[i*stride .. (i+1)*stride])` for
/// each `i`. Result is `f64` ∈ `[0.0, 1.0]`; `1.0` when both query and
/// row are all-zeros.
///
/// # Panics
///
/// See module docs. Use [`try_jaccard_u64_one_to_many`] for a fallible
/// variant that returns [`BatchShapeError`] instead.
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_jaccard_u64_one_to_many`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn jaccard_u64_one_to_many(query: &[u64], db: &[u64], stride: usize, out: &mut [f64]) {
    assert_batch_shape(query.len(), db.len(), stride, out.len());
    jaccard_u64_one_to_many_inner(query, db, stride, out);
}

/// Inner kernel for [`jaccard_u64_one_to_many`] /
/// [`try_jaccard_u64_one_to_many`]. Assumes shape invariants have
/// already been checked by the caller.
#[inline]
fn jaccard_u64_one_to_many_inner(query: &[u64], db: &[u64], stride: usize, out: &mut [f64]) {
    if out.is_empty() {
        return;
    }
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        *slot = kernels::auto::jaccard_u64(query, row).unwrap_or(0.0);
    }
}

/// Fallible variant of [`jaccard_u64_one_to_many`] that returns
/// [`BatchShapeError`] when the caller-supplied buffer shapes are
/// inconsistent, instead of panicking.
pub fn try_jaccard_u64_one_to_many(
    query: &[u64],
    db: &[u64],
    stride: usize,
    out: &mut [f64],
) -> Result<(), BatchShapeError> {
    check_batch_shape(query.len(), db.len(), stride, out.len())?;
    jaccard_u64_one_to_many_inner(query, db, stride, out);
    Ok(())
}

/// Validates the batched-API shape invariants and panics with a clear
/// message on mismatch.
///
/// Only used by the panicking entry points (gated on
/// `panicking-shape-apis`); the `try_*` variants call
/// [`check_batch_shape`] instead.
#[cfg(feature = "panicking-shape-apis")]
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

/// Validates the batched-API shape invariants without panicking.
///
/// Returns the matching [`BatchShapeError`] variant when any invariant
/// fails. Used as the entry-point check inside the `try_*` batched
/// APIs in this module.
#[inline]
fn check_batch_shape(
    query_len: usize,
    db_len: usize,
    stride: usize,
    out_len: usize,
) -> Result<(), BatchShapeError> {
    if stride == 0 {
        return Err(BatchShapeError::StrideZero);
    }
    if query_len != stride {
        return Err(BatchShapeError::QueryStrideMismatch { query_len, stride });
    }
    if !db_len.is_multiple_of(stride) {
        return Err(BatchShapeError::DbStrideMismatch { db_len, stride });
    }
    let expected = db_len / stride;
    if out_len != expected {
        return Err(BatchShapeError::OutLenMismatch {
            expected,
            actual: out_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

    extern crate alloc;

    use alloc::vec;

    use super::{
        BatchShapeError, try_cosine_similarity_f32_one_to_many, try_dot_f32_one_to_many,
        try_hamming_u64_one_to_many, try_jaccard_u64_one_to_many, try_l2_squared_f32_one_to_many,
    };
    #[cfg(feature = "panicking-shape-apis")]
    use super::{
        cosine_similarity_f32_one_to_many, dot_f32_one_to_many, hamming_u64_one_to_many,
        jaccard_u64_one_to_many, l2_squared_f32_one_to_many,
    };

    #[test]
    fn try_dot_returns_err_on_stride_zero() {
        let mut out = [0.0_f32; 1];
        let err = try_dot_f32_one_to_many(&[], &[], 0, &mut out).unwrap_err();
        assert_eq!(err, BatchShapeError::StrideZero);
    }

    #[test]
    fn try_dot_returns_err_on_query_stride_mismatch() {
        let mut out = [0.0_f32; 1];
        let err = try_dot_f32_one_to_many(&[1.0, 2.0], &[1.0, 2.0, 3.0], 3, &mut out).unwrap_err();
        assert_eq!(
            err,
            BatchShapeError::QueryStrideMismatch {
                query_len: 2,
                stride: 3
            }
        );
    }

    #[test]
    fn try_dot_returns_err_on_db_stride_mismatch() {
        let mut out = [0.0_f32; 1];
        let err = try_dot_f32_one_to_many(&[1.0, 2.0], &[1.0, 2.0, 3.0], 2, &mut out).unwrap_err();
        assert_eq!(
            err,
            BatchShapeError::DbStrideMismatch {
                db_len: 3,
                stride: 2
            }
        );
    }

    #[test]
    fn try_dot_returns_err_on_out_len_mismatch() {
        let mut out = [0.0_f32; 5];
        let err =
            try_dot_f32_one_to_many(&[1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 2, &mut out).unwrap_err();
        assert_eq!(
            err,
            BatchShapeError::OutLenMismatch {
                expected: 2,
                actual: 5
            }
        );
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_dot_returns_ok_and_matches_panic_version() {
        let query = [1.0_f32, 2.0, 3.0];
        let db = [1.0_f32, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut try_out = [0.0_f32; 2];
        try_dot_f32_one_to_many(&query, &db, 3, &mut try_out).unwrap();
        let mut panic_out = [0.0_f32; 2];
        dot_f32_one_to_many(&query, &db, 3, &mut panic_out);
        assert_eq!(try_out, panic_out);
        assert_eq!(try_out, [1.0, 2.0]);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_l2_squared_returns_ok_and_matches_panic_version() {
        let query = [1.0_f32, 2.0];
        let db = [1.0_f32, 2.0, 3.0, 4.0];
        let mut try_out = [0.0_f32; 2];
        try_l2_squared_f32_one_to_many(&query, &db, 2, &mut try_out).unwrap();
        let mut panic_out = [0.0_f32; 2];
        l2_squared_f32_one_to_many(&query, &db, 2, &mut panic_out);
        assert_eq!(try_out, panic_out);
        assert_eq!(try_out, [0.0, 8.0]);
    }

    #[test]
    fn try_l2_squared_returns_err_on_out_len_mismatch() {
        let query = [1.0_f32, 2.0];
        let db = [1.0_f32, 2.0, 3.0, 4.0];
        let mut out = [0.0_f32; 1];
        let err = try_l2_squared_f32_one_to_many(&query, &db, 2, &mut out).unwrap_err();
        assert_eq!(
            err,
            BatchShapeError::OutLenMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_cosine_returns_ok_and_matches_panic_version() {
        let query = [1.0_f32, 0.0];
        let db = [1.0_f32, 0.0, 0.0, 1.0];
        let mut try_out = [0.0_f32; 2];
        try_cosine_similarity_f32_one_to_many(&query, &db, 2, &mut try_out).unwrap();
        let mut panic_out = [0.0_f32; 2];
        cosine_similarity_f32_one_to_many(&query, &db, 2, &mut panic_out);
        assert_eq!(try_out, panic_out);
    }

    #[test]
    fn try_cosine_returns_err_on_db_stride_mismatch() {
        let query = [1.0_f32, 2.0];
        let db = [1.0_f32, 2.0, 3.0];
        let mut out = [0.0_f32; 1];
        let err = try_cosine_similarity_f32_one_to_many(&query, &db, 2, &mut out).unwrap_err();
        assert_eq!(
            err,
            BatchShapeError::DbStrideMismatch {
                db_len: 3,
                stride: 2
            }
        );
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_hamming_returns_ok_and_matches_panic_version() {
        let query = [0_u64, 0xff];
        let db = [0_u64, 0xff, 1, 0xff];
        let mut try_out = [0_u32; 2];
        try_hamming_u64_one_to_many(&query, &db, 2, &mut try_out).unwrap();
        let mut panic_out = [0_u32; 2];
        hamming_u64_one_to_many(&query, &db, 2, &mut panic_out);
        assert_eq!(try_out, panic_out);
    }

    #[test]
    fn try_hamming_returns_err_on_out_len_mismatch() {
        let query = [0_u64, 0xff];
        let db = [0_u64, 0xff];
        let mut out = vec![0_u32; 5];
        let err = try_hamming_u64_one_to_many(&query, &db, 2, &mut out).unwrap_err();
        assert_eq!(
            err,
            BatchShapeError::OutLenMismatch {
                expected: 1,
                actual: 5
            }
        );
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_jaccard_returns_ok_and_matches_panic_version() {
        let query = [0_u64, 0xff];
        let db = [0_u64, 0xff, 1, 0xff];
        let mut try_out = [0.0_f64; 2];
        try_jaccard_u64_one_to_many(&query, &db, 2, &mut try_out).unwrap();
        let mut panic_out = [0.0_f64; 2];
        jaccard_u64_one_to_many(&query, &db, 2, &mut panic_out);
        assert_eq!(try_out, panic_out);
    }

    #[test]
    fn try_jaccard_returns_err_on_stride_zero() {
        let mut out = [0.0_f64; 1];
        let err = try_jaccard_u64_one_to_many(&[], &[], 0, &mut out).unwrap_err();
        assert_eq!(err, BatchShapeError::StrideZero);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "batched API: stride must be > 0")]
    fn dot_panics_still_on_stride_zero() {
        let mut out = [0.0_f32; 1];
        dot_f32_one_to_many(&[], &[], 0, &mut out);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "batched API")]
    fn cosine_panics_still_on_db_stride_mismatch() {
        let mut out = [0.0_f32; 1];
        cosine_similarity_f32_one_to_many(&[1.0, 2.0], &[1.0, 2.0, 3.0], 2, &mut out);
    }
}

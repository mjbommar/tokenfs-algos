//! Scalar reference distance kernels.
//!
//! Phase 1 of the HNSW landing. Every kernel here is the parity
//! oracle for the SIMD backends added in Phases 2-3. Tests in
//! `tests.rs` validate against hand-computed reference values; future
//! per-backend parity tests assert SIMD agrees with these scalars
//! exactly (for integer metrics) or within a documented tolerance
//! (for f32 metrics — see `docs/hnsw/research/DETERMINISM.md` §9 on
//! cross-arch FMA behavior).

use super::super::Distance;
use super::encode_f32;

/// Errors from distance kernel calls. Length mismatch is the only
/// caller-visible failure mode for integer metrics; cosine adds a
/// zero-vector case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HnswKernelError {
    /// Inputs `a` and `b` had different lengths.
    LengthMismatch {
        /// Length of `a`.
        len_a: usize,
        /// Length of `b`.
        len_b: usize,
    },
    /// Cosine distance with a zero-magnitude input.
    CosineZeroVector,
}

impl core::fmt::Display for HnswKernelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch { len_a, len_b } => {
                write!(f, "length mismatch: a={len_a}, b={len_b}")
            }
            Self::CosineZeroVector => {
                f.write_str("cosine distance undefined for zero-magnitude vector")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HnswKernelError {}

// ---------------------------------------------------------------------
// f32 kernels
// ---------------------------------------------------------------------

/// L2² (squared Euclidean) over f32 vectors.
///
/// Returns the encoded integer distance ([`encode_f32`]) so the caller
/// can compare across metrics with a single integer comparator.
pub fn try_l2_squared_f32(a: &[f32], b: &[f32]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = x - y;
        sum += d * d;
    }
    Ok(encode_f32(sum))
}

/// Cosine distance over f32 vectors: `1 - dot(a, b) / (|a| * |b|)`.
///
/// Returns [`HnswKernelError::CosineZeroVector`] if either input has
/// zero magnitude (the metric is undefined there).
pub fn try_cosine_f32(a: &[f32], b: &[f32]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return Err(HnswKernelError::CosineZeroVector);
    }
    // Use crate::math::sqrt so this kernel is no_std-clean (libm fallback).
    let denom = crate::math::sqrt_f64(norm_a as f64) * crate::math::sqrt_f64(norm_b as f64);
    let cos_sim = (dot as f64) / denom;
    let distance = 1.0_f32 - cos_sim as f32;
    Ok(encode_f32(distance))
}

/// Negative dot product over f32 vectors.
///
/// usearch's "inner product" metric uses `-dot(a, b)` so the walker's
/// "smaller is better" invariant holds. We follow the same convention.
pub fn try_dot_f32(a: &[f32], b: &[f32]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut sum = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    Ok(encode_f32(-sum))
}

// ---------------------------------------------------------------------
// i8 kernels
// ---------------------------------------------------------------------

/// L2² (squared Euclidean) over i8 vectors.
///
/// Accumulator is i64 to avoid overflow even on long vectors with
/// extremes (max squared difference per lane is `255 * 255 = 65025`;
/// 2^63 / 65025 ≈ 1.4 × 10^14 lanes safe).
pub fn try_l2_squared_i8(a: &[i8], b: &[i8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut sum: i64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x as i32) - (y as i32);
        sum += (d * d) as i64;
    }
    // Result is non-negative; saturate to u32::MAX for absurdly long inputs.
    Ok(sum.try_into().unwrap_or(Distance::MAX))
}

/// Cosine distance over i8 vectors. Returns the encoded integer
/// distance computed via promotion to f64 (the i8 dot product values
/// are too large to fit in f32 reliably for long vectors).
pub fn try_cosine_i8(a: &[i8], b: &[i8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut dot: i64 = 0;
    let mut norm_a: i64 = 0;
    let mut norm_b: i64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as i32 * y as i32) as i64;
        norm_a += (x as i32 * x as i32) as i64;
        norm_b += (y as i32 * y as i32) as i64;
    }
    if norm_a == 0 || norm_b == 0 {
        return Err(HnswKernelError::CosineZeroVector);
    }
    let denom = crate::math::sqrt_f64(norm_a as f64) * crate::math::sqrt_f64(norm_b as f64);
    let cos_sim = (dot as f64) / denom;
    let distance = 1.0_f32 - cos_sim as f32;
    Ok(encode_f32(distance))
}

/// Negative dot product over i8 vectors.
pub fn try_dot_i8(a: &[i8], b: &[i8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut sum: i64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += (x as i32 * y as i32) as i64;
    }
    // -sum encoded into f32 for cross-metric heap compat.
    let distance = -(sum as f32);
    Ok(encode_f32(distance))
}

// ---------------------------------------------------------------------
// u8 kernels
// ---------------------------------------------------------------------

/// L2² (squared Euclidean) over u8 vectors.
pub fn try_l2_squared_u8(a: &[u8], b: &[u8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut sum: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x as i32) - (y as i32);
        sum += (d * d) as u64;
    }
    Ok(sum.try_into().unwrap_or(Distance::MAX))
}

/// Cosine distance over u8 vectors.
pub fn try_cosine_u8(a: &[u8], b: &[u8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut dot: u64 = 0;
    let mut norm_a: u64 = 0;
    let mut norm_b: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as u32 * y as u32) as u64;
        norm_a += (x as u32 * x as u32) as u64;
        norm_b += (y as u32 * y as u32) as u64;
    }
    if norm_a == 0 || norm_b == 0 {
        return Err(HnswKernelError::CosineZeroVector);
    }
    let denom = crate::math::sqrt_f64(norm_a as f64) * crate::math::sqrt_f64(norm_b as f64);
    let cos_sim = (dot as f64) / denom;
    let distance = 1.0_f32 - cos_sim as f32;
    Ok(encode_f32(distance))
}

/// Negative dot product over u8 vectors.
pub fn try_dot_u8(a: &[u8], b: &[u8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut sum: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += (x as u32 * y as u32) as u64;
    }
    let distance = -(sum as f32);
    Ok(encode_f32(distance))
}

// ---------------------------------------------------------------------
// binary (b1x8) kernels
// ---------------------------------------------------------------------

/// Hamming distance over packed binary vectors. Both inputs must have
/// the same byte length; each byte holds 8 dimensions (bit 0 is dim 0).
///
/// Returns the popcount of `a XOR b` — number of differing bits.
pub fn try_hamming_binary(a: &[u8], b: &[u8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut total: u32 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        total += (x ^ y).count_ones();
    }
    Ok(total)
}

/// Jaccard distance over packed binary vectors:
/// `1 - |a ∩ b| / |a ∪ b|`, returned as `Distance` (encoded f32).
///
/// Returns 0 (identical, max similarity) when both inputs are all-zero.
pub fn try_jaccard_binary(a: &[u8], b: &[u8]) -> Result<Distance, HnswKernelError> {
    check_len(a.len(), b.len())?;
    let mut intersect: u32 = 0;
    let mut union: u32 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        intersect += (x & y).count_ones();
        union += (x | y).count_ones();
    }
    if union == 0 {
        // Both vectors all-zero: define distance as 0 (identical).
        return Ok(encode_f32(0.0));
    }
    let similarity = (intersect as f32) / (union as f32);
    Ok(encode_f32(1.0_f32 - similarity))
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

#[inline]
fn check_len(a: usize, b: usize) -> Result<(), HnswKernelError> {
    if a == b {
        Ok(())
    } else {
        Err(HnswKernelError::LengthMismatch { len_a: a, len_b: b })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::super::decode_f32;
    use super::*;

    // Tolerance for f32 round-trip equality. Distance encoding is exact
    // for representable values, but cosine routes through f64 sqrt and
    // back — give a small absolute tolerance.
    const F32_TOL: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() <= F32_TOL || (a - b).abs() <= F32_TOL * a.abs().max(b.abs())
    }

    // ---- length-mismatch path

    #[test]
    fn length_mismatch_returns_error() {
        let err = try_l2_squared_f32(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert_eq!(err, HnswKernelError::LengthMismatch { len_a: 2, len_b: 1 });
    }

    // ---- f32

    #[test]
    fn l2_squared_f32_known_values() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        let d = try_l2_squared_f32(&a, &b).unwrap();
        assert!(approx_eq(decode_f32(d), 27.0));
    }

    #[test]
    fn l2_squared_f32_identical_returns_zero() {
        let a = [1.5_f32, -2.5, 3.5];
        let d = try_l2_squared_f32(&a, &a).unwrap();
        assert_eq!(d, encode_f32(0.0));
    }

    #[test]
    fn cosine_f32_orthogonal_returns_one() {
        let a = [1.0_f32, 0.0];
        let b = [0.0_f32, 1.0];
        let d = try_cosine_f32(&a, &b).unwrap();
        assert!(approx_eq(decode_f32(d), 1.0));
    }

    #[test]
    fn cosine_f32_identical_returns_zero() {
        let a = [3.0_f32, 4.0];
        let d = try_cosine_f32(&a, &a).unwrap();
        assert!(approx_eq(decode_f32(d), 0.0));
    }

    #[test]
    fn cosine_f32_zero_vector_returns_error() {
        let a = [0.0_f32, 0.0, 0.0];
        let b = [1.0_f32, 2.0, 3.0];
        let err = try_cosine_f32(&a, &b).unwrap_err();
        assert_eq!(err, HnswKernelError::CosineZeroVector);
    }

    #[test]
    fn dot_f32_known_values() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // dot = 4 + 10 + 18 = 32; we return -dot = -32
        let d = try_dot_f32(&a, &b).unwrap();
        assert!(approx_eq(decode_f32(d), -32.0));
    }

    // ---- i8

    #[test]
    fn l2_squared_i8_known_values() {
        let a: [i8; 3] = [10, 20, 30];
        let b: [i8; 3] = [13, 24, 35];
        // (10-13)^2 + (20-24)^2 + (30-35)^2 = 9 + 16 + 25 = 50
        let d = try_l2_squared_i8(&a, &b).unwrap();
        assert_eq!(d, 50);
    }

    #[test]
    fn l2_squared_i8_handles_extremes() {
        // (-128 - 127)^2 = 65025 per lane; 4 lanes = 260100
        let a: [i8; 4] = [-128, -128, -128, -128];
        let b: [i8; 4] = [127, 127, 127, 127];
        let d = try_l2_squared_i8(&a, &b).unwrap();
        assert_eq!(d, 260100);
    }

    #[test]
    fn cosine_i8_identical_returns_zero() {
        let a: [i8; 3] = [3, 4, 5];
        let d = try_cosine_i8(&a, &a).unwrap();
        assert!(approx_eq(decode_f32(d), 0.0));
    }

    #[test]
    fn dot_i8_known_values() {
        let a: [i8; 3] = [1, 2, 3];
        let b: [i8; 3] = [4, 5, 6];
        let d = try_dot_i8(&a, &b).unwrap();
        // 4 + 10 + 18 = 32; we return -32 encoded
        assert!(approx_eq(decode_f32(d), -32.0));
    }

    // ---- u8

    #[test]
    fn l2_squared_u8_known_values() {
        let a: [u8; 3] = [10, 20, 30];
        let b: [u8; 3] = [13, 24, 35];
        let d = try_l2_squared_u8(&a, &b).unwrap();
        assert_eq!(d, 50);
    }

    #[test]
    fn cosine_u8_identical_returns_zero() {
        let a: [u8; 3] = [3, 4, 5];
        let d = try_cosine_u8(&a, &a).unwrap();
        assert!(approx_eq(decode_f32(d), 0.0));
    }

    #[test]
    fn dot_u8_known_values() {
        let a: [u8; 3] = [1, 2, 3];
        let b: [u8; 3] = [4, 5, 6];
        let d = try_dot_u8(&a, &b).unwrap();
        assert!(approx_eq(decode_f32(d), -32.0));
    }

    // ---- binary

    #[test]
    fn hamming_binary_known_values() {
        // 0xFF XOR 0x00 = 0xFF (8 bits set)
        // 0x0F XOR 0xF0 = 0xFF (8 bits set)
        // 0xAA XOR 0x55 = 0xFF (8 bits set)
        // Total: 24 differing bits across 3 bytes (24 dimensions).
        let a = [0xFF, 0x0F, 0xAA];
        let b = [0x00, 0xF0, 0x55];
        let d = try_hamming_binary(&a, &b).unwrap();
        assert_eq!(d, 24);
    }

    #[test]
    fn hamming_binary_identical_returns_zero() {
        let a = [0x12, 0x34, 0x56, 0x78];
        let d = try_hamming_binary(&a, &a).unwrap();
        assert_eq!(d, 0);
    }

    #[test]
    fn jaccard_binary_identical_returns_zero() {
        let a = [0xFF, 0x0F, 0xAA];
        let d = try_jaccard_binary(&a, &a).unwrap();
        assert!(approx_eq(decode_f32(d), 0.0));
    }

    #[test]
    fn jaccard_binary_disjoint_returns_one() {
        let a = [0xFF, 0x00];
        let b = [0x00, 0xFF];
        // intersect = 0 bits; union = 16 bits → 1 - 0/16 = 1.0
        let d = try_jaccard_binary(&a, &b).unwrap();
        assert!(approx_eq(decode_f32(d), 1.0));
    }

    #[test]
    fn jaccard_binary_both_zero_returns_zero() {
        let a = [0x00, 0x00];
        let d = try_jaccard_binary(&a, &a).unwrap();
        assert_eq!(d, encode_f32(0.0));
    }

    #[test]
    fn jaccard_binary_partial_overlap() {
        // a = 0b1111_1100; b = 0b1111_0011
        // intersect = 0b1111_0000 (4 bits)
        // union     = 0b1111_1111 (8 bits)
        // similarity = 4/8 = 0.5; distance = 0.5
        let a = [0b1111_1100u8];
        let b = [0b1111_0011u8];
        let d = try_jaccard_binary(&a, &b).unwrap();
        assert!(approx_eq(decode_f32(d), 0.5));
    }

    // ---- distance ordering — the load-bearing property the candidate
    // heap relies on. Smaller-distance candidates must encode to smaller
    // integers across each metric.

    #[test]
    fn l2_f32_orders_consistent_with_raw_distance() {
        let a = [1.0, 1.0, 1.0];
        let close = [1.0, 1.0, 1.5];
        let far = [1.0, 1.0, 5.0];
        let d_close = try_l2_squared_f32(&a, &close).unwrap();
        let d_far = try_l2_squared_f32(&a, &far).unwrap();
        assert!(d_close < d_far);
    }

    #[test]
    fn hamming_binary_orders_consistent_with_raw_distance() {
        let a = [0xFFu8, 0xFF, 0xFF];
        let close = [0xFE, 0xFF, 0xFF]; // 1 bit different
        let far = [0x00, 0x00, 0x00]; // 24 bits different
        let d_close = try_hamming_binary(&a, &close).unwrap();
        let d_far = try_hamming_binary(&a, &far).unwrap();
        assert_eq!(d_close, 1);
        assert_eq!(d_far, 24);
        assert!(d_close < d_far);
    }
}

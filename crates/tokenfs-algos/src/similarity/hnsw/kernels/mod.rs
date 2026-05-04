//! Per-backend distance kernels for HNSW search.
//!
//! Phase 1 (this commit) ships the **scalar** reference kernels for
//! every supported (metric, scalar_kind) pair. These are the parity
//! oracle for every SIMD backend that lands in Phase 2 (AVX2 / NEON /
//! SSE4.1 / SSSE3) and Phase 3 (AVX-512).
//!
//! Per `docs/HNSW_PATH_DECISION.md` §10, the v1 backend matrix:
//!
//! | Metric            | f32 | i8 / u8 | binary (b1x8) |
//! |-------------------|:---:|:-------:|:-------------:|
//! | L2² (squared L2)  |  ✅ |    ✅   |       —       |
//! | cosine            |  ✅ |    ✅   |       —       |
//! | dot               |  ✅ |    ✅   |       —       |
//! | Hamming           |  — |    —    |       ✅      |
//! | Jaccard           |  — |    —    |       ✅      |
//!
//! Tanimoto on binary collapses to Jaccard (per `USEARCH_DEEP_DIVE` §5
//! / `SIMD_PRIOR_ART`), so we don't ship a separate kernel slot.
//!
//! # Distance encoding
//!
//! Per `docs/hnsw/components/WALKER.md`, all metrics return the
//! crate-wide [`Distance`] integer alias. f32 results are pre-encoded
//! via the IEEE-754 total-ordering bit trick so the candidate min-heap
//! comparator works correctly across all metrics. Helpers:
//! [`encode_f32`] / [`decode_f32`].
//!
//! # Audit posture
//!
//! - All kernels return `Result<Distance, HnswKernelError>` rather than
//!   panicking. Length mismatches and zero-vector cosine (division by
//!   zero) are explicit error variants.
//! - `no_std + core`-clean. Uses `crate::math` for `sqrt`/`ln` so the
//!   kernel works without `std` (via `libm`).

#[cfg(any(feature = "std", feature = "alloc"))]
pub mod scalar;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::scalar::HnswKernelError;

use super::Distance;

/// Encode an `f32` distance into the integer [`Distance`] type so the
/// candidate min-heap can use integer comparison for ALL metrics.
///
/// Uses the standard IEEE-754 → biased-uint total-ordering trick:
/// non-negative finites map monotonically into the upper half of the
/// `u32` range, negatives flip into the lower half. NaN is forced to
/// `Distance::MAX` so it sorts past every finite value and the walker
/// can safely treat NaN as "drop this candidate."
///
/// HNSW distances should always be non-negative (L2², cosine
/// distance = 1 - cosine_similarity, dot in inner-product metric is
/// stored as `-dot` so the walker's "smaller is better" invariant
/// holds), but we encode the full f32 range for completeness.
#[inline]
pub fn encode_f32(distance: f32) -> Distance {
    if distance.is_nan() {
        return Distance::MAX;
    }
    let bits = distance.to_bits();
    if (bits & 0x8000_0000) == 0 {
        // Non-negative: flip the sign bit so 0.0 maps to 0x8000_0000
        // and large positives map to high u32 values.
        bits ^ 0x8000_0000
    } else {
        // Negative: flip ALL bits so the most-negative f32 maps to 0
        // and -0.0 maps to 0x7FFF_FFFF.
        !bits
    }
}

/// Inverse of [`encode_f32`]. Round-trips except for `Distance::MAX`,
/// which decodes to `f32::INFINITY` (the original NaN can't be recovered).
#[inline]
pub fn decode_f32(encoded: Distance) -> f32 {
    if encoded == Distance::MAX {
        return f32::INFINITY;
    }
    let bits = if (encoded & 0x8000_0000) != 0 {
        encoded ^ 0x8000_0000
    } else {
        !encoded
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn encode_f32_preserves_total_order_for_non_negative() {
        let xs: &[f32] = &[0.0, 1e-30, 0.5, 1.0, 1.5, 1e10, f32::INFINITY];
        let mut encoded: Vec<u32> = xs.iter().copied().map(encode_f32).collect();
        let original_encoded = encoded.clone();
        encoded.sort_unstable();
        assert_eq!(encoded, original_encoded);
    }

    #[test]
    fn encode_f32_round_trip() {
        let xs: &[f32] = &[0.0, 1.0, 1e-10, 1e10, f32::INFINITY];
        for &x in xs {
            let r = decode_f32(encode_f32(x));
            assert_eq!(r.to_bits(), x.to_bits(), "{x} != {r}");
        }
    }

    #[test]
    fn encode_f32_nan_to_max() {
        assert_eq!(encode_f32(f32::NAN), Distance::MAX);
    }

    #[test]
    fn encode_f32_negative_orders_below_zero() {
        // -1.0 encoded should be less than 0.0 encoded.
        assert!(encode_f32(-1.0) < encode_f32(0.0));
        // And -1.0 encoded < -0.5 encoded < -0.1 encoded.
        assert!(encode_f32(-1.0) < encode_f32(-0.5));
        assert!(encode_f32(-0.5) < encode_f32(-0.1));
    }
}

//! Min-entropy estimators.

use crate::{histogram::ByteHistogram, math};

/// Computes byte min-entropy in bits.
///
/// Min-entropy is `-log2(max p_i)`. Empty inputs return `0.0`.
#[must_use]
pub fn h1(histogram: &ByteHistogram) -> f32 {
    min_entropy_counts_u64(histogram.counts(), histogram.total())
}

/// Computes min-entropy from `u64` counts.
#[must_use]
pub fn min_entropy_counts_u64(counts: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let max = counts.iter().copied().max().unwrap_or(0);
    if max == 0 {
        return 0.0;
    }
    -math::log2_f64(max as f64 / total as f64) as f32
}

/// Computes min-entropy from `u32` counts.
#[must_use]
pub fn min_entropy_counts_u32(counts: &[u32], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let max = counts.iter().copied().max().unwrap_or(0);
    if max == 0 {
        return 0.0;
    }
    -math::log2_f64(f64::from(max) / total as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::h1;
    use crate::histogram::ByteHistogram;

    #[test]
    fn min_entropy_constant_is_zero() {
        let histogram = ByteHistogram::from_block(&[7; 1024]);
        assert_eq!(h1(&histogram), 0.0);
    }

    #[test]
    fn min_entropy_uniform_byte_distribution_is_eight() {
        let bytes = (0_u8..=255).collect::<Vec<_>>();
        let histogram = ByteHistogram::from_block(&bytes);
        assert!((h1(&histogram) - 8.0).abs() < 1e-6);
    }
}

//! Renyi entropy estimators.

use crate::{histogram::ByteHistogram, math};

/// Computes Renyi entropy of order `alpha` over byte counts.
///
/// `alpha` must be positive and not equal to `1.0`. Use Shannon entropy for
/// the `alpha == 1` case. Empty histograms return `Some(0.0)`.
#[must_use]
pub fn h1_alpha(histogram: &ByteHistogram, alpha: f64) -> Option<f32> {
    renyi_entropy_counts_u64(histogram.counts(), histogram.total(), alpha)
}

/// Computes collision entropy, Renyi entropy of order 2.
#[must_use]
pub fn collision_h1(histogram: &ByteHistogram) -> f32 {
    collision_entropy_counts_u64(histogram.counts(), histogram.total())
}

/// Computes Renyi entropy from `u64` counts.
#[must_use]
pub fn renyi_entropy_counts_u64(counts: &[u64], total: u64, alpha: f64) -> Option<f32> {
    if total == 0 {
        return Some(0.0);
    }
    if alpha <= 0.0 || (alpha - 1.0).abs() <= f64::EPSILON {
        return None;
    }

    let total = total as f64;
    let mut sum = 0.0_f64;
    for &count in counts {
        if count != 0 {
            sum += math::powf_f64(count as f64 / total, alpha);
        }
    }
    Some((math::log2_f64(sum) / (1.0 - alpha)).max(0.0) as f32)
}

/// Computes collision entropy from `u64` counts.
#[must_use]
pub fn collision_entropy_counts_u64(counts: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    let mut sum = 0.0_f64;
    for &count in counts {
        if count != 0 {
            let p = count as f64 / total;
            sum += p * p;
        }
    }
    if sum == 0.0 {
        0.0
    } else {
        -math::log2_f64(sum) as f32
    }
}

/// Computes collision entropy from `u32` counts.
#[must_use]
pub fn collision_entropy_counts_u32(counts: &[u32], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    let mut sum = 0.0_f64;
    for &count in counts {
        if count != 0 {
            let p = f64::from(count) / total;
            sum += p * p;
        }
    }
    if sum == 0.0 {
        0.0
    } else {
        -math::log2_f64(sum) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{collision_h1, h1_alpha};
    use crate::histogram::ByteHistogram;
    // `Vec` is not in the no-std prelude; alias it from `alloc` for the
    // alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    #[test]
    fn collision_entropy_matches_uniform_byte_distribution() {
        let bytes = (0_u8..=255).collect::<Vec<_>>();
        let histogram = ByteHistogram::from_block(&bytes);
        assert!((collision_h1(&histogram) - 8.0).abs() < 1e-6);
    }

    #[test]
    fn renyi_rejects_shannon_order() {
        let histogram = ByteHistogram::from_block(b"abc");
        assert_eq!(h1_alpha(&histogram, 1.0), None);
    }
}

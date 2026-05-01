//! Joint entropy estimators over adjacent byte pairs.

use crate::{histogram::BytePairHistogram, math};

/// Computes exact adjacent-pair joint entropy `H(X_i, X_{i+1})`.
///
/// This uses a dense 65,536-bin byte-pair histogram and does not allocate.
#[must_use]
pub fn h2_pairs(bytes: &[u8]) -> f32 {
    let histogram = BytePairHistogram::from_bytes(bytes);
    h2_from_pair_histogram(&histogram)
}

/// Computes joint entropy from a dense byte-pair histogram.
#[must_use]
pub fn h2_from_pair_histogram(histogram: &BytePairHistogram) -> f32 {
    entropy_counts_u32(histogram.counts(), histogram.observations())
}

pub(crate) fn entropy_counts_u32(counts: &[u32], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }

    let total = total as f64;
    let mut entropy = 0.0_f64;
    for &count in counts {
        if count != 0 {
            let p = f64::from(count) / total;
            entropy -= p * math::log2_f64(p);
        }
    }
    entropy as f32
}

#[cfg(test)]
mod tests {
    use super::h2_pairs;

    #[test]
    fn repeated_pair_joint_entropy_is_zero() {
        assert_eq!(h2_pairs(b"aaaaaaaa"), 0.0);
    }

    #[test]
    fn alternating_pairs_have_one_bit_joint_entropy() {
        let entropy = h2_pairs(b"abababa");
        assert!((entropy - 1.0).abs() < 1e-6, "h2_pairs={entropy}");
    }
}

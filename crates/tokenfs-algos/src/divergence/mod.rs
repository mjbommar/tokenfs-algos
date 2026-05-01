//! Distribution divergence and distance measures.
//!
//! These functions compare fixed histograms without allocation. They are meant
//! for calibration experiments such as "does this random disk block look closer
//! to known gzip, source-code, font, or database byte distributions?"

/// Computes total-variation distance between two count vectors.
///
/// Returns `None` when lengths differ. Empty distributions compare as zero.
#[must_use]
pub fn total_variation_counts(a: &[u64], b: &[u64]) -> Option<f64> {
    let totals = CountTotals::new(a, b)?;
    if totals.both_empty() {
        return Some(0.0);
    }

    let mut sum = 0.0;
    for (&left, &right) in a.iter().zip(b) {
        let p = probability(left, totals.left);
        let q = probability(right, totals.right);
        sum += (p - q).abs();
    }

    Some(0.5 * sum)
}

/// Computes the Kolmogorov-Smirnov statistic between two count vectors.
///
/// For byte histograms this is the maximum gap between cumulative byte
/// distributions.
#[must_use]
pub fn ks_statistic_counts(a: &[u64], b: &[u64]) -> Option<f64> {
    let totals = CountTotals::new(a, b)?;
    if totals.both_empty() {
        return Some(0.0);
    }

    let mut left_cdf = 0.0;
    let mut right_cdf = 0.0;
    let mut best = 0.0_f64;
    for (&left, &right) in a.iter().zip(b) {
        left_cdf += probability(left, totals.left);
        right_cdf += probability(right, totals.right);
        best = best.max((left_cdf - right_cdf).abs());
    }

    Some(best)
}

/// Computes smoothed KL divergence `D_KL(a || b)` in natural-log units.
///
/// `smoothing` is added to every bin before normalization. Use a small positive
/// value such as `0.5` for MIME-profile comparisons so unseen bins do not force
/// infinite divergence.
#[must_use]
pub fn kl_divergence_counts(a: &[u64], b: &[u64], smoothing: f64) -> Option<f64> {
    let totals = CountTotals::new(a, b)?;
    if totals.both_empty() {
        return Some(0.0);
    }
    if smoothing < 0.0 {
        return None;
    }

    let bins = a.len() as f64;
    let left_total = totals.left as f64 + smoothing * bins;
    let right_total = totals.right as f64 + smoothing * bins;
    if left_total == 0.0 || right_total == 0.0 {
        return Some(0.0);
    }

    let mut divergence = 0.0;
    for (&left, &right) in a.iter().zip(b) {
        let p = (left as f64 + smoothing) / left_total;
        let q = (right as f64 + smoothing) / right_total;
        if p == 0.0 {
            continue;
        }
        if q == 0.0 {
            return Some(f64::INFINITY);
        }
        divergence += p * (p / q).ln();
    }

    Some(divergence.max(0.0))
}

/// Computes Jensen-Shannon distance between two count vectors.
///
/// This returns the square root of Jensen-Shannon divergence, making it a
/// symmetric distance. `smoothing` has the same meaning as in
/// [`kl_divergence_counts`].
#[must_use]
pub fn jensen_shannon_distance_counts(a: &[u64], b: &[u64], smoothing: f64) -> Option<f64> {
    let totals = CountTotals::new(a, b)?;
    if totals.both_empty() {
        return Some(0.0);
    }
    if smoothing < 0.0 {
        return None;
    }

    let bins = a.len() as f64;
    let left_total = totals.left as f64 + smoothing * bins;
    let right_total = totals.right as f64 + smoothing * bins;
    if left_total == 0.0 || right_total == 0.0 {
        return Some(0.0);
    }

    let mut left_kl = 0.0;
    let mut right_kl = 0.0;
    for (&left, &right) in a.iter().zip(b) {
        let p = (left as f64 + smoothing) / left_total;
        let q = (right as f64 + smoothing) / right_total;
        let m = 0.5 * (p + q);
        if p != 0.0 {
            left_kl += p * (p / m).ln();
        }
        if q != 0.0 {
            right_kl += q * (q / m).ln();
        }
    }

    Some((0.5 * (left_kl + right_kl)).max(0.0).sqrt())
}

/// Computes Hellinger distance between two count vectors.
#[must_use]
pub fn hellinger_distance_counts(a: &[u64], b: &[u64]) -> Option<f64> {
    let totals = CountTotals::new(a, b)?;
    if totals.both_empty() {
        return Some(0.0);
    }

    let mut sum = 0.0;
    for (&left, &right) in a.iter().zip(b) {
        let delta = probability(left, totals.left).sqrt() - probability(right, totals.right).sqrt();
        sum += delta * delta;
    }

    Some((0.5 * sum).sqrt())
}

/// Computes triangular discrimination between two count vectors.
///
/// This is a stable chi-squared-like symmetric distance over probabilities:
/// `sum((p - q)^2 / (p + q))`.
#[must_use]
pub fn triangular_discrimination_counts(a: &[u64], b: &[u64]) -> Option<f64> {
    let totals = CountTotals::new(a, b)?;
    if totals.both_empty() {
        return Some(0.0);
    }

    let mut sum = 0.0;
    for (&left, &right) in a.iter().zip(b) {
        let p = probability(left, totals.left);
        let q = probability(right, totals.right);
        let denom = p + q;
        if denom != 0.0 {
            let delta = p - q;
            sum += delta * delta / denom;
        }
    }

    Some(sum)
}

#[derive(Clone, Copy)]
struct CountTotals {
    left: u64,
    right: u64,
}

impl CountTotals {
    fn new(a: &[u64], b: &[u64]) -> Option<Self> {
        if a.len() != b.len() {
            return None;
        }
        Some(Self {
            left: a.iter().copied().sum(),
            right: b.iter().copied().sum(),
        })
    }

    const fn both_empty(self) -> bool {
        self.left == 0 && self.right == 0
    }
}

fn probability(count: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 / total as f64
    }
}

#[cfg(test)]
mod tests {
    use super::{
        hellinger_distance_counts, jensen_shannon_distance_counts, kl_divergence_counts,
        ks_statistic_counts, total_variation_counts, triangular_discrimination_counts,
    };

    #[test]
    fn identical_distributions_have_zero_distance() {
        let a = [1_u64, 2, 3, 4];
        assert_eq!(total_variation_counts(&a, &a), Some(0.0));
        assert_eq!(ks_statistic_counts(&a, &a), Some(0.0));
        assert_eq!(kl_divergence_counts(&a, &a, 0.0), Some(0.0));
        assert_eq!(jensen_shannon_distance_counts(&a, &a, 0.0), Some(0.0));
        assert_eq!(hellinger_distance_counts(&a, &a), Some(0.0));
        assert_eq!(triangular_discrimination_counts(&a, &a), Some(0.0));
    }

    #[test]
    fn distances_are_positive_for_disjoint_mass() {
        let a = [10_u64, 0];
        let b = [0_u64, 10];
        assert_eq!(total_variation_counts(&a, &b), Some(1.0));
        assert_eq!(ks_statistic_counts(&a, &b), Some(1.0));
        assert!(
            hellinger_distance_counts(&a, &b).expect("same-length counts should compare") > 0.99
        );
        assert!(
            triangular_discrimination_counts(&a, &b).expect("same-length counts should compare")
                > 1.9
        );
        assert!(
            jensen_shannon_distance_counts(&a, &b, 0.5).expect("same-length counts should compare")
                > 0.5
        );
    }

    #[test]
    fn mismatched_lengths_return_none() {
        assert_eq!(total_variation_counts(&[1], &[1, 2]), None);
        assert_eq!(kl_divergence_counts(&[1], &[1, 2], 0.5), None);
    }
}

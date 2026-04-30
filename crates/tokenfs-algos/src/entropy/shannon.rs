//! Shannon entropy estimators.

use crate::histogram::ByteHistogram;

/// Computes first-order Shannon entropy in bits per byte.
///
/// Empty histograms return `0.0`.
#[must_use]
pub fn h1(histogram: &ByteHistogram) -> f32 {
    let total = histogram.total();
    if total == 0 {
        return 0.0;
    }

    let total = total as f64;
    let entropy = histogram
        .counts()
        .iter()
        .copied()
        .filter(|count| *count != 0)
        .map(|count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum::<f64>();

    entropy as f32
}

#[cfg(test)]
mod tests {
    use super::h1;
    use crate::histogram::ByteHistogram;

    #[test]
    fn constant_entropy_is_zero() {
        let histogram = ByteHistogram::from_block(&[7; 1024]);
        assert_eq!(h1(&histogram), 0.0);
    }

    #[test]
    fn byte_uniform_entropy_is_eight() {
        let bytes = (0_u8..=255).collect::<Vec<_>>();
        let histogram = ByteHistogram::from_block(&bytes);
        assert!((h1(&histogram) - 8.0).abs() < 0.000_001);
    }
}

//! Calibrated byte-distribution fingerprints.
//!
//! This module is intentionally small and allocation-free. It gives callers a
//! stable shape for comparing a random disk block or streaming window against
//! calibrated byte-histogram references such as MIME/type profiles.

use crate::{divergence, histogram};

/// A 256-bin byte distribution fingerprint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ByteDistribution {
    counts: [u64; 256],
    total: u64,
}

impl ByteDistribution {
    /// Builds a byte distribution from raw bytes.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let histogram = histogram::kernels::direct_u64::block(bytes);
        Self::from_histogram(&histogram)
    }

    /// Builds a byte distribution from exact byte counts.
    #[must_use]
    pub fn from_counts(counts: [u64; 256]) -> Self {
        let total = counts.iter().copied().sum();
        Self { counts, total }
    }

    /// Builds a byte distribution from an existing histogram.
    #[must_use]
    pub fn from_histogram(histogram: &histogram::ByteHistogram) -> Self {
        Self {
            counts: *histogram.counts(),
            total: histogram.total(),
        }
    }

    /// Returns the byte counts.
    #[must_use]
    pub const fn counts(&self) -> &[u64; 256] {
        &self.counts
    }

    /// Returns the number of observed bytes.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.total
    }

    /// Returns true when no bytes were observed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Compares this distribution with another distribution using `metric`.
    #[must_use]
    pub fn distance(&self, other: &Self, metric: ByteDistributionMetric) -> f64 {
        metric.distance(self, other)
    }

    /// Computes the common byte-distribution distances in one pass-friendly
    /// diagnostic struct.
    #[must_use]
    pub fn distances(&self, other: &Self) -> ByteDistributionDistances {
        ByteDistributionDistances {
            total_variation: divergence::total_variation_counts(&self.counts, &other.counts)
                .unwrap_or(0.0),
            ks: divergence::ks_statistic_counts(&self.counts, &other.counts).unwrap_or(0.0),
            jensen_shannon: divergence::jensen_shannon_distance_counts(
                &self.counts,
                &other.counts,
                DEFAULT_BYTE_DISTRIBUTION_SMOOTHING,
            )
            .unwrap_or(0.0),
            hellinger: divergence::hellinger_distance_counts(&self.counts, &other.counts)
                .unwrap_or(0.0),
        }
    }
}

impl Default for ByteDistribution {
    fn default() -> Self {
        Self {
            counts: [0; 256],
            total: 0,
        }
    }
}

/// Default smoothing for sparse calibrated byte histograms.
pub const DEFAULT_BYTE_DISTRIBUTION_SMOOTHING: f64 = 0.5;

/// Distance metric for byte-distribution nearest-reference lookup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ByteDistributionMetric {
    /// Jensen-Shannon distance with default smoothing.
    JensenShannon,
    /// Hellinger distance.
    Hellinger,
    /// Total-variation distance.
    TotalVariation,
    /// Kolmogorov-Smirnov cumulative-distribution statistic.
    KolmogorovSmirnov,
}

impl ByteDistributionMetric {
    /// Stable metric identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::JensenShannon => "jensen-shannon",
            Self::Hellinger => "hellinger",
            Self::TotalVariation => "total-variation",
            Self::KolmogorovSmirnov => "kolmogorov-smirnov",
        }
    }

    /// Computes this metric between two byte distributions.
    #[must_use]
    pub fn distance(self, left: &ByteDistribution, right: &ByteDistribution) -> f64 {
        match self {
            Self::JensenShannon => divergence::jensen_shannon_distance_counts(
                left.counts(),
                right.counts(),
                DEFAULT_BYTE_DISTRIBUTION_SMOOTHING,
            ),
            Self::Hellinger => divergence::hellinger_distance_counts(left.counts(), right.counts()),
            Self::TotalVariation => {
                divergence::total_variation_counts(left.counts(), right.counts())
            }
            Self::KolmogorovSmirnov => {
                divergence::ks_statistic_counts(left.counts(), right.counts())
            }
        }
        .unwrap_or(f64::INFINITY)
    }
}

/// Common byte-distribution distances.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ByteDistributionDistances {
    /// Total-variation distance.
    pub total_variation: f64,
    /// Kolmogorov-Smirnov statistic.
    pub ks: f64,
    /// Jensen-Shannon distance.
    pub jensen_shannon: f64,
    /// Hellinger distance.
    pub hellinger: f64,
}

/// One calibrated byte-distribution reference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ByteDistributionReference<'a> {
    /// Human-readable reference label.
    pub label: &'a str,
    /// MIME type or other calibrated class label.
    pub mime_type: &'a str,
    /// Reference byte distribution.
    pub distribution: ByteDistribution,
}

impl<'a> ByteDistributionReference<'a> {
    /// Creates a labeled reference.
    #[must_use]
    pub const fn new(label: &'a str, mime_type: &'a str, distribution: ByteDistribution) -> Self {
        Self {
            label,
            mime_type,
            distribution,
        }
    }
}

/// Result of a nearest-reference lookup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NearestByteDistribution<'a> {
    /// Index into the reference slice.
    pub index: usize,
    /// Reference label.
    pub label: &'a str,
    /// Reference MIME/type label.
    pub mime_type: &'a str,
    /// Metric used for the lookup.
    pub metric: ByteDistributionMetric,
    /// Smaller means closer.
    pub distance: f64,
}

/// Finds the nearest calibrated reference to `query`.
#[must_use]
pub fn nearest_byte_distribution<'a>(
    query: &ByteDistribution,
    references: &'a [ByteDistributionReference<'a>],
    metric: ByteDistributionMetric,
) -> Option<NearestByteDistribution<'a>> {
    references
        .iter()
        .enumerate()
        .map(|(index, reference)| NearestByteDistribution {
            index,
            label: reference.label,
            mime_type: reference.mime_type,
            metric,
            distance: metric.distance(query, &reference.distribution),
        })
        .min_by(|left, right| {
            left.distance
                .partial_cmp(&right.distance)
                .unwrap_or(core::cmp::Ordering::Equal)
        })
}

/// Finds the nearest calibrated byte-distribution reference.
///
/// This is the short public-contract alias for callers that are already inside
/// the `distribution` module namespace.
#[must_use]
pub fn nearest_reference<'a>(
    query: &ByteDistribution,
    references: &'a [ByteDistributionReference<'a>],
    metric: ByteDistributionMetric,
) -> Option<NearestByteDistribution<'a>> {
    nearest_byte_distribution(query, references, metric)
}

#[cfg(test)]
mod tests {
    use super::{
        ByteDistribution, ByteDistributionMetric, ByteDistributionReference,
        nearest_byte_distribution, nearest_reference,
    };

    #[test]
    fn identical_distribution_has_zero_distance() {
        let distribution = ByteDistribution::from_bytes(b"aaaabbbbcccc");

        let distances = distribution.distances(&distribution);

        assert_eq!(distances.total_variation, 0.0);
        assert_eq!(distances.ks, 0.0);
        assert_eq!(distances.jensen_shannon, 0.0);
        assert_eq!(distances.hellinger, 0.0);
    }

    #[test]
    fn nearest_reference_picks_matching_shape() {
        let zeros = ByteDistribution::from_bytes(&[0; 4096]);
        let text = ByteDistribution::from_bytes(b"fn main() { println!(\"hello\"); }\n");
        let randomish = ByteDistribution::from_bytes(
            &(0..4096)
                .scan(0x0BAD_5EED_u64, |state, _| {
                    *state ^= *state >> 12;
                    *state ^= *state << 25;
                    *state ^= *state >> 27;
                    Some(state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8)
                })
                .collect::<Vec<_>>(),
        );
        let references = [
            ByteDistributionReference::new("zeros", "application/x-zero", zeros.clone()),
            ByteDistributionReference::new("text", "text/plain", text),
            ByteDistributionReference::new("random", "application/octet-stream", randomish),
        ];

        let nearest =
            nearest_byte_distribution(&zeros, &references, ByteDistributionMetric::JensenShannon)
                .expect("non-empty references");

        assert_eq!(nearest.index, 0);
        assert_eq!(nearest.mime_type, "application/x-zero");
        assert_eq!(nearest.distance, 0.0);

        let nearest_alias =
            nearest_reference(&zeros, &references, ByteDistributionMetric::JensenShannon)
                .expect("non-empty references");
        assert_eq!(nearest_alias.index, nearest.index);
    }

    #[test]
    fn metrics_have_stable_labels() {
        assert_eq!(
            ByteDistributionMetric::KolmogorovSmirnov.as_str(),
            "kolmogorov-smirnov"
        );
    }
}

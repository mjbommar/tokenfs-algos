//! Exact n-gram histograms.
//!
//! Exact histograms for `N > 1` require state proportional to the number of
//! distinct n-grams, so this module is available with `std` or `alloc`. Hash-bin
//! sketches remain the allocation-free hot-path option for high-cardinality
//! windows.

#[cfg(feature = "std")]
use std::collections::BTreeMap;

#[cfg(all(not(feature = "std"), feature = "alloc"))]
use alloc::collections::BTreeMap;

#[cfg(any(feature = "std", feature = "alloc"))]
use crate::windows;

/// Exact histogram of packed little-endian `N`-grams for `1 <= N <= 8`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NGramHistogram<const N: usize> {
    counts: BTreeMap<u64, u64>,
    observations: u64,
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl<const N: usize> NGramHistogram<N> {
    /// Builds an exact n-gram histogram from bytes.
    ///
    /// Unsupported widths (`N == 0` or `N > 8`) produce an empty histogram.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut histogram = Self::new();
        histogram.add_bytes(bytes);
        histogram
    }

    /// Creates an empty exact n-gram histogram.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: BTreeMap::new(),
            observations: 0,
        }
    }

    /// Adds all overlapping n-grams from `bytes`.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        if N == 0 || N > 8 || bytes.len() < N {
            return;
        }

        for offset in 0..=bytes.len() - N {
            let Some(key) = windows::pack_ngram_le::<N>(&bytes[offset..]) else {
                continue;
            };
            *self.counts.entry(key).or_insert(0) += 1;
            self.observations += 1;
        }
    }

    /// Returns the number of observed n-gram windows.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns the number of distinct observed n-grams.
    #[must_use]
    pub fn distinct(&self) -> usize {
        self.counts.len()
    }

    /// Returns true when no n-grams were observed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.observations == 0
    }

    /// Returns the count for one n-gram byte sequence.
    #[must_use]
    pub fn count_bytes(&self, ngram: &[u8]) -> Option<u64> {
        let key = windows::pack_ngram_le::<N>(ngram)?;
        Some(self.count_packed(key))
    }

    /// Returns the count for one packed little-endian n-gram.
    #[must_use]
    pub fn count_packed(&self, packed: u64) -> u64 {
        self.counts.get(&packed).copied().unwrap_or(0)
    }

    /// Iterates over packed n-gram/count pairs.
    pub fn iter(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.counts.iter().map(|(&ngram, &count)| (ngram, count))
    }
}

/// Builds an exact n-gram histogram.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn exact<const N: usize>(bytes: &[u8]) -> NGramHistogram<N> {
    NGramHistogram::from_bytes(bytes)
}

#[cfg(test)]
mod tests {
    use super::NGramHistogram;

    #[test]
    fn counts_overlapping_bigrams() {
        let histogram = NGramHistogram::<2>::from_bytes(b"ababa");
        assert_eq!(histogram.observations(), 4);
        assert_eq!(histogram.distinct(), 2);
        assert_eq!(histogram.count_bytes(b"ab"), Some(2));
        assert_eq!(histogram.count_bytes(b"ba"), Some(2));
        assert_eq!(histogram.count_bytes(b"aa"), Some(0));
    }

    #[test]
    fn unsupported_width_is_empty() {
        assert!(NGramHistogram::<0>::from_bytes(b"abc").is_empty());
        assert!(NGramHistogram::<9>::from_bytes(b"abcdefghi").is_empty());
    }
}

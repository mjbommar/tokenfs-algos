//! Dense byte-pair histograms.
//!
//! This is the allocation-free exact H2 building block. It keeps all 65,536
//! byte-pair counters in fixed storage, so callers should reuse the struct for
//! hot streaming paths rather than repeatedly placing it on small stacks.

use core::fmt;

/// Exact dense histogram of adjacent byte pairs.
#[derive(Clone, Eq, PartialEq)]
pub struct BytePairHistogram {
    counts: [u32; 256 * 256],
    observations: u64,
}

impl BytePairHistogram {
    /// Creates an empty pair histogram.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; 256 * 256],
            observations: 0,
        }
    }

    /// Builds a pair histogram from adjacent byte pairs in `bytes`.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut histogram = Self::new();
        histogram.add_bytes(bytes);
        histogram
    }

    /// Clears all counters.
    pub fn clear(&mut self) {
        self.counts = [0; 256 * 256];
        self.observations = 0;
    }

    /// Adds adjacent byte pairs from `bytes`.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        for pair in bytes.windows(2) {
            self.add_pair(pair[0], pair[1]);
        }
    }

    /// Adds one byte pair.
    pub fn add_pair(&mut self, first: u8, second: u8) {
        self.counts[pair_index(first, second)] += 1;
        self.observations += 1;
    }

    /// Returns the count for one byte pair.
    #[must_use]
    pub const fn count_pair(&self, first: u8, second: u8) -> u32 {
        self.counts[pair_index(first, second)]
    }

    /// Returns all dense pair counts.
    #[must_use]
    pub const fn counts(&self) -> &[u32; 256 * 256] {
        &self.counts
    }

    /// Returns the number of observed adjacent pairs.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns true when no pairs were observed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.observations == 0
    }

    /// Iterates over non-zero pair counts as `(first, second, count)`.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (u8, u8, u32)> + '_ {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(index, count)| {
                let first = (index >> 8) as u8;
                let second = (index & 0xff) as u8;
                (first, second, *count)
            })
    }
}

impl Default for BytePairHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BytePairHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytePairHistogram")
            .field("observations", &self.observations)
            .field(
                "distinct_pairs",
                &self.counts.iter().filter(|count| **count != 0).count(),
            )
            .finish_non_exhaustive()
    }
}

const fn pair_index(first: u8, second: u8) -> usize {
    ((first as usize) << 8) | second as usize
}

#[cfg(test)]
mod tests {
    use super::BytePairHistogram;

    #[test]
    fn counts_adjacent_pairs() {
        let histogram = BytePairHistogram::from_bytes(b"ababa");
        assert_eq!(histogram.observations(), 4);
        assert_eq!(histogram.count_pair(b'a', b'b'), 2);
        assert_eq!(histogram.count_pair(b'b', b'a'), 2);
        assert_eq!(histogram.count_pair(b'a', b'a'), 0);
    }

    #[test]
    fn clear_resets_pair_histogram() {
        let mut histogram = BytePairHistogram::from_bytes(b"abcdef");
        histogram.clear();
        assert!(histogram.is_empty());
        assert_eq!(histogram.counts().iter().sum::<u32>(), 0);
    }
}

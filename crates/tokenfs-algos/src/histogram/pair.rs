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

/// Reusable exact byte-pair scratch space with lazy counter reset.
///
/// `BytePairScratch` is intended for hot paths that repeatedly compute exact
/// adjacent-pair statistics. It keeps dense counters, but reset is proportional
/// to the next call's active pairs rather than 65,536 counters. The struct is
/// large by design; allocate it once per worker/file/stream and reuse it.
#[derive(Clone, Eq, PartialEq)]
pub struct BytePairScratch {
    counts: [u32; 256 * 256],
    pair_stamps: [u32; 256 * 256],
    active_pairs: [u16; 256 * 256],
    active_pair_len: usize,
    predecessor_counts: [u32; 256],
    predecessor_stamps: [u32; 256],
    active_predecessors: [u8; 256],
    active_predecessor_len: usize,
    generation: u32,
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

impl BytePairScratch {
    /// Creates empty reusable scratch state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; 256 * 256],
            pair_stamps: [0; 256 * 256],
            active_pairs: [0; 256 * 256],
            active_pair_len: 0,
            predecessor_counts: [0; 256],
            predecessor_stamps: [0; 256],
            active_predecessors: [0; 256],
            active_predecessor_len: 0,
            generation: 1,
            observations: 0,
        }
    }

    /// Lazily clears all counters.
    ///
    /// This does not touch every pair counter unless the internal generation
    /// wraps, which is effectively never for normal process lifetimes.
    pub fn clear(&mut self) {
        self.active_pair_len = 0;
        self.active_predecessor_len = 0;
        self.observations = 0;
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.pair_stamps = [0; 256 * 256];
            self.predecessor_stamps = [0; 256];
            self.generation = 1;
        }
    }

    /// Clears and adds adjacent byte pairs from `bytes`.
    pub fn reset_and_add_bytes(&mut self, bytes: &[u8]) {
        self.clear();
        self.add_bytes(bytes);
    }

    /// Adds adjacent byte pairs from `bytes`.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        for pair in bytes.windows(2) {
            self.add_pair(pair[0], pair[1]);
        }
    }

    /// Adds one byte pair.
    pub fn add_pair(&mut self, first: u8, second: u8) {
        let index = pair_index(first, second);
        if self.pair_stamps[index] != self.generation {
            self.pair_stamps[index] = self.generation;
            self.counts[index] = 0;
            self.active_pairs[self.active_pair_len] = index as u16;
            self.active_pair_len += 1;
        }
        self.counts[index] += 1;

        let predecessor = first as usize;
        if self.predecessor_stamps[predecessor] != self.generation {
            self.predecessor_stamps[predecessor] = self.generation;
            self.predecessor_counts[predecessor] = 0;
            self.active_predecessors[self.active_predecessor_len] = first;
            self.active_predecessor_len += 1;
        }
        self.predecessor_counts[predecessor] += 1;
        self.observations += 1;
    }

    /// Returns the count for one byte pair in the current generation.
    #[must_use]
    pub fn count_pair(&self, first: u8, second: u8) -> u32 {
        let index = pair_index(first, second);
        if self.pair_stamps[index] == self.generation {
            self.counts[index]
        } else {
            0
        }
    }

    /// Returns the predecessor count for one first byte.
    #[must_use]
    pub fn predecessor_count(&self, first: u8) -> u32 {
        let index = first as usize;
        if self.predecessor_stamps[index] == self.generation {
            self.predecessor_counts[index]
        } else {
            0
        }
    }

    /// Returns the number of observed adjacent pairs.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns the number of distinct active pairs.
    #[must_use]
    pub const fn distinct_pairs(&self) -> usize {
        self.active_pair_len
    }

    /// Returns true when no pairs were observed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.observations == 0
    }

    /// Iterates over non-zero pair counts as `(first, second, count)`.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (u8, u8, u32)> + '_ {
        self.active_pairs[..self.active_pair_len]
            .iter()
            .copied()
            .map(|index| {
                let index = index as usize;
                let first = (index >> 8) as u8;
                let second = (index & 0xff) as u8;
                (first, second, self.counts[index])
            })
    }

    /// Iterates over non-zero predecessor counts as `(byte, count)`.
    pub fn iter_predecessors(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.active_predecessors[..self.active_predecessor_len]
            .iter()
            .copied()
            .map(|byte| (byte, self.predecessor_counts[byte as usize]))
    }
}

impl Default for BytePairScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BytePairScratch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytePairScratch")
            .field("observations", &self.observations)
            .field("distinct_pairs", &self.active_pair_len)
            .field("distinct_predecessors", &self.active_predecessor_len)
            .finish_non_exhaustive()
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
    use super::{BytePairHistogram, BytePairScratch};
    // `Box` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;

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

    #[test]
    fn scratch_reuses_storage_with_lazy_clear() {
        let mut scratch = Box::new(BytePairScratch::new());
        scratch.reset_and_add_bytes(b"ababa");
        assert_eq!(scratch.observations(), 4);
        assert_eq!(scratch.count_pair(b'a', b'b'), 2);
        assert_eq!(scratch.predecessor_count(b'a'), 2);

        scratch.reset_and_add_bytes(b"zzzz");
        assert_eq!(scratch.observations(), 3);
        assert_eq!(scratch.count_pair(b'a', b'b'), 0);
        assert_eq!(scratch.count_pair(b'z', b'z'), 3);
        assert_eq!(scratch.distinct_pairs(), 1);
    }
}

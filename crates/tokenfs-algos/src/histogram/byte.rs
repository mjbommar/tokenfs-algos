//! Byte histogram implementation.

use core::ops::{Add, AddAssign};

use crate::primitives::histogram_scalar;

/// A 256-bin byte histogram.
///
/// Counts are stored as `u64` values so filesystem-scale extents can be
/// accumulated without narrowing. Default block construction goes through the
/// histogram planner; pinned kernels are available under
/// [`crate::histogram::kernels`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ByteHistogram {
    counts: [u64; 256],
    total: u64,
}

impl ByteHistogram {
    /// Creates an empty byte histogram.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; 256],
            total: 0,
        }
    }

    /// Builds a byte histogram from `block`.
    #[must_use]
    pub fn from_block(block: &[u8]) -> Self {
        crate::histogram::block(block)
    }

    /// Adds all bytes from `block` to this histogram.
    ///
    /// This method keeps the direct scalar path for predictable incremental
    /// accumulation. Use [`crate::histogram::block`] for planned whole-block
    /// construction.
    pub fn add_block(&mut self, block: &[u8]) {
        histogram_scalar::add_block_direct_u64(block, &mut self.counts);
        self.total += block.len() as u64;
    }

    /// Adds one byte to this histogram.
    pub fn add_byte(&mut self, byte: u8) {
        self.counts[byte as usize] += 1;
        self.total += 1;
    }

    /// Returns the 256 byte counts.
    #[must_use]
    pub const fn counts(&self) -> &[u64; 256] {
        &self.counts
    }

    /// Returns the total number of observed bytes.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.total
    }

    /// Returns true when the histogram contains no observations.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Resets all counts to zero.
    pub fn clear(&mut self) {
        self.counts = [0; 256];
        self.total = 0;
    }

    /// Iterates over non-zero counts as `(byte, count)` pairs.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (u8, u64)> + '_ {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(byte, count)| (byte as u8, *count))
    }

    pub(crate) const fn counts_mut_for_primitives(&mut self) -> &mut [u64; 256] {
        &mut self.counts
    }

    pub(crate) fn add_to_total_for_primitives(&mut self, amount: u64) {
        self.total += amount;
    }
}

impl Default for ByteHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl AddAssign<&Self> for ByteHistogram {
    fn add_assign(&mut self, rhs: &Self) {
        for (left, right) in self.counts.iter_mut().zip(rhs.counts) {
            *left += right;
        }
        self.total += rhs.total;
    }
}

impl Add<&Self> for ByteHistogram {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self += rhs;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::ByteHistogram;

    #[test]
    fn counts_bytes() {
        let histogram = ByteHistogram::from_block(b"abbccc");

        assert_eq!(histogram.total(), 6);
        assert_eq!(histogram.counts()[b'a' as usize], 1);
        assert_eq!(histogram.counts()[b'b' as usize], 2);
        assert_eq!(histogram.counts()[b'c' as usize], 3);
    }

    #[test]
    fn add_combines_counts() {
        let left = ByteHistogram::from_block(b"abc");
        let right = ByteHistogram::from_block(b"cde");

        let combined = left + &right;

        assert_eq!(combined.total(), 6);
        assert_eq!(combined.counts()[b'c' as usize], 2);
    }
}

//! Content-defined chunking primitives.
//!
//! This module provides allocation-free Gear/FastCDC-style boundary detection.
//! It does not perform file I/O and does not own chunk payloads.

use crate::windows::{GearHash64, gear_update};

/// One chunk boundary over an input byte slice.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Chunk {
    /// Start offset, inclusive.
    pub start: usize,
    /// End offset, exclusive.
    pub end: usize,
    /// Rolling hash value observed at the boundary.
    pub boundary_hash: u64,
}

impl Chunk {
    /// Returns the chunk length in bytes.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Returns true when the chunk is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start >= self.end
    }
}

/// Configuration for Gear/FastCDC-style chunking.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkConfig {
    /// Minimum chunk size.
    pub min_size: usize,
    /// Target average chunk size. This is rounded to a power-of-two mask.
    pub avg_size: usize,
    /// Maximum chunk size.
    pub max_size: usize,
}

impl ChunkConfig {
    /// Creates a conservative configuration from an average chunk size.
    #[must_use]
    pub fn new(avg_size: usize) -> Self {
        let avg_size = avg_size.max(64).next_power_of_two();
        Self {
            min_size: (avg_size / 4).max(16),
            avg_size,
            max_size: (avg_size * 4).max(avg_size),
        }
    }

    /// Creates a configuration with explicit min/avg/max sizes.
    #[must_use]
    pub fn with_sizes(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        let avg_size = avg_size.max(1).next_power_of_two();
        let min_size = min_size.min(max_size).min(avg_size).max(1);
        let max_size = max_size.max(min_size).max(avg_size);
        Self {
            min_size,
            avg_size,
            max_size,
        }
    }

    /// Returns the boundary mask implied by the average chunk size.
    #[must_use]
    pub fn mask(self) -> u64 {
        (self.avg_size.next_power_of_two() as u64).saturating_sub(1)
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self::new(64 * 1024)
    }
}

/// Streaming Gear/FastCDC-style chunk iterator.
#[derive(Clone, Debug)]
pub struct Chunker<'a> {
    bytes: &'a [u8],
    config: ChunkConfig,
    position: usize,
    hash: GearHash64,
}

impl<'a> Chunker<'a> {
    /// Creates a chunk iterator over `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8], config: ChunkConfig) -> Self {
        Self {
            bytes,
            config,
            position: 0,
            hash: GearHash64::new(),
        }
    }
}

impl Iterator for Chunker<'_> {
    type Item = Chunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.bytes.len() {
            return None;
        }

        let start = self.position;
        let boundary =
            find_boundary_with_state(&self.bytes[start..], self.config, self.hash.value());

        let end = start + boundary.bytes;
        self.position = end;
        self.hash = boundary.hash;

        Some(Chunk {
            start,
            end,
            boundary_hash: boundary.hash.value(),
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Boundary {
    bytes: usize,
    hash: GearHash64,
}

/// Returns a streaming chunk iterator over `bytes`.
#[must_use]
pub const fn chunks(bytes: &[u8], config: ChunkConfig) -> Chunker<'_> {
    Chunker::new(bytes, config)
}

/// Finds the first chunk boundary in `bytes`.
///
/// The returned value is a length in bytes, never larger than `bytes.len()`.
#[must_use]
pub fn find_boundary(bytes: &[u8], config: ChunkConfig) -> usize {
    find_boundary_with_state(bytes, config, 0).bytes
}

fn find_boundary_with_state(bytes: &[u8], config: ChunkConfig, initial_hash: u64) -> Boundary {
    if bytes.is_empty() {
        return Boundary {
            bytes: 0,
            hash: GearHash64::new(),
        };
    }

    let mut hash = GearHash64::from_value(initial_hash);
    let min = config.min_size.min(bytes.len());
    let max = config.max_size.min(bytes.len()).max(min);
    let mask = config.mask();

    for &byte in &bytes[..min] {
        hash.update(byte);
    }

    if min == bytes.len() {
        return Boundary { bytes: min, hash };
    }

    for (offset, &byte) in bytes[min..max].iter().enumerate() {
        let value = hash.update(byte);
        let boundary = min + offset + 1;
        if value & mask == 0 {
            return Boundary {
                bytes: boundary,
                hash,
            };
        }
    }

    Boundary { bytes: max, hash }
}

/// Computes the Gear hash of `bytes` from an initial hash value.
#[must_use]
pub fn gear_hash(bytes: &[u8], initial: u64) -> u64 {
    let mut hash = initial;
    for &byte in bytes {
        hash = gear_update(hash, byte);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{ChunkConfig, chunks, find_boundary, gear_hash};

    #[test]
    fn explicit_config_is_normalized() {
        let config = ChunkConfig::with_sizes(0, 1000, 10);
        assert_eq!(config.avg_size, 1024);
        assert_eq!(config.min_size, 1);
        assert_eq!(config.max_size, 1024);
        assert_eq!(config.mask(), 1023);
    }

    #[test]
    fn boundary_respects_limits() {
        let bytes = (0_usize..100_000)
            .map(|i| (i.wrapping_mul(17) ^ (i >> 3)) as u8)
            .collect::<Vec<_>>();
        let config = ChunkConfig::with_sizes(1024, 4096, 8192);
        let boundary = find_boundary(&bytes, config);
        assert!((1024..=8192).contains(&boundary));
    }

    #[test]
    fn chunker_covers_input_without_overlap() {
        let bytes = (0_usize..65_537)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 2)) as u8)
            .collect::<Vec<_>>();
        let config = ChunkConfig::with_sizes(256, 1024, 4096);
        let parts = chunks(&bytes, config).collect::<Vec<_>>();

        assert_eq!(parts.first().map(|chunk| chunk.start), Some(0));
        assert_eq!(parts.last().map(|chunk| chunk.end), Some(bytes.len()));
        for pair in parts.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
            assert!(!pair[0].is_empty());
        }
    }

    #[test]
    fn gear_hash_is_streamable() {
        let full = gear_hash(b"abcdef", 0);
        let left = gear_hash(b"abc", 0);
        let split = gear_hash(b"def", left);
        assert_eq!(full, split);
    }
}

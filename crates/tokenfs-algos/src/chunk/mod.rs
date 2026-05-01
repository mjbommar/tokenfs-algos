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

/// Configuration for normalized FastCDC-style chunking.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FastCdcConfig {
    /// Minimum chunk size.
    pub min_size: usize,
    /// Target average chunk size. This is rounded to a power-of-two mask.
    pub avg_size: usize,
    /// Maximum chunk size.
    pub max_size: usize,
    /// Number of mask bits used for pre/post-average normalization.
    pub normalization_level: u8,
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

impl FastCdcConfig {
    /// Creates a normalized FastCDC configuration from an average chunk size.
    #[must_use]
    pub fn new(avg_size: usize) -> Self {
        let base = ChunkConfig::new(avg_size);
        Self {
            min_size: base.min_size,
            avg_size: base.avg_size,
            max_size: base.max_size,
            normalization_level: 1,
        }
    }

    /// Creates a normalized FastCDC configuration with explicit sizes.
    #[must_use]
    pub fn with_sizes(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        let base = ChunkConfig::with_sizes(min_size, avg_size, max_size);
        Self {
            min_size: base.min_size,
            avg_size: base.avg_size,
            max_size: base.max_size,
            normalization_level: 1,
        }
    }

    /// Sets the normalization level.
    ///
    /// A higher level makes the pre-average mask stricter and the post-average
    /// mask looser. Values above 8 are clamped.
    #[must_use]
    pub const fn with_normalization_level(mut self, level: u8) -> Self {
        self.normalization_level = if level > 8 { 8 } else { level };
        self
    }

    /// Returns the strict pre-average mask.
    #[must_use]
    pub fn small_mask(self) -> u64 {
        mask_for_bits(mask_bits(self.avg_size).saturating_add(u32::from(self.normalization_level)))
    }

    /// Returns the loose post-average mask.
    #[must_use]
    pub fn large_mask(self) -> u64 {
        mask_for_bits(mask_bits(self.avg_size).saturating_sub(u32::from(self.normalization_level)))
    }

    /// Converts to the basic Gear chunking configuration.
    #[must_use]
    pub const fn as_gear_config(self) -> ChunkConfig {
        ChunkConfig {
            min_size: self.min_size,
            avg_size: self.avg_size,
            max_size: self.max_size,
        }
    }
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self::new(64 * 1024)
    }
}

impl Default for FastCdcConfig {
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

/// Streaming normalized FastCDC-style chunk iterator.
#[derive(Clone, Debug)]
pub struct FastCdcChunker<'a> {
    bytes: &'a [u8],
    config: FastCdcConfig,
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

impl<'a> FastCdcChunker<'a> {
    /// Creates a normalized FastCDC chunk iterator over `bytes`.
    #[must_use]
    pub const fn new(bytes: &'a [u8], config: FastCdcConfig) -> Self {
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

impl Iterator for FastCdcChunker<'_> {
    type Item = Chunk;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.bytes.len() {
            return None;
        }

        let start = self.position;
        let boundary =
            fastcdc_boundary_with_state(&self.bytes[start..], self.config, self.hash.value());

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

/// Pinned chunking kernels.
pub mod kernels {
    use super::{Chunk, ChunkConfig, Chunker, FastCdcChunker, FastCdcConfig};

    /// Basic Gear-mask chunking.
    pub mod gear {
        use super::{Chunk, ChunkConfig, Chunker};
        use crate::chunk::{chunks as gear_chunks, find_boundary as gear_find_boundary};

        /// Returns a Gear chunk iterator.
        #[must_use]
        pub const fn chunks(bytes: &[u8], config: ChunkConfig) -> Chunker<'_> {
            gear_chunks(bytes, config)
        }

        /// Finds the first Gear chunk boundary.
        #[must_use]
        pub fn find_boundary(bytes: &[u8], config: ChunkConfig) -> usize {
            gear_find_boundary(bytes, config)
        }

        /// Collects the first chunk descriptor for callers that need a fixed
        /// pinned-kernel function shape.
        #[must_use]
        pub fn first_chunk(bytes: &[u8], config: ChunkConfig) -> Option<Chunk> {
            gear_chunks(bytes, config).next()
        }
    }

    /// Normalized FastCDC-style chunking.
    pub mod fastcdc {
        use super::{Chunk, FastCdcChunker, FastCdcConfig};
        use crate::chunk::{
            fastcdc_chunks as normalized_chunks, fastcdc_find_boundary as normalized_find_boundary,
        };

        /// Returns a normalized FastCDC chunk iterator.
        #[must_use]
        pub const fn chunks(bytes: &[u8], config: FastCdcConfig) -> FastCdcChunker<'_> {
            normalized_chunks(bytes, config)
        }

        /// Finds the first normalized FastCDC chunk boundary.
        #[must_use]
        pub fn find_boundary(bytes: &[u8], config: FastCdcConfig) -> usize {
            normalized_find_boundary(bytes, config)
        }

        /// Collects the first chunk descriptor for callers that need a fixed
        /// pinned-kernel function shape.
        #[must_use]
        pub fn first_chunk(bytes: &[u8], config: FastCdcConfig) -> Option<Chunk> {
            normalized_chunks(bytes, config).next()
        }
    }
}

/// Returns a streaming chunk iterator over `bytes`.
#[must_use]
pub const fn chunks(bytes: &[u8], config: ChunkConfig) -> Chunker<'_> {
    Chunker::new(bytes, config)
}

/// Returns a normalized FastCDC chunk iterator over `bytes`.
#[must_use]
pub const fn fastcdc_chunks(bytes: &[u8], config: FastCdcConfig) -> FastCdcChunker<'_> {
    FastCdcChunker::new(bytes, config)
}

/// Finds the first chunk boundary in `bytes`.
///
/// The returned value is a length in bytes, never larger than `bytes.len()`.
#[must_use]
pub fn find_boundary(bytes: &[u8], config: ChunkConfig) -> usize {
    find_boundary_with_state(bytes, config, 0).bytes
}

/// Finds the first normalized FastCDC boundary in `bytes`.
///
/// The returned value is a length in bytes, never larger than `bytes.len()`.
#[must_use]
pub fn fastcdc_find_boundary(bytes: &[u8], config: FastCdcConfig) -> usize {
    fastcdc_boundary_with_state(bytes, config, 0).bytes
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

fn fastcdc_boundary_with_state(bytes: &[u8], config: FastCdcConfig, initial_hash: u64) -> Boundary {
    if bytes.is_empty() {
        return Boundary {
            bytes: 0,
            hash: GearHash64::new(),
        };
    }

    let mut hash = GearHash64::from_value(initial_hash);
    let min = config.min_size.min(bytes.len());
    let normal = config.avg_size.min(bytes.len()).max(min);
    let max = config.max_size.min(bytes.len()).max(normal);
    let small_mask = config.small_mask();
    let large_mask = config.large_mask();

    for &byte in &bytes[..min] {
        hash.update(byte);
    }

    if min == bytes.len() {
        return Boundary { bytes: min, hash };
    }

    for (offset, &byte) in bytes[min..normal].iter().enumerate() {
        let value = hash.update(byte);
        let boundary = min + offset + 1;
        if value & small_mask == 0 {
            return Boundary {
                bytes: boundary,
                hash,
            };
        }
    }

    if normal == bytes.len() {
        return Boundary {
            bytes: normal,
            hash,
        };
    }

    for (offset, &byte) in bytes[normal..max].iter().enumerate() {
        let value = hash.update(byte);
        let boundary = normal + offset + 1;
        if value & large_mask == 0 {
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

fn mask_bits(avg_size: usize) -> u32 {
    avg_size.max(1).next_power_of_two().trailing_zeros()
}

fn mask_for_bits(bits: u32) -> u64 {
    if bits >= 63 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ChunkConfig, FastCdcConfig, chunks, fastcdc_chunks, fastcdc_find_boundary, find_boundary,
        gear_hash,
    };

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

    #[test]
    fn fastcdc_masks_are_normalized_around_average() {
        let config = FastCdcConfig::with_sizes(1024, 4096, 16 * 1024).with_normalization_level(2);
        assert_eq!(config.small_mask(), 16_383);
        assert_eq!(config.large_mask(), 1023);
    }

    #[test]
    fn fastcdc_boundary_respects_limits() {
        let bytes = (0_usize..100_000)
            .map(|i| (i.wrapping_mul(17) ^ (i >> 3)) as u8)
            .collect::<Vec<_>>();
        let config = FastCdcConfig::with_sizes(1024, 4096, 8192);
        let boundary = fastcdc_find_boundary(&bytes, config);
        assert!((1024..=8192).contains(&boundary));
    }

    #[test]
    fn fastcdc_chunker_covers_input_without_overlap() {
        let bytes = (0_usize..1_048_576)
            .map(|i| (i.wrapping_mul(131) ^ (i >> 5).wrapping_mul(17)) as u8)
            .collect::<Vec<_>>();
        let config = FastCdcConfig::with_sizes(1024, 4096, 16 * 1024);
        let parts = fastcdc_chunks(&bytes, config).collect::<Vec<_>>();

        assert_eq!(parts.first().map(|chunk| chunk.start), Some(0));
        assert_eq!(parts.last().map(|chunk| chunk.end), Some(bytes.len()));
        for pair in parts.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
            assert!(!pair[0].is_empty());
            assert!(pair[0].len() <= config.max_size);
        }

        let average = bytes.len() as f64 / parts.len() as f64;
        assert!(
            (2048.0..=8192.0).contains(&average),
            "average={average}, chunks={}",
            parts.len()
        );
    }
}

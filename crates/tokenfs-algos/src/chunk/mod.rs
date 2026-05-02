//! Content-defined chunking primitives.
//!
//! This module provides allocation-free Gear/FastCDC-style boundary detection.
//! It does not perform file I/O and does not own chunk payloads.

pub mod recursive;

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

/// Aggregate quality statistics for a chunk stream.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChunkQuality {
    /// Number of chunks produced.
    pub chunks: usize,
    /// Total bytes covered by the chunk stream.
    pub total_bytes: usize,
    /// Smallest chunk length.
    pub min_len: usize,
    /// Largest chunk length.
    pub max_len: usize,
    /// Mean chunk length.
    pub mean_len: f64,
    /// Fraction of chunks smaller than the configured minimum.
    pub below_min_fraction: f64,
    /// Fraction of chunks larger than the configured maximum.
    pub above_max_fraction: f64,
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

impl Default for ChunkQuality {
    fn default() -> Self {
        Self {
            chunks: 0,
            total_bytes: 0,
            min_len: 0,
            max_len: 0,
            mean_len: 0.0,
            below_min_fraction: 0.0,
            above_max_fraction: 0.0,
        }
    }
}

/// Failure modes for the fallible `ChunkConfig` constructors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChunkConfigError {
    /// `avg_size` exceeds `CHUNK_AVG_SIZE_CAP`. The infallible
    /// constructors saturate; the `try_*` variants surface this.
    AvgSizeOverflow {
        /// Caller-supplied avg_size.
        requested: usize,
        /// Hard cap before saturation kicks in.
        cap: usize,
    },
}

impl core::fmt::Display for ChunkConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AvgSizeOverflow { requested, cap } => write!(
                f,
                "ChunkConfig avg_size {requested} exceeds the saturation cap {cap}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ChunkConfigError {}

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

/// Largest `avg_size` accepted before the constructor saturates rather
/// than overflowing. Set to the largest power of two whose `× 4`
/// expansion still lives within `usize`. `next_power_of_two()` panics
/// in debug + wraps in release on inputs `> usize::MAX / 2`, and
/// `avg_size * 4` wraps on inputs `> usize::MAX / 4`; saturating both
/// keeps the public constructors panic-free for any user-controlled
/// input. On 64-bit platforms this is `1 << 62` ≈ 4.6 EiB.
pub const CHUNK_AVG_SIZE_CAP: usize = 1 << (usize::BITS as usize - 2);

/// Saturating equivalent of `n.next_power_of_two()` — returns
/// `1 << (usize::BITS - 1)` (the largest power of two representable
/// in a `usize`) instead of panicking/wrapping when `n > 1 << (BITS - 1)`.
const fn saturating_next_power_of_two(n: usize) -> usize {
    let max_pot = 1_usize << (usize::BITS as usize - 1);
    if n > max_pot {
        max_pot
    } else {
        n.next_power_of_two()
    }
}

impl ChunkConfig {
    /// Creates a conservative configuration from an average chunk size.
    ///
    /// Any user-controlled input is accepted: oversized values saturate
    /// to `CHUNK_AVG_SIZE_CAP` before the next-power-of-two + `* 4`
    /// arithmetic, so this constructor cannot panic or wrap. Pair with
    /// [`Self::try_new`] in kernel-adjacent code that wants an explicit
    /// overflow signal.
    #[must_use]
    pub fn new(avg_size: usize) -> Self {
        let avg_size = saturating_next_power_of_two(avg_size.clamp(64, CHUNK_AVG_SIZE_CAP));
        Self {
            min_size: (avg_size / 4).max(16),
            avg_size,
            max_size: avg_size.saturating_mul(4).max(avg_size),
        }
    }

    /// Fallible variant of [`Self::new`]: returns `Err` when
    /// `avg_size > CHUNK_AVG_SIZE_CAP`. Use in kernel-adjacent
    /// callers that need to reject hostile sizes explicitly.
    pub fn try_new(avg_size: usize) -> Result<Self, ChunkConfigError> {
        if avg_size > CHUNK_AVG_SIZE_CAP {
            return Err(ChunkConfigError::AvgSizeOverflow {
                requested: avg_size,
                cap: CHUNK_AVG_SIZE_CAP,
            });
        }
        Ok(Self::new(avg_size))
    }

    /// Creates a configuration with explicit min/avg/max sizes.
    ///
    /// Saturating semantics match [`Self::new`]: oversized values
    /// are clamped before any arithmetic. Pair with [`Self::try_with_sizes`]
    /// to surface overflow as an error.
    #[must_use]
    pub fn with_sizes(min_size: usize, avg_size: usize, max_size: usize) -> Self {
        let avg_size = saturating_next_power_of_two(avg_size.clamp(1, CHUNK_AVG_SIZE_CAP));
        let min_size = min_size.min(max_size).min(avg_size).max(1);
        let max_size = max_size.max(min_size).max(avg_size);
        Self {
            min_size,
            avg_size,
            max_size,
        }
    }

    /// Fallible variant of [`Self::with_sizes`].
    pub fn try_with_sizes(
        min_size: usize,
        avg_size: usize,
        max_size: usize,
    ) -> Result<Self, ChunkConfigError> {
        if avg_size > CHUNK_AVG_SIZE_CAP {
            return Err(ChunkConfigError::AvgSizeOverflow {
                requested: avg_size,
                cap: CHUNK_AVG_SIZE_CAP,
            });
        }
        Ok(Self::with_sizes(min_size, avg_size, max_size))
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

/// Summarizes chunk quality for a produced chunk stream.
#[must_use]
pub fn summarize_chunk_quality<I>(chunks: I, min_size: usize, max_size: usize) -> ChunkQuality
where
    I: IntoIterator<Item = Chunk>,
{
    let mut count = 0_usize;
    let mut total = 0_usize;
    let mut min_len = usize::MAX;
    let mut max_len = 0_usize;
    let mut below_min = 0_usize;
    let mut above_max = 0_usize;

    for chunk in chunks {
        let len = chunk.len();
        count += 1;
        total = total.saturating_add(len);
        min_len = min_len.min(len);
        max_len = max_len.max(len);
        if len < min_size {
            below_min += 1;
        }
        if len > max_size {
            above_max += 1;
        }
    }

    if count == 0 {
        return ChunkQuality::default();
    }

    ChunkQuality {
        chunks: count,
        total_bytes: total,
        min_len,
        max_len,
        mean_len: total as f64 / count as f64,
        below_min_fraction: below_min as f64 / count as f64,
        above_max_fraction: above_max as f64 / count as f64,
    }
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
#[allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.
mod tests {
    use super::{
        CHUNK_AVG_SIZE_CAP, ChunkConfig, ChunkConfigError, FastCdcConfig, chunks, fastcdc_chunks,
        fastcdc_find_boundary, find_boundary, gear_hash, summarize_chunk_quality,
    };
    // `Vec` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    #[test]
    fn chunk_config_new_saturates_on_extreme_avg_size() {
        // Pre-fix `avg_size.next_power_of_two()` panicked in debug
        // and `avg_size * 4` wrapped on usize::MAX. Saturating
        // semantics keep the constructor panic-free for any input.
        let cfg = ChunkConfig::new(usize::MAX);
        assert!(cfg.avg_size <= CHUNK_AVG_SIZE_CAP);
        assert!(cfg.max_size >= cfg.avg_size);
        assert!(cfg.min_size >= 16);
    }

    #[test]
    fn chunk_config_with_sizes_saturates() {
        let cfg = ChunkConfig::with_sizes(0, usize::MAX, usize::MAX);
        assert!(cfg.avg_size <= CHUNK_AVG_SIZE_CAP);
        assert!(cfg.max_size >= cfg.avg_size);
    }

    #[test]
    fn chunk_config_try_new_rejects_overflow() {
        let err = ChunkConfig::try_new(usize::MAX).unwrap_err();
        assert!(matches!(err, ChunkConfigError::AvgSizeOverflow { .. }));
        // Sane inputs still construct.
        assert!(ChunkConfig::try_new(8 * 1024).is_ok());
    }

    #[test]
    fn chunk_config_try_with_sizes_rejects_overflow() {
        let err = ChunkConfig::try_with_sizes(0, usize::MAX, usize::MAX).unwrap_err();
        assert!(matches!(err, ChunkConfigError::AvgSizeOverflow { .. }));
        assert!(ChunkConfig::try_with_sizes(1024, 4096, 8192).is_ok());
    }

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

    #[test]
    fn chunk_quality_summarizes_bounds() {
        let bytes = (0_usize..65_536)
            .map(|i| (i.wrapping_mul(31) ^ (i >> 2)) as u8)
            .collect::<Vec<_>>();
        let config = ChunkConfig::with_sizes(256, 1024, 4096);
        let quality =
            summarize_chunk_quality(chunks(&bytes, config), config.min_size, config.max_size);

        assert!(quality.chunks > 0);
        assert_eq!(quality.total_bytes, bytes.len());
        assert_eq!(quality.above_max_fraction, 0.0);
    }
}

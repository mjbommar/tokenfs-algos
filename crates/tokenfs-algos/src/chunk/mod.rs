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

/// Failure modes for the fallible `ChunkConfig` / `FastCdcConfig` constructors.
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
    /// `min_size` was zero. Zero-progress iterators emit chunks of
    /// length `0` indefinitely, a CPU/memory DoS hazard. The infallible
    /// constructors clamp to `1`; `try_*` rejects.
    ZeroMin,
    /// `avg_size` was zero. The mask derived from `avg_size`
    /// (`avg_size.next_power_of_two() - 1`) becomes degenerate at zero
    /// and the boundary scan terminates with `bytes: 0`. The infallible
    /// constructors clamp to a sane floor; `try_*` rejects.
    ZeroAvg,
    /// `max_size` was zero. Zero-progress iterators emit chunks of
    /// length `0` indefinitely. The infallible constructors clamp to at
    /// least `avg_size`; `try_*` rejects.
    ZeroMax,
    /// `min_size > avg_size` violates the size ordering. The infallible
    /// constructors silently re-clamp; `try_*` rejects so kernel callers
    /// catch the misconfiguration up-front.
    MinExceedsAvg {
        /// Caller-supplied min_size.
        min: usize,
        /// Caller-supplied avg_size (post-saturation).
        avg: usize,
    },
    /// `avg_size > max_size` violates the size ordering. The infallible
    /// constructors silently re-clamp; `try_*` rejects so kernel callers
    /// catch the misconfiguration up-front.
    AvgExceedsMax {
        /// Caller-supplied avg_size (post-saturation).
        avg: usize,
        /// Caller-supplied max_size.
        max: usize,
    },
}

impl core::fmt::Display for ChunkConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AvgSizeOverflow { requested, cap } => write!(
                f,
                "ChunkConfig avg_size {requested} exceeds the saturation cap {cap}"
            ),
            Self::ZeroMin => f.write_str("ChunkConfig min_size must be > 0"),
            Self::ZeroAvg => f.write_str("ChunkConfig avg_size must be > 0"),
            Self::ZeroMax => f.write_str("ChunkConfig max_size must be > 0"),
            Self::MinExceedsAvg { min, avg } => write!(
                f,
                "ChunkConfig min_size {min} exceeds avg_size {avg} (post-saturation)"
            ),
            Self::AvgExceedsMax { avg, max } => write!(
                f,
                "ChunkConfig avg_size {avg} (post-saturation) exceeds max_size {max}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ChunkConfigError {}

/// Configuration for Gear/FastCDC-style chunking.
///
/// **Construction.** Always build a `ChunkConfig` through one of the
/// constructors ([`Self::new`], [`Self::try_new`], [`Self::with_sizes`],
/// [`Self::try_with_sizes`]). Direct struct-literal construction
/// (`ChunkConfig { min_size: 0, avg_size: 0, max_size: 0 }`) bypasses
/// the size validation the constructors enforce and can lead to
/// zero-progress iterators that emit empty chunks indefinitely
/// (CPU/memory DoS in kernel/FUSE callers).
///
/// The fields are kept `pub` for back-compatibility with read-only
/// inspection (benches, tests, examples, fuzz harnesses already match
/// against them). The iterator hot path holds a defensive runtime
/// progress guard so even a hand-built zero-config still terminates
/// cleanly, but kernel-adjacent callers SHOULD prefer
/// [`Self::try_with_sizes`] to fail-fast on misconfiguration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkConfig {
    /// Minimum chunk size in bytes. Must be `>= 1` for the iterator to
    /// make progress. The infallible constructors clamp; the `try_*`
    /// constructors reject `0` with [`ChunkConfigError::ZeroMin`].
    pub min_size: usize,
    /// Target average chunk size in bytes. Rounded to a power-of-two
    /// mask. Must be `>= 1` and `>= min_size`. The infallible
    /// constructors clamp; `try_*` rejects `0`
    /// ([`ChunkConfigError::ZeroAvg`]) and inversion
    /// ([`ChunkConfigError::MinExceedsAvg`]).
    pub avg_size: usize,
    /// Maximum chunk size in bytes. Must be `>= 1` and `>= avg_size`.
    /// The infallible constructors clamp; `try_*` rejects `0`
    /// ([`ChunkConfigError::ZeroMax`]) and inversion
    /// ([`ChunkConfigError::AvgExceedsMax`]).
    pub max_size: usize,
}

/// Configuration for normalized FastCDC-style chunking.
///
/// **Construction.** Same posture as [`ChunkConfig`]: build through
/// constructors only. Hand-built zero values will be neutralized by the
/// runtime progress guard inside [`FastCdcChunker`], but kernel callers
/// SHOULD use [`Self::try_with_sizes`] to fail-fast.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FastCdcConfig {
    /// Minimum chunk size in bytes. See [`ChunkConfig::min_size`].
    pub min_size: usize,
    /// Target average chunk size. Rounded to a power-of-two mask.
    /// See [`ChunkConfig::avg_size`].
    pub avg_size: usize,
    /// Maximum chunk size. See [`ChunkConfig::max_size`].
    pub max_size: usize,
    /// Number of mask bits used for pre/post-average normalization.
    /// Values above 8 are clamped by the constructors.
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

    /// Fallible variant of [`Self::new`]: returns `Err` for any input
    /// the infallible constructor would silently clamp.
    ///
    /// Rejects:
    /// - `avg_size == 0` ([`ChunkConfigError::ZeroAvg`]) — would yield
    ///   a degenerate mask.
    /// - `avg_size > CHUNK_AVG_SIZE_CAP` ([`ChunkConfigError::AvgSizeOverflow`])
    ///   — would saturate.
    ///
    /// Use in kernel-adjacent callers that need to reject hostile or
    /// misconfigured sizes explicitly rather than receive a clamped
    /// (but functional) config.
    pub fn try_new(avg_size: usize) -> Result<Self, ChunkConfigError> {
        if avg_size == 0 {
            return Err(ChunkConfigError::ZeroAvg);
        }
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

    /// Fallible variant of [`Self::with_sizes`]: returns `Err` for any
    /// input the infallible constructor would silently clamp.
    ///
    /// Rejects:
    /// - `min_size == 0` ([`ChunkConfigError::ZeroMin`])
    /// - `avg_size == 0` ([`ChunkConfigError::ZeroAvg`])
    /// - `max_size == 0` ([`ChunkConfigError::ZeroMax`])
    /// - `avg_size > CHUNK_AVG_SIZE_CAP` ([`ChunkConfigError::AvgSizeOverflow`])
    /// - `min_size > avg_size` ([`ChunkConfigError::MinExceedsAvg`]) —
    ///   evaluated against `avg_size` post-saturation.
    /// - `avg_size > max_size` ([`ChunkConfigError::AvgExceedsMax`]) —
    ///   evaluated against `avg_size` post-saturation.
    ///
    /// Use in kernel-adjacent callers that need to fail-fast on
    /// misconfiguration rather than rely on the silent re-clamp.
    pub fn try_with_sizes(
        min_size: usize,
        avg_size: usize,
        max_size: usize,
    ) -> Result<Self, ChunkConfigError> {
        if min_size == 0 {
            return Err(ChunkConfigError::ZeroMin);
        }
        if avg_size == 0 {
            return Err(ChunkConfigError::ZeroAvg);
        }
        if max_size == 0 {
            return Err(ChunkConfigError::ZeroMax);
        }
        if avg_size > CHUNK_AVG_SIZE_CAP {
            return Err(ChunkConfigError::AvgSizeOverflow {
                requested: avg_size,
                cap: CHUNK_AVG_SIZE_CAP,
            });
        }
        // Validate ordering against the post-saturation avg_size so the
        // error message matches the value the constructor would actually
        // pin down.
        let saturated_avg = saturating_next_power_of_two(avg_size);
        if min_size > saturated_avg {
            return Err(ChunkConfigError::MinExceedsAvg {
                min: min_size,
                avg: saturated_avg,
            });
        }
        if saturated_avg > max_size {
            return Err(ChunkConfigError::AvgExceedsMax {
                avg: saturated_avg,
                max: max_size,
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

    /// Fallible variant of [`Self::new`]: returns `Err` under the same
    /// conditions as [`ChunkConfig::try_new`]. Use in kernel-adjacent
    /// callers that need to reject hostile sizes explicitly.
    pub fn try_new(avg_size: usize) -> Result<Self, ChunkConfigError> {
        let base = ChunkConfig::try_new(avg_size)?;
        Ok(Self {
            min_size: base.min_size,
            avg_size: base.avg_size,
            max_size: base.max_size,
            normalization_level: 1,
        })
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

    /// Fallible variant of [`Self::with_sizes`]: returns `Err` under the
    /// same conditions as [`ChunkConfig::try_with_sizes`].
    pub fn try_with_sizes(
        min_size: usize,
        avg_size: usize,
        max_size: usize,
    ) -> Result<Self, ChunkConfigError> {
        let base = ChunkConfig::try_with_sizes(min_size, avg_size, max_size)?;
        Ok(Self {
            min_size: base.min_size,
            avg_size: base.avg_size,
            max_size: base.max_size,
            normalization_level: 1,
        })
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

        // Zero-progress guard (audit-R8 #2): a hand-built `ChunkConfig`
        // that bypasses the constructors (e.g. `min_size = avg_size =
        // max_size = 0`) makes `find_boundary_with_state` return
        // `bytes: 0`, which would otherwise emit empty chunks forever
        // and never advance `position` — a CPU/memory DoS hazard for
        // kernel/FUSE callers. Treat zero forward progress as
        // end-of-stream and stop iterating cleanly.
        if boundary.bytes == 0 {
            self.position = self.bytes.len();
            return None;
        }

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

        // Zero-progress guard (audit-R8 #2): mirror of the `Chunker`
        // guard. A bypassed-constructor zero-config (`min_size =
        // avg_size = max_size = 0`) would emit empty chunks indefinitely
        // — terminate cleanly instead.
        if boundary.bytes == 0 {
            self.position = self.bytes.len();
            return None;
        }

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
    // `Vec` and the `vec!` macro are not in the no-std prelude; alias
    // both from `alloc` for the alloc-only build (audit-R6 #164,
    // audit-R8 followup).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
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
        // After audit-R8 #2 the zero-field validation triggers first, so
        // probe overflow with a non-zero min_size.
        let err = ChunkConfig::try_with_sizes(1, usize::MAX, usize::MAX).unwrap_err();
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

    // -----------------------------------------------------------------
    // audit-R8 #2: zero-config validation + iterator progress guard
    // -----------------------------------------------------------------

    #[test]
    fn chunk_config_try_new_rejects_zero_avg() {
        // try_new now distinguishes ZeroAvg from AvgSizeOverflow — the
        // infallible `new()` would silently clamp to 64.
        let err = ChunkConfig::try_new(0).unwrap_err();
        assert!(matches!(err, ChunkConfigError::ZeroAvg));
    }

    #[test]
    fn chunk_config_try_with_sizes_rejects_zero_min() {
        let err = ChunkConfig::try_with_sizes(0, 4096, 8192).unwrap_err();
        assert!(matches!(err, ChunkConfigError::ZeroMin));
    }

    #[test]
    fn chunk_config_try_with_sizes_rejects_zero_avg() {
        let err = ChunkConfig::try_with_sizes(1024, 0, 8192).unwrap_err();
        assert!(matches!(err, ChunkConfigError::ZeroAvg));
    }

    #[test]
    fn chunk_config_try_with_sizes_rejects_zero_max() {
        let err = ChunkConfig::try_with_sizes(1024, 4096, 0).unwrap_err();
        assert!(matches!(err, ChunkConfigError::ZeroMax));
    }

    #[test]
    fn chunk_config_try_with_sizes_rejects_zero_config() {
        // The full-zero misconfiguration that produces zero-progress
        // iterators if it bypasses the constructor — must be rejected.
        let err = ChunkConfig::try_with_sizes(0, 0, 0).unwrap_err();
        // Order of checks: ZeroMin fires first.
        assert!(matches!(err, ChunkConfigError::ZeroMin));
    }

    #[test]
    fn chunk_config_try_with_sizes_rejects_min_exceeds_avg() {
        // min_size > avg_size (post-saturation). avg_size=1024 is already
        // a power of two, so post-saturation it stays 1024. min=2048 > 1024.
        let err = ChunkConfig::try_with_sizes(2048, 1024, 4096).unwrap_err();
        assert!(matches!(
            err,
            ChunkConfigError::MinExceedsAvg {
                min: 2048,
                avg: 1024
            }
        ));
    }

    #[test]
    fn chunk_config_try_with_sizes_rejects_avg_exceeds_max() {
        // avg_size > max_size (post-saturation). avg=4096 > max=2048.
        let err = ChunkConfig::try_with_sizes(512, 4096, 2048).unwrap_err();
        assert!(matches!(
            err,
            ChunkConfigError::AvgExceedsMax {
                avg: 4096,
                max: 2048
            }
        ));
    }

    #[test]
    fn chunk_config_try_with_sizes_accepts_valid_ordering() {
        let cfg = ChunkConfig::try_with_sizes(1024, 4096, 16 * 1024).unwrap();
        assert_eq!(cfg.min_size, 1024);
        assert_eq!(cfg.avg_size, 4096);
        assert_eq!(cfg.max_size, 16 * 1024);
    }

    #[test]
    fn fastcdc_config_try_new_rejects_zero_avg() {
        // FastCdcConfig now has its own try_new mirroring ChunkConfig's.
        let err = FastCdcConfig::try_new(0).unwrap_err();
        assert!(matches!(err, ChunkConfigError::ZeroAvg));
        assert!(FastCdcConfig::try_new(8 * 1024).is_ok());
    }

    #[test]
    fn fastcdc_config_try_with_sizes_rejects_zero_config() {
        let err = FastCdcConfig::try_with_sizes(0, 0, 0).unwrap_err();
        assert!(matches!(err, ChunkConfigError::ZeroMin));
        assert!(FastCdcConfig::try_with_sizes(1024, 4096, 16 * 1024).is_ok());
    }

    #[test]
    fn chunker_zero_config_progress_guard_terminates() {
        // Bypass the constructors with a hand-built zero-config and
        // verify the iterator's progress guard terminates instead of
        // looping forever emitting empty chunks. A non-empty input is
        // mandatory because the iterator's own `position >= bytes.len()`
        // early-return would otherwise mask the guard.
        let bytes = vec![0_u8; 4096];
        let bypassed = ChunkConfig {
            min_size: 0,
            avg_size: 0,
            max_size: 0,
        };
        let mut iter = chunks(&bytes, bypassed);
        // Must return None on the very first call — no zero-length chunk
        // should be emitted.
        assert!(
            iter.next().is_none(),
            "Chunker progress guard must terminate on zero-config bypass"
        );
        // And subsequent calls remain None.
        assert!(iter.next().is_none());
    }

    #[test]
    fn fastcdc_chunker_zero_config_progress_guard_terminates() {
        let bytes = vec![0_u8; 4096];
        let bypassed = FastCdcConfig {
            min_size: 0,
            avg_size: 0,
            max_size: 0,
            normalization_level: 0,
        };
        let mut iter = fastcdc_chunks(&bytes, bypassed);
        assert!(
            iter.next().is_none(),
            "FastCdcChunker progress guard must terminate on zero-config bypass"
        );
        assert!(iter.next().is_none());
    }

    #[test]
    fn chunker_zero_config_collect_does_not_oom() {
        // Belt-and-suspenders: `.collect()` would allocate forever on a
        // zero-progress iterator. The guard makes the collected vec empty.
        let bytes = vec![0_u8; 4096];
        let bypassed = ChunkConfig {
            min_size: 0,
            avg_size: 0,
            max_size: 0,
        };
        let parts: Vec<_> = chunks(&bytes, bypassed).collect();
        assert!(parts.is_empty());
    }

    #[test]
    fn fastcdc_chunker_zero_config_collect_does_not_oom() {
        let bytes = vec![0_u8; 4096];
        let bypassed = FastCdcConfig {
            min_size: 0,
            avg_size: 0,
            max_size: 0,
            normalization_level: 0,
        };
        let parts: Vec<_> = fastcdc_chunks(&bytes, bypassed).collect();
        assert!(parts.is_empty());
    }
}

//! Experimental histogram kernels for benchmarking.
//!
//! This module is available only with the `bench-internals` feature. It is not
//! part of the stable public API; it exists so Criterion benches and parity
//! tests can compare alternative implementation strategies without exposing raw
//! primitives as normal user-facing APIs.

use core::fmt;

use crate::{histogram::ByteHistogram, primitives::histogram_scalar};

/// Experimental byte-histogram kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HistogramKernel {
    /// Current direct scalar implementation with one `[u64; 256]` table.
    DirectU64,
    /// Local `[u32; 256]` table, reduced into the public `u64` histogram.
    LocalU32,
    /// Four independent `[u32; 256]` stripes, reduced at the end.
    Stripe4U32,
    /// Eight independent `[u32; 256]` stripes, reduced at the end.
    Stripe8U32,
    /// Run-length scan that adds whole runs to one counter at a time.
    RunLengthU64,
    /// AVX2 palette counter with exact scalar fallback.
    Avx2PaletteU32,
    /// Adaptive classifier using the first 1 KiB as a sample.
    AdaptivePrefix1K,
    /// Adaptive classifier using the first 4 KiB as a sample.
    AdaptivePrefix4K,
    /// Adaptive classifier using four spread-out 1 KiB samples.
    AdaptiveSpread4K,
    /// Conservative adaptive classifier that only diverts obvious huge runs.
    AdaptiveRunSentinel4K,
    /// Adaptive classifier applied independently to each 64 KiB chunk.
    AdaptiveChunked64K,
    /// Adaptive sequential planner that updates the choice at 64 KiB chunk boundaries.
    AdaptiveSequentialOnline64K,
    /// Adaptive file-level planner that samples once and applies the choice to all chunks.
    AdaptiveFileCached64K,
    /// Low-entropy fast path that aggressively promotes obvious runs.
    AdaptiveLowEntropyFast,
    /// ASCII/text-biased path that avoids extra sampling once text dominance is clear.
    AdaptiveAsciiFast,
    /// High-entropy path that skips specialized logic when the sample looks random.
    AdaptiveHighEntropySkip,
    /// Meso-pattern detector tuned for block-palette-like files.
    AdaptiveMesoDetector,
}

impl HistogramKernel {
    /// Returns all experimental kernels in a stable order.
    #[must_use]
    pub const fn all() -> [Self; 17] {
        [
            Self::DirectU64,
            Self::LocalU32,
            Self::Stripe4U32,
            Self::Stripe8U32,
            Self::RunLengthU64,
            Self::Avx2PaletteU32,
            Self::AdaptivePrefix1K,
            Self::AdaptivePrefix4K,
            Self::AdaptiveSpread4K,
            Self::AdaptiveRunSentinel4K,
            Self::AdaptiveChunked64K,
            Self::AdaptiveSequentialOnline64K,
            Self::AdaptiveFileCached64K,
            Self::AdaptiveLowEntropyFast,
            Self::AdaptiveAsciiFast,
            Self::AdaptiveHighEntropySkip,
            Self::AdaptiveMesoDetector,
        ]
    }

    /// Returns only the adaptive experimental kernels in a stable order.
    #[must_use]
    pub const fn adaptive() -> [Self; 11] {
        [
            Self::AdaptivePrefix1K,
            Self::AdaptivePrefix4K,
            Self::AdaptiveSpread4K,
            Self::AdaptiveRunSentinel4K,
            Self::AdaptiveChunked64K,
            Self::AdaptiveSequentialOnline64K,
            Self::AdaptiveFileCached64K,
            Self::AdaptiveLowEntropyFast,
            Self::AdaptiveAsciiFast,
            Self::AdaptiveHighEntropySkip,
            Self::AdaptiveMesoDetector,
        ]
    }

    /// Returns a short stable identifier for benchmark names.
    #[must_use]
    pub const fn id(self) -> &'static str {
        match self {
            Self::DirectU64 => "direct-u64",
            Self::LocalU32 => "local-u32",
            Self::Stripe4U32 => "stripe4-u32",
            Self::Stripe8U32 => "stripe8-u32",
            Self::RunLengthU64 => "run-length-u64",
            Self::Avx2PaletteU32 => "avx2-palette-u32",
            Self::AdaptivePrefix1K => "adaptive-prefix-1k",
            Self::AdaptivePrefix4K => "adaptive-prefix-4k",
            Self::AdaptiveSpread4K => "adaptive-spread-4k",
            Self::AdaptiveRunSentinel4K => "adaptive-run-sentinel-4k",
            Self::AdaptiveChunked64K => "adaptive-chunked-64k",
            Self::AdaptiveSequentialOnline64K => "adaptive-sequential-online-64k",
            Self::AdaptiveFileCached64K => "adaptive-file-cached-64k",
            Self::AdaptiveLowEntropyFast => "adaptive-low-entropy-fast",
            Self::AdaptiveAsciiFast => "adaptive-ascii-fast",
            Self::AdaptiveHighEntropySkip => "adaptive-high-entropy-skip",
            Self::AdaptiveMesoDetector => "adaptive-meso-detector",
        }
    }
}

impl fmt::Display for HistogramKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

/// Builds a [`ByteHistogram`] with the selected experimental kernel.
#[must_use]
pub fn byte_histogram_with_kernel(block: &[u8], kernel: HistogramKernel) -> ByteHistogram {
    let mut histogram = ByteHistogram::new();
    add_block_with_kernel(block, &mut histogram, kernel);
    histogram
}

/// Adds `block` to `histogram` with the selected experimental kernel.
pub fn add_block_with_kernel(block: &[u8], histogram: &mut ByteHistogram, kernel: HistogramKernel) {
    let counts = histogram.counts_mut_for_primitives();

    match kernel {
        HistogramKernel::DirectU64 => histogram_scalar::add_block_direct_u64(block, counts),
        HistogramKernel::LocalU32 => histogram_scalar::add_block_local_u32(block, counts),
        HistogramKernel::Stripe4U32 => histogram_scalar::add_block_striped_u32::<4>(block, counts),
        HistogramKernel::Stripe8U32 => histogram_scalar::add_block_striped_u32::<8>(block, counts),
        HistogramKernel::RunLengthU64 => histogram_scalar::add_block_run_length_u64(block, counts),
        HistogramKernel::Avx2PaletteU32 => {
            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                if crate::histogram::kernels::avx2_palette_u32::is_available() {
                    // SAFETY: availability was checked immediately above.
                    unsafe {
                        crate::primitives::histogram_avx2::add_block_palette_u32(block, counts);
                    }
                } else {
                    histogram_scalar::add_block_local_u32(block, counts);
                }
            }
            #[cfg(not(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64"))))]
            {
                histogram_scalar::add_block_local_u32(block, counts);
            }
        }
        HistogramKernel::AdaptivePrefix1K => {
            histogram_scalar::add_block_adaptive_prefix::<1024>(block, counts);
        }
        HistogramKernel::AdaptivePrefix4K => {
            histogram_scalar::add_block_adaptive_prefix::<4096>(block, counts);
        }
        HistogramKernel::AdaptiveSpread4K => {
            histogram_scalar::add_block_adaptive_spread_4k(block, counts);
        }
        HistogramKernel::AdaptiveRunSentinel4K => {
            histogram_scalar::add_block_adaptive_run_sentinel_4k(block, counts);
        }
        HistogramKernel::AdaptiveChunked64K => {
            histogram_scalar::add_block_adaptive_chunked::<65_536>(block, counts);
        }
        HistogramKernel::AdaptiveSequentialOnline64K => {
            histogram_scalar::add_block_adaptive_sequential_online::<65_536>(block, counts);
        }
        HistogramKernel::AdaptiveFileCached64K => {
            histogram_scalar::add_block_adaptive_file_cached::<65_536>(block, counts);
        }
        HistogramKernel::AdaptiveLowEntropyFast => {
            histogram_scalar::add_block_adaptive_low_entropy_fast(block, counts);
        }
        HistogramKernel::AdaptiveAsciiFast => {
            histogram_scalar::add_block_adaptive_ascii_fast(block, counts);
        }
        HistogramKernel::AdaptiveHighEntropySkip => {
            histogram_scalar::add_block_adaptive_high_entropy_skip(block, counts);
        }
        HistogramKernel::AdaptiveMesoDetector => {
            histogram_scalar::add_block_adaptive_meso_detector(block, counts);
        }
    }

    histogram.add_to_total_for_primitives(block.len() as u64);
}

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
}

impl HistogramKernel {
    /// Returns all experimental kernels in a stable order.
    #[must_use]
    pub const fn all() -> [Self; 10] {
        [
            Self::DirectU64,
            Self::LocalU32,
            Self::Stripe4U32,
            Self::Stripe8U32,
            Self::RunLengthU64,
            Self::AdaptivePrefix1K,
            Self::AdaptivePrefix4K,
            Self::AdaptiveSpread4K,
            Self::AdaptiveRunSentinel4K,
            Self::AdaptiveChunked64K,
        ]
    }

    /// Returns only the adaptive experimental kernels in a stable order.
    #[must_use]
    pub const fn adaptive() -> [Self; 5] {
        [
            Self::AdaptivePrefix1K,
            Self::AdaptivePrefix4K,
            Self::AdaptiveSpread4K,
            Self::AdaptiveRunSentinel4K,
            Self::AdaptiveChunked64K,
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
            Self::AdaptivePrefix1K => "adaptive-prefix-1k",
            Self::AdaptivePrefix4K => "adaptive-prefix-4k",
            Self::AdaptiveSpread4K => "adaptive-spread-4k",
            Self::AdaptiveRunSentinel4K => "adaptive-run-sentinel-4k",
            Self::AdaptiveChunked64K => "adaptive-chunked-64k",
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
    }

    histogram.add_to_total_for_primitives(block.len() as u64);
}

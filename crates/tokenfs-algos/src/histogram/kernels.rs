//! Pinned byte-histogram kernels.
//!
//! These functions bypass the planner. They are intended for reproducible
//! experiments, paper calibration, regression debugging, and forensic
//! comparisons where the selected kernel must not change implicitly.

use crate::{histogram::ByteHistogram, primitives::histogram_scalar};

macro_rules! kernel_module {
    ($module:ident, $add:path, $doc:literal) => {
        #[doc = $doc]
        pub mod $module {
            use super::{ByteHistogram, histogram_scalar};

            /// Builds a byte histogram with this pinned kernel.
            #[must_use]
            pub fn block(bytes: &[u8]) -> ByteHistogram {
                let mut histogram = ByteHistogram::new();
                add_block(bytes, &mut histogram);
                histogram
            }

            /// Adds bytes into an existing histogram with this pinned kernel.
            pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
                $add(bytes, histogram.counts_mut_for_primitives());
                histogram.add_to_total_for_primitives(bytes.len() as u64);
            }
        }
    };
}

kernel_module!(
    direct_u64,
    histogram_scalar::add_block_direct_u64,
    "Direct scalar counting into one public `u64` table."
);

kernel_module!(
    local_u32,
    histogram_scalar::add_block_local_u32,
    "Private `u32` table reduced into public `u64` counts."
);

/// Four private `u32` stripes reduced into public `u64` counts.
pub mod stripe4_u32 {
    use super::{ByteHistogram, histogram_scalar};

    /// Builds a byte histogram with this pinned kernel.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        histogram_scalar::add_block_striped_u32::<4>(bytes, histogram.counts_mut_for_primitives());
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

/// Eight private `u32` stripes reduced into public `u64` counts.
pub mod stripe8_u32 {
    use super::{ByteHistogram, histogram_scalar};

    /// Builds a byte histogram with this pinned kernel.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        histogram_scalar::add_block_striped_u32::<8>(bytes, histogram.counts_mut_for_primitives());
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

kernel_module!(
    run_length_u64,
    histogram_scalar::add_block_run_length_u64,
    "Run-length counting that increments once per equal-byte run."
);

/// Adaptive classifier using the first 1 KiB as a sample.
pub mod adaptive_prefix_1k {
    use super::{ByteHistogram, histogram_scalar};

    /// Builds a byte histogram with this pinned kernel.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        histogram_scalar::add_block_adaptive_prefix::<1024>(
            bytes,
            histogram.counts_mut_for_primitives(),
        );
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

/// Adaptive classifier using the first 4 KiB as a sample.
pub mod adaptive_prefix_4k {
    use super::{ByteHistogram, histogram_scalar};

    /// Builds a byte histogram with this pinned kernel.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        histogram_scalar::add_block_adaptive_prefix::<4096>(
            bytes,
            histogram.counts_mut_for_primitives(),
        );
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

kernel_module!(
    adaptive_spread_4k,
    histogram_scalar::add_block_adaptive_spread_4k,
    "Adaptive classifier using four 1 KiB samples spread across the block."
);

kernel_module!(
    adaptive_run_sentinel_4k,
    histogram_scalar::add_block_adaptive_run_sentinel_4k,
    "Conservative adaptive classifier that diverts only obvious long runs."
);

kernel_module!(
    adaptive_low_entropy_fast,
    histogram_scalar::add_block_adaptive_low_entropy_fast,
    "Low-entropy fast path that aggressively promotes obvious long runs."
);

kernel_module!(
    adaptive_ascii_fast,
    histogram_scalar::add_block_adaptive_ascii_fast,
    "ASCII/text-biased path that avoids extra sampling once text dominance is clear."
);

kernel_module!(
    adaptive_high_entropy_skip,
    histogram_scalar::add_block_adaptive_high_entropy_skip,
    "High-entropy path that skips specialized logic when the sample looks random."
);

kernel_module!(
    adaptive_meso_detector,
    histogram_scalar::add_block_adaptive_meso_detector,
    "Meso-pattern detector tuned for block-palette-like files."
);

/// Adaptive classifier applied independently to each 64 KiB chunk.
pub mod adaptive_chunked_64k {
    use super::{ByteHistogram, histogram_scalar};

    /// Builds a byte histogram with this pinned kernel.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        histogram_scalar::add_block_adaptive_chunked::<65_536>(
            bytes,
            histogram.counts_mut_for_primitives(),
        );
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

/// Adaptive sequential planner that updates the choice at 64 KiB chunk boundaries.
pub mod adaptive_sequential_online_64k {
    use super::{ByteHistogram, histogram_scalar};

    /// Builds a byte histogram with this pinned kernel.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        histogram_scalar::add_block_adaptive_sequential_online::<65_536>(
            bytes,
            histogram.counts_mut_for_primitives(),
        );
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

/// Adaptive file-level planner that samples once and applies the choice to all chunks.
pub mod adaptive_file_cached_64k {
    use super::{ByteHistogram, histogram_scalar};

    /// Builds a byte histogram with this pinned kernel.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        histogram_scalar::add_block_adaptive_file_cached::<65_536>(
            bytes,
            histogram.counts_mut_for_primitives(),
        );
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

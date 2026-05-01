//! Paper-lineage compatibility aliases.
//!
//! Normal crate users should prefer product names such as [`crate::fingerprint`],
//! [`crate::selector`], and [`crate::sketch`]. This namespace keeps the F21/F22
//! and F23a labels available for calibration and replication code.

/// F21 selector compatibility aliases.
pub mod f21 {
    pub use crate::selector::{
        RepresentationHint, SelectorSignals, hint, signals, signals_from_parts,
    };
}

/// F22 fingerprint compatibility aliases.
pub mod f22 {
    pub use crate::fingerprint::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint};

    /// Computes the productized F22 block fingerprint.
    #[must_use]
    pub fn fingerprint_block(block: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
        crate::fingerprint::block(block)
    }

    /// Computes the productized F22 extent fingerprint.
    #[must_use]
    pub fn fingerprint_extent(bytes: &[u8]) -> ExtentFingerprint {
        crate::fingerprint::extent(bytes)
    }

    /// Pinned F22 scalar aliases.
    pub mod scalar {
        pub use crate::fingerprint::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint};

        /// Computes the pinned scalar F22 block fingerprint.
        #[must_use]
        pub fn fingerprint_block(block: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
            crate::fingerprint::kernels::scalar::block(block)
        }

        /// Computes the pinned scalar F22 extent fingerprint.
        #[must_use]
        pub fn fingerprint_extent(bytes: &[u8]) -> ExtentFingerprint {
            crate::fingerprint::kernels::scalar::extent(bytes)
        }
    }
}

/// F23a sketch compatibility aliases.
pub mod f23a {
    pub use crate::sketch::{
        CLog2Lut, CountMinSketch, MisraGries, c_log2_c, concentration_ratio_u32, crc32_hash4_bins,
        crc32c_u32, entropy_from_counts_u32, entropy_from_counts_u32_lut, top_k_coverage_u32,
    };
}

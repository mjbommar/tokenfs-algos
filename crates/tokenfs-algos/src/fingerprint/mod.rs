//! F22/content byte-stream fingerprints.
//!
//! The F21 paper prototype used `H1`, `H2`, `H3`, top-16 coverage,
//! run-length fraction, and byte-entropy skew. The F22 crate primitive keeps
//! the same cheap-feature spirit while replacing expensive `H3` with a
//! hardware-friendly CRC32-hashed 4-gram entropy estimate.

use crate::{
    dispatch::{KernelIsa, KernelStatefulness, PrimitiveFamily, WorkingSetClass},
    histogram::ByteHistogram,
    sketch,
};

/// F22 block size. A 64 KiB extent contains 256 such blocks.
pub const BLOCK_SIZE: usize = 256;

/// Number of hash bins for extent-level 4-gram entropy.
pub const QUAD_HASH_BINS: usize = 4096;

/// Number of hash bins for block-level 4-gram entropy.
pub const QUAD_HASH_BLOCK_BINS: usize = 256;

/// Compact per-block fingerprint.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(C)]
pub struct BlockFingerprint {
    /// H1 byte entropy, bits/byte scaled by 16.
    pub h1_q4: u8,
    /// Hashed H4 entropy estimate, bits/byte scaled by 16.
    pub h4_q4: u8,
    /// Number of equal-byte runs of length at least 4.
    pub rl_runs_ge4: u16,
    /// Top-4 byte coverage fraction scaled by 256.
    pub top4_coverage_q8: u8,
    /// Byte-class dominance bitmap.
    pub byte_class: u8,
    /// Reserved for stable 8-byte layout.
    pub reserved: u8,
}

const _: () = assert!(core::mem::size_of::<BlockFingerprint>() == 8);

/// Per-extent aggregate fingerprint.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ExtentFingerprint {
    /// H1 byte entropy in bits/byte.
    pub h1: f32,
    /// Hashed H4 entropy estimate in bits/byte.
    pub h4: f32,
    /// Fraction of bytes that belong to equal-byte runs of length at least 4.
    pub rl_fraction: f32,
    /// Fraction of bytes covered by the 16 most common byte values.
    pub top16_coverage: f32,
    /// `h1 / 8.0`, retained for F21/F22 calibration compatibility.
    pub byte_entropy_skew: f32,
}

/// Fingerprint kernel exposed for pinned benchmarks and reproducibility.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FingerprintKernel {
    /// Runtime-dispatched/default kernel.
    Auto,
    /// Portable scalar reference kernel.
    Scalar,
}

impl FingerprintKernel {
    /// Stable benchmark identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
        }
    }
}

/// Catalog metadata for one fingerprint kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FingerprintKernelInfo {
    /// Primitive family.
    pub family: PrimitiveFamily,
    /// Stable kernel identifier.
    pub kernel: FingerprintKernel,
    /// ISA/backend requirement.
    pub isa: KernelIsa,
    /// Working-set class.
    pub working_set: WorkingSetClass,
    /// State model.
    pub statefulness: KernelStatefulness,
    /// Block size this kernel is tuned around.
    pub block_bytes: usize,
    /// Whether the hot path allocates.
    pub hot_path_allocates: bool,
    /// Short description.
    pub description: &'static str,
}

const FINGERPRINT_KERNEL_CATALOG: [FingerprintKernelInfo; 2] = [
    FingerprintKernelInfo {
        family: PrimitiveFamily::Fingerprint,
        kernel: FingerprintKernel::Auto,
        isa: KernelIsa::RuntimeDispatch,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        block_bytes: BLOCK_SIZE,
        hot_path_allocates: false,
        description: "default fingerprint path; may use runtime-dispatched CRC32C hash bins",
    },
    FingerprintKernelInfo {
        family: PrimitiveFamily::Fingerprint,
        kernel: FingerprintKernel::Scalar,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        block_bytes: BLOCK_SIZE,
        hot_path_allocates: false,
        description: "portable scalar reference path for F22 calibration and parity",
    },
];

/// Returns catalog metadata for currently known fingerprint kernels.
#[must_use]
pub const fn kernel_catalog() -> &'static [FingerprintKernelInfo] {
    &FINGERPRINT_KERNEL_CATALOG
}

/// Pinned fingerprint kernels.
pub mod kernels {
    use super::{
        BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, block_auto, block_scalar, extent_auto,
        extent_scalar,
    };

    /// Runtime-dispatched/default fingerprint kernel.
    pub mod auto {
        use super::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, block_auto, extent_auto};

        /// Computes a compact F22/content fingerprint for one 256-byte block.
        #[must_use]
        pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
            block_auto(bytes)
        }

        /// Computes an aggregate F22/content fingerprint for any byte slice.
        #[must_use]
        pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
            extent_auto(bytes)
        }
    }

    /// Portable scalar reference fingerprint kernel.
    pub mod scalar {
        use super::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, block_scalar, extent_scalar};

        /// Computes a compact F22/content fingerprint for one 256-byte block.
        #[must_use]
        pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
            block_scalar(bytes)
        }

        /// Computes an aggregate F22/content fingerprint for any byte slice.
        #[must_use]
        pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
            extent_scalar(bytes)
        }
    }
}

/// Computes a compact F22/content fingerprint for one 256-byte block.
///
/// This is the ergonomic default and may use runtime-dispatched sub-primitives.
/// Use [`kernels::scalar::block`] when a pinned scalar reference is required.
#[must_use]
pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    kernels::auto::block(bytes)
}

fn block_auto(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    block_with_hash4(bytes, sketch::crc32_hash4_bins)
}

fn block_scalar(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    block_with_hash4(bytes, sketch::kernels::scalar::crc32_hash4_bins)
}

fn block_with_hash4(
    bytes: &[u8; BLOCK_SIZE],
    hash4_bins: fn(&[u8], &mut [u32; QUAD_HASH_BLOCK_BINS]),
) -> BlockFingerprint {
    let mut histogram = [0_u32; 256];
    for &byte in bytes {
        histogram[byte as usize] += 1;
    }

    let h1 = sketch::entropy_from_counts_u32(&histogram, BLOCK_SIZE as u64);
    let (rl_runs_ge4, _) = runlength(bytes);
    let h4 = if h1 >= 7.875 && rl_runs_ge4 == 0 {
        h1
    } else {
        let mut bins = [0_u32; QUAD_HASH_BLOCK_BINS];
        hash4_bins(bytes, &mut bins);
        sketch::entropy_from_counts_u32(&bins, (BLOCK_SIZE - 3) as u64)
    };

    BlockFingerprint {
        h1_q4: quantize_q4(h1),
        h4_q4: quantize_q4(h4),
        rl_runs_ge4,
        top4_coverage_q8: top_k_coverage_q8(&histogram, 4, BLOCK_SIZE as u32),
        byte_class: byte_class_bitmap(&histogram),
        reserved: 0,
    }
}

/// Computes an F22/content aggregate fingerprint for any byte slice.
///
/// This is the ergonomic default and may use runtime-dispatched sub-primitives.
/// Use [`kernels::scalar::extent`] when a pinned scalar reference is required.
#[must_use]
pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
    kernels::auto::extent(bytes)
}

fn extent_auto(bytes: &[u8]) -> ExtentFingerprint {
    extent_with_hash4(bytes, sketch::crc32_hash4_bins)
}

fn extent_scalar(bytes: &[u8]) -> ExtentFingerprint {
    extent_with_hash4(bytes, sketch::kernels::scalar::crc32_hash4_bins)
}

fn extent_with_hash4(
    bytes: &[u8],
    hash4_bins: fn(&[u8], &mut [u32; QUAD_HASH_BINS]),
) -> ExtentFingerprint {
    if bytes.is_empty() {
        return ExtentFingerprint::default();
    }

    let histogram = ByteHistogram::from_block(bytes);
    let h1 = crate::entropy::shannon::h1(&histogram);
    let mut bins = [0_u32; QUAD_HASH_BINS];
    hash4_bins(bytes, &mut bins);
    let h4 = sketch::entropy_from_counts_u32(&bins, bytes.len().saturating_sub(3).max(1) as u64);
    let (_, rl_bytes) = runlength(bytes);
    let top16_coverage = top_k_coverage_u64(histogram.counts(), 16, histogram.total());

    ExtentFingerprint {
        h1,
        h4,
        rl_fraction: rl_bytes as f32 / bytes.len() as f32,
        top16_coverage,
        byte_entropy_skew: h1 / 8.0,
    }
}

fn quantize_q4(bits_per_byte: f32) -> u8 {
    let value = bits_per_byte * 16.0;
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value.round() as u8
    }
}

fn runlength(bytes: &[u8]) -> (u16, u32) {
    if bytes.is_empty() {
        return (0, 0);
    }

    let mut runs = 0_u16;
    let mut bytes_in_runs = 0_u32;
    let mut index = 0_usize;
    while index < bytes.len() {
        let byte = bytes[index];
        let start = index;
        index += 1;
        while index < bytes.len() && bytes[index] == byte {
            index += 1;
        }
        let len = index - start;
        if len >= 4 {
            runs = runs.saturating_add(1);
            bytes_in_runs = bytes_in_runs.saturating_add(len as u32);
        }
    }
    (runs, bytes_in_runs)
}

fn top_k_coverage_q8(histogram: &[u32; 256], k: usize, total: u32) -> u8 {
    if total == 0 {
        return 0;
    }
    let mut counts = *histogram;
    let mut top = 0_u32;
    for _ in 0..k {
        let Some((index, count)) = counts
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(_, count)| *count)
        else {
            break;
        };
        top += count;
        counts[index] = 0;
    }
    let value = ((top as f32 / total as f32) * 256.0).round();
    if value >= 255.0 {
        255
    } else if value <= 0.0 {
        0
    } else {
        value as u8
    }
}

fn top_k_coverage_u64(histogram: &[u64; 256], k: usize, total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let mut counts = *histogram;
    let mut top = 0_u64;
    for _ in 0..k {
        let Some((index, count)) = counts
            .iter()
            .copied()
            .enumerate()
            .max_by_key(|(_, count)| *count)
        else {
            break;
        };
        top += count;
        counts[index] = 0;
    }
    top as f32 / total as f32
}

fn byte_class_bitmap(histogram: &[u32; 256]) -> u8 {
    let total = histogram.iter().sum::<u32>().max(1);
    let half = total / 2;
    let printable = (0x21..=0x7e).map(|byte| histogram[byte]).sum::<u32>();
    let whitespace = [b' ', b'\n', b'\r', b'\t']
        .into_iter()
        .map(|byte| histogram[byte as usize])
        .sum::<u32>();
    let control = (0x00..0x20).map(|byte| histogram[byte]).sum::<u32>();
    let high = (0x80..=0xff).map(|byte| histogram[byte]).sum::<u32>();

    let mut bitmap = 0_u8;
    if printable >= half {
        bitmap |= 1 << 0;
    }
    if whitespace >= half {
        bitmap |= 1 << 1;
    }
    if control >= half {
        bitmap |= 1 << 2;
    }
    if high >= half {
        bitmap |= 1 << 3;
    }
    bitmap
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_SIZE, FingerprintKernel, block, extent, kernel_catalog, kernels};

    #[test]
    fn zero_block_has_zero_entropy_and_one_run() {
        let bytes = [0_u8; BLOCK_SIZE];
        let fp = block(&bytes);
        assert_eq!(fp.h1_q4, 0);
        assert_eq!(fp.rl_runs_ge4, 1);
    }

    #[test]
    fn random_extent_has_high_entropy() {
        let bytes = (0..65_536)
            .scan(0xF22_u64, |state, _| {
                *state ^= *state >> 12;
                *state ^= *state << 25;
                *state ^= *state >> 27;
                Some(state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8)
            })
            .collect::<Vec<_>>();
        let fp = extent(&bytes);
        assert!(fp.h1 > 7.9, "h1={}", fp.h1);
        assert!(fp.h4 > 7.0, "h4={}", fp.h4);
    }

    #[test]
    fn extent_skew_is_h1_over_eight() {
        let fp = extent(b"abcdabcdabcdabcd");
        assert!((fp.byte_entropy_skew - fp.h1 / 8.0).abs() < 1e-6);
    }

    #[test]
    fn public_default_matches_pinned_scalar() {
        let mut bytes = [0_u8; BLOCK_SIZE];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index.wrapping_mul(37) ^ (index >> 1)) as u8;
        }

        assert_eq!(block(&bytes), kernels::scalar::block(&bytes));
        assert_eq!(extent(&bytes), kernels::scalar::extent(&bytes));
    }

    #[test]
    fn catalog_exposes_default_and_scalar_paths() {
        let catalog = kernel_catalog();
        assert!(
            catalog
                .iter()
                .any(|info| info.kernel == FingerprintKernel::Auto)
        );
        assert!(
            catalog
                .iter()
                .any(|info| info.kernel == FingerprintKernel::Scalar)
        );
        assert!(catalog.iter().all(|info| !info.hot_path_allocates));
    }
}

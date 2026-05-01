//! F22/content byte-stream fingerprints.
//!
//! The F21 paper prototype used `H1`, `H2`, `H3`, top-16 coverage,
//! run-length fraction, and byte-entropy skew. The F22 crate primitive keeps
//! the same cheap-feature spirit while replacing expensive `H3` with a
//! hardware-friendly CRC32-hashed 4-gram entropy estimate.
//!
//! [`block`] and [`kernels::scalar::block`] are exact parity surfaces. For
//! extents, [`kernels::scalar::extent`] is the exact reference path. The
//! ergonomic [`extent`] path computes exact H1, run-length, top-16 coverage,
//! and skew for all inputs, and uses exact H4 hash bins up to 64 KiB. Larger
//! extents sample H4 hash windows to keep FUSE/read-path latency bounded. The
//! current sampled-H4 regression tolerance is 2.5 bits on a periodic-text
//! fixture; use the scalar extent path when exact H4 is required.

use crate::{
    dispatch::{KernelIsa, KernelStatefulness, PrimitiveFamily, WorkingSetClass},
    math, sketch,
};

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod neon;

/// F22 block size. A 64 KiB extent contains 256 such blocks.
pub const BLOCK_SIZE: usize = 256;

/// Number of hash bins for extent-level 4-gram entropy.
pub const QUAD_HASH_BINS: usize = 4096;

/// Number of hash bins for block-level 4-gram entropy.
pub const QUAD_HASH_BLOCK_BINS: usize = 256;

/// Largest extent for exact H4 hash-bin counting on the default extent path.
pub const EXTENT_HASH_EXACT_MAX_BYTES: usize = 64 * 1024;

/// H4 hash-bin sampling stride for large extents on the default extent path.
pub const EXTENT_HASH_SAMPLE_STRIDE: usize = 4;

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
    /// x86 AVX2/SSE4.2 fused block kernel.
    Avx2,
}

impl FingerprintKernel {
    /// Stable benchmark identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Scalar => "scalar",
            Self::Avx2 => "avx2",
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

const FINGERPRINT_KERNEL_CATALOG: [FingerprintKernelInfo; 3] = [
    FingerprintKernelInfo {
        family: PrimitiveFamily::Fingerprint,
        kernel: FingerprintKernel::Auto,
        isa: KernelIsa::RuntimeDispatch,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        block_bytes: BLOCK_SIZE,
        hot_path_allocates: false,
        description: "default fingerprint path; exact scalar-compatible blocks, sampled large-extent H4",
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
    FingerprintKernelInfo {
        family: PrimitiveFamily::Fingerprint,
        kernel: FingerprintKernel::Avx2,
        isa: KernelIsa::X86Avx2,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        block_bytes: BLOCK_SIZE,
        hot_path_allocates: false,
        description: "x86 AVX2/SSE4.2 fused F22 block path with runtime-dispatched extent path",
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

    /// AArch64 NEON fused F22 block fingerprint kernel.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    pub use super::neon::public as neon;

    /// Runtime-dispatched/default fingerprint kernel.
    pub mod auto {
        use super::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, block_auto, extent_auto};

        /// Computes a compact F22/content fingerprint for one 256-byte block.
        #[must_use]
        pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
            block_auto(bytes)
        }

        /// Computes an aggregate F22/content fingerprint for any byte slice.
        ///
        /// H1, run-length, top-16 coverage, and skew are exact. H4 is exact
        /// up to [`crate::fingerprint::EXTENT_HASH_EXACT_MAX_BYTES`] and
        /// sampled on larger extents.
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

        /// Computes an exact aggregate F22/content fingerprint for any byte
        /// slice.
        #[must_use]
        pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
            extent_scalar(bytes)
        }
    }

    /// x86 AVX2/SSE4.2 fused block fingerprint kernel.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, extent_auto};

        /// Returns true when the fused AVX2/SSE4.2 block kernel is available.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("sse4.2")
        }

        /// Returns true when the fused AVX2/SSE4.2 block kernel is available.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// Computes a compact F22/content fingerprint for one 256-byte block.
        ///
        /// If the current CPU does not support AVX2 and SSE4.2, this falls
        /// back to the runtime-dispatched default path.
        #[must_use]
        pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
            if is_available() {
                // SAFETY: availability was checked immediately above.
                unsafe { super::super::block_avx2_unchecked(bytes) }
            } else {
                super::block_auto(bytes)
            }
        }

        /// Computes a compact F22/content fingerprint without checking CPU
        /// features.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2 and SSE4.2.
        #[must_use]
        pub unsafe fn block_unchecked(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
            unsafe { super::super::block_avx2_unchecked(bytes) }
        }

        /// Computes an aggregate F22/content fingerprint for any byte slice.
        ///
        /// The extent path uses the runtime-dispatched default extent
        /// accumulator; the AVX2-specific implementation is block-scoped.
        #[must_use]
        pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
            extent_auto(bytes)
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
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::avx2::is_available() {
            // SAFETY: availability was checked immediately above.
            return unsafe { block_avx2_unchecked(bytes) };
        }
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        if kernels::neon::is_available() {
            // SAFETY: NEON is mandatory on AArch64; CRC extension checked
            // by `kernels::neon::is_available()`.
            return unsafe { neon::block_neon_unchecked(bytes) };
        }
    }

    block_with_hash4(bytes, sketch::crc32_hash4_bins)
}

fn block_scalar(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    block_with_hash4(bytes, sketch::kernels::scalar::crc32_hash4_bins)
}

#[cfg(all(feature = "avx2", target_arch = "x86"))]
use core::arch::x86::_mm_crc32_u32 as crc32_u32_intrinsic;

#[cfg(all(feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::_mm_crc32_u32 as crc32_u32_intrinsic;

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2,sse4.2")]
unsafe fn block_avx2_unchecked(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    // SAFETY: target_feature(enable = "avx2,sse4.2") on this fn satisfies the
    // requirements of histogram_byte_block_avx2 (avx2) and
    // histogram_hash4_block_avx2 (avx2,sse4.2).
    let histogram = unsafe { histogram_byte_block_avx2(bytes) };
    let h1 = sketch::entropy_from_counts_u32(&histogram, BLOCK_SIZE as u64);
    let h1_q4 = quantize_q4(h1);
    let (rl_runs_ge4, _) = runlength(bytes);

    let h4_q4 = if h1_q4 >= 126 && rl_runs_ge4 == 0 {
        h1_q4
    } else {
        let bins = unsafe { histogram_hash4_block_avx2(bytes) };
        let h4 = sketch::entropy_from_counts_u32(&bins, (BLOCK_SIZE - 3) as u64);
        quantize_q4(h4)
    };

    BlockFingerprint {
        h1_q4,
        h4_q4,
        rl_runs_ge4,
        top4_coverage_q8: top_k_coverage_q8(&histogram, 4, BLOCK_SIZE as u32),
        byte_class: byte_class_bitmap(&histogram),
        reserved: 0,
    }
}

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn histogram_byte_block_avx2(bytes: &[u8; BLOCK_SIZE]) -> [u32; 256] {
    let mut h0 = [0_u32; 256];
    let mut h1 = [0_u32; 256];
    let mut h2 = [0_u32; 256];
    let mut h3 = [0_u32; 256];

    for index in 0..64 {
        h0[bytes[index] as usize] += 1;
        h1[bytes[64 + index] as usize] += 1;
        h2[bytes[128 + index] as usize] += 1;
        h3[bytes[192 + index] as usize] += 1;
    }

    // SAFETY: AVX2 enabled by target_feature; 4 stripes are 4 KiB each.
    unsafe { merge_4_stripes_u32_avx2::<256>(&h0, &h1, &h2, &h3) }
}

/// AVX2 sum-reduce four `[u32; N]` stripes into one. Replaces the prior
/// scalar `for i in 0..N { out[i] = a[i] + b[i] + c[i] + d[i]; }` loop —
/// 8 lanes per `_mm256_add_epi32`, so 32 iterations per 256-element merge
/// instead of 256.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `N` is a multiple of 8.
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2")]
unsafe fn merge_4_stripes_u32_avx2<const N: usize>(
    a: &[u32; N],
    b: &[u32; N],
    c: &[u32; N],
    d: &[u32; N],
) -> [u32; N] {
    debug_assert!(
        N.is_multiple_of(8),
        "N must be a multiple of 8 for the AVX2 merge"
    );

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{__m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_storeu_si256};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{__m256i, _mm256_add_epi32, _mm256_loadu_si256, _mm256_storeu_si256};

    let mut out = [0_u32; N];
    let mut i = 0;
    while i < N {
        // SAFETY: i + 8 <= N is guaranteed by the loop condition + N % 8
        // assertion; arrays a/b/c/d/out all have length N.
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
        let vc = unsafe { _mm256_loadu_si256(c.as_ptr().add(i).cast::<__m256i>()) };
        let vd = unsafe { _mm256_loadu_si256(d.as_ptr().add(i).cast::<__m256i>()) };
        let sum = _mm256_add_epi32(_mm256_add_epi32(va, vb), _mm256_add_epi32(vc, vd));
        // SAFETY: same.
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast::<__m256i>(), sum);
        }
        i += 8;
    }
    out
}

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx2,sse4.2")]
unsafe fn histogram_hash4_block_avx2(bytes: &[u8; BLOCK_SIZE]) -> [u32; QUAD_HASH_BLOCK_BINS] {
    let mut c0 = [0_u32; QUAD_HASH_BLOCK_BINS];
    let mut c1 = [0_u32; QUAD_HASH_BLOCK_BINS];
    let mut c2 = [0_u32; QUAD_HASH_BLOCK_BINS];
    let mut c3 = [0_u32; QUAD_HASH_BLOCK_BINS];
    let mask = (QUAD_HASH_BLOCK_BINS as u32) - 1;
    let ngrams = BLOCK_SIZE - 3;
    let groups = ngrams / 4;

    for group in 0..groups {
        let base = group * 4;
        let q0 = u32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ]);
        let q1 = u32::from_le_bytes([
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
            bytes[base + 4],
        ]);
        let q2 = u32::from_le_bytes([
            bytes[base + 2],
            bytes[base + 3],
            bytes[base + 4],
            bytes[base + 5],
        ]);
        let q3 = u32::from_le_bytes([
            bytes[base + 3],
            bytes[base + 4],
            bytes[base + 5],
            bytes[base + 6],
        ]);

        c0[(crc32_u32_intrinsic(0, q0) & mask) as usize] += 1;
        c1[(crc32_u32_intrinsic(0, q1) & mask) as usize] += 1;
        c2[(crc32_u32_intrinsic(0, q2) & mask) as usize] += 1;
        c3[(crc32_u32_intrinsic(0, q3) & mask) as usize] += 1;
    }

    let tail = groups * 4;
    if tail < ngrams {
        let q = u32::from_le_bytes([
            bytes[tail],
            bytes[tail + 1],
            bytes[tail + 2],
            bytes[tail + 3],
        ]);
        c0[(crc32_u32_intrinsic(0, q) & mask) as usize] += 1;
    }

    // SAFETY: avx2 + sse4.2 enabled by target_feature; QUAD_HASH_BLOCK_BINS=256
    // is a multiple of 8.
    unsafe { merge_4_stripes_u32_avx2::<QUAD_HASH_BLOCK_BINS>(&c0, &c1, &c2, &c3) }
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
/// H1, run-length, top-16 coverage, and skew are exact for all inputs. H4 is
/// exact up to [`EXTENT_HASH_EXACT_MAX_BYTES`] and sampled every
/// [`EXTENT_HASH_SAMPLE_STRIDE`] hash windows for larger inputs.
/// The sampled large-extent H4 tolerance currently has a regression bound of
/// 2.5 bits on a periodic-text fixture; it is an estimator, not a replacement
/// for exact H4.
///
/// Use [`kernels::scalar::extent`] when an exact pinned scalar reference is
/// required.
#[must_use]
pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
    kernels::auto::extent(bytes)
}

fn extent_auto(bytes: &[u8]) -> ExtentFingerprint {
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::avx2::is_available() {
            // SAFETY: availability was checked immediately above.
            return unsafe { extent_fused_sse42_unchecked(bytes) };
        }
    }

    extent_fused_auto_scalar(bytes)
}

fn extent_scalar(bytes: &[u8]) -> ExtentFingerprint {
    extent_fused_exact_scalar(bytes)
}

fn extent_fused_auto_scalar(bytes: &[u8]) -> ExtentFingerprint {
    extent_fused_with_crc32(
        bytes,
        sketch::kernels::scalar::crc32c_u32,
        extent_hash_stride(bytes),
    )
}

fn extent_fused_exact_scalar(bytes: &[u8]) -> ExtentFingerprint {
    extent_fused_with_crc32(bytes, sketch::kernels::scalar::crc32c_u32, 1)
}

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "sse4.2")]
unsafe fn extent_fused_sse42_unchecked(bytes: &[u8]) -> ExtentFingerprint {
    extent_fused_with_crc32(
        bytes,
        |seed, value| crc32_u32_intrinsic(seed, value),
        extent_hash_stride(bytes),
    )
}

const fn extent_hash_stride(bytes: &[u8]) -> usize {
    if bytes.len() > EXTENT_HASH_EXACT_MAX_BYTES {
        EXTENT_HASH_SAMPLE_STRIDE
    } else {
        1
    }
}

fn extent_fused_with_crc32<F>(bytes: &[u8], crc32: F, hash_stride: usize) -> ExtentFingerprint
where
    F: FnMut(u32, u32) -> u32,
{
    if bytes.is_empty() {
        return ExtentFingerprint::default();
    }

    let mut histogram = [0_u64; 256];
    let mut current_byte = bytes[0];
    let mut current_run = 0_usize;
    let mut rl_bytes = 0_u32;

    for (index, &byte) in bytes.iter().enumerate() {
        histogram[byte as usize] += 1;

        if index == 0 || byte == current_byte {
            current_byte = byte;
            current_run += 1;
        } else {
            if current_run >= 4 {
                rl_bytes = rl_bytes.saturating_add(current_run as u32);
            }
            current_byte = byte;
            current_run = 1;
        }
    }
    if current_run >= 4 {
        rl_bytes = rl_bytes.saturating_add(current_run as u32);
    }

    let h1 = entropy_from_counts_u64(&histogram, bytes.len() as u64);
    let h4 = if bytes.len() < 4 || (h1 >= 7.875 && rl_bytes == 0) {
        h1
    } else {
        let mut bins = [0_u32; QUAD_HASH_BINS];
        let observations = if hash_stride <= 1 {
            hash4_bins_with_crc32(bytes, &mut bins, crc32)
        } else {
            hash4_bins_strided_with_crc32(bytes, &mut bins, hash_stride, crc32)
        };
        sketch::entropy_from_counts_u32(&bins, observations.max(1))
    };
    let top16_coverage = top_k_coverage_u64(&histogram, 16, bytes.len() as u64);

    ExtentFingerprint {
        h1,
        h4,
        rl_fraction: rl_bytes as f32 / bytes.len() as f32,
        top16_coverage,
        byte_entropy_skew: h1 / 8.0,
    }
}

#[inline(always)]
fn hash4_bins_with_crc32<F>(bytes: &[u8], bins: &mut [u32; QUAD_HASH_BINS], mut crc32: F) -> u64
where
    F: FnMut(u32, u32) -> u32,
{
    if bytes.len() < 4 {
        return 0;
    }

    let mut c0 = [0_u32; QUAD_HASH_BINS];
    let mut c1 = [0_u32; QUAD_HASH_BINS];
    let mut c2 = [0_u32; QUAD_HASH_BINS];
    let mut c3 = [0_u32; QUAD_HASH_BINS];
    let mask = (QUAD_HASH_BINS as u32) - 1;
    let ngrams = bytes.len() - 3;
    let groups = ngrams / 4;

    for group in 0..groups {
        let base = group * 4;
        let q0 = u32::from_le_bytes([
            bytes[base],
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
        ]);
        let q1 = u32::from_le_bytes([
            bytes[base + 1],
            bytes[base + 2],
            bytes[base + 3],
            bytes[base + 4],
        ]);
        let q2 = u32::from_le_bytes([
            bytes[base + 2],
            bytes[base + 3],
            bytes[base + 4],
            bytes[base + 5],
        ]);
        let q3 = u32::from_le_bytes([
            bytes[base + 3],
            bytes[base + 4],
            bytes[base + 5],
            bytes[base + 6],
        ]);

        c0[(crc32(0, q0) & mask) as usize] += 1;
        c1[(crc32(0, q1) & mask) as usize] += 1;
        c2[(crc32(0, q2) & mask) as usize] += 1;
        c3[(crc32(0, q3) & mask) as usize] += 1;
    }

    for offset in (groups * 4)..ngrams {
        let q = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        c0[(crc32(0, q) & mask) as usize] += 1;
    }

    for index in 0..QUAD_HASH_BINS {
        bins[index] = c0[index] + c1[index] + c2[index] + c3[index];
    }

    ngrams as u64
}

#[inline(always)]
fn hash4_bins_strided_with_crc32<F>(
    bytes: &[u8],
    bins: &mut [u32; QUAD_HASH_BINS],
    stride: usize,
    mut crc32: F,
) -> u64
where
    F: FnMut(u32, u32) -> u32,
{
    if bytes.len() < 4 {
        return 0;
    }

    let mask = (QUAD_HASH_BINS as u32) - 1;
    let ngrams = bytes.len() - 3;
    let stride = stride.max(1);
    let mut observations = 0_u64;
    let mut offset = 0_usize;
    while offset < ngrams {
        let q = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]);
        bins[(crc32(0, q) & mask) as usize] += 1;
        observations += 1;
        offset += stride;
    }
    observations
}

fn entropy_from_counts_u64(counts: &[u64], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }

    let total = total as f64;
    let mut entropy = 0.0_f64;
    for &count in counts {
        if count != 0 {
            let p = count as f64 / total;
            entropy -= p * math::log2_f64(p);
        }
    }
    entropy as f32
}

fn quantize_q4(bits_per_byte: f32) -> u8 {
    let value = bits_per_byte * 16.0;
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        math::round_f32(value) as u8
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
    let value = math::round_f32((top as f32 / total as f32) * 256.0);
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
    fn public_default_matches_pinned_scalar_for_blocks_and_small_extents() {
        let mut bytes = [0_u8; BLOCK_SIZE];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index.wrapping_mul(37) ^ (index >> 1)) as u8;
        }

        assert_eq!(block(&bytes), kernels::scalar::block(&bytes));
        assert_eq!(extent(&bytes), kernels::scalar::extent(&bytes));
    }

    #[test]
    fn large_default_extent_keeps_exact_stable_fields() {
        let bytes = (0..(super::EXTENT_HASH_EXACT_MAX_BYTES * 2))
            .map(|index| {
                let motif = b"tokenfs-algos-f22-fingerprint-calibration:";
                motif[index % motif.len()].wrapping_add(((index / motif.len()) & 7) as u8)
            })
            .collect::<Vec<_>>();

        let default = extent(&bytes);
        let exact = kernels::scalar::extent(&bytes);

        assert_eq!(default.h1, exact.h1);
        assert_eq!(default.rl_fraction, exact.rl_fraction);
        assert_eq!(default.top16_coverage, exact.top16_coverage);
        assert_eq!(default.byte_entropy_skew, exact.byte_entropy_skew);
        assert!(
            (default.h4 - exact.h4).abs() <= 2.5,
            "default_h4={}, exact_h4={}",
            default.h4,
            exact.h4
        );
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
        assert!(
            catalog
                .iter()
                .any(|info| info.kernel == FingerprintKernel::Avx2)
        );
        assert!(catalog.iter().all(|info| !info.hot_path_allocates));
    }

    #[test]
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx2_block_matches_scalar_when_available() {
        if !kernels::avx2::is_available() {
            return;
        }

        let mut bytes = [0_u8; BLOCK_SIZE];
        for (index, byte) in bytes.iter_mut().enumerate() {
            *byte = (index.wrapping_mul(41) ^ (index >> 3).wrapping_mul(19)) as u8;
        }

        assert_eq!(kernels::avx2::block(&bytes), kernels::scalar::block(&bytes));
    }
}

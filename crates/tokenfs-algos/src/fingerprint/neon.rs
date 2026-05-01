//! AArch64 NEON port of the F22 fused per-block fingerprint kernel.
//!
//! Mirrors `block_avx2_unchecked` + `histogram_byte_block_avx2` +
//! `histogram_hash4_block_avx2` from [`super`]. Output is bit-exact with
//! the AVX2 path on identical input: histogram math is identical, and
//! CRC32C uses the same Castagnoli polynomial (0x1EDC6F41) on both ISAs.
//!
//! # Algorithm
//!
//! Same three-stage pipeline as the AVX2 port:
//!
//! 1. **Stripe-and-merge byte histogram** ([`histogram_byte_block_neon`]).
//!    Four private 256-entry tables, each filled scalar by 64 bytes of the
//!    256-byte block. The cross-stripe merge replaces a 256-iteration
//!    scalar reduce with 32 iterations of `vaddq_u32` over 8 lanes (two
//!    `uint32x4_t` halves per iter via `vld1q_u32` + `vaddq_u32`).
//!
//! 2. **Stripe-and-merge hash4 histogram** ([`histogram_hash4_block_neon`]).
//!    Four CRC32C-hashed bin tables (256 bins each), one per stagger of
//!    the sliding 4-byte window: 63 groups × 4 staggered `u32` reads →
//!    four independent `__crc32cw(0, q_i)` calls → four bin scatters,
//!    plus a tail window. Same `vaddq_u32` merge as (1). The 4-in-flight
//!    pipelining matches Skylake's `_mm_crc32_u32` shape and exploits the
//!    Cortex-A76 / Apple Firestorm 3-cycle latency / 1-cycle throughput
//!    on the CRC unit.
//!
//! 3. **Run-length** ([`super::runlength`]). Stays scalar — same as AVX2
//!    at the 256-byte block size.
//!
//! # NEON intrinsics
//!
//! `vld1q_u32`/`vst1q_u32` (unaligned u32×4 loads/stores), `vaddq_u32`
//! (replaces `_mm256_add_epi32`), `__crc32cw` (single-round Castagnoli
//! CRC32; stable since 1.80.0 under `target_feature("crc")`, classified
//! safe in stdarch — matches how the AVX2 path calls `_mm_crc32_u32` bare).
//!
//! # Safety convention (matches `byteclass/utf8_neon.rs`)
//!
//! Public entry [`block_neon_unchecked`] is `unsafe` and tagged
//! `#[target_feature(enable = "neon,crc")]`. NEON is mandatory in the
//! AArch64 ABI; `crc` is FEAT_CRC32, mandatory on ARMv8.1-A and present
//! on every aarch64-linux core in production (Cortex-A53 onward). Runtime
//! confirmation goes through `is_aarch64_feature_detected!("crc")`.
//!
//! Per workspace `unsafe_op_in_unsafe_fn = "deny"`: `vld1q_*`/`vst1q_*`
//! are `unsafe fn` (raw pointer ops) and wrapped in inner `unsafe { }`;
//! pure-compute `vaddq_u32` and the safe `__crc32cw` are called bare.
//!
//! # Required edits to `fingerprint/mod.rs`
//!
//! 1. At the top, declare the submodule:
//!    ```ignore
//!    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
//!    mod neon;
//!    ```
//! 2. In `pub mod kernels`, re-export the NEON public surface:
//!    ```ignore
//!    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
//!    pub use super::neon::public as neon;
//!    ```
//! 3. In `block_auto`, add a NEON dispatch branch mirroring AVX2:
//!    ```ignore
//!    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
//!    {
//!        if kernels::neon::is_available() {
//!            // SAFETY: availability checked above.
//!            return unsafe { neon::block_neon_unchecked(bytes) };
//!        }
//!    }
//!    ```
//! 4. Promote these parent-private helpers to `pub(super)` so the child
//!    module can call them: `runlength`, `quantize_q4`, `top_k_coverage_q8`,
//!    `byte_class_bitmap`.
//! 5. (Optional) Add a `FingerprintKernel::Neon` variant + matching
//!    `FingerprintKernelInfo` catalog entry with `KernelIsa::Aarch64Neon`.

use super::{
    BLOCK_SIZE, BlockFingerprint, QUAD_HASH_BLOCK_BINS, byte_class_bitmap, quantize_q4, runlength,
    top_k_coverage_q8,
};
use crate::sketch;

use core::arch::aarch64::{__crc32cw, vaddq_u32, vld1q_u32, vst1q_u32};

/// AArch64 NEON fused F22 block fingerprint. Bit-exact with
/// [`super::block_avx2_unchecked`] on identical input.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON and the FEAT_CRC32
/// (`crc`) extension. NEON is mandatory in the AArch64 base ABI; `crc` is
/// universally available on ARMv8.1-A and later (every aarch64-linux core
/// shipped this decade). Runtime confirmation is available via
/// [`public::is_available`].
#[target_feature(enable = "neon,crc")]
pub(crate) unsafe fn block_neon_unchecked(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    // SAFETY: `target_feature(enable = "neon,crc")` on this fn satisfies the
    // requirements of `histogram_byte_block_neon` (neon) and
    // `histogram_hash4_block_neon` (neon,crc).
    let histogram = unsafe { histogram_byte_block_neon(bytes) };
    let h1 = sketch::entropy_from_counts_u32(&histogram, BLOCK_SIZE as u64);
    let h1_q4 = quantize_q4(h1);
    let (rl_runs_ge4, _) = runlength(bytes);

    // Same short-circuit as the AVX2 path: a near-uniform block with no
    // long runs lets H4 ≈ H1 and skips the hash4 stage. Threshold mirrors
    // `block_avx2_unchecked` (h1_q4 >= 126 ⇔ h1 >= 7.875 bits/byte).
    let h4_q4 = if h1_q4 >= 126 && rl_runs_ge4 == 0 {
        h1_q4
    } else {
        // SAFETY: target_feature(enable = "neon,crc") covers the helper.
        let bins = unsafe { histogram_hash4_block_neon(bytes) };
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

/// Four-stripe scalar histogram + NEON merge. Identical fill pattern to
/// [`super::histogram_byte_block_avx2`].
///
/// # Safety
///
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
unsafe fn histogram_byte_block_neon(bytes: &[u8; BLOCK_SIZE]) -> [u32; 256] {
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

    // SAFETY: NEON enabled by target_feature; 4 stripes are 1 KiB each
    // (256 * 4 bytes).
    unsafe { merge_4_stripes_u32_neon::<256>(&h0, &h1, &h2, &h3) }
}

/// Four-stripe pipelined CRC32C hash4 histogram + NEON merge. Identical
/// stagger pattern to [`super::histogram_hash4_block_avx2`].
///
/// # Safety
///
/// Caller must ensure NEON and the `crc` (FEAT_CRC32) extension are
/// available.
#[target_feature(enable = "neon,crc")]
unsafe fn histogram_hash4_block_neon(bytes: &[u8; BLOCK_SIZE]) -> [u32; QUAD_HASH_BLOCK_BINS] {
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

        // `__crc32cw` is `pub fn` in stdarch (safe under target_feature),
        // matching how the AVX2 path leaves `_mm_crc32_u32` bare. Issuing
        // four independent CRCs keeps the CRC unit saturated despite the
        // 3-cycle latency — same hand-pipelined shape as the AVX2 sibling
        // and `sketch::kernels::sse42::crc32_hash4_bins_pipelined`.
        c0[(__crc32cw(0, q0) & mask) as usize] += 1;
        c1[(__crc32cw(0, q1) & mask) as usize] += 1;
        c2[(__crc32cw(0, q2) & mask) as usize] += 1;
        c3[(__crc32cw(0, q3) & mask) as usize] += 1;
    }

    let tail = groups * 4;
    if tail < ngrams {
        let q = u32::from_le_bytes([
            bytes[tail],
            bytes[tail + 1],
            bytes[tail + 2],
            bytes[tail + 3],
        ]);
        c0[(__crc32cw(0, q) & mask) as usize] += 1;
    }

    // SAFETY: NEON enabled by target_feature; QUAD_HASH_BLOCK_BINS = 256
    // is a multiple of 4.
    unsafe { merge_4_stripes_u32_neon::<QUAD_HASH_BLOCK_BINS>(&c0, &c1, &c2, &c3) }
}

/// NEON sum-reduce four `[u32; N]` stripes into one. Mirrors
/// [`super::merge_4_stripes_u32_avx2`]: replaces the scalar
/// `for i in 0..N { out[i] = a[i] + b[i] + c[i] + d[i]; }` loop with
/// 4-lane SIMD adds (`vaddq_u32`) plus a manual unroll to two vectors per
/// iteration so 8 elements collapse per loop body — matching AVX2's 8-lane
/// reduction throughput at the same `N`.
///
/// # Safety
///
/// Caller must ensure NEON is available and `N` is a multiple of 8.
#[target_feature(enable = "neon")]
unsafe fn merge_4_stripes_u32_neon<const N: usize>(
    a: &[u32; N],
    b: &[u32; N],
    c: &[u32; N],
    d: &[u32; N],
) -> [u32; N] {
    debug_assert!(N.is_multiple_of(8), "N must be a multiple of 8 for the NEON merge");

    let mut out = [0_u32; N];
    let mut i = 0;
    while i < N {
        // SAFETY: `i + 8 <= N` is guaranteed by the loop condition + the
        // `N % 8 == 0` debug-asserted invariant; arrays a/b/c/d/out all
        // have length N. `vld1q_u32` reads 4 contiguous u32 (16 bytes);
        // we issue two such loads per stripe (offsets +0 and +4), keeping
        // the reduction at 8 lanes per iteration.
        let va_lo = unsafe { vld1q_u32(a.as_ptr().add(i)) };
        let va_hi = unsafe { vld1q_u32(a.as_ptr().add(i + 4)) };
        let vb_lo = unsafe { vld1q_u32(b.as_ptr().add(i)) };
        let vb_hi = unsafe { vld1q_u32(b.as_ptr().add(i + 4)) };
        let vc_lo = unsafe { vld1q_u32(c.as_ptr().add(i)) };
        let vc_hi = unsafe { vld1q_u32(c.as_ptr().add(i + 4)) };
        let vd_lo = unsafe { vld1q_u32(d.as_ptr().add(i)) };
        let vd_hi = unsafe { vld1q_u32(d.as_ptr().add(i + 4)) };

        // Tree-shaped reduction so the two halves of the lane group can
        // be issued in parallel. `vaddq_u32` is a safe pure-compute
        // intrinsic under `target_feature(enable = "neon")` and is
        // called bare (no inner `unsafe { }` block).
        let sum_lo = vaddq_u32(vaddq_u32(va_lo, vb_lo), vaddq_u32(vc_lo, vd_lo));
        let sum_hi = vaddq_u32(vaddq_u32(va_hi, vb_hi), vaddq_u32(vc_hi, vd_hi));

        // SAFETY: same in-bounds reasoning as the loads above.
        unsafe {
            vst1q_u32(out.as_mut_ptr().add(i), sum_lo);
            vst1q_u32(out.as_mut_ptr().add(i + 4), sum_hi);
        }
        i += 8;
    }
    out
}

/// Public NEON kernel surface. Re-exported by `fingerprint::kernels::neon`
/// (see the top-of-file note for the required `pub use` line in
/// `mod.rs::kernels`). The `unreachable_pub` and `dead_code` allows
/// expect that bridge: without it the items here have no in-crate caller,
/// but once `mod.rs::kernels` re-exports `super::neon::public as neon` the
/// transitive path `tokenfs_algos::fingerprint::kernels::neon::block`
/// becomes the public entry point and both lints quiet on their own.
#[allow(unreachable_pub, dead_code)]
pub mod public {
    use super::{BLOCK_SIZE, BlockFingerprint, block_neon_unchecked};
    use crate::fingerprint::ExtentFingerprint;

    /// Returns true when the fused NEON+CRC32 block kernel is available.
    ///
    /// NEON is mandatory in the AArch64 base ABI, so the only runtime
    /// question is whether the CPU exposes the FEAT_CRC32 extension. That
    /// extension is universally present on ARMv8.1-A and later, and on
    /// every aarch64-linux core in production today; the check exists for
    /// API symmetry with [`super::super::kernels::avx2::is_available`] and
    /// for forward compatibility with hypothetical bare-ARMv8.0 targets.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::arch::is_aarch64_feature_detected!("crc")
    }

    /// Returns true when the fused NEON+CRC32 block kernel is available.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        // Without `std`, `is_aarch64_feature_detected!` is unavailable.
        // Conservatively report unavailable so callers fall back to the
        // scalar reference.
        false
    }

    /// Computes a compact F22/content fingerprint for one 256-byte block.
    ///
    /// If the current CPU does not support NEON+CRC, this falls back to
    /// the runtime-dispatched default path (which itself selects scalar).
    #[must_use]
    pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
        if is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe { block_neon_unchecked(bytes) }
        } else {
            super::super::block(bytes)
        }
    }

    /// Computes a compact F22/content fingerprint without checking CPU
    /// features.
    ///
    /// # Safety
    ///
    /// The caller must ensure the current CPU supports NEON and the
    /// FEAT_CRC32 (`crc`) extension.
    #[must_use]
    pub unsafe fn block_unchecked(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
        // SAFETY: caller upholds the precondition documented above.
        unsafe { block_neon_unchecked(bytes) }
    }

    /// Computes an aggregate F22/content fingerprint for any byte slice.
    ///
    /// The extent path uses the runtime-dispatched default extent
    /// accumulator; the NEON-specific implementation is block-scoped, the
    /// same scoping decision the AVX2 sibling makes.
    #[allow(dead_code)] // wired in once mod.rs re-exports `public as neon`
    #[must_use]
    pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
        super::super::extent(bytes)
    }
}

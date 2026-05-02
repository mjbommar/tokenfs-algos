//! Pinned byte-histogram kernels.
//!
//! These functions bypass the planner. They are intended for reproducible
//! experiments, paper calibration, regression debugging, and forensic
//! comparisons where the selected kernel must not change implicitly.

use crate::{histogram::ByteHistogram, primitives::histogram_scalar};

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
use crate::primitives::histogram_avx2;

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

/// AVX2-dispatched general four-stripe counter with scalar fallback.
///
/// **Status:** the body is currently scalar four-stripe counting under an
/// `#[target_feature(enable = "avx2")]` gate. It exists so the planner has a
/// pinned, feature-dispatched general-cardinality slot distinct from the
/// palette specialization (`avx2_palette_u32`). A real AVX2 byte-histogram
/// implementation is tracked separately; until it lands, this module's output
/// is bit-exact with [`stripe4_u32`] and `tests/avx2_parity.rs` enforces that.
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2_stripe4_u32 {
    use super::{ByteHistogram, histogram_scalar};
    use crate::primitives::histogram_avx2;

    /// Returns true when AVX2 is available at runtime.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// Returns true when AVX2 is available at runtime.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Builds a byte histogram with this pinned kernel.
    ///
    /// If AVX2 is unavailable at runtime, this falls back to `stripe4-u32`.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        if is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe {
                histogram_avx2::add_block_stripe4_u32(bytes, histogram.counts_mut_for_primitives());
            }
        } else {
            histogram_scalar::add_block_striped_u32::<4>(
                bytes,
                histogram.counts_mut_for_primitives(),
            );
        }
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }

    /// Adds bytes with AVX2 without checking runtime availability.
    ///
    /// # Safety
    ///
    /// The caller must ensure the current CPU supports AVX2.
    pub unsafe fn add_block_unchecked(bytes: &[u8], histogram: &mut ByteHistogram) {
        unsafe {
            histogram_avx2::add_block_stripe4_u32(bytes, histogram.counts_mut_for_primitives());
        }
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

/// AVX2 four-stripe counter with a constant-chunk RLE fast path.
///
/// Scans the input in 32-byte chunks. Each chunk that is a single
/// repeated byte is counted with one `bin += 32` update; any
/// heterogeneous chunk falls back to a scalar four-stripe pass over the
/// chunk's bytes. Bytes after the last full 32-byte chunk are counted
/// scalar.
///
/// Output is bit-exact with [`stripe4_u32`]; only the work distribution
/// changes.
///
/// **Expected speedup.** Real-world inputs (text, code, executables)
/// frequently contain runs of zero, space, or padding bytes that satisfy
/// the constant-chunk predicate, so the kernel sees 2-5x throughput vs.
/// the plain four-stripe path on those workloads. On uniform random data
/// no chunk is constant and the kernel collapses to the four-stripe
/// scalar core plus one AVX2 broadcast / cmpeq / movemask probe per
/// chunk; the probe overhead is small (single-digit percent) so the
/// kernel runs at roughly the four-stripe baseline.
///
/// **Why not a per-byte RLE inside heterogeneous chunks?** Tracking
/// partial runs across the cmpeq mask costs more in branch and shift
/// bookkeeping than the four-stripe scalar fallback saves. The
/// constant-chunk fast path captures the bulk of the wins on real data
/// without inflating the cost of random-byte inputs.
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2_rle_stripe4_u32 {
    use super::{ByteHistogram, histogram_avx2, histogram_scalar};

    /// Returns true when AVX2 is available at runtime.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// Returns true when AVX2 is available at runtime.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Builds a byte histogram with this pinned kernel.
    ///
    /// If AVX2 is unavailable at runtime, this falls back to
    /// `stripe4-u32`.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        if is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe {
                histogram_avx2::add_block_rle_stripe4_u32(
                    bytes,
                    histogram.counts_mut_for_primitives(),
                );
            }
        } else {
            histogram_scalar::add_block_striped_u32::<4>(
                bytes,
                histogram.counts_mut_for_primitives(),
            );
        }
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }

    /// Adds bytes with AVX2 without checking runtime availability.
    ///
    /// # Safety
    ///
    /// The caller must ensure the current CPU supports AVX2.
    pub unsafe fn add_block_unchecked(bytes: &[u8], histogram: &mut ByteHistogram) {
        unsafe {
            histogram_avx2::add_block_rle_stripe4_u32(bytes, histogram.counts_mut_for_primitives());
        }
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

/// AVX2 palette-counting fast path with scalar local-table fallback.
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2_palette_u32 {
    use super::{ByteHistogram, histogram_scalar};
    use crate::primitives::histogram_avx2;

    /// Returns true when AVX2 is available at runtime.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// Returns true when AVX2 is available at runtime.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Builds a byte histogram with this pinned kernel.
    ///
    /// If AVX2 is unavailable at runtime, this falls back to `local-u32` while
    /// preserving exact counts.
    #[must_use]
    pub fn block(bytes: &[u8]) -> ByteHistogram {
        let mut histogram = ByteHistogram::new();
        add_block(bytes, &mut histogram);
        histogram
    }

    /// Adds bytes into an existing histogram with this pinned kernel.
    ///
    /// If AVX2 is unavailable at runtime, this falls back to `local-u32` while
    /// preserving exact counts.
    pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
        if is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe {
                histogram_avx2::add_block_palette_u32(bytes, histogram.counts_mut_for_primitives());
            }
        } else {
            histogram_scalar::add_block_local_u32(bytes, histogram.counts_mut_for_primitives());
        }
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }

    /// Adds bytes with AVX2 without checking runtime availability.
    ///
    /// # Safety
    ///
    /// The caller must ensure the current CPU supports AVX2.
    pub unsafe fn add_block_unchecked(bytes: &[u8], histogram: &mut ByteHistogram) {
        unsafe {
            histogram_avx2::add_block_palette_u32(bytes, histogram.counts_mut_for_primitives());
        }
        histogram.add_to_total_for_primitives(bytes.len() as u64);
    }
}

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

/// AVX-512 BITALG bit-sliced 8-marginal byte histogram.
///
/// **What this is.** A *partial* histogram: 8 marginal bit-frequencies
/// `marginals[k]` = count of input bytes with bit `k` set, for k in 0..8.
/// This is **not** the 256-bin byte histogram — it discards which byte
/// values appeared and keeps only their per-bit projections. Use the
/// other kernels when you need the full distribution; use this one when
/// per-bit density is the actual signal (entropy bounds, randomness
/// checks, sparse-byte detection).
///
/// **Why BITALG.** Two AVX-512 BITALG instructions carry the kernel:
/// - `_mm512_popcnt_epi8` — per-byte popcount, used to fold the
///   per-byte Hamming weight into a scalar `total_bits` total in one
///   pass over the input. Without BITALG the same statistic would need
///   `pshufb`-based byte-popcount LUTs (8-12 ops/64 bytes) or scalar
///   `popcntq` after vector reduction.
/// - `_mm512_movepi8_mask` — extracts bit-7 of each byte as a 64-bit
///   mask, used per bit-plane to count the marginal directly.
///
/// **Cost.** ~10 ops per 64-byte chunk for the 8 marginals (8 shifts +
/// 8 movepi8_mask + 8 scalar popcountq accumulations) plus 1
/// `_mm512_popcnt_epi8` and 1 horizontal reduce per chunk for the
/// total-bits side-channel — all from the AVX-512 BITALG / AVX-512BW
/// pipeline, no `pshufb`-based byte-popcount LUT required.
///
/// **Marginal vs full histogram.** Eight marginals are NOT enough to
/// reconstruct the 256-bin distribution (256 > 2^8 = 256 only when bits
/// are independent, which they aren't for real byte streams). Treat the
/// 8 marginals as a low-rank summary, not as a histogram replacement.
#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512_bitalg_bitsliced {
    /// 8 marginal bit-frequencies plus a `total_bits` set-bit count.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct BitMarginals {
        /// `marginals[k]` = number of input bytes with bit `k` set, for
        /// k in 0..8. Bit 0 is the LSB.
        pub marginals: [u64; 8],
        /// Total number of set bits across the entire input. Equal to
        /// `marginals.iter().sum()`. Reported separately because the
        /// AVX-512 BITALG `_mm512_popcnt_epi8` path computes it without
        /// extra cost; on the scalar reference this is also a sum of
        /// `byte.count_ones()`.
        pub total_bits: u64,
        /// Number of input bytes processed.
        pub total_bytes: u64,
    }

    /// Returns true when AVX-512BW + AVX-512 BITALG are both available
    /// at runtime.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512bitalg")
    }

    /// Returns true when AVX-512BW + AVX-512 BITALG are both available.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Scalar reference: 8 marginal bit-frequencies + total-bits.
    #[must_use]
    pub fn block_scalar(bytes: &[u8]) -> BitMarginals {
        let mut out = BitMarginals {
            marginals: [0; 8],
            total_bits: 0,
            total_bytes: bytes.len() as u64,
        };
        for &b in bytes {
            for (k, marginal) in out.marginals.iter_mut().enumerate() {
                if (b >> k) & 1 != 0 {
                    *marginal += 1;
                }
            }
            out.total_bits += u64::from(b.count_ones());
        }
        out
    }

    /// AVX-512 BITALG bit-sliced kernel. Falls back to
    /// [`block_scalar`] when the runtime CPU lacks AVX-512BW or
    /// AVX-512 BITALG.
    #[must_use]
    pub fn block(bytes: &[u8]) -> BitMarginals {
        if is_available() {
            // SAFETY: availability checked immediately above.
            unsafe { block_unchecked(bytes) }
        } else {
            block_scalar(bytes)
        }
    }

    /// AVX-512 BITALG kernel without runtime feature checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure both AVX-512BW AND AVX-512 BITALG are
    /// available on the current CPU.
    #[target_feature(enable = "avx512bw,avx512bitalg")]
    #[must_use]
    pub unsafe fn block_unchecked(bytes: &[u8]) -> BitMarginals {
        // SAFETY: caller guarantees the required CPU features.
        unsafe { block_avx512_impl(bytes) }
    }

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_movepi8_mask, _mm512_popcnt_epi8,
        _mm512_reduce_add_epi64, _mm512_sad_epu8, _mm512_setzero_si512, _mm512_slli_epi16,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_movepi8_mask, _mm512_popcnt_epi8,
        _mm512_reduce_add_epi64, _mm512_sad_epu8, _mm512_setzero_si512, _mm512_slli_epi16,
    };

    /// 64 bytes per AVX-512 vector iteration.
    const SIMD_CHUNK_SIZE: usize = 64;

    /// AVX-512 implementation. Per chunk: per-bit `slli + movepi8_mask`
    /// followed by scalar `popcntq` for the 8 marginals;
    /// `_mm512_popcnt_epi8 + _mm512_sad_epu8` for the total-bits
    /// side-channel.
    #[target_feature(enable = "avx512bw,avx512bitalg")]
    unsafe fn block_avx512_impl(bytes: &[u8]) -> BitMarginals {
        let len = bytes.len();
        let mut marginals = [0_u64; 8];

        // Total-bits accumulator. `_mm512_sad_epu8(x, 0)` horizontally
        // sums each 8-byte qword of `x` into a u16 within a u64 lane;
        // we accumulate those 8 partial sums in a __m512i and reduce at
        // the end. Each qword holds at most 8 * 8 = 64 (a single chunk's
        // contribution), and we only run while the 64 chunks fit in a
        // u16 per qword (saturate-safe up to ~1024 chunks = 64 KiB before
        // any qword overflows — we flush every 256 chunks to be safe).
        let mut total_acc = _mm512_setzero_si512();
        let mut total_bits: u64 = 0;
        let zero = _mm512_setzero_si512();

        let iter_lim = len - (len % SIMD_CHUNK_SIZE);
        let ptr = bytes.as_ptr();

        let mut idx = 0;
        let mut chunks_in_acc = 0_u32;
        // SAFETY (entire block): AVX-512BW + BITALG enabled by
        // target_feature; pointer adds use `idx + 64 <= iter_lim <= len`.
        unsafe {
            while idx < iter_lim {
                let v = _mm512_loadu_si512(ptr.add(idx).cast::<__m512i>());

                // Eight bit-marginals via shift-mask-popcount.
                // Loop unrolled (k in 0..8) so the shift immediates are
                // const-folded by LLVM into individual `vpsllw` ops.
                let m0 = _mm512_movepi8_mask(_mm512_slli_epi16::<7>(v));
                let m1 = _mm512_movepi8_mask(_mm512_slli_epi16::<6>(v));
                let m2 = _mm512_movepi8_mask(_mm512_slli_epi16::<5>(v));
                let m3 = _mm512_movepi8_mask(_mm512_slli_epi16::<4>(v));
                let m4 = _mm512_movepi8_mask(_mm512_slli_epi16::<3>(v));
                let m5 = _mm512_movepi8_mask(_mm512_slli_epi16::<2>(v));
                let m6 = _mm512_movepi8_mask(_mm512_slli_epi16::<1>(v));
                let m7 = _mm512_movepi8_mask(v);
                marginals[0] += u64::from(m0.count_ones());
                marginals[1] += u64::from(m1.count_ones());
                marginals[2] += u64::from(m2.count_ones());
                marginals[3] += u64::from(m3.count_ones());
                marginals[4] += u64::from(m4.count_ones());
                marginals[5] += u64::from(m5.count_ones());
                marginals[6] += u64::from(m6.count_ones());
                marginals[7] += u64::from(m7.count_ones());

                // Per-byte popcount (BITALG), then horizontal sum into
                // total_acc qwords via `_mm512_sad_epu8` against zero.
                let pc = _mm512_popcnt_epi8(v);
                let qsums = _mm512_sad_epu8(pc, zero);
                total_acc = _mm512_add_epi64(total_acc, qsums);

                idx += SIMD_CHUNK_SIZE;
                chunks_in_acc += 1;
                // Flush every 256 chunks (= 16 KiB) to keep each u64
                // lane under 256 * 8 * 8 = 16384, well under saturation.
                if chunks_in_acc == 256 {
                    total_bits = total_bits.wrapping_add(_mm512_reduce_add_epi64(total_acc) as u64);
                    total_acc = _mm512_setzero_si512();
                    chunks_in_acc = 0;
                }
            }
            if chunks_in_acc != 0 {
                total_bits = total_bits.wrapping_add(_mm512_reduce_add_epi64(total_acc) as u64);
            }
        }

        // Tail (< 64 bytes): scalar fold.
        if idx < len {
            for &b in &bytes[idx..] {
                for (k, marginal) in marginals.iter_mut().enumerate() {
                    if (b >> k) & 1 != 0 {
                        *marginal += 1;
                    }
                }
                total_bits += u64::from(b.count_ones());
            }
        }

        BitMarginals {
            marginals,
            total_bits,
            total_bytes: len as u64,
        }
    }
}

/// AVX-512 GFNI + BITALG bit-sliced 8-marginal byte histogram.
///
/// Same output and parity contract as
/// [`avx512_bitalg_bitsliced::block`]: 8 marginal bit-frequencies +
/// total set-bits + total bytes. The difference is the inner loop:
/// instead of 8 `_mm512_slli_epi16` shifts per chunk, this kernel does
/// the per-bit projection via a SINGLE `_mm512_gf2p8affine_epi64_epi8`
/// per bit-position, picking that bit into byte-position 7 directly
/// from the affine matrix definition.
///
/// **Honest expectation: this is roughly the same speed as the BITALG
/// path on Zen 4.** GFNI affine and `vpsllw` both run on FP0/FP1 with
/// 1-cycle throughput on Zen 4 (per AMD SOG); the affine has slightly
/// higher latency (3 cycles vs 1 for slli) but the loop is bottlenecked
/// on `_mm512_movepi8_mask` and scalar `popcntq`, not on the per-bit
/// projection. The GFNI variant exists primarily to **document** the
/// trade-off and to expose a callable surface for the s7 benchmark
/// table — if Zen 4 surprises us and `vgf2p8affineqb` schedules better
/// than `vpsllw` on a particular workload, we'll have data; if it
/// doesn't, the bench column will say so.
///
/// **What this is NOT.** It is *not* an 8x8 byte-level bit-transpose
/// followed by BITALG popcount, despite that being the common framing
/// in research-summary text. A true 8x8 byte-tile bit-transpose cannot
/// be done in a single `vgf2p8affineqb` because the affine's per-qword
/// formula `y[j].bit[i] = parity(a[7-i] & x[j])` only depends on input
/// byte `x[j]`, not on the other 7 bytes of the qword — so the
/// per-byte affine output cannot read bits from neighbouring bytes.
/// The full transpose requires affine + a cross-byte permute (`vpermb`,
/// from VBMI) + a second affine, totalling 3 instructions per 8-byte
/// tile and 24 per 64-byte chunk — strictly worse than the per-bit
/// shift-and-mask BITALG path. This kernel uses the simpler
/// per-bit-projection affine instead.
#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512_gfni_bitsliced {
    pub use super::avx512_bitalg_bitsliced::BitMarginals;

    /// Returns true when AVX-512BW + GFNI are both available at runtime.
    /// (BITALG is not strictly required for this kernel — we only use
    /// `_mm512_movepi8_mask` and scalar `popcntq` from BW for the
    /// reduction — but we keep the `total_bits` side-channel running
    /// scalar so the same `BitMarginals` shape stays comparable.)
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx512bw")
            && std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("gfni")
    }

    /// Returns true when AVX-512BW + AVX-512F + GFNI are all available.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// AVX-512 GFNI kernel. Falls back to the BITALG scalar reference
    /// when the runtime CPU lacks AVX-512BW / AVX-512F / GFNI.
    #[must_use]
    pub fn block(bytes: &[u8]) -> BitMarginals {
        if is_available() {
            // SAFETY: availability checked immediately above.
            unsafe { block_unchecked(bytes) }
        } else {
            super::avx512_bitalg_bitsliced::block_scalar(bytes)
        }
    }

    /// AVX-512 GFNI kernel without runtime feature checks.
    ///
    /// # Safety
    ///
    /// The caller must ensure that AVX-512BW, AVX-512F, and GFNI are
    /// all available on the current CPU.
    #[target_feature(enable = "avx512bw,avx512f,gfni")]
    #[must_use]
    pub unsafe fn block_unchecked(bytes: &[u8]) -> BitMarginals {
        // SAFETY: caller guarantees the required CPU features.
        unsafe { block_avx512_impl(bytes) }
    }

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_movepi8_mask,
        _mm512_set1_epi64,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m512i, _mm512_gf2p8affine_epi64_epi8, _mm512_loadu_si512, _mm512_movepi8_mask,
        _mm512_set1_epi64,
    };

    /// 64 bytes per AVX-512 vector iteration.
    const SIMD_CHUNK_SIZE: usize = 64;

    /// Per-bit-projection affine matrices. `BIT_PROJ[k]` is an 8x8
    /// matrix whose only non-zero row is row 0, with bit `(7-k)` set.
    /// Applying `vgf2p8affineqb(x, BIT_PROJ[k], 0)` produces an output
    /// where, for each input byte `x[j]`:
    ///   `y[j].bit[i] = parity(BIT_PROJ[k][7-i] & x[j])`
    /// Only `i == 7` gives a non-zero parity (since only row 0 = `7-7`
    /// is non-zero), and that parity = `parity((1 << (7-k)) & x[j])` =
    /// `bit-(7-k) of x[j]`.
    ///
    /// To make `y[j].bit[7]` equal to `bit-k of x[j]` we set row 0 =
    /// `(1 << k)`. Then `movepi8_mask(y)` extracts a 64-bit mask where
    /// bit-j is set iff `bit-k of x[j]` is set — a per-bit marginal
    /// indicator without any explicit `vpsllw`.
    ///
    /// Matrix layout per qword: byte 0 holds row 0 (low byte = `(1 <<
    /// k)`); bytes 1..8 are zero. As a 64-bit integer in
    /// little-endian byte-order: `(1 << k)` in the low byte, zeros
    /// elsewhere → integer value `(1 << k)`.
    const BIT_PROJ: [i64; 8] = [
        1 << 0,
        1 << 1,
        1 << 2,
        1 << 3,
        1 << 4,
        1 << 5,
        1 << 6,
        1 << 7,
    ];

    /// AVX-512 GFNI implementation.
    #[target_feature(enable = "avx512bw,avx512f,gfni")]
    unsafe fn block_avx512_impl(bytes: &[u8]) -> BitMarginals {
        let len = bytes.len();
        let mut marginals = [0_u64; 8];
        let mut total_bits: u64 = 0;

        let iter_lim = len - (len % SIMD_CHUNK_SIZE);
        let ptr = bytes.as_ptr();

        // SAFETY (entire block): AVX-512BW + AVX-512F + GFNI enabled by
        // target_feature; pointer adds use `idx + 64 <= iter_lim <= len`.
        // Each of these per-bit affine matrices is splat across all 8
        // qwords of the __m512i.
        unsafe {
            let m0 = _mm512_set1_epi64(BIT_PROJ[0]);
            let m1 = _mm512_set1_epi64(BIT_PROJ[1]);
            let m2 = _mm512_set1_epi64(BIT_PROJ[2]);
            let m3 = _mm512_set1_epi64(BIT_PROJ[3]);
            let m4 = _mm512_set1_epi64(BIT_PROJ[4]);
            let m5 = _mm512_set1_epi64(BIT_PROJ[5]);
            let m6 = _mm512_set1_epi64(BIT_PROJ[6]);
            let m7 = _mm512_set1_epi64(BIT_PROJ[7]);

            let mut idx = 0;
            while idx < iter_lim {
                let v = _mm512_loadu_si512(ptr.add(idx).cast::<__m512i>());

                // 8 per-bit-projection affine calls, each followed by
                // movepi8_mask + scalar popcountq accumulation.
                let p0 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m0));
                let p1 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m1));
                let p2 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m2));
                let p3 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m3));
                let p4 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m4));
                let p5 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m5));
                let p6 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m6));
                let p7 = _mm512_movepi8_mask(_mm512_gf2p8affine_epi64_epi8::<0>(v, m7));

                let c0 = u64::from(p0.count_ones());
                let c1 = u64::from(p1.count_ones());
                let c2 = u64::from(p2.count_ones());
                let c3 = u64::from(p3.count_ones());
                let c4 = u64::from(p4.count_ones());
                let c5 = u64::from(p5.count_ones());
                let c6 = u64::from(p6.count_ones());
                let c7 = u64::from(p7.count_ones());

                marginals[0] += c0;
                marginals[1] += c1;
                marginals[2] += c2;
                marginals[3] += c3;
                marginals[4] += c4;
                marginals[5] += c5;
                marginals[6] += c6;
                marginals[7] += c7;

                // total_bits = sum of all 8 marginals over this chunk.
                total_bits += c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;

                idx += SIMD_CHUNK_SIZE;
            }
        }

        // Tail (< 64 bytes): scalar fold.
        if !len.is_multiple_of(SIMD_CHUNK_SIZE) {
            for &b in &bytes[iter_lim..] {
                for (k, marginal) in marginals.iter_mut().enumerate() {
                    if (b >> k) & 1 != 0 {
                        *marginal += 1;
                    }
                }
                total_bits += u64::from(b.count_ones());
            }
        }

        BitMarginals {
            marginals,
            total_bits,
            total_bytes: len as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    use super::{avx2_rle_stripe4_u32, stripe4_u32};

    /// Random-but-deterministic byte stream for parity testing.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(n);
        out
    }

    /// Heuristic ASCII-text payload similar to the bench-compare filler.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn ascii_text_bytes(n: usize) -> Vec<u8> {
        const FILLER: &[u8] = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
            Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n";
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let take = (n - out.len()).min(FILLER.len());
            out.extend_from_slice(&FILLER[..take]);
        }
        out
    }

    /// Executable-like payload: short header, long zero pad, mixed code,
    /// long zero pad. Mimics typical PE/ELF section layout for the RLE
    /// fast path.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn executable_like_bytes(n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        out.extend_from_slice(b"\x7fELF\x02\x01\x01\x00");
        out.extend(core::iter::repeat_n(0, 248));
        // Code-ish bytes.
        let mut state = 0xC0FF_EE12_3456_7890_u64;
        while out.len() < n.min(out.len() + 4096) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.push((state >> 32) as u8 | 0x40);
        }
        // Long .bss-style zero pad.
        while out.len() < n {
            out.push(0);
        }
        out.truncate(n);
        out
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn assert_parity(bytes: &[u8], label: &str) {
        let expected = stripe4_u32::block(bytes);
        let actual = avx2_rle_stripe4_u32::block(bytes);
        assert_eq!(
            actual.counts(),
            expected.counts(),
            "{label}: avx2_rle_stripe4_u32 diverged from stripe4_u32 (len {})",
            bytes.len()
        );
        assert_eq!(actual.total(), expected.total(), "{label}: total mismatch");
    }

    #[test]
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx2_rle_stripe4_matches_stripe4_on_random() {
        if !avx2_rle_stripe4_u32::is_available() {
            return;
        }
        for &n in &[0_usize, 1, 31, 32, 33, 64, 1023, 1024, 65_536] {
            assert_parity(&random_bytes(n, 0xC8C2_5E0F_2C5C_3F6D), "random");
        }
    }

    #[test]
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx2_rle_stripe4_matches_stripe4_on_zeros() {
        if !avx2_rle_stripe4_u32::is_available() {
            return;
        }
        for &n in &[0_usize, 32, 64, 1024, 65_536] {
            assert_parity(&vec![0_u8; n], "all-zero");
        }
        // Long all-0xff run.
        assert_parity(&vec![0xff_u8; 4096], "all-0xff");
    }

    #[test]
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx2_rle_stripe4_matches_stripe4_on_ascii_text() {
        if !avx2_rle_stripe4_u32::is_available() {
            return;
        }
        for &n in &[0_usize, 1, 31, 65, 1024, 8192, 65_536] {
            assert_parity(&ascii_text_bytes(n), "ascii-text");
        }
    }

    #[test]
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx2_rle_stripe4_matches_stripe4_on_executable_like() {
        if !avx2_rle_stripe4_u32::is_available() {
            return;
        }
        for &n in &[256_usize, 1024, 4096, 65_536] {
            assert_parity(&executable_like_bytes(n), "executable-like");
        }
    }

    /// Random-but-deterministic byte stream for bitalg parity (always
    /// available — does not depend on the avx2 feature gate).
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    fn bitalg_random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(n);
        out
    }

    #[test]
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx512_bitalg_bitsliced_matches_scalar_when_available() {
        use super::avx512_bitalg_bitsliced::{block_scalar, block_unchecked, is_available};
        if !is_available() {
            return;
        }
        for &n in &[
            0_usize, 1, 7, 31, 63, 64, 65, 127, 128, 129, 255, 256, 1023, 1024, 4097, 65_536,
            131_072,
        ] {
            let bytes = bitalg_random_bytes(n, 0xCAFE_F00D_DEAD_BEEF);
            let expected = block_scalar(&bytes);
            // SAFETY: availability checked above.
            let actual = unsafe { block_unchecked(&bytes) };
            assert_eq!(
                actual, expected,
                "bitalg vs scalar mismatch on {n}-byte payload"
            );
        }
        // All-zero and all-0xff edge cases.
        for &n in &[0_usize, 64, 1024] {
            let zeros = vec![0_u8; n];
            // SAFETY: availability checked above.
            let actual = unsafe { block_unchecked(&zeros) };
            assert_eq!(actual, block_scalar(&zeros), "all-zero {n}");
            let ones = vec![0xff_u8; n];
            // SAFETY: availability checked above.
            let actual = unsafe { block_unchecked(&ones) };
            assert_eq!(actual, block_scalar(&ones), "all-0xff {n}");
        }
    }

    #[test]
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    fn avx512_gfni_bitsliced_matches_scalar_when_available() {
        use super::avx512_bitalg_bitsliced::block_scalar;
        use super::avx512_gfni_bitsliced::{block_unchecked, is_available};
        if !is_available() {
            return;
        }
        for &n in &[
            0_usize, 1, 7, 31, 63, 64, 65, 127, 128, 129, 255, 256, 1023, 1024, 4097, 65_536,
            131_072,
        ] {
            let bytes = bitalg_random_bytes(n, 0xC0FF_EE00_BAAD_F00D);
            let expected = block_scalar(&bytes);
            // SAFETY: availability checked above.
            let actual = unsafe { block_unchecked(&bytes) };
            assert_eq!(
                actual, expected,
                "gfni vs scalar mismatch on {n}-byte payload"
            );
        }
        for &n in &[0_usize, 64, 1024] {
            let zeros = vec![0_u8; n];
            // SAFETY: availability checked above.
            let actual = unsafe { block_unchecked(&zeros) };
            assert_eq!(actual, block_scalar(&zeros), "all-zero {n}");
            let ones = vec![0xff_u8; n];
            // SAFETY: availability checked above.
            let actual = unsafe { block_unchecked(&ones) };
            assert_eq!(actual, block_scalar(&ones), "all-0xff {n}");
        }
    }
}

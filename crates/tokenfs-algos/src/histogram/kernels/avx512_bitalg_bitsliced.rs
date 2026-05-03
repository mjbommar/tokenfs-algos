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

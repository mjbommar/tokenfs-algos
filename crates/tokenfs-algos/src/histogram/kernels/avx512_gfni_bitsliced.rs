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

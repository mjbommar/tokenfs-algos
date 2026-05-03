use super::POW2_NEG_LUT;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, __m256i, __m512i, _mm_loadu_si128, _mm256_loadu_si256, _mm512_add_pd,
    _mm512_cvtepu8_epi32, _mm512_i32gather_pd, _mm512_loadu_si512, _mm512_max_epu8,
    _mm512_reduce_add_pd, _mm512_setzero_pd, _mm512_storeu_si512,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256i, __m512i, _mm_loadu_si128, _mm256_loadu_si256, _mm512_add_pd,
    _mm512_cvtepu8_epi32, _mm512_i32gather_pd, _mm512_loadu_si512, _mm512_max_epu8,
    _mm512_reduce_add_pd, _mm512_setzero_pd, _mm512_storeu_si512,
};

/// 64 u8 registers per AVX-512 vector.
const VEC_BYTES: usize = 64;

/// Returns true when AVX-512BW (for VPMAXUB) is available
/// at runtime. AVX-512BW is the byte-and-word AVX-512
/// extension (Skylake-X, Ice Lake, Zen 4); without it the
/// 64-byte VPMAXUB on `__m512i` is unavailable.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw")
}

/// Returns true when AVX-512F + AVX-512BW are available.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX-512 per-bucket max merge.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F
/// and AVX-512BW and that `dst.len() == src.len()`.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn merge(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let mut i = 0_usize;
    while i + VEC_BYTES <= len {
        // SAFETY: `i + 64 <= len`; AVX-512F+BW enabled by
        // the enclosing target_feature.
        let a = unsafe { _mm512_loadu_si512(dst.as_ptr().add(i).cast::<__m512i>()) };
        let b = unsafe { _mm512_loadu_si512(src.as_ptr().add(i).cast::<__m512i>()) };
        let m = _mm512_max_epu8(a, b);
        // SAFETY: same range bound as load above.
        unsafe {
            _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast::<__m512i>(), m);
        }
        i += VEC_BYTES;
    }
    // Scalar tail (≤ 63 bytes). For valid HLL precisions
    // (4..=16) the register count is `2^p` ∈ {16..=65_536},
    // which is always a multiple of 16; the only sub-vector
    // tail occurs at `p ∈ {4, 5}` (16 / 32 registers).
    while i < len {
        let b = src[i];
        if b > dst[i] {
            dst[i] = b;
        }
        i += 1;
    }
}

/// AVX-512 harmonic-mean cardinality `alpha * m^2 / Z`.
///
/// Loads 16 u8 registers per inner iteration, zero-extends
/// to 16x i32 with VPMOVZXBD, gathers via two
/// `_mm512_i32gather_pd` calls (each 8-wide), and
/// accumulates into two `__m512d` accumulators.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F.
#[target_feature(enable = "avx512f,avx512bw")]
#[must_use]
pub unsafe fn count_raw(registers: &[u8], alpha: f64) -> f64 {
    let m = registers.len() as f64;
    let lut_ptr = POW2_NEG_LUT.as_ptr();
    let mut acc0 = _mm512_setzero_pd();
    let mut acc1 = _mm512_setzero_pd();
    let mut i = 0_usize;
    let n = registers.len();

    while i + 16 <= n {
        // Load 16 u8 register bytes into the low half of a
        // 32-byte zmm register, then zero-extend to 16x i32.
        // SAFETY: `i + 16 <= n` bounds the 16-byte read.
        let xmm = unsafe { _mm_loadu_si128(registers.as_ptr().add(i).cast::<__m128i>()) };
        // SAFETY: VPMOVZXBD widens 16x u8 → 16x i32; AVX-512F
        // enabled by the enclosing target_feature.
        let indices_512: __m512i = _mm512_cvtepu8_epi32(xmm);

        // Split the 16x i32 indices into two 8x i32 halves
        // for the two 8-wide gathers.
        let mut tmp_idx = [0_i32; 16];
        // SAFETY: tmp_idx is i32-aligned and 64 bytes long.
        unsafe {
            _mm512_storeu_si512(tmp_idx.as_mut_ptr().cast::<__m512i>(), indices_512);
        }
        let lo_idx_256 = unsafe { _mm256_loadu_si256(tmp_idx.as_ptr().cast::<__m256i>()) };
        let hi_idx_256 = unsafe { _mm256_loadu_si256(tmp_idx.as_ptr().add(8).cast::<__m256i>()) };

        // SAFETY: indices ∈ 0..=255; gather offsets ≤ 2040.
        // Intrinsic signature for AVX-512:
        // <SCALE>(offsets: __m256i, base: *const f64).
        let g0 = unsafe { _mm512_i32gather_pd::<8>(lo_idx_256, lut_ptr) };
        let g1 = unsafe { _mm512_i32gather_pd::<8>(hi_idx_256, lut_ptr) };
        acc0 = _mm512_add_pd(acc0, g0);
        acc1 = _mm512_add_pd(acc1, g1);
        i += 16;
    }

    let mut sum = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));

    // Scalar tail.
    while i < n {
        sum += POW2_NEG_LUT[registers[i] as usize];
        i += 1;
    }

    alpha * m * m / sum
}

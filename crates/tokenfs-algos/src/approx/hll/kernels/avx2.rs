use super::POW2_NEG_LUT;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, __m256d, __m256i, _mm_cvtepu8_epi32, _mm_set_epi64x, _mm_srli_si128, _mm256_add_pd,
    _mm256_i32gather_pd, _mm256_loadu_si256, _mm256_max_epu8, _mm256_setzero_pd,
    _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, __m256d, __m256i, _mm_cvtepu8_epi32, _mm_set_epi64x, _mm_srli_si128, _mm256_add_pd,
    _mm256_i32gather_pd, _mm256_loadu_si256, _mm256_max_epu8, _mm256_setzero_pd,
    _mm256_storeu_si256,
};

/// 32 u8 registers per AVX2 vector.
const VEC_BYTES: usize = 32;

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

/// AVX2 per-bucket max merge.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2 and
/// that `dst.len() == src.len()`.
#[target_feature(enable = "avx2")]
pub unsafe fn merge(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let mut i = 0_usize;
    while i + VEC_BYTES <= len {
        // SAFETY: `i + 32 <= len` and AVX2 enabled by the
        // enclosing target_feature.
        let a = unsafe { _mm256_loadu_si256(dst.as_ptr().add(i).cast::<__m256i>()) };
        let b = unsafe { _mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>()) };
        let m = _mm256_max_epu8(a, b);
        // SAFETY: same range bound as load above.
        unsafe {
            _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), m);
        }
        i += VEC_BYTES;
    }
    // Scalar tail.
    while i < len {
        let b = src[i];
        if b > dst[i] {
            dst[i] = b;
        }
        i += 1;
    }
}

/// AVX2 harmonic-mean cardinality `alpha * m^2 / Z`.
///
/// Loads 8 u8 registers per inner iteration, gathers their
/// `2^-r` f64 values from [`super::POW2_NEG_LUT`] in two
/// 4-wide AVX2 gathers, and accumulates into two parallel
/// `__m256d` accumulators (8 doubles total) to break the
/// dependency chain through `_mm256_add_pd`.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn count_raw(registers: &[u8], alpha: f64) -> f64 {
    let m = registers.len() as f64;
    let lut_ptr = POW2_NEG_LUT.as_ptr();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut i = 0_usize;
    let n = registers.len();

    // Inner loop: process 8 registers per iteration via
    // two 4-wide AVX2 gathers from the f64 LUT. Each gather
    // requires 4 i32 indices; we load the 8-byte group into
    // the low qword of an xmm, do PMOVZXBD on bytes 0..4
    // for the low gather, then byte-shift the xmm right by
    // 4 bytes (PSRLDQ) to move bytes 4..8 down into bytes
    // 0..4, and PMOVZXBD again for the high gather.
    while i + 8 <= n {
        // Load 8 u8 register values into the low qword of
        // an xmm register. SAFETY: `i + 8 <= n`.
        let bytes_u64 =
            unsafe { core::ptr::read_unaligned(registers.as_ptr().add(i).cast::<i64>()) };
        let packed_xmm: __m128i = _mm_set_epi64x(0, bytes_u64);

        // Low 4 bytes (registers i..i+4) → 4x i32 indices.
        let lo_indices = _mm_cvtepu8_epi32(packed_xmm);
        // High 4 bytes (registers i+4..i+8): byte-shift the
        // xmm right by 4 to expose bytes 4..8 in the low
        // 4-byte slot, then PMOVZXBD.
        let shifted = _mm_srli_si128::<4>(packed_xmm);
        let hi_indices = _mm_cvtepu8_epi32(shifted);

        // SAFETY: indices ∈ 0..=255, LUT length 256, so
        // gather offsets in bytes (= 8*index) are ≤ 2040.
        // Intrinsic signature: <SCALE>(base_ptr, vindex).
        let g0 = unsafe { _mm256_i32gather_pd::<8>(lut_ptr, lo_indices) };
        let g1 = unsafe { _mm256_i32gather_pd::<8>(lut_ptr, hi_indices) };
        acc0 = _mm256_add_pd(acc0, g0);
        acc1 = _mm256_add_pd(acc1, g1);
        i += 8;
    }

    // Horizontal sum the two parallel f64x4 accumulators.
    // SAFETY: AVX2 enabled by the enclosing target_feature.
    let mut sum = unsafe { horizontal_sum_pd(_mm256_add_pd(acc0, acc1)) };

    // Scalar tail.
    while i < n {
        sum += POW2_NEG_LUT[registers[i] as usize];
        i += 1;
    }

    alpha * m * m / sum
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_pd(v: __m256d) -> f64 {
    // Cast the __m256d into two __m128d halves via the
    // integer-cast helpers, then sum lanes scalar-style.
    // We use a memory store for portability across rustc
    // versions where some hadd/permute intrinsics are not
    // stable on AVX2 alone.
    let mut tmp = [0.0_f64; 4];
    // SAFETY: tmp is f64-aligned (Rust guarantees this for
    // local arrays of f64); the store is 32-byte aligned
    // enough for VMOVUPD.
    unsafe {
        core::arch::asm!(
            "vmovupd [{ptr}], {vec}",
            ptr = in(reg) tmp.as_mut_ptr(),
            vec = in(ymm_reg) v,
            options(nostack, preserves_flags),
        );
    }
    tmp[0] + tmp[1] + tmp[2] + tmp[3]
}

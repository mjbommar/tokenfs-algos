#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_add_epi64, _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_setr_epi64x,
    _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_setr_epi64x,
    _mm256_slli_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
};

/// 4 u64 lanes per AVX2 vector.
const LANES: usize = 4;

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

/// 64-bit lane-wise multiply emulation for AVX2.
///
/// AVX2 has `_mm256_mul_epu32` which multiplies the low 32
/// bits of each 64-bit lane (producing 64-bit products in
/// even-indexed lanes only) but no native 64-bit multiply.
/// We emulate `a * b` via three 32x32→64 multiplies and two
/// shifts — the standard schoolbook recipe:
///
/// `a * b = (a.lo * b.lo) + ((a.hi * b.lo) << 32) + ((a.lo * b.hi) << 32)`
///
/// (the `a.hi * b.hi` term is dropped because we only keep
/// the low 64 bits of the product, and that term contributes
/// only to bits 64+).
///
/// # Safety
///
/// AVX2 must be available; caller asserts via `target_feature`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_epi64_lo(a: __m256i, b: __m256i) -> __m256i {
    let a_lo = a; // low 32 of each 64-bit lane (high 32 ignored by mul_epu32)
    let b_lo = b;
    let a_hi = _mm256_srli_epi64::<32>(a);
    let b_hi = _mm256_srli_epi64::<32>(b);

    let lo_lo = _mm256_mul_epu32(a_lo, b_lo);
    let hi_lo = _mm256_mul_epu32(a_hi, b_lo);
    let lo_hi = _mm256_mul_epu32(a_lo, b_hi);

    // Shift cross terms into the high half of each 64-bit lane
    // and add to the lo*lo product.
    let cross = _mm256_add_epi64(hi_lo, lo_hi);
    let cross_shifted = _mm256_slli_epi64::<32>(cross);
    _mm256_add_epi64(lo_lo, cross_shifted)
}

/// AVX2 position-computation kernel.
///
/// Computes K positions in parallel across 4-wide vectors,
/// stores to the caller-supplied buffer, then performs the
/// scalar `% bits` reduction. Bit-exact with
/// [`super::scalar::positions`].
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
///
/// # Panics
///
/// Panics if `out.len() < k` or `bits == 0`.
#[target_feature(enable = "avx2")]
pub unsafe fn positions(h1: u64, h2: u64, k: usize, bits: usize, out: &mut [u64]) {
    assert!(bits > 0, "BloomFilter bits must be > 0");
    assert!(out.len() >= k, "out buffer too small: {} < {k}", out.len());

    // Stage 1: vector-add `h1 + i*h2` into a stack buffer of
    // u64 lanes. Process LANES (4) i-values per iteration.
    let h1_v = _mm256_set1_epi64x(h1 as i64);
    let h2_v = _mm256_set1_epi64x(h2 as i64);

    let mut i = 0_usize;
    while i + LANES <= k {
        // i-vector for this block: {i+0, i+1, i+2, i+3}
        let i_v = _mm256_setr_epi64x(i as i64, (i + 1) as i64, (i + 2) as i64, (i + 3) as i64);
        // SAFETY: `mul_epi64_lo` requires AVX2; the enclosing
        // target_feature supplies it.
        let prod = unsafe { mul_epi64_lo(i_v, h2_v) };
        let sum = _mm256_add_epi64(h1_v, prod);
        // Store the 4 raw u64 positions to `out[i..i+4]`.
        // SAFETY: `out.len() >= k >= i + LANES` holds by the
        // loop condition; the cast to `__m256i*` is align(1)
        // via the unaligned store intrinsic.
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().add(i).cast::<__m256i>(), sum);
        }
        i += LANES;
    }

    // Stage 1 tail: remaining 0..LANES positions, scalar.
    let bits_u64 = bits as u64;
    while i < k {
        let raw = h1.wrapping_add((i as u64).wrapping_mul(h2));
        out[i] = raw;
        i += 1;
    }

    // Stage 2: scalar modular reduction. AVX2 has no vector
    // u64 divide; the per-lane `% bits` reduction is cheap
    // compared to the multiply-add chain anyway.
    for slot in out.iter_mut().take(k) {
        *slot %= bits_u64;
    }
}

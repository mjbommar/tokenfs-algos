#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_add_epi64, _mm512_mullo_epi64, _mm512_set1_epi64, _mm512_setr_epi64,
    _mm512_storeu_si512,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_mullo_epi64, _mm512_set1_epi64, _mm512_setr_epi64,
    _mm512_storeu_si512,
};

/// 8 u64 lanes per AVX-512 vector.
const LANES: usize = 8;

/// Returns true when AVX-512F + AVX-512DQ are available at
/// runtime.
///
/// `_mm512_mullo_epi64` is part of AVX-512DQ. The base
/// AVX-512F flag is implied by DQ but checked independently
/// for clarity (and to match the dispatch convention used by
/// [`crate::bits::popcount::kernels::avx512::is_available`]).
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512dq")
}

/// Returns true when AVX-512F + AVX-512DQ are available.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX-512 position-computation kernel.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F
/// and AVX-512DQ.
///
/// # Panics
///
/// Panics if `out.len() < k` or `bits == 0`.
#[target_feature(enable = "avx512f,avx512dq")]
pub unsafe fn positions(h1: u64, h2: u64, k: usize, bits: usize, out: &mut [u64]) {
    assert!(bits > 0, "BloomFilter bits must be > 0");
    assert!(out.len() >= k, "out buffer too small: {} < {k}", out.len());

    let h1_v = _mm512_set1_epi64(h1 as i64);
    let h2_v = _mm512_set1_epi64(h2 as i64);

    let mut i = 0_usize;
    while i + LANES <= k {
        let i_v = _mm512_setr_epi64(
            i as i64,
            (i + 1) as i64,
            (i + 2) as i64,
            (i + 3) as i64,
            (i + 4) as i64,
            (i + 5) as i64,
            (i + 6) as i64,
            (i + 7) as i64,
        );
        let prod = _mm512_mullo_epi64(i_v, h2_v);
        let sum = _mm512_add_epi64(h1_v, prod);
        // SAFETY: `out.len() >= k >= i + LANES`; unaligned
        // store is align(1) via the intrinsic.
        unsafe {
            _mm512_storeu_si512(out.as_mut_ptr().add(i).cast::<__m512i>(), sum);
        }
        i += LANES;
    }

    // Tail: scalar.
    while i < k {
        let raw = h1.wrapping_add((i as u64).wrapping_mul(h2));
        out[i] = raw;
        i += 1;
    }

    let bits_u64 = bits as u64;
    for slot in out.iter_mut().take(k) {
        *slot %= bits_u64;
    }
}

use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64,
    _mm512_setzero_si512,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64,
    _mm512_setzero_si512,
};

/// 64 bytes (8 u64) per AVX-512 vector iteration.
const VEC_BYTES: usize = 64;

/// 4x unrolled = 256 bytes per outer iteration. Four independent
/// accumulators break the dependency chain through the
/// `_mm512_add_epi64` reductions and let the OoO scheduler issue
/// VPOPCNTQ + VPADDQ pairs in parallel.
const UNROLL_VECTORS: usize = 4;

/// Returns true when AVX-512F + AVX-512VPOPCNTDQ are available at
/// runtime.
///
/// `AVX512VPOPCNTDQ` shipped on Intel Ice Lake (2019) and AMD
/// Zen 4 (2022). The base AVX-512F flag is implied by VPOPCNTDQ
/// support but checked independently for clarity.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vpopcntdq")
}

/// Returns true when AVX-512F + AVX-512VPOPCNTDQ are available.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// AVX-512 VPOPCNTQ popcount over a `&[u64]` slice.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F and
/// AVX-512VPOPCNTDQ.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[must_use]
pub unsafe fn popcount_u64_slice(words: &[u64]) -> u64 {
    // Reuse the byte-driven inner kernel; popcount is associative
    // over the bit decomposition so the result is identical.
    let bytes_ptr = words.as_ptr().cast::<u8>();
    let bytes_len = core::mem::size_of_val(words);
    // SAFETY: `bytes_ptr`/`bytes_len` describe the same memory as
    // `words`, borrowed for the duration of this call.
    let bytes = unsafe { core::slice::from_raw_parts(bytes_ptr, bytes_len) };
    unsafe { popcount_bytes_avx512(bytes) }
}

/// AVX-512 VPOPCNTQ popcount over a `&[u8]` slice.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512F and
/// AVX-512VPOPCNTDQ.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[must_use]
pub unsafe fn popcount_u8_slice(bytes: &[u8]) -> u64 {
    // SAFETY: target_feature on this fn forwards to the inner
    // kernel.
    unsafe { popcount_bytes_avx512(bytes) }
}

/// Inner VPOPCNTQ kernel.
///
/// # Safety
///
/// AVX-512F + AVX-512VPOPCNTDQ must be available.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[inline]
unsafe fn popcount_bytes_avx512(bytes: &[u8]) -> u64 {
    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();

    let mut index = 0_usize;
    let unroll_bytes = VEC_BYTES * UNROLL_VECTORS;

    while index + unroll_bytes <= bytes.len() {
        // SAFETY: each load reads 64 bytes and `index + 4*64 <=
        // bytes.len()` is enforced by the loop condition.
        let v0 = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast::<__m512i>()) };
        let v1 =
            unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index + VEC_BYTES).cast::<__m512i>()) };
        let v2 = unsafe {
            _mm512_loadu_si512(bytes.as_ptr().add(index + 2 * VEC_BYTES).cast::<__m512i>())
        };
        let v3 = unsafe {
            _mm512_loadu_si512(bytes.as_ptr().add(index + 3 * VEC_BYTES).cast::<__m512i>())
        };
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v0));
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(v1));
        acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(v2));
        acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(v3));
        index += unroll_bytes;
    }

    while index + VEC_BYTES <= bytes.len() {
        // SAFETY: index + 64 <= bytes.len() bounds the load.
        let v = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast::<__m512i>()) };
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v));
        index += VEC_BYTES;
    }

    let sum01 = _mm512_add_epi64(acc0, acc1);
    let sum23 = _mm512_add_epi64(acc2, acc3);
    let sum = _mm512_add_epi64(sum01, sum23);
    let total_simd = _mm512_reduce_add_epi64(sum) as u64;

    // Scalar tail for the remaining 0..63 bytes. Aligned-by-8
    // tails go through the chunked u64 path inside
    // `scalar::popcount_u8_slice`.
    total_simd + scalar::popcount_u8_slice(&bytes[index..])
}

use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_extract_epi64,
    _mm256_loadu_si256, _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_srli_epi16,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_extract_epi64,
    _mm256_loadu_si256, _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setr_epi8, _mm256_setzero_si256,
    _mm256_shuffle_epi8, _mm256_srli_epi16,
};

/// 32 bytes (4 u64) per inner SIMD iteration.
const VEC_BYTES: usize = 32;

/// 8x unrolled = 256 bytes per outer iteration. Each per-byte
/// popcount is in `0..=8`; eight unrolled per-byte sums fit in
/// `0..=64`, well below the u8 saturation threshold of 255, so we
/// can defer the SAD reduction to the end of the unrolled block.
const UNROLL_VECTORS: usize = 8;

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

/// AVX2 Mula nibble-LUT popcount over a `&[u64]` slice.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn popcount_u64_slice(words: &[u64]) -> u64 {
    // The byte-slice kernel is the same workload (popcount over a
    // contiguous run of bytes) and writing it once keeps the two
    // entry points bit-exact.
    // SAFETY: target_feature(enable = "avx2") on this fn satisfies
    // the unsafe precondition of `popcount_bytes_avx2`.
    let bytes_ptr = words.as_ptr().cast::<u8>();
    let bytes_len = core::mem::size_of_val(words);
    // SAFETY: `bytes_ptr` and `bytes_len` describe the same memory
    // region as `words`, which is borrowed for the duration of
    // this call.
    let bytes = unsafe { core::slice::from_raw_parts(bytes_ptr, bytes_len) };
    unsafe { popcount_bytes_avx2(bytes) }
}

/// AVX2 Mula nibble-LUT popcount over a `&[u8]` slice.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn popcount_u8_slice(bytes: &[u8]) -> u64 {
    // SAFETY: target_feature on this fn forwards to the inner
    // kernel.
    unsafe { popcount_bytes_avx2(bytes) }
}

/// Inner Mula nibble-LUT kernel.
///
/// # Safety
///
/// AVX2 must be available; caller asserts via `target_feature`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn popcount_bytes_avx2(bytes: &[u8]) -> u64 {
    // Nibble-popcount LUT, broadcast to both 128-bit lanes of the
    // AVX2 vector. Built via `_mm256_setr_epi8` so the source
    // ordering reads naturally as nibble values 0..=15.
    //
    // _mm256_shuffle_epi8 is **per-128-bit-lane** — the same 16
    // entries populate both halves and the shuffle indices are
    // interpreted modulo 16 within each lane.
    let lookup: __m256i = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0F);

    let mut acc_u64 = _mm256_setzero_si256();

    let mut index = 0_usize;
    let unroll_bytes = VEC_BYTES * UNROLL_VECTORS;

    // Outer loop: process UNROLL_VECTORS vectors per iteration,
    // accumulating per-byte popcounts in an 8-bit vector and
    // folding to u64 lanes only once per outer iteration. Each
    // per-byte popcount is in 0..=8, so 8 iterations sum to
    // 0..=64 < 255 — safe in u8 lanes without overflow.
    while index + unroll_bytes <= bytes.len() {
        let mut acc_u8 = _mm256_setzero_si256();
        for k in 0..UNROLL_VECTORS {
            // SAFETY: `index + k * VEC_BYTES + 32 <= index +
            // UNROLL_VECTORS * VEC_BYTES <= bytes.len()`; AVX2
            // is enabled by the enclosing target_feature so the
            // helper `popcnt_per_byte` precondition is met.
            let v = unsafe {
                _mm256_loadu_si256(bytes.as_ptr().add(index + k * VEC_BYTES).cast::<__m256i>())
            };
            acc_u8 = _mm256_add_epi8(acc_u8, unsafe { popcnt_per_byte(v, lookup, low_mask) });
        }
        // Horizontal sum each 8-byte qword of `acc_u8` into a u16
        // within a 64-bit lane via `_mm256_sad_epu8(acc, 0)`, then
        // accumulate into the persistent u64 register.
        let zero = _mm256_setzero_si256();
        acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(acc_u8, zero));
        index += unroll_bytes;
    }

    // Single-vector loop for the leftover 32-byte windows after
    // the unrolled block.
    while index + VEC_BYTES <= bytes.len() {
        // SAFETY: index + 32 <= bytes.len() bounds the load;
        // AVX2 is enabled by the enclosing target_feature.
        let v = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
        let per_byte = unsafe { popcnt_per_byte(v, lookup, low_mask) };
        let zero = _mm256_setzero_si256();
        acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(per_byte, zero));
        index += VEC_BYTES;
    }

    // Reduce the four 64-bit lanes of `acc_u64` to a scalar.
    let s0 = _mm256_extract_epi64::<0>(acc_u64) as u64;
    let s1 = _mm256_extract_epi64::<1>(acc_u64) as u64;
    let s2 = _mm256_extract_epi64::<2>(acc_u64) as u64;
    let s3 = _mm256_extract_epi64::<3>(acc_u64) as u64;
    let mut total = s0 + s1 + s2 + s3;

    // Scalar tail for the remaining 0..VEC_BYTES bytes.
    total += scalar::popcount_u8_slice(&bytes[index..]);
    total
}

/// One Mula step on a 32-byte input vector: per-byte popcount.
///
/// Returns a `__m256i` whose byte at lane `i` holds
/// `bytes[i].count_ones()` for the input vector `v`.
///
/// # Safety
///
/// AVX2 must be available; caller asserts via `target_feature`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn popcnt_per_byte(v: __m256i, lookup: __m256i, low_mask: __m256i) -> __m256i {
    // Low nibbles: AND with 0x0F.
    let lo = _mm256_and_si256(v, low_mask);
    // High nibbles: shift right by 4 (per 16-bit element; both
    // halves get the same shift, so AND-ing with 0x0F afterwards
    // recovers the per-byte high nibble).
    let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), low_mask);
    let lo_pc = _mm256_shuffle_epi8(lookup, lo);
    let hi_pc = _mm256_shuffle_epi8(lookup, hi);
    _mm256_add_epi8(lo_pc, hi_pc)
}

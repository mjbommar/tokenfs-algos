//! x86 AVX2 dense-distance kernels.
//!
//! Each integer kernel processes 8 lanes per iteration (256-bit register,
//! 8×u32 or 8×f32). Tails fall back to scalar.
//!
//! ## Numeric tolerance for SIMD reductions
//!
//! Integer kernels (`dot_u32`, `l1_u32`, `l2_squared_u32`) are bit-exact
//! with [`super::scalar`]. Floating-point kernels (`dot_f32`,
//! `l2_squared_f32`, `cosine_similarity_f32`) use **8-way pairwise tree
//! reduction** vs. the scalar left-to-right summation, so results can
//! differ by a few ULP on long vectors.
//!
//! The cross-backend tolerance follows the Higham §3 / Wilkinson model:
//! `|fl(dot) - exact| <= n * eps * sum(|a_i*b_i|)`. Parity tests assert
//! `(scalar - avx2).abs() / sum(|a*b|) < 1e-3`. See `tests/avx2_parity.rs`
//! for the exact assertion shape and `docs/v0.2_planning/13_VECTOR.md`
//! § 5 for rationale.
//!
//! Bit-exact reproducibility *within* the AVX2 backend is guaranteed: the
//! 8-way pairwise tree is part of the public contract.
//!
//! Hamming/Jaccard for `u64` packed bitvectors reuse the
//! [`crate::bits::popcount`] kernels for the inner popcount; the wrapper
//! XORs/ANDs/ORs the words and forwards to the popcount kernel.

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256, __m256i, _mm_add_epi64, _mm_extract_epi64, _mm256_add_epi64, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_max_epu32, _mm256_min_epu32, _mm256_mul_epu32, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_shuffle_epi32, _mm256_storeu_ps, _mm256_sub_epi32, _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256, __m256i, _mm_add_epi64, _mm_extract_epi64, _mm256_add_epi64, _mm256_castsi256_si128,
    _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_loadu_si256,
    _mm256_max_epu32, _mm256_min_epu32, _mm256_mul_epu32, _mm256_setzero_ps, _mm256_setzero_si256,
    _mm256_shuffle_epi32, _mm256_storeu_ps, _mm256_sub_epi32, _mm256_sub_ps,
};

use super::scalar;

const LANES_U32: usize = 8;
const LANES_F32: usize = 8;

/// Returns true when AVX2 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
}

/// Returns true when AVX2 is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Reduces a `__m256i` of four `u64` lanes to a scalar `u64`.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hadd_u64x4(v: __m256i) -> u64 {
    // SAFETY: AVX2 enabled.
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let sum = _mm_add_epi64(lo, hi);
    // sum = [a, b]; we want a + b.
    let a = _mm_extract_epi64::<0>(sum) as u64;
    let b = _mm_extract_epi64::<1>(sum) as u64;
    a.wrapping_add(b)
}

/// Reduces a `__m256` of eight `f32` lanes to a scalar `f32` using the
/// canonical 8-way pairwise tree.
///
/// The reduction tree is part of the AVX2 public contract:
/// `((p0 + p1) + (p2 + p3))` with `p_i = lane[2i] + lane[2i+1]`.
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hadd_f32x8(v: __m256) -> f32 {
    let mut tmp = [0.0_f32; 8];
    // SAFETY: tmp is 32-byte writable; alignment-tolerant store.
    unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), v) };
    // 8-way pairwise tree reduction (deterministic, lower error than
    // left-to-right). Documented as the canonical AVX2 reduction order.
    let p0 = tmp[0] + tmp[1];
    let p1 = tmp[2] + tmp[3];
    let p2 = tmp[4] + tmp[5];
    let p3 = tmp[6] + tmp[7];
    (p0 + p1) + (p2 + p3)
}

/// Multiplies low-32-of-each-u64 lane and accumulates into u64 lanes.
/// `_mm256_mul_epu32` reads the *even* 32-bit positions (0, 2, 4, 6).
/// To cover the odd positions (1, 3, 5, 7) we shuffle them down first.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_accum_u32_pair(
    a: __m256i,
    b: __m256i,
    acc_lo: __m256i,
    acc_hi: __m256i,
) -> (__m256i, __m256i) {
    // SAFETY: AVX2 enabled.
    let prod_even = _mm256_mul_epu32(a, b);
    // Shuffle 0xB1 = swap each adjacent 32-bit pair so odd lanes become even.
    let a_odd = _mm256_shuffle_epi32::<0xB1>(a);
    let b_odd = _mm256_shuffle_epi32::<0xB1>(b);
    let prod_odd = _mm256_mul_epu32(a_odd, b_odd);
    (
        _mm256_add_epi64(acc_lo, prod_even),
        _mm256_add_epi64(acc_hi, prod_odd),
    )
}

/// Inner product of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn dot_u32(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    let mut i = 0;
    while i + LANES_U32 <= a.len() {
        // SAFETY: bounds checked by loop condition.
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
        (acc_lo, acc_hi) = unsafe { mul_accum_u32_pair(va, vb, acc_lo, acc_hi) };
        i += LANES_U32;
    }
    // SAFETY: AVX2 enabled.
    let total = unsafe { hadd_u64x4(_mm256_add_epi64(acc_lo, acc_hi)) };
    let tail = scalar::dot_u32(&a[i..], &b[i..]).unwrap_or(0);
    total.wrapping_add(tail)
}

/// L1 distance of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn l1_u32(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    // Use max - min for unsigned absolute-difference (no overflow risk).
    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    let one_vec = unsafe {
        // Build a vector of 1u32 by storing then reloading; cheaper than _mm256_set1.
        let mut buf = [1_u32; LANES_U32];
        _mm256_loadu_si256(buf.as_mut_ptr().cast::<__m256i>())
    };
    let mut i = 0;
    while i + LANES_U32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
        let mx = _mm256_max_epu32(va, vb);
        let mn = _mm256_min_epu32(va, vb);
        let d = _mm256_sub_epi32(mx, mn);
        // Multiply each diff by 1 to widen to u64 lanes via mul_epu32.
        (acc_lo, acc_hi) = unsafe { mul_accum_u32_pair(d, one_vec, acc_lo, acc_hi) };
        i += LANES_U32;
    }
    // SAFETY: AVX2 enabled.
    let total = unsafe { hadd_u64x4(_mm256_add_epi64(acc_lo, acc_hi)) };
    let tail = scalar::l1_u32(&a[i..], &b[i..]).unwrap_or(0);
    total.wrapping_add(tail)
}

/// Squared L2 distance of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc_lo = _mm256_setzero_si256();
    let mut acc_hi = _mm256_setzero_si256();
    let mut i = 0;
    while i + LANES_U32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
        let mx = _mm256_max_epu32(va, vb);
        let mn = _mm256_min_epu32(va, vb);
        let d = _mm256_sub_epi32(mx, mn);
        // Square: d * d via mul_epu32 (safe: d fits in u32, d*d fits in u64).
        (acc_lo, acc_hi) = unsafe { mul_accum_u32_pair(d, d, acc_lo, acc_hi) };
        i += LANES_U32;
    }
    // SAFETY: AVX2 enabled.
    let total = unsafe { hadd_u64x4(_mm256_add_epi64(acc_lo, acc_hi)) };
    let tail = scalar::l2_squared_u32(&a[i..], &b[i..]).unwrap_or(0);
    total.wrapping_add(tail)
}

/// Cosine similarity of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    // Compute dot, norm_a, norm_b in three SIMD passes through the data.
    // Three passes vs. one fused pass: the integer multiply-accumulate
    // pattern is L1-bandwidth-bound for these vector widths, so the
    // simpler three-pass code is competitive and easier to reason about.
    // SAFETY: AVX2 enabled and lengths match.
    let dot = unsafe { dot_u32(a, b) };
    let norm_a = unsafe { dot_u32(a, a) };
    let norm_b = unsafe { dot_u32(b, b) };
    if norm_a == 0 || norm_b == 0 {
        return 0.0;
    }
    let denom = crate::math::sqrt_f64((norm_a as f64) * (norm_b as f64));
    (dot as f64) / denom
}

/// Inner product of two `f32` vectors.
///
/// # Reduction order
///
/// 8-lane FMA accumulator; final reduction is the 8-way pairwise tree
/// `((p0 + p1) + (p2 + p3))` with `p_i = lane[2i] + lane[2i+1]`.
///
/// # Safety
///
/// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
#[must_use]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    // Prefetch lookahead: 256 bytes (64 f32 lanes) places the
    // prefetch window roughly one L1 fill latency ahead of the
    // FMA. We only fire prefetches when the input is large enough
    // to outrun the L1 streamer (about 8 KiB on a 12th-gen Intel
    // Core), and only every 4 iterations to avoid saturating the
    // load-store unit. Inside the gate, the prefetch is two
    // independent lines per fire (one per input stream).
    const PREFETCH_AHEAD_FLOATS: usize = 64;
    const PREFETCH_GROUP_ITERS: usize = 4;
    // Threshold tuned on a 12th-gen Intel: below ~64 KiB the L1
    // streamer alone keeps up with the FMA, and the prefetch
    // overhead becomes a net loss. See bench_compare's
    // similarity-dot-f32 row for the cross-over data.
    let prefetch_enable = a.len() >= 16_384;

    let mut iter_index = 0_usize;
    while i + LANES_F32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        if prefetch_enable
            && iter_index.is_multiple_of(PREFETCH_GROUP_ITERS)
            && i + LANES_F32 + PREFETCH_AHEAD_FLOATS <= a.len()
        {
            // SAFETY: bound-checked above; prefetch is hint-only.
            unsafe {
                crate::primitives::prefetch::prefetch_t0(
                    a.as_ptr().add(i + PREFETCH_AHEAD_FLOATS).cast::<u8>(),
                );
                crate::primitives::prefetch::prefetch_t0(
                    b.as_ptr().add(i + PREFETCH_AHEAD_FLOATS).cast::<u8>(),
                );
            }
        }
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += LANES_F32;
        iter_index += 1;
    }
    // SAFETY: AVX2 enabled.
    let total = unsafe { hadd_f32x8(acc) };
    let tail = scalar::dot_f32(&a[i..], &b[i..]).unwrap_or(0.0);
    total + tail
}

/// Squared L2 distance of two `f32` vectors.
///
/// Same 8-way pairwise tree reduction order as [`dot_f32`].
///
/// # Safety
///
/// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
#[must_use]
pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + LANES_F32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
        let d = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
        i += LANES_F32;
    }
    // SAFETY: AVX2 enabled.
    let total = unsafe { hadd_f32x8(acc) };
    let tail = scalar::l2_squared_f32(&a[i..], &b[i..]).unwrap_or(0.0);
    total + tail
}

/// Cosine similarity of two `f32` vectors.
///
/// # Safety
///
/// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
#[target_feature(enable = "avx2,fma")]
#[must_use]
pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: AVX2+FMA enabled, lengths match.
    let dot = unsafe { dot_f32(a, b) };
    let norm_a = unsafe { dot_f32(a, a) };
    let norm_b = unsafe { dot_f32(b, b) };
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / crate::math::sqrt_f32(norm_a * norm_b)
}

/// Hamming distance of two packed `u64` bitvector slices.
///
/// Strategy: stream-load 32 bytes (4 u64) of each input, XOR in-register,
/// run the per-byte Mula nibble-LUT popcount, accumulate. Fully fused —
/// no scratch buffer, no extra memory traffic. This is the AVX2 analog
/// of the AVX-512 VPOPCNTQ kernel.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn hamming_u64(a: &[u64], b: &[u64]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        _mm256_add_epi8, _mm256_and_si256, _mm256_extract_epi64, _mm256_sad_epu8, _mm256_set1_epi8,
        _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        _mm256_add_epi8, _mm256_and_si256, _mm256_extract_epi64, _mm256_sad_epu8, _mm256_set1_epi8,
        _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_xor_si256,
    };

    // 4 u64 words per __m256i load.
    const VEC_WORDS: usize = 4;
    // 8x unrolled = 32 words per outer iteration. Each per-byte popcount
    // is in 0..=8; eight unrolled per-byte sums fit in 0..=64 < 255 — safe
    // in u8 lanes without overflow before the SAD reduction.
    const UNROLL: usize = 8;

    let lookup: __m256i = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0F);

    let mut acc_u64 = _mm256_setzero_si256();
    let mut i = 0_usize;
    let unroll_words = VEC_WORDS * UNROLL;
    while i + unroll_words <= a.len() {
        let mut acc_u8 = _mm256_setzero_si256();
        for k in 0..UNROLL {
            // SAFETY: bounds checked by loop condition.
            let va =
                unsafe { _mm256_loadu_si256(a.as_ptr().add(i + k * VEC_WORDS).cast::<__m256i>()) };
            let vb =
                unsafe { _mm256_loadu_si256(b.as_ptr().add(i + k * VEC_WORDS).cast::<__m256i>()) };
            let x = _mm256_xor_si256(va, vb);
            let lo = _mm256_and_si256(x, low_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), low_mask);
            let lo_pc = _mm256_shuffle_epi8(lookup, lo);
            let hi_pc = _mm256_shuffle_epi8(lookup, hi);
            acc_u8 = _mm256_add_epi8(acc_u8, _mm256_add_epi8(lo_pc, hi_pc));
        }
        let zero = _mm256_setzero_si256();
        acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(acc_u8, zero));
        i += unroll_words;
    }
    while i + VEC_WORDS <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
        let x = _mm256_xor_si256(va, vb);
        let lo = _mm256_and_si256(x, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(x), low_mask);
        let lo_pc = _mm256_shuffle_epi8(lookup, lo);
        let hi_pc = _mm256_shuffle_epi8(lookup, hi);
        let per_byte = _mm256_add_epi8(lo_pc, hi_pc);
        let zero = _mm256_setzero_si256();
        acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(per_byte, zero));
        i += VEC_WORDS;
    }
    let s0 = _mm256_extract_epi64::<0>(acc_u64) as u64;
    let s1 = _mm256_extract_epi64::<1>(acc_u64) as u64;
    let s2 = _mm256_extract_epi64::<2>(acc_u64) as u64;
    let s3 = _mm256_extract_epi64::<3>(acc_u64) as u64;
    let total = s0 + s1 + s2 + s3;
    // Scalar tail (u64-aligned) using `count_ones` which lowers to POPCNT.
    let tail = scalar::hamming_u64(&a[i..], &b[i..]).unwrap_or(0);
    total + tail
}

/// Jaccard similarity of two packed `u64` bitvector slices.
///
/// Returns `f64` ∈ `[0.0, 1.0]`. By convention returns `1.0` when both
/// inputs are all-zeros.
///
/// Strategy: single fused streaming pass — load each input vector once,
/// compute `popcount(a & b)` and `popcount(a | b)` from the same SIMD
/// registers. No scratch buffer.
///
/// # Safety
///
/// Caller must ensure AVX2 is available and `a.len() == b.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn jaccard_u64(a: &[u64], b: &[u64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        _mm256_add_epi8, _mm256_and_si256, _mm256_extract_epi64, _mm256_or_si256, _mm256_sad_epu8,
        _mm256_set1_epi8, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_srli_epi16,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        _mm256_add_epi8, _mm256_and_si256, _mm256_extract_epi64, _mm256_or_si256, _mm256_sad_epu8,
        _mm256_set1_epi8, _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_srli_epi16,
    };

    const VEC_WORDS: usize = 4;
    const UNROLL: usize = 4; // smaller unroll than hamming because we double the per-iter work.

    let lookup: __m256i = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let low_mask = _mm256_set1_epi8(0x0F);

    let mut acc_and = _mm256_setzero_si256();
    let mut acc_or = _mm256_setzero_si256();
    let mut i = 0_usize;
    let unroll_words = VEC_WORDS * UNROLL;
    while i + unroll_words <= a.len() {
        let mut acc8_and = _mm256_setzero_si256();
        let mut acc8_or = _mm256_setzero_si256();
        for k in 0..UNROLL {
            // SAFETY: bounds checked.
            let va =
                unsafe { _mm256_loadu_si256(a.as_ptr().add(i + k * VEC_WORDS).cast::<__m256i>()) };
            let vb =
                unsafe { _mm256_loadu_si256(b.as_ptr().add(i + k * VEC_WORDS).cast::<__m256i>()) };
            let xa = _mm256_and_si256(va, vb);
            let xo = _mm256_or_si256(va, vb);
            // Per-byte popcount of (a & b).
            let lo_a = _mm256_and_si256(xa, low_mask);
            let hi_a = _mm256_and_si256(_mm256_srli_epi16::<4>(xa), low_mask);
            let pc_a = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo_a),
                _mm256_shuffle_epi8(lookup, hi_a),
            );
            // Per-byte popcount of (a | b).
            let lo_o = _mm256_and_si256(xo, low_mask);
            let hi_o = _mm256_and_si256(_mm256_srli_epi16::<4>(xo), low_mask);
            let pc_o = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo_o),
                _mm256_shuffle_epi8(lookup, hi_o),
            );
            acc8_and = _mm256_add_epi8(acc8_and, pc_a);
            acc8_or = _mm256_add_epi8(acc8_or, pc_o);
        }
        let zero = _mm256_setzero_si256();
        acc_and = _mm256_add_epi64(acc_and, _mm256_sad_epu8(acc8_and, zero));
        acc_or = _mm256_add_epi64(acc_or, _mm256_sad_epu8(acc8_or, zero));
        i += unroll_words;
    }
    while i + VEC_WORDS <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
        let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
        let xa = _mm256_and_si256(va, vb);
        let xo = _mm256_or_si256(va, vb);
        let lo_a = _mm256_and_si256(xa, low_mask);
        let hi_a = _mm256_and_si256(_mm256_srli_epi16::<4>(xa), low_mask);
        let pc_a = _mm256_add_epi8(
            _mm256_shuffle_epi8(lookup, lo_a),
            _mm256_shuffle_epi8(lookup, hi_a),
        );
        let lo_o = _mm256_and_si256(xo, low_mask);
        let hi_o = _mm256_and_si256(_mm256_srli_epi16::<4>(xo), low_mask);
        let pc_o = _mm256_add_epi8(
            _mm256_shuffle_epi8(lookup, lo_o),
            _mm256_shuffle_epi8(lookup, hi_o),
        );
        let zero = _mm256_setzero_si256();
        acc_and = _mm256_add_epi64(acc_and, _mm256_sad_epu8(pc_a, zero));
        acc_or = _mm256_add_epi64(acc_or, _mm256_sad_epu8(pc_o, zero));
        i += VEC_WORDS;
    }
    let mut intersect = (_mm256_extract_epi64::<0>(acc_and) as u64)
        + (_mm256_extract_epi64::<1>(acc_and) as u64)
        + (_mm256_extract_epi64::<2>(acc_and) as u64)
        + (_mm256_extract_epi64::<3>(acc_and) as u64);
    let mut union = (_mm256_extract_epi64::<0>(acc_or) as u64)
        + (_mm256_extract_epi64::<1>(acc_or) as u64)
        + (_mm256_extract_epi64::<2>(acc_or) as u64)
        + (_mm256_extract_epi64::<3>(acc_or) as u64);
    // Scalar tail.
    for (&x, &y) in a[i..].iter().zip(&b[i..]) {
        intersect += u64::from((x & y).count_ones());
        union += u64::from((x | y).count_ones());
    }
    if union == 0 {
        return 1.0;
    }
    (intersect as f64) / (union as f64)
}

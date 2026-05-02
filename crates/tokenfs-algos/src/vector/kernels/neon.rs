//! AArch64 NEON dense-distance kernels.
//!
//! Same shape as the AVX2 path: 4-lane integer / 4-lane f32 inner loop,
//! with scalar tail. NEON's natural register width is 128 bits.
//!
//! ## Numeric tolerance for SIMD reductions
//!
//! Integer kernels are bit-exact with [`super::scalar`]. Floating-point
//! kernels (`dot_f32`, `l2_squared_f32`, `cosine_similarity_f32`) use
//! a **4-way pairwise tree reduction** combined with the AArch64
//! `vaddvq_f32` horizontal add. Cross-backend tolerance follows the
//! Higham §3 / Wilkinson model: `|fl(dot) - exact| <= n * eps *
//! sum(|a_i*b_i|)`. Parity tests assert `(scalar - neon).abs() /
//! sum(|a*b|) < 1e-3`.
//!
//! Bit-exact reproducibility *within* the NEON backend is guaranteed:
//! the 4-way reduction order through `vaddvq_f32` is part of the public
//! contract.
//!
//! Hamming/Jaccard for `u64` packed bitvectors reuse the
//! [`crate::bits::popcount`] NEON VCNT kernel for the inner popcount.

use core::arch::aarch64::{
    vaddvq_f32, vaddvq_u64, vdupq_n_f32, vdupq_n_u32, vfmaq_f32, vget_high_u32, vget_low_u32,
    vld1q_f32, vld1q_u32, vmaxq_u32, vminq_u32, vmlal_u32, vsubq_f32, vsubq_u32,
};

use super::scalar;

const LANES_U32: usize = 4;
const LANES_F32: usize = 4;

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this is unconditionally true.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// Inner product of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on AArch64) and
/// `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn dot_u32(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc64 = core::arch::aarch64::vdupq_n_u64(0);
    let mut acc64_hi = core::arch::aarch64::vdupq_n_u64(0);
    let mut i = 0;
    while i + LANES_U32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { vld1q_u32(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_u32(b.as_ptr().add(i)) };
        // vmlal_u32 multiplies two uint32x2_t into uint64x2_t.
        // Process low-2 and high-2 lanes separately.
        acc64 = vmlal_u32(acc64, vget_low_u32(va), vget_low_u32(vb));
        acc64_hi = vmlal_u32(acc64_hi, vget_high_u32(va), vget_high_u32(vb));
        i += LANES_U32;
    }
    // Combine the two u64×2 accumulators and horizontally add.
    let total_vec = core::arch::aarch64::vaddq_u64(acc64, acc64_hi);
    let total = vaddvq_u64(total_vec);
    let tail = scalar::dot_u32(&a[i..], &b[i..]).unwrap_or(0);
    total.wrapping_add(tail)
}

/// L1 distance of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn l1_u32(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let one = vdupq_n_u32(1);
    let mut acc64 = core::arch::aarch64::vdupq_n_u64(0);
    let mut acc64_hi = core::arch::aarch64::vdupq_n_u64(0);
    let mut i = 0;
    while i + LANES_U32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { vld1q_u32(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_u32(b.as_ptr().add(i)) };
        // unsigned absolute difference via max - min.
        let mx = vmaxq_u32(va, vb);
        let mn = vminq_u32(va, vb);
        let d = vsubq_u32(mx, mn);
        // Multiply by 1 to widen to u64 lanes.
        acc64 = vmlal_u32(acc64, vget_low_u32(d), vget_low_u32(one));
        acc64_hi = vmlal_u32(acc64_hi, vget_high_u32(d), vget_high_u32(one));
        i += LANES_U32;
    }
    let total_vec = core::arch::aarch64::vaddq_u64(acc64, acc64_hi);
    let total = vaddvq_u64(total_vec);
    let tail = scalar::l1_u32(&a[i..], &b[i..]).unwrap_or(0);
    total.wrapping_add(tail)
}

/// Squared L2 distance of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc64 = core::arch::aarch64::vdupq_n_u64(0);
    let mut acc64_hi = core::arch::aarch64::vdupq_n_u64(0);
    let mut i = 0;
    while i + LANES_U32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { vld1q_u32(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_u32(b.as_ptr().add(i)) };
        let mx = vmaxq_u32(va, vb);
        let mn = vminq_u32(va, vb);
        let d = vsubq_u32(mx, mn);
        acc64 = vmlal_u32(acc64, vget_low_u32(d), vget_low_u32(d));
        acc64_hi = vmlal_u32(acc64_hi, vget_high_u32(d), vget_high_u32(d));
        i += LANES_U32;
    }
    let total_vec = core::arch::aarch64::vaddq_u64(acc64, acc64_hi);
    let total = vaddvq_u64(total_vec);
    let tail = scalar::l2_squared_u32(&a[i..], &b[i..]).unwrap_or(0);
    total.wrapping_add(tail)
}

/// Cosine similarity of two `u32` vectors.
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: NEON enabled, lengths match.
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
/// 4-lane FMA accumulator; final reduction is the NEON `vaddvq_f32`
/// horizontal add (4-way pairwise tree). Pinned for reproducibility.
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + LANES_F32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(i)) };
        acc = vfmaq_f32(acc, va, vb);
        i += LANES_F32;
    }
    let total = vaddvq_f32(acc);
    let tail = scalar::dot_f32(&a[i..], &b[i..]).unwrap_or(0.0);
    total + tail
}

/// Squared L2 distance of two `f32` vectors.
///
/// Same 4-way reduction order as [`dot_f32`].
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = vdupq_n_f32(0.0);
    let mut i = 0;
    while i + LANES_F32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { vld1q_f32(a.as_ptr().add(i)) };
        let vb = unsafe { vld1q_f32(b.as_ptr().add(i)) };
        let d = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, d, d);
        i += LANES_F32;
    }
    let total = vaddvq_f32(acc);
    let tail = scalar::l2_squared_f32(&a[i..], &b[i..]).unwrap_or(0.0);
    total + tail
}

/// Cosine similarity of two `f32` vectors.
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: NEON enabled, lengths match.
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
/// Strategy: stream-load 16 bytes (2 u64) of each input as a `uint8x16_t`
/// vector, XOR in-register, popcount per byte via `vcntq_u8`, accumulate
/// in a u8 vector and fold to u16 lanes via `vpaddlq_u8`. Fully fused —
/// no scratch buffer.
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn hamming_u64(a: &[u64], b: &[u64]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    use core::arch::aarch64::{
        uint8x16_t, vaddq_u8, vaddvq_u16, vcntq_u8, vdupq_n_u8, veorq_u8, vld1q_u8, vpaddlq_u8,
    };

    // 2 u64 = 16 bytes per load.
    const VEC_WORDS: usize = 2;
    // 8x unrolled = 16 words per outer iteration. Eight `vcntq_u8`
    // outputs each in 0..=8 sum to 0..=64 — safe under u8 saturation.
    const UNROLL: usize = 8;

    let mut total = 0_u64;
    let unroll_words = VEC_WORDS * UNROLL;
    let mut i = 0_usize;
    while i + unroll_words <= a.len() {
        let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
        for k in 0..UNROLL {
            // SAFETY: bounds checked.
            let va = unsafe { vld1q_u8(a.as_ptr().add(i + k * VEC_WORDS).cast::<u8>()) };
            let vb = unsafe { vld1q_u8(b.as_ptr().add(i + k * VEC_WORDS).cast::<u8>()) };
            acc_u8 = vaddq_u8(acc_u8, vcntq_u8(veorq_u8(va, vb)));
        }
        total += u64::from(vaddvq_u16(vpaddlq_u8(acc_u8)));
        i += unroll_words;
    }
    let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
    while i + VEC_WORDS <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { vld1q_u8(a.as_ptr().add(i).cast::<u8>()) };
        let vb = unsafe { vld1q_u8(b.as_ptr().add(i).cast::<u8>()) };
        acc_u8 = vaddq_u8(acc_u8, vcntq_u8(veorq_u8(va, vb)));
        i += VEC_WORDS;
    }
    total += u64::from(vaddvq_u16(vpaddlq_u8(acc_u8)));
    total + scalar::hamming_u64(&a[i..], &b[i..]).unwrap_or(0)
}

/// Jaccard similarity of two packed `u64` bitvector slices.
///
/// Returns `f64` ∈ `[0.0, 1.0]`. By convention returns `1.0` when both
/// inputs are all-zeros.
///
/// Strategy: single fused streaming pass — load each input vector once,
/// compute `popcount(a & b)` and `popcount(a | b)` from the same NEON
/// registers. No scratch buffer.
///
/// # Safety
///
/// Caller must ensure NEON is available and `a.len() == b.len()`.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn jaccard_u64(a: &[u64], b: &[u64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    use core::arch::aarch64::{
        uint8x16_t, vaddq_u8, vaddvq_u16, vandq_u8, vcntq_u8, vdupq_n_u8, vld1q_u8, vorrq_u8,
        vpaddlq_u8,
    };

    const VEC_WORDS: usize = 2;
    const UNROLL: usize = 4; // smaller unroll because we double the per-iter work.

    let mut intersect = 0_u64;
    let mut union = 0_u64;
    let unroll_words = VEC_WORDS * UNROLL;
    let mut i = 0_usize;
    while i + unroll_words <= a.len() {
        let mut acc_and: uint8x16_t = vdupq_n_u8(0);
        let mut acc_or: uint8x16_t = vdupq_n_u8(0);
        for k in 0..UNROLL {
            // SAFETY: bounds checked.
            let va = unsafe { vld1q_u8(a.as_ptr().add(i + k * VEC_WORDS).cast::<u8>()) };
            let vb = unsafe { vld1q_u8(b.as_ptr().add(i + k * VEC_WORDS).cast::<u8>()) };
            acc_and = vaddq_u8(acc_and, vcntq_u8(vandq_u8(va, vb)));
            acc_or = vaddq_u8(acc_or, vcntq_u8(vorrq_u8(va, vb)));
        }
        intersect += u64::from(vaddvq_u16(vpaddlq_u8(acc_and)));
        union += u64::from(vaddvq_u16(vpaddlq_u8(acc_or)));
        i += unroll_words;
    }
    let mut acc_and: uint8x16_t = vdupq_n_u8(0);
    let mut acc_or: uint8x16_t = vdupq_n_u8(0);
    while i + VEC_WORDS <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { vld1q_u8(a.as_ptr().add(i).cast::<u8>()) };
        let vb = unsafe { vld1q_u8(b.as_ptr().add(i).cast::<u8>()) };
        acc_and = vaddq_u8(acc_and, vcntq_u8(vandq_u8(va, vb)));
        acc_or = vaddq_u8(acc_or, vcntq_u8(vorrq_u8(va, vb)));
        i += VEC_WORDS;
    }
    intersect += u64::from(vaddvq_u16(vpaddlq_u8(acc_and)));
    union += u64::from(vaddvq_u16(vpaddlq_u8(acc_or)));
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

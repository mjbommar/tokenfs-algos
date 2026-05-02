//! x86 AVX-512 dense-distance kernels.
//!
//! Each `f32` kernel processes 16 lanes per iteration (512-bit register).
//! Tails fall back to scalar.
//!
//! ## Numeric tolerance for SIMD reductions
//!
//! Floating-point kernels (`dot_f32`, `l2_squared_f32`,
//! `cosine_similarity_f32`) use **16-way pairwise tree reduction** vs.
//! the scalar left-to-right summation, so results can differ by a few
//! ULP on long vectors.
//!
//! The cross-backend tolerance follows the Higham §3 / Wilkinson model:
//! `|fl(dot) - exact| <= n * eps * sum(|a_i*b_i|)`. Parity tests assert
//! `(scalar - avx512).abs() / sum(|a*b|) < 1e-3`.
//!
//! Bit-exact reproducibility *within* the AVX-512 backend is guaranteed:
//! the 16-way pairwise tree (built from
//! `_mm512_reduce_add_ps`'s implementation-defined order — but we
//! materialize via store + manual tree to pin the order) is part of the
//! public contract.
//!
//! Hamming/Jaccard for `u64` packed bitvectors reuse the
//! [`crate::bits::popcount`] AVX-512 VPOPCNTQ kernels for the inner
//! popcount; the wrapper XORs/ANDs/ORs the words and forwards.
//!
//! ## CPU feature requirements
//!
//! - `avx512f` for the f32 FMA / load / store / sub intrinsics.
//! - `avx512vpopcntdq` for the popcount-driven hamming/jaccard kernels
//!   (delegated to [`crate::bits::popcount::kernels::avx512`]).
//! - Both are available on Intel Ice Lake (2019+) and AMD Zen 4 (2022+).

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512, __m512i, _mm512_add_epi64, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_loadu_si512,
    _mm512_reduce_add_epi64, _mm512_setzero_ps, _mm512_setzero_si512, _mm512_storeu_ps,
    _mm512_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512, __m512i, _mm512_add_epi64, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_loadu_si512,
    _mm512_reduce_add_epi64, _mm512_setzero_ps, _mm512_setzero_si512, _mm512_storeu_ps,
    _mm512_sub_ps,
};

use super::scalar;

const LANES_F32: usize = 16;
const LANES_U64: usize = 8;

/// Returns true when AVX-512F is available at runtime.
///
/// `AVX512F` shipped on Intel Skylake-X (2017) and AMD Zen 4 (2022).
/// The hamming/jaccard kernels additionally require `avx512vpopcntdq`
/// (Ice Lake / Zen 4); they have a separate availability check.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512f")
}

/// Returns true when AVX-512F is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Returns true when AVX-512F + AVX-512VPOPCNTDQ are both available at
/// runtime — the precondition for the hamming/jaccard kernels.
#[cfg(feature = "std")]
#[must_use]
pub fn is_popcnt_available() -> bool {
    std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512vpopcntdq")
}

/// Returns true when AVX-512F + AVX-512VPOPCNTDQ are available.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_popcnt_available() -> bool {
    false
}

/// Reduces a `__m512` of sixteen `f32` lanes to a scalar `f32` using the
/// canonical 16-way pairwise tree.
///
/// The reduction tree is part of the AVX-512 public contract:
/// `(((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7)))` with `p_i = lane[2i] +
/// lane[2i+1]`. This is bit-equivalent to the AVX2 8-way tree applied
/// to the lower and upper halves and then summed.
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hadd_f32x16(v: __m512) -> f32 {
    let mut tmp = [0.0_f32; 16];
    // SAFETY: tmp is 64-byte writable; alignment-tolerant store.
    unsafe { _mm512_storeu_ps(tmp.as_mut_ptr(), v) };
    // 16-way pairwise tree reduction. Pinned order for reproducibility.
    let p0 = tmp[0] + tmp[1];
    let p1 = tmp[2] + tmp[3];
    let p2 = tmp[4] + tmp[5];
    let p3 = tmp[6] + tmp[7];
    let p4 = tmp[8] + tmp[9];
    let p5 = tmp[10] + tmp[11];
    let p6 = tmp[12] + tmp[13];
    let p7 = tmp[14] + tmp[15];
    let q0 = (p0 + p1) + (p2 + p3);
    let q1 = (p4 + p5) + (p6 + p7);
    q0 + q1
}

/// Inner product of two `f32` vectors.
///
/// # Reduction order
///
/// 16-lane FMA accumulator; final reduction is the 16-way pairwise tree
/// `((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7))` with `p_i = lane[2i] +
/// lane[2i+1]`.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + LANES_F32 <= a.len() {
        // SAFETY: bounds checked by loop condition; AVX-512F enabled.
        let va = unsafe { _mm512_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm512_loadu_ps(b.as_ptr().add(i)) };
        acc = _mm512_fmadd_ps(va, vb, acc);
        i += LANES_F32;
    }
    // SAFETY: AVX-512F enabled.
    let total = unsafe { hadd_f32x16(acc) };
    let tail = scalar::dot_f32(&a[i..], &b[i..]).unwrap_or(0.0);
    total + tail
}

/// Squared L2 distance of two `f32` vectors.
///
/// Same 16-way pairwise tree reduction order as [`dot_f32`].
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm512_setzero_ps();
    let mut i = 0;
    while i + LANES_F32 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm512_loadu_ps(a.as_ptr().add(i)) };
        let vb = unsafe { _mm512_loadu_ps(b.as_ptr().add(i)) };
        let d = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc);
        i += LANES_F32;
    }
    // SAFETY: AVX-512F enabled.
    let total = unsafe { hadd_f32x16(acc) };
    let tail = scalar::l2_squared_f32(&a[i..], &b[i..]).unwrap_or(0.0);
    total + tail
}

/// Cosine similarity of two `f32` vectors.
///
/// # Safety
///
/// Caller must ensure AVX-512F is available and `a.len() == b.len()`.
#[target_feature(enable = "avx512f")]
#[must_use]
pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // SAFETY: AVX-512F enabled, lengths match.
    let dot = unsafe { dot_f32(a, b) };
    let norm_a = unsafe { dot_f32(a, a) };
    let norm_b = unsafe { dot_f32(b, b) };
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / crate::math::sqrt_f32(norm_a * norm_b)
}

/// Hamming distance of two packed `u64` bitvector slices using AVX-512
/// VPOPCNTQ.
///
/// Strategy: stream-XOR `a ^ b` 64 bytes (8 u64) at a time, popcount
/// per-lane via `_mm512_popcnt_epi64`, accumulate. Inlined directly
/// rather than via the scratch-buffer pattern of the AVX2 kernel —
/// VPOPCNTQ's 1-cycle latency makes the fused XOR-popcount the fast
/// path, no need to materialize.
///
/// # Safety
///
/// Caller must ensure AVX-512F + AVX-512VPOPCNTDQ are available and
/// `a.len() == b.len()`.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[must_use]
pub unsafe fn hamming_u64(a: &[u64], b: &[u64]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_mm512_popcnt_epi64, _mm512_xor_si512};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_mm512_popcnt_epi64, _mm512_xor_si512};

    let mut acc0 = _mm512_setzero_si512();
    let mut acc1 = _mm512_setzero_si512();
    let mut acc2 = _mm512_setzero_si512();
    let mut acc3 = _mm512_setzero_si512();

    let mut i = 0;
    let unroll = LANES_U64 * 4;
    while i + unroll <= a.len() {
        // SAFETY: bounds checked; avx512f+vpopcntdq enabled.
        let va0 = unsafe { _mm512_loadu_si512(a.as_ptr().add(i).cast::<__m512i>()) };
        let vb0 = unsafe { _mm512_loadu_si512(b.as_ptr().add(i).cast::<__m512i>()) };
        let va1 = unsafe { _mm512_loadu_si512(a.as_ptr().add(i + LANES_U64).cast::<__m512i>()) };
        let vb1 = unsafe { _mm512_loadu_si512(b.as_ptr().add(i + LANES_U64).cast::<__m512i>()) };
        let va2 =
            unsafe { _mm512_loadu_si512(a.as_ptr().add(i + 2 * LANES_U64).cast::<__m512i>()) };
        let vb2 =
            unsafe { _mm512_loadu_si512(b.as_ptr().add(i + 2 * LANES_U64).cast::<__m512i>()) };
        let va3 =
            unsafe { _mm512_loadu_si512(a.as_ptr().add(i + 3 * LANES_U64).cast::<__m512i>()) };
        let vb3 =
            unsafe { _mm512_loadu_si512(b.as_ptr().add(i + 3 * LANES_U64).cast::<__m512i>()) };
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(va0, vb0)));
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(va1, vb1)));
        acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(_mm512_xor_si512(va2, vb2)));
        acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(_mm512_xor_si512(va3, vb3)));
        i += unroll;
    }
    while i + LANES_U64 <= a.len() {
        // SAFETY: bounds checked.
        let va = unsafe { _mm512_loadu_si512(a.as_ptr().add(i).cast::<__m512i>()) };
        let vb = unsafe { _mm512_loadu_si512(b.as_ptr().add(i).cast::<__m512i>()) };
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
        i += LANES_U64;
    }
    let sum01 = _mm512_add_epi64(acc0, acc1);
    let sum23 = _mm512_add_epi64(acc2, acc3);
    let sum = _mm512_add_epi64(sum01, sum23);
    let total = _mm512_reduce_add_epi64(sum) as u64;
    let tail = scalar::hamming_u64(&a[i..], &b[i..]).unwrap_or(0);
    total + tail
}

/// Jaccard similarity of two packed `u64` bitvector slices using AVX-512
/// VPOPCNTQ.
///
/// Returns `f64` ∈ `[0.0, 1.0]`. By convention returns `1.0` when both
/// inputs are all-zeros.
///
/// Strategy: single fused pass — popcount(a & b) and popcount(a | b)
/// computed in alternating accumulators per iteration, so the input
/// lines are touched only once.
///
/// # Safety
///
/// Caller must ensure AVX-512F + AVX-512VPOPCNTDQ are available and
/// `a.len() == b.len()`.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[must_use]
pub unsafe fn jaccard_u64(a: &[u64], b: &[u64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_mm512_and_si512, _mm512_or_si512, _mm512_popcnt_epi64};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_mm512_and_si512, _mm512_or_si512, _mm512_popcnt_epi64};

    let mut acc_and = _mm512_setzero_si512();
    let mut acc_or = _mm512_setzero_si512();

    let mut i = 0;
    while i + LANES_U64 <= a.len() {
        // SAFETY: bounds checked; avx512f+vpopcntdq enabled.
        let va = unsafe { _mm512_loadu_si512(a.as_ptr().add(i).cast::<__m512i>()) };
        let vb = unsafe { _mm512_loadu_si512(b.as_ptr().add(i).cast::<__m512i>()) };
        acc_and = _mm512_add_epi64(acc_and, _mm512_popcnt_epi64(_mm512_and_si512(va, vb)));
        acc_or = _mm512_add_epi64(acc_or, _mm512_popcnt_epi64(_mm512_or_si512(va, vb)));
        i += LANES_U64;
    }
    let mut intersect = _mm512_reduce_add_epi64(acc_and) as u64;
    let mut union = _mm512_reduce_add_epi64(acc_or) as u64;
    // Tail.
    for (&x, &y) in a[i..].iter().zip(&b[i..]) {
        intersect += u64::from((x & y).count_ones());
        union += u64::from((x | y).count_ones());
    }
    if union == 0 {
        return 1.0;
    }
    (intersect as f64) / (union as f64)
}

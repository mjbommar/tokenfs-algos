//! AVX2 modularity-gain kernel.
//!
//! 4 lanes per iteration via `_mm256_mul_epu32` (low 32 bits of
//! each 64-bit lane → 64-bit product). When the i64 fast path
//! is not eligible (see [`super::scalar::fast_path_eligible`]),
//! delegates to [`super::scalar::modularity_gains_neighbor_batch`].

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_storeu_si256,
    _mm256_sub_epi64,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_loadu_si256, _mm256_mul_epu32, _mm256_set1_epi64x, _mm256_storeu_si256,
    _mm256_sub_epi64,
};

/// 4 i64 lanes per AVX2 vector.
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

/// AVX2 implementation of the modularity-gain batch kernel.
///
/// See [`super::scalar::modularity_gains_neighbor_batch`] for
/// the score definition. Bit-exact with the scalar reference
/// (integer arithmetic; no FP reduction order in play).
///
/// # Safety
///
/// Caller must ensure AVX2 is available and that
/// `neighbor_weights.len() == neighbor_degrees.len()`.
#[target_feature(enable = "avx2")]
#[must_use]
pub unsafe fn modularity_gains_neighbor_batch(
    neighbor_weights: &[u64],
    neighbor_degrees: &[u64],
    self_degree: u64,
    m_doubled: u128,
) -> Vec<i128> {
    debug_assert_eq!(neighbor_weights.len(), neighbor_degrees.len());
    let n = neighbor_weights.len();
    if !super::scalar::fast_path_eligible(
        neighbor_weights,
        neighbor_degrees,
        self_degree,
        m_doubled,
    ) {
        return super::scalar::modularity_gains_neighbor_batch_unchecked(
            neighbor_weights,
            neighbor_degrees,
            self_degree,
            m_doubled,
        );
    }

    // Pre-allocate the output Vec; we write through the
    // backing storage directly to avoid per-element `Vec::push`
    // bookkeeping in the hot loop.
    let mut out: Vec<i128> = Vec::with_capacity(n);
    // SIMD lanes operate in i64; widen to i128 on store. This
    // avoids needing a portable 128-bit lane-wise multiply,
    // which neither AVX2 nor AVX-512 provides.
    let two_m = m_doubled as i64;
    let deg_u = self_degree as i64;
    // SAFETY: avx2 is enabled by the enclosing target_feature.
    let two_m_v = _mm256_set1_epi64x(two_m);
    let deg_u_v = _mm256_set1_epi64x(deg_u);

    let out_ptr = out.as_mut_ptr();
    let mut tmp = [0_i64; LANES];
    let mut i = 0;
    while i + LANES <= n {
        // SAFETY: bounds checked by loop condition.
        let w_v = unsafe { _mm256_loadu_si256(neighbor_weights.as_ptr().add(i).cast::<__m256i>()) };
        let d_v = unsafe { _mm256_loadu_si256(neighbor_degrees.as_ptr().add(i).cast::<__m256i>()) };
        // `_mm256_mul_epu32` multiplies the LOW 32 bits of each
        // 64-bit lane. Eligibility ensures every lane's value
        // fits in u32, so the upper 32 bits are zero and we
        // get the full product.
        let prod_w = _mm256_mul_epu32(two_m_v, w_v);
        let prod_d = _mm256_mul_epu32(deg_u_v, d_v);
        let score = _mm256_sub_epi64(prod_w, prod_d);
        // SAFETY: tmp is 32-byte writable; aligned-tolerant store.
        unsafe { _mm256_storeu_si256(tmp.as_mut_ptr().cast::<__m256i>(), score) };
        // Widen each i64 lane to i128 and write directly into
        // the pre-allocated Vec storage. Faster than four
        // `Vec::push` calls because it bypasses the length
        // bookkeeping per element.
        // SAFETY: `i + LANES <= n <= out.capacity()` ensures
        // every write is in-bounds; `out_ptr` is non-null and
        // properly aligned for `i128` (Vec::with_capacity
        // guarantees both).
        for (lane_idx, &lane) in tmp.iter().enumerate() {
            unsafe {
                out_ptr.add(i + lane_idx).write(i128::from(lane));
            }
        }
        i += LANES;
    }
    // Tail: scalar with the same eligibility-guarded i64 path.
    while i < n {
        let w = neighbor_weights[i] as i64;
        let d = neighbor_degrees[i] as i64;
        let score = two_m * w - deg_u * d;
        // SAFETY: `i < n <= out.capacity()`.
        unsafe {
            out_ptr.add(i).write(i128::from(score));
        }
        i += 1;
    }
    // SAFETY: every slot in `0..n` has been initialised above.
    unsafe {
        out.set_len(n);
    }
    out
}

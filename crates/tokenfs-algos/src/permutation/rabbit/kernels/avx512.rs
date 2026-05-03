//! AVX-512 modularity-gain kernel.
//!
//! 8 lanes per iteration via `_mm512_mullo_epi64` (native i64
//! multiply, AVX-512DQ). Falls back to scalar when the i64
//! fast path is not eligible.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_loadu_si512, _mm512_mullo_epi64, _mm512_set1_epi64, _mm512_storeu_si512,
    _mm512_sub_epi64,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_loadu_si512, _mm512_mullo_epi64, _mm512_set1_epi64, _mm512_storeu_si512,
    _mm512_sub_epi64,
};

/// 8 i64 lanes per AVX-512 vector.
const LANES: usize = 8;

/// Returns true when AVX-512F + AVX-512DQ are available.
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

/// AVX-512 implementation of the modularity-gain batch kernel.
///
/// See [`super::scalar::modularity_gains_neighbor_batch`] for
/// the score definition. Bit-exact with the scalar reference.
///
/// # Safety
///
/// Caller must ensure AVX-512F + AVX-512DQ are available and
/// that `neighbor_weights.len() == neighbor_degrees.len()`.
#[target_feature(enable = "avx512f,avx512dq")]
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

    let mut out: Vec<i128> = Vec::with_capacity(n);
    let two_m = m_doubled as i64;
    let deg_u = self_degree as i64;
    // SAFETY: avx512f+dq enabled by the enclosing target_feature.
    let two_m_v = _mm512_set1_epi64(two_m);
    let deg_u_v = _mm512_set1_epi64(deg_u);

    let out_ptr = out.as_mut_ptr();
    let mut tmp = [0_i64; LANES];
    let mut i = 0;
    while i + LANES <= n {
        // SAFETY: bounds checked.
        let w_v = unsafe { _mm512_loadu_si512(neighbor_weights.as_ptr().add(i).cast::<__m512i>()) };
        let d_v = unsafe { _mm512_loadu_si512(neighbor_degrees.as_ptr().add(i).cast::<__m512i>()) };
        let prod_w = _mm512_mullo_epi64(two_m_v, w_v);
        let prod_d = _mm512_mullo_epi64(deg_u_v, d_v);
        let score = _mm512_sub_epi64(prod_w, prod_d);
        // SAFETY: tmp is 64-byte writable; aligned-tolerant store.
        unsafe { _mm512_storeu_si512(tmp.as_mut_ptr().cast::<__m512i>(), score) };
        // SAFETY: `i + LANES <= n <= out.capacity()`.
        for (lane_idx, &lane) in tmp.iter().enumerate() {
            unsafe {
                out_ptr.add(i + lane_idx).write(i128::from(lane));
            }
        }
        i += LANES;
    }
    while i < n {
        let w = neighbor_weights[i] as i64;
        let d = neighbor_degrees[i] as i64;
        let score = two_m.wrapping_mul(w).wrapping_sub(deg_u.wrapping_mul(d));
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

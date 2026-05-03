//! AArch64 NEON modularity-gain kernel.
//!
//! 2 lanes per iteration via `vmull_u32` (widening 32x32→64
//! multiply). Falls back to scalar when the i64 fast path is
//! not eligible.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::arch::aarch64::{
    int64x2_t, uint32x2_t, vld1_u32, vmull_u32, vreinterpretq_s64_u64, vst1q_s64, vsubq_s64,
};

/// 2 i64 lanes per NEON vector.
const LANES: usize = 2;

/// Returns true when NEON is available.
///
/// NEON is mandatory on AArch64; this is unconditionally true.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// Loads two `u32` lanes from a `&[u64]` slice into a NEON
/// `uint32x2_t` register.
///
/// Eligibility-checked callers ensure each `u64` value fits in
/// `u32`, so we read the low 32 bits of each 64-bit slot via a
/// scratch buffer. A direct gather would require `vld2q_u32` /
/// `vuzp` shuffles; the scratch is simpler and the loop is
/// memory-bandwidth bound at this lane width regardless.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load_u32x2_low_from_u64(slice: &[u64], i: usize) -> uint32x2_t {
    let a = slice[i] as u32;
    let b = slice[i + 1] as u32;
    let buf = [a, b];
    // SAFETY: buf is 8 bytes contiguous, vld1_u32 reads 8 bytes.
    unsafe { vld1_u32(buf.as_ptr()) }
}

/// NEON implementation of the modularity-gain batch kernel.
///
/// See [`super::scalar::modularity_gains_neighbor_batch`] for
/// the score definition. Bit-exact with the scalar reference.
///
/// # Safety
///
/// Caller must ensure NEON is available (always true on
/// AArch64) and that `neighbor_weights.len() ==
/// neighbor_degrees.len()`.
#[target_feature(enable = "neon")]
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
        return super::scalar::modularity_gains_neighbor_batch(
            neighbor_weights,
            neighbor_degrees,
            self_degree,
            m_doubled,
        );
    }

    let mut out: Vec<i128> = Vec::with_capacity(n);
    let two_m = m_doubled as i64;
    let deg_u = self_degree as i64;
    // Broadcast the scalar low-32 of two_m / deg_u as a uint32x2_t.
    // Eligibility guarantees both fit in u32.
    let two_m_lo = [two_m as u32, two_m as u32];
    let deg_u_lo = [deg_u as u32, deg_u as u32];
    // SAFETY: arrays are 8 bytes contiguous; neon enabled.
    let two_m_v = unsafe { vld1_u32(two_m_lo.as_ptr()) };
    let deg_u_v = unsafe { vld1_u32(deg_u_lo.as_ptr()) };

    let out_ptr = out.as_mut_ptr();
    let mut tmp = [0_i64; LANES];
    let mut i = 0;
    while i + LANES <= n {
        // SAFETY: bounds checked; eligibility ensures values fit u32.
        let w_v = unsafe { load_u32x2_low_from_u64(neighbor_weights, i) };
        let d_v = unsafe { load_u32x2_low_from_u64(neighbor_degrees, i) };
        // vmull_u32 widens 32x32→64 per lane: uint32x2_t * uint32x2_t → uint64x2_t.
        let prod_w_u = vmull_u32(two_m_v, w_v);
        let prod_d_u = vmull_u32(deg_u_v, d_v);
        // Reinterpret as signed for the subtraction. The values
        // fit in u63 by eligibility, so the bit pattern is the
        // same as i64.
        let prod_w_s = vreinterpretq_s64_u64(prod_w_u);
        let prod_d_s = vreinterpretq_s64_u64(prod_d_u);
        let score: int64x2_t = vsubq_s64(prod_w_s, prod_d_s);
        // SAFETY: tmp is 16 bytes contiguous.
        unsafe { vst1q_s64(tmp.as_mut_ptr(), score) };
        // SAFETY: `i + LANES <= n <= out.capacity()`.
        for (lane_idx, &lane) in tmp.iter().enumerate() {
            unsafe {
                out_ptr.add(i + lane_idx).write(i128::from(lane));
            }
        }
        i += LANES;
    }
    // Tail (n is odd).
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

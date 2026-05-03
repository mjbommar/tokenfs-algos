use super::POW2_NEG_LUT;

use core::arch::aarch64::{vld1q_u8, vmaxq_u8, vst1q_u8};

/// 16 u8 registers per NEON vector.
const VEC_BYTES: usize = 16;

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory in the AArch64 ABI; this helper exists
/// for API symmetry with the x86 kernels.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// NEON per-bucket max merge.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON and
/// that `dst.len() == src.len()`.
#[target_feature(enable = "neon")]
pub unsafe fn merge(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();
    let mut i = 0_usize;
    while i + VEC_BYTES <= len {
        // SAFETY: `i + 16 <= len`; NEON enabled by the
        // enclosing target_feature.
        let a = unsafe { vld1q_u8(dst.as_ptr().add(i)) };
        let b = unsafe { vld1q_u8(src.as_ptr().add(i)) };
        let m = vmaxq_u8(a, b);
        // SAFETY: same range bound as the load above.
        unsafe { vst1q_u8(dst.as_mut_ptr().add(i), m) };
        i += VEC_BYTES;
    }
    // Scalar tail.
    while i < len {
        let b = src[i];
        if b > dst[i] {
            dst[i] = b;
        }
        i += 1;
    }
}

/// NEON harmonic-mean cardinality `alpha * m^2 / Z`.
///
/// AArch64 NEON has no efficient gather instruction, so the
/// inner loop runs four parallel scalar f64 accumulators
/// (driving four LUT lookups per iteration). The pipelined
/// AArch64 backends (Apple M-series, Graviton, Snapdragon)
/// extract enough ILP from this to clear the same throughput
/// the gather-based x86 backends hit.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn count_raw(registers: &[u8], alpha: f64) -> f64 {
    let m = registers.len() as f64;
    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let mut i = 0_usize;
    let n = registers.len();
    while i + 4 <= n {
        a0 += POW2_NEG_LUT[registers[i] as usize];
        a1 += POW2_NEG_LUT[registers[i + 1] as usize];
        a2 += POW2_NEG_LUT[registers[i + 2] as usize];
        a3 += POW2_NEG_LUT[registers[i + 3] as usize];
        i += 4;
    }
    let mut sum = (a0 + a1) + (a2 + a3);
    while i < n {
        sum += POW2_NEG_LUT[registers[i] as usize];
        i += 1;
    }
    alpha * m * m / sum
}

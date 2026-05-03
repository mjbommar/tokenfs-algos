use core::arch::aarch64::{uint64x2_t, vaddq_u64, vdupq_n_u64, vld1q_u64, vst1q_u64};

/// 2 u64 lanes per NEON vector.
const LANES: usize = 2;

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this exists for API symmetry
/// with the x86 `is_available` helpers.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// NEON position-computation kernel.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
///
/// # Panics
///
/// Panics if `out.len() < k` or `bits == 0`. Available only with
/// `feature = "userspace"` (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
#[target_feature(enable = "neon")]
pub unsafe fn positions(h1: u64, h2: u64, k: usize, bits: usize, out: &mut [u64]) {
    assert!(bits > 0, "BloomFilter bits must be > 0");
    assert!(out.len() >= k, "out buffer too small: {} < {k}", out.len());
    // SAFETY: precondition checked above.
    unsafe { positions_unchecked(h1, h2, k, bits, out) }
}

/// Unchecked variant of [`positions`].
///
/// # Safety
///
/// Caller must ensure NEON is available, `out.len() >= k`, and
/// `bits > 0` (audit-R10 #1 / #216).
#[target_feature(enable = "neon")]
pub unsafe fn positions_unchecked(h1: u64, h2: u64, k: usize, bits: usize, out: &mut [u64]) {
    let h1_v = vdupq_n_u64(h1);

    let mut i = 0_usize;
    while i + LANES <= k {
        // Compute `i*h2` and `(i+1)*h2` scalar (no NEON
        // 64-bit multiply pre-SVE2), pack into a vector,
        // vector-add `h1` and store the two raw u64
        // positions.
        let prod = [
            (i as u64).wrapping_mul(h2),
            ((i + 1) as u64).wrapping_mul(h2),
        ];
        // SAFETY: `prod` is on the stack with 8-byte alignment;
        // `vld1q_u64` accepts unaligned loads.
        let prod_v: uint64x2_t = unsafe { vld1q_u64(prod.as_ptr()) };
        let sum = vaddq_u64(h1_v, prod_v);
        // SAFETY: `out.len() >= k >= i + LANES` holds by the
        // loop condition.
        unsafe {
            vst1q_u64(out.as_mut_ptr().add(i), sum);
        }
        i += LANES;
    }

    // Tail (k odd).
    while i < k {
        let raw = h1.wrapping_add((i as u64).wrapping_mul(h2));
        out[i] = raw;
        i += 1;
    }

    // Stage 2: scalar modular reduction.
    let bits_u64 = bits as u64;
    for slot in out.iter_mut().take(k) {
        *slot %= bits_u64;
    }
}

use super::scalar;

use core::arch::aarch64::{
    uint8x16_t, vaddq_u8, vaddvq_u16, vcntq_u8, vdupq_n_u8, vld1q_u8, vpaddlq_u8,
};

/// 16 bytes (2 u64) per NEON vector.
const VEC_BYTES: usize = 16;

/// 8x unrolled = 128 bytes per outer iteration. Eight `vcntq_u8`
/// outputs each in 0..=8 sum to 0..=64 — safe under the u8
/// saturation threshold of 255 and large enough to amortize the
/// `vpaddlq_u8` pairwise widening cost across many bytes.
const UNROLL_VECTORS: usize = 8;

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this exists for API symmetry
/// with the x86 `is_available` helpers.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// NEON VCNT popcount over a `&[u64]` slice.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn popcount_u64_slice(words: &[u64]) -> u64 {
    let bytes_ptr = words.as_ptr().cast::<u8>();
    let bytes_len = core::mem::size_of_val(words);
    // SAFETY: `bytes_ptr`/`bytes_len` describe the same memory as
    // `words`, borrowed for the duration of this call.
    let bytes = unsafe { core::slice::from_raw_parts(bytes_ptr, bytes_len) };
    unsafe { popcount_bytes_neon(bytes) }
}

/// NEON VCNT popcount over a `&[u8]` slice.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn popcount_u8_slice(bytes: &[u8]) -> u64 {
    // SAFETY: target_feature on this fn forwards to the inner
    // kernel.
    unsafe { popcount_bytes_neon(bytes) }
}

/// Inner NEON VCNT kernel.
///
/// # Safety
///
/// NEON must be available; caller asserts via `target_feature`.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn popcount_bytes_neon(bytes: &[u8]) -> u64 {
    let mut total = 0_u64;
    let mut index = 0_usize;
    let unroll_bytes = VEC_BYTES * UNROLL_VECTORS;

    while index + unroll_bytes <= bytes.len() {
        // SAFETY: each load reads 16 bytes; the loop condition
        // bounds `index + UNROLL_VECTORS * 16 <= bytes.len()`.
        let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
        for k in 0..UNROLL_VECTORS {
            // SAFETY: as above.
            let v = unsafe { vld1q_u8(bytes.as_ptr().add(index + k * VEC_BYTES)) };
            acc_u8 = vaddq_u8(acc_u8, vcntq_u8(v));
        }
        // `vpaddlq_u8` widens 16x u8 → 8x u16 via pairwise
        // addition, avoiding u8 overflow (max sum per byte =
        // 8 * 8 = 64, sum of pairs = 128 < u16::MAX).
        let widened = vpaddlq_u8(acc_u8);
        total += u64::from(vaddvq_u16(widened));
        index += unroll_bytes;
    }

    // Single-vector loop for the residual after the unrolled
    // block. Up to UNROLL_VECTORS - 1 vectors remain, so a u8
    // accumulator stays well below saturation; no inner flush
    // is needed.
    let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
    while index + VEC_BYTES <= bytes.len() {
        // SAFETY: index + 16 <= bytes.len() bounds the load.
        let v = unsafe { vld1q_u8(bytes.as_ptr().add(index)) };
        acc_u8 = vaddq_u8(acc_u8, vcntq_u8(v));
        index += VEC_BYTES;
    }
    total += u64::from(vaddvq_u16(vpaddlq_u8(acc_u8)));

    // Scalar tail for the residual bytes < VEC_BYTES.
    total + scalar::popcount_u8_slice(&bytes[index..])
}

use super::TABLE_ROWS;

use core::arch::aarch64::{uint64x2_t, vbslq_u64, vcgtq_u64, vld1q_u64, vst1q_u64};

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this exists for API symmetry with
/// the x86 `is_available` helpers.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// Synthesized lane-wise unsigned 64-bit minimum on AArch64.
///
/// AArch64 stable Rust exposes `vminq_u32` but not `vminq_u64`.
/// We use `vcgtq_u64(a, b)` which produces 0xFFFF... per lane where
/// `a > b` (unsigned), then `vbslq_u64(mask, b, a)` selects `b`
/// where the mask is set (the smaller of the pair).
///
/// # Safety
///
/// NEON is mandatory on AArch64; the helper is `#[inline]` so the
/// caller's `target_feature` propagates.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn min_u64x2(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    // mask lanes where a > b unsigned.
    let mask = vcgtq_u64(a, b);
    // bsl: if mask bit set, take b; else take a.
    vbslq_u64(mask, b, a)
}

/// Vectorized K-min `MinHash` update for `K = 8` using NEON.
///
/// Each input byte triggers four 128-bit `vld1q_u64` loads from
/// `table[b]` (covering all 8 u64 slots) and four lane-wise
/// minimum reductions into the running signature.
///
/// # Safety
///
/// The caller must ensure NEON is available (always true on
/// AArch64) and that `sig` and `table` outlive the call.
#[target_feature(enable = "neon")]
pub unsafe fn update_minhash_8way(
    bytes: &[u8],
    table: &[[u64; 8]; TABLE_ROWS],
    sig: &mut [u64; 8],
) {
    // Four 128-bit accumulators (2 × u64 each = 8 lanes total).
    // SAFETY: `sig` is &mut [u64; 8] = 64 bytes; loads are 16-byte
    // each from contiguous offsets 0/2/4/6.
    let mut a0: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr()) };
    let mut a1: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr().add(2)) };
    let mut a2: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr().add(4)) };
    let mut a3: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr().add(6)) };

    for &b in bytes {
        let row_ptr = unsafe { table.as_ptr().add(b as usize).cast::<u64>() };
        // SAFETY: each row is 8 × u64 (64 bytes); loads cover the
        // entire row in 4 × 16-byte halves.
        let r0 = unsafe { vld1q_u64(row_ptr) };
        let r1 = unsafe { vld1q_u64(row_ptr.add(2)) };
        let r2 = unsafe { vld1q_u64(row_ptr.add(4)) };
        let r3 = unsafe { vld1q_u64(row_ptr.add(6)) };
        // SAFETY: NEON enabled by enclosing target_feature.
        a0 = unsafe { min_u64x2(a0, r0) };
        a1 = unsafe { min_u64x2(a1, r1) };
        a2 = unsafe { min_u64x2(a2, r2) };
        a3 = unsafe { min_u64x2(a3, r3) };
    }

    // SAFETY: 64 writable bytes at `sig`.
    unsafe { vst1q_u64(sig.as_mut_ptr(), a0) };
    unsafe { vst1q_u64(sig.as_mut_ptr().add(2), a1) };
    unsafe { vst1q_u64(sig.as_mut_ptr().add(4), a2) };
    unsafe { vst1q_u64(sig.as_mut_ptr().add(6), a3) };
}

/// Vectorized K-min `MinHash` update for general `K` using NEON.
///
/// Iterates the `K` slots in groups of 2 — each group is one
/// `vld1q_u64` (loading two contiguous u64 hashes from
/// `table[b][group_start..group_start+2]`) followed by a
/// synthesised lane-wise minimum reduction. Tail slots (when `K`
/// is odd) fall back to scalar updates.
///
/// `K` is a const generic so the compiler unrolls the per-group
/// loop and propagates the per-row stride constant.
///
/// # Safety
///
/// The caller must ensure NEON is available (always true on
/// AArch64).
#[target_feature(enable = "neon")]
pub unsafe fn update_minhash_kway<const K: usize>(
    bytes: &[u8],
    table: &[[u64; K]; TABLE_ROWS],
    sig: &mut [u64; K],
) {
    let groups = K / 2;
    let tail = K - groups * 2;

    for &b in bytes {
        let row_ptr = unsafe { table.as_ptr().add(b as usize).cast::<u64>() };
        for g in 0..groups {
            let base = g * 2;
            // SAFETY: row is K × u64; base + 2 <= K because g < K/2.
            let r = unsafe { vld1q_u64(row_ptr.add(base)) };
            // SAFETY: sig is K × u64 (8K bytes); load 16 bytes.
            let cur = unsafe { vld1q_u64(sig.as_ptr().add(base)) };
            // SAFETY: NEON enabled by enclosing target_feature.
            let merged = unsafe { min_u64x2(cur, r) };
            // SAFETY: 16 writable bytes at sig[base..base+2].
            unsafe { vst1q_u64(sig.as_mut_ptr().add(base), merged) };
        }
        if tail > 0 {
            let row = unsafe { &*table.as_ptr().add(b as usize) };
            let base = groups * 2;
            for k in base..K {
                let h = row[k];
                if h < sig[k] {
                    sig[k] = h;
                }
            }
        }
    }
}

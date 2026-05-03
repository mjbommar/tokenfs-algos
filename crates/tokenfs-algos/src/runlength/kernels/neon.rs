use super::scalar;

use core::arch::aarch64::{vaddlvq_u8, vandq_u8, vceqq_u8, vdupq_n_u8, vld1q_u8};

const LANES: usize = 16;

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; the function exists for API
/// symmetry with [`super::avx2::is_available`].
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// Counts transitions where `bytes[i] != bytes[i - 1]`, NEON path.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON. On AArch64
/// this is part of the base ABI; the precondition is always met for
/// `target_arch = "aarch64"` builds.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn transitions(bytes: &[u8]) -> u64 {
    if bytes.len() < 2 {
        return 0;
    }

    let mut count = 0_u64;
    let mut index = 1_usize;
    let one = vdupq_n_u8(1);

    while index + LANES <= bytes.len() {
        // SAFETY: index >= 1 and index + LANES <= bytes.len(), so
        // both loads stay inside `bytes`.
        let curr = unsafe { vld1q_u8(bytes.as_ptr().add(index)) };
        let prev = unsafe { vld1q_u8(bytes.as_ptr().add(index - 1)) };
        let eq = vceqq_u8(curr, prev);
        // eq is 0xff for equal lanes, 0x00 otherwise; AND with 1 to
        // get a per-lane 0/1 indicator, then horizontal-add as u16
        // for an equality count in 0..=16.
        let eq_count = u64::from(vaddlvq_u8(vandq_u8(eq, one)));
        count += LANES as u64 - eq_count;
        index += LANES;
    }

    count + scalar::transitions(&bytes[index - 1..])
}

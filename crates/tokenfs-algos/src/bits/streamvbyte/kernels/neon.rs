use super::super::{GROUP, streamvbyte_control_len};
use super::scalar;
use super::tables::{length_table, shuffle_table};

use core::arch::aarch64::{uint8x16_t, vld1q_u8, vqtbl1q_u8, vst1q_u8};

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this exists for API symmetry.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// NEON `vqtbl1q_u8` decode.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
#[target_feature(enable = "neon")]
pub unsafe fn decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize {
    let ctrl_needed = streamvbyte_control_len(n);
    assert!(
        control.len() >= ctrl_needed,
        "control too small: {} < {}",
        control.len(),
        ctrl_needed
    );
    assert!(
        out.len() >= n,
        "decode output buffer too small: {} < {}",
        out.len(),
        n
    );
    // SAFETY: NEON availability and buffer-length preconditions
    // are both established above.
    unsafe { decode_u32_unchecked(control, data, n, out) }
}

/// NEON `vqtbl1q_u8` decode without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure the current CPU supports NEON,
/// `control.len() >= streamvbyte_control_len(n)`,
/// `out.len() >= n`, and `data.len()` covers the implied byte
/// sum from the control stream.
#[target_feature(enable = "neon")]
pub unsafe fn decode_u32_unchecked(
    control: &[u8],
    data: &[u8],
    n: usize,
    out: &mut [u32],
) -> usize {
    let shuf = shuffle_table();
    let lens = length_table();

    let full_groups = n / GROUP;
    let mut data_pos = 0_usize;
    let mut g = 0_usize;

    while g < full_groups {
        let c = control[g] as usize;
        let len = lens[c] as usize;
        if data_pos + 16 > data.len() {
            break;
        }

        // SAFETY: bounds checked above; NEON enabled on enclosing
        // fn; output lane `g*4..g*4+4` is in-bounds because
        // `g < full_groups <= n / 4`.
        unsafe {
            let v: uint8x16_t = vld1q_u8(data.as_ptr().add(data_pos));
            let s: uint8x16_t = vld1q_u8(shuf[c].as_ptr());
            let r: uint8x16_t = vqtbl1q_u8(v, s);
            vst1q_u8(out.as_mut_ptr().add(g * GROUP).cast::<u8>(), r);
        }

        data_pos += len;
        g += 1;
    }

    let written = g * GROUP;
    if written < n {
        // SAFETY: caller upholds the buffer-length preconditions
        // on `control`, `data`, and `out`; the slice subranges
        // share their parents' validity.
        data_pos += unsafe {
            scalar::decode_u32_unchecked(
                &control[g..],
                &data[data_pos..],
                n - written,
                &mut out[written..],
            )
        };
    }

    data_pos
}

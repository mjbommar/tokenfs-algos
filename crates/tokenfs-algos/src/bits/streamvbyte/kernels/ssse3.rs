//! SSSE3 + SSE4.1 decode kernel.
//!
//! Extracted from streamvbyte.rs as part of the audit-R10 T1.3
//! follow-up to v0.4.2's #180 file-split — ssse3 was missed in
//! the original split. Now properly file-gated behind
//! `arch-pinned-kernels`.

use super::super::GROUP;
#[cfg(feature = "userspace")]
use super::super::streamvbyte_control_len;
use super::scalar;
use super::tables::{length_table, shuffle_table};

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128i, _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128i, _mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128};

/// Returns true when SSSE3 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("ssse3")
}

/// Returns true when SSSE3 is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// SSSE3 PSHUFB-based decode.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SSSE3.
///
/// Available only with `feature = "userspace"`; kernel-safe callers
/// must use [`decode_u32_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
#[target_feature(enable = "ssse3")]
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
    // SAFETY: SSSE3 availability and the buffer-length
    // preconditions are both established above.
    unsafe { decode_u32_unchecked(control, data, n, out) }
}

/// SSSE3 PSHUFB-based decode without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure the current CPU supports SSSE3,
/// `control.len() >= streamvbyte_control_len(n)`,
/// `out.len() >= n`, and `data.len()` covers the implied byte
/// sum from the control stream (so the scalar tail fallback can
/// finish without overrunning `data`).
#[target_feature(enable = "ssse3")]
pub unsafe fn decode_u32_unchecked(
    control: &[u8],
    data: &[u8],
    n: usize,
    out: &mut [u32],
) -> usize {
    let shuf = shuffle_table();
    let lens = length_table();

    let full_groups = n / GROUP;
    // The SIMD path reads 16 bytes per group regardless of how
    // many it actually consumes. The last group is decoded by
    // the scalar tail when fewer than 16 bytes of data remain
    // after `data_pos`.
    let mut data_pos = 0_usize;
    let mut g = 0_usize;

    while g < full_groups {
        let c = control[g] as usize;
        let len = lens[c] as usize;
        if data_pos + 16 > data.len() {
            break;
        }

        // SAFETY: `data_pos + 16 <= data.len()` checked above; SSSE3
        // enabled on enclosing fn; `out[g*4..g*4+4]` is in-bounds
        // because `g < full_groups <= n / 4`, so g*4+4 <= n <= out.len().
        unsafe {
            let v = _mm_loadu_si128(data.as_ptr().add(data_pos).cast::<__m128i>());
            let s = _mm_loadu_si128(shuf[c].as_ptr().cast::<__m128i>());
            let r = _mm_shuffle_epi8(v, s);
            _mm_storeu_si128(out.as_mut_ptr().add(g * GROUP).cast::<__m128i>(), r);
        }

        data_pos += len;
        g += 1;
    }

    // Scalar tail covers (a) any group that couldn't safely read
    // 16 bytes in the SIMD inner loop, and (b) the partial group
    // when `n % 4 != 0`.
    let written = g * GROUP;
    if written < n {
        // SAFETY: caller upholds the buffer-length preconditions
        // for the unchecked scalar fallback (the slice subranges
        // share their parents' validity).
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

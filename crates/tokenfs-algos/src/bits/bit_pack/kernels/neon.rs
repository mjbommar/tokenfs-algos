use super::super::encoded_len_bytes;
use super::scalar;

use core::arch::aarch64::{
    uint32x4_t, uint64x2_t, vandq_u32, vandq_u64, vdupq_n_u32, vdupq_n_u64, vld1q_s32, vld1q_s64,
    vld1q_u32, vld1q_u64, vshlq_u32, vshlq_u64, vst1q_u32, vst1q_u64,
};

/// Returns true when NEON is available at runtime.
///
/// NEON is mandatory on AArch64; this exists for API symmetry.
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// Decodes `n` values of `w` bits each from `input` into `out`.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON.
///
/// Available only with `feature = "userspace"`; kernel-safe callers
/// must use [`decode_u32_slice_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
#[target_feature(enable = "neon")]
pub unsafe fn decode_u32_slice(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    assert!((1..=32).contains(&w), "width must be 1..=32");
    let needed = encoded_len_bytes(n, w);
    assert!(
        input.len() >= needed,
        "decode input buffer too small: {} < {}",
        input.len(),
        needed
    );
    assert!(
        out.len() >= n,
        "decode output buffer too small: {} < {}",
        out.len(),
        n
    );
    // SAFETY: NEON availability and buffer-length preconditions
    // are both established above.
    unsafe { decode_u32_slice_unchecked(w, input, n, out) };
}

/// Decodes `n` values of `w` bits each from `input` into `out`
/// without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure the current CPU supports NEON,
/// `1 <= w <= 32`, `input.len() >= ceil(n * w / 8)`, and
/// `out.len() >= n`.
#[target_feature(enable = "neon")]
pub unsafe fn decode_u32_slice_unchecked(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    match w {
        1 | 2 | 4 | 8 | 16 | 32 => {
            // SAFETY: caller upholds the buffer-length precondition.
            unsafe { scalar::decode_u32_slice_unchecked(w, input, n, out) };
            return;
        }
        _ => {}
    }

    if w <= 8 {
        // SAFETY: target_feature on this fn forwards to the inner kernel.
        unsafe { decode_le8_neon(w, input, n, out) };
    } else if w <= 24 {
        // SAFETY: target_feature on this fn forwards to the inner kernel.
        unsafe { decode_le24_neon(w, input, n, out) };
    } else {
        // SAFETY: caller upholds the buffer-length precondition.
        unsafe { scalar::decode_u32_slice_unchecked(w, input, n, out) };
    }
}

/// NEON decode for `W ∈ 3..=7`. Processes 2 elements per SIMD
/// iteration via a `uint64x2_t` with per-lane negative-shift.
///
/// # Safety
///
/// NEON must be available; caller asserts via `target_feature`.
#[target_feature(enable = "neon")]
unsafe fn decode_le8_neon(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    let mask = (1_u64 << w) - 1;
    let mask_v: uint64x2_t = vdupq_n_u64(mask);

    let mut idx = 0_usize;
    let max_byte = if input.len() >= 8 { input.len() - 8 } else { 0 };

    while idx + 2 <= n {
        let bit0 = idx * w as usize;
        let bit1 = (idx + 1) * w as usize;
        let byte0 = bit0 / 8;
        let byte1 = bit1 / 8;
        if input.len() < 8 || byte1 > max_byte {
            break;
        }
        let shift0 = (bit0 % 8) as i64;
        let shift1 = (bit1 % 8) as i64;

        // SAFETY: bounded by max_byte check above.
        let raw0 = u64::from_le_bytes(unsafe {
            [
                *input.get_unchecked(byte0),
                *input.get_unchecked(byte0 + 1),
                *input.get_unchecked(byte0 + 2),
                *input.get_unchecked(byte0 + 3),
                *input.get_unchecked(byte0 + 4),
                *input.get_unchecked(byte0 + 5),
                *input.get_unchecked(byte0 + 6),
                *input.get_unchecked(byte0 + 7),
            ]
        });
        let raw1 = u64::from_le_bytes(unsafe {
            [
                *input.get_unchecked(byte1),
                *input.get_unchecked(byte1 + 1),
                *input.get_unchecked(byte1 + 2),
                *input.get_unchecked(byte1 + 3),
                *input.get_unchecked(byte1 + 4),
                *input.get_unchecked(byte1 + 5),
                *input.get_unchecked(byte1 + 6),
                *input.get_unchecked(byte1 + 7),
            ]
        });

        let data_arr = [raw0, raw1];
        let neg_arr = [-shift0, -shift1];
        // SAFETY: NEON enabled; both arrays expose 16
        // contiguous bytes for the unaligned vector load.
        let data = unsafe { vld1q_u64(data_arr.as_ptr()) };
        let neg_v = unsafe { vld1q_s64(neg_arr.as_ptr()) };

        let shifted = vshlq_u64(data, neg_v);
        let masked = vandq_u64(shifted, mask_v);

        let mut buf = [0_u64; 2];
        // SAFETY: NEON enabled; `buf` provides 16 contiguous bytes.
        unsafe { vst1q_u64(buf.as_mut_ptr(), masked) };
        // SAFETY: idx+1 < n by while-loop guard.
        unsafe {
            *out.get_unchecked_mut(idx) = buf[0] as u32;
            *out.get_unchecked_mut(idx + 1) = buf[1] as u32;
        }
        idx += 2;
    }

    if idx < n {
        tail_scalar(w, input, idx, n, out);
    }
}

/// NEON decode for `W ∈ 9..=24`. Processes 4 elements per
/// SIMD iteration via `uint32x4_t`; per-lane right shift via
/// `vshlq_u32(src, neg_shift)`.
///
/// # Safety
///
/// NEON must be available; caller asserts via `target_feature`.
#[target_feature(enable = "neon")]
unsafe fn decode_le24_neon(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    let mask = (1_u32 << w) - 1;
    let mask_v: uint32x4_t = vdupq_n_u32(mask);

    let mut idx = 0_usize;
    let max_byte = if input.len() >= 4 { input.len() - 4 } else { 0 };

    while idx + 4 <= n {
        let bits = [
            idx * w as usize,
            (idx + 1) * w as usize,
            (idx + 2) * w as usize,
            (idx + 3) * w as usize,
        ];
        let bytes = [bits[0] / 8, bits[1] / 8, bits[2] / 8, bits[3] / 8];
        if input.len() < 4 || bytes[3] > max_byte {
            break;
        }
        let neg_shifts = [
            -((bits[0] % 8) as i32),
            -((bits[1] % 8) as i32),
            -((bits[2] % 8) as i32),
            -((bits[3] % 8) as i32),
        ];
        let mut srcs = [0_u32; 4];
        for k in 0..4 {
            // SAFETY: bounded by max_byte check above.
            srcs[k] = u32::from_le_bytes(unsafe {
                [
                    *input.get_unchecked(bytes[k]),
                    *input.get_unchecked(bytes[k] + 1),
                    *input.get_unchecked(bytes[k] + 2),
                    *input.get_unchecked(bytes[k] + 3),
                ]
            });
        }
        // SAFETY: NEON enabled; arrays expose 16 contiguous
        // bytes for the unaligned vector load.
        let src_v = unsafe { vld1q_u32(srcs.as_ptr()) };
        let shift_v = unsafe { vld1q_s32(neg_shifts.as_ptr()) };

        let shifted = vshlq_u32(src_v, shift_v);
        let masked = vandq_u32(shifted, mask_v);

        let mut buf = [0_u32; 4];
        // SAFETY: NEON enabled; `buf` provides 16 contiguous bytes.
        unsafe { vst1q_u32(buf.as_mut_ptr(), masked) };
        // SAFETY: idx+3 < n by while-loop guard.
        unsafe {
            *out.get_unchecked_mut(idx) = buf[0];
            *out.get_unchecked_mut(idx + 1) = buf[1];
            *out.get_unchecked_mut(idx + 2) = buf[2];
            *out.get_unchecked_mut(idx + 3) = buf[3];
        }
        idx += 4;
    }

    if idx < n {
        tail_scalar(w, input, idx, n, out);
    }
}

fn tail_scalar(w: u32, input: &[u8], start: usize, end: usize, out: &mut [u32]) {
    let mask: u64 = if w == 32 {
        u32::MAX as u64
    } else {
        (1_u64 << w) - 1
    };
    for i in start..end {
        let bit_pos = i * (w as usize);
        let byte = bit_pos / 8;
        let shift = (bit_pos % 8) as u32;
        let span_bits = shift + w;
        let span_bytes = (span_bits as usize).div_ceil(8);
        let mut acc = 0_u64;
        for k in 0..span_bytes {
            acc |= (input[byte + k] as u64) << (8 * k);
        }
        out[i] = ((acc >> shift) & mask) as u32;
    }
}

use super::super::encoded_len_bytes;
use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_and_si256, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_setr_epi32,
    _mm256_setr_epi64x, _mm256_srlv_epi32, _mm256_srlv_epi64, _mm256_storeu_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_and_si256, _mm256_set1_epi32, _mm256_set1_epi64x, _mm256_setr_epi32,
    _mm256_setr_epi64x, _mm256_srlv_epi32, _mm256_srlv_epi64, _mm256_storeu_si256,
};

/// Returns true when AVX2 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Returns true when AVX2 is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Decodes `n` values of `w` bits each from `input` into `out`.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
///
/// Available only with `feature = "userspace"`; kernel-safe callers
/// must use [`decode_u32_slice_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
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
    // SAFETY: AVX2 availability and buffer-length preconditions
    // are both established above.
    unsafe { decode_u32_slice_unchecked(w, input, n, out) };
}

/// Decodes `n` values of `w` bits each from `input` into `out`
/// without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure the current CPU supports AVX2,
/// `1 <= w <= 32`, `input.len() >= ceil(n * w / 8)`, and
/// `out.len() >= n`.
#[target_feature(enable = "avx2")]
pub unsafe fn decode_u32_slice_unchecked(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    // Byte-aligned widths and the very small widths fall back
    // to scalar — the SIMD setup overhead exceeds the savings
    // on a memcpy-shaped workload.
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
        unsafe { decode_le8_avx2(w, input, n, out) };
    } else if w <= 24 {
        // SAFETY: target_feature on this fn forwards to the inner kernel.
        unsafe { decode_le24_avx2(w, input, n, out) };
    } else {
        // 25..=31: 5-byte spans break the u32 lane load.
        // SAFETY: caller upholds the buffer-length precondition.
        unsafe { scalar::decode_u32_slice_unchecked(w, input, n, out) };
    }
}

/// AVX2 decode for `W ∈ 3..=7` (after the byte-aligned 4-bit
/// width is forwarded to scalar): broadcast a 64-bit word then
/// shift each of four lanes by its per-lane bit offset.
///
/// # Safety
///
/// AVX2 must be available; caller asserts via `target_feature`.
#[target_feature(enable = "avx2")]
unsafe fn decode_le8_avx2(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    // Process 4 elements per SIMD iteration. Each block of 4
    // values spans `4*W` bits = at most 32 bits — comfortably
    // inside a 64-bit word, even with up to 7 bits of in-byte
    // start offset.
    let mask = (1_u32 << w) - 1;

    let group_bits = 4 * w as usize;
    let mut group = 0_usize;
    let total_groups = n / 4;

    // The last allowed `byte` index for a u64 load is `len-8`.
    let safe_groups = if input.len() >= 8 {
        let max_byte = input.len() - 8;
        let mut count = 0_usize;
        while count < total_groups {
            let start_bit = count * group_bits;
            let last_lane_byte = (start_bit + 3 * w as usize) / 8;
            if last_lane_byte > max_byte {
                break;
            }
            count += 1;
        }
        count
    } else {
        0
    };

    while group < safe_groups {
        let start_bit = group * group_bits;
        let byte = start_bit / 8;
        let shift0 = (start_bit % 8) as i64;

        // SAFETY: bounded by `safe_groups`.
        let raw = u64::from_le_bytes(unsafe {
            [
                *input.get_unchecked(byte),
                *input.get_unchecked(byte + 1),
                *input.get_unchecked(byte + 2),
                *input.get_unchecked(byte + 3),
                *input.get_unchecked(byte + 4),
                *input.get_unchecked(byte + 5),
                *input.get_unchecked(byte + 6),
                *input.get_unchecked(byte + 7),
            ]
        });
        let broadcast = _mm256_set1_epi64x(raw as i64);

        let shifts = _mm256_setr_epi64x(
            shift0,
            shift0 + w as i64,
            shift0 + 2 * w as i64,
            shift0 + 3 * w as i64,
        );
        let shifted = _mm256_srlv_epi64(broadcast, shifts);

        // Mask each 64-bit lane down to W bits via a 64-bit
        // wide AND. The result is in the low 32 bits of each
        // 64-bit lane; we extract via a stack store.
        let masked = _mm256_and_si256(shifted, _mm256_set1_epi64x(mask as i64));

        let mut buf = [0_u64; 4];
        // SAFETY: AVX2 enabled; `buf` provides 32 contiguous
        // bytes for the unaligned store.
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr().cast::<__m256i>(), masked) };
        let dst = group * 4;
        // SAFETY: dst+3 < n by construction.
        unsafe {
            *out.get_unchecked_mut(dst) = buf[0] as u32;
            *out.get_unchecked_mut(dst + 1) = buf[1] as u32;
            *out.get_unchecked_mut(dst + 2) = buf[2] as u32;
            *out.get_unchecked_mut(dst + 3) = buf[3] as u32;
        }
        group += 1;
    }

    let done = safe_groups * 4;
    if done < n {
        tail_scalar(w, input, done, n, out);
    }
}

/// AVX2 decode for `W ∈ 9..=24`. Each iteration extracts 8
/// values via per-lane VPSRLVD over u32 lanes. Each value is
/// 9..=24 bits, so a 4-byte (`u32`) load per lane covers the
/// span (`24 + 7 = 31` bits ≤ 4 bytes).
///
/// `W ∈ {25..=31}` would need a 5-byte source per lane, which
/// breaks the per-lane u32 load — the entry point dispatches
/// those widths to scalar.
///
/// # Safety
///
/// AVX2 must be available; caller asserts via `target_feature`.
#[target_feature(enable = "avx2")]
unsafe fn decode_le24_avx2(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    let mask = (1_u32 << w) - 1;
    let mask_v = _mm256_set1_epi32(mask as i32);

    let mut group = 0_usize;
    let total_groups = n / 8;
    let safe_groups = if input.len() >= 4 {
        let max_byte = input.len() - 4;
        let mut count = 0_usize;
        while count < total_groups {
            let start_bit = 8 * count * w as usize;
            let last_lane_byte = (start_bit + 7 * w as usize) / 8;
            if last_lane_byte > max_byte {
                break;
            }
            count += 1;
        }
        count
    } else {
        0
    };

    while group < safe_groups {
        let start_bit = 8 * group * w as usize;
        let mut srcs = [0_u32; 8];
        let mut shifts_arr = [0_i32; 8];
        for k in 0..8 {
            let bit = start_bit + k * w as usize;
            let byte = bit / 8;
            shifts_arr[k] = (bit % 8) as i32;
            // SAFETY: bounded by safe_groups.
            srcs[k] = u32::from_le_bytes(unsafe {
                [
                    *input.get_unchecked(byte),
                    *input.get_unchecked(byte + 1),
                    *input.get_unchecked(byte + 2),
                    *input.get_unchecked(byte + 3),
                ]
            });
        }
        let src_v = _mm256_setr_epi32(
            srcs[0] as i32,
            srcs[1] as i32,
            srcs[2] as i32,
            srcs[3] as i32,
            srcs[4] as i32,
            srcs[5] as i32,
            srcs[6] as i32,
            srcs[7] as i32,
        );
        let shift_v = _mm256_setr_epi32(
            shifts_arr[0],
            shifts_arr[1],
            shifts_arr[2],
            shifts_arr[3],
            shifts_arr[4],
            shifts_arr[5],
            shifts_arr[6],
            shifts_arr[7],
        );
        let shifted = _mm256_srlv_epi32(src_v, shift_v);
        let masked = _mm256_and_si256(shifted, mask_v);

        let dst = group * 8;
        let mut buf = [0_u32; 8];
        // SAFETY: AVX2 enabled; `buf` provides 32 contiguous
        // bytes for the unaligned store.
        unsafe { _mm256_storeu_si256(buf.as_mut_ptr().cast::<__m256i>(), masked) };
        // SAFETY: dst+7 < n by construction.
        unsafe {
            *out.get_unchecked_mut(dst) = buf[0];
            *out.get_unchecked_mut(dst + 1) = buf[1];
            *out.get_unchecked_mut(dst + 2) = buf[2];
            *out.get_unchecked_mut(dst + 3) = buf[3];
            *out.get_unchecked_mut(dst + 4) = buf[4];
            *out.get_unchecked_mut(dst + 5) = buf[5];
            *out.get_unchecked_mut(dst + 6) = buf[6];
            *out.get_unchecked_mut(dst + 7) = buf[7];
        }
        group += 1;
    }

    let done = safe_groups * 8;
    if done < n {
        tail_scalar(w, input, done, n, out);
    }
}

/// Decodes elements `[start, end)` via the scalar reference,
/// writing into `out[start..end]`.
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

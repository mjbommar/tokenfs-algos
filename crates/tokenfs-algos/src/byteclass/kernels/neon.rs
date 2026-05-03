use super::scalar;
use crate::byteclass::ByteClassCounts;

use core::arch::aarch64::{
    uint8x16_t, vaddlvq_u8, vandq_u8, vceqq_u8, vcgeq_u8, vcltq_u8, vdupq_n_u8, vld1q_u8, vmvnq_u8,
    vorrq_u8,
};

const LANES: usize = 32;

/// Returns true when NEON is available at runtime.
///
/// On AArch64, NEON is mandatory (it's part of the base ABI), so
/// this is unconditionally true. The function exists for API
/// symmetry with [`super::avx2::is_available`].
#[must_use]
pub const fn is_available() -> bool {
    true
}

/// Counts coarse byte classes with NEON.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON. On AArch64
/// this is part of the base ABI; the precondition is always met for
/// `target_arch = "aarch64"` builds.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn classify(bytes: &[u8]) -> ByteClassCounts {
    let mut counts = ByteClassCounts::default();
    let mut index = 0;

    let one = vdupq_n_u8(1);
    let space = vdupq_n_u8(b' ');
    let tab = vdupq_n_u8(b'\t');
    let newline = vdupq_n_u8(b'\n');
    let carriage_return = vdupq_n_u8(b'\r');
    let delete = vdupq_n_u8(0x7f);
    let high_bit_threshold = vdupq_n_u8(0x80);

    while index + LANES <= bytes.len() {
        // SAFETY: index + 32 <= bytes.len() bounds both 16-byte loads.
        let v0 = unsafe { vld1q_u8(bytes.as_ptr().add(index)) };
        let v1 = unsafe { vld1q_u8(bytes.as_ptr().add(index + 16)) };

        // SAFETY: target_feature("neon") is enabled on this fn; the
        // helper requires the same precondition.
        unsafe {
            accumulate(
                v0,
                &mut counts,
                one,
                space,
                tab,
                newline,
                carriage_return,
                delete,
                high_bit_threshold,
            );
            accumulate(
                v1,
                &mut counts,
                one,
                space,
                tab,
                newline,
                carriage_return,
                delete,
                high_bit_threshold,
            );
        }

        index += LANES;
    }

    scalar::add(&bytes[index..], &mut counts);
    counts
}

/// Folds one 16-byte vector into the running [`ByteClassCounts`].
///
/// # Safety
///
/// Must be called from a function tagged `target_feature = "neon"`.
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate(
    chunk: uint8x16_t,
    counts: &mut ByteClassCounts,
    one: uint8x16_t,
    space: uint8x16_t,
    tab: uint8x16_t,
    newline: uint8x16_t,
    carriage_return: uint8x16_t,
    delete: uint8x16_t,
    high_bit_threshold: uint8x16_t,
) {
    // High-bit (>= 0x80): one-byte indicator per lane, horizontally summed.
    let high_bit = vcgeq_u8(chunk, high_bit_threshold);
    let high_bit_count = vaddlvq_u8(vandq_u8(high_bit, one)) as u64;

    // Whitespace = space | tab | newline | carriage_return.
    let is_space = vceqq_u8(chunk, space);
    let is_tab = vceqq_u8(chunk, tab);
    let is_nl = vceqq_u8(chunk, newline);
    let is_cr = vceqq_u8(chunk, carriage_return);
    let whitespace = vorrq_u8(vorrq_u8(is_space, is_tab), vorrq_u8(is_nl, is_cr));
    let whitespace_count = vaddlvq_u8(vandq_u8(whitespace, one)) as u64;

    // Low control = byte < 0x20 AND high-bit == 0.
    let is_low = vcltq_u8(chunk, space);
    let low_control = vandq_u8(is_low, vmvnq_u8(high_bit));
    // DEL (0x7f) is also classed as control; subtract it from whitespace overlap.
    let is_delete = vceqq_u8(chunk, delete);
    // Control = (low_control | delete) & !whitespace.
    let control_raw = vorrq_u8(low_control, is_delete);
    let control = vandq_u8(control_raw, vmvnq_u8(whitespace));
    let control_count = vaddlvq_u8(vandq_u8(control, one)) as u64;

    // Printable = !(high_bit | low_control | delete | whitespace).
    // Equivalently: !high_bit & !low_control & !is_delete & !whitespace.
    let not_high = vmvnq_u8(high_bit);
    let not_low = vmvnq_u8(low_control);
    let not_del = vmvnq_u8(is_delete);
    let not_ws = vmvnq_u8(whitespace);
    let printable = vandq_u8(vandq_u8(not_high, not_low), vandq_u8(not_del, not_ws));
    let printable_count = vaddlvq_u8(vandq_u8(printable, one)) as u64;

    counts.high_bit += high_bit_count;
    counts.whitespace += whitespace_count;
    counts.control += control_count;
    counts.printable_ascii += printable_count;
}

/// Validates UTF-8 with the NEON Keiser-Lemire 3-pshufb DFA.
///
/// Returns the same triple as [`super::scalar::validate_utf8`] /
/// `core::str::from_utf8`. Implementation lives in
/// [`crate::byteclass::utf8_neon`]; on detected error, falls back
/// to scalar diagnosis to recover precise `valid_up_to` and
/// `error_len`.
///
/// # Safety
///
/// The caller must ensure the current CPU supports NEON. On
/// AArch64 NEON is mandatory; the precondition is always met for
/// `target_arch = "aarch64"` builds.
#[target_feature(enable = "neon")]
#[must_use]
pub unsafe fn validate_utf8(bytes: &[u8]) -> crate::byteclass::Utf8Validation {
    // SAFETY: target_feature("neon") propagates the requirement to
    // the inner module-level entry point.
    unsafe { crate::byteclass::utf8_neon::validate_utf8(bytes) }
}

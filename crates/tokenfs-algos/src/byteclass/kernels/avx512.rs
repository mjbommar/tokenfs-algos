use super::scalar;
use crate::byteclass::ByteClassCounts;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_cmpeq_epi8_mask, _mm512_cmplt_epi8_mask, _mm512_loadu_si512,
    _mm512_movepi8_mask, _mm512_set1_epi8,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_cmpeq_epi8_mask, _mm512_cmplt_epi8_mask, _mm512_loadu_si512,
    _mm512_movepi8_mask, _mm512_set1_epi8,
};

const LANES: usize = 64;

/// Returns true when AVX-512BW is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx512bw")
}

/// Returns true when AVX-512BW is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Counts coarse byte classes with AVX-512BW.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512BW.
#[target_feature(enable = "avx512bw")]
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub unsafe fn classify(bytes: &[u8]) -> ByteClassCounts {
    let mut counts = ByteClassCounts::default();
    let mut index = 0;

    let space = _mm512_set1_epi8(b' ' as i8);
    let tab = _mm512_set1_epi8(b'\t' as i8);
    let newline = _mm512_set1_epi8(b'\n' as i8);
    let carriage_return = _mm512_set1_epi8(b'\r' as i8);
    let delete = _mm512_set1_epi8(0x7f_i8);

    while index + LANES <= bytes.len() {
        // SAFETY: `index + 64 <= bytes.len()`; AVX-512BW enabled.
        let chunk = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast::<__m512i>()) };

        // High bit: extract sign bits — equivalent to AVX2's
        // `_mm256_movemask_epi8` but as a 64-bit mask.
        let high_bit_mask = _mm512_movepi8_mask(chunk);
        // `cmplt(space, chunk)` ⇒ `chunk < space` ⇒ low ASCII
        // controls. Mask off bytes whose high bit is set so we
        // only see 0x00..0x1f (the AVX2 path uses `cmpgt(space,
        // chunk)`; both treat bytes as signed but high-bit bytes
        // appear as negative, so they wrap into "less than" any
        // small positive byte and are rejected via the AND).
        let low_control_mask = _mm512_cmplt_epi8_mask(chunk, space) & !high_bit_mask;
        let space_mask = _mm512_cmpeq_epi8_mask(chunk, space);
        let whitespace_mask = space_mask
            | _mm512_cmpeq_epi8_mask(chunk, tab)
            | _mm512_cmpeq_epi8_mask(chunk, newline)
            | _mm512_cmpeq_epi8_mask(chunk, carriage_return);
        let delete_mask = _mm512_cmpeq_epi8_mask(chunk, delete);
        let control_mask = (low_control_mask | delete_mask) & !whitespace_mask;
        let printable_mask = !(high_bit_mask | low_control_mask | delete_mask | space_mask);

        counts.high_bit += u64::from(high_bit_mask.count_ones());
        counts.whitespace += u64::from(whitespace_mask.count_ones());
        counts.control += u64::from(control_mask.count_ones());
        counts.printable_ascii += u64::from(printable_mask.count_ones());

        index += LANES;
    }

    scalar::add(&bytes[index..], &mut counts);
    counts
}

/// Validates UTF-8 with the AVX-512BW Keiser-Lemire 3-pshufb DFA.
///
/// Returns the same triple as [`super::scalar::validate_utf8`] /
/// `core::str::from_utf8`. On detected error, falls back to scalar
/// diagnosis to recover precise `valid_up_to` and `error_len`.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512BW.
#[target_feature(enable = "avx512bw")]
#[must_use]
pub unsafe fn validate_utf8(bytes: &[u8]) -> crate::byteclass::Utf8Validation {
    // SAFETY: `target_feature(enable = "avx512bw")` propagates the
    // AVX-512BW requirement to the inner module-level entry point.
    unsafe { crate::byteclass::utf8_avx512::validate_utf8(bytes) }
}

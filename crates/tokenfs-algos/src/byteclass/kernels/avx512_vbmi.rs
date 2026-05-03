use crate::byteclass::ByteClassCounts;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_mask_blend_epi8,
    _mm512_movepi8_mask, _mm512_permutex2var_epi8, _mm512_set1_epi8,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_cmpeq_epi8_mask, _mm512_loadu_si512, _mm512_mask_blend_epi8,
    _mm512_movepi8_mask, _mm512_permutex2var_epi8, _mm512_set1_epi8,
};

const LANES: usize = 64;

/// Maximum number of distinct class IDs supported by the LUT
/// kernels. Class IDs in tables passed to [`classify_with_lut`]
/// must be in `0..MAX_CLASSES`.
pub const MAX_CLASSES: usize = 16;

/// Returns true when AVX-512 VBMI is available at runtime.
///
/// VBMI is **not** implied by AVX-512BW. AMD Zen 4 has BW but not
/// VBMI; Intel Ice Lake and later do.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    // VBMI is the precondition for `_mm512_permutex2var_epi8`
    // (`vpermi2b`); BW is required for the per-class compare-mask
    // intrinsics; the F base is implied by both.
    std::is_x86_feature_detected!("avx512vbmi") && std::is_x86_feature_detected!("avx512bw")
}

/// Returns true when AVX-512 VBMI is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Loads a contiguous 64-byte slice into an `__m512i`.
///
/// # Safety
///
/// Requires AVX-512 VBMI (and implicitly BW + F). `src` must be
/// readable for at least 64 bytes.
#[target_feature(enable = "avx512vbmi,avx512bw")]
#[inline]
unsafe fn load_table_chunk(src: &[u8; 64]) -> __m512i {
    // SAFETY: `src` is a 64-byte array, which is the exact input
    // width of `_mm512_loadu_si512`.
    unsafe { _mm512_loadu_si512(src.as_ptr().cast::<__m512i>()) }
}

/// Counts coarse byte classes against an arbitrary `[u8; 256]`
/// class-index table.
///
/// `class_table[b]` is the class index assigned to byte value `b`.
/// Indices must be in `0..MAX_CLASSES`. The returned array is
/// indexed by class: `counts[c]` is the number of input bytes that
/// mapped to class `c`. Entries beyond the maximum class index
/// used by the table remain zero.
///
/// # Algorithm
///
/// 1. The 256-byte `class_table` is split into four 64-byte halves.
///    The first two form the "low" pair (covering byte values
///    `0x00..0x7F`); the last two form the "high" pair (covering
///    `0x80..0xFF`).
/// 2. For each 64-byte input chunk:
///    * `lo = vpermi2b(chunk, low_pair)` — looks up `chunk[i]` for
///      bytes in `0x00..0x7F`. `vpermi2b` ignores the high bit of
///      its index, so this also produces a (wrong) value for bytes
///      `0x80..0xFF`.
///    * `hi = vpermi2b(chunk, high_pair)` — same shape, with the
///      high half of the table.
///    * `mask = movepi8_mask(chunk)` — high-bit-of-each-lane.
///    * `classes = mask_blend(mask, lo, hi)` — picks `hi` where
///      the input byte's high bit is set, `lo` otherwise.
/// 3. For each class `c` in `0..MAX_CLASSES`:
///    `counts[c] += popcnt(cmpeq_mask(classes, splat(c)))`.
///
/// The trailing tail (`bytes.len() % 64`) is handled by a scalar
/// loop against the same table.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX-512 VBMI
/// (see [`is_available`]). Entries of `class_table` that are
/// `>= MAX_CLASSES` are silently discarded (the per-class
/// `cmpeq` loop only matches indices `0..MAX_CLASSES`); the
/// returned counts therefore may not sum to `bytes.len()` if
/// out-of-range entries were used.
#[target_feature(enable = "avx512vbmi,avx512bw")]
#[must_use]
#[allow(clippy::cast_possible_wrap)]
pub unsafe fn classify_with_lut(bytes: &[u8], class_table: &[u8; 256]) -> [u64; MAX_CLASSES] {
    let mut counts = [0_u64; MAX_CLASSES];

    // Pre-load the four 64-byte table halves once. The four
    // sub-slices each have length exactly 64; the `try_into`
    // calls into `&[u8; 64]` cannot fail. We use
    // `unwrap_unchecked` to avoid a panic branch that the
    // compiler might otherwise leave around the inlined loads.
    //
    // SAFETY: each of `[0..64]`, `[64..128]`, `[128..192]`,
    // `[192..256]` slices a length-256 array at length-64
    // strides, so the conversion to `&[u8; 64]` always
    // succeeds. `load_table_chunk` requires `avx512vbmi+bw`,
    // which is asserted by this function's `target_feature`.
    let t0_arr: &[u8; 64] = unsafe { (&class_table[0..64]).try_into().unwrap_unchecked() };
    let t1_arr: &[u8; 64] = unsafe { (&class_table[64..128]).try_into().unwrap_unchecked() };
    let t2_arr: &[u8; 64] = unsafe { (&class_table[128..192]).try_into().unwrap_unchecked() };
    let t3_arr: &[u8; 64] = unsafe { (&class_table[192..256]).try_into().unwrap_unchecked() };
    // SAFETY: target_feature(enable = "avx512vbmi,avx512bw")
    // propagates to the helper.
    let (t0, t1, t2, t3) = unsafe {
        (
            load_table_chunk(t0_arr),
            load_table_chunk(t1_arr),
            load_table_chunk(t2_arr),
            load_table_chunk(t3_arr),
        )
    };

    // Splat constants for each candidate class index.
    // `_mm512_set1_epi8` is a safe-to-call `pub fn` when the
    // enclosing function has `target_feature(enable = "avx512bw")`,
    // which `target_feature(enable = "avx512vbmi,avx512bw")` does.
    let class_splats: [__m512i; MAX_CLASSES] = core::array::from_fn(|i| _mm512_set1_epi8(i as i8));

    let mut index = 0;
    while index + LANES <= bytes.len() {
        // SAFETY: `index + 64 <= bytes.len()`; the unaligned 64-byte
        // load reads from `bytes` which the bounds check above
        // proves is in range.
        let chunk = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast::<__m512i>()) };

        // Two LUT halves; blend by high-bit of each input byte.
        // All four intrinsics are safe to call under
        // `target_feature(avx512vbmi,avx512bw)`.
        let lo = _mm512_permutex2var_epi8(t0, chunk, t1);
        let hi = _mm512_permutex2var_epi8(t2, chunk, t3);
        let high_bit_mask = _mm512_movepi8_mask(chunk);
        let classes = _mm512_mask_blend_epi8(high_bit_mask, lo, hi);

        // Per-class popcount via cmpeq-mask.
        for (c, splat_c) in class_splats.iter().enumerate() {
            let class_mask = _mm512_cmpeq_epi8_mask(classes, *splat_c);
            counts[c] += u64::from(class_mask.count_ones());
        }

        index += LANES;
    }

    // Scalar tail.
    for &byte in &bytes[index..] {
        let c = class_table[byte as usize] as usize;
        if c < MAX_CLASSES {
            counts[c] += 1;
        }
    }

    counts
}

/// Convenience wrapper that translates the LUT result back into
/// the legacy [`ByteClassCounts`] shape, given a `class_table`
/// built by [`super::super::printable_control_whitespace_high_bit_table`].
///
/// # Safety
///
/// Same precondition as [`classify_with_lut`].
#[target_feature(enable = "avx512vbmi,avx512bw")]
#[must_use]
pub unsafe fn classify(bytes: &[u8]) -> ByteClassCounts {
    let table = super::super::printable_control_whitespace_high_bit_table();
    // SAFETY: target_feature(enable = "avx512vbmi,avx512bw") is set.
    let counts = unsafe { classify_with_lut(bytes, &table) };
    ByteClassCounts {
        printable_ascii: counts[super::super::CLASS_PRINTABLE as usize],
        whitespace: counts[super::super::CLASS_WHITESPACE as usize],
        control: counts[super::super::CLASS_CONTROL as usize],
        high_bit: counts[super::super::CLASS_HIGH_BIT as usize],
        other: 0,
    }
}

/// Validates UTF-8 with the AVX-512 VBMI fused-table DFA.
///
/// Same triple as [`super::scalar::validate_utf8`] /
/// `core::str::from_utf8`. Single 256-entry vpermi2b lookup
/// replaces the AVX-512BW path's two-shuffle pair.
///
/// # Safety
///
/// Caller must ensure both AVX-512BW and AVX-512 VBMI are
/// available on the current CPU.
#[target_feature(enable = "avx512bw,avx512vbmi")]
#[must_use]
pub unsafe fn validate_utf8(bytes: &[u8]) -> crate::byteclass::Utf8Validation {
    // SAFETY: target_feature(enable = "avx512bw,avx512vbmi") on
    // this function propagates both requirements to the inner
    // module-level entry point.
    unsafe { crate::byteclass::utf8_avx512::validate_utf8_vbmi(bytes) }
}

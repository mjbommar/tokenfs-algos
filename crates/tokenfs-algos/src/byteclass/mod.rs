//! Byte classification primitives.
//!
//! The public [`classify`] path uses runtime dispatch when a tested optimized
//! backend is available. Pinned kernels live under [`kernels`] for
//! reproducible benchmarks and forensic comparisons.

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
mod utf8_avx2;

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
mod utf8_avx512;

#[cfg(all(feature = "neon", target_arch = "aarch64"))]
mod utf8_neon;

/// Counts coarse byte classes in one pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ByteClassCounts {
    /// Printable ASCII bytes, excluding bytes counted as whitespace.
    pub printable_ascii: u64,
    /// ASCII whitespace bytes.
    pub whitespace: u64,
    /// ASCII control bytes excluding whitespace.
    pub control: u64,
    /// Bytes with the high bit set.
    pub high_bit: u64,
    /// Other bytes.
    pub other: u64,
}

/// UTF-8 validation summary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Utf8Validation {
    /// True when the entire byte slice is valid UTF-8.
    pub valid: bool,
    /// Number of valid bytes before the first error.
    pub valid_up_to: usize,
    /// Error length in bytes. Zero means valid or incomplete trailing sequence.
    pub error_len: u8,
}

impl Utf8Validation {
    /// Returns true when validation ended at an incomplete trailing sequence.
    #[must_use]
    pub const fn is_incomplete(self) -> bool {
        !self.valid && self.error_len == 0
    }
}

impl ByteClassCounts {
    /// Counts all bytes in the class summary.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.printable_ascii + self.whitespace + self.control + self.high_bit + self.other
    }
}

/// Byte-class kernels.
pub mod kernels {
    /// Runtime-dispatched byte-class classifier.
    pub mod auto {
        use crate::byteclass::{ByteClassCounts, Utf8Validation};

        /// Counts coarse byte classes using the best available kernel.
        #[must_use]
        pub fn classify(bytes: &[u8]) -> ByteClassCounts {
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512::classify(bytes) };
                }
            }

            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::classify(bytes) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                // NEON is part of the base AArch64 ABI, so `is_available()`
                // is unconditionally true. The check is kept for API
                // symmetry with the AVX2 path.
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64 builds.
                    return unsafe { super::neon::classify(bytes) };
                }
            }

            super::scalar::classify(bytes)
        }

        /// Validates UTF-8 using the best available kernel.
        #[must_use]
        pub fn validate_utf8(bytes: &[u8]) -> Utf8Validation {
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512_vbmi::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512_vbmi::validate_utf8(bytes) };
                }
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512::validate_utf8(bytes) };
                }
            }

            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::validate_utf8(bytes) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64 builds.
                    return unsafe { super::neon::validate_utf8(bytes) };
                }
            }

            super::scalar::validate_utf8(bytes)
        }
    }

    /// Portable scalar byte-class classifier.
    pub mod scalar {
        use crate::byteclass::{ByteClassCounts, Utf8Validation};

        /// Counts coarse byte classes in one scalar pass.
        #[must_use]
        pub fn classify(bytes: &[u8]) -> ByteClassCounts {
            let mut counts = ByteClassCounts::default();
            add(bytes, &mut counts);
            counts
        }

        pub(super) fn add(bytes: &[u8], counts: &mut ByteClassCounts) {
            for &byte in bytes {
                match byte {
                    b'\t' | b'\n' | b'\r' | b' ' => counts.whitespace += 1,
                    0x20..=0x7e => counts.printable_ascii += 1,
                    0x00..=0x1f | 0x7f => counts.control += 1,
                    0x80..=0xff => counts.high_bit += 1,
                }
            }
        }

        /// Validates UTF-8 with the scalar reference path.
        #[must_use]
        pub fn validate_utf8(bytes: &[u8]) -> Utf8Validation {
            match core::str::from_utf8(bytes) {
                Ok(_) => Utf8Validation {
                    valid: true,
                    valid_up_to: bytes.len(),
                    error_len: 0,
                },
                Err(error) => Utf8Validation {
                    valid: false,
                    valid_up_to: error.valid_up_to(),
                    error_len: error.error_len().unwrap_or(0) as u8,
                },
            }
        }
    }

    /// AVX2 byte-class classifier.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::scalar;
        use crate::byteclass::ByteClassCounts;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m256i, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_loadu_si256,
            _mm256_movemask_epi8, _mm256_set1_epi8,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m256i, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_loadu_si256,
            _mm256_movemask_epi8, _mm256_set1_epi8,
        };

        const LANES: usize = 32;

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

        /// Counts coarse byte classes with AVX2.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn classify(bytes: &[u8]) -> ByteClassCounts {
            let mut counts = ByteClassCounts::default();
            let mut index = 0;

            let space = _mm256_set1_epi8(b' ' as i8);
            let tab = _mm256_set1_epi8(b'\t' as i8);
            let newline = _mm256_set1_epi8(b'\n' as i8);
            let carriage_return = _mm256_set1_epi8(b'\r' as i8);
            let delete = _mm256_set1_epi8(0x7f_i8);

            while index + LANES <= bytes.len() {
                let chunk =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };

                let high_bit_mask = _mm256_movemask_epi8(chunk) as u32;
                let low_control_mask =
                    (_mm256_movemask_epi8(_mm256_cmpgt_epi8(space, chunk)) as u32) & !high_bit_mask;
                let space_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, space)) as u32;
                let whitespace_mask = space_mask
                    | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, tab)) as u32)
                    | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newline)) as u32)
                    | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, carriage_return)) as u32);
                let delete_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, delete)) as u32;
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

        /// Validates UTF-8 with the AVX2 Keiser-Lemire 3-pshufb DFA.
        ///
        /// Returns the same triple as [`super::scalar::validate_utf8`] /
        /// `core::str::from_utf8`. On detected error, falls back to scalar
        /// diagnosis to recover precise `valid_up_to` and `error_len`.
        ///
        /// # When this wins (and when it doesn't)
        ///
        /// On valid UTF-8 text at multi-KiB lengths this is roughly 2× the
        /// scalar `core::str::from_utf8` path (measured ~134 GiB/s vs ~61
        /// GiB/s on a 1 MiB text fixture). On inputs whose very first bytes
        /// are invalid UTF-8 (e.g. random binary), the scalar path is
        /// faster because it bails on the first illegal byte; this kernel
        /// must run a full 64-byte SIMD chunk before falling back to scalar
        /// for precise diagnosis. Callers that already know the input is
        /// likely binary should call [`super::scalar::validate_utf8`]
        /// directly.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn validate_utf8(bytes: &[u8]) -> crate::byteclass::Utf8Validation {
            // SAFETY: `target_feature(enable = "avx2")` propagates the AVX2
            // requirement to the inner module-level entry point.
            unsafe { crate::byteclass::utf8_avx2::validate_utf8(bytes) }
        }
    }

    /// AVX-512 VBMI UTF-8 validator. Same DFA as the AVX-512BW path but
    /// fuses the two `prev1`-indexed lookups into a single 256-entry
    /// `vpermi2b` byte-permute lookup. The byte-class `classify` kernel
    /// has no VBMI variant — its hot path is already a `_mm512_movepi8_mask`
    /// + `_mm512_cmpeq_epi8_mask` cascade that VBMI does not improve.
    ///
    /// Runtime gate: requires both AVX-512BW (for the kernel base) AND
    /// AVX-512 VBMI (for the `vpermi2b` instruction).
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx512_vbmi {
        /// Returns true when AVX-512BW + AVX-512 VBMI are both available
        /// at runtime.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vbmi")
        }

        /// Returns true when AVX-512BW + AVX-512 VBMI are both available.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// Validates UTF-8 with the AVX-512 VBMI fused-table DFA.
        ///
        /// Returns the same triple as
        /// [`super::scalar::validate_utf8`] / `core::str::from_utf8`.
        ///
        /// # Safety
        ///
        /// The caller must ensure both AVX-512BW and AVX-512 VBMI are
        /// available on the current CPU.
        #[target_feature(enable = "avx512bw,avx512vbmi")]
        #[must_use]
        pub unsafe fn validate_utf8(bytes: &[u8]) -> crate::byteclass::Utf8Validation {
            // SAFETY: target_feature(enable = "avx512bw,avx512vbmi") on this
            // function propagates both requirements to the inner module-level
            // entry point.
            unsafe { crate::byteclass::utf8_avx512::validate_utf8_vbmi(bytes) }
        }
    }

    /// AVX-512BW byte-class classifier.
    ///
    /// Mirrors the AVX2 path but processes 64-byte chunks via `__m512i`
    /// and uses native `__mmask64`-returning compare intrinsics
    /// (`_mm512_cmpeq_epi8_mask`, `_mm512_cmplt_epi8_mask`) instead of
    /// `movemask`. Per-class counts come straight from `mask.count_ones()`.
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx512 {
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
                let chunk =
                    unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast::<__m512i>()) };

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
    }

    /// AArch64 NEON byte-class classifier.
    ///
    /// Mirrors the AVX2 byte-class path. NEON has no movemask, so per-class
    /// counts are derived by ANDing each comparison mask with `0x01` and
    /// horizontally summing the resulting one-byte indicators across two
    /// `uint8x16_t` halves of the 32-byte window.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    pub mod neon {
        use super::scalar;
        use crate::byteclass::ByteClassCounts;

        use core::arch::aarch64::{
            uint8x16_t, vaddlvq_u8, vandq_u8, vceqq_u8, vcgeq_u8, vcltq_u8, vdupq_n_u8, vld1q_u8,
            vmvnq_u8, vorrq_u8,
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
    }
}

/// Counts coarse byte classes using the public runtime-dispatched path.
#[must_use]
pub fn classify(bytes: &[u8]) -> ByteClassCounts {
    kernels::auto::classify(bytes)
}

/// Validates UTF-8 using the public runtime-dispatched path.
#[must_use]
pub fn validate_utf8(bytes: &[u8]) -> Utf8Validation {
    kernels::auto::validate_utf8(bytes)
}

/// Returns true when the slice is strongly ASCII/text dominated.
#[must_use]
pub fn is_ascii_dominant(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let counts = classify(bytes);
    (counts.printable_ascii + counts.whitespace) * 10 >= bytes.len() as u64 * 9
}

#[cfg(test)]
mod tests {
    use super::{classify, is_ascii_dominant, kernels, validate_utf8};

    #[test]
    fn classifies_ascii_text() {
        let counts = classify(b"abc 123\n");
        assert_eq!(counts.whitespace, 2);
        assert_eq!(counts.printable_ascii, 6);
        assert!(is_ascii_dominant(b"abc 123\n"));
    }

    #[test]
    fn public_default_matches_scalar_on_edge_cases() {
        for bytes in byteclass_cases() {
            assert_eq!(classify(&bytes), kernels::scalar::classify(&bytes));
            assert_eq!(
                validate_utf8(&bytes),
                kernels::scalar::validate_utf8(&bytes)
            );
        }
    }

    #[test]
    fn validates_utf8_with_error_offsets() {
        let valid = validate_utf8("hello \u{2603}".as_bytes());
        assert!(valid.valid);
        assert_eq!(valid.valid_up_to, "hello \u{2603}".len());

        let invalid = validate_utf8(b"abc\xffdef");
        assert!(!invalid.valid);
        assert_eq!(invalid.valid_up_to, 3);
        assert_eq!(invalid.error_len, 1);

        let incomplete = validate_utf8(b"abc\xe2\x98");
        assert!(incomplete.is_incomplete());
        assert_eq!(incomplete.valid_up_to, 3);
    }

    #[cfg(all(
        feature = "std",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !kernels::avx2::is_available() {
            return;
        }

        let base = byteclass_cases()
            .into_iter()
            .flatten()
            .cycle()
            .take(16 * 1024 + 63)
            .collect::<Vec<_>>();
        for start in 0..64 {
            for len in [0_usize, 1, 2, 7, 31, 32, 33, 255, 256, 4096, 8191] {
                let end = (start + len).min(base.len());
                let bytes = &base[start..end];
                // SAFETY: availability was checked above.
                let actual = unsafe { kernels::avx2::classify(bytes) };
                assert_eq!(
                    actual,
                    kernels::scalar::classify(bytes),
                    "AVX2 mismatch at start {start}, len {}",
                    bytes.len()
                );
            }
        }
    }

    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx512_classify_matches_scalar_when_available() {
        if !kernels::avx512::is_available() {
            return;
        }

        let base = byteclass_cases()
            .into_iter()
            .flatten()
            .cycle()
            .take(16 * 1024 + 127)
            .collect::<Vec<_>>();
        for start in 0..64 {
            for len in [
                0_usize, 1, 2, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 4096, 8191,
            ] {
                let end = (start + len).min(base.len());
                let bytes = &base[start..end];
                // SAFETY: availability was checked above.
                let actual = unsafe { kernels::avx512::classify(bytes) };
                assert_eq!(
                    actual,
                    kernels::scalar::classify(bytes),
                    "AVX-512 classify mismatch at start {start}, len {}",
                    bytes.len()
                );
            }
        }
    }

    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx512_validate_utf8_matches_scalar_when_available() {
        if !kernels::avx512::is_available() {
            return;
        }

        for bytes in byteclass_cases() {
            // SAFETY: availability was checked above.
            let actual = unsafe { kernels::avx512::validate_utf8(&bytes) };
            assert_eq!(
                actual,
                kernels::scalar::validate_utf8(&bytes),
                "AVX-512 validate_utf8 mismatch on {}-byte case",
                bytes.len(),
            );
        }

        // A long valid UTF-8 stream that exercises the DFA across many
        // 64-byte blocks (multi-byte snowman + ASCII + 4-byte emoji).
        let mut text = String::new();
        for _ in 0..512 {
            text.push_str("hello \u{2603} world \u{1F600} ");
        }
        // SAFETY: availability checked above.
        let actual = unsafe { kernels::avx512::validate_utf8(text.as_bytes()) };
        assert_eq!(actual, kernels::scalar::validate_utf8(text.as_bytes()));
    }

    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx512_vbmi_validate_utf8_matches_scalar_when_available() {
        if !kernels::avx512_vbmi::is_available() {
            return;
        }

        for bytes in byteclass_cases() {
            // SAFETY: availability was checked above.
            let actual = unsafe { kernels::avx512_vbmi::validate_utf8(&bytes) };
            assert_eq!(
                actual,
                kernels::scalar::validate_utf8(&bytes),
                "AVX-512 VBMI validate_utf8 mismatch on {}-byte case",
                bytes.len(),
            );
        }

        // Long valid stream across many 64-byte blocks.
        let mut text = String::new();
        for _ in 0..512 {
            text.push_str("hello \u{2603} world \u{1F600} ");
        }
        // SAFETY: availability checked above.
        let actual = unsafe { kernels::avx512_vbmi::validate_utf8(text.as_bytes()) };
        assert_eq!(actual, kernels::scalar::validate_utf8(text.as_bytes()));
    }

    /// Direct parity check between the AVX-512 VBMI variant and the
    /// AVX-512BW variant on a fixed mixed-content payload. Required by
    /// the task spec: "parity with scalar reference (or with the
    /// existing AVX-512BW path for utf8) on a fixed mixed payload".
    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx512_vbmi_matches_avx512bw_on_mixed_payload() {
        if !kernels::avx512_vbmi::is_available() || !kernels::avx512::is_available() {
            return;
        }

        // Build a 1 MiB mixed payload covering ASCII, multi-byte UTF-8,
        // and binary tails. Cycle through a heterogeneous filler to
        // exercise the DFA's continuation-byte and high-leader paths
        // across many 64-byte boundaries.
        let mut payload = Vec::with_capacity(1 << 20);
        let filler: &[u8] = b"hello world ";
        let snowman = "\u{2603}".as_bytes();
        let emoji = "\u{1F600}".as_bytes();
        while payload.len() < (1 << 20) {
            payload.extend_from_slice(filler);
            payload.extend_from_slice(snowman);
            payload.extend_from_slice(filler);
            payload.extend_from_slice(emoji);
        }
        payload.truncate(1 << 20);

        // SAFETY: availability checked above.
        let bw = unsafe { kernels::avx512::validate_utf8(&payload) };
        // SAFETY: availability checked above.
        let vbmi = unsafe { kernels::avx512_vbmi::validate_utf8(&payload) };
        assert_eq!(
            bw, vbmi,
            "VBMI vs AVX-512BW mismatch on 1 MiB mixed payload"
        );

        // Also verify on a payload with an injected error mid-stream.
        let mut bad = payload.clone();
        bad[768] = 0xFF;
        // SAFETY: availability checked above.
        let bw_bad = unsafe { kernels::avx512::validate_utf8(&bad) };
        // SAFETY: availability checked above.
        let vbmi_bad = unsafe { kernels::avx512_vbmi::validate_utf8(&bad) };
        assert_eq!(
            bw_bad, vbmi_bad,
            "VBMI vs AVX-512BW mismatch on bad-byte payload"
        );
        assert!(!bw_bad.valid);
    }

    fn byteclass_cases() -> Vec<Vec<u8>> {
        vec![
            Vec::new(),
            vec![0],
            vec![0; 4096],
            b"abc 123\n\t\r".to_vec(),
            (0_u8..=255).collect(),
            (0_u8..=255).cycle().take(4097).collect(),
            (0_usize..8192)
                .map(|i| (i.wrapping_mul(37) ^ (i >> 3).wrapping_mul(11)) as u8)
                .collect(),
            b"\x00\x01\x02\t\n\r hello world \x7f\x80\xff"
                .repeat(257)
                .to_vec(),
        ]
    }
}

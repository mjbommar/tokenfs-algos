//! Byte classification primitives.
//!
//! The public [`classify`] path uses runtime dispatch when a tested optimized
//! backend is available. Pinned kernels live under [`kernels`] for
//! reproducible benchmarks and forensic comparisons.

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
mod utf8_avx2;

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
mod utf8_avx512;

// `utf8_neon` is reused by both the NEON path and the SVE2 path (the
// SVE2 fast-path defers to the NEON DFA on non-ASCII input), so the
// module declaration follows the union of the two cargo features.
#[cfg(all(any(feature = "neon", feature = "sve2"), target_arch = "aarch64"))]
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

            #[cfg(all(feature = "std", feature = "sve2", target_arch = "aarch64"))]
            {
                if super::sve2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::sve2::classify(bytes) };
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

            #[cfg(all(feature = "std", feature = "sve2", target_arch = "aarch64"))]
            {
                if super::sve2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::sve2::validate_utf8(bytes) };
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
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    /// AVX2 byte-class classifier.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx2;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx2;

    /// AVX-512BW byte-class classifier.
    ///
    /// Mirrors the AVX2 path but processes 64-byte chunks via `__m512i`
    /// and uses native `__mmask64`-returning compare intrinsics
    /// (`_mm512_cmpeq_epi8_mask`, `_mm512_cmplt_epi8_mask`) instead of
    /// `movemask`. Per-class counts come straight from `mask.count_ones()`.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx512;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx512;

    /// Permutation-LUT byte classifier built on AVX-512 VBMI.
    ///
    /// Replaces the per-class `cmpeq + popcount` chain in
    /// `super::avx512::classify` with a single 256-entry table lookup
    /// per byte, then a small per-class popcount loop. The lookup is
    /// driven by `_mm512_permutex2var_epi8` (`vpermi2b`), which selects
    /// 64 bytes from the 128-byte concatenation of two source registers.
    /// Two such lookups blended by the high bit of each input byte cover
    /// the full 256-byte table.
    ///
    /// # Hardware requirements
    ///
    /// AVX-512 VBMI ships on Intel Ice Lake (2019) and newer client/server
    /// parts (Tiger Lake, Sapphire Rapids, Granite Rapids). It is **not**
    /// part of the AVX-512BW base. Notable absences:
    ///
    /// * AMD Zen 4 (e.g. EPYC 9004 / Ryzen 7000) implements AVX-512F/BW/VL
    ///   and even VBMI2 + BITALG, but does **not** expose the original
    ///   `vpermi2b` (`AVX512VBMI`) bit. Code paths that gate on `avx512vbmi`
    ///   will fall back on those CPUs.
    /// * Intel Alder Lake / Raptor Lake disabled AVX-512 in production
    ///   microcode; this kernel never runs there.
    ///
    /// At time of writing only Intel Ice Lake-and-later runners can
    /// execute this path; everywhere else the dispatch falls back to
    /// `super::avx512` or below.
    ///
    /// # Why use it
    ///
    /// This kernel is a **generalization** of
    /// `super::avx512::classify`: instead of hard-coding 4 named byte
    /// classes (printable / control / whitespace / high-bit), it accepts
    /// any `[u8; 256]` table mapping byte values to class indices in
    /// `0..16`. That makes it suitable for ad-hoc classifiers (URL-safe
    /// bytes, hex digits, base64 alphabets, JSON structural characters,
    /// etc.) without writing bespoke kernels for each.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx512_vbmi;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx512_vbmi;

    /// AArch64 NEON byte-class classifier.
    ///
    /// Mirrors the AVX2 byte-class path. NEON has no movemask, so per-class
    /// counts are derived by ANDing each comparison mask with `0x01` and
    /// horizontally summing the resulting one-byte indicators across two
    /// `uint8x16_t` halves of the 32-byte window.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "neon",
        target_arch = "aarch64"
    ))]
    pub mod neon;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "neon",
        target_arch = "aarch64"
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod neon;

    /// AArch64 SVE2 byte-class classifier.
    ///
    /// # Vector-length-agnostic (VLA) programming model
    ///
    /// Unlike NEON's fixed 128-bit `uint8x16_t`, SVE/SVE2 vectors have a
    /// runtime-determined width: anything from 128 bits up to 2048 bits in
    /// 128-bit increments. The CPU exposes its physical width via
    /// [`core::arch::aarch64::svcntb`] (bytes per vector). One source
    /// compiles to optimal code for every supported width.
    ///
    /// The classifier loop is shaped as:
    ///
    /// ```text
    /// loop {
    ///     pg = svwhilelt_b8(i, n);          // active-lane predicate
    ///     if !svptest_any(svptrue_b8(), pg) { break; }
    ///     v  = svld1_u8(pg, ptr.add(i));    // predicated load (zeros inactive lanes)
    ///     // classify, popcount each class via `svcntp_b8`
    ///     i += svcntb() as usize;
    ///  }
    /// ```
    ///
    /// `svwhilelt_b8(i, n)` is true for lane `j` when `i + j < n`, so the
    /// final iteration cleanly handles the tail with no scalar fall-back —
    /// the inactive lanes contribute zero to every class. This eliminates
    /// the explicit `bytes[index..]` scalar epilogue that the AVX2 / NEON
    /// paths need.
    ///
    /// # Per-runner vector width
    ///
    /// The kernels in this module work on every SVE-capable CPU regardless
    /// of width. For reference, common widths in 2024-2026 silicon:
    ///
    /// * Neoverse-N2 / Cobalt-100 (Linux GitHub-hosted aarch64 runner): 128
    ///   bits (`svcntb() == 16`).
    /// * Neoverse-V1 (Graviton 3): 256 bits (`svcntb() == 32`).
    /// * Neoverse-V2 (Graviton 4): 128 bits (it implements SVE2 at 128b
    ///   width to keep peak throughput per cycle aligned with NEON).
    /// * A64FX (Fujitsu): 512 bits (`svcntb() == 64`).
    /// * Apple M4 Pro: 128 bits (introduced in 2024).
    ///
    /// QEMU's default user-mode CPU (`max`) emulates 512-bit SVE, so
    /// cross-tests run wider lanes than the Cobalt-100 hardware will.
    /// Both must produce identical results — that is the parity test.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "sve2",
        target_arch = "aarch64"
    ))]
    pub mod sve2;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "sve2",
        target_arch = "aarch64"
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod sve2;
}

/// Class index used by [`printable_control_whitespace_high_bit_table`]
/// for printable ASCII (`0x20..=0x7e`, excluding the space character).
pub const CLASS_PRINTABLE: u8 = 0;
/// Class index used by [`printable_control_whitespace_high_bit_table`]
/// for ASCII whitespace (`\t`, `\n`, `\r`, ` `).
pub const CLASS_WHITESPACE: u8 = 1;
/// Class index used by [`printable_control_whitespace_high_bit_table`]
/// for ASCII control bytes excluding whitespace
/// (`0x00..=0x1f` minus whitespace, plus `0x7f`).
pub const CLASS_CONTROL: u8 = 2;
/// Class index used by [`printable_control_whitespace_high_bit_table`]
/// for high-bit bytes (`0x80..=0xff`).
pub const CLASS_HIGH_BIT: u8 = 3;

/// Builds a 256-byte class-index table from a closure mapping each byte
/// value to a class index.
///
/// Class indices must be in `0..16` (see
/// [`kernels::avx512_vbmi::MAX_CLASSES`]) for the LUT-based classifier
/// to produce well-defined per-class counts. The scalar reference path
/// in [`classify_with_table`] handles arbitrary `u8` indices safely; it
/// just rejects entries `>= 16`.
#[must_use]
pub fn class_table_from_fn<F: Fn(u8) -> u8>(f: F) -> [u8; 256] {
    let mut out = [0_u8; 256];
    let mut b: u32 = 0;
    while b < 256 {
        out[b as usize] = f(b as u8);
        b += 1;
    }
    out
}

/// Returns the canonical 4-class table used by
/// [`kernels::auto::classify`]: printable / whitespace / control /
/// high-bit. Matches the indices declared by [`CLASS_PRINTABLE`],
/// [`CLASS_WHITESPACE`], [`CLASS_CONTROL`], and [`CLASS_HIGH_BIT`].
///
/// Useful as a sanity-check input for [`classify_with_table`].
#[must_use]
pub fn printable_control_whitespace_high_bit_table() -> [u8; 256] {
    class_table_from_fn(|b| match b {
        b'\t' | b'\n' | b'\r' | b' ' => CLASS_WHITESPACE,
        0x20..=0x7e => CLASS_PRINTABLE,
        0x00..=0x1f | 0x7f => CLASS_CONTROL,
        0x80..=0xff => CLASS_HIGH_BIT,
    })
}

/// Counts bytes per class against an arbitrary `[u8; 256]` class-index
/// table.
///
/// Uses [`kernels::avx512_vbmi::classify_with_lut`] when AVX-512 VBMI is
/// available at runtime; otherwise falls back to a portable scalar loop.
/// Class indices in the table must be in `0..16`; entries `>= 16` are
/// counted into nothing (silently dropped).
///
/// The returned array is indexed by class: `counts[c]` is the number of
/// input bytes that mapped to class `c`.
#[must_use]
pub fn classify_with_table(bytes: &[u8], class_table: &[u8; 256]) -> [u64; 16] {
    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if kernels::avx512_vbmi::is_available() {
            // SAFETY: availability was checked immediately above.
            return unsafe { kernels::avx512_vbmi::classify_with_lut(bytes, class_table) };
        }
    }

    classify_with_table_scalar(bytes, class_table)
}

/// Portable scalar reference for [`classify_with_table`].
fn classify_with_table_scalar(bytes: &[u8], class_table: &[u8; 256]) -> [u64; 16] {
    let mut counts = [0_u64; 16];
    for &byte in bytes {
        let c = class_table[byte as usize] as usize;
        if c < counts.len() {
            counts[c] += 1;
        }
    }
    counts
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
    use super::{
        CLASS_CONTROL, CLASS_HIGH_BIT, CLASS_PRINTABLE, CLASS_WHITESPACE, class_table_from_fn,
        classify, classify_with_table, classify_with_table_scalar, is_ascii_dominant, kernels,
        printable_control_whitespace_high_bit_table, validate_utf8,
    };
    // `Vec` and `vec!` are not in the no-std prelude; alias them from
    // `alloc` for the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

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

    #[test]
    fn class_table_from_fn_round_trip() {
        let table = printable_control_whitespace_high_bit_table();
        for b in 0_u32..256 {
            let expected = match b as u8 {
                b'\t' | b'\n' | b'\r' | b' ' => CLASS_WHITESPACE,
                0x20..=0x7e => CLASS_PRINTABLE,
                0x00..=0x1f | 0x7f => CLASS_CONTROL,
                0x80..=0xff => CLASS_HIGH_BIT,
            };
            assert_eq!(table[b as usize], expected, "table mismatch at byte {b}");
        }
    }

    #[test]
    fn classify_with_table_default_matches_named_classify() {
        let table = printable_control_whitespace_high_bit_table();
        for bytes in byteclass_cases() {
            let lut_counts = classify_with_table(&bytes, &table);
            let named = classify(&bytes);
            assert_eq!(
                lut_counts[CLASS_PRINTABLE as usize],
                named.printable_ascii,
                "printable mismatch on {}-byte case",
                bytes.len()
            );
            assert_eq!(
                lut_counts[CLASS_WHITESPACE as usize],
                named.whitespace,
                "whitespace mismatch on {}-byte case",
                bytes.len()
            );
            assert_eq!(
                lut_counts[CLASS_CONTROL as usize],
                named.control,
                "control mismatch on {}-byte case",
                bytes.len()
            );
            assert_eq!(
                lut_counts[CLASS_HIGH_BIT as usize],
                named.high_bit,
                "high-bit mismatch on {}-byte case",
                bytes.len()
            );
        }
    }

    #[test]
    fn classify_with_table_alpha_digit_other() {
        // 4-class table: 0=upper, 1=lower, 2=digit, 3=other.
        let table = class_table_from_fn(|b| match b {
            b'A'..=b'Z' => 0,
            b'a'..=b'z' => 1,
            b'0'..=b'9' => 2,
            _ => 3,
        });

        let payload = b"Hello, World 12345! Foo Bar 678";
        let counts = classify_with_table(payload, &table);

        // "H", "W", "F", "B" -> 4 uppercase
        // "ello", "orld", "oo", "ar" -> 4+4+2+2 = 12 lowercase
        // "12345", "678" -> 8 digits
        // remainder = total - (upper + lower + digit)
        let upper = counts[0];
        let lower = counts[1];
        let digit = counts[2];
        let other = counts[3];

        assert_eq!(upper, 4, "uppercase count");
        assert_eq!(lower, 12, "lowercase count");
        assert_eq!(digit, 8, "digit count");
        assert_eq!(
            upper + lower + digit + other,
            payload.len() as u64,
            "total covers payload"
        );
    }

    #[test]
    fn classify_with_table_scalar_fallback_matches_self() {
        // Drives the scalar fallback path explicitly to confirm it agrees
        // with itself across edge cases (length 0, 1, 63, 64, 65, ...).
        let table = printable_control_whitespace_high_bit_table();
        for bytes in byteclass_cases() {
            assert_eq!(
                classify_with_table_scalar(&bytes, &table),
                {
                    let mut expected = [0_u64; 16];
                    for &b in &bytes {
                        let c = table[b as usize] as usize;
                        if c < expected.len() {
                            expected[c] += 1;
                        }
                    }
                    expected
                },
                "scalar fallback self-consistency on {}-byte case",
                bytes.len()
            );
        }
    }

    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx512_vbmi_classify_with_lut_matches_scalar_when_available() {
        if !kernels::avx512_vbmi::is_available() {
            return;
        }

        let table = printable_control_whitespace_high_bit_table();
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
                let actual = unsafe { kernels::avx512_vbmi::classify_with_lut(bytes, &table) };
                let expected = classify_with_table_scalar(bytes, &table);
                assert_eq!(
                    actual,
                    expected,
                    "VBMI LUT mismatch at start {start}, len {}",
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
    fn avx512_vbmi_classify_named_matches_scalar_when_available() {
        if !kernels::avx512_vbmi::is_available() {
            return;
        }

        for bytes in byteclass_cases() {
            // SAFETY: availability was checked above.
            let actual = unsafe { kernels::avx512_vbmi::classify(&bytes) };
            let expected = kernels::scalar::classify(&bytes);
            assert_eq!(
                actual.printable_ascii,
                expected.printable_ascii,
                "VBMI named printable mismatch ({}-byte)",
                bytes.len()
            );
            assert_eq!(
                actual.whitespace,
                expected.whitespace,
                "VBMI named whitespace mismatch ({}-byte)",
                bytes.len()
            );
            assert_eq!(
                actual.control,
                expected.control,
                "VBMI named control mismatch ({}-byte)",
                bytes.len()
            );
            assert_eq!(
                actual.high_bit,
                expected.high_bit,
                "VBMI named high-bit mismatch ({}-byte)",
                bytes.len()
            );
        }
    }

    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx512_vbmi_lut_alpha_digit_table_when_available() {
        if !kernels::avx512_vbmi::is_available() {
            return;
        }

        let table = class_table_from_fn(|b| match b {
            b'A'..=b'Z' => 0,
            b'a'..=b'z' => 1,
            b'0'..=b'9' => 2,
            _ => 3,
        });
        // Build a chunk longer than one 64-byte stride so the SIMD loop
        // body runs at least once and the scalar tail is exercised.
        let mut text = Vec::with_capacity(200);
        text.extend_from_slice(b"Hello, World 12345! Foo Bar 678 ");
        text.extend_from_slice(b"qwertyuiopASDFGHJKL12345!@#$%^&*()_+");
        text.extend_from_slice(b"ZZZZZZZZZ aaaaa  9090909090 ABCDEFG ");
        // SAFETY: availability checked above.
        let actual = unsafe { kernels::avx512_vbmi::classify_with_lut(&text, &table) };
        let expected = classify_with_table_scalar(&text, &table);
        assert_eq!(actual, expected);
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

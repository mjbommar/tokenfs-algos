//! Pack and unpack `u32` values at arbitrary widths `W ∈ 1..=32`.
//!
//! See `docs/v0.2_planning/10_BITS.md` § 2 for the spec.
//!
//! ## Bit layout
//!
//! Bits within a byte are little-endian (lowest bit first). `values[i]`
//! occupies bits `[i*W, (i+1)*W)` of the packed stream; bytes are written
//! in order. Encoded length is `ceil(n*W/8)` bytes.
//!
//! ## API surface
//!
//! * [`BitPacker<W>`] — const-generic specialization for compile-time
//!   widths. Use when the width is fixed at the call site (token
//!   decoders for a known vocabulary).
//! * [`DynamicBitPacker`] — runtime-width form. Dispatches on a single
//!   `match` over the width.
//! * [`kernels::scalar`] — portable shift-and-OR reference oracle.
//! * `kernels::avx2` — VPSRLVQ / VPSRLVD per-lane shift + mask.
//! * `kernels::neon` — TBL byte permute + per-lane shifts.
//!
//! ## Hot path
//!
//! Decode is the hot kernel. The implementation has two regimes:
//!
//! * **Byte-aligned widths** (`W ∈ {8, 16, 32}`): degenerate to a
//!   `from_le_bytes` cast. `W ∈ {1, 2, 4}` use a single-byte expansion
//!   loop. These hard-coded fast paths are used by both the scalar and
//!   SIMD entry points.
//! * **Other widths**: load up to 8 input bytes into a `u64`, shift
//!   right by the in-byte offset, mask to `W` bits.
//!
//! The const-generic [`BitPacker<W>`] lets the compiler specialize at
//! the call site; the dynamic form dispatches on width via `match`.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_range_loop)]

use core::marker::PhantomData;

/// Failure modes for the fallible bit-pack APIs ([`BitPacker::try_encode_u32_slice`],
/// [`BitPacker::try_decode_u32_slice`], [`DynamicBitPacker::try_encode_u32_slice`],
/// and [`DynamicBitPacker::try_decode_u32_slice`]).
///
/// Returned instead of panicking when a caller-supplied buffer is too
/// small or the configured width is out of the supported range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BitPackError {
    /// Decode input was shorter than `ceil(n * width / 8)` bytes.
    InputTooShort {
        /// Number of input bytes the operation needed.
        needed: usize,
        /// Length of the caller-supplied input buffer.
        actual: usize,
    },
    /// Encode output (or decode output slot count) was too short for
    /// the operation.
    OutputTooShort {
        /// Number of output bytes / slots the operation needed.
        needed: usize,
        /// Length of the caller-supplied output buffer.
        actual: usize,
    },
    /// Width was outside the accepted `1..=32` range.
    WidthOutOfRange {
        /// Caller-supplied width.
        width: u32,
    },
}

impl core::fmt::Display for BitPackError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InputTooShort { needed, actual } => write!(
                f,
                "bit-pack input buffer too small: needed {needed}, got {actual}"
            ),
            Self::OutputTooShort { needed, actual } => write!(
                f,
                "bit-pack output buffer too small: needed {needed}, got {actual}"
            ),
            Self::WidthOutOfRange { width } => write!(
                f,
                "bit-pack width {width} outside the accepted range 1..=32"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BitPackError {}

/// Returns `ceil(n * w / 8)`, the number of bytes needed to pack `n`
/// values of `w` bits each.
///
/// `w` must satisfy `1 <= w <= 32`; higher widths overflow `usize` only
/// for absurd `n` and are already rejected by [`BitPacker`] /
/// [`DynamicBitPacker`].
#[inline]
const fn encoded_len_bytes(n: usize, w: u32) -> usize {
    let bits = n.saturating_mul(w as usize);
    bits.div_ceil(8)
}

/// Const-generic bit packer for compile-time-known widths.
///
/// Width `W` must satisfy `1 <= W <= 32`; out-of-range widths panic at
/// the encode/decode entry points.
#[derive(Copy, Clone, Debug, Default)]
pub struct BitPacker<const W: u32>(PhantomData<()>);

impl<const W: u32> BitPacker<W> {
    /// Constructs a new packer. The const generic carries the width.
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// Returns the encoded length in bytes for `n` values.
    #[must_use]
    pub const fn encoded_len(n: usize) -> usize {
        encoded_len_bytes(n, W)
    }

    /// Packs `values.len()` integers of `W` bits each into `out`.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < Self::encoded_len(values.len())` or if
    /// `W` is out of range `1..=32`. Values whose high bits exceed `W`
    /// are masked silently — callers that need range-checking should
    /// validate upstream. Use [`Self::try_encode_u32_slice`] for a
    /// fallible variant that returns [`BitPackError`] instead.
    ///
    /// Only compiled when the `panicking-shape-apis` Cargo feature is
    /// enabled (default). Kernel/FUSE consumers should disable that
    /// feature and use [`Self::try_encode_u32_slice`] (audit-R5 #157).
    #[cfg(feature = "panicking-shape-apis")]
    pub fn encode_u32_slice(values: &[u32], out: &mut [u8]) {
        assert!(W >= 1 && W <= 32, "BitPacker width must be 1..=32");
        kernels::auto::encode_u32_slice(W, values, out);
    }

    /// Fallible variant of [`Self::encode_u32_slice`] that returns
    /// [`BitPackError`] when the output buffer is too small or the
    /// const-generic width `W` is out of range, instead of panicking.
    ///
    /// Validates `W` and `out.len()` upfront and dispatches to the
    /// `_unchecked` kernel that omits the kernel-internal `assert!`
    /// guards, so the call is panic-free even when the
    /// `panicking-shape-apis` feature is disabled (audit-R6 finding
    /// #162).
    pub fn try_encode_u32_slice(values: &[u32], out: &mut [u8]) -> Result<(), BitPackError> {
        if !(1..=32).contains(&W) {
            return Err(BitPackError::WidthOutOfRange { width: W });
        }
        let needed = Self::encoded_len(values.len());
        if out.len() < needed {
            return Err(BitPackError::OutputTooShort {
                needed,
                actual: out.len(),
            });
        }
        // SAFETY: pre-validation above ensures `1 <= W <= 32` and
        // `out.len() >= ceil(values.len() * W / 8)`.
        unsafe { kernels::auto::encode_u32_slice_unchecked(W, values, out) };
        Ok(())
    }

    /// Unpacks `n` integers of `W` bits each from `input` into `out`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() < Self::encoded_len(n)`, `out.len() < n`,
    /// or `W` is out of range `1..=32`. Use [`Self::try_decode_u32_slice`]
    /// for a fallible variant that returns [`BitPackError`] instead.
    ///
    /// Only compiled when the `panicking-shape-apis` Cargo feature is
    /// enabled (default). Kernel/FUSE consumers should disable that
    /// feature and use [`Self::try_decode_u32_slice`] (audit-R5 #157).
    #[cfg(feature = "panicking-shape-apis")]
    pub fn decode_u32_slice(input: &[u8], n: usize, out: &mut [u32]) {
        assert!(W >= 1 && W <= 32, "BitPacker width must be 1..=32");
        kernels::auto::decode_u32_slice(W, input, n, out);
    }

    /// Fallible variant of [`Self::decode_u32_slice`] that returns
    /// [`BitPackError`] when the input buffer is too short, the output
    /// buffer is too small, or the const-generic width `W` is out of
    /// range, instead of panicking.
    ///
    /// Validates `W`, `input.len()`, and `out.len()` upfront and
    /// dispatches to the `_unchecked` kernel that omits the
    /// kernel-internal `assert!` guards, so the call is panic-free even
    /// when the `panicking-shape-apis` feature is disabled (audit-R6
    /// finding #162).
    pub fn try_decode_u32_slice(
        input: &[u8],
        n: usize,
        out: &mut [u32],
    ) -> Result<(), BitPackError> {
        if !(1..=32).contains(&W) {
            return Err(BitPackError::WidthOutOfRange { width: W });
        }
        let needed = Self::encoded_len(n);
        if input.len() < needed {
            return Err(BitPackError::InputTooShort {
                needed,
                actual: input.len(),
            });
        }
        if out.len() < n {
            return Err(BitPackError::OutputTooShort {
                needed: n,
                actual: out.len(),
            });
        }
        // SAFETY: pre-validation above ensures `1 <= W <= 32`,
        // `input.len() >= ceil(n * W / 8)`, and `out.len() >= n`.
        unsafe { kernels::auto::decode_u32_slice_unchecked(W, input, n, out) };
        Ok(())
    }
}

/// Runtime-width bit packer.
///
/// Use when the width is configured per-image rather than known at
/// compile time. Internally dispatches on the width via a single match
/// in the kernel.
#[derive(Copy, Clone, Debug)]
pub struct DynamicBitPacker {
    width: u32,
}

impl DynamicBitPacker {
    /// Returns a packer for the given width.
    ///
    /// # Panics
    ///
    /// Panics if `width` is outside `1..=32`. Use [`Self::try_new`] for
    /// a panic-free variant.
    ///
    /// Only compiled when the `panicking-shape-apis` Cargo feature is
    /// enabled (default). Kernel/FUSE consumers should disable that
    /// feature and use [`Self::try_new`] (audit-R5 #157).
    #[cfg(feature = "panicking-shape-apis")]
    #[must_use]
    pub const fn new(width: u32) -> Self {
        assert!(
            width >= 1 && width <= 32,
            "DynamicBitPacker width must be 1..=32"
        );
        Self { width }
    }

    /// Returns a packer for the given width, or `None` if `width` is
    /// outside `1..=32`.
    #[must_use]
    pub const fn try_new(width: u32) -> Option<Self> {
        if width >= 1 && width <= 32 {
            Some(Self { width })
        } else {
            None
        }
    }

    /// Returns the configured width in bits (1..=32).
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Returns the encoded length in bytes for `n` values.
    #[must_use]
    pub const fn encoded_len(&self, n: usize) -> usize {
        encoded_len_bytes(n, self.width)
    }

    /// Packs `values.len()` integers of `self.width()` bits each into
    /// `out`.
    ///
    /// # Panics
    ///
    /// Panics if `out.len() < self.encoded_len(values.len())`. Values
    /// whose high bits exceed `self.width()` are masked silently. Use
    /// [`Self::try_encode_u32_slice`] for a fallible variant that
    /// returns [`BitPackError`] instead.
    ///
    /// Only compiled when the `panicking-shape-apis` Cargo feature is
    /// enabled (default). Kernel/FUSE consumers should disable that
    /// feature and use [`Self::try_encode_u32_slice`] (audit-R5 #157).
    #[cfg(feature = "panicking-shape-apis")]
    pub fn encode_u32_slice(&self, values: &[u32], out: &mut [u8]) {
        kernels::auto::encode_u32_slice(self.width, values, out);
    }

    /// Fallible variant of [`Self::encode_u32_slice`] that returns
    /// [`BitPackError`] when the output buffer is too small or the
    /// configured width is out of range, instead of panicking.
    ///
    /// Validates the configured width and `out.len()` upfront and
    /// dispatches to the `_unchecked` kernel that omits the
    /// kernel-internal `assert!` guards, so the call is panic-free even
    /// when the `panicking-shape-apis` feature is disabled (audit-R6
    /// finding #162).
    pub fn try_encode_u32_slice(&self, values: &[u32], out: &mut [u8]) -> Result<(), BitPackError> {
        if !(1..=32).contains(&self.width) {
            return Err(BitPackError::WidthOutOfRange { width: self.width });
        }
        let needed = self.encoded_len(values.len());
        if out.len() < needed {
            return Err(BitPackError::OutputTooShort {
                needed,
                actual: out.len(),
            });
        }
        // SAFETY: pre-validation above ensures `1 <= width <= 32` and
        // `out.len() >= ceil(values.len() * width / 8)`.
        unsafe { kernels::auto::encode_u32_slice_unchecked(self.width, values, out) };
        Ok(())
    }

    /// Unpacks `n` integers of `self.width()` bits each from `input`
    /// into `out`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() < self.encoded_len(n)` or `out.len() < n`.
    /// Use [`Self::try_decode_u32_slice`] for a fallible variant that
    /// returns [`BitPackError`] instead.
    ///
    /// Only compiled when the `panicking-shape-apis` Cargo feature is
    /// enabled (default). Kernel/FUSE consumers should disable that
    /// feature and use [`Self::try_decode_u32_slice`] (audit-R5 #157).
    #[cfg(feature = "panicking-shape-apis")]
    pub fn decode_u32_slice(&self, input: &[u8], n: usize, out: &mut [u32]) {
        kernels::auto::decode_u32_slice(self.width, input, n, out);
    }

    /// Fallible variant of [`Self::decode_u32_slice`] that returns
    /// [`BitPackError`] when the input buffer is too short, the output
    /// buffer is too small, or the configured width is out of range,
    /// instead of panicking.
    ///
    /// Validates the configured width, `input.len()`, and `out.len()`
    /// upfront and dispatches to the `_unchecked` kernel that omits the
    /// kernel-internal `assert!` guards, so the call is panic-free even
    /// when the `panicking-shape-apis` feature is disabled (audit-R6
    /// finding #162).
    pub fn try_decode_u32_slice(
        &self,
        input: &[u8],
        n: usize,
        out: &mut [u32],
    ) -> Result<(), BitPackError> {
        if !(1..=32).contains(&self.width) {
            return Err(BitPackError::WidthOutOfRange { width: self.width });
        }
        let needed = self.encoded_len(n);
        if input.len() < needed {
            return Err(BitPackError::InputTooShort {
                needed,
                actual: input.len(),
            });
        }
        if out.len() < n {
            return Err(BitPackError::OutputTooShort {
                needed: n,
                actual: out.len(),
            });
        }
        // SAFETY: pre-validation above ensures `1 <= width <= 32`,
        // `input.len() >= ceil(n * width / 8)`, and `out.len() >= n`.
        unsafe { kernels::auto::decode_u32_slice_unchecked(self.width, input, n, out) };
        Ok(())
    }
}

/// Pinned bit-pack kernels.
pub mod kernels {
    /// Runtime-dispatched bit-pack kernels.
    pub mod auto {
        /// Packs `values` at width `w` into `out` using the best
        /// available kernel (panicking variant).
        ///
        /// # Panics
        ///
        /// Panics if `w` is outside `1..=32` or
        /// `out.len() < ceil(values.len() * w / 8)`. Available only with
        /// `feature = "userspace"` (audit-R10 #1) — kernel/FUSE callers
        /// should use the `_unchecked` sibling after validating, or go
        /// through [`super::super::BitPacker::try_encode_u32_slice`] /
        /// [`super::super::DynamicBitPacker::try_encode_u32_slice`].
        #[cfg(feature = "userspace")]
        pub fn encode_u32_slice(w: u32, values: &[u32], out: &mut [u8]) {
            // Encode is bandwidth-modest and the per-element work is
            // dominated by cross-byte stores; the scalar path already
            // runs ~2-3 GB/s and SIMD wins are small. Keep one path.
            super::scalar::encode_u32_slice(w, values, out);
        }

        /// Encode kernel without bounds-checking asserts.
        ///
        /// # Safety
        ///
        /// Caller must ensure `1 <= w <= 32` and
        /// `out.len() >= ceil(values.len() * w / 8)`. Used by the
        /// fallible bit-pack APIs after pre-validation; eliminates the
        /// kernel-internal `assert!` panic sites that would otherwise
        /// leak through the fallible API surface (audit-R6 finding
        /// #162).
        pub unsafe fn encode_u32_slice_unchecked(w: u32, values: &[u32], out: &mut [u8]) {
            // SAFETY: caller upholds the precondition.
            unsafe { super::scalar::encode_u32_slice_unchecked(w, values, out) };
        }

        /// Unpacks `n` values at width `w` from `input` into `out`
        /// using the best available kernel (panicking variant).
        ///
        /// # Panics
        ///
        /// Panics if `w` is outside `1..=32`,
        /// `input.len() < ceil(n * w / 8)`, or `out.len() < n`.
        ///
        /// Available only with `feature = "userspace"`; kernel-safe
        /// callers must use [`decode_u32_slice_unchecked`]
        /// (audit-R10 #1 / #216).
        #[cfg(feature = "userspace")]
        pub fn decode_u32_slice(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    unsafe { super::avx2::decode_u32_slice(w, input, n, out) };
                    return;
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { super::neon::decode_u32_slice(w, input, n, out) };
                    return;
                }
            }

            super::scalar::decode_u32_slice(w, input, n, out);
        }

        /// Decode kernel without bounds-checking asserts.
        ///
        /// # Safety
        ///
        /// Caller must ensure `1 <= w <= 32`,
        /// `input.len() >= ceil(n * w / 8)`, and `out.len() >= n`.
        /// Used by the fallible bit-pack APIs after pre-validation;
        /// eliminates the kernel-internal `assert!` panic sites that
        /// would otherwise leak through the fallible API surface
        /// (audit-R6 finding #162).
        pub unsafe fn decode_u32_slice_unchecked(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: AVX2 availability checked above; caller
                    // upholds the buffer-length precondition.
                    unsafe { super::avx2::decode_u32_slice_unchecked(w, input, n, out) };
                    return;
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64; caller
                    // upholds the buffer-length precondition.
                    unsafe { super::neon::decode_u32_slice_unchecked(w, input, n, out) };
                    return;
                }
            }

            // SAFETY: caller upholds the precondition.
            unsafe { super::scalar::decode_u32_slice_unchecked(w, input, n, out) };
        }
    }

    /// Portable scalar bit-pack reference.
    ///
    /// This is the oracle every SIMD backend must match bit-exactly.
    /// The hot loops are intentionally simple — the compiler can
    /// usually inline and unroll the const-generic
    /// [`super::super::BitPacker`] call site to within ~10% of the
    /// SIMD paths for `W <= 8`.
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    /// x86 AVX2 bit-pack decode kernels.
    ///
    /// Two regimes share the entry point:
    ///
    /// * `W ≤ 8` — broadcast a 64-bit input word to four 64-bit lanes
    ///   and shift each lane by its per-lane bit offset using
    ///   `_mm256_srlv_epi64`. Each block extracts four packed values.
    /// * `8 < W ≤ 24` — gather eight u32 source words and shift each
    ///   lane by its per-lane offset via `_mm256_srlv_epi32`. Each
    ///   block extracts eight packed values.
    /// * `W ∈ {25..=31}` — falls through to scalar; a per-lane source
    ///   would need a 5-byte load, losing the SIMD win.
    ///
    /// Byte-aligned widths (`W ∈ {8, 16, 32}`) and `W ∈ {1, 2, 4}`
    /// fall through to the scalar fast paths — those are already a
    /// memcpy and adding SIMD on top doesn't beat the scalar
    /// `to_le_bytes` loop.
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

    /// AArch64 NEON bit-pack decode kernels.
    ///
    /// NEON has no per-lane variable right-shift on integer vectors;
    /// the `vshlq_u32` / `vshlq_u64` instructions are *signed* left
    /// shifts whose negative-magnitude shift counts behave as logical
    /// right shifts. We exploit that for per-lane right shifts.
    ///
    /// * `W ≤ 8` — extract two 64-bit lanes per iteration.
    /// * `8 < W ≤ 24` — extract four 32-bit lanes per iteration.
    /// * `W ∈ {25..=31}` — falls through to scalar; a per-lane source
    ///   would need a 5-byte load, losing the SIMD win.
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
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

    extern crate alloc;
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn deterministic_values(n: usize, w: u32, seed: u64) -> Vec<u32> {
        let mask = if w == 32 { u32::MAX } else { (1_u32 << w) - 1 };
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let v = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32;
                v & mask
            })
            .collect()
    }

    #[test]
    fn encoded_len_matches_ceil_formula() {
        for w in 1_u32..=32 {
            for n in [0_usize, 1, 7, 8, 33, 1024] {
                let expected = (n * w as usize).div_ceil(8);
                assert_eq!(encoded_len_bytes(n, w), expected);
            }
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn dynamic_packer_round_trip_every_width_and_length() {
        for w in 1_u32..=32 {
            for &n in &[0_usize, 1, 7, 8, 33, 1024] {
                let packer = DynamicBitPacker::new(w);
                let values =
                    deterministic_values(n, w, 0xC0FFEE_u64 ^ ((w as u64) << 16) ^ n as u64);
                let mut encoded = vec![0_u8; packer.encoded_len(n)];
                packer.encode_u32_slice(&values, &mut encoded);
                let mut decoded = vec![0_u32; n];
                packer.decode_u32_slice(&encoded, n, &mut decoded);
                assert_eq!(decoded, values, "round-trip failed at w={w} n={n}");
            }
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn const_generic_packer_matches_dynamic() {
        // Spot-check the canonical token widths.
        macro_rules! check {
            ($w:expr) => {{
                const W: u32 = $w;
                for &n in &[0_usize, 1, 7, 8, 33, 1024] {
                    let values = deterministic_values(n, W, 0x1234_u64 ^ n as u64);
                    let mut encoded_const = vec![0_u8; BitPacker::<W>::encoded_len(n)];
                    BitPacker::<W>::encode_u32_slice(&values, &mut encoded_const);
                    let mut encoded_dyn = vec![0_u8; DynamicBitPacker::new(W).encoded_len(n)];
                    DynamicBitPacker::new(W).encode_u32_slice(&values, &mut encoded_dyn);
                    assert_eq!(encoded_const, encoded_dyn, "encode w={W} n={n}");

                    let mut decoded_const = vec![0_u32; n];
                    BitPacker::<W>::decode_u32_slice(&encoded_const, n, &mut decoded_const);
                    let mut decoded_dyn = vec![0_u32; n];
                    DynamicBitPacker::new(W).decode_u32_slice(&encoded_const, n, &mut decoded_dyn);
                    assert_eq!(decoded_const, decoded_dyn, "decode w={W} n={n}");
                    assert_eq!(decoded_const, values);
                }
            }};
        }
        check!(1);
        check!(8);
        check!(11);
        check!(12);
        check!(16);
        check!(32);
    }

    #[test]
    fn try_new_rejects_out_of_range() {
        assert!(DynamicBitPacker::try_new(0).is_none());
        assert!(DynamicBitPacker::try_new(33).is_none());
        for w in 1_u32..=32 {
            assert!(DynamicBitPacker::try_new(w).is_some());
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn empty_inputs_produce_empty_output() {
        for w in 1_u32..=32 {
            let packer = DynamicBitPacker::new(w);
            let mut encoded = vec![0_u8; 0];
            packer.encode_u32_slice(&[], &mut encoded);
            let mut decoded: Vec<u32> = Vec::new();
            packer.decode_u32_slice(&encoded, 0, &mut decoded);
            assert!(decoded.is_empty(), "w={w}");
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn single_value_round_trips_at_every_width() {
        for w in 1_u32..=32 {
            let packer = DynamicBitPacker::new(w);
            let mask = if w == 32 { u32::MAX } else { (1_u32 << w) - 1 };
            for v in [0_u32, 1, mask, mask >> 1] {
                let mut encoded = vec![0_u8; packer.encoded_len(1)];
                packer.encode_u32_slice(&[v], &mut encoded);
                let mut decoded = vec![0_u32; 1];
                packer.decode_u32_slice(&encoded, 1, &mut decoded);
                assert_eq!(decoded[0], v, "w={w} v={v}");
            }
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn high_bits_above_width_are_masked_silently() {
        // The contract: callers may pass arbitrary u32, but only the
        // low W bits round-trip. This documents the silent-mask
        // behavior the spec calls out.
        let packer = DynamicBitPacker::new(11);
        let raw = vec![0xffff_ffff_u32; 4];
        let expected = vec![(1_u32 << 11) - 1; 4];
        let mut encoded = vec![0_u8; packer.encoded_len(4)];
        packer.encode_u32_slice(&raw, &mut encoded);
        let mut decoded = vec![0_u32; 4];
        packer.decode_u32_slice(&encoded, 4, &mut decoded);
        assert_eq!(decoded, expected);
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_decode_matches_scalar_for_every_width() {
        if !kernels::avx2::is_available() {
            eprintln!("avx2 unavailable; skipping inline AVX2 parity test");
            return;
        }
        for w in 1_u32..=32 {
            for &n in &[0_usize, 1, 7, 8, 33, 1024] {
                let values = deterministic_values(n, w, 0xABCDu64 ^ ((w as u64) << 8) ^ n as u64);
                let needed = encoded_len_bytes(n, w);
                let mut encoded = vec![0_u8; needed];
                // SAFETY: `encoded` sized to `encoded_len_bytes(n, w)` and `w` is in 1..=32.
                unsafe { kernels::scalar::encode_u32_slice_unchecked(w, &values, &mut encoded) };

                let mut expected = vec![0_u32; n];
                // SAFETY: `expected` sized to n; `encoded` sized to required bytes.
                unsafe {
                    kernels::scalar::decode_u32_slice_unchecked(w, &encoded, n, &mut expected)
                };

                let mut actual = vec![0_u32; n];
                // SAFETY: avx2_available() returned true above; buffers sized as above.
                unsafe { kernels::avx2::decode_u32_slice_unchecked(w, &encoded, n, &mut actual) };
                assert_eq!(actual, expected, "avx2 decode diverged at w={w} n={n}");
            }
        }
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_decode_matches_scalar_for_every_width() {
        for w in 1_u32..=32 {
            for &n in &[0_usize, 1, 7, 8, 33, 1024] {
                let values = deterministic_values(n, w, 0xABCDu64 ^ ((w as u64) << 8) ^ n as u64);
                let needed = encoded_len_bytes(n, w);
                let mut encoded = vec![0_u8; needed];
                kernels::scalar::encode_u32_slice(w, &values, &mut encoded);

                let mut expected = vec![0_u32; n];
                kernels::scalar::decode_u32_slice(w, &encoded, n, &mut expected);

                let mut actual = vec![0_u32; n];
                // SAFETY: NEON is mandatory on AArch64.
                unsafe { kernels::neon::decode_u32_slice(w, &encoded, n, &mut actual) };
                assert_eq!(actual, expected, "neon decode diverged at w={w} n={n}");
            }
        }
    }

    #[test]
    fn try_encode_dynamic_returns_err_on_undersized_output() {
        let packer = DynamicBitPacker::try_new(11).expect("width valid");
        let values = vec![1_u32; 8];
        let needed = packer.encoded_len(8);
        let mut out = vec![0_u8; needed - 1];
        let err = packer.try_encode_u32_slice(&values, &mut out).unwrap_err();
        assert_eq!(
            err,
            BitPackError::OutputTooShort {
                needed,
                actual: needed - 1
            }
        );
    }

    #[test]
    fn try_decode_dynamic_returns_err_on_undersized_input() {
        let packer = DynamicBitPacker::try_new(11).expect("width valid");
        let needed = packer.encoded_len(8);
        let input = vec![0_u8; needed - 1];
        let mut out = vec![0_u32; 8];
        let err = packer
            .try_decode_u32_slice(&input, 8, &mut out)
            .unwrap_err();
        assert_eq!(
            err,
            BitPackError::InputTooShort {
                needed,
                actual: needed - 1
            }
        );
    }

    #[test]
    fn try_decode_dynamic_returns_err_on_undersized_output() {
        let packer = DynamicBitPacker::try_new(11).expect("width valid");
        let needed = packer.encoded_len(8);
        let input = vec![0_u8; needed];
        let mut out = vec![0_u32; 1];
        let err = packer
            .try_decode_u32_slice(&input, 8, &mut out)
            .unwrap_err();
        assert_eq!(
            err,
            BitPackError::OutputTooShort {
                needed: 8,
                actual: 1
            }
        );
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_dynamic_round_trip_matches_panicking_version() {
        for w in 1_u32..=32 {
            let packer = DynamicBitPacker::new(w);
            let values = deterministic_values(33, w, 0xCAFEBABE_u64 ^ ((w as u64) << 8));
            let mut try_encoded = vec![0_u8; packer.encoded_len(33)];
            packer
                .try_encode_u32_slice(&values, &mut try_encoded)
                .unwrap();
            let mut panic_encoded = vec![0_u8; packer.encoded_len(33)];
            packer.encode_u32_slice(&values, &mut panic_encoded);
            assert_eq!(try_encoded, panic_encoded, "encode w={w}");

            let mut try_decoded = vec![0_u32; 33];
            packer
                .try_decode_u32_slice(&try_encoded, 33, &mut try_decoded)
                .unwrap();
            assert_eq!(try_decoded, values, "decode w={w}");
        }
    }

    #[test]
    fn try_const_packer_returns_err_on_undersized_output() {
        const W: u32 = 11;
        let needed = BitPacker::<W>::encoded_len(8);
        let values = vec![1_u32; 8];
        let mut out = vec![0_u8; needed - 1];
        let err = BitPacker::<W>::try_encode_u32_slice(&values, &mut out).unwrap_err();
        assert_eq!(
            err,
            BitPackError::OutputTooShort {
                needed,
                actual: needed - 1
            }
        );
    }

    #[test]
    fn try_const_packer_returns_err_on_width_out_of_range() {
        const W: u32 = 0;
        let mut out = vec![0_u8; 4];
        let err = BitPacker::<W>::try_encode_u32_slice(&[], &mut out).unwrap_err();
        assert_eq!(err, BitPackError::WidthOutOfRange { width: 0 });
        let err = BitPacker::<W>::try_decode_u32_slice(&out, 0, &mut [0_u32; 0]).unwrap_err();
        assert_eq!(err, BitPackError::WidthOutOfRange { width: 0 });

        const W2: u32 = 33;
        let err = BitPacker::<W2>::try_encode_u32_slice(&[], &mut out).unwrap_err();
        assert_eq!(err, BitPackError::WidthOutOfRange { width: 33 });
    }

    #[test]
    fn try_const_packer_decode_returns_err_on_undersized_input() {
        const W: u32 = 11;
        let needed = BitPacker::<W>::encoded_len(8);
        let input = vec![0_u8; needed - 1];
        let mut out = vec![0_u32; 8];
        let err = BitPacker::<W>::try_decode_u32_slice(&input, 8, &mut out).unwrap_err();
        assert_eq!(
            err,
            BitPackError::InputTooShort {
                needed,
                actual: needed - 1
            }
        );
    }

    #[test]
    fn try_const_packer_round_trip_matches_panicking_version() {
        const W: u32 = 11;
        let values = deterministic_values(33, W, 0xC0FFEE);
        let mut try_encoded = vec![0_u8; BitPacker::<W>::encoded_len(33)];
        BitPacker::<W>::try_encode_u32_slice(&values, &mut try_encoded).unwrap();
        let mut decoded = vec![0_u32; 33];
        BitPacker::<W>::try_decode_u32_slice(&try_encoded, 33, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "encode output buffer too small")]
    fn dynamic_encode_still_panics_on_undersized_output() {
        let packer = DynamicBitPacker::new(11);
        let values = vec![1_u32; 8];
        let needed = packer.encoded_len(8);
        let mut out = vec![0_u8; needed - 1];
        packer.encode_u32_slice(&values, &mut out);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "decode input buffer too small")]
    fn dynamic_decode_still_panics_on_undersized_input() {
        let packer = DynamicBitPacker::new(11);
        let needed = packer.encoded_len(8);
        let input = vec![0_u8; needed - 1];
        let mut out = vec![0_u32; 8];
        packer.decode_u32_slice(&input, 8, &mut out);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn unaligned_output_offsets_round_trip() {
        // The decoder writes to the caller's slice; verify that
        // writing into a sub-slice of a larger buffer behaves
        // identically. This catches accidental absolute indexing into
        // `out` from inside SIMD kernels.
        for &w in &[1_u32, 7, 8, 11, 12, 16, 25, 32] {
            for &n in &[1_usize, 33, 257] {
                let values = deterministic_values(n, w, 0xBEEF_u64 ^ ((w as u64) << 4) ^ n as u64);
                let packer = DynamicBitPacker::new(w);
                let mut encoded = vec![0_u8; packer.encoded_len(n)];
                packer.encode_u32_slice(&values, &mut encoded);
                let mut padded = vec![0_u32; n + 16];
                packer.decode_u32_slice(&encoded, n, &mut padded[..n]);
                assert_eq!(&padded[..n], &values[..], "w={w} n={n}");
                assert!(
                    padded[n..].iter().all(|&v| v == 0),
                    "tail clobbered w={w} n={n}"
                );
            }
        }
    }

    // ------------------------------------------------------------------
    // Audit-R6 finding #162 regression tests for the `try_*` paths.
    //
    // These tests intentionally avoid the panicking entry points so the
    // assertion that the bit-pack `try_*` decoders are panic-free still
    // holds when the `panicking-shape-apis` Cargo feature is disabled
    // (the kernel/FUSE deployment build).
    // ------------------------------------------------------------------

    #[test]
    fn try_decode_dynamic_input_too_short_returns_err_not_panic() {
        // n=8, W=11 -> ceil(88/8) = 11 bytes needed; supply 7.
        let packer = DynamicBitPacker::try_new(11).expect("width valid");
        let input = [0_u8; 7];
        let mut out = [0_u32; 8];
        let err = packer
            .try_decode_u32_slice(&input, 8, &mut out)
            .expect_err("must return Err");
        assert!(
            matches!(err, BitPackError::InputTooShort { .. }),
            "expected InputTooShort, got {err:?}"
        );
    }

    #[test]
    fn try_decode_dynamic_output_too_short_returns_err_not_panic() {
        let packer = DynamicBitPacker::try_new(11).expect("width valid");
        let needed = packer.encoded_len(8);
        let input = vec![0_u8; needed];
        let mut out = vec![0_u32; 1];
        let err = packer
            .try_decode_u32_slice(&input, 8, &mut out)
            .expect_err("must return Err");
        assert_eq!(
            err,
            BitPackError::OutputTooShort {
                needed: 8,
                actual: 1
            }
        );
    }

    #[test]
    fn try_decode_const_packer_at_every_width_panic_free_round_trip() {
        // Encode + decode via try_ APIs only at every legal width so we
        // exercise the byte-aligned fast paths, the AVX2 / NEON regimes,
        // and the 25..=31 fall-through to the unchecked scalar without
        // any panicking entry point in the call chain.
        macro_rules! check {
            ($w:expr) => {{
                const W: u32 = $w;
                let n = 33_usize;
                let values = deterministic_values(n, W, 0xC0FFEE_u64 ^ (W as u64));
                let mut encoded = vec![0_u8; BitPacker::<W>::encoded_len(n)];
                BitPacker::<W>::try_encode_u32_slice(&values, &mut encoded).unwrap();
                let mut decoded = vec![0_u32; n];
                BitPacker::<W>::try_decode_u32_slice(&encoded, n, &mut decoded).unwrap();
                assert_eq!(decoded, values, "fallible round-trip diverged at W={W}");
            }};
        }
        check!(1);
        check!(3);
        check!(5);
        check!(7);
        check!(8);
        check!(11);
        check!(13);
        check!(16);
        check!(20);
        check!(24);
        check!(25);
        check!(27);
        check!(31);
        check!(32);
    }

    #[test]
    fn try_const_decode_width_out_of_range_returns_err_not_panic() {
        const W: u32 = 33;
        let input = [0_u8; 4];
        let mut out = [0_u32; 0];
        let err =
            BitPacker::<W>::try_decode_u32_slice(&input, 0, &mut out).expect_err("must return Err");
        assert_eq!(err, BitPackError::WidthOutOfRange { width: 33 });
    }

    #[test]
    fn try_dynamic_decode_width_out_of_range_returns_err_not_panic() {
        // Build via try_new bypass: forge a packer with width=0 by
        // re-using the type's bit-pattern through a struct literal —
        // but the public API only allows valid widths via try_new. To
        // verify the in-method check still triggers, we construct a
        // legal packer and rely on the const-generic test above for
        // out-of-range coverage on that path.
        // Instead, exercise the legal path's panic-free decode tail:
        let packer = DynamicBitPacker::try_new(7).expect("width valid");
        // n*W = 30 bits -> 4 bytes needed; supply 3.
        let input = [0_u8; 3];
        let mut out = [0_u32; 6];
        let err = packer
            .try_decode_u32_slice(&input, 6, &mut out)
            .expect_err("must return Err");
        assert!(
            matches!(err, BitPackError::InputTooShort { .. }),
            "expected InputTooShort, got {err:?}"
        );
    }
}

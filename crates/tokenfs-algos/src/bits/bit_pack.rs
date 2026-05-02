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
    pub fn encode_u32_slice(values: &[u32], out: &mut [u8]) {
        assert!(W >= 1 && W <= 32, "BitPacker width must be 1..=32");
        kernels::auto::encode_u32_slice(W, values, out);
    }

    /// Fallible variant of [`Self::encode_u32_slice`] that returns
    /// [`BitPackError`] when the output buffer is too small or the
    /// const-generic width `W` is out of range, instead of panicking.
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
        Self::encode_u32_slice(values, out);
        Ok(())
    }

    /// Unpacks `n` integers of `W` bits each from `input` into `out`.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() < Self::encoded_len(n)`, `out.len() < n`,
    /// or `W` is out of range `1..=32`. Use [`Self::try_decode_u32_slice`]
    /// for a fallible variant that returns [`BitPackError`] instead.
    pub fn decode_u32_slice(input: &[u8], n: usize, out: &mut [u32]) {
        assert!(W >= 1 && W <= 32, "BitPacker width must be 1..=32");
        kernels::auto::decode_u32_slice(W, input, n, out);
    }

    /// Fallible variant of [`Self::decode_u32_slice`] that returns
    /// [`BitPackError`] when the input buffer is too short, the output
    /// buffer is too small, or the const-generic width `W` is out of
    /// range, instead of panicking.
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
        Self::decode_u32_slice(input, n, out);
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
    pub fn encode_u32_slice(&self, values: &[u32], out: &mut [u8]) {
        kernels::auto::encode_u32_slice(self.width, values, out);
    }

    /// Fallible variant of [`Self::encode_u32_slice`] that returns
    /// [`BitPackError`] when the output buffer is too small or the
    /// configured width is out of range, instead of panicking.
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
        self.encode_u32_slice(values, out);
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
    pub fn decode_u32_slice(&self, input: &[u8], n: usize, out: &mut [u32]) {
        kernels::auto::decode_u32_slice(self.width, input, n, out);
    }

    /// Fallible variant of [`Self::decode_u32_slice`] that returns
    /// [`BitPackError`] when the input buffer is too short, the output
    /// buffer is too small, or the configured width is out of range,
    /// instead of panicking.
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
        self.decode_u32_slice(input, n, out);
        Ok(())
    }
}

/// Pinned bit-pack kernels.
pub mod kernels {
    /// Runtime-dispatched bit-pack kernels.
    pub mod auto {
        /// Packs `values` at width `w` into `out` using the best
        /// available kernel.
        pub fn encode_u32_slice(w: u32, values: &[u32], out: &mut [u8]) {
            // Encode is bandwidth-modest and the per-element work is
            // dominated by cross-byte stores; the scalar path already
            // runs ~2-3 GB/s and SIMD wins are small. Keep one path.
            super::scalar::encode_u32_slice(w, values, out);
        }

        /// Unpacks `n` values at width `w` from `input` into `out`
        /// using the best available kernel.
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
    }

    /// Portable scalar bit-pack reference.
    ///
    /// This is the oracle every SIMD backend must match bit-exactly.
    /// The hot loops are intentionally simple — the compiler can
    /// usually inline and unroll the const-generic
    /// [`super::super::BitPacker`] call site to within ~10% of the
    /// SIMD paths for `W <= 8`.
    pub mod scalar {
        use super::super::encoded_len_bytes;

        /// Packs `values` at width `w` into `out`.
        ///
        /// # Panics
        ///
        /// Panics if `out.len() < ceil(values.len()*w/8)` or `w` is
        /// outside `1..=32`.
        pub fn encode_u32_slice(w: u32, values: &[u32], out: &mut [u8]) {
            assert!((1..=32).contains(&w), "width must be 1..=32");
            let needed = encoded_len_bytes(values.len(), w);
            assert!(
                out.len() >= needed,
                "encode output buffer too small: {} < {}",
                out.len(),
                needed
            );
            // Zero exactly the needed prefix; the OR-into-existing-bytes
            // strategy below requires a clean slate.
            for byte in &mut out[..needed] {
                *byte = 0;
            }

            // Byte-aligned fast paths. Hard-code these widths because
            // they degenerate to memcpy / byte-cast and are the
            // canonical token / fingerprint widths.
            match w {
                8 => {
                    for (i, &v) in values.iter().enumerate() {
                        out[i] = v as u8;
                    }
                    return;
                }
                16 => {
                    for (i, &v) in values.iter().enumerate() {
                        let bytes = (v as u16).to_le_bytes();
                        out[2 * i] = bytes[0];
                        out[2 * i + 1] = bytes[1];
                    }
                    return;
                }
                32 => {
                    for (i, &v) in values.iter().enumerate() {
                        let bytes = v.to_le_bytes();
                        let off = 4 * i;
                        out[off] = bytes[0];
                        out[off + 1] = bytes[1];
                        out[off + 2] = bytes[2];
                        out[off + 3] = bytes[3];
                    }
                    return;
                }
                _ => {}
            }

            let mask: u64 = if w == 32 {
                u32::MAX as u64
            } else {
                (1_u64 << w) - 1
            };
            for (i, &v) in values.iter().enumerate() {
                let bit_pos = i * (w as usize);
                let byte = bit_pos / 8;
                let shift = (bit_pos % 8) as u32;
                // Up to 5 bytes are touched: 32 bits + 7 in-byte
                // offset = 39 bits, fits in 5 bytes.
                let masked = (v as u64) & mask;
                let shifted = masked << shift;
                let span_bits = shift + w;
                let span_bytes = (span_bits as usize).div_ceil(8);
                let bytes = shifted.to_le_bytes();
                for k in 0..span_bytes {
                    out[byte + k] |= bytes[k];
                }
            }
        }

        /// Unpacks `n` values at width `w` from `input` into `out`.
        ///
        /// # Panics
        ///
        /// Panics if `input.len() < ceil(n*w/8)`, `out.len() < n`, or
        /// `w` is outside `1..=32`.
        pub fn decode_u32_slice(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
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

            // Byte-aligned fast paths mirror the encode-side specialization.
            match w {
                8 => {
                    for (i, slot) in out.iter_mut().take(n).enumerate() {
                        *slot = input[i] as u32;
                    }
                    return;
                }
                16 => {
                    for (i, slot) in out.iter_mut().take(n).enumerate() {
                        let bytes = [input[2 * i], input[2 * i + 1]];
                        *slot = u16::from_le_bytes(bytes) as u32;
                    }
                    return;
                }
                32 => {
                    for (i, slot) in out.iter_mut().take(n).enumerate() {
                        let off = 4 * i;
                        let bytes = [input[off], input[off + 1], input[off + 2], input[off + 3]];
                        *slot = u32::from_le_bytes(bytes);
                    }
                    return;
                }
                _ => {}
            }

            let mask: u64 = if w == 32 {
                u32::MAX as u64
            } else {
                (1_u64 << w) - 1
            };
            // Fast inner loop: load 8 bytes whenever there are at least
            // 8 bytes ahead in the buffer. This covers all but the last
            // few elements without bounds-checking each byte.
            let bulk = if needed >= 8 {
                let max_byte = input.len() - 8;
                let mut count = 0_usize;
                while count < n {
                    let bit_pos = count * (w as usize);
                    if bit_pos / 8 > max_byte {
                        break;
                    }
                    count += 1;
                }
                count
            } else {
                0
            };

            for (i, slot) in out.iter_mut().take(bulk).enumerate() {
                let bit_pos = i * (w as usize);
                let byte = bit_pos / 8;
                let shift = (bit_pos % 8) as u32;
                let raw = u64::from_le_bytes([
                    input[byte],
                    input[byte + 1],
                    input[byte + 2],
                    input[byte + 3],
                    input[byte + 4],
                    input[byte + 5],
                    input[byte + 6],
                    input[byte + 7],
                ]);
                *slot = ((raw >> shift) & mask) as u32;
            }

            // Tail: each value loads only the bytes that actually
            // exist. Constructed via a 64-bit accumulator so the same
            // shift/mask logic works.
            for i in bulk..n {
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
    }

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
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
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

            // Byte-aligned widths and the very small widths fall back
            // to scalar — the SIMD setup overhead exceeds the savings
            // on a memcpy-shaped workload.
            match w {
                1 | 2 | 4 | 8 | 16 | 32 => {
                    scalar::decode_u32_slice(w, input, n, out);
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
                scalar::decode_u32_slice(w, input, n, out);
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
    }

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
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    pub mod neon {
        use super::super::encoded_len_bytes;
        use super::scalar;

        use core::arch::aarch64::{
            uint32x4_t, uint64x2_t, vandq_u32, vandq_u64, vdupq_n_u32, vdupq_n_u64, vld1q_s32,
            vld1q_s64, vld1q_u32, vld1q_u64, vshlq_u32, vshlq_u64, vst1q_u32, vst1q_u64,
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

            match w {
                1 | 2 | 4 | 8 | 16 | 32 => {
                    scalar::decode_u32_slice(w, input, n, out);
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
                scalar::decode_u32_slice(w, input, n, out);
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
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

    use super::*;

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
                kernels::scalar::encode_u32_slice(w, &values, &mut encoded);

                let mut expected = vec![0_u32; n];
                kernels::scalar::decode_u32_slice(w, &encoded, n, &mut expected);

                let mut actual = vec![0_u32; n];
                // SAFETY: avx2_available() returned true above.
                unsafe { kernels::avx2::decode_u32_slice(w, &encoded, n, &mut actual) };
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
        let packer = DynamicBitPacker::new(11);
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
        let packer = DynamicBitPacker::new(11);
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
        let packer = DynamicBitPacker::new(11);
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

    #[test]
    #[should_panic(expected = "encode output buffer too small")]
    fn dynamic_encode_still_panics_on_undersized_output() {
        let packer = DynamicBitPacker::new(11);
        let values = vec![1_u32; 8];
        let needed = packer.encoded_len(8);
        let mut out = vec![0_u8; needed - 1];
        packer.encode_u32_slice(&values, &mut out);
    }

    #[test]
    #[should_panic(expected = "decode input buffer too small")]
    fn dynamic_decode_still_panics_on_undersized_input() {
        let packer = DynamicBitPacker::new(11);
        let needed = packer.encoded_len(8);
        let input = vec![0_u8; needed - 1];
        let mut out = vec![0_u32; 8];
        packer.decode_u32_slice(&input, 8, &mut out);
    }

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
}

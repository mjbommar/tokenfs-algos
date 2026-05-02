//! Stream-VByte codec — Lemire & Kurz 2017 variable-byte integer compression
//! for `u32` streams with SIMD-accelerated decode.
//!
//! See `docs/v0.2_planning/10_BITS.md` § 3 for the spec, hardware plan, and
//! vendor decision. The format is **wire-compatible** with Lemire's canonical
//! C reference at <https://github.com/lemire/streamvbyte> and the upstream
//! `stream-vbyte` Rust crate.
//!
//! ## Format
//!
//! Every group of four `u32` values produces:
//!
//! * One **control byte** packing four 2-bit length codes. Code `cc` means
//!   the integer takes `cc + 1` little-endian bytes (`00`→1B, `01`→2B,
//!   `10`→3B, `11`→4B). The first 2-bit word is the **least** significant
//!   bits of the control byte; `x1` is bits 0-1, `x4` is bits 6-7.
//! * Each integer's `len_i` low-order bytes appended to the data stream.
//!
//! All control bytes go into a separate stream of `ceil(N/4)` bytes; all
//! data bytes follow in `Σ_i len_i` bytes. The element count `N` is
//! length-prefixed by the container — it is **not** stored in the stream.
//!
//! For `N % 4 != 0` the encoder pads the final group with `00` codes (one
//! data byte each, value zero); the decoder writes only `N` outputs and
//! ignores the trailing slots.
//!
//! ## Public API
//!
//! * [`streamvbyte_encode_u32`] — encode `&[u32]` into separate control +
//!   data streams.
//! * [`streamvbyte_decode_u32`] — decode the streams back into `&mut [u32]`.
//! * [`streamvbyte_control_len`] — `ceil(N/4)`.
//! * [`streamvbyte_data_max_len`] — worst-case data byte length: `4 * N`
//!   plus padding bytes for the partial final group when `N % 4 != 0`.
//! * [`kernels::scalar`] — portable byte-by-byte reference.
//! * `kernels::ssse3` — PSHUFB-based 16-byte chunks (`feature = "avx2"`,
//!   uses SSSE3 only at runtime).
//! * `kernels::avx2` — dual-pumped PSHUFB, two control bytes per iteration
//!   (`feature = "avx2"`).
//! * `kernels::neon` — `vqtbl1q_u8`-based 16-byte chunks
//!   (`feature = "neon"`).
//!
//! ## Decode hot path
//!
//! For each control byte `c`:
//! 1. `LENGTH_TABLE[c]` → number of data bytes consumed (4..=16).
//! 2. Load 16 data bytes (overshoot OK if buffer padded ≥16B at tail).
//! 3. `SHUFFLE_TABLE[c]` → `[u8; 16]` permutation; index `0xFF` zeros the
//!    target lane on both PSHUFB and `vqtbl1q_u8`.
//! 4. One PSHUFB / `vqtbl1q_u8` produces four little-endian `u32` lanes.
//! 5. Advance the data pointer by the table-derived length.
//!
//! The final group is always decoded via the scalar tail when fewer than
//! 16 bytes of valid data remain, which keeps the SIMD inner loop free of
//! buffer-overrun checks.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_range_loop)]

/// Number of `u32` values packed per control byte.
const GROUP: usize = 4;

/// Returns the number of control bytes needed to encode `n` `u32` values.
///
/// One control byte covers up to four values, so this is `ceil(n / 4)`.
#[must_use]
pub const fn streamvbyte_control_len(n: usize) -> usize {
    n.div_ceil(GROUP)
}

/// Returns the worst-case data-byte length for `n` `u32` values.
///
/// Each actual value occupies 1..=4 bytes (worst case 4); when `n` is not
/// a multiple of 4, the encoder pads the trailing partial group with
/// 1-byte zero codes so encoder and decoder data offsets always agree.
/// The bound therefore covers all four slots of the final group:
/// `4 * ceil(n / 4) * 4` in the absolute worst case, but tightened to
/// `4 * n + 3 * (4 - n % 4) % 4` since padding slots are always 1B each.
///
/// Callers sizing fixed buffers should use this; the actual encoded
/// length is returned by [`streamvbyte_encode_u32`].
#[must_use]
pub const fn streamvbyte_data_max_len(n: usize) -> usize {
    let pad = (GROUP - (n % GROUP)) % GROUP;
    n.saturating_mul(4).saturating_add(pad)
}

/// Encodes `values` into separate control + data byte streams.
///
/// Writes `streamvbyte_control_len(values.len())` control bytes to
/// `control_out` and the variable-length data bytes to `data_out`. Returns
/// the number of data bytes written.
///
/// # Panics
///
/// Panics if `control_out.len() < streamvbyte_control_len(values.len())`
/// or `data_out.len() < streamvbyte_data_max_len(values.len())`. Use
/// [`streamvbyte_data_max_len`] to size `data_out`; the actual byte count
/// is the return value.
pub fn streamvbyte_encode_u32(
    values: &[u32],
    control_out: &mut [u8],
    data_out: &mut [u8],
) -> usize {
    kernels::auto::encode_u32(values, control_out, data_out)
}

/// Decodes `n` `u32` values from separate control + data byte streams.
///
/// Reads `streamvbyte_control_len(n)` control bytes from `control` and the
/// matching data bytes from `data`; writes `n` values into `out`. Returns
/// the number of data bytes consumed.
///
/// # Panics
///
/// Panics if `control.len() < streamvbyte_control_len(n)` or `out.len() < n`.
/// `data` must be long enough to hold the bytes implied by the control
/// stream — when in doubt, size it via [`streamvbyte_data_max_len`].
pub fn streamvbyte_decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize {
    kernels::auto::decode_u32(control, data, n, out)
}

/// Pinned Stream-VByte kernels.
pub mod kernels {
    /// Runtime-dispatched Stream-VByte kernels.
    pub mod auto {
        /// Runtime-dispatched encode.
        pub fn encode_u32(values: &[u32], control_out: &mut [u8], data_out: &mut [u8]) -> usize {
            // Encode is bandwidth-modest; the scalar path already runs at
            // near memory speed. SIMD-encode wins are small and the spec
            // explicitly defers them.
            super::scalar::encode_u32(values, control_out, data_out)
        }

        /// Runtime-dispatched decode.
        pub fn decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize {
            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability checked immediately above.
                    return unsafe { super::avx2::decode_u32(control, data, n, out) };
                }
                if super::ssse3::is_available() {
                    // SAFETY: availability checked immediately above.
                    return unsafe { super::ssse3::decode_u32(control, data, n, out) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { super::neon::decode_u32(control, data, n, out) };
                }
            }

            super::scalar::decode_u32(control, data, n, out)
        }
    }

    /// Portable scalar reference oracle.
    ///
    /// Byte-by-byte encode and decode following the Lemire/Kurz spec
    /// directly. SIMD backends must match this bit-exactly.
    pub mod scalar {
        use super::super::{GROUP, streamvbyte_control_len, streamvbyte_data_max_len};

        /// Returns the per-value length code for `v`: `0..=3`, where the
        /// encoded byte width is `code + 1`.
        #[inline]
        const fn code_for(v: u32) -> u8 {
            // `v.leading_zeros()` is 32 for v=0, so the formula yields 0
            // and the value occupies 1 byte (the 0x00 byte). Otherwise
            // `(32 - lz + 7) / 8 - 1` ∈ {0, 1, 2, 3}.
            if v < 1 << 8 {
                0
            } else if v < 1 << 16 {
                1
            } else if v < 1 << 24 {
                2
            } else {
                3
            }
        }

        /// Encodes `values` using the spec's reference algorithm.
        ///
        /// # Panics
        ///
        /// Panics if `control_out` is shorter than
        /// [`streamvbyte_control_len`] or `data_out` is shorter than
        /// [`streamvbyte_data_max_len`].
        pub fn encode_u32(values: &[u32], control_out: &mut [u8], data_out: &mut [u8]) -> usize {
            let n = values.len();
            assert!(
                control_out.len() >= streamvbyte_control_len(n),
                "control_out too small: {} < {}",
                control_out.len(),
                streamvbyte_control_len(n)
            );
            assert!(
                data_out.len() >= streamvbyte_data_max_len(n),
                "data_out too small: {} < {}",
                data_out.len(),
                streamvbyte_data_max_len(n)
            );

            let mut data_pos = 0_usize;
            let mut ctrl_pos = 0_usize;

            // Full groups of 4 → one control byte each.
            let full_groups = n / GROUP;
            for g in 0..full_groups {
                let base = g * GROUP;
                let v0 = values[base];
                let v1 = values[base + 1];
                let v2 = values[base + 2];
                let v3 = values[base + 3];
                let c0 = code_for(v0);
                let c1 = code_for(v1);
                let c2 = code_for(v2);
                let c3 = code_for(v3);
                control_out[ctrl_pos] = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                ctrl_pos += 1;

                data_pos += write_value(v0, c0, &mut data_out[data_pos..]);
                data_pos += write_value(v1, c1, &mut data_out[data_pos..]);
                data_pos += write_value(v2, c2, &mut data_out[data_pos..]);
                data_pos += write_value(v3, c3, &mut data_out[data_pos..]);
            }

            // Tail group with N % 4 ∈ {1, 2, 3}: pad missing slots with
            // code 0 (1 data byte = 0x00). The padding bytes still count
            // toward `data_pos` so a round-trip read sees the encoded
            // bytes the encoder wrote.
            let tail = n - full_groups * GROUP;
            if tail > 0 {
                let mut codes = [0_u8; GROUP];
                let mut payload = [0_u32; GROUP];
                for k in 0..tail {
                    payload[k] = values[full_groups * GROUP + k];
                    codes[k] = code_for(payload[k]);
                }
                control_out[ctrl_pos] =
                    codes[0] | (codes[1] << 2) | (codes[2] << 4) | (codes[3] << 6);
                ctrl_pos += 1;
                for k in 0..GROUP {
                    data_pos += write_value(payload[k], codes[k], &mut data_out[data_pos..]);
                }
            }

            let _ = ctrl_pos; // Asserted-via-bounds rather than returned.
            data_pos
        }

        /// Writes the `code + 1` low-order little-endian bytes of `v` into
        /// `dst` and returns that length.
        #[inline]
        fn write_value(v: u32, code: u8, dst: &mut [u8]) -> usize {
            let len = (code as usize) + 1;
            let bytes = v.to_le_bytes();
            dst[..len].copy_from_slice(&bytes[..len]);
            len
        }

        /// Decodes `n` values; returns the number of data bytes consumed.
        ///
        /// # Panics
        ///
        /// Panics if `control` has fewer than `streamvbyte_control_len(n)`
        /// bytes, `out` is shorter than `n`, or `data` runs out before the
        /// implied length is reached.
        pub fn decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize {
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

            let mut data_pos = 0_usize;
            let mut written = 0_usize;
            let full_groups = n / GROUP;
            for g in 0..full_groups {
                let c = control[g];
                let codes = [c & 0b11, (c >> 2) & 0b11, (c >> 4) & 0b11, (c >> 6) & 0b11];
                for k in 0..GROUP {
                    let len = (codes[k] as usize) + 1;
                    out[written + k] = read_value(&data[data_pos..], len);
                    data_pos += len;
                }
                written += GROUP;
            }

            // Tail: only `n - full_groups * GROUP` outputs requested,
            // even though the control byte covers four codes. We must
            // still advance `data_pos` past every code the encoder wrote
            // to make decoder/encoder offset bookkeeping match.
            let tail = n - full_groups * GROUP;
            if tail > 0 {
                let c = control[full_groups];
                let codes = [c & 0b11, (c >> 2) & 0b11, (c >> 4) & 0b11, (c >> 6) & 0b11];
                for k in 0..GROUP {
                    let len = (codes[k] as usize) + 1;
                    if k < tail {
                        out[written + k] = read_value(&data[data_pos..], len);
                    }
                    data_pos += len;
                }
            }

            data_pos
        }

        /// Reads a little-endian integer of `len ∈ 1..=4` bytes from
        /// `src` into a `u32`.
        #[inline]
        fn read_value(src: &[u8], len: usize) -> u32 {
            let mut bytes = [0_u8; 4];
            bytes[..len].copy_from_slice(&src[..len]);
            u32::from_le_bytes(bytes)
        }
    }

    /// Per-control-byte length and shuffle tables for the SIMD decoders.
    ///
    /// Both tables are pure functions of the control byte and are derived
    /// once at first use. The shuffle table follows the Lemire C
    /// reference: index `0xFF` zeros the destination lane (both
    /// `_mm_shuffle_epi8` and `vqtbl1q_u8` clear bytes for indices with
    /// bit 7 set / >= 16).
    #[cfg(any(
        all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    pub(super) mod tables {
        /// `LENGTH_TABLE[c]` is the number of data bytes the control byte
        /// `c` consumes — the sum of `(code_i + 1)` over the four codes.
        /// Range is 4..=16.
        ///
        /// `SHUFFLE_TABLE[c]` is the 16-byte permutation that turns a
        /// 16-byte data window into four little-endian `u32` lanes; lanes
        /// past the encoded width hold `0xFF` so the SIMD shuffle zeros
        /// them.
        ///
        /// Constructed at runtime once and shared via [`OnceLock`]. The
        /// table is 256 × 16 = 4 KiB plus 256 × 1 = 256 B; at first use
        /// the cost is dominated by computing the shuffle pattern, which
        /// is constant work.
        use std::sync::OnceLock;

        /// Shuffle table: 256 entries of `[u8; 16]`. 4 KiB total.
        static SHUFFLE: OnceLock<[[u8; 16]; 256]> = OnceLock::new();
        /// Length table: 256 entries of `u8`. 256 B total.
        static LENGTHS: OnceLock<[u8; 256]> = OnceLock::new();

        /// Returns the (potentially first-use-initialized) shuffle table.
        #[inline]
        pub(crate) fn shuffle_table() -> &'static [[u8; 16]; 256] {
            SHUFFLE.get_or_init(build_shuffle_table)
        }

        /// Returns the (potentially first-use-initialized) length table.
        #[inline]
        pub(crate) fn length_table() -> &'static [u8; 256] {
            LENGTHS.get_or_init(build_length_table)
        }

        /// Builds the shuffle table.
        ///
        /// For each control byte `c`, decode the four 2-bit codes
        /// `c1, c2, c3, c4` (low→high). Each integer takes `code_i + 1`
        /// data bytes. The output is four little-endian u32s in 16
        /// contiguous bytes; we fill the first `code_i + 1` of each
        /// 4-byte lane with consecutive source byte indices, and the
        /// remaining lanes with `0xFF` so PSHUFB / TBL writes zero.
        fn build_shuffle_table() -> [[u8; 16]; 256] {
            let mut table = [[0xff_u8; 16]; 256];
            for c in 0_usize..256 {
                let codes = [
                    (c & 0b11) as u8,
                    ((c >> 2) & 0b11) as u8,
                    ((c >> 4) & 0b11) as u8,
                    ((c >> 6) & 0b11) as u8,
                ];
                let mut shuffle = [0xff_u8; 16];
                let mut src = 0_u8;
                for lane in 0..4 {
                    let len = codes[lane] + 1;
                    for byte in 0..len {
                        shuffle[lane * 4 + byte as usize] = src;
                        src += 1;
                    }
                    // Lane bytes [len..4] stay 0xFF → zeroed by shuffle.
                }
                table[c] = shuffle;
            }
            table
        }

        /// Builds the length table.
        fn build_length_table() -> [u8; 256] {
            let mut table = [0_u8; 256];
            for c in 0_usize..256 {
                let l = (c & 0b11) + ((c >> 2) & 0b11) + ((c >> 4) & 0b11) + ((c >> 6) & 0b11) + 4;
                table[c] = l as u8;
            }
            table
        }
    }

    /// SSSE3 + SSE4.1 decode kernel.
    ///
    /// One PSHUFB per control byte produces four little-endian `u32`s
    /// from a 16-byte data window. Gated on the crate's `avx2` feature
    /// (which is the umbrella for runtime-detected SIMD x86 backends);
    /// runtime check is for SSSE3 only.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod ssse3 {
        use super::super::{GROUP, streamvbyte_control_len};
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
                data_pos += scalar::decode_u32(
                    &control[g..],
                    &data[data_pos..],
                    n - written,
                    &mut out[written..],
                );
            }

            data_pos
        }
    }

    /// AVX2 dual-pumped decode kernel.
    ///
    /// Each iteration consumes two control bytes and stores eight
    /// little-endian `u32`s. The two halves are decoded into a `__m128i`
    /// each via PSHUFB (per-lane on AVX2 the equivalent is two PSHUFBs);
    /// stitched together via `_mm256_inserti128_si256`. Falls back to the
    /// SSSE3 inner loop for the trailing odd group, then to scalar for
    /// the unsafe tail.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::super::{GROUP, streamvbyte_control_len};
        use super::scalar;
        use super::tables::{length_table, shuffle_table};

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m128i, __m256i, _mm_loadu_si128, _mm_shuffle_epi8, _mm256_inserti128_si256,
            _mm256_storeu_si256,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m128i, __m256i, _mm_loadu_si128, _mm_shuffle_epi8, _mm256_inserti128_si256,
            _mm256_storeu_si256,
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

        /// AVX2 dual-pumped PSHUFB decode.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
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

            let shuf = shuffle_table();
            let lens = length_table();

            let full_groups = n / GROUP;
            let mut data_pos = 0_usize;
            let mut g = 0_usize;

            // Dual-pumped: process two control bytes per iteration.
            // Each half needs 16 bytes of safe data; check both before
            // entering the body.
            while g + 2 <= full_groups {
                let c0 = control[g] as usize;
                let c1 = control[g + 1] as usize;
                let len0 = lens[c0] as usize;
                let len1 = lens[c1] as usize;

                // Bounds check both halves against `data` before issuing
                // the unaligned 16-byte loads.
                if data_pos + 16 > data.len() || data_pos + len0 + 16 > data.len() {
                    break;
                }

                // SAFETY: bounds checked above; AVX2 (which implies SSSE3)
                // enabled on the enclosing fn; the two output stores fall
                // inside `out[g*4 .. g*4 + 8]`, in-bounds because
                // `g + 2 <= full_groups`, so `g*4 + 8 <= n <= out.len()`.
                unsafe {
                    let v0 = _mm_loadu_si128(data.as_ptr().add(data_pos).cast::<__m128i>());
                    let s0 = _mm_loadu_si128(shuf[c0].as_ptr().cast::<__m128i>());
                    let r0 = _mm_shuffle_epi8(v0, s0);

                    let v1 = _mm_loadu_si128(data.as_ptr().add(data_pos + len0).cast::<__m128i>());
                    let s1 = _mm_loadu_si128(shuf[c1].as_ptr().cast::<__m128i>());
                    let r1 = _mm_shuffle_epi8(v1, s1);

                    // Compose two __m128i into one __m256i and store both
                    // 16-byte halves in a single 32-byte unaligned store.
                    let lo = _mm256_inserti128_si256::<0>(core::mem::zeroed::<__m256i>(), r0);
                    let combined = _mm256_inserti128_si256::<1>(lo, r1);
                    _mm256_storeu_si256(
                        out.as_mut_ptr().add(g * GROUP).cast::<__m256i>(),
                        combined,
                    );
                }

                data_pos += len0 + len1;
                g += 2;
            }

            // Single-group SSSE3 path for the residual full group when
            // `full_groups` is odd.
            while g < full_groups {
                let c = control[g] as usize;
                let len = lens[c] as usize;
                if data_pos + 16 > data.len() {
                    break;
                }
                // SAFETY: bounds checked above; AVX2 implies SSSE3 so
                // PSHUFB is available; output lane indexing matches the
                // SSSE3 kernel.
                unsafe {
                    let v = _mm_loadu_si128(data.as_ptr().add(data_pos).cast::<__m128i>());
                    let s = _mm_loadu_si128(shuf[c].as_ptr().cast::<__m128i>());
                    let r = _mm_shuffle_epi8(v, s);
                    let dst = out.as_mut_ptr().add(g * GROUP).cast::<__m128i>();
                    core::arch::x86_64::_mm_storeu_si128(dst, r);
                }
                data_pos += len;
                g += 1;
            }

            // Scalar tail: residual full groups that didn't have 16
            // bytes of data slack, plus any partial trailing group.
            let written = g * GROUP;
            if written < n {
                data_pos += scalar::decode_u32(
                    &control[g..],
                    &data[data_pos..],
                    n - written,
                    &mut out[written..],
                );
            }

            data_pos
        }
    }

    /// AArch64 NEON decode kernel.
    ///
    /// `vqtbl1q_u8` performs the same 16-byte permute as `_mm_shuffle_epi8`;
    /// indices outside `[0, 15]` produce zero, so the same `0xFF` sentinel
    /// works on both ISAs and the shuffle table is shared between them.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    pub mod neon {
        use super::super::{GROUP, streamvbyte_control_len};
        use super::scalar;
        use super::tables::{length_table, shuffle_table};

        use core::arch::aarch64::{uint8x16_t, vld1q_u8, vqtbl1q_u8, vst1q_u8};

        /// Returns true when NEON is available at runtime.
        ///
        /// NEON is mandatory on AArch64; this exists for API symmetry.
        #[must_use]
        pub const fn is_available() -> bool {
            true
        }

        /// NEON `vqtbl1q_u8` decode.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports NEON.
        #[target_feature(enable = "neon")]
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

            let shuf = shuffle_table();
            let lens = length_table();

            let full_groups = n / GROUP;
            let mut data_pos = 0_usize;
            let mut g = 0_usize;

            while g < full_groups {
                let c = control[g] as usize;
                let len = lens[c] as usize;
                if data_pos + 16 > data.len() {
                    break;
                }

                // SAFETY: bounds checked above; NEON enabled on enclosing
                // fn; output lane `g*4..g*4+4` is in-bounds because
                // `g < full_groups <= n / 4`.
                unsafe {
                    let v: uint8x16_t = vld1q_u8(data.as_ptr().add(data_pos));
                    let s: uint8x16_t = vld1q_u8(shuf[c].as_ptr());
                    let r: uint8x16_t = vqtbl1q_u8(v, s);
                    vst1q_u8(out.as_mut_ptr().add(g * GROUP).cast::<u8>(), r);
                }

                data_pos += len;
                g += 1;
            }

            let written = g * GROUP;
            if written < n {
                data_pos += scalar::decode_u32(
                    &control[g..],
                    &data[data_pos..],
                    n - written,
                    &mut out[written..],
                );
            }

            data_pos
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_values(n: usize, seed: u64, max_bytes: u32) -> Vec<u32> {
        let mask: u32 = match max_bytes {
            1 => 0x0000_00ff,
            2 => 0x0000_ffff,
            3 => 0x00ff_ffff,
            _ => u32::MAX,
        };
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32) & mask
            })
            .collect()
    }

    #[test]
    fn control_len_matches_ceil_formula() {
        for n in [0_usize, 1, 2, 3, 4, 5, 99, 100, 1024] {
            assert_eq!(streamvbyte_control_len(n), n.div_ceil(4));
        }
    }

    #[test]
    fn data_max_len_includes_padding_for_partial_groups() {
        // Full groups: bound is 4 * n.
        assert_eq!(streamvbyte_data_max_len(0), 0);
        assert_eq!(streamvbyte_data_max_len(4), 16);
        assert_eq!(streamvbyte_data_max_len(100), 400);
        // Partial groups: each missing slot pads with 1 byte.
        assert_eq!(streamvbyte_data_max_len(1), 4 + 3);
        assert_eq!(streamvbyte_data_max_len(2), 8 + 2);
        assert_eq!(streamvbyte_data_max_len(3), 12 + 1);
        assert_eq!(streamvbyte_data_max_len(5), 20 + 3);
    }

    #[test]
    fn round_trip_n_zero_writes_nothing() {
        let mut ctrl = [0_u8; 0];
        let mut data = [0_u8; 0];
        let written = streamvbyte_encode_u32(&[], &mut ctrl, &mut data);
        assert_eq!(written, 0);
        let mut out: Vec<u32> = Vec::new();
        let consumed = streamvbyte_decode_u32(&ctrl, &data, 0, &mut out);
        assert_eq!(consumed, 0);
    }

    fn round_trip(values: &[u32]) {
        let n = values.len();
        let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
        let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
        let written = streamvbyte_encode_u32(values, &mut ctrl, &mut data);
        let mut out = vec![0_u32; n];
        let consumed = streamvbyte_decode_u32(&ctrl, &data[..written], n, &mut out);
        assert_eq!(consumed, written, "encoder vs decoder offset disagreed");
        assert_eq!(out, values, "round-trip diverged at n={n}");
    }

    #[test]
    fn round_trip_all_widths_and_sizes() {
        for n in [0_usize, 1, 2, 3, 4, 5, 7, 8, 16, 100, 1024] {
            for max in [1_u32, 2, 3, 4] {
                let values =
                    deterministic_values(n, 0xC0FFEE_u64 ^ (n as u64) ^ ((max as u64) << 16), max);
                round_trip(&values);
            }
        }
    }

    #[test]
    fn round_trip_each_code_specifically() {
        // Mix one of each code in a single group; verify byte budget and
        // value recovery.
        let values: Vec<u32> = vec![0x00, 0xff, 0x1234, 0x00ab_cdef, 0x1234_5678];
        round_trip(&values);
    }

    #[test]
    fn round_trip_n_one_through_seven() {
        // Touches every (full-groups, tail) shape:
        // n=1 → 0 full + tail 1
        // n=2 → 0 full + tail 2
        // n=3 → 0 full + tail 3
        // n=4 → 1 full + tail 0
        // n=5 → 1 full + tail 1
        // n=6 → 1 full + tail 2
        // n=7 → 1 full + tail 3
        for n in 1_usize..=7 {
            for max in [1_u32, 4] {
                let values = deterministic_values(n, 0x1234_5678 ^ (n as u64), max);
                round_trip(&values);
            }
        }
    }

    #[test]
    fn round_trip_max_u32_values_use_four_bytes() {
        // Force the 4-byte code on every value in a full group.
        let values = vec![u32::MAX, u32::MAX - 1, 0xffff_fff0, 0x8000_0000];
        let n = values.len();
        let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
        let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
        let written = streamvbyte_encode_u32(&values, &mut ctrl, &mut data);
        // Each value is 4 bytes → 16 total.
        assert_eq!(written, 16);
        // Control byte: every code is 11 → 0xFF.
        assert_eq!(ctrl[0], 0xff);
        let mut out = vec![0_u32; n];
        streamvbyte_decode_u32(&ctrl, &data[..written], n, &mut out);
        assert_eq!(out, values);
    }

    #[test]
    fn encoded_size_matches_spec_for_known_values() {
        // Each value is 1 byte → 4 data bytes total.
        let values = vec![0_u32, 1, 2, 3];
        let n = values.len();
        let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
        let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
        let written = streamvbyte_encode_u32(&values, &mut ctrl, &mut data);
        assert_eq!(written, 4);
        assert_eq!(ctrl[0], 0x00);
        // Bytes themselves should match the value LSBs.
        assert_eq!(&data[..4], &[0, 1, 2, 3]);
    }

    // Table tests apply only when at least one SIMD backend (which owns
    // the tables module) is compiled in. The encoder/decoder don't use
    // the tables themselves; they're a pure-SIMD asset.
    #[cfg(any(
        all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    #[test]
    fn shuffle_table_known_entries() {
        // Spot-check a few well-known shuffle patterns. The byte index
        // 0xFF is the "zero" sentinel for both PSHUFB and vqtbl1q_u8.
        let table = kernels::tables::shuffle_table();

        // c=0x00: every code is 0 → every value is 1 byte. Source bytes
        // are at positions 0..4; each is placed in the low byte of its
        // u32 lane and the upper three bytes are zeroed.
        assert_eq!(
            table[0x00],
            [
                0, 0xff, 0xff, 0xff, 1, 0xff, 0xff, 0xff, 2, 0xff, 0xff, 0xff, 3, 0xff, 0xff, 0xff
            ]
        );

        // c=0xFF: every code is 3 → every value is 4 bytes. Source bytes
        // are 0..16, each placed in its u32 lane verbatim.
        assert_eq!(
            table[0xff],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        );

        // c=0x55: every pair of bits is 01 → every value is 2 bytes.
        // Source bytes 0..8.
        assert_eq!(
            table[0x55],
            [
                0, 1, 0xff, 0xff, 2, 3, 0xff, 0xff, 4, 5, 0xff, 0xff, 6, 7, 0xff, 0xff
            ]
        );
    }

    #[cfg(any(
        all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    #[test]
    fn length_table_known_entries() {
        let lens = kernels::tables::length_table();
        assert_eq!(lens[0x00], 4); // 4 × 1B
        assert_eq!(lens[0xff], 16); // 4 × 4B
        assert_eq!(lens[0x55], 8); // 4 × 2B
        assert_eq!(lens[0xaa], 12); // 4 × 3B
        // Mixed: c1=00 (1B), c2=11 (4B), c3=00 (1B), c4=11 (4B) = 10
        // bits pattern: c4 c3 c2 c1 = 11 00 11 00 = 0xCC
        assert_eq!(lens[0xcc], 10);
    }

    #[cfg(any(
        all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    #[test]
    fn length_table_matches_sum_of_codes() {
        let lens = kernels::tables::length_table();
        for c in 0_usize..256 {
            let expected =
                (c & 0b11) + ((c >> 2) & 0b11) + ((c >> 4) & 0b11) + ((c >> 6) & 0b11) + 4;
            assert_eq!(lens[c] as usize, expected, "length table c=0x{c:02x}");
        }
    }

    #[cfg(any(
        all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    #[test]
    fn shuffle_table_consistent_with_length_table() {
        let table = kernels::tables::shuffle_table();
        let lens = kernels::tables::length_table();
        for c in 0_usize..256 {
            // Number of non-0xFF entries == data byte length.
            let nonzero = table[c].iter().filter(|&&b| b != 0xff).count();
            assert_eq!(nonzero, lens[c] as usize, "c=0x{c:02x}");
            // Source byte indices are 0, 1, …, len-1 in order.
            let mut expected = 0_u8;
            for &b in table[c].iter() {
                if b != 0xff {
                    assert_eq!(b, expected, "c=0x{c:02x}");
                    expected += 1;
                }
            }
        }
    }

    #[test]
    fn scalar_decode_matches_encoded_bit_pattern_for_canonical_inputs() {
        // Shape the inputs to hit each code exactly once per group.
        let values: Vec<u32> = (0..1024)
            .map(|i| (i as u32).wrapping_mul(0x9e37_79b9))
            .collect();
        round_trip(&values);
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn ssse3_decode_matches_scalar_when_available() {
        if !kernels::ssse3::is_available() {
            eprintln!("ssse3 unavailable; skipping inline SSSE3 parity test");
            return;
        }
        for n in [0_usize, 1, 4, 5, 8, 100, 1024] {
            for max in [1_u32, 2, 3, 4] {
                let values = deterministic_values(n, 0xABCD ^ (n as u64), max);
                let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
                let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
                let written = streamvbyte_encode_u32(&values, &mut ctrl, &mut data);

                let mut expected = vec![0_u32; n];
                kernels::scalar::decode_u32(&ctrl, &data[..written], n, &mut expected);

                let mut actual = vec![0_u32; n];
                // SAFETY: ssse3 is_available() returned true above.
                unsafe {
                    kernels::ssse3::decode_u32(&ctrl, &data[..written], n, &mut actual);
                }
                assert_eq!(actual, expected, "ssse3 diverged at n={n} max_bytes={max}");
            }
        }
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_decode_matches_scalar_when_available() {
        if !kernels::avx2::is_available() {
            eprintln!("avx2 unavailable; skipping inline AVX2 parity test");
            return;
        }
        for n in [0_usize, 1, 4, 5, 8, 9, 100, 1023, 1024] {
            for max in [1_u32, 2, 3, 4] {
                let values =
                    deterministic_values(n, 0x1234 ^ (n as u64) ^ ((max as u64) << 8), max);
                let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
                let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
                let written = streamvbyte_encode_u32(&values, &mut ctrl, &mut data);

                let mut expected = vec![0_u32; n];
                kernels::scalar::decode_u32(&ctrl, &data[..written], n, &mut expected);

                let mut actual = vec![0_u32; n];
                // SAFETY: avx2 is_available() returned true above.
                unsafe {
                    kernels::avx2::decode_u32(&ctrl, &data[..written], n, &mut actual);
                }
                assert_eq!(actual, expected, "avx2 diverged at n={n} max_bytes={max}");
            }
        }
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_decode_matches_scalar_when_available() {
        for n in [0_usize, 1, 4, 5, 8, 100, 1024] {
            for max in [1_u32, 2, 3, 4] {
                let values = deterministic_values(n, 0xBEEF ^ (n as u64), max);
                let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
                let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
                let written = streamvbyte_encode_u32(&values, &mut ctrl, &mut data);

                let mut expected = vec![0_u32; n];
                kernels::scalar::decode_u32(&ctrl, &data[..written], n, &mut expected);

                let mut actual = vec![0_u32; n];
                // SAFETY: NEON is mandatory on AArch64.
                unsafe {
                    kernels::neon::decode_u32(&ctrl, &data[..written], n, &mut actual);
                }
                assert_eq!(actual, expected, "neon diverged at n={n} max_bytes={max}");
            }
        }
    }

    #[test]
    fn dispatched_decode_matches_scalar_for_random_corpus() {
        // Independent of feature detection: the auto-dispatcher must
        // always agree with scalar. This is the contract every caller
        // depends on.
        for n in [0_usize, 1, 3, 4, 5, 100, 1024, 4097] {
            let values = deterministic_values(n, 0xCAFE ^ (n as u64), 4);
            let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
            let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
            let written = streamvbyte_encode_u32(&values, &mut ctrl, &mut data);

            let mut expected = vec![0_u32; n];
            kernels::scalar::decode_u32(&ctrl, &data[..written], n, &mut expected);

            let mut actual = vec![0_u32; n];
            let consumed = streamvbyte_decode_u32(&ctrl, &data[..written], n, &mut actual);
            assert_eq!(consumed, written, "dispatched offset diverged at n={n}");
            assert_eq!(actual, expected, "dispatched decode diverged at n={n}");
        }
    }
}

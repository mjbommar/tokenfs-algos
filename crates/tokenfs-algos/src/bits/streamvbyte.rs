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
//!
//! ## Tables
//!
//! The 4 KiB shuffle table and 256 B length table are pure functions of
//! the 256 possible control-byte values, so they are produced at compile
//! time by [`const fn`] initializers and stored as `static` rodata. This
//! keeps the SIMD kernels usable in `no_std` / kernel-mode builds (no
//! `std::sync::OnceLock` runtime initializer required) and removes the
//! first-use latency from the hot path.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_range_loop)]

/// Number of `u32` values packed per control byte.
const GROUP: usize = 4;

/// Failure modes for the fallible Stream-VByte codec entry points
/// ([`try_streamvbyte_encode_u32`] and [`try_streamvbyte_decode_u32`]).
///
/// Returned instead of panicking when a caller-supplied buffer is too
/// small for the requested element count, or when the data stream runs
/// out partway through a decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamvbyteError {
    /// `control` is too short for `n` integers
    /// (`< streamvbyte_control_len(n)`).
    ControlTooShort {
        /// Number of control bytes required for the requested `n`.
        needed: usize,
        /// Length of the caller-supplied control buffer.
        actual: usize,
    },
    /// `out` (or, for encode, `data_out`) is too short to hold the
    /// requested output.
    OutputTooShort {
        /// Number of output slots / bytes the operation needed.
        needed: usize,
        /// Length of the caller-supplied output buffer.
        actual: usize,
    },
    /// `data` runs out partway through decoding — the control byte
    /// implied more bytes than were available in the data stream.
    DataExhausted {
        /// Byte position within `data` at which the underflow was
        /// detected.
        position: usize,
    },
}

impl core::fmt::Display for StreamvbyteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ControlTooShort { needed, actual } => write!(
                f,
                "Stream-VByte control buffer too small: needed {needed}, got {actual}"
            ),
            Self::OutputTooShort { needed, actual } => write!(
                f,
                "Stream-VByte output buffer too small: needed {needed}, got {actual}"
            ),
            Self::DataExhausted { position } => write!(
                f,
                "Stream-VByte data stream exhausted at byte position {position}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for StreamvbyteError {}

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
/// is the return value. For a fallible variant that returns
/// [`StreamvbyteError`] instead of panicking, use
/// [`try_streamvbyte_encode_u32`].
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_streamvbyte_encode_u32`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn streamvbyte_encode_u32(
    values: &[u32],
    control_out: &mut [u8],
    data_out: &mut [u8],
) -> usize {
    kernels::auto::encode_u32(values, control_out, data_out)
}

/// Fallible variant of [`streamvbyte_encode_u32`] that returns
/// [`StreamvbyteError`] when the caller-supplied buffers are too small,
/// instead of panicking.
///
/// Returns the number of data bytes written on success.
///
/// This routine validates every length precondition upfront and then
/// dispatches to `_unchecked` kernels that contain no `assert!` /
/// panicking-index sites, so the call is panic-free even when the
/// `panicking-shape-apis` feature is disabled (audit-R6 finding #162).
pub fn try_streamvbyte_encode_u32(
    values: &[u32],
    control_out: &mut [u8],
    data_out: &mut [u8],
) -> Result<usize, StreamvbyteError> {
    let n = values.len();
    let ctrl_needed = streamvbyte_control_len(n);
    if control_out.len() < ctrl_needed {
        return Err(StreamvbyteError::ControlTooShort {
            needed: ctrl_needed,
            actual: control_out.len(),
        });
    }
    let data_needed = streamvbyte_data_max_len(n);
    if data_out.len() < data_needed {
        return Err(StreamvbyteError::OutputTooShort {
            needed: data_needed,
            actual: data_out.len(),
        });
    }
    // SAFETY: pre-validation above ensures both buffers meet the
    // _unchecked kernel preconditions (control_out big enough for the
    // ceil(n/4) control bytes, data_out big enough for the worst-case
    // 4*n + tail-padding bytes).
    Ok(unsafe { kernels::auto::encode_u32_unchecked(values, control_out, data_out) })
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
/// stream — when in doubt, size it via [`streamvbyte_data_max_len`]. For a
/// fallible variant that returns [`StreamvbyteError`] instead of
/// panicking, use [`try_streamvbyte_decode_u32`].
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_streamvbyte_decode_u32`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn streamvbyte_decode_u32(control: &[u8], data: &[u8], n: usize, out: &mut [u32]) -> usize {
    kernels::auto::decode_u32(control, data, n, out)
}

/// Fallible variant of [`streamvbyte_decode_u32`] that returns
/// [`StreamvbyteError`] when the caller-supplied buffers are too small or
/// when the data stream runs out partway through, instead of panicking.
///
/// Returns the number of data bytes consumed on success.
///
/// Validates `control.len()`, `out.len()`, and walks the control stream
/// to bound the implied data length against `data.len()` before
/// dispatching to the `_unchecked` kernel. The kernels themselves no
/// longer contain `assert!` / panicking-index sites, so this routine is
/// panic-free even when the `panicking-shape-apis` feature is disabled
/// (audit-R6 finding #162).
pub fn try_streamvbyte_decode_u32(
    control: &[u8],
    data: &[u8],
    n: usize,
    out: &mut [u32],
) -> Result<usize, StreamvbyteError> {
    let ctrl_needed = streamvbyte_control_len(n);
    if control.len() < ctrl_needed {
        return Err(StreamvbyteError::ControlTooShort {
            needed: ctrl_needed,
            actual: control.len(),
        });
    }
    if out.len() < n {
        return Err(StreamvbyteError::OutputTooShort {
            needed: n,
            actual: out.len(),
        });
    }
    // Walk the control stream to verify the implied data length fits
    // before invoking the kernel. This matches the kernel's
    // own data-byte accounting (full groups + padded tail).
    let full_groups = n / GROUP;
    let mut implied_data: usize = 0;
    for g in 0..full_groups {
        let c = control[g];
        for k in 0..GROUP {
            let code = (c >> (2 * k)) & 0b11;
            implied_data += (code as usize) + 1;
            if implied_data > data.len() {
                return Err(StreamvbyteError::DataExhausted {
                    position: data.len(),
                });
            }
        }
    }
    let tail = n - full_groups * GROUP;
    if tail > 0 {
        let c = control[full_groups];
        for k in 0..GROUP {
            let code = (c >> (2 * k)) & 0b11;
            implied_data += (code as usize) + 1;
            if implied_data > data.len() {
                return Err(StreamvbyteError::DataExhausted {
                    position: data.len(),
                });
            }
        }
    }
    // SAFETY: pre-validation above ensures `control.len() >= ceil(n/4)`,
    // `out.len() >= n`, and the implied data length fits inside `data`.
    Ok(unsafe { kernels::auto::decode_u32_unchecked(control, data, n, out) })
}

/// Pinned Stream-VByte kernels.
pub mod kernels {
    /// Runtime-dispatched Stream-VByte kernels.
    pub mod auto {
        /// Runtime-dispatched encode (panicking variant).
        ///
        /// Asserts caller-supplied buffer lengths before dispatching to
        /// the `_unchecked` kernel.
        ///
        /// # Panics
        ///
        /// Panics if `control_out.len() < ceil(values.len()/4)` or
        /// `data_out.len() < streamvbyte_data_max_len(values.len())`.
        pub fn encode_u32(values: &[u32], control_out: &mut [u8], data_out: &mut [u8]) -> usize {
            // Encode is bandwidth-modest; the scalar path already runs at
            // near memory speed. SIMD-encode wins are small and the spec
            // explicitly defers them.
            super::scalar::encode_u32(values, control_out, data_out)
        }

        /// Runtime-dispatched encode kernel without bounds-checking
        /// asserts.
        ///
        /// # Safety
        ///
        /// Caller must ensure
        /// `control_out.len() >= streamvbyte_control_len(values.len())`
        /// and
        /// `data_out.len() >= streamvbyte_data_max_len(values.len())`.
        /// Used by [`super::super::try_streamvbyte_encode_u32`] after
        /// pre-validation; eliminates the `assert!` panic sites that
        /// would otherwise leak through the fallible API surface
        /// (audit-R6 finding #162).
        pub unsafe fn encode_u32_unchecked(
            values: &[u32],
            control_out: &mut [u8],
            data_out: &mut [u8],
        ) -> usize {
            // SAFETY: caller upholds the buffer-length precondition.
            unsafe { super::scalar::encode_u32_unchecked(values, control_out, data_out) }
        }

        /// Runtime-dispatched decode (panicking variant).
        ///
        /// Asserts caller-supplied buffer lengths before dispatching to
        /// the `_unchecked` kernel.
        ///
        /// # Panics
        ///
        /// Panics if `control.len() < ceil(n/4)`, `out.len() < n`, or
        /// `data` runs out partway through the decode.
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

        /// Runtime-dispatched decode kernel without bounds-checking
        /// asserts.
        ///
        /// # Safety
        ///
        /// Caller must ensure `control.len() >= ceil(n/4)`,
        /// `out.len() >= n`, and `data.len()` is at least the implied
        /// length encoded in the control stream (sum of `code+1` over
        /// each 2-bit code in the first `ceil(n/4)` control bytes).
        /// Used by [`super::super::try_streamvbyte_decode_u32`] after
        /// pre-validation; eliminates the `assert!` panic sites that
        /// would otherwise leak through the fallible API surface
        /// (audit-R6 finding #162).
        pub unsafe fn decode_u32_unchecked(
            control: &[u8],
            data: &[u8],
            n: usize,
            out: &mut [u32],
        ) -> usize {
            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability checked above; caller upholds
                    // the buffer-length precondition.
                    return unsafe { super::avx2::decode_u32_unchecked(control, data, n, out) };
                }
                if super::ssse3::is_available() {
                    // SAFETY: availability checked above; caller upholds
                    // the buffer-length precondition.
                    return unsafe { super::ssse3::decode_u32_unchecked(control, data, n, out) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64; caller
                    // upholds the buffer-length precondition.
                    return unsafe { super::neon::decode_u32_unchecked(control, data, n, out) };
                }
            }

            // SAFETY: caller upholds the buffer-length precondition.
            unsafe { super::scalar::decode_u32_unchecked(control, data, n, out) }
        }
    }

    /// Portable scalar reference oracle.
    ///
    /// Byte-by-byte encode and decode following the Lemire/Kurz spec
    /// directly. SIMD backends must match this bit-exactly.
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    /// Per-control-byte length and shuffle tables for the SIMD decoders.
    ///
    /// Both tables are pure functions of the control byte. They are
    /// produced at compile time via `const fn` initializers and live in
    /// `static` rodata, so this module compiles unchanged in `no_std`
    /// kernel-mode builds (no `OnceLock` / runtime initializer required)
    /// and the first SIMD decode pays no synchronization or table-build
    /// cost.
    ///
    /// The shuffle table follows the Lemire C reference: index `0xFF`
    /// zeros the destination lane (both `_mm_shuffle_epi8` and
    /// `vqtbl1q_u8` clear bytes for indices with bit 7 set / >= 16).
    #[cfg(any(
        all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")),
        all(feature = "neon", target_arch = "aarch64")
    ))]
    pub(super) mod tables {
        /// Shuffle table: 256 entries of `[u8; 16]`. 4 KiB total.
        ///
        /// `SHUFFLE_TABLE[c]` is the 16-byte permutation that turns a
        /// 16-byte data window into four little-endian `u32` lanes; lanes
        /// past the encoded width hold `0xFF` so the SIMD shuffle zeros
        /// them.
        pub(crate) static SHUFFLE_TABLE: [[u8; 16]; 256] = build_shuffle_table();

        /// Length table: 256 entries of `u8`. 256 B total.
        ///
        /// `LENGTH_TABLE[c]` is the number of data bytes the control byte
        /// `c` consumes — the sum of `(code_i + 1)` over the four codes.
        /// Range is 4..=16.
        pub(crate) static LENGTH_TABLE: [u8; 256] = build_length_table();

        /// Returns a reference to the static shuffle table.
        ///
        /// Kept as a function (vs. exposing `SHUFFLE_TABLE` directly) so
        /// callers do not have to spell the array type and so the SIMD
        /// kernels read from the same accessor in every build.
        #[inline]
        pub(crate) fn shuffle_table() -> &'static [[u8; 16]; 256] {
            &SHUFFLE_TABLE
        }

        /// Returns a reference to the static length table.
        ///
        /// See [`shuffle_table`] for the rationale behind the accessor.
        #[inline]
        pub(crate) fn length_table() -> &'static [u8; 256] {
            &LENGTH_TABLE
        }

        /// Builds the shuffle table at compile time.
        ///
        /// For each control byte `c`, decode the four 2-bit codes
        /// `c1, c2, c3, c4` (low→high). Each integer takes `code_i + 1`
        /// data bytes. The output is four little-endian u32s in 16
        /// contiguous bytes; we fill the first `code_i + 1` of each
        /// 4-byte lane with consecutive source byte indices, and the
        /// remaining lanes with `0xFF` so PSHUFB / TBL writes zero.
        const fn build_shuffle_table() -> [[u8; 16]; 256] {
            let mut table = [[0xff_u8; 16]; 256];
            let mut c = 0_usize;
            while c < 256 {
                let codes = [
                    (c & 0b11) as u8,
                    ((c >> 2) & 0b11) as u8,
                    ((c >> 4) & 0b11) as u8,
                    ((c >> 6) & 0b11) as u8,
                ];
                let mut shuffle = [0xff_u8; 16];
                let mut src = 0_u8;
                let mut lane = 0_usize;
                while lane < 4 {
                    let len = codes[lane] + 1;
                    let mut byte = 0_u8;
                    while byte < len {
                        shuffle[lane * 4 + byte as usize] = src;
                        src += 1;
                        byte += 1;
                    }
                    // Lane bytes [len..4] stay 0xFF → zeroed by shuffle.
                    lane += 1;
                }
                table[c] = shuffle;
                c += 1;
            }
            table
        }

        /// Builds the length table at compile time.
        const fn build_length_table() -> [u8; 256] {
            let mut table = [0_u8; 256];
            let mut c = 0_usize;
            while c < 256 {
                let l = (c & 0b11) + ((c >> 2) & 0b11) + ((c >> 4) & 0b11) + ((c >> 6) & 0b11) + 4;
                table[c] = l as u8;
                c += 1;
            }
            table
        }
    }

    /// SSSE3 + SSE4.1 decode kernel.
    ///
    /// One PSHUFB per control byte produces four little-endian `u32`s
    /// from a 16-byte data window. Gated on the crate's `avx2` feature
    /// (which is the umbrella for runtime-detected SIMD x86 backends);
    /// runtime check is for SSSE3 only. Per-backend visibility gated
    /// on `arch-pinned-kernels` per the v0.4.2 #180 + audit-R10 T1.3
    /// pattern.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod ssse3;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod ssse3;

    /// AVX2 dual-pumped decode kernel.
    ///
    /// Each iteration consumes two control bytes and stores eight
    /// little-endian `u32`s. The two halves are decoded into a `__m128i`
    /// each via PSHUFB (per-lane on AVX2 the equivalent is two PSHUFBs);
    /// stitched together via `_mm256_inserti128_si256`. Falls back to the
    /// SSSE3 inner loop for the trailing odd group, then to scalar for
    /// the unsafe tail.
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

    /// AArch64 NEON decode kernel.
    ///
    /// `vqtbl1q_u8` performs the same 16-byte permute as `_mm_shuffle_epi8`;
    /// indices outside `[0, 15]` produce zero, so the same `0xFF` sentinel
    /// works on both ISAs and the shuffle table is shared between them.
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

    #[cfg(all(feature = "alloc", not(feature = "std")))]
    extern crate alloc;
    use super::*;
    // The `vec!` macro and `Vec` type are not in the no-std prelude;
    // alias them from `alloc` for the alloc-only build (audit-R6 #164).
    // Both helpers below use `Vec` regardless of `panicking-shape-apis`
    // (the `try_*` round-trip in particular).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

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

    // The panicking `streamvbyte_encode_u32` / `streamvbyte_decode_u32`
    // entry points only exist when the on-by-default
    // `panicking-shape-apis` feature is enabled (audit-R5 #157). The
    // round-trip helpers below call them directly, so gate the whole
    // family on that feature; the fallible `try_*` siblings are exercised
    // by their own tests further down (audit-R6 finding #164).
    #[cfg(feature = "panicking-shape-apis")]
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

    #[cfg(feature = "panicking-shape-apis")]
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

    #[cfg(feature = "panicking-shape-apis")]
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

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn round_trip_each_code_specifically() {
        // Mix one of each code in a single group; verify byte budget and
        // value recovery.
        let values: Vec<u32> = vec![0x00, 0xff, 0x1234, 0x00ab_cdef, 0x1234_5678];
        round_trip(&values);
    }

    #[cfg(feature = "panicking-shape-apis")]
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

    #[cfg(feature = "panicking-shape-apis")]
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

    #[cfg(feature = "panicking-shape-apis")]
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
        all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")),
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
        all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")),
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
        all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")),
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
        all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")),
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

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn scalar_decode_matches_encoded_bit_pattern_for_canonical_inputs() {
        // Shape the inputs to hit each code exactly once per group.
        let values: Vec<u32> = (0..1024)
            .map(|i| (i as u32).wrapping_mul(0x9e37_79b9))
            .collect();
        round_trip(&values);
    }

    #[cfg(all(
        feature = "panicking-shape-apis",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
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

    #[cfg(all(
        feature = "panicking-shape-apis",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
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

    #[cfg(all(
        feature = "panicking-shape-apis",
        feature = "neon",
        target_arch = "aarch64"
    ))]
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

    #[cfg(feature = "panicking-shape-apis")]
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

    #[test]
    fn try_encode_returns_err_on_undersized_control() {
        let values = vec![1_u32, 2, 3, 4, 5];
        let mut ctrl = vec![0_u8; 0]; // need 2 bytes for n=5.
        let mut data = vec![0_u8; streamvbyte_data_max_len(values.len())];
        let err = try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data).unwrap_err();
        assert_eq!(
            err,
            StreamvbyteError::ControlTooShort {
                needed: 2,
                actual: 0
            }
        );
    }

    #[test]
    fn try_encode_returns_err_on_undersized_data() {
        let values = vec![0xff_ff_ff_ff_u32; 4]; // 4-byte code each.
        let mut ctrl = vec![0_u8; streamvbyte_control_len(values.len())];
        let mut data = vec![0_u8; 1];
        let err = try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data).unwrap_err();
        // Worst case is 4 bytes per value -> 16 bytes; max_len for n=4
        // is 16.
        assert_eq!(
            err,
            StreamvbyteError::OutputTooShort {
                needed: streamvbyte_data_max_len(4),
                actual: 1
            }
        );
    }

    #[test]
    fn try_encode_returns_ok_with_byte_count_on_valid_inputs() {
        let values = vec![0_u32, 1, 2, 3];
        let mut ctrl = vec![0_u8; streamvbyte_control_len(values.len())];
        let mut data = vec![0_u8; streamvbyte_data_max_len(values.len())];
        let written = try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data).unwrap();
        // Bytes themselves are 1 each → total 4.
        assert_eq!(written, 4);
        assert_eq!(ctrl[0], 0x00);
        assert_eq!(&data[..4], &[0, 1, 2, 3]);
    }

    #[test]
    fn try_decode_returns_err_on_undersized_control() {
        let mut out = vec![0_u32; 5];
        let ctrl = vec![0_u8; 1]; // need 2 for n=5.
        let data = vec![0_u8; 32];
        let err = try_streamvbyte_decode_u32(&ctrl, &data, 5, &mut out).unwrap_err();
        assert_eq!(
            err,
            StreamvbyteError::ControlTooShort {
                needed: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn try_decode_returns_err_on_undersized_output() {
        let ctrl = vec![0_u8; 2];
        let data = vec![0_u8; 16];
        let mut out = vec![0_u32; 1];
        let err = try_streamvbyte_decode_u32(&ctrl, &data, 5, &mut out).unwrap_err();
        assert_eq!(
            err,
            StreamvbyteError::OutputTooShort {
                needed: 5,
                actual: 1
            }
        );
    }

    // The next two tests exercise the fallible `try_*` decode error paths
    // but rely on the panicking `streamvbyte_encode_u32` to materialise
    // the bit pattern they probe; gate them on `panicking-shape-apis`.
    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_decode_returns_err_when_data_is_exhausted() {
        // Encode 4 four-byte values -> 1 control byte (0xff) + 16 data
        // bytes; truncate the data to 8 bytes so the decode would read
        // past the end.
        let values = vec![0xff_ff_ff_ff_u32; 4];
        let mut ctrl = vec![0_u8; streamvbyte_control_len(values.len())];
        let mut data = vec![0_u8; streamvbyte_data_max_len(values.len())];
        try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data).unwrap();
        let truncated = &data[..8];
        let mut out = vec![0_u32; 4];
        let err = try_streamvbyte_decode_u32(&ctrl, truncated, 4, &mut out).unwrap_err();
        assert!(matches!(err, StreamvbyteError::DataExhausted { .. }));
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_decode_returns_ok_and_matches_scalar_on_valid_inputs() {
        for n in [0_usize, 1, 3, 4, 5, 100, 1024] {
            let values = deterministic_values(n, 0xC0DE ^ (n as u64), 4);
            let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
            let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
            let written = try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data).unwrap();
            let mut out = vec![0_u32; n];
            let consumed =
                try_streamvbyte_decode_u32(&ctrl, &data[..written], n, &mut out).unwrap();
            assert_eq!(consumed, written);
            assert_eq!(out, values);
        }
    }

    // The "still panics" checks asset that the panicking variant retains
    // its panicking behaviour; they're meaningless when the panicking API
    // is compiled out.
    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "control_out too small")]
    fn encode_still_panics_on_undersized_control() {
        let values = vec![1_u32, 2, 3, 4, 5];
        let mut ctrl = vec![0_u8; 0];
        let mut data = vec![0_u8; streamvbyte_data_max_len(values.len())];
        streamvbyte_encode_u32(&values, &mut ctrl, &mut data);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "decode output buffer too small")]
    fn decode_still_panics_on_undersized_output() {
        let ctrl = vec![0_u8; 2];
        let data = vec![0_u8; 16];
        let mut out = vec![0_u32; 1];
        streamvbyte_decode_u32(&ctrl, &data, 5, &mut out);
    }

    // ------------------------------------------------------------------
    // Audit-R6 finding #162 regression tests for the `try_*` paths.
    //
    // These tests intentionally avoid the panicking entry points so the
    // assertion that `try_streamvbyte_decode_u32` is panic-free still
    // holds when the `panicking-shape-apis` Cargo feature is disabled
    // (the kernel/FUSE deployment build).
    // ------------------------------------------------------------------

    /// Helper: encode via the fallible API and assert success. Used by
    /// the `try_*` regression tests below so they compile without the
    /// `panicking-shape-apis` feature.
    fn try_encode_or_panic(values: &[u32]) -> (Vec<u8>, Vec<u8>, usize) {
        let n = values.len();
        let mut ctrl = vec![0_u8; streamvbyte_control_len(n)];
        let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
        let written =
            try_streamvbyte_encode_u32(values, &mut ctrl, &mut data).expect("encode succeeded");
        (ctrl, data, written)
    }

    #[test]
    fn try_decode_data_too_short_for_implied_length_returns_err_not_panic() {
        // Control byte 0xFF implies 4 four-byte values -> 16 data bytes.
        // Provide only 7 data bytes so the third value's 4-byte read
        // would exceed the buffer.
        let ctrl = [0xff_u8];
        let data = [0_u8; 7];
        let mut out = [0_u32; 4];
        let err =
            try_streamvbyte_decode_u32(&ctrl, &data, 4, &mut out).expect_err("must return Err");
        assert!(
            matches!(err, StreamvbyteError::DataExhausted { .. }),
            "expected DataExhausted, got {err:?}"
        );
    }

    #[test]
    fn try_decode_undersized_output_returns_err_not_panic() {
        // n=4 but out has length 0.
        let ctrl = [0x00_u8];
        let data = [0_u8; 4];
        let mut out: [u32; 0] = [];
        let err =
            try_streamvbyte_decode_u32(&ctrl, &data, 4, &mut out).expect_err("must return Err");
        assert_eq!(
            err,
            StreamvbyteError::OutputTooShort {
                needed: 4,
                actual: 0
            }
        );
    }

    #[test]
    fn try_decode_undersized_control_returns_err_not_panic() {
        // n=5 needs ceil(5/4) = 2 control bytes; supply only 1.
        let ctrl = [0x00_u8; 1];
        let data = [0_u8; 32];
        let mut out = [0_u32; 5];
        let err =
            try_streamvbyte_decode_u32(&ctrl, &data, 5, &mut out).expect_err("must return Err");
        assert_eq!(
            err,
            StreamvbyteError::ControlTooShort {
                needed: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn try_decode_tail_padding_data_underflow_returns_err_not_panic() {
        // n=5 -> 1 full group + tail of 1; the encoder pads the
        // remaining 3 tail slots with 1-byte zero codes. If the data
        // is one byte short of even the implied tail, the try_ path
        // must surface DataExhausted rather than panic in the kernel.
        let values = vec![1_u32, 2, 3, 4, 5];
        let (ctrl, data, written) = try_encode_or_panic(&values);
        // Drop one byte off the end of the encoded data.
        let truncated = &data[..written - 1];
        let mut out = vec![0_u32; values.len()];
        let err = try_streamvbyte_decode_u32(&ctrl, truncated, values.len(), &mut out)
            .expect_err("must return Err");
        assert!(
            matches!(err, StreamvbyteError::DataExhausted { .. }),
            "expected DataExhausted, got {err:?}"
        );
    }

    #[test]
    fn try_encode_undersized_data_returns_err_not_panic() {
        let values = vec![0xff_ff_ff_ff_u32; 4];
        let mut ctrl = vec![0_u8; streamvbyte_control_len(values.len())];
        let mut data = vec![0_u8; 1]; // far too small.
        let err =
            try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data).expect_err("must return Err");
        assert!(
            matches!(err, StreamvbyteError::OutputTooShort { .. }),
            "expected OutputTooShort, got {err:?}"
        );
    }

    #[test]
    fn try_round_trip_via_fallible_apis_only() {
        // Exercise the full fallible round-trip across (full groups +
        // tail) shapes without ever invoking the panicking variants.
        for n in [0_usize, 1, 3, 4, 5, 7, 8, 100, 1024] {
            let values = deterministic_values(n, 0xF00D ^ (n as u64), 4);
            let (ctrl, data, written) = try_encode_or_panic(&values);
            let mut out = vec![0_u32; n];
            let consumed = try_streamvbyte_decode_u32(&ctrl, &data[..written], n, &mut out)
                .expect("decode succeeded");
            assert_eq!(consumed, written);
            assert_eq!(out, values, "fallible round-trip diverged at n={n}");
        }
    }
}

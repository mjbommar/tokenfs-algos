//! AVX-512BW UTF-8 validator returning a [`Utf8Validation`] compatible with
//! the scalar reference implementation in [`super`].
//!
//! # Algorithm
//!
//! Same Keiser-Lemire 3-pshufb DFA as [`super::utf8_avx2`], scaled to
//! 64-byte blocks via `__m512i`. Each per-byte error predicate is computed
//! lane-wise using `_mm512_shuffle_epi8` lookups; the cross-block "previous
//! N" shift uses `_mm512_permutex2var_epi32` to rotate the four 128-bit
//! lanes by one dword followed by a per-lane `_mm512_alignr_epi8`. This is
//! the same trick used by simdutf's icelake backend (see
//! `_references/simdutf/src/icelake/icelake_common.inl.cpp`).
//!
//! Unlike the AVX2 path, which processes a 64-byte chunk as two 32-byte
//! lanes, the AVX-512 path processes one 64-byte block per iteration and
//! relies on natural `__mmask64` returns from the byte compare intrinsics.
//!
//! # Safety contract
//!
//! [`validate_utf8`] is `unsafe` and tagged
//! `#[target_feature(enable = "avx512bw")]`. The caller MUST verify
//! `avx512bw` is available on the running CPU (e.g. via
//! `std::is_x86_feature_detected!("avx512bw")`) before invoking it. Calling
//! on a non-AVX-512BW CPU is undefined behavior.
//!
//! # Trade-off: vectorized check, scalar diagnosis
//!
//! Same approach as the AVX2 path: SIMD answers "is there an error in this
//! stream?". On the first failing 64-byte block we re-validate from a safe
//! re-entry point with `core::str::from_utf8` so `valid_up_to` /
//! `error_len` exactly match `from_utf8`.
//!
//! # Provenance
//!
//! Algorithm and tables follow `rusticstuff/simdutf8` (the AVX2 port lives
//! in [`super::utf8_avx2`]) and `simdutf` icelake (`__m512i` shape, the
//! `prev<N>` permute trick). Both upstreams are MIT OR Apache-2.0; this
//! file inherits the same terms by re-derivation.

use super::Utf8Validation;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m512i, _mm512_alignr_epi8, _mm512_and_si512, _mm512_loadu_si512, _mm512_or_si512,
    _mm512_permutex2var_epi32, _mm512_set_epi8, _mm512_set_epi32, _mm512_set1_epi8,
    _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_srli_epi16, _mm512_subs_epu8,
    _mm512_test_epi8_mask, _mm512_xor_si512,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m512i, _mm512_alignr_epi8, _mm512_and_si512, _mm512_loadu_si512, _mm512_or_si512,
    _mm512_permutex2var_epi32, _mm512_set_epi8, _mm512_set_epi32, _mm512_set1_epi8,
    _mm512_setzero_si512, _mm512_shuffle_epi8, _mm512_srli_epi16, _mm512_subs_epu8,
    _mm512_test_epi8_mask, _mm512_xor_si512,
};

/// 64 bytes per iteration: one full AVX-512 register.
const SIMD_CHUNK_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// SIMD primitive helpers
// ---------------------------------------------------------------------------

/// Loads 64 bytes (unaligned) into an AVX-512 register.
///
/// SAFETY: precondition — AVX-512BW available; `ptr..ptr+64` readable.
#[target_feature(enable = "avx512bw")]
#[inline]
unsafe fn load64(ptr: *const u8) -> __m512i {
    // SAFETY: caller upholds the precondition above.
    unsafe { _mm512_loadu_si512(ptr.cast::<__m512i>()) }
}

/// Splat `val` into all 64 byte lanes.
///
/// SAFETY: precondition — AVX-512BW available.
#[target_feature(enable = "avx512bw")]
#[inline]
#[allow(clippy::cast_possible_wrap)]
unsafe fn splat(val: u8) -> __m512i {
    _mm512_set1_epi8(val as i8)
}

/// Per-byte right-shift by 4 (high nibble extraction). Mirrors `shr4` in
/// the AVX2 port.
///
/// SAFETY: precondition — AVX-512BW available.
#[target_feature(enable = "avx512bw")]
#[inline]
unsafe fn shr4(v: __m512i) -> __m512i {
    // SAFETY: AVX-512BW enabled by target_feature.
    unsafe { _mm512_and_si512(_mm512_srli_epi16(v, 4), splat(0x0F)) }
}

/// 16-entry lookup table replicated across all four 128-bit halves,
/// indexed by the low nibble of each byte of `v`. This is the `pshufb`
/// table dispatch.
///
/// SAFETY: precondition — AVX-512BW available.
#[target_feature(enable = "avx512bw")]
#[inline]
#[allow(clippy::cast_possible_wrap, clippy::too_many_arguments)]
unsafe fn lookup16(
    v: __m512i,
    t0: u8,
    t1: u8,
    t2: u8,
    t3: u8,
    t4: u8,
    t5: u8,
    t6: u8,
    t7: u8,
    t8: u8,
    t9: u8,
    ta: u8,
    tb: u8,
    tc: u8,
    td: u8,
    te: u8,
    tf: u8,
) -> __m512i {
    // `_mm512_set_epi8` lays out high-to-low; replicate the same 16-entry
    // table four times so each 128-bit lane has the full table for the
    // per-lane `_mm512_shuffle_epi8`.
    let table = _mm512_set_epi8(
        tf as i8, te as i8, td as i8, tc as i8, tb as i8, ta as i8, t9 as i8, t8 as i8, t7 as i8,
        t6 as i8, t5 as i8, t4 as i8, t3 as i8, t2 as i8, t1 as i8, t0 as i8, tf as i8, te as i8,
        td as i8, tc as i8, tb as i8, ta as i8, t9 as i8, t8 as i8, t7 as i8, t6 as i8, t5 as i8,
        t4 as i8, t3 as i8, t2 as i8, t1 as i8, t0 as i8, tf as i8, te as i8, td as i8, tc as i8,
        tb as i8, ta as i8, t9 as i8, t8 as i8, t7 as i8, t6 as i8, t5 as i8, t4 as i8, t3 as i8,
        t2 as i8, t1 as i8, t0 as i8, tf as i8, te as i8, td as i8, tc as i8, tb as i8, ta as i8,
        t9 as i8, t8 as i8, t7 as i8, t6 as i8, t5 as i8, t4 as i8, t3 as i8, t2 as i8, t1 as i8,
        t0 as i8,
    );
    _mm512_shuffle_epi8(table, v)
}

/// Cross-block "previous N" — returns a vector whose byte `i` is
/// `input[i-N]` with the first `N` bytes coming from `prev_block`. Mirrors
/// `prev<N>` in `simdutf/src/icelake/icelake_common.inl.cpp`.
///
/// The 16-element index permutes the four 128-bit lanes such that lane `j`
/// of the rotated vector contains lane `j-1` of the input (with lane -1
/// coming from `prev_block`). Then a per-lane `palignr` finishes the
/// 1-/2-/3-byte rotation. `IMM` must equal `16 - N`.
///
/// SAFETY: precondition — AVX-512BW available; `IMM` in 13..=15.
#[target_feature(enable = "avx512bw")]
#[inline]
#[allow(clippy::cast_possible_wrap)]
unsafe fn prev<const IMM: i32>(input: __m512i, prev_block: __m512i) -> __m512i {
    // Indices 28..31 select dwords 12..15 from `prev_block` (its high
    // 128-bit lane); indices 0..11 select dwords 0..11 from `input` (its
    // low three 128-bit lanes). Result: high 16 bytes of `prev_block`
    // followed by low 48 bytes of `input`. Per-lane `alignr` then slides
    // each 128-bit lane left by `16 - IMM` bytes against `input`.
    let idx = _mm512_set_epi32(11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 31, 30, 29, 28);
    let rotated = _mm512_permutex2var_epi32(input, idx, prev_block);
    _mm512_alignr_epi8::<IMM>(input, rotated)
}

// ---------------------------------------------------------------------------
// DFA tables — verbatim from `algorithm.rs::check_special_cases` (same as
// the AVX2 port; bit assignments are load-bearing, do not renumber).
// ---------------------------------------------------------------------------

const TOO_SHORT: u8 = 1 << 0;
const TOO_LONG: u8 = 1 << 1;
const OVERLONG_3: u8 = 1 << 2;
const TOO_LARGE: u8 = 1 << 3;
const SURROGATE: u8 = 1 << 4;
const OVERLONG_2: u8 = 1 << 5;
const TOO_LARGE_1000: u8 = 1 << 6;
const OVERLONG_4: u8 = 1 << 6;
const TWO_CONTS: u8 = 1 << 7;
const CARRY: u8 = TOO_SHORT | TOO_LONG | TWO_CONTS;

/// Per-block error predicate. Returns a vector whose byte `i` is non-zero
/// iff the pair `(prev1[i], input[i])` violates UTF-8.
///
/// SAFETY: precondition — AVX-512BW available.
#[target_feature(enable = "avx512bw")]
#[inline]
unsafe fn check_special_cases(input: __m512i, prev1: __m512i) -> __m512i {
    // SAFETY: AVX-512BW enabled by target_feature; helpers preserve it.
    unsafe {
        let byte_1_high = lookup16(
            shr4(prev1),
            TOO_LONG,
            TOO_LONG,
            TOO_LONG,
            TOO_LONG,
            TOO_LONG,
            TOO_LONG,
            TOO_LONG,
            TOO_LONG,
            TWO_CONTS,
            TWO_CONTS,
            TWO_CONTS,
            TWO_CONTS,
            TOO_SHORT | OVERLONG_2,
            TOO_SHORT,
            TOO_SHORT | OVERLONG_3 | SURROGATE,
            TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4,
        );
        let byte_1_low = lookup16(
            _mm512_and_si512(prev1, splat(0x0F)),
            CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
            CARRY | OVERLONG_2,
            CARRY,
            CARRY,
            CARRY | TOO_LARGE,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
            CARRY | TOO_LARGE | TOO_LARGE_1000,
        );
        let byte_2_high = lookup16(
            shr4(input),
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
            TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
            TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
            TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
            TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
            TOO_SHORT,
        );
        _mm512_and_si512(_mm512_and_si512(byte_1_high, byte_1_low), byte_2_high)
    }
}

/// "Must-be-continuation" check: bytes ≥ 0xE0 require a 3rd cont; bytes ≥
/// 0xF0 require a 4th. XOR with `special_cases` so non-continuations show
/// up as errors. Mirrors `check_multibyte_lengths` from the AVX2 port.
///
/// SAFETY: precondition — AVX-512BW available.
#[target_feature(enable = "avx512bw")]
#[inline]
unsafe fn check_multibyte_lengths(
    input: __m512i,
    prev_block: __m512i,
    special_cases: __m512i,
) -> __m512i {
    // SAFETY: AVX-512BW enabled by target_feature; 16 - 2 = 14, 16 - 3 = 13.
    unsafe {
        let prev2 = prev::<14>(input, prev_block);
        let prev3 = prev::<13>(input, prev_block);
        let is_third = _mm512_subs_epu8(prev2, splat(0xE0 - 0x80));
        let is_fourth = _mm512_subs_epu8(prev3, splat(0xF0 - 0x80));
        let must23 = _mm512_or_si512(is_third, is_fourth);
        let must23_80 = _mm512_and_si512(must23, splat(0x80));
        _mm512_xor_si512(must23_80, special_cases)
    }
}

/// Builds the "incomplete trailing sequence" mask for the last 64-byte
/// block. Mirrors `is_incomplete` from simdutf icelake. The terminating
/// pattern keeps the final three bytes lowered to detect 2/3/4-byte
/// starters in the last 1/2/3 lanes that lack their continuations.
///
/// SAFETY: precondition — AVX-512BW available.
#[target_feature(enable = "avx512bw")]
#[inline]
#[allow(clippy::cast_possible_wrap)]
unsafe fn is_incomplete(input: __m512i) -> __m512i {
    let max = _mm512_set_epi8(
        (0b1100_0000 - 1) as i8,
        (0b1110_0000 - 1) as i8,
        (0b1111_0000 - 1) as i8,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    );
    _mm512_subs_epu8(input, max)
}

/// Per-block accumulator state mirroring `Utf8CheckAlgorithm`.
struct State {
    prev: __m512i,
    incomplete: __m512i,
    error: __m512i,
}

impl State {
    /// SAFETY: precondition — AVX-512BW available.
    #[target_feature(enable = "avx512bw")]
    #[inline]
    unsafe fn new() -> Self {
        let z = _mm512_setzero_si512();
        Self {
            prev: z,
            incomplete: z,
            error: z,
        }
    }

    /// Process one 64-byte block.
    ///
    /// SAFETY: precondition — AVX-512BW available.
    #[target_feature(enable = "avx512bw")]
    #[inline]
    unsafe fn check_block(&mut self, input: __m512i) {
        // SAFETY: AVX-512BW enabled by target_feature; 16 - 1 = 15 is the
        // const generic argument.
        unsafe {
            let p1 = prev::<15>(input, self.prev);
            let sc = check_special_cases(input, p1);
            let lengths = check_multibyte_lengths(input, self.prev, sc);
            self.error = _mm512_or_si512(self.error, lengths);
            self.incomplete = is_incomplete(input);
            self.prev = input;
        }
    }

    /// Fold pending incomplete-sequence state into the error vector.
    ///
    /// SAFETY: precondition — AVX-512BW available.
    #[target_feature(enable = "avx512bw")]
    #[inline]
    unsafe fn check_incomplete_pending(&mut self) {
        self.error = _mm512_or_si512(self.error, self.incomplete);
    }

    /// True iff any error bit was ever set.
    ///
    /// SAFETY: precondition — AVX-512BW available.
    #[target_feature(enable = "avx512bw")]
    #[inline]
    unsafe fn has_error(&self) -> bool {
        // `_mm512_test_epi8_mask` returns the per-byte non-zero predicate
        // as a 64-bit mask; non-zero ⇒ at least one error bit was set.
        _mm512_test_epi8_mask(self.error, self.error) != 0
    }
}

/// Scalar fallback that converts a `core::str::from_utf8` outcome on
/// `bytes[offset..]` into a [`Utf8Validation`] expressed in absolute
/// offsets. Mirrors `helpers::validate_utf8_at_offset`.
#[inline]
fn scalar_diagnose(bytes: &[u8], offset: usize) -> Utf8Validation {
    let tail = &bytes[offset..];
    match core::str::from_utf8(tail) {
        Ok(_) => Utf8Validation {
            valid: true,
            valid_up_to: bytes.len(),
            error_len: 0,
        },
        Err(err) => Utf8Validation {
            valid: false,
            valid_up_to: offset + err.valid_up_to(),
            error_len: err.error_len().unwrap_or(0) as u8,
        },
    }
}

/// Pick a safe scalar re-entry point near a failing 64-byte block. Mirrors
/// the AVX2 port's `safe_reentry`: walk back up to three bytes from the
/// start of the failing block until we find a non-continuation. That byte
/// is the earliest possible start of a sequence whose tail could lie inside
/// the failing block.
#[inline]
fn safe_reentry(bytes: &[u8], failing_block_pos: usize) -> usize {
    if failing_block_pos == 0 {
        return 0;
    }
    for i in 1..=3 {
        if failing_block_pos < i {
            break;
        }
        // High two bits != 0b10 ⇒ not a continuation byte ⇒ valid restart.
        if bytes[failing_block_pos - i] >> 6 != 0b10 {
            return failing_block_pos - i;
        }
    }
    // All three look like continuations: the previous block ended on a
    // complete 4-byte codepoint; restart at the failing block boundary.
    failing_block_pos
}

/// AVX-512BW UTF-8 validator. Returns the same triple `core::str::from_utf8`
/// would produce — see [`Utf8Validation`].
///
/// # Safety
///
/// `#[target_feature(enable = "avx512bw")]` requires the host CPU to
/// actually support AVX-512BW. Caller MUST gate this behind a runtime
/// feature check (e.g. `std::is_x86_feature_detected!("avx512bw")`).
/// Calling on a non-AVX-512BW CPU is undefined behavior.
#[target_feature(enable = "avx512bw")]
#[must_use]
pub(crate) unsafe fn validate_utf8(bytes: &[u8]) -> Utf8Validation {
    let len = bytes.len();
    if len < SIMD_CHUNK_SIZE {
        // Tail-only path: scalar handles short inputs.
        return scalar_diagnose(bytes, 0);
    }

    let iter_lim = len - (len % SIMD_CHUNK_SIZE);
    let ptr = bytes.as_ptr();

    // SAFETY (entire block below): AVX-512BW is enabled on this fn via
    // target_feature; the caller guarantees the CPU supports it. Pointer
    // adds use `idx + 64 <= iter_lim <= len`, so all loads are in-bounds.
    // All `unsafe` calls inside are AVX-512BW intrinsics or AVX-512BW
    // helpers.
    let processed = unsafe {
        let mut state = State::new();
        let mut idx: usize = 0;

        // Two-mode loop matching simdutf's icelake checker: stay in
        // pure-ASCII fast path when we can; switch to DFA mode on the first
        // non-ASCII block. Whenever the error vector becomes non-zero we
        // abandon SIMD and re-validate from a safe re-entry point.
        let mut only_ascii = true;
        let v_80 = splat(0x80);
        'outer: loop {
            if only_ascii {
                while idx < iter_lim {
                    let block = load64(ptr.add(idx));
                    // High-bit test on all 64 lanes; zero ⇒ pure ASCII.
                    if _mm512_test_epi8_mask(block, v_80) != 0 {
                        state.check_block(block);
                        if state.has_error() {
                            return scalar_diagnose(bytes, safe_reentry(bytes, idx));
                        }
                        only_ascii = false;
                        idx += SIMD_CHUNK_SIZE;
                        continue 'outer;
                    }
                    idx += SIMD_CHUNK_SIZE;
                }
            } else {
                while idx < iter_lim {
                    let block = load64(ptr.add(idx));
                    if _mm512_test_epi8_mask(block, v_80) == 0 {
                        // Pure-ASCII block flushes any pending incomplete
                        // state.
                        state.check_incomplete_pending();
                        if state.has_error() {
                            return scalar_diagnose(bytes, safe_reentry(bytes, idx));
                        }
                        only_ascii = true;
                        idx += SIMD_CHUNK_SIZE;
                        continue 'outer;
                    }
                    state.check_block(block);
                    if state.has_error() {
                        return scalar_diagnose(bytes, safe_reentry(bytes, idx));
                    }
                    idx += SIMD_CHUNK_SIZE;
                }
            }
            break;
        }

        // Flush incomplete state from the last full block before consulting
        // `has_error`, mirroring the AVX2 path's final
        // `check_incomplete_pending`.
        state.check_incomplete_pending();
        if state.has_error() {
            return scalar_diagnose(bytes, safe_reentry(bytes, iter_lim));
        }
        idx
    };

    // Tail (< 64 bytes) — scalar handles it. Restart at a safe boundary
    // that includes any in-flight multibyte sequence from the previous
    // block, so error_len / valid_up_to match `from_utf8` exactly.
    if processed < len {
        let reentry = safe_reentry(bytes, processed);
        return scalar_diagnose(bytes, reentry);
    }

    Utf8Validation {
        valid: true,
        valid_up_to: len,
        error_len: 0,
    }
}

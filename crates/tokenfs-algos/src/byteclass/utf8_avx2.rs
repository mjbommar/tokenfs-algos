//! AVX2 UTF-8 validator returning a [`Utf8Validation`] compatible with the
//! scalar reference implementation in [`super`].
//!
//! # Algorithm (Keiser-Lemire 3-pshufb DFA)
//!
//! UTF-8 validity for a stream of bytes can be expressed as a set of
//! per-byte error predicates that depend only on the current byte and the
//! previous one. Keiser and Lemire (2020, "Validating UTF-8 in less than
//! one instruction per byte") encode that table as three 16-entry lookups
//! evaluated with `pshufb`:
//!
//! 1. `byte_1_high` — indexed by the high nibble of `prev1`. Tells us what
//!    classes of error the leading byte could produce (TOO_LONG, TWO_CONTS,
//!    TOO_SHORT, OVERLONG_*, SURROGATE, TOO_LARGE_*).
//! 2. `byte_1_low`  — indexed by the low nibble of `prev1`. Refines
//!    OVERLONG / TOO_LARGE / SURROGATE for specific leader values.
//! 3. `byte_2_high` — indexed by the high nibble of the current byte. Says
//!    what kinds of follow-byte the current byte can be (continuation vs.
//!    leader vs. ASCII).
//!
//! AND-ing the three lookups yields a non-zero byte exactly when `(prev1,
//! input)` is an illegal pair. A separate check ensures bytes that *must*
//! be 2nd or 3rd continuations actually carry the `0b10xx_xxxx` mark
//! (`check_multibyte_lengths`). Errors accumulate into a single OR-vector;
//! `is_incomplete` keeps a tail vector so a 2-, 3-, or 4-byte sequence
//! that straddles the final 64-byte block is reported correctly.
//!
//! # Safety contract
//!
//! [`validate_utf8`] is `unsafe` and tagged `#[target_feature(enable =
//! "avx2")]`. The caller MUST verify AVX2 is available on the running CPU
//! (e.g. via `std::is_x86_feature_detected!("avx2")`) before invoking it.
//! Calling on a non-AVX2 CPU is undefined behavior.
//!
//! # Trade-off: vectorized check, scalar diagnosis
//!
//! The vector pipeline only answers "is there an error in this stream?".
//! Recovering the precise `valid_up_to` and `error_len` from vector state
//! is expensive and historically bug-prone. simdutf8 itself defers to
//! `core::str::from_utf8` for diagnosis (see `helpers::get_compat_error`
//! and `validate_utf8_compat_simd0`). We do the same: when the SIMD pass
//! flags an error we re-run the scalar validator from a safe re-entry
//! point near the failing 64-byte block. Pure-ASCII chunks short-circuit
//! via `_mm256_movemask_epi8`. Tails (< 64 bytes) go straight to scalar.
//!
//! # Provenance
//!
//! Ported from `rusticstuff/simdutf8` @ commit `a02d0cace1787e4a683e75530d826d96201b2060`:
//! - `src/implementation/x86/avx2.rs`        (SIMD primitive wrappers)
//! - `src/implementation/algorithm.rs`       (DFA tables + outer loop in `algorithm_simd!`)
//! - `src/implementation/helpers.rs`         (`get_compat_error`, `SIMD_CHUNK_SIZE = 64`)
//!
//! Upstream is dual-licensed MIT OR Apache-2.0; this file is distributed
//! under the same terms by the original author below.
//
// ============================================================================
// Upstream attribution (verbatim from rusticstuff/simdutf8 Cargo.toml):
//   authors = ["Hans Kratz <hans@appfour.com>"]
//   license = "MIT OR Apache-2.0"
//
// MIT License notice (excerpt from upstream LICENSE-MIT):
//   Permission is hereby granted, free of charge, to any person obtaining
//   a copy of this software and associated documentation files (the
//   "Software"), to deal in the Software without restriction... THE
//   SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
//
// This port retains the same MIT OR Apache-2.0 terms.
// ============================================================================

use super::Utf8Validation;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_alignr_epi8, _mm256_and_si256, _mm256_loadu_si256, _mm256_movemask_epi8,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi8, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_subs_epu8,
    _mm256_testz_si256, _mm256_xor_si256,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_alignr_epi8, _mm256_and_si256, _mm256_loadu_si256, _mm256_movemask_epi8,
    _mm256_or_si256, _mm256_permute2x128_si256, _mm256_set1_epi8, _mm256_setr_epi8,
    _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16, _mm256_subs_epu8,
    _mm256_testz_si256, _mm256_xor_si256,
};

/// 64 bytes / iteration: two AVX2 lanes. Matches `helpers::SIMD_CHUNK_SIZE`.
const SIMD_CHUNK_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// SIMD primitive helpers — direct port of `x86/avx2.rs::SimdU8Value` methods.
// All operations here are unsafe because they require AVX2; safety is
// established by the `#[target_feature(enable = "avx2")]` boundary on the
// public entry point and the caller's CPU-feature precondition.
// ---------------------------------------------------------------------------

/// Loads 32 bytes (unaligned) into an AVX2 lane.
///
/// SAFETY: precondition — AVX2 is available; `ptr..ptr+32` is readable.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load32(ptr: *const u8) -> __m256i {
    // SAFETY: caller upholds the precondition above.
    unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) }
}

/// Splat `val` into all 32 lanes. SAFETY: precondition — AVX2 is available.
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::cast_possible_wrap)]
unsafe fn splat(val: u8) -> __m256i {
    // `_mm256_set1_epi8` is safe in current stdarch when reached via a
    // `target_feature(avx2)` boundary; no inner unsafe block needed.
    _mm256_set1_epi8(val as i8)
}

/// Per-byte right-shift by 4 (high nibble extraction). Mirrors `shr4` in
/// upstream `avx2.rs`: 16-bit logical shift, then mask off neighboring bits.
///
/// SAFETY: precondition — AVX2 is available.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn shr4(v: __m256i) -> __m256i {
    // SAFETY: AVX2 enabled by target_feature.
    unsafe { _mm256_and_si256(_mm256_srli_epi16(v, 4), splat(0x0F)) }
}

/// 16-entry lookup table replicated across both 128-bit halves, indexed by
/// the low nibble of each byte of `v`. This is the `pshufb` table dispatch.
///
/// SAFETY: precondition — AVX2 is available.
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::cast_possible_wrap, clippy::too_many_arguments)]
unsafe fn lookup16(
    v: __m256i,
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
) -> __m256i {
    let table = _mm256_setr_epi8(
        t0 as i8, t1 as i8, t2 as i8, t3 as i8, t4 as i8, t5 as i8, t6 as i8, t7 as i8, t8 as i8,
        t9 as i8, ta as i8, tb as i8, tc as i8, td as i8, te as i8, tf as i8, t0 as i8, t1 as i8,
        t2 as i8, t3 as i8, t4 as i8, t5 as i8, t6 as i8, t7 as i8, t8 as i8, t9 as i8, ta as i8,
        tb as i8, tc as i8, td as i8, te as i8, tf as i8,
    );
    _mm256_shuffle_epi8(table, v)
}

/// Cross-lane "previous N" — returns a vector whose byte `i` is `input[i-N]`
/// with the first N bytes coming from `prev`. Mirrors `prev1`/`prev2`/`prev3`
/// in upstream `avx2.rs`; 0x21 swaps the halves so `palignr` can splice.
/// `IMM` is `16 - N`; Rust const generics don't yet allow arithmetic in the
/// instantiation site, so callers pass the precomputed value (15, 14, 13).
///
/// SAFETY: precondition — AVX2 is available; `IMM` in 13..=15.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn prev<const IMM: i32>(input: __m256i, prev_block: __m256i) -> __m256i {
    let shifted = _mm256_permute2x128_si256(prev_block, input, 0x21);
    _mm256_alignr_epi8::<IMM>(input, shifted)
}

// ---------------------------------------------------------------------------
// DFA tables — verbatim from `algorithm.rs::check_special_cases` and
// `is_incomplete`. Bit assignments are load-bearing; do not renumber.
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
/// SAFETY: precondition — AVX2 is available.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn check_special_cases(input: __m256i, prev1: __m256i) -> __m256i {
    // SAFETY: AVX2 enabled by target_feature; helpers below preserve it.
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
            _mm256_and_si256(prev1, splat(0x0F)),
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
        _mm256_and_si256(_mm256_and_si256(byte_1_high, byte_1_low), byte_2_high)
    }
}

/// "Must-be-continuation" check: bytes ≥ 0xE0 require a 3rd cont; bytes ≥
/// 0xF0 require a 4th. XOR with `special_cases` so non-continuations show
/// up as errors. Mirrors `must_be_2_3_continuation` + `check_multibyte_lengths`.
///
/// SAFETY: precondition — AVX2 is available.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn check_multibyte_lengths(
    input: __m256i,
    prev_block: __m256i,
    special_cases: __m256i,
) -> __m256i {
    // SAFETY: AVX2 enabled by target_feature; 16 - 2 = 14, 16 - 3 = 13.
    unsafe {
        let prev2 = prev::<14>(input, prev_block);
        let prev3 = prev::<13>(input, prev_block);
        let is_third = _mm256_subs_epu8(prev2, splat(0xE0 - 0x80));
        let is_fourth = _mm256_subs_epu8(prev3, splat(0xF0 - 0x80));
        let must23 = _mm256_or_si256(is_third, is_fourth);
        let must23_80 = _mm256_and_si256(must23, splat(0x80));
        _mm256_xor_si256(must23_80, special_cases)
    }
}

/// Builds the "incomplete trailing sequence" mask for the last 32-byte
/// vector of a block. Mirrors `algorithm.rs::is_incomplete`. The 0xFF
/// pattern with the final three bytes lowered detects 2/3/4-byte starters
/// in the last 1/2/3 lanes that lack their continuations.
///
/// SAFETY: precondition — AVX2 is available.
#[target_feature(enable = "avx2")]
#[inline]
#[allow(clippy::cast_possible_wrap)]
unsafe fn is_incomplete(input: __m256i) -> __m256i {
    let max = _mm256_setr_epi8(
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
        (0b1111_0000 - 1) as i8,
        (0b1110_0000 - 1) as i8,
        (0b1100_0000 - 1) as i8,
    );
    _mm256_subs_epu8(input, max)
}

/// Per-block accumulator state mirroring `Utf8CheckAlgorithm`.
struct State {
    prev: __m256i,
    incomplete: __m256i,
    error: __m256i,
}

impl State {
    /// SAFETY: precondition — AVX2 is available.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn new() -> Self {
        let z = _mm256_setzero_si256();
        Self {
            prev: z,
            incomplete: z,
            error: z,
        }
    }

    /// Process one 32-byte lane. SAFETY: precondition — AVX2 available.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn check_bytes(&mut self, input: __m256i) {
        // SAFETY: AVX2 enabled; 16 - 1 = 15 is the literal const generic arg.
        unsafe {
            let p1 = prev::<15>(input, self.prev);
            let sc = check_special_cases(input, p1);
            let lengths = check_multibyte_lengths(input, self.prev, sc);
            self.error = _mm256_or_si256(self.error, lengths);
            self.prev = input;
        }
    }

    /// Process a full 64-byte block (two 32-byte lanes). SAFETY: precondition — AVX2 available.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn check_block(&mut self, lane0: __m256i, lane1: __m256i) {
        // SAFETY: AVX2 enabled by target_feature.
        unsafe {
            self.check_bytes(lane0);
            self.check_bytes(lane1);
            self.incomplete = is_incomplete(lane1);
        }
    }

    /// Fold pending incomplete-sequence state into the error vector.
    /// Mirrors `check_incomplete_pending`. SAFETY: precondition — AVX2 available.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn check_incomplete_pending(&mut self) {
        self.error = _mm256_or_si256(self.error, self.incomplete);
    }

    /// True iff any error bit was ever set. SAFETY: precondition — AVX2 available.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn has_error(&self) -> bool {
        _mm256_testz_si256(self.error, self.error) != 1
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

/// Pick a safe scalar re-entry point near a failing 64-byte block.
/// Mirrors `helpers::get_compat_error`: walk back up to three bytes from
/// the start of the failing block until we find a non-continuation. That
/// byte is the earliest possible start of a sequence whose tail could lie
/// inside the failing block.
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

/// AVX2 UTF-8 validator. Returns the same triple `core::str::from_utf8`
/// would produce — see [`Utf8Validation`].
///
/// # Safety
///
/// `#[target_feature(enable = "avx2")]` requires the host CPU to actually
/// support AVX2. Caller MUST gate this behind a runtime feature check
/// (e.g. `std::is_x86_feature_detected!("avx2")`). Calling on a non-AVX2
/// CPU is undefined behavior.
#[target_feature(enable = "avx2")]
#[must_use]
pub(crate) unsafe fn validate_utf8(bytes: &[u8]) -> Utf8Validation {
    let len = bytes.len();
    if len < SIMD_CHUNK_SIZE {
        // Tail-only path: scalar handles short inputs. Mirrors upstream
        // `validate_utf8_compat_simd0` which also delegates the < 64 case.
        return scalar_diagnose(bytes, 0);
    }

    let iter_lim = len - (len % SIMD_CHUNK_SIZE);
    let ptr = bytes.as_ptr();

    // SAFETY (entire block below): AVX2 is enabled on this fn via
    // target_feature; the caller guarantees the CPU supports it. Pointer
    // adds use `idx + 32 <= iter_lim <= len`, so all loads are in-bounds.
    // All `unsafe` calls inside are AVX2 intrinsics or AVX2 helpers.
    let processed = unsafe {
        let mut state = State::new();
        let mut idx: usize = 0;

        // Two-mode loop matching simdutf8: stay in pure-ASCII fast path when
        // we can; switch to DFA mode on the first non-ASCII block. Whenever
        // the error vector becomes non-zero we abandon SIMD and re-validate
        // from a safe re-entry point with scalar code.
        let mut only_ascii = true;
        'outer: loop {
            if only_ascii {
                while idx < iter_lim {
                    let l0 = load32(ptr.add(idx));
                    let l1 = load32(ptr.add(idx + 32));
                    let combined = _mm256_or_si256(l0, l1);
                    // High-bit movemask == 0 ⇒ all 64 bytes are ASCII.
                    if _mm256_movemask_epi8(combined) != 0 {
                        state.check_block(l0, l1);
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
                    let l0 = load32(ptr.add(idx));
                    let l1 = load32(ptr.add(idx + 32));
                    let combined = _mm256_or_si256(l0, l1);
                    if _mm256_movemask_epi8(combined) == 0 {
                        // Pure-ASCII block flushes any pending incomplete state.
                        state.check_incomplete_pending();
                        if state.has_error() {
                            return scalar_diagnose(bytes, safe_reentry(bytes, idx));
                        }
                        only_ascii = true;
                        idx += SIMD_CHUNK_SIZE;
                        continue 'outer;
                    }
                    state.check_block(l0, l1);
                    if state.has_error() {
                        return scalar_diagnose(bytes, safe_reentry(bytes, idx));
                    }
                    idx += SIMD_CHUNK_SIZE;
                }
            }
            break;
        }

        // Flush incomplete state from the last full block before consulting
        // `has_error`, mirroring upstream's final `check_incomplete_pending`.
        state.check_incomplete_pending();
        if state.has_error() {
            return scalar_diagnose(bytes, safe_reentry(bytes, iter_lim));
        }
        idx
    };

    // Tail (< 64 bytes) — scalar handles it. We restart at a safe boundary
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

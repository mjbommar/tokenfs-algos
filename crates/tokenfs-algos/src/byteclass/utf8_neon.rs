//! AArch64 NEON UTF-8 validator returning a [`Utf8Validation`] compatible
//! with the scalar reference implementation in [`super`].
//!
//! # Algorithm (Keiser-Lemire 3-pshufb DFA, NEON port)
//!
//! Same DFA as `utf8_avx2.rs`: per-byte UTF-8 validity is expressed as a
//! function of `(prev_byte, current_byte)` and evaluated with three
//! 16-entry table lookups whose results are AND-ed together. On NEON the
//! `pshufb`-style dispatch is `vqtbl1q_u8(table, indices)` — a 16-entry
//! byte-table lookup over a `uint8x16_t` selector. Cross-block "previous N"
//! splices use `vextq_u8(prev, current, IMM)` instead of AVX2's
//! `_mm256_alignr_epi8`. Error accumulation OR's a single state vector
//! across the stream; `is_incomplete` keeps a tail vector so a 2-, 3-, or
//! 4-byte sequence that straddles the final 64-byte block is reported
//! correctly.
//!
//! Each iteration consumes 64 bytes as four 16-byte lanes (NEON's natural
//! register width), which matches simdutf8's upstream NEON path and the
//! 64-byte `SIMD_CHUNK_SIZE` invariant the safe-reentry helper depends on.
//!
//! # Safety contract
//!
//! [`validate_utf8`] is `unsafe` and tagged `#[target_feature(enable =
//! "neon")]`. NEON is mandatory in the AArch64 ABI, so the precondition is
//! satisfied unconditionally for `target_arch = "aarch64"` builds —
//! callers do not need a runtime feature check. The function is still
//! `unsafe` for symmetry with the AVX2 entry point and to keep the option
//! open of running on hypothetical AArch64 targets without NEON.
//!
//! # Trade-off: vectorized check, scalar diagnosis
//!
//! The vector pipeline only answers "is there an error in this stream?".
//! Recovering the precise `valid_up_to` and `error_len` from vector state
//! is expensive and historically bug-prone. simdutf8 itself defers to
//! `core::str::from_utf8` for diagnosis. We do the same: when the SIMD
//! pass flags an error we re-run the scalar validator from a safe
//! re-entry point near the failing 64-byte block. Pure-ASCII chunks
//! short-circuit via `vmaxvq_u8` on the high-bit mask. Tails (< 64 bytes)
//! go straight to scalar.
//!
//! # Provenance
//!
//! Ported from `rusticstuff/simdutf8` @ commit `a02d0cace1787e4a683e75530d826d96201b2060`:
//! - `src/implementation/aarch64/neon.rs`     (NEON SIMD primitive wrappers)
//! - `src/implementation/algorithm.rs`        (DFA tables + outer loop in `algorithm_simd!`)
//! - `src/implementation/helpers.rs`          (`get_compat_error`, `SIMD_CHUNK_SIZE = 64`)
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

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::{
    uint8x16_t, vandq_u8, vdupq_n_u8, veorq_u8, vextq_u8, vld1q_u8, vmaxvq_u8, vorrq_u8, vqsubq_u8,
    vqtbl1q_u8, vshrq_n_u8,
};

/// 64 bytes / iteration: four 16-byte NEON lanes. Matches `helpers::SIMD_CHUNK_SIZE`.
const SIMD_CHUNK_SIZE: usize = 64;
/// One NEON register. The DFA processes 16 bytes per `check_bytes` call.
const LANE_BYTES: usize = 16;

// ---------------------------------------------------------------------------
// SIMD primitive helpers — direct port of `aarch64/neon.rs::SimdU8Value`
// methods. All operations here are unsafe because they require NEON; safety
// is established by the `#[target_feature(enable = "neon")]` boundary on
// the public entry point. Pure-compute intrinsics (`vdupq_n_u8`,
// `vandq_u8`, etc.) are SAFE in current stdarch when reached from a
// `target_feature("neon")` context, so they are NOT wrapped in inner
// `unsafe { }` blocks. Only the loads and helper calls (which carry their
// own `unsafe fn` signature) need explicit `unsafe { }` wrappers.
// ---------------------------------------------------------------------------

/// Loads 16 bytes (unaligned) into a NEON register.
///
/// SAFETY: precondition — NEON is available; `ptr..ptr+16` is readable.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn load16(ptr: *const u8) -> uint8x16_t {
    // SAFETY: caller upholds the precondition above.
    unsafe { vld1q_u8(ptr) }
}

/// Per-byte right-shift by 4 (high nibble extraction). Mirrors `shr4` in
/// upstream `aarch64/neon.rs`. NEON's `vshrq_n_u8` is a per-lane logical
/// shift, so no follow-up mask is needed (unlike the AVX2 16-bit shift).
///
/// SAFETY: precondition — NEON is available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn shr4(v: uint8x16_t) -> uint8x16_t {
    vshrq_n_u8::<4>(v)
}

/// 16-entry lookup table indexed by the low nibble of each byte of `v`.
/// This is the `pshufb`-equivalent table dispatch: `vqtbl1q_u8(table,
/// indices)` returns `table[indices[i] & 0x0F]` for `indices[i] < 16` and
/// 0 otherwise (the high bit on an index produces 0 — exactly the
/// behavior we want for inputs whose high nibble we already masked off).
///
/// SAFETY: precondition — NEON is available.
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn lookup16(
    v: uint8x16_t,
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
) -> uint8x16_t {
    let table_bytes: [u8; 16] = [
        t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf,
    ];
    // SAFETY: the array is 16 bytes and properly aligned for `vld1q_u8`.
    let table = unsafe { vld1q_u8(table_bytes.as_ptr()) };
    vqtbl1q_u8(table, v)
}

/// "Previous N" — returns a vector whose byte `i` is `input[i-N]` with
/// the first N bytes coming from `prev_block`. NEON's `vextq_u8(a, b,
/// IMM)` concatenates `a` and `b` and extracts a 16-byte window starting
/// at offset `IMM` (in bytes). Passing `prev` then `input` with `IMM =
/// 16 - N` reconstructs `[prev[16-N..], input[..16-N]]` — exactly the
/// "shift in N bytes from prev" semantics we need.
///
/// SAFETY: precondition — NEON is available; `IMM` in 13..=15.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn prev<const IMM: i32>(input: uint8x16_t, prev_block: uint8x16_t) -> uint8x16_t {
    vextq_u8(prev_block, input, IMM)
}

// ---------------------------------------------------------------------------
// DFA tables — Mirror of utf8_avx2.rs DFA tables; bit assignments are
// load-bearing — keep in sync. (A future refactor can pull these into a
// shared `pub(super) mod utf8_dfa_tables`; for now the tiny duplication
// keeps each file self-contained, which simplifies cfg gating.)
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
/// SAFETY: precondition — NEON is available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn check_special_cases(input: uint8x16_t, prev1: uint8x16_t) -> uint8x16_t {
    // SAFETY: NEON enabled by target_feature; helpers below preserve it.
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
            vandq_u8(prev1, vdupq_n_u8(0x0F)),
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
        vandq_u8(vandq_u8(byte_1_high, byte_1_low), byte_2_high)
    }
}

/// "Must-be-continuation" check: bytes 0xE0-0xEF require a 3rd
/// continuation, bytes 0xF0-0xF7 require a 4th. XOR with `special_cases`
/// so non-continuations show up as errors. Mirrors
/// `must_be_2_3_continuation` + `check_multibyte_lengths` from upstream
/// `algorithm.rs`.
///
/// SAFETY: precondition — NEON is available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn check_multibyte_lengths(
    input: uint8x16_t,
    prev_block: uint8x16_t,
    special_cases: uint8x16_t,
) -> uint8x16_t {
    // SAFETY: NEON enabled by target_feature; 16 - 2 = 14, 16 - 3 = 13.
    unsafe {
        let prev2 = prev::<14>(input, prev_block);
        let prev3 = prev::<13>(input, prev_block);
        let is_third = vqsubq_u8(prev2, vdupq_n_u8(0xE0 - 0x80));
        let is_fourth = vqsubq_u8(prev3, vdupq_n_u8(0xF0 - 0x80));
        let must23 = vorrq_u8(is_third, is_fourth);
        let must23_80 = vandq_u8(must23, vdupq_n_u8(0x80));
        veorq_u8(must23_80, special_cases)
    }
}

/// Builds the "incomplete trailing sequence" mask for the last 16-byte
/// vector of a block. Mirrors `algorithm.rs::is_incomplete`. The 0xFF
/// pattern with the final three bytes lowered detects 2/3/4-byte starters
/// in the last 1/2/3 lanes that lack their continuations.
///
/// SAFETY: precondition — NEON is available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn is_incomplete(input: uint8x16_t) -> uint8x16_t {
    let max_bytes: [u8; 16] = [
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0xFF,
        0b1111_0000 - 1,
        0b1110_0000 - 1,
        0b1100_0000 - 1,
    ];
    // SAFETY: 16 bytes, properly aligned for `vld1q_u8`.
    let max = unsafe { vld1q_u8(max_bytes.as_ptr()) };
    vqsubq_u8(input, max)
}

/// Per-block accumulator state mirroring `Utf8CheckAlgorithm`. We keep
/// the previous 16-byte lane so `prev1`/`prev2`/`prev3` can be spliced
/// across block boundaries.
struct State {
    prev: uint8x16_t,
    incomplete: uint8x16_t,
    error: uint8x16_t,
}

impl State {
    /// SAFETY: precondition — NEON is available.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn new() -> Self {
        let z = vdupq_n_u8(0);
        Self {
            prev: z,
            incomplete: z,
            error: z,
        }
    }

    /// Process one 16-byte lane. SAFETY: precondition — NEON available.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn check_bytes(&mut self, input: uint8x16_t) {
        // SAFETY: NEON enabled; 16 - 1 = 15 is the literal const generic arg.
        unsafe {
            let p1 = prev::<15>(input, self.prev);
            let sc = check_special_cases(input, p1);
            let lengths = check_multibyte_lengths(input, self.prev, sc);
            self.error = vorrq_u8(self.error, lengths);
            self.prev = input;
        }
    }

    /// Process a full 64-byte block (four 16-byte lanes). SAFETY:
    /// precondition — NEON available.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn check_block(
        &mut self,
        lane0: uint8x16_t,
        lane1: uint8x16_t,
        lane2: uint8x16_t,
        lane3: uint8x16_t,
    ) {
        // SAFETY: NEON enabled by target_feature.
        unsafe {
            self.check_bytes(lane0);
            self.check_bytes(lane1);
            self.check_bytes(lane2);
            self.check_bytes(lane3);
            self.incomplete = is_incomplete(lane3);
        }
    }

    /// Fold pending incomplete-sequence state into the error vector.
    /// Mirrors `check_incomplete_pending`. SAFETY: precondition — NEON
    /// available.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn check_incomplete_pending(&mut self) {
        self.error = vorrq_u8(self.error, self.incomplete);
    }

    /// True iff any error bit was ever set. NEON has no movemask, so we
    /// reduce horizontally with `vmaxvq_u8`. SAFETY: precondition — NEON
    /// available.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn has_error(&self) -> bool {
        vmaxvq_u8(self.error) != 0
    }
}

/// Returns true iff every byte across the four lanes has its high bit
/// clear (pure ASCII). One `vmaxvq_u8` over the OR-combined vector
/// suffices: any byte >= 0x80 forces the max into the 0x80..=0xFF range.
///
/// SAFETY: precondition — NEON is available.
#[target_feature(enable = "neon")]
#[inline]
unsafe fn is_pure_ascii(l0: uint8x16_t, l1: uint8x16_t, l2: uint8x16_t, l3: uint8x16_t) -> bool {
    let or01 = vorrq_u8(l0, l1);
    let or23 = vorrq_u8(l2, l3);
    let combined = vorrq_u8(or01, or23);
    (vmaxvq_u8(combined) & 0x80) == 0
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

/// AArch64 NEON UTF-8 validator. Returns the same triple
/// `core::str::from_utf8` would produce — see [`Utf8Validation`].
///
/// # Safety
///
/// `#[target_feature(enable = "neon")]` requires the host CPU to actually
/// support NEON. On AArch64 NEON is part of the base ABI; the
/// precondition is unconditionally satisfied. Calling on a hypothetical
/// AArch64 target without NEON would be undefined behavior.
#[target_feature(enable = "neon")]
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

    // SAFETY (entire block below): NEON is enabled on this fn via
    // target_feature; AArch64 mandates NEON in the base ABI. Pointer adds
    // use `idx + 48 + 16 <= iter_lim <= len`, so all four 16-byte loads
    // are in-bounds. All `unsafe` calls inside are NEON intrinsics or
    // NEON helpers.
    let processed = unsafe {
        let mut state = State::new();
        let mut idx: usize = 0;

        // Two-mode loop matching simdutf8: stay in pure-ASCII fast path
        // when we can; switch to DFA mode on the first non-ASCII block.
        // Whenever the error vector becomes non-zero we abandon SIMD and
        // re-validate from a safe re-entry point with scalar code.
        let mut only_ascii = true;
        'outer: loop {
            if only_ascii {
                while idx < iter_lim {
                    let l0 = load16(ptr.add(idx));
                    let l1 = load16(ptr.add(idx + LANE_BYTES));
                    let l2 = load16(ptr.add(idx + 2 * LANE_BYTES));
                    let l3 = load16(ptr.add(idx + 3 * LANE_BYTES));
                    if !is_pure_ascii(l0, l1, l2, l3) {
                        state.check_block(l0, l1, l2, l3);
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
                    let l0 = load16(ptr.add(idx));
                    let l1 = load16(ptr.add(idx + LANE_BYTES));
                    let l2 = load16(ptr.add(idx + 2 * LANE_BYTES));
                    let l3 = load16(ptr.add(idx + 3 * LANE_BYTES));
                    if is_pure_ascii(l0, l1, l2, l3) {
                        // Pure-ASCII block flushes any pending incomplete state.
                        state.check_incomplete_pending();
                        if state.has_error() {
                            return scalar_diagnose(bytes, safe_reentry(bytes, idx));
                        }
                        only_ascii = true;
                        idx += SIMD_CHUNK_SIZE;
                        continue 'outer;
                    }
                    state.check_block(l0, l1, l2, l3);
                    if state.has_error() {
                        return scalar_diagnose(bytes, safe_reentry(bytes, idx));
                    }
                    idx += SIMD_CHUNK_SIZE;
                }
            }
            break;
        }

        // Flush incomplete state from the last full block before
        // consulting `has_error`, mirroring upstream's final
        // `check_incomplete_pending`.
        state.check_incomplete_pending();
        if state.has_error() {
            return scalar_diagnose(bytes, safe_reentry(bytes, iter_lim));
        }
        idx
    };

    // Tail (< 64 bytes) — scalar handles it. We restart at a safe
    // boundary that includes any in-flight multibyte sequence from the
    // previous block, so error_len / valid_up_to match `from_utf8`
    // exactly.
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

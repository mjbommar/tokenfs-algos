//! Fuzz target: the dispatched UTF-8 validator must agree with
//! `core::str::from_utf8` byte-for-byte on every input.
//!
//! The AVX2 implementation runs a Keiser-Lemire shuffle DFA over 64-byte
//! chunks and falls back to scalar for diagnosis. A divergence in
//! `valid_up_to` or `error_len` would silently corrupt downstream consumers
//! that act on those offsets — fuzz wide.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::byteclass::{self, Utf8Validation};

fuzz_target!(|data: &[u8]| {
    let actual = byteclass::validate_utf8(data);

    let expected = match core::str::from_utf8(data) {
        Ok(_) => Utf8Validation {
            valid: true,
            valid_up_to: data.len(),
            error_len: 0,
        },
        Err(error) => Utf8Validation {
            valid: false,
            valid_up_to: error.valid_up_to(),
            error_len: error.error_len().unwrap_or(0) as u8,
        },
    };

    assert_eq!(
        actual, expected,
        "dispatched validate_utf8 diverged from core::str::from_utf8 on input \
         of length {}",
        data.len()
    );
});

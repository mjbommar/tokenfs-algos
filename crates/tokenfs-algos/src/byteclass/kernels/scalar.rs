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

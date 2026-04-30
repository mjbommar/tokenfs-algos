//! Byte classification primitives.
//!
//! The current implementation is scalar and portable. The API is shaped so
//! AVX2/NEON preclassification kernels can slot in without changing callers.

/// Counts coarse byte classes in one pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ByteClassCounts {
    /// Printable ASCII bytes, including space.
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

impl ByteClassCounts {
    /// Counts all bytes in the class summary.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.printable_ascii + self.whitespace + self.control + self.high_bit + self.other
    }
}

/// Scalar byte-class preclassification pass.
#[must_use]
pub fn classify(bytes: &[u8]) -> ByteClassCounts {
    let mut counts = ByteClassCounts::default();
    for &byte in bytes {
        match byte {
            b'\t' | b'\n' | b'\r' | b' ' => counts.whitespace += 1,
            0x20..=0x7e => counts.printable_ascii += 1,
            0x00..=0x1f | 0x7f => counts.control += 1,
            0x80..=0xff => counts.high_bit += 1,
        }
    }
    counts
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
    use super::{classify, is_ascii_dominant};

    #[test]
    fn classifies_ascii_text() {
        let counts = classify(b"abc 123\n");
        assert_eq!(counts.whitespace, 2);
        assert_eq!(counts.printable_ascii, 6);
        assert!(is_ascii_dominant(b"abc 123\n"));
    }
}

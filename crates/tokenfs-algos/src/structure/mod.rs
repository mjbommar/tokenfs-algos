//! Cheap byte-structure detectors.
//!
//! These detectors are intentionally simple, scalar, and allocation-free. They
//! provide planner and selector signals that are cheaper than full parsing:
//! sparse pages, ASCII dominance, transition rate, repeated motifs, and
//! low-cardinality palettes.

use crate::{byteclass, runlength};

/// Summary of structural byte signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructureSummary {
    /// Total bytes scanned.
    pub total_bytes: usize,
    /// Run-length summary.
    pub runs: runlength::RunLengthSummary,
    /// Coarse byte-class counts.
    pub byte_classes: byteclass::ByteClassCounts,
    /// Number of zero bytes.
    pub zero_bytes: u64,
    /// Number of distinct byte values.
    pub unique_bytes: u16,
    /// Smallest repeated period detected from the candidate set.
    pub repeated_period: Option<usize>,
    /// Number of all-zero 4 KiB pages/chunks.
    pub zero_pages_4k: u64,
    /// Whether the whole slice is valid UTF-8.
    pub utf8_valid: bool,
}

impl Default for StructureSummary {
    fn default() -> Self {
        Self {
            total_bytes: 0,
            runs: runlength::RunLengthSummary::default(),
            byte_classes: byteclass::ByteClassCounts::default(),
            zero_bytes: 0,
            unique_bytes: 0,
            repeated_period: None,
            zero_pages_4k: 0,
            utf8_valid: true,
        }
    }
}

impl StructureSummary {
    /// Fraction of bytes that are zero.
    #[must_use]
    pub fn zero_fraction(self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.zero_bytes as f32 / self.total_bytes as f32
        }
    }

    /// Fraction of bytes that are printable ASCII or ASCII whitespace.
    #[must_use]
    pub fn ascii_text_fraction(self) -> f32 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        let ascii = self.byte_classes.printable_ascii + self.byte_classes.whitespace;
        ascii as f32 / self.total_bytes as f32
    }

    /// Fraction of bytes with the high bit set.
    #[must_use]
    pub fn high_bit_fraction(self) -> f32 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        self.byte_classes.high_bit as f32 / self.total_bytes as f32
    }

    /// True when the slice is zero-filled.
    #[must_use]
    pub fn is_zero_filled(self) -> bool {
        self.total_bytes != 0 && self.zero_bytes as usize == self.total_bytes
    }

    /// True when zero bytes dominate the slice.
    #[must_use]
    pub fn is_sparse_like(self) -> bool {
        self.total_bytes != 0 && self.zero_bytes.saturating_mul(100) >= self.total_bytes as u64 * 95
    }

    /// True when the slice has a small byte palette.
    #[must_use]
    pub fn is_low_cardinality(self) -> bool {
        self.total_bytes != 0 && self.unique_bytes <= 16
    }

    /// True when text-like bytes dominate.
    #[must_use]
    pub fn is_text_like(self) -> bool {
        self.total_bytes != 0 && self.ascii_text_fraction() >= 0.90
    }
}

/// Builds a structural summary in allocation-free scalar passes.
#[must_use]
pub fn summarize(bytes: &[u8]) -> StructureSummary {
    if bytes.is_empty() {
        return StructureSummary::default();
    }

    let runs = runlength::summarize(bytes);
    let byte_classes = byteclass::classify(bytes);
    let mut seen = [false; 256];
    let mut unique = 0_u16;
    let mut zero_bytes = 0_u64;

    for &byte in bytes {
        if byte == 0 {
            zero_bytes += 1;
        }
        let slot = &mut seen[byte as usize];
        if !*slot {
            *slot = true;
            unique += 1;
        }
    }

    StructureSummary {
        total_bytes: bytes.len(),
        runs,
        byte_classes,
        zero_bytes,
        unique_bytes: unique,
        repeated_period: repeated_period(bytes),
        zero_pages_4k: count_zero_pages_4k(bytes),
        utf8_valid: core::str::from_utf8(bytes).is_ok(),
    }
}

/// Returns true when every byte is zero.
#[must_use]
pub fn is_zero_page(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.iter().all(|&byte| byte == 0)
}

/// Returns the smallest repeated period from a fixed candidate set.
#[must_use]
pub fn repeated_period(bytes: &[u8]) -> Option<usize> {
    const CANDIDATES: [usize; 13] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

    if bytes.len() < 2 {
        return None;
    }

    for period in CANDIDATES {
        if period >= bytes.len() || !bytes.len().is_multiple_of(period) {
            continue;
        }
        if bytes[period..]
            .iter()
            .enumerate()
            .all(|(index, &byte)| byte == bytes[index % period])
        {
            return Some(period);
        }
    }

    None
}

/// Counts all-zero 4 KiB chunks.
#[must_use]
pub fn count_zero_pages_4k(bytes: &[u8]) -> u64 {
    bytes
        .chunks(4 * 1024)
        .filter(|chunk| chunk.len() == 4 * 1024 && chunk.iter().all(|&byte| byte == 0))
        .count() as u64
}

#[cfg(test)]
mod tests {
    use super::{is_zero_page, repeated_period, summarize};
    // `vec!` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;

    #[test]
    fn detects_zero_and_sparse_data() {
        let bytes = vec![0_u8; 8192];
        let summary = summarize(&bytes);
        assert!(summary.is_zero_filled());
        assert!(summary.is_sparse_like());
        assert_eq!(summary.zero_pages_4k, 2);
        assert!(is_zero_page(&bytes));
    }

    #[test]
    fn detects_repeated_motif_period() {
        assert_eq!(repeated_period(b"abcdabcdabcdabcd"), Some(4));
        assert_eq!(summarize(b"abcdabcdabcdabcd").repeated_period, Some(4));
    }

    #[test]
    fn classifies_text_like_and_low_cardinality() {
        let summary = summarize(b"hello tokenfs\nhello tokenfs\n");
        assert!(summary.is_text_like());
        assert!(summary.utf8_valid);
        assert!(summary.unique_bytes <= 16);
    }
}

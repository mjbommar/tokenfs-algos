//! Selector feature extraction.
//!
//! This module is the product-facing name for the F21 selector inputs. It does
//! not implement a learned model; it exposes stable, cheap signals that
//! downstream selectors can use to choose representation, compression, or
//! dispatch paths.

use crate::{fingerprint, structure};

/// Low-level features used by representation selectors.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectorSignals {
    /// Content fingerprint.
    pub fingerprint: fingerprint::ExtentFingerprint,
    /// Structural byte summary.
    pub structure: structure::StructureSummary,
    /// True when entropy is high enough that compression should usually be
    /// skipped unless a downstream policy says otherwise.
    pub skip_compression_candidate: bool,
    /// True when text-like byte classes dominate.
    pub text_like: bool,
    /// True when sparse/zero bytes dominate.
    pub sparse_like: bool,
    /// True when a small byte palette dominates.
    pub low_cardinality: bool,
}

/// Conservative representation hint derived from selector signals.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RepresentationHint {
    /// Empty input.
    Empty,
    /// Sparse/zero-dominated content.
    Sparse,
    /// Text-like content.
    Text,
    /// Small-palette or repeated content.
    LowCardinality,
    /// High-entropy content where raw storage or compression skip is plausible.
    HighEntropy,
    /// General binary or mixed content.
    Binary,
}

/// Computes selector signals for a byte slice.
#[must_use]
pub fn signals(bytes: &[u8]) -> SelectorSignals {
    let fingerprint = fingerprint::extent(bytes);
    let structure = structure::summarize(bytes);
    signals_from_parts(fingerprint, structure)
}

/// Builds selector signals from precomputed primitive summaries.
///
/// This lets callers fuse their own passes and avoid recomputing fingerprint or
/// structure summaries when those values are already available.
#[must_use]
pub fn signals_from_parts(
    fingerprint: fingerprint::ExtentFingerprint,
    structure: structure::StructureSummary,
) -> SelectorSignals {
    let skip_compression_candidate = fingerprint.h1 >= 7.90 && structure.runs.run_fraction() < 0.01;
    let text_like = structure.is_text_like();
    let sparse_like = structure.is_sparse_like();
    let low_cardinality = structure.is_low_cardinality();

    SelectorSignals {
        fingerprint,
        structure,
        skip_compression_candidate,
        text_like,
        sparse_like,
        low_cardinality,
    }
}

/// Computes a conservative representation hint.
#[must_use]
pub fn hint(bytes: &[u8]) -> RepresentationHint {
    if bytes.is_empty() {
        return RepresentationHint::Empty;
    }

    let signals = signals(bytes);
    if signals.sparse_like {
        RepresentationHint::Sparse
    } else if signals.text_like {
        RepresentationHint::Text
    } else if signals.low_cardinality || signals.structure.repeated_period.is_some() {
        RepresentationHint::LowCardinality
    } else if signals.skip_compression_candidate {
        RepresentationHint::HighEntropy
    } else {
        RepresentationHint::Binary
    }
}

#[cfg(test)]
mod tests {
    use super::{RepresentationHint, hint, signals};
    // `Vec` is not in the no-std prelude; alias it from `alloc` for the
    // alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    #[test]
    fn hints_sparse_text_and_binary() {
        assert_eq!(hint(&[]), RepresentationHint::Empty);
        assert_eq!(hint(&[0_u8; 4096]), RepresentationHint::Sparse);
        assert_eq!(
            hint(b"hello tokenfs\nhello tokenfs\n"),
            RepresentationHint::Text
        );
    }

    #[test]
    fn high_entropy_signal_is_set_for_prng_data() {
        let bytes = (0..65_536)
            .scan(0x515e_1ec7_u64, |state, _| {
                *state ^= *state >> 12;
                *state ^= *state << 25;
                *state ^= *state >> 27;
                Some(state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8)
            })
            .collect::<Vec<_>>();
        let signals = signals(&bytes);
        assert!(signals.fingerprint.h1 > 7.9);
        assert!(signals.skip_compression_candidate);
    }
}

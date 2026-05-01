//! Exact Shannon entropy over byte n-grams.
//!
//! `H2` through `H8` use exact n-gram counts and therefore require `std` or
//! `alloc`. For allocation-free high-cardinality summaries, use
//! [`crate::sketch::HashBinSketch`] and the sketch entropy helpers.

#[cfg(any(feature = "std", feature = "alloc"))]
use crate::histogram::ngram::NGramHistogram;

/// Computes exact Shannon entropy in bits per observed n-gram.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h<const N: usize>(bytes: &[u8]) -> f32 {
    let histogram = NGramHistogram::<N>::from_bytes(bytes);
    h_from_histogram(&histogram)
}

/// Computes exact Shannon entropy from an n-gram histogram.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h_from_histogram<const N: usize>(histogram: &NGramHistogram<N>) -> f32 {
    let total = histogram.observations();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let mut entropy = 0.0_f64;
    for (_, count) in histogram.iter() {
        let p = count as f64 / total_f;
        entropy -= p * p.log2();
    }
    entropy as f32
}

/// Exact bigram entropy.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h2(bytes: &[u8]) -> f32 {
    h::<2>(bytes)
}

/// Exact trigram entropy.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h3(bytes: &[u8]) -> f32 {
    h::<3>(bytes)
}

/// Exact 4-gram entropy.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h4(bytes: &[u8]) -> f32 {
    h::<4>(bytes)
}

/// Exact 5-gram entropy.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h5(bytes: &[u8]) -> f32 {
    h::<5>(bytes)
}

/// Exact 6-gram entropy.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h6(bytes: &[u8]) -> f32 {
    h::<6>(bytes)
}

/// Exact 7-gram entropy.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h7(bytes: &[u8]) -> f32 {
    h::<7>(bytes)
}

/// Exact 8-gram entropy.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn h8(bytes: &[u8]) -> f32 {
    h::<8>(bytes)
}

#[cfg(test)]
mod tests {
    use super::{h2, h3, h4, h8};

    #[test]
    fn repeated_ngram_entropy_is_zero() {
        assert_eq!(h2(b"aaaaaaaa"), 0.0);
        assert_eq!(h3(b"aaaaaaaa"), 0.0);
    }

    #[test]
    fn two_equally_likely_bigrams_have_one_bit() {
        let entropy = h2(b"abababa");
        assert!((entropy - 1.0).abs() < 1e-6, "h2={entropy}");
    }

    #[test]
    fn unsupported_or_too_short_inputs_are_zero() {
        assert_eq!(h4(b"abc"), 0.0);
        assert_eq!(h8(b""), 0.0);
    }
}

//! Substring and pattern-matching primitives.
//!
//! This module bundles a family of substring and multi-pattern search
//! algorithms:
//!
//! - [`bitap`] — Bitap (Wu-Manber) shift-bitmap matcher for short needles
//!   (`Bitap16`: ≤16 bytes, `Bitap64`: ≤64 bytes). Pure scalar.
//! - [`two_way`] — Crochemore-Perrin two-way matcher. General `O(n+m)` time
//!   with `O(1)` extra space.
//! - [`rabin_karp`] — Rolling-hash matcher with low init cost.
//! - [`shift_or`] — Branchless bitwise pattern matcher for needles ≤64 bytes.
//! - [`packed_dfa`] — Aho-Corasick-style multi-pattern DFA with byte-class
//!   alphabet compression. Requires `alloc`.
//! - [`packed_pair`] — memchr's hybrid 2-byte pre-filter with verification.
//!   AVX2 fast path on x86_64 with a portable scalar fallback.
//!
//! All algorithms are `no_std` compatible. The packed DFA additionally
//! requires `alloc` for its state table.

pub mod bitap;
pub mod packed_pair;
pub mod rabin_karp;
pub mod shift_or;
pub mod two_way;

#[cfg(any(feature = "std", feature = "alloc"))]
pub mod packed_dfa;

#[cfg(test)]
mod cross_check_tests {
    extern crate alloc;

    use alloc::vec::Vec;
    use proptest::prelude::*;

    use super::bitap::{Bitap16, Bitap64};
    use super::packed_pair::PackedPair;
    use super::rabin_karp::RabinKarp;
    use super::shift_or::ShiftOr;
    use super::two_way::TwoWay;

    fn naive_find(needle: &[u8], haystack: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if haystack.len() < needle.len() {
            return None;
        }
        for i in 0..=haystack.len() - needle.len() {
            if &haystack[i..i + needle.len()] == needle {
                return Some(i);
            }
        }
        None
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 256,
            ..ProptestConfig::default()
        })]

        /// All five single-pattern algorithms must agree with the naive
        /// reference for every (needle, haystack) pair when the needle is
        /// short enough that all five accept it (1..=16 bytes; PackedPair
        /// also requires len >= 2).
        #[test]
        fn all_short_needle_algos_agree(
            needle in proptest::collection::vec(any::<u8>(), 1..=16),
            haystack in proptest::collection::vec(any::<u8>(), 0..=512),
        ) {
            let expected = naive_find(&needle, &haystack);

            let bitap = Bitap16::new(&needle).expect("≤16 bytes accepted");
            prop_assert_eq!(bitap.find(&haystack), expected, "Bitap16 disagrees");

            let bitap64 = Bitap64::new(&needle).expect("≤64 bytes accepted");
            prop_assert_eq!(bitap64.find(&haystack), expected, "Bitap64 disagrees");

            let so = ShiftOr::new(&needle).expect("≤64 bytes accepted");
            prop_assert_eq!(so.find(&haystack), expected, "ShiftOr disagrees");

            let tw = TwoWay::new(&needle);
            prop_assert_eq!(tw.find(&haystack), expected, "TwoWay disagrees");

            let rk = RabinKarp::new(&needle);
            prop_assert_eq!(rk.find(&haystack), expected, "RabinKarp disagrees");

            if needle.len() >= 2 {
                let pp = PackedPair::new(&needle).expect("len >= 2");
                prop_assert_eq!(pp.find(&haystack), expected, "PackedPair disagrees");
            }
        }

        /// Long needles (17..=64 bytes): Bitap16 is excluded; the rest
        /// must still agree with the naive reference.
        #[test]
        fn long_needle_algos_agree(
            needle in proptest::collection::vec(any::<u8>(), 17..=64),
            haystack in proptest::collection::vec(any::<u8>(), 0..=512),
        ) {
            let expected = naive_find(&needle, &haystack);

            let bitap64 = Bitap64::new(&needle).expect("≤64 bytes accepted");
            prop_assert_eq!(bitap64.find(&haystack), expected, "Bitap64 disagrees");

            let so = ShiftOr::new(&needle).expect("≤64 bytes accepted");
            prop_assert_eq!(so.find(&haystack), expected, "ShiftOr disagrees");

            let tw = TwoWay::new(&needle);
            prop_assert_eq!(tw.find(&haystack), expected, "TwoWay disagrees");

            let rk = RabinKarp::new(&needle);
            prop_assert_eq!(rk.find(&haystack), expected, "RabinKarp disagrees");

            let pp = PackedPair::new(&needle).expect("len >= 2");
            prop_assert_eq!(pp.find(&haystack), expected, "PackedPair disagrees");
        }

        /// `find_iter` for both Bitap and Two-Way reports the same set
        /// of starting positions (the naive enumeration).
        #[test]
        fn find_iter_agrees_with_naive(
            needle in proptest::collection::vec(any::<u8>(), 1..=8),
            haystack in proptest::collection::vec(any::<u8>(), 0..=128),
        ) {
            let expected: Vec<usize> = if needle.is_empty() || haystack.len() < needle.len() {
                Vec::new()
            } else {
                (0..=haystack.len() - needle.len())
                    .filter(|&i| &haystack[i..i + needle.len()] == needle.as_slice())
                    .collect()
            };

            let bitap = Bitap16::new(&needle).expect("≤16 bytes");
            let actual: Vec<usize> = bitap.find_iter(&haystack).collect();
            prop_assert_eq!(&actual, &expected, "Bitap16 find_iter disagrees");

            let tw = TwoWay::new(&needle);
            let actual: Vec<usize> = tw.find_iter(&haystack).collect();
            prop_assert_eq!(&actual, &expected, "TwoWay find_iter disagrees");
        }
    }
}

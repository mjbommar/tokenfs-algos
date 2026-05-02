//! SimHash signatures for cosine-similarity estimation over weighted feature
//! vectors.
//!
//! For each feature `(token_hash, weight)`:
//!   - examine the bits of `token_hash`;
//!   - add `weight` to the per-bit accumulator when the bit is `1`;
//!   - subtract `weight` when the bit is `0`.
//!
//! The signature is the sign vector of the per-bit accumulators (1 bit per
//! position, packed into a `u64`). Hamming distance between two signatures
//! is a strong proxy for angular distance between the original weighted
//! vectors. Reference: Charikar (2002), "Similarity estimation techniques
//! from rounding algorithms".
//!
//! Width is fixed at 64 bits — that's the standard size used in production
//! near-duplicate systems (Google, ssdeep variants) and matches one machine
//! word exactly. A 128-bit / `u128` variant is straightforward to add later
//! if needed.

use crate::hash::mix64;

/// 64-bit SimHash signature.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct Signature64 {
    bits: u64,
}

impl Signature64 {
    /// Returns the underlying 64-bit signature.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.bits
    }

    /// Wraps a raw 64-bit signature value (e.g. when reading from storage).
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self { bits }
    }

    /// Hamming distance to another 64-bit signature.
    #[must_use]
    pub const fn hamming_distance(self, other: Self) -> u32 {
        (self.bits ^ other.bits).count_ones()
    }

    /// Estimated cosine similarity. Maps Hamming distance `d` over 64 bits
    /// to `cos(pi * d / 64)`. Charikar's proof bounds this estimate.
    #[must_use]
    pub fn estimated_cosine(self, other: Self) -> f64 {
        let d = self.hamming_distance(other);
        let theta = core::f64::consts::PI * (d as f64) / 64.0;
        crate::math::cos_f64(theta)
    }
}

/// Builds a 64-bit SimHash from `(feature_hash, weight)` pairs.
///
/// Weights are `i64` so callers can express positive (present) and negative
/// (anti-correlated) features. The accumulator uses i128 internally to
/// avoid overflow on long feature streams.
#[must_use]
pub fn from_weighted_hashes<I>(features: I) -> Signature64
where
    I: IntoIterator<Item = (u64, i64)>,
{
    let mut accumulators = [0_i128; 64];
    for (hash, weight) in features {
        let weight = i128::from(weight);
        for (bit, acc) in accumulators.iter_mut().enumerate() {
            if hash & (1_u64 << bit) != 0 {
                *acc += weight;
            } else {
                *acc -= weight;
            }
        }
    }
    let mut bits = 0_u64;
    for (bit, &acc) in accumulators.iter().enumerate() {
        if acc > 0 {
            bits |= 1_u64 << bit;
        }
    }
    Signature64 { bits }
}

/// Builds a 64-bit SimHash from raw byte features, each contributing weight 1.
///
/// Each item is hashed via [`crate::hash::mix64`] with `seed`, then folded
/// into the accumulator. This is the right shape for n-gram features (e.g.
/// 4-byte windows from a byte stream) where every gram counts once.
#[must_use]
pub fn from_unweighted_bytes<'a, I>(items: I, seed: u64) -> Signature64
where
    I: IntoIterator<Item = &'a [u8]>,
{
    from_weighted_hashes(items.into_iter().map(|b| (mix64(b, seed), 1_i64)))
}

/// Builds a 64-bit SimHash from `(feature_bytes, weight)` pairs.
#[must_use]
pub fn from_weighted_bytes<'a, I>(features: I, seed: u64) -> Signature64
where
    I: IntoIterator<Item = (&'a [u8], i64)>,
{
    from_weighted_hashes(features.into_iter().map(|(b, w)| (mix64(b, seed), w)))
}

#[cfg(any(feature = "std", feature = "alloc"))]
mod table {
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;

    use super::Signature64;
    use crate::similarity::kernels_gather;

    /// Number of bits in this SimHash signature (matches `Signature64`).
    pub const BITS: usize = kernels_gather::simhash::BITS;

    /// Builds a per-byte SimHash contribution table from one seed per
    /// bit. See [`kernels_gather::simhash`] for the table layout and
    /// state footprint.
    #[must_use]
    pub fn build_byte_table_from_seeds(seeds: &[u64; BITS]) -> Box<kernels_gather::simhash::Table> {
        kernels_gather::simhash::build_table_from_seeds(seeds)
    }

    /// Builds a 64-bit SimHash from a byte slice using the
    /// runtime-dispatched gather kernel.
    ///
    /// Each byte contributes one ±1 to every accumulator lane via the
    /// precomputed table. This is **not** bit-equivalent to
    /// [`super::from_unweighted_bytes`], which streams whole inputs
    /// through `mix64`. Two callers using the same seeds via either
    /// the scalar table-based path or the dispatched gather path
    /// produce **bit-identical** signatures.
    #[must_use]
    pub fn from_bytes_table(bytes: &[u8], table: &kernels_gather::simhash::Table) -> Signature64 {
        let mut acc = [0_i32; BITS];
        kernels_gather::simhash::update_accumulator_auto(bytes, table, &mut acc);
        Signature64::from_bits(kernels_gather::simhash::finalize(&acc))
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
pub use table::{BITS as TABLE_BITS, build_byte_table_from_seeds, from_bytes_table};

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn empty_input_is_zero_signature() {
        let sig = from_weighted_hashes(core::iter::empty());
        assert_eq!(sig.bits(), 0);
        assert_eq!(sig.hamming_distance(Signature64::default()), 0);
    }

    #[test]
    fn identical_inputs_produce_identical_signatures() {
        let features: Vec<(u64, i64)> = (0..1000_u64)
            .map(|i| (i.wrapping_mul(0x9E37_79B9), 1))
            .collect();
        let s1 = from_weighted_hashes(features.iter().copied());
        let s2 = from_weighted_hashes(features.iter().copied());
        assert_eq!(s1, s2);
    }

    #[test]
    fn order_does_not_matter() {
        let mut features: Vec<(u64, i64)> = (0..1000_u64)
            .map(|i| (i.wrapping_mul(0x1234_5678), 2))
            .collect();
        let s1 = from_weighted_hashes(features.iter().copied());
        features.reverse();
        let s2 = from_weighted_hashes(features.iter().copied());
        assert_eq!(s1, s2);
    }

    #[test]
    fn near_duplicate_inputs_have_low_hamming_distance() {
        // 100 identical features + 5 unique to each side.
        let common: Vec<(u64, i64)> = (0..100_u64)
            .map(|i| (crate::hash::mix_word(i), 1))
            .collect();
        let mut a = common.clone();
        a.extend((1000..1005_u64).map(|i| (crate::hash::mix_word(i), 1)));
        let mut b = common.clone();
        b.extend((2000..2005_u64).map(|i| (crate::hash::mix_word(i), 1)));
        let sa = from_weighted_hashes(a.iter().copied());
        let sb = from_weighted_hashes(b.iter().copied());
        let d = sa.hamming_distance(sb);
        assert!(
            d < 16,
            "near-duplicate Hamming distance too high: {d} (want <16)"
        );
    }

    #[test]
    fn unrelated_inputs_have_high_hamming_distance() {
        let a: Vec<(u64, i64)> = (0..500_u64)
            .map(|i| (crate::hash::mix_word(i), 1))
            .collect();
        let b: Vec<(u64, i64)> = (10_000..10_500_u64)
            .map(|i| (crate::hash::mix_word(i), 1))
            .collect();
        let sa = from_weighted_hashes(a.iter().copied());
        let sb = from_weighted_hashes(b.iter().copied());
        let d = sa.hamming_distance(sb);
        // Random unrelated 64-bit signatures average 32 bits apart with
        // standard deviation ~4. We expect significantly above 16 here.
        assert!(d > 16, "unrelated Hamming distance unexpectedly low: {d}");
    }

    #[test]
    fn from_unweighted_bytes_is_deterministic() {
        let items: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta"];
        let s1 = from_unweighted_bytes(items.iter().copied(), 0x42);
        let s2 = from_unweighted_bytes(items.iter().copied(), 0x42);
        assert_eq!(s1, s2);
    }

    #[test]
    fn estimated_cosine_is_one_for_identical_signatures() {
        let s = Signature64::from_bits(0xDEADBEEF_CAFEF00D);
        let cos = s.estimated_cosine(s);
        assert!((cos - 1.0).abs() < 1e-9, "cos(0)={cos}");
    }

    #[test]
    fn estimated_cosine_is_minus_one_for_inverted_signatures() {
        let s = Signature64::from_bits(0xAAAA_AAAA_AAAA_AAAA);
        let inverted = Signature64::from_bits(!s.bits());
        let cos = s.estimated_cosine(inverted);
        // Hamming = 64, theta = pi, cos(pi) = -1.
        assert!((cos - (-1.0)).abs() < 1e-9, "cos(pi)={cos}");
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn from_bytes_table_matches_per_byte_reference() {
        let seeds: [u64; super::TABLE_BITS] =
            core::array::from_fn(|i| 0x123_4567_89AB_CDEF_u64.wrapping_mul((i as u64) + 1));
        let table = super::build_byte_table_from_seeds(&seeds);

        let payload = b"the quick brown fox jumps over the lazy dog 0123456789";
        let actual = super::from_bytes_table(payload, &table);

        // Hand-compute reference.
        let mut acc = [0_i32; super::TABLE_BITS];
        for &b in payload {
            for i in 0..super::TABLE_BITS {
                let h = crate::hash::mix_word((b as u64) ^ seeds[i]);
                acc[i] += if h & 1 == 1 { 1 } else { -1 };
            }
        }
        let mut expected_bits = 0_u64;
        for (i, &a) in acc.iter().enumerate() {
            if a > 0 {
                expected_bits |= 1_u64 << i;
            }
        }
        assert_eq!(actual.bits(), expected_bits);
    }

    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn from_bytes_table_is_deterministic() {
        let seeds: [u64; super::TABLE_BITS] = core::array::from_fn(|i| (i as u64) * 7);
        let table = super::build_byte_table_from_seeds(&seeds);
        let payload = b"hello, world!";
        let s1 = super::from_bytes_table(payload, &table);
        let s2 = super::from_bytes_table(payload, &table);
        assert_eq!(s1, s2);
    }
}

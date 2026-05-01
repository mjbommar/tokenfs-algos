//! In-memory LSH (Locality-Sensitive Hashing) indexes over MinHash and
//! SimHash signatures.
//!
//! Two index families:
//!
//! - **Banded MinHash LSH** ([`MinHashIndex`]): partitions a `Signature<K>`
//!   into `B` bands of `R = K / B` rows each; two signatures collide in a
//!   band iff all `R` slots in that band agree. Probability of collision
//!   for two sets with Jaccard similarity `s` is `1 - (1 - s^R)^B`. Tuning
//!   `(B, R)` shifts the S-curve to favor recall vs. precision.
//!
//! - **Hamming-radius SimHash buckets** ([`SimHashIndex`]): groups 64-bit
//!   SimHash signatures by a fixed-width prefix and answers radius queries
//!   over Hamming distance. The default uses signature equality (radius 0)
//!   plus per-band rotation; callers wanting larger radii should issue
//!   multiple queries with bit-flips applied to the seed signature.
//!
//! Both indexes report `candidate_reduction()` so callers can quantify the
//! pre-filter speedup vs. exhaustive comparison. Quality fixtures in tests
//! verify recall / precision over controlled overlap.
//!
//! Memory model: indexes hold owned `Vec` storage and are gated on the
//! `std` feature (since the audit calls out alloc-bearing structures
//! belong to a separate crate eventually). The static fixed-array
//! signature primitives in [`super::minhash`] / [`super::simhash`] remain
//! `no_std` + alloc-free.

#![cfg(feature = "std")]

use std::collections::HashMap;
use std::hash::Hash;

use super::minhash::Signature as MinHashSignature;
use super::simhash::Signature64;

/// Per-query report.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub struct QueryStats {
    /// Number of items inserted into the index.
    pub corpus_size: usize,
    /// Number of items returned as candidates (before any rerank).
    pub candidates: usize,
}

impl QueryStats {
    /// Fraction of corpus eliminated by the LSH prefilter, in `[0, 1]`.
    /// `1.0` means everything was eliminated; `0.0` means the index gave
    /// no reduction over a full scan.
    #[must_use]
    pub fn candidate_reduction(self) -> f64 {
        if self.corpus_size == 0 {
            return 0.0;
        }
        1.0 - (self.candidates as f64 / self.corpus_size as f64)
    }
}

/// Banded MinHash LSH index over fixed-size MinHash signatures.
///
/// Construction parameters `(B, R)` partition the `K`-slot signature into
/// `B` bands of `R` slots each (`K` MUST equal `B * R` — checked in
/// [`MinHashIndex::new`]).
///
/// The S-curve `1 - (1 - s^R)^B` peaks at the threshold `s* ≈ (1/B)^(1/R)`:
/// pairs with Jaccard ≥ `s*` are very likely to be returned; pairs below
/// are very unlikely. Common tunings:
///
/// | B | R | Threshold s* | Bandwidth (signature bytes per band) |
/// |---|---|--------------|-------------------------------------|
/// | 8 | 16 | ~0.74 | 128 |
/// | 16 | 8 | ~0.69 | 64  |
/// | 32 | 4 | ~0.58 | 32  |
/// | 64 | 2 | ~0.40 | 16  |
///
/// Lower `s*` means broader recall and more false positives.
pub struct MinHashIndex<Id, const K: usize> {
    bands: Vec<HashMap<u64, Vec<Id>>>,
    rows_per_band: usize,
    corpus_size: usize,
}

impl<Id: Clone + Eq + Hash, const K: usize> MinHashIndex<Id, K> {
    /// Builds an empty index with `bands * rows_per_band == K`.
    ///
    /// # Panics
    ///
    /// Panics if `bands == 0` or `bands * rows_per_band != K`.
    #[must_use]
    pub fn new(bands: usize, rows_per_band: usize) -> Self {
        assert!(bands > 0, "bands must be > 0");
        assert!(rows_per_band > 0, "rows_per_band must be > 0");
        assert_eq!(
            bands * rows_per_band,
            K,
            "bands * rows_per_band must equal K"
        );
        let bands = (0..bands).map(|_| HashMap::new()).collect();
        Self {
            bands,
            rows_per_band,
            corpus_size: 0,
        }
    }

    /// Number of items inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.corpus_size
    }

    /// True when no items have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.corpus_size == 0
    }

    /// Inserts an item under its MinHash signature.
    pub fn insert(&mut self, id: Id, signature: &MinHashSignature<K>) {
        let slots = signature.slots();
        for (band_idx, bucket) in self.bands.iter_mut().enumerate() {
            let band_hash = hash_band(slots, band_idx, self.rows_per_band);
            bucket.entry(band_hash).or_default().push(id.clone());
        }
        self.corpus_size += 1;
    }

    /// Queries for candidate matches. Returns the deduplicated set of ids
    /// that collide with `query` in at least one band, plus a [`QueryStats`].
    #[must_use]
    pub fn query(&self, query: &MinHashSignature<K>) -> (Vec<Id>, QueryStats) {
        let slots = query.slots();
        // Use a HashMap-as-set for deduplication; preserves insertion order
        // is not promised by the contract.
        let mut seen: HashMap<Id, ()> = HashMap::new();
        for (band_idx, bucket) in self.bands.iter().enumerate() {
            let band_hash = hash_band(slots, band_idx, self.rows_per_band);
            if let Some(ids) = bucket.get(&band_hash) {
                for id in ids {
                    seen.insert(id.clone(), ());
                }
            }
        }
        let candidates: Vec<Id> = seen.into_keys().collect();
        let stats = QueryStats {
            corpus_size: self.corpus_size,
            candidates: candidates.len(),
        };
        (candidates, stats)
    }
}

fn hash_band(slots: &[u64], band_idx: usize, rows_per_band: usize) -> u64 {
    // Build a 64-bit hash for a band by FNV-folding the rows. Independent
    // of crate::hash to keep the LSH layer self-contained.
    let start = band_idx * rows_per_band;
    let end = start + rows_per_band;
    let mut acc = 0xcbf2_9ce4_8422_2325_u64; // FNV offset basis.
    for &slot in &slots[start..end] {
        acc = acc.wrapping_mul(0x100_0000_01b3); // FNV prime.
        acc ^= slot;
        // Final SplitMix step so all 64 bits matter.
        acc = crate::hash::mix_word(acc);
    }
    acc
}

/// Hamming-radius SimHash bucket index.
///
/// Splits the 64-bit signature into `BANDS` equal-width segments. Two
/// signatures will collide in at least one band if their Hamming distance
/// is at most `BANDS - 1` — this is the standard "indexing for similarity
/// search in high dimensions" trick (Manku-Jain-Sarma 2007). For example
/// with `BANDS = 8` the index reliably finds matches within Hamming
/// distance 7 with one band-equality check per query.
pub struct SimHashIndex<Id, const BANDS: usize> {
    bands: [HashMap<u64, Vec<Id>>; BANDS],
    band_width_bits: usize,
    corpus_size: usize,
}

impl<Id: Clone + Eq + Hash, const BANDS: usize> SimHashIndex<Id, BANDS> {
    /// Builds an empty index with `BANDS` partitions of the 64-bit signature.
    ///
    /// # Panics
    ///
    /// Panics if `BANDS == 0` or `BANDS > 64`. Recommended: 4, 8, or 16
    /// (corresponding to 16-, 8-, and 4-bit band widths).
    #[must_use]
    pub fn new() -> Self {
        assert!(BANDS > 0, "BANDS must be > 0");
        assert!(BANDS <= 64, "BANDS must be <= 64");
        let band_width_bits = 64 / BANDS;
        // [HashMap<u64, Vec<Id>>; BANDS] without Copy bound on Id requires
        // initialization via std::array::from_fn.
        let bands = std::array::from_fn(|_| HashMap::new());
        Self {
            bands,
            band_width_bits,
            corpus_size: 0,
        }
    }

    /// Number of items inserted.
    #[must_use]
    pub fn len(&self) -> usize {
        self.corpus_size
    }

    /// True when no items have been inserted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.corpus_size == 0
    }

    /// Inserts an item under its SimHash signature.
    pub fn insert(&mut self, id: Id, signature: Signature64) {
        let bits = signature.bits();
        for (band_idx, bucket) in self.bands.iter_mut().enumerate() {
            let key = extract_band(bits, band_idx, self.band_width_bits);
            bucket.entry(key).or_default().push(id.clone());
        }
        self.corpus_size += 1;
    }

    /// Queries for candidate matches that share at least one full band with
    /// `query`. Two signatures with Hamming distance <= `BANDS - 1` are
    /// guaranteed to share at least one band by pigeonhole (Manku-Jain-Sarma).
    #[must_use]
    pub fn query(&self, query: Signature64) -> (Vec<Id>, QueryStats) {
        let bits = query.bits();
        let mut seen: HashMap<Id, ()> = HashMap::new();
        for (band_idx, bucket) in self.bands.iter().enumerate() {
            let key = extract_band(bits, band_idx, self.band_width_bits);
            if let Some(ids) = bucket.get(&key) {
                for id in ids {
                    seen.insert(id.clone(), ());
                }
            }
        }
        let candidates: Vec<Id> = seen.into_keys().collect();
        let stats = QueryStats {
            corpus_size: self.corpus_size,
            candidates: candidates.len(),
        };
        (candidates, stats)
    }
}

impl<Id: Clone + Eq + Hash, const BANDS: usize> Default for SimHashIndex<Id, BANDS> {
    fn default() -> Self {
        Self::new()
    }
}

fn extract_band(bits: u64, band_idx: usize, width: usize) -> u64 {
    let shift = band_idx * width;
    let mask = if width >= 64 {
        u64::MAX
    } else {
        (1_u64 << width) - 1
    };
    (bits >> shift) & mask
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::similarity::minhash::{classic_from_hashes, jaccard_similarity};
    use crate::similarity::simhash::from_weighted_hashes;

    fn elements(items: &[u32]) -> impl Iterator<Item = u64> + '_ {
        items.iter().map(|&x| crate::hash::mix_word(u64::from(x)))
    }

    #[test]
    fn minhash_index_finds_known_overlap_pair() {
        // High-overlap pair (Jaccard ≈ 0.82): 0..200 vs 20..220.
        let high_a: Vec<u32> = (0..200).collect();
        let high_b: Vec<u32> = (20..220).collect();
        // Disjoint distractor set.
        let disjoint: Vec<u32> = (10_000..10_200).collect();

        const K: usize = 128;
        let sig_a = classic_from_hashes::<_, K>(elements(&high_a), 0xCAFE);
        let sig_b = classic_from_hashes::<_, K>(elements(&high_b), 0xCAFE);
        let sig_d = classic_from_hashes::<_, K>(elements(&disjoint), 0xCAFE);

        // 16 bands of 8 rows each: threshold ~0.69.
        let mut idx = MinHashIndex::<&'static str, K>::new(16, 8);
        idx.insert("b", &sig_b);
        idx.insert("d", &sig_d);

        let (candidates, stats) = idx.query(&sig_a);
        assert!(
            candidates.contains(&"b"),
            "near-duplicate not found in candidates: {candidates:?}"
        );
        assert_eq!(stats.corpus_size, 2);
    }

    #[test]
    fn minhash_index_eliminates_dissimilar_inserts_at_high_threshold() {
        // 1 high-overlap pair + 50 disjoint inserts with Jaccard ~ 0.
        const K: usize = 128;
        let target_a: Vec<u32> = (0..200).collect();
        let target_b: Vec<u32> = (10..200).chain(200..210).collect();

        let sig_a = classic_from_hashes::<_, K>(elements(&target_a), 0x1234);
        let sig_b = classic_from_hashes::<_, K>(elements(&target_b), 0x1234);

        // 32 bands of 4 rows each: threshold ~0.58 — narrower than the
        // expected high-overlap Jaccard, so target_b should be a candidate
        // while disjoint inserts should not.
        let mut idx = MinHashIndex::<usize, K>::new(32, 4);
        idx.insert(0, &sig_b);
        for i in 1..51 {
            let disjoint: Vec<u32> = (1000 + i * 100..1000 + i * 100 + 200)
                .map(|x| x as u32)
                .collect();
            let sig = classic_from_hashes::<_, K>(elements(&disjoint), 0x1234);
            idx.insert(i, &sig);
        }
        let (candidates, stats) = idx.query(&sig_a);
        assert!(
            candidates.contains(&0),
            "near-duplicate (id=0) not in candidates {candidates:?}"
        );
        // Reduction must be > 50% — most disjoint inserts should be filtered.
        assert!(
            stats.candidate_reduction() > 0.5,
            "stats={stats:?}, expected reduction > 50%"
        );

        // Sanity: confirm the actual Jaccard estimate justifies the filter.
        let est = jaccard_similarity(&sig_a, &sig_b);
        assert!(est > 0.5);
    }

    #[test]
    fn minhash_index_panics_on_invalid_band_split() {
        let result = std::panic::catch_unwind(|| {
            // 3 * 5 = 15 != K=16 → must panic.
            MinHashIndex::<&'static str, 16>::new(3, 5)
        });
        assert!(result.is_err());
    }

    #[test]
    fn simhash_index_finds_near_duplicate_within_radius() {
        // Two signatures differing in 5 bits — should collide in at least
        // one band when BANDS = 8 (Hamming threshold 7).
        let sig_a = from_weighted_hashes((0..200_u64).map(|i| (i.wrapping_mul(0x9E37), 1)));
        let mut bits_b = sig_a.bits();
        // Flip 5 bits at known positions.
        for &pos in &[3_u32, 17, 31, 44, 58] {
            bits_b ^= 1_u64 << pos;
        }
        let sig_b = Signature64::from_bits(bits_b);
        assert_eq!(sig_a.hamming_distance(sig_b), 5);

        let mut idx = SimHashIndex::<&'static str, 8>::new();
        idx.insert("near", sig_b);
        // Add some unrelated entries.
        for i in 0..20 {
            let other =
                from_weighted_hashes((10_000_u64..10_200).map(|x| (x.wrapping_mul(0x1000 + i), 1)));
            idx.insert("other", other);
        }
        let (candidates, _stats) = idx.query(sig_a);
        assert!(
            candidates.contains(&"near"),
            "near-duplicate at Hamming=5 not found"
        );
    }

    #[test]
    fn simhash_index_handles_distant_signatures_per_lsh_contract() {
        // Construct a signature with maximum Hamming distance from sig_a
        // by inverting all bits — distance is exactly 64.
        let sig_a = from_weighted_hashes((0..100_u64).map(|i| (i.wrapping_mul(0x33), 1)));
        let sig_far = Signature64::from_bits(!sig_a.bits());
        assert_eq!(sig_a.hamming_distance(sig_far), 64);

        let mut idx = SimHashIndex::<&'static str, 8>::new();
        idx.insert("far", sig_far);
        let (candidates, stats) = idx.query(sig_a);
        // With BANDS=8, every band has the all-bits-inverted pattern, so
        // no band agreement is possible.
        assert_eq!(
            candidates.len(),
            0,
            "fully-inverted signature should not collide"
        );
        assert_eq!(stats.corpus_size, 1);
    }

    #[test]
    fn query_stats_candidate_reduction_is_correct() {
        let stats = QueryStats {
            corpus_size: 100,
            candidates: 25,
        };
        assert!((stats.candidate_reduction() - 0.75).abs() < 1e-9);

        let zero = QueryStats {
            corpus_size: 0,
            candidates: 0,
        };
        assert_eq!(zero.candidate_reduction(), 0.0);
    }
}

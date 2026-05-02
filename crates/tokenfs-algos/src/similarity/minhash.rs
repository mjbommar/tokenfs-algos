//! MinHash signatures for set-similarity (Jaccard) estimation.
//!
//! `MinHash` answers "how similar are these two sets?" without storing the
//! sets themselves. The signature is `K` smallest hash values across `K`
//! independent hash functions; the fraction of equal positions in two
//! signatures is an unbiased estimator of the Jaccard similarity.
//!
//! Variants implemented:
//!
//! - **Classic K-min MinHash**: `K` independent seeded hashers; signature
//!   stores `min` per hasher. Most established and statistically clean.
//! - **One-permutation MinHash (OPH)**: a single hash function is partitioned
//!   into `K` buckets; the signature is the per-bucket minimum. Cheaper to
//!   build (one hash per element instead of `K`), with documented degraded
//!   accuracy on sparse inputs (see `densified_one_permutation` for the fix).
//! - **b-bit MinHash**: any of the above signatures truncated to the lowest
//!   `b` bits per slot, trading collision probability for compactness.
//!
//! Hashing uses [`crate::hash::mix64`] with per-slot seeds for deterministic
//! reproducibility.

use crate::hash::mix64;
use crate::similarity::kernels_gather;

/// Fixed-size MinHash signature. Each slot holds the minimum observed hash
/// for one of `K` independent hash functions (classic) or `K` partitions
/// (OPH).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Signature<const K: usize> {
    slots: [u64; K],
    /// True for slots that have never been updated. Used by densification.
    populated: [bool; K],
}

impl<const K: usize> Signature<K> {
    /// Empty signature; every slot is unpopulated and set to `u64::MAX`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            slots: [u64::MAX; K],
            populated: [false; K],
        }
    }

    /// Number of slots `K`.
    #[must_use]
    pub const fn len(&self) -> usize {
        K
    }

    /// Returns true when no slot has been populated.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.populated.iter().all(|&p| !p)
    }

    /// Returns the raw signature slice.
    #[must_use]
    pub const fn slots(&self) -> &[u64; K] {
        &self.slots
    }

    /// Returns the slot at `index`. Out-of-range returns `u64::MAX`.
    #[must_use]
    pub fn slot(&self, index: usize) -> u64 {
        if index >= K {
            u64::MAX
        } else {
            self.slots[index]
        }
    }

    /// Truncate every slot to its lowest `b` bits. Used by b-bit MinHash.
    /// `b` is clamped to `1..=64`.
    #[must_use]
    pub fn b_bit(self, b: u32) -> Self {
        let b = b.clamp(1, 64);
        let mask = if b == 64 { u64::MAX } else { (1_u64 << b) - 1 };
        let mut out = Self::new();
        for i in 0..K {
            out.slots[i] = self.slots[i] & mask;
            out.populated[i] = self.populated[i];
        }
        out
    }
}

impl<const K: usize> Default for Signature<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builds a classic K-min MinHash signature from an iterator of `u64`
/// element hashes.
///
/// Each slot `k` is updated with `min(slot_k, mix64_of(element ^ seed_k))`,
/// where `seed_k = base_seed + k as u64`. Iterating elements in any order
/// produces the same signature.
#[must_use]
pub fn classic_from_hashes<I, const K: usize>(elements: I, base_seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = u64>,
{
    let mut sig = Signature::<K>::new();
    for element in elements {
        for k in 0..K {
            // Seed each hasher with base_seed + k; mix the element's hash with
            // that seed via the public mix function used elsewhere.
            let seed_k = base_seed.wrapping_add(k as u64);
            let h = mix_two(element, seed_k);
            if h < sig.slots[k] {
                sig.slots[k] = h;
                sig.populated[k] = true;
            }
        }
    }
    sig
}

/// Builds a classic MinHash signature from raw byte slices.
///
/// Each item's hash is computed via [`crate::hash::mix64`] with `base_seed`
/// before being fed into the signature builder. Useful for n-gram features.
#[must_use]
pub fn classic_from_bytes<'a, I, const K: usize>(items: I, base_seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    classic_from_hashes::<_, K>(
        items.into_iter().map(|bytes| mix64(bytes, base_seed)),
        base_seed,
    )
}

/// Per-byte MinHash signature backed by a precomputed gather table.
///
/// The table-based representation defines an alternative hash family
/// where each input byte contributes `K` independent hash values via a
/// single row of `T : [u8 -> [u64; K]]`. Building the table from seeds
/// is one-shot (`build_byte_table_from_seeds`); per-byte updates are
/// then table loads instead of per-byte hash evaluations.
///
/// State footprint: `K * 256 * 8` bytes — see
/// [`kernels_gather`] for the L1/L2 trade-off discussion.
///
/// **Hash family**: the table-based variant is **not** bit-equivalent
/// to [`classic_from_bytes`], which streams whole inputs through
/// [`mix64`]. It defines its own per-byte family:
/// `h_k(byte) = mix_word(byte ^ seeds[k])`. Two callers that build
/// signatures with the same seeds — one via the scalar table-based
/// path, one via the gather kernels — produce **bit-identical**
/// signatures.
#[must_use]
pub fn build_byte_table_from_seeds<const K: usize>(
    seeds: &[u64; K],
) -> [[u64; K]; kernels_gather::TABLE_ROWS] {
    kernels_gather::build_table_from_seeds(seeds)
}

/// Updates an 8-way `Signature` from a byte slice using the
/// runtime-dispatched gather kernel.
///
/// The signature uses the per-byte hash family defined above. Falls
/// through to the scalar implementation when no SIMD path is
/// available.
pub fn update_bytes_table_8(
    sig: &mut Signature<8>,
    table: &[[u64; 8]; kernels_gather::TABLE_ROWS],
    bytes: &[u8],
) {
    if bytes.is_empty() {
        return;
    }
    kernels_gather::update_minhash_8way_auto(bytes, table, &mut sig.slots);
    for k in 0..8 {
        if sig.slots[k] != u64::MAX {
            sig.populated[k] = true;
        }
    }
}

/// Builds a fresh table-based 8-way MinHash signature from a byte
/// slice and a precomputed gather table. Convenience wrapper over
/// [`update_bytes_table_8`].
#[must_use]
pub fn classic_from_bytes_table_8(
    bytes: &[u8],
    table: &[[u64; 8]; kernels_gather::TABLE_ROWS],
) -> Signature<8> {
    let mut sig = Signature::<8>::new();
    update_bytes_table_8(&mut sig, table, bytes);
    sig
}

/// Streaming K-way table-based MinHash signature builder.
///
/// Wraps a [`Signature<K>`] plus a borrowed gather table so callers can feed
/// bytes incrementally — one byte at a time, or in 4 KiB-ish chunks — and
/// snapshot the running signature at any point. The hash family matches the
/// table-based [`classic_from_bytes_table_8`] / [`update_bytes_table_8`]
/// path: `h_k(byte) = mix_word(byte ^ seeds[k])`.
///
/// This is the right primitive for FUSE-style write paths and for any caller
/// that needs both a "live" Jaccard estimate and a final signature without
/// re-hashing the input.
///
/// ## Memory shape
///
/// The table itself is borrowed (`&[[u64; K]; 256]`); the builder owns only
/// `Signature<K>` (16 bytes per slot plus a populated bitmap) so it stays
/// `Copy`-ish and cheap to keep around per write handle.
///
/// ## Bit-exact across chunkings
///
/// Two streams that consume the same bytes (in any order — the hash family
/// is order-independent) produce identical signatures regardless of chunk
/// boundaries. This is verified by [`tests::stream_chunking_invariant`].
///
/// ## Example
///
/// ```
/// use tokenfs_algos::similarity::minhash::{
///     build_byte_table_from_seeds, IncrementalSignature,
/// };
///
/// let seeds: [u64; 8] = core::array::from_fn(|i| 0x9E37_79B9_u64 ^ i as u64);
/// let table = build_byte_table_from_seeds::<8>(&seeds);
///
/// let mut builder = IncrementalSignature::<8>::new(&table);
/// for chunk in b"abcdef".chunks(2) {
///     builder.update_bytes(chunk);
/// }
/// let sig = builder.finalize();
/// assert!(!sig.is_empty());
/// ```
#[derive(Debug)]
pub struct IncrementalSignature<'a, const K: usize> {
    sig: Signature<K>,
    table: &'a [[u64; K]; kernels_gather::TABLE_ROWS],
}

impl<'a, const K: usize> IncrementalSignature<'a, K> {
    /// Construct an empty incremental signature backed by the given gather
    /// table. The table is borrowed for the lifetime of the builder; build
    /// it once via [`build_byte_table_from_seeds`] and reuse.
    #[must_use]
    pub fn new(table: &'a [[u64; K]; kernels_gather::TABLE_ROWS]) -> Self {
        Self {
            sig: Signature::<K>::new(),
            table,
        }
    }

    /// Construct an incremental signature seeded from an existing
    /// [`Signature<K>`]. Useful for resuming a partial computation that was
    /// snapshotted with [`Self::snapshot`] or merged via [`Self::merge`].
    #[must_use]
    pub fn from_signature(
        table: &'a [[u64; K]; kernels_gather::TABLE_ROWS],
        sig: Signature<K>,
    ) -> Self {
        Self { sig, table }
    }

    /// Feed a single byte into the running K-min update.
    pub fn update_byte(&mut self, byte: u8) {
        let row = &self.table[byte as usize];
        for (k, &h) in row.iter().enumerate() {
            if h < self.sig.slots[k] {
                self.sig.slots[k] = h;
                self.sig.populated[k] = true;
            }
        }
    }

    /// Feed `bytes` into the running K-min update.
    ///
    /// For `K == 8`, dispatches to the runtime-selected gather kernel
    /// (`kernels_gather::update_minhash_8way_auto`). For other `K`, falls
    /// back to the scalar reference. Either way the output matches
    /// [`classic_from_bytes_table_8`] / [`update_bytes_table_8`] for the
    /// concatenated input.
    pub fn update_bytes(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        if K == 8 {
            // Cast through `[u64; K]` -> `[u64; 8]` and `[[u64; K]; 256]`
            // -> `[[u64; 8]; 256]`; both layouts are identical when `K == 8`
            // because const-generic arrays are simply `K * size_of::<T>()`
            // contiguous bytes (Rust 2024 / const generics MSRV).
            // SAFETY: K == 8 so the type representations are byte-identical.
            let table_8: &[[u64; 8]; kernels_gather::TABLE_ROWS] = unsafe {
                &*(self.table as *const [[u64; K]; kernels_gather::TABLE_ROWS])
                    .cast::<[[u64; 8]; kernels_gather::TABLE_ROWS]>()
            };
            // SAFETY: K == 8 so the slot array sizes match.
            let sig_8: &mut [u64; 8] =
                unsafe { &mut *(self.sig.slots.as_mut_ptr().cast::<[u64; 8]>()) };
            kernels_gather::update_minhash_8way_auto(bytes, table_8, sig_8);
            for k in 0..K {
                if self.sig.slots[k] != u64::MAX {
                    self.sig.populated[k] = true;
                }
            }
        } else {
            kernels_gather::update_minhash_scalar::<K>(bytes, self.table, &mut self.sig.slots);
            for k in 0..K {
                if self.sig.slots[k] != u64::MAX {
                    self.sig.populated[k] = true;
                }
            }
        }
    }

    /// Read the current signature mid-stream without consuming the builder.
    ///
    /// Useful for emitting periodic snapshots while the underlying byte
    /// stream is still being written.
    #[must_use]
    pub fn snapshot(&self) -> Signature<K> {
        self.sig
    }

    /// Return the final signature and consume the builder.
    #[must_use]
    pub fn finalize(self) -> Signature<K> {
        self.sig
    }

    /// Reset the running signature to empty. The table reference is
    /// preserved; reuse the builder for the next stream without re-binding.
    pub fn reset(&mut self) {
        self.sig = Signature::<K>::new();
    }

    /// Merge another signature into the running state via per-slot `min`.
    ///
    /// Lets a caller hash disjoint shards of the same input in parallel
    /// (each shard accumulates its own [`IncrementalSignature`], all sharing
    /// the same table) then fold the results: the final merged signature is
    /// identical to one produced by feeding the concatenated input through a
    /// single builder, because the K-min update is associative and
    /// commutative under `min`.
    pub fn merge(&mut self, other: &Signature<K>) {
        for k in 0..K {
            if other.slots[k] < self.sig.slots[k] {
                self.sig.slots[k] = other.slots[k];
            }
            if other.populated[k] {
                self.sig.populated[k] = true;
            }
        }
    }

    /// Number of slots `K`.
    #[must_use]
    pub const fn len(&self) -> usize {
        K
    }

    /// True when the running signature has no populated slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sig.is_empty()
    }
}

/// Builds a one-permutation MinHash (OPH) signature.
///
/// One hash function partitions the universe into `K` equal-sized buckets;
/// each slot stores the minimum hash that landed in its bucket. This is `K`x
/// cheaper to build than [`classic_from_hashes`] — single hash per element
/// instead of `K` — at the cost of accuracy on sparse inputs (slots may stay
/// `u64::MAX`). Use [`densified_one_permutation`] to repair.
#[must_use]
pub fn one_permutation_from_hashes<I, const K: usize>(elements: I, seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = u64>,
{
    assert!(K > 0, "MinHash signature size K must be > 0");
    let mut sig = Signature::<K>::new();
    for element in elements {
        let h = mix_two(element, seed);
        let bucket = (h % K as u64) as usize;
        // Within-bucket score: rotate so the bucket index doesn't dominate.
        let score = h.rotate_right((bucket as u32) & 63);
        if score < sig.slots[bucket] {
            sig.slots[bucket] = score;
            sig.populated[bucket] = true;
        }
    }
    sig
}

/// Builds a one-permutation MinHash signature from raw byte slices.
#[must_use]
pub fn one_permutation_from_bytes<'a, I, const K: usize>(items: I, seed: u64) -> Signature<K>
where
    I: IntoIterator<Item = &'a [u8]>,
{
    one_permutation_from_hashes::<_, K>(items.into_iter().map(|bytes| mix64(bytes, seed)), seed)
}

/// Densified one-permutation MinHash: fills empty slots by borrowing from
/// the nearest populated neighbor (round-robin both directions).
///
/// On sparse inputs OPH leaves many slots at `u64::MAX`. Densification
/// preserves the OPH speed advantage while restoring the unbiased Jaccard
/// estimator. Reference: Shrivastava & Li, "Densifying One-Permutation
/// Hashing via Rotation for Fast Near Neighbor Search" (ICML 2014).
///
/// Implementation here is the simple "rotation" variant: for each empty
/// slot, walk forward (mod K) to the first populated slot. Returns the
/// signature unchanged if it was already fully populated or if no slots are
/// populated at all.
#[must_use]
pub fn densified_one_permutation<const K: usize>(mut sig: Signature<K>) -> Signature<K> {
    if sig.is_empty() {
        return sig;
    }
    if sig.populated.iter().all(|&p| p) {
        return sig;
    }
    // For each empty slot, find the next populated slot mod K and copy.
    for i in 0..K {
        if sig.populated[i] {
            continue;
        }
        for offset in 1..K {
            let j = (i + offset) % K;
            if sig.populated[j] {
                sig.slots[i] = sig.slots[j];
                sig.populated[i] = true;
                break;
            }
        }
    }
    sig
}

/// Estimates the Jaccard similarity between two signatures of the same size.
///
/// Returns the fraction of slots where the two signatures agree. For
/// classic MinHash this is an unbiased estimator of `|A ∩ B| / |A ∪ B|`.
#[must_use]
pub fn jaccard_similarity<const K: usize>(a: &Signature<K>, b: &Signature<K>) -> f64 {
    if K == 0 {
        return 0.0;
    }
    // Slots that are unpopulated in both don't contribute information.
    let mut equal = 0_usize;
    let mut total = 0_usize;
    for i in 0..K {
        match (a.populated[i], b.populated[i]) {
            (false, false) => continue,
            _ => {
                total += 1;
                if a.slots[i] == b.slots[i] {
                    equal += 1;
                }
            }
        }
    }
    if total == 0 {
        return 0.0;
    }
    equal as f64 / total as f64
}

/// Estimates Jaccard similarity from b-bit signatures.
///
/// b-bit MinHash collisions are more frequent than classic, so the raw
/// agreement rate over-estimates similarity. The unbiased estimator is
/// `(observed_match - 1/2^b) / (1 - 1/2^b)`, clamped to `[0, 1]`. See
/// Li & König (2010), "b-Bit Minwise Hashing".
#[must_use]
pub fn b_bit_jaccard_similarity<const K: usize>(
    a: &Signature<K>,
    b: &Signature<K>,
    b_bits: u32,
) -> f64 {
    let raw = jaccard_similarity(a, b);
    let b_bits = b_bits.clamp(1, 63);
    let collision = 1.0 / ((1_u64 << b_bits) as f64);
    let est = (raw - collision) / (1.0 - collision);
    est.clamp(0.0, 1.0)
}

/// Mixes two `u64` values into one. Cheaper than [`mix64`] over bytes.
#[inline]
fn mix_two(a: u64, b: u64) -> u64 {
    crate::hash::mix_word(a ^ crate::hash::mix_word(b.wrapping_add(0x9E37_79B9_7F4A_7C15)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// Helper: hash a Rust set into u64 element hashes for test inputs.
    fn elements(items: &[u32]) -> impl Iterator<Item = u64> + '_ {
        items.iter().map(|&x| crate::hash::mix_word(u64::from(x)))
    }

    #[test]
    fn empty_signature_is_max() {
        let sig: Signature<8> = Signature::new();
        for slot in sig.slots() {
            assert_eq!(*slot, u64::MAX);
        }
        assert!(sig.is_empty());
    }

    #[test]
    fn identical_inputs_produce_identical_signatures() {
        let a = (0..100_u32).collect::<Vec<_>>();
        let s1 = classic_from_hashes::<_, 64>(elements(&a), 0xCAFE);
        let s2 = classic_from_hashes::<_, 64>(elements(&a), 0xCAFE);
        assert_eq!(s1, s2);
    }

    #[test]
    fn order_does_not_matter() {
        let mut a = (0..100_u32).collect::<Vec<_>>();
        let s1 = classic_from_hashes::<_, 64>(elements(&a), 0x1234);
        a.reverse();
        let s2 = classic_from_hashes::<_, 64>(elements(&a), 0x1234);
        assert_eq!(s1, s2);
    }

    #[test]
    fn jaccard_estimate_is_close_for_known_overlap() {
        // Two sets with known Jaccard similarity 0.5: {0..200} and {100..300}.
        // Intersection 100, union 300, Jaccard = 1/3.
        let a: Vec<u32> = (0..200).collect();
        let b: Vec<u32> = (100..300).collect();
        let sa = classic_from_hashes::<_, 256>(elements(&a), 0xABCD);
        let sb = classic_from_hashes::<_, 256>(elements(&b), 0xABCD);
        let est = jaccard_similarity(&sa, &sb);
        let expected = 1.0 / 3.0;
        assert!(
            (est - expected).abs() < 0.10,
            "est={est}, expected≈{expected}"
        );
    }

    #[test]
    fn jaccard_estimate_is_close_for_high_overlap() {
        // |A| = 200, |B| = 200, intersection 180, union 220, Jaccard = 180/220.
        let a: Vec<u32> = (0..200).collect();
        let b: Vec<u32> = (20..220).collect();
        let sa = classic_from_hashes::<_, 512>(elements(&a), 0x99);
        let sb = classic_from_hashes::<_, 512>(elements(&b), 0x99);
        let est = jaccard_similarity(&sa, &sb);
        let expected = 180.0 / 220.0;
        assert!(
            (est - expected).abs() < 0.10,
            "est={est}, expected≈{expected}"
        );
    }

    #[test]
    fn jaccard_estimate_is_close_for_disjoint_sets() {
        let a: Vec<u32> = (0..100).collect();
        let b: Vec<u32> = (1000..1100).collect();
        let sa = classic_from_hashes::<_, 256>(elements(&a), 0xDDDD);
        let sb = classic_from_hashes::<_, 256>(elements(&b), 0xDDDD);
        let est = jaccard_similarity(&sa, &sb);
        assert!(est < 0.10, "disjoint est={est}");
    }

    #[test]
    fn one_permutation_with_densification_recovers_jaccard() {
        let a: Vec<u32> = (0..200).collect();
        let b: Vec<u32> = (100..300).collect();
        let sa =
            densified_one_permutation(one_permutation_from_hashes::<_, 256>(elements(&a), 0xBEEF));
        let sb =
            densified_one_permutation(one_permutation_from_hashes::<_, 256>(elements(&b), 0xBEEF));
        let est = jaccard_similarity(&sa, &sb);
        let expected = 1.0 / 3.0;
        assert!(
            (est - expected).abs() < 0.15,
            "OPH est={est}, expected≈{expected}"
        );
    }

    #[test]
    fn b_bit_jaccard_close_to_classic() {
        let a: Vec<u32> = (0..500).collect();
        let b: Vec<u32> = (250..750).collect();
        let sa_full = classic_from_hashes::<_, 512>(elements(&a), 0x4242);
        let sb_full = classic_from_hashes::<_, 512>(elements(&b), 0x4242);
        let full_est = jaccard_similarity(&sa_full, &sb_full);

        let sa = sa_full.b_bit(8);
        let sb = sb_full.b_bit(8);
        let bbit_est = b_bit_jaccard_similarity(&sa, &sb, 8);

        assert!(
            (full_est - bbit_est).abs() < 0.05,
            "b-bit est={bbit_est} drifted from full est={full_est}"
        );
    }

    #[test]
    fn classic_from_bytes_matches_classic_from_hashes() {
        let items: Vec<&[u8]> = vec![b"alpha", b"beta", b"gamma", b"delta", b"epsilon"];
        let s_bytes = classic_from_bytes::<_, 64>(items.iter().copied(), 0x77);
        let s_hashed = classic_from_hashes::<_, 64>(items.iter().map(|b| mix64(b, 0x77)), 0x77);
        assert_eq!(s_bytes, s_hashed);
    }

    #[test]
    fn table_based_8way_matches_per_byte_reference() {
        // Hand-compute the per-byte hash family directly and compare
        // against the dispatched gather path.
        let seeds: [u64; 8] = [
            0x1111_1111_u64,
            0x2222_2222,
            0x3333_3333,
            0x4444_4444,
            0x5555_5555,
            0x6666_6666,
            0x7777_7777,
            0x8888_8888,
        ];
        let table = build_byte_table_from_seeds(&seeds);
        let payload = b"the quick brown fox jumps over the lazy dog 0123456789!@#$%^&*()";

        let actual = classic_from_bytes_table_8(payload, &table);

        let mut expected = [u64::MAX; 8];
        for &b in payload {
            for k in 0..8 {
                let h = crate::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] {
                    expected[k] = h;
                }
            }
        }
        assert_eq!(actual.slots(), &expected);
        // populated flags should all be true since we updated every slot.
        for k in 0..8 {
            assert!(actual.populated[k], "slot {k} should be populated");
        }
    }

    #[test]
    fn empty_input_does_not_populate_table_signature() {
        let seeds: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let table = build_byte_table_from_seeds(&seeds);
        let sig = classic_from_bytes_table_8(b"", &table);
        for slot in sig.slots() {
            assert_eq!(*slot, u64::MAX);
        }
        assert!(sig.is_empty());
    }

    // ----- IncrementalSignature streaming tests -----------------------------

    fn make_test_seeds_8() -> [u64; 8] {
        [
            0x1111_1111_u64,
            0x2222_2222,
            0x3333_3333,
            0x4444_4444,
            0x5555_5555,
            0x6666_6666,
            0x7777_7777,
            0x8888_8888,
        ]
    }

    fn make_test_seeds_4() -> [u64; 4] {
        [0xAA_u64, 0xBB, 0xCC, 0xDD]
    }

    fn random_bytes(n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        while out.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(n);
        out
    }

    #[test]
    fn incremental_empty_signature_is_empty() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);
        let builder = IncrementalSignature::<8>::new(&table);
        let sig = builder.finalize();
        assert!(sig.is_empty());
    }

    #[test]
    fn incremental_update_byte_matches_one_shot() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = b"the quick brown fox jumps over the lazy dog";

        let mut builder = IncrementalSignature::<8>::new(&table);
        for &b in payload {
            builder.update_byte(b);
        }
        let stream_sig = builder.finalize();

        let one_shot = classic_from_bytes_table_8(payload, &table);
        assert_eq!(stream_sig.slots(), one_shot.slots());
    }

    #[test]
    fn incremental_update_bytes_matches_one_shot() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = random_bytes(4096);

        let mut builder = IncrementalSignature::<8>::new(&table);
        builder.update_bytes(&payload);
        let stream_sig = builder.finalize();

        let one_shot = classic_from_bytes_table_8(&payload, &table);
        assert_eq!(stream_sig.slots(), one_shot.slots());
    }

    /// Two streams chunked differently must produce identical signatures.
    #[test]
    fn stream_chunking_invariant() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = random_bytes(64 * 1024);
        let one_shot = classic_from_bytes_table_8(&payload, &table);

        for &chunk in &[1_usize, 7, 17, 63, 64, 65, 1024, 4096] {
            let mut builder = IncrementalSignature::<8>::new(&table);
            for block in payload.chunks(chunk) {
                builder.update_bytes(block);
            }
            assert_eq!(
                builder.finalize().slots(),
                one_shot.slots(),
                "chunk={chunk}"
            );
        }
    }

    /// The non-K=8 path delegates to the scalar reference. It must match the
    /// per-byte hand-rolled K-min update.
    #[test]
    fn incremental_k4_matches_scalar_reference() {
        let seeds = make_test_seeds_4();
        let table = build_byte_table_from_seeds::<4>(&seeds);

        let payload = random_bytes(4096);

        let mut builder = IncrementalSignature::<4>::new(&table);
        builder.update_bytes(&payload);
        let stream_sig = builder.finalize();

        let mut expected = [u64::MAX; 4];
        for &b in &payload {
            for k in 0..4 {
                let h = crate::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] {
                    expected[k] = h;
                }
            }
        }
        assert_eq!(stream_sig.slots(), &expected);
    }

    /// `snapshot()` returns the live signature without consuming the builder;
    /// further updates must continue to refine it monotonically.
    #[test]
    fn snapshot_does_not_consume_builder() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let mut builder = IncrementalSignature::<8>::new(&table);
        builder.update_bytes(b"first half");
        let snap = builder.snapshot();

        builder.update_bytes(b" + second half");
        let later = builder.snapshot();

        // K-min slots only get smaller as more bytes are observed.
        for k in 0..8 {
            assert!(later.slots[k] <= snap.slots[k]);
        }

        // After all updates, the builder's signature equals what we'd get
        // from a fresh one-shot pass over the concatenated input.
        let final_sig = builder.finalize();
        let one_shot = classic_from_bytes_table_8(b"first half + second half", &table);
        assert_eq!(final_sig.slots(), one_shot.slots());
    }

    #[test]
    fn reset_clears_signature() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let mut builder = IncrementalSignature::<8>::new(&table);
        builder.update_bytes(b"poison");
        builder.reset();
        builder.update_bytes(b"abcdef");

        let one_shot = classic_from_bytes_table_8(b"abcdef", &table);
        assert_eq!(builder.finalize().slots(), one_shot.slots());
    }

    /// Merging two shard-signatures matches the signature of the
    /// concatenated input. K-min is associative + commutative under min.
    #[test]
    fn merge_matches_concatenation() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let payload = random_bytes(8192);
        let (left, right) = payload.split_at(payload.len() / 2);

        let sig_left = classic_from_bytes_table_8(left, &table);
        let sig_right = classic_from_bytes_table_8(right, &table);

        let mut merged = IncrementalSignature::<8>::from_signature(&table, sig_left);
        merged.merge(&sig_right);

        let one_shot = classic_from_bytes_table_8(&payload, &table);
        assert_eq!(merged.finalize().slots(), one_shot.slots());
    }

    /// `from_signature` round-trips: building a signature, snapshotting it,
    /// resuming with `from_signature`, and feeding more bytes equals the
    /// signature of the concatenated input.
    #[test]
    fn from_signature_resumes_correctly() {
        let seeds = make_test_seeds_8();
        let table = build_byte_table_from_seeds::<8>(&seeds);

        let mut a = IncrementalSignature::<8>::new(&table);
        a.update_bytes(b"first half");
        let snapshot = a.snapshot();

        let mut b = IncrementalSignature::<8>::from_signature(&table, snapshot);
        b.update_bytes(b" rest");

        let one_shot = classic_from_bytes_table_8(b"first half rest", &table);
        assert_eq!(b.finalize().slots(), one_shot.slots());
    }
}

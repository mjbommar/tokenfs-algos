//! Approximate data structures: SpaceSaving heavy hitters, Bloom filter,
//! and HyperLogLog cardinality estimator.
//!
//! Per `docs/SIMILARITY_APPROXIMATION_ROADMAP.md` Phase 4. Each structure
//! reports its memory footprint and documented error bounds.
//!
//! Memory model:
//! - [`SpaceSaving`] is fixed-size (`[T; K]`), allocation-free, no_std/alloc-clean.
//! - [`BloomFilter`] and [`HyperLogLog`] hold `Vec`-backed storage; gated on
//!   the `std` feature (matches the LSH module).
//!
//! `CountMinSketch` already lives in [`crate::sketch`] and is not duplicated
//! here.

// ============================================================================
// SpaceSaving heavy hitters (fixed-size, no_std-clean)
// ============================================================================

/// SpaceSaving algorithm for top-K heavy hitters.
///
/// Stronger than [`crate::sketch::MisraGries`]: supports incremental updates
/// in O(1) amortized while maintaining a strict approximate count for each
/// of the K monitored items. Counts have an error bound of `N / K` where
/// `N` is the total stream length. Reference: Metwally, Agrawal, El Abbadi,
/// "Efficient Computation of Frequent and Top-k Elements in Data Streams",
/// ICDT 2005.
///
/// Memory footprint: `K * (size_of::<u64>() + size_of::<u32>()) = 12 * K` bytes
/// of monitored slots, plus 16 bytes for `populated` and `total`. No heap.
#[derive(Clone, Debug)]
pub struct SpaceSaving<const K: usize> {
    /// Monitored items.
    items: [u64; K],
    /// Approximate counts. `counts[i]` is the count for `items[i]`.
    counts: [u32; K],
    /// Number of populated slots (0..=K).
    populated: usize,
    /// Total observations.
    total: u64,
}

impl<const K: usize> Default for SpaceSaving<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize> SpaceSaving<K> {
    /// Builds an empty SpaceSaving sketch.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            items: [0; K],
            counts: [0; K],
            populated: 0,
            total: 0,
        }
    }

    /// Total observations seen.
    #[must_use]
    pub const fn total(&self) -> u64 {
        self.total
    }

    /// Memory footprint of the sketch in bytes.
    #[must_use]
    pub const fn memory_bytes() -> usize {
        K * (core::mem::size_of::<u64>() + core::mem::size_of::<u32>())
            + 2 * core::mem::size_of::<u64>()
    }

    /// Records one observation of `item`.
    pub fn update(&mut self, item: u64) {
        self.update_by(item, 1);
    }

    /// Records `count` observations of `item`.
    pub fn update_by(&mut self, item: u64, count: u32) {
        if count == 0 {
            return;
        }
        self.total = self.total.wrapping_add(u64::from(count));

        // Already monitored? Increment its slot.
        if let Some(idx) = self.find_index(item) {
            self.counts[idx] = self.counts[idx].saturating_add(count);
            return;
        }

        // Free slot? Use it.
        if self.populated < K {
            let idx = self.populated;
            self.items[idx] = item;
            self.counts[idx] = count;
            self.populated += 1;
            return;
        }

        // Full: evict the slot with the lowest count and let the incoming
        // item inherit `displaced + count` per SpaceSaving's contract.
        let evict_idx = self.min_index();
        let displaced = self.counts[evict_idx];
        self.items[evict_idx] = item;
        self.counts[evict_idx] = displaced.saturating_add(count);
    }

    /// Returns the K (item, count) pairs in **arbitrary order**. Unpopulated
    /// slots have count 0.
    #[must_use]
    pub fn snapshot(&self) -> [(u64, u32); K] {
        let mut out = [(0_u64, 0_u32); K];
        for (slot, dst) in out.iter_mut().enumerate().take(self.populated) {
            *dst = (self.items[slot], self.counts[slot]);
        }
        out
    }

    /// Returns the count estimate for `item`. Items not monitored are
    /// reported as 0. A monitored item's count is an over-estimate of its
    /// true frequency, bounded by the count of the displaced item.
    #[must_use]
    pub fn estimate(&self, item: u64) -> u32 {
        self.find_index(item).map(|i| self.counts[i]).unwrap_or(0)
    }

    fn find_index(&self, item: u64) -> Option<usize> {
        self.items[..self.populated].iter().position(|&x| x == item)
    }

    fn min_index(&self) -> usize {
        let mut idx = 0;
        let mut min = self.counts[0];
        for i in 1..self.populated {
            if self.counts[i] < min {
                min = self.counts[i];
                idx = i;
            }
        }
        idx
    }
}

// ============================================================================
// Bloom filter (Vec-backed, std-gated)
// ============================================================================

#[cfg(feature = "std")]
mod bloom {
    use crate::hash::mix64;

    /// Vec-backed Bloom filter.
    ///
    /// `bits` is the size of the bit vector (rounded up to a multiple of 64);
    /// `k` is the number of hash positions per item (typical: 3-7). False
    /// positive rate for `n` items is approximately
    /// `(1 - exp(-k * n / bits))^k`. Optimal `k = (bits / n) * ln(2)`.
    ///
    /// Hash functions are derived via the Kirsch-Mitzenmacher double-hashing
    /// trick: two base hashes `h1`, `h2` are combined as `h1 + i * h2` for
    /// `i = 0..k`.
    #[derive(Clone, Debug)]
    pub struct BloomFilter {
        words: Vec<u64>,
        bits: usize,
        k: usize,
        inserted: u64,
    }

    impl BloomFilter {
        /// Builds an empty Bloom filter with `bits` bits and `k` hash
        /// functions. `bits` is rounded up to the next multiple of 64.
        ///
        /// # Panics
        ///
        /// Panics if `bits == 0` or `k == 0`.
        #[must_use]
        pub fn new(bits: usize, k: usize) -> Self {
            assert!(bits > 0, "BloomFilter bits must be > 0");
            assert!(k > 0, "BloomFilter k must be > 0");
            let words_needed = bits.div_ceil(64);
            let actual_bits = words_needed * 64;
            Self {
                words: vec![0; words_needed],
                bits: actual_bits,
                k,
                inserted: 0,
            }
        }

        /// Builds a Bloom filter sized for `expected_items` with the
        /// requested `target_fpr` (false-positive rate). Picks bit count
        /// and `k` per the optimal-loading formulas.
        #[must_use]
        pub fn with_target(expected_items: usize, target_fpr: f64) -> Self {
            assert!(expected_items > 0);
            assert!(target_fpr > 0.0 && target_fpr < 1.0);
            // m = -n ln(p) / (ln 2)^2; k = (m/n) ln 2.
            let n = expected_items as f64;
            let ln2_sq = core::f64::consts::LN_2 * core::f64::consts::LN_2;
            let m = (-n * target_fpr.ln() / ln2_sq).ceil() as usize;
            let k = ((m as f64 / n) * core::f64::consts::LN_2).round().max(1.0) as usize;
            Self::new(m, k)
        }

        /// Number of items inserted.
        #[must_use]
        pub fn inserted(&self) -> u64 {
            self.inserted
        }

        /// Total bit count (rounded to multiple of 64).
        #[must_use]
        pub fn bits(&self) -> usize {
            self.bits
        }

        /// Number of hash positions per item.
        #[must_use]
        pub fn k(&self) -> usize {
            self.k
        }

        /// Memory footprint in bytes (storage only — fixed overhead is small).
        #[must_use]
        pub fn memory_bytes(&self) -> usize {
            self.words.len() * core::mem::size_of::<u64>()
        }

        /// Estimated false-positive rate at the current load.
        #[must_use]
        pub fn estimated_false_positive_rate(&self) -> f64 {
            if self.inserted == 0 {
                return 0.0;
            }
            let n = self.inserted as f64;
            let m = self.bits as f64;
            let k = self.k as f64;
            let p = 1.0 - (-k * n / m).exp();
            p.powf(k)
        }

        /// Inserts `item`.
        pub fn insert(&mut self, item: &[u8]) {
            let h1 = mix64(item, 0x9E37_79B9_7F4A_7C15);
            let h2 = mix64(item, 0x6E5E_2E5C_DEAD_BEEF);
            for i in 0..self.k {
                let position = h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize % self.bits;
                let word = position / 64;
                let bit = position % 64;
                self.words[word] |= 1_u64 << bit;
            }
            self.inserted = self.inserted.wrapping_add(1);
        }

        /// True if `item` *might* be present; false if definitely not.
        #[must_use]
        pub fn contains(&self, item: &[u8]) -> bool {
            let h1 = mix64(item, 0x9E37_79B9_7F4A_7C15);
            let h2 = mix64(item, 0x6E5E_2E5C_DEAD_BEEF);
            for i in 0..self.k {
                let position = h1.wrapping_add((i as u64).wrapping_mul(h2)) as usize % self.bits;
                let word = position / 64;
                let bit = position % 64;
                if self.words[word] & (1_u64 << bit) == 0 {
                    return false;
                }
            }
            true
        }

        /// Resets the filter.
        pub fn clear(&mut self) {
            for w in &mut self.words {
                *w = 0;
            }
            self.inserted = 0;
        }
    }
}

#[cfg(feature = "std")]
pub use bloom::BloomFilter;

// ============================================================================
// HyperLogLog cardinality estimator (Vec-backed, std-gated)
// ============================================================================

#[cfg(feature = "std")]
mod hll {
    use crate::hash::mix64;

    /// HyperLogLog cardinality estimator with `2^precision` registers.
    ///
    /// `precision` is in `4..=16`. Memory footprint is `2^precision` bytes.
    /// Standard error is `1.04 / sqrt(2^precision)`:
    ///
    /// | precision | registers | memory  | std err |
    /// |-----------|-----------|---------|---------|
    /// | 8  | 256   | 256 B  | ~6.5%  |
    /// | 10 | 1024  | 1 KiB  | ~3.3%  |
    /// | 12 | 4096  | 4 KiB  | ~1.6%  |
    /// | 14 | 16384 | 16 KiB | ~0.81% |
    /// | 16 | 65536 | 64 KiB | ~0.41% |
    ///
    /// Reference: Flajolet et al., "HyperLogLog: the analysis of a
    /// near-optimal cardinality estimation algorithm", AofA 2007. Uses the
    /// original HLL formulation with small/large-range corrections; HLL++
    /// improvements (sparse representation, refined bias correction) are
    /// out of scope for this scalar-first pass.
    #[derive(Clone, Debug)]
    pub struct HyperLogLog {
        registers: Vec<u8>,
        precision: u32,
    }

    impl HyperLogLog {
        /// Builds an empty HLL with the given precision (4..=16).
        ///
        /// # Panics
        ///
        /// Panics if precision is outside `4..=16`.
        #[must_use]
        pub fn new(precision: u32) -> Self {
            assert!(
                (4..=16).contains(&precision),
                "HyperLogLog precision must be in 4..=16, got {precision}"
            );
            let m = 1_usize << precision;
            Self {
                registers: vec![0; m],
                precision,
            }
        }

        /// Number of registers (`2^precision`).
        #[must_use]
        pub fn registers(&self) -> usize {
            self.registers.len()
        }

        /// Precision parameter.
        #[must_use]
        pub fn precision(&self) -> u32 {
            self.precision
        }

        /// Memory footprint in bytes.
        #[must_use]
        pub fn memory_bytes(&self) -> usize {
            self.registers.len()
        }

        /// Adds an item to the sketch.
        pub fn insert(&mut self, item: &[u8]) {
            self.insert_hash(mix64(item, 0xC8C2_5E0F_2C5C_3F6D));
        }

        /// Adds a precomputed `u64` hash to the sketch.
        pub fn insert_hash(&mut self, hash: u64) {
            let p = self.precision;
            let idx = (hash >> (64 - p)) as usize;
            // Rank: leading-zeros count in the low (64 - p) bits, plus 1.
            let masked = hash << p;
            let rank = if masked == 0 {
                (64 - p) as u8 + 1
            } else {
                masked.leading_zeros() as u8 + 1
            };
            if rank > self.registers[idx] {
                self.registers[idx] = rank;
            }
        }

        /// Returns the estimated cardinality.
        #[must_use]
        pub fn estimate(&self) -> u64 {
            let m = self.registers.len() as f64;
            let alpha = match self.precision {
                4 => 0.673,
                5 => 0.697,
                _ => 0.7213 / (1.0 + 1.079 / m),
            };
            let mut sum = 0.0_f64;
            let mut zeros = 0_usize;
            for &r in &self.registers {
                sum += 2.0_f64.powi(-i32::from(r));
                if r == 0 {
                    zeros += 1;
                }
            }
            let raw = alpha * m * m / sum;

            // Small-range correction.
            if raw <= 2.5 * m && zeros > 0 {
                let corrected = m * (m / zeros as f64).ln();
                return corrected.round() as u64;
            }

            // Large-range correction (saturation near 2^32).
            let two_32 = 4_294_967_296.0_f64;
            if raw > two_32 / 30.0 {
                let corrected = -two_32 * (1.0 - raw / two_32).ln();
                return corrected.round() as u64;
            }

            raw.round() as u64
        }

        /// Merges another sketch into this one.
        ///
        /// # Panics
        ///
        /// Panics if `other.precision != self.precision`.
        pub fn merge(&mut self, other: &Self) {
            assert_eq!(self.precision, other.precision);
            for (a, &b) in self.registers.iter_mut().zip(&other.registers) {
                if b > *a {
                    *a = b;
                }
            }
        }

        /// Clears all registers.
        pub fn clear(&mut self) {
            for r in &mut self.registers {
                *r = 0;
            }
        }
    }
}

#[cfg(feature = "std")]
pub use hll::HyperLogLog;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    // ----- SpaceSaving -----

    #[test]
    fn space_saving_finds_obvious_heavy_hitter() {
        let mut sk = SpaceSaving::<4>::new();
        for _ in 0..1000 {
            sk.update(42);
        }
        for i in 0..30_u64 {
            sk.update(i);
        }
        let snap = sk.snapshot();
        let (top_item, top_count) = snap.iter().max_by_key(|(_, c)| *c).copied().unwrap();
        assert_eq!(top_item, 42, "snap={snap:?}");
        assert!(top_count >= 1000);
    }

    #[test]
    fn space_saving_zipfian_top1_recovers() {
        // Zipfian: item i appears 100/i times (i in 1..=20). Total ≈ 354.
        // K=8 has error bound N/K ≈ 44 per slot, so the *very* heavy
        // hitter (item 1, count 100) is reliably top-1 but rank shuffles
        // among items 2-8 are within the documented bound.
        let mut sk = SpaceSaving::<8>::new();
        // Shuffle the insertion order so we don't accidentally test a
        // monotonic-decreasing-frequency edge case.
        let mut ops: Vec<u64> = Vec::new();
        for i in 1_u64..=20 {
            for _ in 0..(100 / i) {
                ops.push(i);
            }
        }
        // Deterministic Fisher-Yates with a fixed seed.
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        for i in (1..ops.len()).rev() {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let j = (state as usize) % (i + 1);
            ops.swap(i, j);
        }
        for x in ops {
            sk.update(x);
        }
        let mut snap = sk.snapshot().to_vec();
        snap.sort_by_key(|p| core::cmp::Reverse(p.1));
        assert_eq!(
            snap[0].0, 1,
            "item 1 (count 100) should be top-1; snap={snap:?}"
        );
        // Item 1's count must be within N/K of the true count 100.
        let bound = sk.total() / 8 + 100;
        assert!(
            snap[0].1 as u64 <= bound,
            "count {} exceeds N/K bound {}",
            snap[0].1,
            bound
        );
        assert!(
            snap[0].1 >= 100,
            "underestimated heavy hitter: count={}",
            snap[0].1
        );
    }

    #[test]
    fn space_saving_memory_footprint_is_documented() {
        // 8 * (8 + 4) + 16 = 112
        assert_eq!(SpaceSaving::<8>::memory_bytes(), 8 * 12 + 16);
    }

    // ----- Bloom filter -----

    #[cfg(feature = "std")]
    #[test]
    fn bloom_filter_no_false_negatives() {
        let mut bf = BloomFilter::new(1024, 5);
        let items: Vec<&[u8]> = vec![
            b"alpha", b"beta", b"gamma", b"delta", b"epsilon", b"zeta", b"eta",
        ];
        for item in &items {
            bf.insert(item);
        }
        for item in &items {
            assert!(bf.contains(item), "false negative for {item:?}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn bloom_filter_false_positive_rate_within_bound() {
        let mut bf = BloomFilter::new(4096, 5);
        for i in 0..200_u32 {
            bf.insert(&i.to_le_bytes());
        }
        let mut false_positives = 0_usize;
        let trials = 10_000;
        for i in 1_000_000_u32..1_000_000_u32 + trials as u32 {
            if bf.contains(&i.to_le_bytes()) {
                false_positives += 1;
            }
        }
        let observed = false_positives as f64 / trials as f64;
        let predicted = bf.estimated_false_positive_rate();
        assert!(
            observed < predicted * 2.5 + 0.005,
            "observed FPR {observed} >> predicted {predicted}"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn bloom_filter_with_target_picks_reasonable_params() {
        let bf = BloomFilter::with_target(1000, 0.01);
        // m = -1000 * ln(0.01) / (ln 2)^2 ≈ 9586 bits, rounded to 9600.
        assert!(bf.bits() >= 9000 && bf.bits() <= 12_000);
        assert!(bf.k() >= 5 && bf.k() <= 10);
    }

    #[cfg(feature = "std")]
    #[test]
    fn bloom_filter_clear_resets_state() {
        let mut bf = BloomFilter::new(512, 3);
        bf.insert(b"x");
        assert!(bf.contains(b"x"));
        bf.clear();
        assert!(!bf.contains(b"x"));
        assert_eq!(bf.inserted(), 0);
    }

    // ----- HyperLogLog -----

    #[cfg(feature = "std")]
    #[test]
    fn hll_cardinality_within_documented_error_bound() {
        let mut hll = HyperLogLog::new(12); // 1.6% std err
        let true_card = 50_000_u64;
        for i in 0..true_card {
            hll.insert(&i.to_le_bytes());
        }
        let est = hll.estimate();
        let err = (est as f64 - true_card as f64).abs() / true_card as f64;
        // ~5x sigma slack for single-run variance.
        assert!(
            err < 0.08,
            "P=12 HLL err={err}, est={est}, true={true_card}"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_small_cardinality_uses_correction() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..50_u64 {
            hll.insert(&i.to_le_bytes());
        }
        let est = hll.estimate();
        let err = (est as f64 - 50.0).abs() / 50.0;
        assert!(err < 0.20, "small-range HLL err={err}, est={est}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_merge_combines_estimates() {
        let mut a = HyperLogLog::new(10);
        let mut b = HyperLogLog::new(10);
        for i in 0..5_000_u64 {
            a.insert(&i.to_le_bytes());
        }
        for i in 5_000..15_000_u64 {
            b.insert(&i.to_le_bytes());
        }
        a.merge(&b);
        let est = a.estimate();
        let true_card = 15_000.0;
        let err = (est as f64 - true_card).abs() / true_card;
        assert!(err < 0.1, "merged HLL err={err}, est={est}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_clear_resets_state() {
        let mut hll = HyperLogLog::new(8);
        for i in 0..1000_u64 {
            hll.insert(&i.to_le_bytes());
        }
        assert!(hll.estimate() > 100);
        hll.clear();
        assert_eq!(hll.estimate(), 0);
    }
}

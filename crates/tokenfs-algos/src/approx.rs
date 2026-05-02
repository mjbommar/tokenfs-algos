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
//!
//! # Fallible constructors
//!
//! Each `new` / `with_target` constructor below panics on out-of-range
//! parameters (zero bits, zero precision, etc.) — convenient for
//! userland callers that pass compile-time constants. For
//! kernel-adjacent or user-input-driven callers a `try_new` /
//! `try_with_target` parallel returns [`ApproxError`] instead. See
//! the per-method docs for the exact preconditions enforced.

/// Failure modes for the fallible approx-data-structure constructors
/// (`BloomFilter::try_new`, `BloomFilter::try_with_target`,
/// `HyperLogLog::try_new`).
#[cfg(feature = "std")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ApproxError {
    /// A required positive-integer parameter was zero (BloomFilter
    /// `bits` / `k`, HyperLogLog `precision` outside the valid band).
    ZeroParameter {
        /// Name of the offending parameter.
        name: &'static str,
    },
    /// A floating-point ratio (e.g. Bloom filter target FPR) was
    /// outside the open interval `(0, 1)`.
    OutOfRangeFraction {
        /// Name of the offending parameter.
        name: &'static str,
        /// Caller-supplied value.
        value: f64,
    },
    /// HyperLogLog precision must be in `4..=16`; this captures
    /// values outside that band.
    PrecisionOutOfRange {
        /// Caller-supplied precision.
        requested: u32,
        /// Smallest accepted precision.
        min: u32,
        /// Largest accepted precision.
        max: u32,
    },
}

#[cfg(feature = "std")]
impl core::fmt::Display for ApproxError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ZeroParameter { name } => {
                write!(f, "approx parameter `{name}` must be > 0")
            }
            Self::OutOfRangeFraction { name, value } => write!(
                f,
                "approx parameter `{name}` = {value} must be in the open interval (0, 1)"
            ),
            Self::PrecisionOutOfRange {
                requested,
                min,
                max,
            } => write!(
                f,
                "HyperLogLog precision {requested} outside accepted band {min}..={max}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ApproxError {}

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

        /// Fallible variant of [`Self::new`] returning [`super::ApproxError`]
        /// on `bits == 0` or `k == 0` instead of panicking.
        pub fn try_new(bits: usize, k: usize) -> Result<Self, super::ApproxError> {
            if bits == 0 {
                return Err(super::ApproxError::ZeroParameter { name: "bits" });
            }
            if k == 0 {
                return Err(super::ApproxError::ZeroParameter { name: "k" });
            }
            Ok(Self::new(bits, k))
        }

        /// Fallible variant of [`Self::with_target`].
        pub fn try_with_target(
            expected_items: usize,
            target_fpr: f64,
        ) -> Result<Self, super::ApproxError> {
            if expected_items == 0 {
                return Err(super::ApproxError::ZeroParameter {
                    name: "expected_items",
                });
            }
            if !(target_fpr > 0.0 && target_fpr < 1.0) {
                return Err(super::ApproxError::OutOfRangeFraction {
                    name: "target_fpr",
                    value: target_fpr,
                });
            }
            Ok(Self::with_target(expected_items, target_fpr))
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

    /// Failure modes for the fallible HyperLogLog merge variants
    /// ([`HyperLogLog::try_merge`] / [`HyperLogLog::try_merge_simd`]).
    ///
    /// The infallible [`HyperLogLog::merge`] / [`HyperLogLog::merge_simd`]
    /// surface panics on the same precondition violations; this enum is
    /// the kernel-adjacent / user-input-driven counterpart.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum HllMergeError {
        /// Two sketches with different `precision` parameters cannot be
        /// merged because their register counts and hash-bucket maps
        /// differ.
        PrecisionMismatch {
            /// Precision of `self` (the destination sketch).
            lhs: u32,
            /// Precision of `other` (the source sketch).
            rhs: u32,
        },
    }

    impl core::fmt::Display for HllMergeError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                Self::PrecisionMismatch { lhs, rhs } => write!(
                    f,
                    "HyperLogLog merge requires equal precision: lhs={lhs}, rhs={rhs}"
                ),
            }
        }
    }

    impl std::error::Error for HllMergeError {}

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
    ///
    /// SIMD-accelerated variants of the merge and harmonic-mean kernels
    /// live in [`kernels`]; the public APIs ([`Self::merge_simd`],
    /// [`Self::count_simd`]) auto-dispatch to the best available backend
    /// (AVX-512 > AVX2 > NEON > scalar) and are bit-identical with the
    /// scalar paths for the merge case and within `~1e-12` relative
    /// tolerance for the harmonic-mean case (parallel f64 accumulation
    /// reorders rounding).
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

        /// Fallible variant of [`Self::new`] returning
        /// [`super::ApproxError`] when `precision` is outside `4..=16`.
        pub fn try_new(precision: u32) -> Result<Self, super::ApproxError> {
            if !(4..=16).contains(&precision) {
                return Err(super::ApproxError::PrecisionOutOfRange {
                    requested: precision,
                    min: 4,
                    max: 16,
                });
            }
            Ok(Self::new(precision))
        }

        /// Number of registers (`2^precision`).
        #[must_use]
        pub fn registers(&self) -> usize {
            self.registers.len()
        }

        /// Read-only view of the underlying register byte slice.
        ///
        /// Exposed for SIMD parity testing and external aggregators that
        /// want to combine sketches without going through [`Self::merge`].
        #[must_use]
        pub fn register_bytes(&self) -> &[u8] {
            &self.registers
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

        /// Returns the alpha constant for this sketch's precision per
        /// Flajolet et al.'s small-precision lookup, otherwise the
        /// asymptotic `0.7213 / (1 + 1.079 / m)` formula.
        #[inline]
        #[must_use]
        pub fn alpha(&self) -> f64 {
            let m = self.registers.len() as f64;
            match self.precision {
                4 => 0.673,
                5 => 0.697,
                _ => 0.7213 / (1.0 + 1.079 / m),
            }
        }

        /// Returns the raw harmonic-mean cardinality estimate `alpha * m^2 / Z`
        /// where `Z = sum(2^-r)` over all registers.
        ///
        /// This is the **uncorrected** estimate before the small-range
        /// linear-counting and large-range saturation corrections that
        /// [`Self::estimate`] applies. Useful as the parity oracle for
        /// SIMD harmonic-mean kernels — see [`Self::count_simd`].
        #[must_use]
        pub fn count_raw(&self) -> f64 {
            kernels::scalar::count_raw(&self.registers, self.alpha())
        }

        /// SIMD-accelerated raw harmonic-mean cardinality estimate.
        ///
        /// Bit-identical numerical contract with [`Self::count_raw`]
        /// modulo floating-point reduction order (see kernel docs); the
        /// tests check parity within a `1e-12` relative tolerance.
        ///
        /// Auto-dispatches to AVX-512 > AVX2 > NEON > scalar on the
        /// current host. Returns the same alpha-scaled `m^2 / Z` value
        /// as [`Self::count_raw`] — apply [`Self::estimate`] for the
        /// fully bias-corrected u64 cardinality.
        #[must_use]
        pub fn count_simd(&self) -> f64 {
            kernels::auto::count_raw(&self.registers, self.alpha())
        }

        /// Returns the estimated cardinality.
        #[must_use]
        pub fn estimate(&self) -> u64 {
            let m = self.registers.len() as f64;
            let mut zeros = 0_usize;
            for &r in &self.registers {
                if r == 0 {
                    zeros += 1;
                }
            }
            let raw = self.count_raw();

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
            assert_eq!(
                self.precision, other.precision,
                "HyperLogLog merge requires equal precision: {} vs {}",
                self.precision, other.precision
            );
            kernels::scalar::merge(&mut self.registers, &other.registers);
        }

        /// Fallible variant of [`Self::merge`].
        ///
        /// Returns [`HllMergeError::PrecisionMismatch`] if the two
        /// sketches have different precisions instead of panicking.
        pub fn try_merge(&mut self, other: &Self) -> Result<(), HllMergeError> {
            if self.precision != other.precision {
                return Err(HllMergeError::PrecisionMismatch {
                    lhs: self.precision,
                    rhs: other.precision,
                });
            }
            kernels::scalar::merge(&mut self.registers, &other.registers);
            Ok(())
        }

        /// SIMD-accelerated in-place per-bucket-max merge.
        ///
        /// Bit-identical contract with [`Self::merge`]: every register
        /// in `self` is replaced with `max(self, other)`. Auto-dispatches
        /// to AVX-512 (`vpmaxub` on `__m512i`), AVX2
        /// (`_mm256_max_epu8`), NEON (`vmaxq_u8`), or the scalar fallback
        /// based on host capability.
        ///
        /// # Panics
        ///
        /// Panics if `other.precision != self.precision`.
        pub fn merge_simd(&mut self, other: &Self) {
            assert_eq!(
                self.precision, other.precision,
                "HyperLogLog merge requires equal precision: {} vs {}",
                self.precision, other.precision
            );
            kernels::auto::merge(&mut self.registers, &other.registers);
        }

        /// Fallible variant of [`Self::merge_simd`].
        ///
        /// Returns [`HllMergeError::PrecisionMismatch`] if the two
        /// sketches have different precisions instead of panicking.
        pub fn try_merge_simd(&mut self, other: &Self) -> Result<(), HllMergeError> {
            if self.precision != other.precision {
                return Err(HllMergeError::PrecisionMismatch {
                    lhs: self.precision,
                    rhs: other.precision,
                });
            }
            kernels::auto::merge(&mut self.registers, &other.registers);
            Ok(())
        }

        /// Clears all registers.
        pub fn clear(&mut self) {
            for r in &mut self.registers {
                *r = 0;
            }
        }
    }

    /// Pinned SIMD kernels for HyperLogLog merge (per-bucket max) and
    /// the harmonic-mean cardinality reduction.
    ///
    /// Each backend is exposed for parity testing; the public
    /// [`HyperLogLog::merge_simd`] / [`HyperLogLog::count_simd`] APIs
    /// auto-dispatch via [`kernels::auto`].
    ///
    /// Algorithmic notes:
    ///
    /// - **Merge.** The per-bucket maximum of two `&[u8]` register vectors.
    ///   AVX2 uses `_mm256_max_epu8` (32 bytes/iter), AVX-512 uses
    ///   `_mm512_max_epu8` (64 bytes/iter), NEON uses `vmaxq_u8`
    ///   (16 bytes/iter). All three are byte-exact with the scalar path
    ///   because `max(u8, u8)` is associative-and-commutative and
    ///   element-wise.
    ///
    /// - **Harmonic mean.** Reduces `Z = sum_i 2^-registers[i]` over the
    ///   register array, then returns `alpha * m^2 / Z`. The SIMD path
    ///   uses a 256-entry f64 lookup table (`POW2_NEG_LUT`) indexed by
    ///   the raw `u8` register value (registers are bounded by
    ///   `64 - precision + 1 <= 61`, but we size the LUT to 256 to keep
    ///   gather indices unconditional). Multiple parallel f64
    ///   accumulators break the floating-point reduction dependency
    ///   chain. Floating-point summation is not associative, so SIMD and
    ///   scalar may differ by `~1e-15` per term; aggregate relative
    ///   error stays well under `1e-12` for all valid precisions.
    pub mod kernels {
        /// Lookup table of `2^-r` for `r ∈ 0..=255` as `f64`. All HLL
        /// register values are bounded by `64` (rank for an all-zero
        /// hash chunk), so entries `r >= 65` are unreachable in
        /// well-formed sketches but kept for unconditional gather
        /// indexing without bounds-check overhead.
        ///
        /// Entries `r ∈ 0..=1074` of `2^-r` are exactly representable in
        /// f64 (subnormals start at `2^-1022`); for `r ∈ 0..=64` the
        /// values are all normalised f64s with zero rounding error.
        pub static POW2_NEG_LUT: [f64; 256] = build_pow2_neg_lut();

        const fn build_pow2_neg_lut() -> [f64; 256] {
            let mut lut = [0.0_f64; 256];
            let mut i = 0;
            while i < 256 {
                // 2^-i for i ∈ 0..=1074 is exactly representable; for
                // i >= 1075 it underflows to 0. const-eval can't call
                // f64::powi, so synthesise the value via bit-pattern
                // arithmetic on the IEEE-754 exponent field.
                if i == 0 {
                    lut[i] = 1.0;
                } else if i <= 1022 {
                    // 2^-i = bit pattern with biased exponent (1023 - i).
                    let exp_bits = ((1023_i64 - i as i64) as u64) & 0x7ff;
                    let bits = exp_bits << 52;
                    lut[i] = f64::from_bits(bits);
                } else if i <= 1074 {
                    // Subnormals: mantissa-only, exponent = 0.
                    let mantissa_bit = 1_u64 << (52 - (i - 1023));
                    lut[i] = f64::from_bits(mantissa_bit);
                } else {
                    lut[i] = 0.0;
                }
                i += 1;
            }
            lut
        }

        /// Auto-dispatched kernels for [`super::HyperLogLog`].
        pub mod auto {
            /// Runtime-dispatched per-bucket max merge: `dst[i] = max(dst[i], src[i])`.
            ///
            /// `src` and `dst` must have the same length (the public
            /// surface enforces equal precision before calling).
            pub fn merge(dst: &mut [u8], src: &[u8]) {
                debug_assert_eq!(
                    dst.len(),
                    src.len(),
                    "HLL merge requires equal-length register arrays"
                );

                #[cfg(all(
                    feature = "std",
                    feature = "avx512",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if super::avx512::is_available() {
                        // SAFETY: availability checked immediately above.
                        return unsafe { super::avx512::merge(dst, src) };
                    }
                }

                #[cfg(all(
                    feature = "std",
                    feature = "avx2",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if super::avx2::is_available() {
                        // SAFETY: availability checked immediately above.
                        return unsafe { super::avx2::merge(dst, src) };
                    }
                }

                #[cfg(all(feature = "neon", target_arch = "aarch64"))]
                {
                    if super::neon::is_available() {
                        // SAFETY: NEON is mandatory on AArch64.
                        return unsafe { super::neon::merge(dst, src) };
                    }
                }

                super::scalar::merge(dst, src);
            }

            /// Runtime-dispatched harmonic-mean cardinality estimate
            /// `alpha * m^2 / Z` where `Z = sum(2^-r)` over `registers`.
            #[must_use]
            pub fn count_raw(registers: &[u8], alpha: f64) -> f64 {
                #[cfg(all(
                    feature = "std",
                    feature = "avx512",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if super::avx512::is_available() {
                        // SAFETY: availability checked immediately above.
                        return unsafe { super::avx512::count_raw(registers, alpha) };
                    }
                }

                #[cfg(all(
                    feature = "std",
                    feature = "avx2",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if super::avx2::is_available() {
                        // SAFETY: availability checked immediately above.
                        return unsafe { super::avx2::count_raw(registers, alpha) };
                    }
                }

                #[cfg(all(feature = "neon", target_arch = "aarch64"))]
                {
                    if super::neon::is_available() {
                        // SAFETY: NEON is mandatory on AArch64.
                        return unsafe { super::neon::count_raw(registers, alpha) };
                    }
                }

                super::scalar::count_raw(registers, alpha)
            }
        }

        /// Portable scalar reference implementations.
        pub mod scalar {
            use super::POW2_NEG_LUT;

            /// Per-bucket max merge: `dst[i] = max(dst[i], src[i])`.
            ///
            /// Reference for the SIMD parity tests; this is the same
            /// loop the original [`super::super::HyperLogLog::merge`]
            /// shipped with.
            pub fn merge(dst: &mut [u8], src: &[u8]) {
                debug_assert_eq!(
                    dst.len(),
                    src.len(),
                    "HLL merge requires equal-length register arrays"
                );
                for (a, &b) in dst.iter_mut().zip(src) {
                    if b > *a {
                        *a = b;
                    }
                }
            }

            /// Reference harmonic-mean cardinality `alpha * m^2 / Z`.
            ///
            /// Uses the [`super::POW2_NEG_LUT`] lookup so the scalar
            /// reference exercises the same numerical path as the SIMD
            /// kernels, eliminating LUT-vs-`powi` rounding differences
            /// from the parity tolerance budget.
            #[must_use]
            pub fn count_raw(registers: &[u8], alpha: f64) -> f64 {
                let m = registers.len() as f64;
                let mut sum = 0.0_f64;
                for &r in registers {
                    sum += POW2_NEG_LUT[r as usize];
                }
                alpha * m * m / sum
            }
        }

        /// x86 AVX2 kernels: `_mm256_max_epu8` for merge and an 8x-unrolled
        /// f64 reduction with `_mm256_i32gather_pd` for the harmonic mean.
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        pub mod avx2 {
            use super::POW2_NEG_LUT;

            #[cfg(target_arch = "x86")]
            use core::arch::x86::{
                __m128i, __m256d, __m256i, _mm_cvtepu8_epi32, _mm_set_epi64x, _mm_srli_si128,
                _mm256_add_pd, _mm256_i32gather_pd, _mm256_loadu_si256, _mm256_max_epu8,
                _mm256_setzero_pd, _mm256_storeu_si256,
            };
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::{
                __m128i, __m256d, __m256i, _mm_cvtepu8_epi32, _mm_set_epi64x, _mm_srli_si128,
                _mm256_add_pd, _mm256_i32gather_pd, _mm256_loadu_si256, _mm256_max_epu8,
                _mm256_setzero_pd, _mm256_storeu_si256,
            };

            /// 32 u8 registers per AVX2 vector.
            const VEC_BYTES: usize = 32;

            /// Returns true when AVX2 is available at runtime.
            #[cfg(feature = "std")]
            #[must_use]
            pub fn is_available() -> bool {
                std::is_x86_feature_detected!("avx2")
            }

            /// Returns true when AVX2 is available at runtime.
            #[cfg(not(feature = "std"))]
            #[must_use]
            pub const fn is_available() -> bool {
                false
            }

            /// AVX2 per-bucket max merge.
            ///
            /// # Safety
            ///
            /// The caller must ensure the current CPU supports AVX2 and
            /// that `dst.len() == src.len()`.
            #[target_feature(enable = "avx2")]
            pub unsafe fn merge(dst: &mut [u8], src: &[u8]) {
                debug_assert_eq!(dst.len(), src.len());
                let len = dst.len();
                let mut i = 0_usize;
                while i + VEC_BYTES <= len {
                    // SAFETY: `i + 32 <= len` and AVX2 enabled by the
                    // enclosing target_feature.
                    let a = unsafe { _mm256_loadu_si256(dst.as_ptr().add(i).cast::<__m256i>()) };
                    let b = unsafe { _mm256_loadu_si256(src.as_ptr().add(i).cast::<__m256i>()) };
                    let m = _mm256_max_epu8(a, b);
                    // SAFETY: same range bound as load above.
                    unsafe {
                        _mm256_storeu_si256(dst.as_mut_ptr().add(i).cast::<__m256i>(), m);
                    }
                    i += VEC_BYTES;
                }
                // Scalar tail.
                while i < len {
                    let b = src[i];
                    if b > dst[i] {
                        dst[i] = b;
                    }
                    i += 1;
                }
            }

            /// AVX2 harmonic-mean cardinality `alpha * m^2 / Z`.
            ///
            /// Loads 8 u8 registers per inner iteration, gathers their
            /// `2^-r` f64 values from [`super::POW2_NEG_LUT`] in two
            /// 4-wide AVX2 gathers, and accumulates into two parallel
            /// `__m256d` accumulators (8 doubles total) to break the
            /// dependency chain through `_mm256_add_pd`.
            ///
            /// # Safety
            ///
            /// The caller must ensure the current CPU supports AVX2.
            #[target_feature(enable = "avx2")]
            #[must_use]
            pub unsafe fn count_raw(registers: &[u8], alpha: f64) -> f64 {
                let m = registers.len() as f64;
                let lut_ptr = POW2_NEG_LUT.as_ptr();
                let mut acc0 = _mm256_setzero_pd();
                let mut acc1 = _mm256_setzero_pd();
                let mut i = 0_usize;
                let n = registers.len();

                // Inner loop: process 8 registers per iteration via
                // two 4-wide AVX2 gathers from the f64 LUT. Each gather
                // requires 4 i32 indices; we load the 8-byte group into
                // the low qword of an xmm, do PMOVZXBD on bytes 0..4
                // for the low gather, then byte-shift the xmm right by
                // 4 bytes (PSRLDQ) to move bytes 4..8 down into bytes
                // 0..4, and PMOVZXBD again for the high gather.
                while i + 8 <= n {
                    // Load 8 u8 register values into the low qword of
                    // an xmm register. SAFETY: `i + 8 <= n`.
                    let bytes_u64 = unsafe {
                        core::ptr::read_unaligned(registers.as_ptr().add(i).cast::<i64>())
                    };
                    let packed_xmm: __m128i = _mm_set_epi64x(0, bytes_u64);

                    // Low 4 bytes (registers i..i+4) → 4x i32 indices.
                    let lo_indices = _mm_cvtepu8_epi32(packed_xmm);
                    // High 4 bytes (registers i+4..i+8): byte-shift the
                    // xmm right by 4 to expose bytes 4..8 in the low
                    // 4-byte slot, then PMOVZXBD.
                    let shifted = _mm_srli_si128::<4>(packed_xmm);
                    let hi_indices = _mm_cvtepu8_epi32(shifted);

                    // SAFETY: indices ∈ 0..=255, LUT length 256, so
                    // gather offsets in bytes (= 8*index) are ≤ 2040.
                    // Intrinsic signature: <SCALE>(base_ptr, vindex).
                    let g0 = unsafe { _mm256_i32gather_pd::<8>(lut_ptr, lo_indices) };
                    let g1 = unsafe { _mm256_i32gather_pd::<8>(lut_ptr, hi_indices) };
                    acc0 = _mm256_add_pd(acc0, g0);
                    acc1 = _mm256_add_pd(acc1, g1);
                    i += 8;
                }

                // Horizontal sum the two parallel f64x4 accumulators.
                // SAFETY: AVX2 enabled by the enclosing target_feature.
                let mut sum = unsafe { horizontal_sum_pd(_mm256_add_pd(acc0, acc1)) };

                // Scalar tail.
                while i < n {
                    sum += POW2_NEG_LUT[registers[i] as usize];
                    i += 1;
                }

                alpha * m * m / sum
            }

            #[target_feature(enable = "avx2")]
            #[inline]
            unsafe fn horizontal_sum_pd(v: __m256d) -> f64 {
                // Cast the __m256d into two __m128d halves via the
                // integer-cast helpers, then sum lanes scalar-style.
                // We use a memory store for portability across rustc
                // versions where some hadd/permute intrinsics are not
                // stable on AVX2 alone.
                let mut tmp = [0.0_f64; 4];
                // SAFETY: tmp is f64-aligned (Rust guarantees this for
                // local arrays of f64); the store is 32-byte aligned
                // enough for VMOVUPD.
                unsafe {
                    core::arch::asm!(
                        "vmovupd [{ptr}], {vec}",
                        ptr = in(reg) tmp.as_mut_ptr(),
                        vec = in(ymm_reg) v,
                        options(nostack, preserves_flags),
                    );
                }
                tmp[0] + tmp[1] + tmp[2] + tmp[3]
            }
        }

        /// x86 AVX-512 kernels: `_mm512_max_epu8` for merge and an
        /// `_mm512_i32gather_pd`-driven 8-wide reduction for harmonic mean.
        #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
        pub mod avx512 {
            use super::POW2_NEG_LUT;

            #[cfg(target_arch = "x86")]
            use core::arch::x86::{
                __m128i, __m256i, __m512i, _mm_loadu_si128, _mm256_loadu_si256, _mm512_add_pd,
                _mm512_cvtepu8_epi32, _mm512_i32gather_pd, _mm512_loadu_si512, _mm512_max_epu8,
                _mm512_reduce_add_pd, _mm512_setzero_pd, _mm512_storeu_si512,
            };
            #[cfg(target_arch = "x86_64")]
            use core::arch::x86_64::{
                __m128i, __m256i, __m512i, _mm_loadu_si128, _mm256_loadu_si256, _mm512_add_pd,
                _mm512_cvtepu8_epi32, _mm512_i32gather_pd, _mm512_loadu_si512, _mm512_max_epu8,
                _mm512_reduce_add_pd, _mm512_setzero_pd, _mm512_storeu_si512,
            };

            /// 64 u8 registers per AVX-512 vector.
            const VEC_BYTES: usize = 64;

            /// Returns true when AVX-512BW (for VPMAXUB) is available
            /// at runtime. AVX-512BW is the byte-and-word AVX-512
            /// extension (Skylake-X, Ice Lake, Zen 4); without it the
            /// 64-byte VPMAXUB on `__m512i` is unavailable.
            #[cfg(feature = "std")]
            #[must_use]
            pub fn is_available() -> bool {
                std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
            }

            /// Returns true when AVX-512F + AVX-512BW are available.
            #[cfg(not(feature = "std"))]
            #[must_use]
            pub const fn is_available() -> bool {
                false
            }

            /// AVX-512 per-bucket max merge.
            ///
            /// # Safety
            ///
            /// The caller must ensure the current CPU supports AVX-512F
            /// and AVX-512BW and that `dst.len() == src.len()`.
            #[target_feature(enable = "avx512f,avx512bw")]
            pub unsafe fn merge(dst: &mut [u8], src: &[u8]) {
                debug_assert_eq!(dst.len(), src.len());
                let len = dst.len();
                let mut i = 0_usize;
                while i + VEC_BYTES <= len {
                    // SAFETY: `i + 64 <= len`; AVX-512F+BW enabled by
                    // the enclosing target_feature.
                    let a = unsafe { _mm512_loadu_si512(dst.as_ptr().add(i).cast::<__m512i>()) };
                    let b = unsafe { _mm512_loadu_si512(src.as_ptr().add(i).cast::<__m512i>()) };
                    let m = _mm512_max_epu8(a, b);
                    // SAFETY: same range bound as load above.
                    unsafe {
                        _mm512_storeu_si512(dst.as_mut_ptr().add(i).cast::<__m512i>(), m);
                    }
                    i += VEC_BYTES;
                }
                // Scalar tail (≤ 63 bytes). For valid HLL precisions
                // (4..=16) the register count is `2^p` ∈ {16..=65_536},
                // which is always a multiple of 16; the only sub-vector
                // tail occurs at `p ∈ {4, 5}` (16 / 32 registers).
                while i < len {
                    let b = src[i];
                    if b > dst[i] {
                        dst[i] = b;
                    }
                    i += 1;
                }
            }

            /// AVX-512 harmonic-mean cardinality `alpha * m^2 / Z`.
            ///
            /// Loads 16 u8 registers per inner iteration, zero-extends
            /// to 16x i32 with VPMOVZXBD, gathers via two
            /// `_mm512_i32gather_pd` calls (each 8-wide), and
            /// accumulates into two `__m512d` accumulators.
            ///
            /// # Safety
            ///
            /// The caller must ensure the current CPU supports AVX-512F.
            #[target_feature(enable = "avx512f,avx512bw")]
            #[must_use]
            pub unsafe fn count_raw(registers: &[u8], alpha: f64) -> f64 {
                let m = registers.len() as f64;
                let lut_ptr = POW2_NEG_LUT.as_ptr();
                let mut acc0 = _mm512_setzero_pd();
                let mut acc1 = _mm512_setzero_pd();
                let mut i = 0_usize;
                let n = registers.len();

                while i + 16 <= n {
                    // Load 16 u8 register bytes into the low half of a
                    // 32-byte zmm register, then zero-extend to 16x i32.
                    // SAFETY: `i + 16 <= n` bounds the 16-byte read.
                    let xmm =
                        unsafe { _mm_loadu_si128(registers.as_ptr().add(i).cast::<__m128i>()) };
                    // SAFETY: VPMOVZXBD widens 16x u8 → 16x i32; AVX-512F
                    // enabled by the enclosing target_feature.
                    let indices_512: __m512i = _mm512_cvtepu8_epi32(xmm);

                    // Split the 16x i32 indices into two 8x i32 halves
                    // for the two 8-wide gathers.
                    let mut tmp_idx = [0_i32; 16];
                    // SAFETY: tmp_idx is i32-aligned and 64 bytes long.
                    unsafe {
                        _mm512_storeu_si512(tmp_idx.as_mut_ptr().cast::<__m512i>(), indices_512);
                    }
                    let lo_idx_256 =
                        unsafe { _mm256_loadu_si256(tmp_idx.as_ptr().cast::<__m256i>()) };
                    let hi_idx_256 =
                        unsafe { _mm256_loadu_si256(tmp_idx.as_ptr().add(8).cast::<__m256i>()) };

                    // SAFETY: indices ∈ 0..=255; gather offsets ≤ 2040.
                    // Intrinsic signature for AVX-512:
                    // <SCALE>(offsets: __m256i, base: *const f64).
                    let g0 = unsafe { _mm512_i32gather_pd::<8>(lo_idx_256, lut_ptr) };
                    let g1 = unsafe { _mm512_i32gather_pd::<8>(hi_idx_256, lut_ptr) };
                    acc0 = _mm512_add_pd(acc0, g0);
                    acc1 = _mm512_add_pd(acc1, g1);
                    i += 16;
                }

                let mut sum = _mm512_reduce_add_pd(_mm512_add_pd(acc0, acc1));

                // Scalar tail.
                while i < n {
                    sum += POW2_NEG_LUT[registers[i] as usize];
                    i += 1;
                }

                alpha * m * m / sum
            }
        }

        /// AArch64 NEON kernels: `vmaxq_u8` for merge and a parallel-f64
        /// LUT-driven reduction for harmonic mean.
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        pub mod neon {
            use super::POW2_NEG_LUT;

            use core::arch::aarch64::{vld1q_u8, vmaxq_u8, vst1q_u8};

            /// 16 u8 registers per NEON vector.
            const VEC_BYTES: usize = 16;

            /// Returns true when NEON is available at runtime.
            ///
            /// NEON is mandatory in the AArch64 ABI; this helper exists
            /// for API symmetry with the x86 kernels.
            #[must_use]
            pub const fn is_available() -> bool {
                true
            }

            /// NEON per-bucket max merge.
            ///
            /// # Safety
            ///
            /// The caller must ensure the current CPU supports NEON and
            /// that `dst.len() == src.len()`.
            #[target_feature(enable = "neon")]
            pub unsafe fn merge(dst: &mut [u8], src: &[u8]) {
                debug_assert_eq!(dst.len(), src.len());
                let len = dst.len();
                let mut i = 0_usize;
                while i + VEC_BYTES <= len {
                    // SAFETY: `i + 16 <= len`; NEON enabled by the
                    // enclosing target_feature.
                    let a = unsafe { vld1q_u8(dst.as_ptr().add(i)) };
                    let b = unsafe { vld1q_u8(src.as_ptr().add(i)) };
                    let m = vmaxq_u8(a, b);
                    // SAFETY: same range bound as the load above.
                    unsafe { vst1q_u8(dst.as_mut_ptr().add(i), m) };
                    i += VEC_BYTES;
                }
                // Scalar tail.
                while i < len {
                    let b = src[i];
                    if b > dst[i] {
                        dst[i] = b;
                    }
                    i += 1;
                }
            }

            /// NEON harmonic-mean cardinality `alpha * m^2 / Z`.
            ///
            /// AArch64 NEON has no efficient gather instruction, so the
            /// inner loop runs four parallel scalar f64 accumulators
            /// (driving four LUT lookups per iteration). The pipelined
            /// AArch64 backends (Apple M-series, Graviton, Snapdragon)
            /// extract enough ILP from this to clear the same throughput
            /// the gather-based x86 backends hit.
            ///
            /// # Safety
            ///
            /// The caller must ensure the current CPU supports NEON.
            #[target_feature(enable = "neon")]
            #[must_use]
            pub unsafe fn count_raw(registers: &[u8], alpha: f64) -> f64 {
                let m = registers.len() as f64;
                let mut a0 = 0.0_f64;
                let mut a1 = 0.0_f64;
                let mut a2 = 0.0_f64;
                let mut a3 = 0.0_f64;
                let mut i = 0_usize;
                let n = registers.len();
                while i + 4 <= n {
                    a0 += POW2_NEG_LUT[registers[i] as usize];
                    a1 += POW2_NEG_LUT[registers[i + 1] as usize];
                    a2 += POW2_NEG_LUT[registers[i + 2] as usize];
                    a3 += POW2_NEG_LUT[registers[i + 3] as usize];
                    i += 4;
                }
                let mut sum = (a0 + a1) + (a2 + a3);
                while i < n {
                    sum += POW2_NEG_LUT[registers[i] as usize];
                    i += 1;
                }
                alpha * m * m / sum
            }
        }
    }
}

#[cfg(feature = "std")]
pub use hll::{HllMergeError, HyperLogLog, kernels as hll_kernels};

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

    #[cfg(feature = "std")]
    #[test]
    fn bloom_try_new_rejects_zero_parameters() {
        // Fallible parallel of the panic-on-zero contract.
        let err = BloomFilter::try_new(0, 4).unwrap_err();
        assert!(matches!(err, ApproxError::ZeroParameter { name: "bits" }));
        let err = BloomFilter::try_new(64, 0).unwrap_err();
        assert!(matches!(err, ApproxError::ZeroParameter { name: "k" }));
        // Sane inputs construct.
        assert!(BloomFilter::try_new(64, 4).is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn bloom_try_with_target_rejects_bad_inputs() {
        let err = BloomFilter::try_with_target(0, 0.01).unwrap_err();
        assert!(matches!(
            err,
            ApproxError::ZeroParameter {
                name: "expected_items"
            }
        ));
        for bad in [0.0, -0.1, 1.0, 2.0, f64::NAN] {
            let err = BloomFilter::try_with_target(1000, bad).unwrap_err();
            assert!(matches!(
                err,
                ApproxError::OutOfRangeFraction {
                    name: "target_fpr",
                    ..
                }
            ));
        }
        assert!(BloomFilter::try_with_target(1000, 0.01).is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_try_new_rejects_precision_out_of_range() {
        for bad in [0_u32, 3, 17, 32, 64] {
            let err = HyperLogLog::try_new(bad).unwrap_err();
            assert!(matches!(
                err,
                ApproxError::PrecisionOutOfRange {
                    requested,
                    min: 4,
                    max: 16,
                } if requested == bad
            ));
        }
        for ok in [4_u32, 8, 12, 16] {
            assert!(
                HyperLogLog::try_new(ok).is_ok(),
                "precision {ok} should be ok"
            );
        }
    }

    // ----- HyperLogLog SIMD merge + count parity (Sprint 44) -----
    //
    // SIMD merge is bit-exact with scalar (per-bucket max is associative,
    // commutative, and order-independent). SIMD harmonic mean is
    // numerically equivalent within a tight relative tolerance because
    // floating-point summation is not strictly associative; the SIMD
    // backends use a different reduction tree than the scalar reference.

    #[cfg(feature = "std")]
    fn deterministic_hash_stream(seed: u64, count: usize) -> Vec<u64> {
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        (0..count)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_F491_4F6C_DD1D)
            })
            .collect()
    }

    #[cfg(feature = "std")]
    fn build_hll_with_inserts(precision: u32, seed: u64, n: usize) -> HyperLogLog {
        let mut hll = HyperLogLog::new(precision);
        for h in deterministic_hash_stream(seed, n) {
            hll.insert_hash(h);
        }
        hll
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_merge_simd_is_byte_identical_to_scalar() {
        // Cover every supported precision with a non-trivial workload.
        for precision in [4_u32, 5, 6, 8, 10, 12, 14, 16] {
            for n in [0_usize, 1, 7, 64, 1024, 64_000] {
                let mut a_scalar = build_hll_with_inserts(precision, 0xA1A1, n);
                let b = build_hll_with_inserts(precision, 0xB2B2, n.saturating_add(13));
                let mut a_simd = a_scalar.clone();

                a_scalar.merge(&b);
                a_simd.merge_simd(&b);

                assert_eq!(
                    a_scalar.register_bytes(),
                    a_simd.register_bytes(),
                    "SIMD merge diverged at precision={precision}, n={n}"
                );
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_merge_simd_is_idempotent_for_self_merge() {
        let original = build_hll_with_inserts(12, 0xC0FFEE, 5_000);
        let mut copy = original.clone();
        copy.merge_simd(&original);
        assert_eq!(
            copy.register_bytes(),
            original.register_bytes(),
            "self-merge must be a no-op"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_merge_simd_handles_alternating_extremes() {
        // Edge case: alternating large/small registers stresses the
        // per-byte max kernel's tail handling and lane-shuffle paths.
        let precision = 10;
        let m = 1_usize << precision;
        let mut a = HyperLogLog::new(precision);
        let mut b = HyperLogLog::new(precision);

        // Manually clobber the registers to a known pattern via the
        // public merge contract: insert hashes that drive each register
        // toward its target rank.
        for i in 0..m {
            // a: zero at even, max-rank at odd.
            if i % 2 == 1 {
                let hash = ((i as u64) << (64 - precision)) | (1_u64 << (63 - precision));
                a.insert_hash(hash);
            }
            // b: max-rank at even, zero at odd (mirrored).
            if i % 2 == 0 {
                let hash = ((i as u64) << (64 - precision)) | (1_u64 << (63 - precision));
                b.insert_hash(hash);
            }
        }

        let mut scalar = a.clone();
        let mut simd = a.clone();
        scalar.merge(&b);
        simd.merge_simd(&b);
        assert_eq!(
            scalar.register_bytes(),
            simd.register_bytes(),
            "alternating-extremes pattern diverged"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_count_simd_matches_scalar_within_tolerance() {
        // Tolerance budget: floating-point summation reorder. Each f64
        // add can introduce ~ulp_2^-r relative error. For ~65k registers
        // the worst-case accumulated error is well under 1e-12 relative.
        let tolerance = 1e-12;
        for precision in [4_u32, 5, 6, 8, 10, 12, 14, 16] {
            for n in [0_usize, 1, 100, 5_000, 100_000] {
                let hll = build_hll_with_inserts(precision, 0xDEAD_BEEF, n);
                let scalar = hll.count_raw();
                let simd = hll.count_simd();

                if scalar == 0.0 || scalar.is_infinite() {
                    assert_eq!(
                        scalar.is_finite(),
                        simd.is_finite(),
                        "finite mismatch p={precision} n={n}"
                    );
                    continue;
                }
                let rel_err = (scalar - simd).abs() / scalar.abs();
                assert!(
                    rel_err < tolerance,
                    "count_simd diverged: precision={precision} n={n} \
                     scalar={scalar} simd={simd} rel_err={rel_err}"
                );
            }
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_count_simd_handles_empty_sketch() {
        // All registers zero ⇒ Z = m (every term is 2^0 = 1.0) ⇒
        // raw = alpha * m^2 / m = alpha * m. Both paths must agree.
        for precision in [4_u32, 8, 12, 16] {
            let hll = HyperLogLog::new(precision);
            let scalar = hll.count_raw();
            let simd = hll.count_simd();
            assert_eq!(
                scalar, simd,
                "empty-sketch count diverged at precision={precision}"
            );
            // Sanity: alpha * m for the all-zero sketch.
            let m = (1_usize << precision) as f64;
            let expected = hll.alpha() * m;
            assert!(
                (scalar - expected).abs() < 1e-9,
                "empty raw count formula mismatch: got {scalar}, expected {expected}"
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_count_simd_handles_max_registers() {
        // Maxed-out registers (rank = 64 - precision + 1 for an all-zero
        // hash chunk). This pushes 2^-r toward subnormals and exercises
        // the LUT entries far from zero.
        for precision in [4_u32, 8, 12] {
            let mut hll = HyperLogLog::new(precision);
            // Force every register to its maximum rank by inserting a
            // hash whose low (64-p) bits are all zero, for each
            // bucket index in turn.
            let m = 1_usize << precision;
            for idx in 0..m {
                let hash = (idx as u64) << (64 - precision);
                hll.insert_hash(hash);
            }
            let scalar = hll.count_raw();
            let simd = hll.count_simd();
            assert!(
                (scalar - simd).abs() / scalar.abs() < 1e-12,
                "maxed-register count diverged at precision={precision}: \
                 scalar={scalar} simd={simd}"
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_try_merge_rejects_precision_mismatch() {
        let mut a = HyperLogLog::new(10);
        let b = HyperLogLog::new(12);
        let err = a.try_merge(&b).unwrap_err();
        assert!(matches!(
            err,
            hll::HllMergeError::PrecisionMismatch { lhs: 10, rhs: 12 }
        ));
        let err = a.try_merge_simd(&b).unwrap_err();
        assert!(matches!(
            err,
            hll::HllMergeError::PrecisionMismatch { lhs: 10, rhs: 12 }
        ));
        // Equal-precision case succeeds.
        let c = HyperLogLog::new(10);
        assert!(a.try_merge(&c).is_ok());
        assert!(a.try_merge_simd(&c).is_ok());
    }

    #[cfg(feature = "std")]
    #[test]
    #[should_panic(expected = "HyperLogLog merge requires equal precision")]
    fn hll_merge_simd_panics_on_precision_mismatch() {
        let mut a = HyperLogLog::new(10);
        let b = HyperLogLog::new(12);
        a.merge_simd(&b);
    }

    #[cfg(feature = "std")]
    #[test]
    fn hll_pow2_neg_lut_matches_powi_for_valid_ranks() {
        // Sanity: the LUT must agree with `2.0_f64.powi(-r)` for every
        // rank a real HLL register can hold (0..=64).
        for r in 0_u8..=64 {
            let lut = hll_kernels::POW2_NEG_LUT[r as usize];
            let powi = 2.0_f64.powi(-i32::from(r));
            assert_eq!(
                lut.to_bits(),
                powi.to_bits(),
                "LUT[{r}] = {lut:e} != powi(-{r}) = {powi:e}"
            );
        }
    }
}

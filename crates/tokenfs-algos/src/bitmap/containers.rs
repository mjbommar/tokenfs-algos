//! Roaring container types.
//!
//! Roaring partitions a `u32` set into 16-bit "high keys" (which container
//! within a top-level array of containers) and 16-bit "low keys" (which
//! 16-bit value within the container). This module ships only the
//! container-level primitives — the high-level Roaring `Bitmap` type is
//! out of scope per `docs/v0.2_planning/11_BITMAP.md` § 8.
//!
//! Three container shapes:
//!
//! * [`BitmapContainer`] — dense 65536-bit bitmap, exactly 1024 `u64`
//!   words = 8 KiB. Used when cardinality > 4096.
//! * [`ArrayContainer`] — sorted, deduplicated `Vec<u16>`, length ≤ 4096.
//!   Used when cardinality ≤ 4096.
//! * [`RunContainer`] — sorted `(start, length_minus_one)` pairs covering
//!   non-overlapping intervals. Used when run-encoding compresses well.
//!
//! The 4096-element threshold is the canonical CRoaring break-even point:
//! at 4096 elements an array container occupies `4096 × 2 = 8 KiB` —
//! exactly the size of a bitmap container. Beyond 4096 the bitmap form is
//! always at least as compact.
//!
//! ## Run length encoding
//!
//! Following the CRoaring convention, [`RunContainer`] stores `(start,
//! length_minus_one)` pairs; the run covers values `start..=start +
//! length_minus_one`, total `length_minus_one + 1` values. The `_minus_one`
//! offset lets a single-value run fit in the same compact format
//! (`length_minus_one = 0`) without wasting an extra bit.

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec};

use core::fmt;

/// Reasons a container's data violated a structural invariant.
///
/// Returned by the validating `try_from_vec` constructors on
/// [`ArrayContainer`] and [`RunContainer`]. Each variant carries the
/// offending index so that callers (or property-test shrinkers) can
/// pinpoint the failure without re-walking the input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ContainerInvariantError {
    /// `ArrayContainer`: data is not strictly ascending sorted.
    ArrayUnsorted {
        /// Index of the first element that broke the ordering, i.e. the
        /// element where `data[i - 1] >= data[i]`.
        offending_index: usize,
    },
    /// `ArrayContainer`: data exceeds the 4096 array→bitmap promotion
    /// threshold. Caller should promote to [`BitmapContainer`].
    ArrayOverThreshold {
        /// Length of the supplied array.
        len: usize,
        /// Threshold the array exceeded ([`ARRAY_MAX_CARDINALITY`]).
        threshold: usize,
    },
    /// `RunContainer`: runs are not sorted by `start`.
    RunsUnsorted {
        /// Index of the first run that broke the ordering, i.e. the run
        /// whose `start` is `<= runs[i - 1].start`.
        offending_index: usize,
    },
    /// `RunContainer`: two adjacent runs overlap or are not coalesced
    /// (i.e. `runs[i - 1]` ends at or after `runs[i].start - 1`).
    RunsNotCoalesced {
        /// Index of the run that should have merged into its predecessor.
        offending_index: usize,
    },
    /// `RunContainer`: a run's `start + length_minus_one + 1` exceeds the
    /// 16-bit value space (65 536).
    RunOverflowsValueSpace {
        /// Index of the offending run.
        offending_index: usize,
        /// Run start.
        start: u16,
        /// Run length minus one (CRoaring on-disk convention).
        length_minus_one: u16,
    },
}

impl fmt::Display for ContainerInvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArrayUnsorted { offending_index } => write!(
                f,
                "ArrayContainer: data not strictly ascending at index {offending_index}"
            ),
            Self::ArrayOverThreshold { len, threshold } => write!(
                f,
                "ArrayContainer: length {len} exceeds promotion threshold {threshold}"
            ),
            Self::RunsUnsorted { offending_index } => write!(
                f,
                "RunContainer: runs not sorted by start at index {offending_index}"
            ),
            Self::RunsNotCoalesced { offending_index } => write!(
                f,
                "RunContainer: runs at index {offending_index} overlap or are adjacent (must be coalesced)"
            ),
            Self::RunOverflowsValueSpace {
                offending_index,
                start,
                length_minus_one,
            } => write!(
                f,
                "RunContainer: run at index {offending_index} (start={start}, length_minus_one={length_minus_one}) overflows the 16-bit value space"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ContainerInvariantError {}

/// Number of `u64` words in a [`BitmapContainer`].
pub const BITMAP_WORDS: usize = 1024;

/// Number of bits represented by a [`BitmapContainer`] (`1024 × 64 = 65_536`).
pub const BITMAP_BITS: usize = BITMAP_WORDS * 64;

/// Maximum cardinality for an [`ArrayContainer`] before promotion to a
/// [`BitmapContainer`] becomes more compact.
///
/// At 4096 entries an [`ArrayContainer`] occupies `4096 × 2 = 8 KiB` —
/// exactly the size of a [`BitmapContainer`]. Beyond this point the
/// bitmap form is always at least as compact.
pub const ARRAY_MAX_CARDINALITY: usize = 4096;

/// Dense 65 536-bit bitmap container.
///
/// Exactly 1024 little-endian `u64` words = 8 KiB. Bit `v` (with `v` in
/// `0..65_536`) lives at `words[v >> 6] & (1 << (v & 63))`.
///
/// The buffer is `Box`-owned so the type is move-cheap; the words array
/// itself is large enough that callers should avoid putting the
/// container on the stack in kernel contexts. SIMD kernels write into a
/// caller-provided `&mut Self` (no internal allocation) so the kernel
/// stack budget stays small.
#[derive(Clone, Debug)]
pub struct BitmapContainer {
    /// 1024 little-endian `u64` words (8 KiB total).
    pub words: Box<[u64; BITMAP_WORDS]>,
}

impl BitmapContainer {
    /// Returns an empty bitmap container (all 1024 words set to zero).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            words: Box::new([0_u64; BITMAP_WORDS]),
        }
    }

    /// Returns a bitmap container from the supplied 1024-word array.
    #[must_use]
    pub fn from_words(words: Box<[u64; BITMAP_WORDS]>) -> Self {
        Self { words }
    }

    /// Returns whether bit `v` is set.
    #[must_use]
    pub fn contains(&self, v: u16) -> bool {
        let v = v as usize;
        (self.words[v >> 6] >> (v & 63)) & 1 == 1
    }

    /// Inserts bit `v` and returns whether the bit was newly set.
    pub fn insert(&mut self, v: u16) -> bool {
        let v = v as usize;
        let word = &mut self.words[v >> 6];
        let mask = 1_u64 << (v & 63);
        let was_set = (*word & mask) != 0;
        *word |= mask;
        !was_set
    }

    /// Removes bit `v` and returns whether the bit had been set.
    pub fn remove(&mut self, v: u16) -> bool {
        let v = v as usize;
        let word = &mut self.words[v >> 6];
        let mask = 1_u64 << (v & 63);
        let was_set = (*word & mask) != 0;
        *word &= !mask;
        was_set
    }

    /// Returns the number of set bits.
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        // Reuse the SIMD-accelerated popcount kernel from `bits` so the
        // bitmap container's cardinality benefits from VPOPCNTQ on
        // hardware that supports it.
        crate::bits::popcount_u64_slice(self.words.as_slice()) as u32
    }

    /// Returns whether the bitmap has no set bits.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        // Short-circuit: a single set word disqualifies emptiness without
        // a full popcount.
        self.words.iter().all(|w| *w == 0)
    }

    /// Iterates the set bits as `u16` values in ascending order.
    pub fn iter(&self) -> BitmapIter<'_> {
        // Seed with `current_word_index = -1` (encoded as wrap-around
        // via `0` then bumped on first refill) and the first refill in
        // `next()` lands at `words[0]`. We set `current = 0` so the
        // first iteration enters the refill path immediately.
        let words = self.words.as_slice();
        let first = if words.is_empty() { 0 } else { words[0] };
        BitmapIter {
            words,
            current_word_index: 0,
            current: first,
        }
    }

    /// Materialises the set bits into a sorted `Vec<u16>`.
    #[must_use]
    pub fn to_array(&self) -> Vec<u16> {
        let card = self.cardinality() as usize;
        let mut out = Vec::with_capacity(card);
        for v in self.iter() {
            out.push(v);
        }
        out
    }
}

impl Default for BitmapContainer {
    fn default() -> Self {
        Self::empty()
    }
}

impl PartialEq for BitmapContainer {
    fn eq(&self, other: &Self) -> bool {
        self.words[..] == other.words[..]
    }
}

impl Eq for BitmapContainer {}

/// Iterator over the set bits of a [`BitmapContainer`].
///
/// Yields `u16` values in ascending order. The implementation tracks the
/// remaining bits within the current word so that hot-loop iteration is
/// dominated by `trailing_zeros` calls rather than per-bit branches.
///
/// Invariant during iteration: `current_word_index` is the index in
/// `words` of the word currently being unpacked into `current`. After
/// the words run out, `current_word_index == words.len()` and `current
/// == 0`, so subsequent `next()` calls return `None`.
#[derive(Debug)]
pub struct BitmapIter<'a> {
    words: &'a [u64],
    /// Index into `words` of the word held in `current`. Starts at 0
    /// with `current = 0` so the first `next()` loads `words[0]`.
    current_word_index: usize,
    /// Bits remaining in the word at `current_word_index`. Cleared as
    /// each set bit is yielded; refilled from the next word when zero.
    current: u64,
}

impl Iterator for BitmapIter<'_> {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        loop {
            if self.current != 0 {
                let bit = self.current.trailing_zeros();
                self.current &= self.current - 1;
                let value = (self.current_word_index as u32) * 64 + bit;
                debug_assert!(value < BITMAP_BITS as u32);
                return Some(value as u16);
            }
            // Refill from the next word.
            self.current_word_index += 1;
            if self.current_word_index >= self.words.len() {
                // Walk past the last word; subsequent calls return None.
                self.current_word_index = self.words.len();
                return None;
            }
            self.current = self.words[self.current_word_index];
        }
    }
}

/// Sparse sorted-`u16` container.
///
/// Invariants:
///
/// * `data` is sorted in strictly ascending order.
/// * `data.len() <= ARRAY_MAX_CARDINALITY` (= 4096) at the conceptual
///   level. The type does **not** enforce the cap so kernels can build
///   intermediate over-cap arrays before promoting to a
///   [`BitmapContainer`]; the per-op dispatch wrappers in
///   [`crate::bitmap::union`] and friends apply the promotion rule.
///
/// # Construction contract
///
/// External callers **MUST** construct via [`ArrayContainer::try_from_vec`]
/// (or [`ArrayContainer::empty`] / [`ArrayContainer::from_sorted`] when the
/// invariants are upheld by construction). The `data` field is only
/// `pub(crate)` so the SIMD kernels and dispatch wrappers inside this
/// crate can construct directly along merge-emitting paths that produce
/// sorted output by design; **direct field construction from outside the
/// crate is not supported** and is enforced by the compiler.
///
/// For read-only access to the underlying values, use
/// [`ArrayContainer::data`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArrayContainer {
    /// Sorted `u16` values.
    ///
    /// `pub(crate)` so SIMD kernels inside this crate can construct
    /// directly when the sortedness invariant is upheld by the producer.
    /// External callers must go through [`ArrayContainer::try_from_vec`].
    pub(crate) data: Vec<u16>,
}

impl ArrayContainer {
    /// Returns an empty array container.
    #[must_use]
    pub const fn empty() -> Self {
        Self { data: Vec::new() }
    }

    /// Returns an array container from the supplied sorted slice.
    ///
    /// # Panics
    ///
    /// Panics if `data` is not strictly ascending.
    #[must_use]
    pub fn from_sorted(data: Vec<u16>) -> Self {
        debug_assert!(
            data.windows(2).all(|w| w[0] < w[1]),
            "ArrayContainer::from_sorted: input is not strictly ascending"
        );
        Self { data }
    }

    /// Validating constructor — recommended path for untrusted input.
    ///
    /// Walks `data` once, verifying:
    ///
    /// 1. each adjacent pair satisfies `data[i - 1] < data[i]`
    ///    (strictly ascending), and
    /// 2. the length does not exceed [`ARRAY_MAX_CARDINALITY`] (4096).
    ///
    /// Returns [`ContainerInvariantError::ArrayUnsorted`] or
    /// [`ContainerInvariantError::ArrayOverThreshold`] on violation; the
    /// returned error carries the offending index / length so callers
    /// can recover or report.
    ///
    /// # Errors
    ///
    /// * [`ContainerInvariantError::ArrayUnsorted`] if `data[i - 1] >= data[i]`.
    /// * [`ContainerInvariantError::ArrayOverThreshold`] if
    ///   `data.len() > ARRAY_MAX_CARDINALITY`.
    pub fn try_from_vec(data: Vec<u16>) -> Result<Self, ContainerInvariantError> {
        if data.len() > ARRAY_MAX_CARDINALITY {
            return Err(ContainerInvariantError::ArrayOverThreshold {
                len: data.len(),
                threshold: ARRAY_MAX_CARDINALITY,
            });
        }
        // Strictly ascending check. We walk indices 1..len so the
        // reported offending_index points at the right-hand element of
        // the violating pair; that matches the convention used by the
        // run-validator below.
        for i in 1..data.len() {
            if data[i - 1] >= data[i] {
                return Err(ContainerInvariantError::ArrayUnsorted { offending_index: i });
            }
        }
        Ok(Self { data })
    }

    /// Returns a read-only view of the sorted `u16` values.
    ///
    /// This is the public accessor for what was previously the `pub data`
    /// field. The slice is borrowed from the container and remains valid
    /// for the lifetime of the borrow; the returned reference is
    /// **immutable** and cannot be used to violate the sortedness or
    /// length invariants. To mutate the contents, drop the container and
    /// rebuild via [`ArrayContainer::try_from_vec`] (or, inside the
    /// crate, via direct field construction along an invariant-preserving
    /// path).
    #[must_use]
    pub fn data(&self) -> &[u16] {
        &self.data
    }

    /// Returns the cardinality (`data.len()`).
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        self.data.len() as u32
    }

    /// Returns whether the container has no values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns whether `v` is present (binary search).
    #[must_use]
    pub fn contains(&self, v: u16) -> bool {
        self.data.binary_search(&v).is_ok()
    }
}

/// Roaring-style container enum — bitmap, array, or run.
///
/// Each variant carries its own representation; the [`Container`]
/// methods route every pairwise operation through the dispatch tables
/// in [`crate::bitmap::intersect`], [`crate::bitmap::union`],
/// [`crate::bitmap::difference`], and [`crate::bitmap::xor`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Container {
    /// Dense 65 536-bit bitmap.
    Bitmap(BitmapContainer),
    /// Sparse sorted `u16` array.
    Array(ArrayContainer),
    /// Sorted run-pairs (start, length-1).
    Run(RunContainer),
}

/// Sorted-run container.
///
/// Stores non-overlapping runs as `(start, length_minus_one)` pairs. A
/// run covers values `start..=start + length_minus_one` for a total of
/// `length_minus_one + 1` values. The `_minus_one` offset matches the
/// CRoaring on-disk format and lets a single-value run be stored as
/// `(v, 0)` without wasting representation.
///
/// Invariants:
///
/// * `runs` is sorted by `start` in ascending order.
/// * Runs do not overlap or touch: for adjacent entries `i, i+1`,
///   `runs[i].0 + runs[i].1 + 1 < runs[i+1].0` (the merged form has been
///   coalesced so no two runs are adjacent or overlapping).
/// * `runs[i].0 + runs[i].1 + 1 <= 65536`, i.e. each run fits inside the
///   16-bit value space.
///
/// # Construction contract
///
/// External callers **MUST** construct via [`RunContainer::try_from_vec`]
/// (or [`RunContainer::empty`] / [`RunContainer::from_runs`] when the
/// invariants are upheld by construction). The `runs` field is only
/// `pub(crate)` so the SIMD kernels and dispatch wrappers inside this
/// crate can construct directly along run-merging paths that emit
/// coalesced output by design; **direct field construction from outside
/// the crate is not supported** and is enforced by the compiler.
///
/// For read-only access to the underlying runs, use
/// [`RunContainer::runs`].
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunContainer {
    /// Sorted, coalesced `(start, length_minus_one)` pairs.
    ///
    /// `pub(crate)` so SIMD kernels inside this crate can construct
    /// directly when the sortedness, coalescing, and value-space
    /// invariants are upheld by the producer. External callers must go
    /// through [`RunContainer::try_from_vec`].
    pub(crate) runs: Vec<(u16, u16)>,
}

impl RunContainer {
    /// Returns an empty run container.
    #[must_use]
    pub const fn empty() -> Self {
        Self { runs: Vec::new() }
    }

    /// Returns a run container from the supplied sorted, coalesced runs.
    ///
    /// In debug builds, asserts the invariants; in release builds the
    /// caller is trusted.
    #[must_use]
    pub fn from_runs(runs: Vec<(u16, u16)>) -> Self {
        debug_assert!(Self::runs_are_valid(&runs));
        Self { runs }
    }

    /// Validating constructor — recommended path for untrusted input.
    ///
    /// Walks `runs` once, verifying for each entry `(start, length_minus_one)`:
    ///
    /// 1. `start + length_minus_one + 1 <= 65 536` (fits in 16-bit value
    ///    space).
    /// 2. For all `i > 0`, `runs[i].0 > runs[i - 1].0` (sorted by start).
    /// 3. For all `i > 0`, `runs[i].0 > runs[i - 1].0 + runs[i - 1].1 + 1`
    ///    (no overlap and no adjacency — i.e. the run list is coalesced).
    ///
    /// Order of checks is `RunOverflowsValueSpace` (per-run, before any
    /// adjacency math is meaningful) → `RunsUnsorted` →
    /// `RunsNotCoalesced`. Errors carry the offending index so callers
    /// can pinpoint the failure.
    ///
    /// # Errors
    ///
    /// * [`ContainerInvariantError::RunOverflowsValueSpace`] if
    ///   `start + length_minus_one + 1 > 65 536`.
    /// * [`ContainerInvariantError::RunsUnsorted`] if `runs[i].0 <= runs[i - 1].0`.
    /// * [`ContainerInvariantError::RunsNotCoalesced`] if `runs[i].0` falls
    ///   inside or immediately after the predecessor's run.
    pub fn try_from_vec(runs: Vec<(u16, u16)>) -> Result<Self, ContainerInvariantError> {
        for i in 0..runs.len() {
            let (start, len_m1) = runs[i];
            // Per-run value-space check first: u32 widening avoids overflow
            // on the `+1` and matches the runs_are_valid math exactly.
            let end = u32::from(start) + u32::from(len_m1);
            if end > u32::from(u16::MAX) {
                return Err(ContainerInvariantError::RunOverflowsValueSpace {
                    offending_index: i,
                    start,
                    length_minus_one: len_m1,
                });
            }
            if i > 0 {
                let (prev_start, prev_len_m1) = runs[i - 1];
                if start <= prev_start {
                    return Err(ContainerInvariantError::RunsUnsorted { offending_index: i });
                }
                let prev_end = u32::from(prev_start) + u32::from(prev_len_m1);
                if u32::from(start) <= prev_end + 1 {
                    return Err(ContainerInvariantError::RunsNotCoalesced { offending_index: i });
                }
            }
        }
        Ok(Self { runs })
    }

    /// Returns a read-only view of the sorted, coalesced run-pairs.
    ///
    /// This is the public accessor for what was previously the `pub runs`
    /// field. The slice is borrowed from the container and remains valid
    /// for the lifetime of the borrow; the returned reference is
    /// **immutable** and cannot be used to violate the sortedness,
    /// coalescing, or value-space invariants. To mutate the contents,
    /// drop the container and rebuild via [`RunContainer::try_from_vec`]
    /// (or, inside the crate, via direct field construction along an
    /// invariant-preserving path).
    #[must_use]
    pub fn runs(&self) -> &[(u16, u16)] {
        &self.runs
    }

    /// Returns the cardinality (sum of `length_minus_one + 1` over all runs).
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        let mut sum: u32 = 0;
        for &(_, len_m1) in &self.runs {
            sum += u32::from(len_m1) + 1;
        }
        sum
    }

    /// Returns whether the container has no runs.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Returns whether `v` lies inside any run.
    #[must_use]
    pub fn contains(&self, v: u16) -> bool {
        // Binary-search for the largest run start ≤ v, then check end.
        let v = u32::from(v);
        match self
            .runs
            .binary_search_by(|&(start, _)| u32::from(start).cmp(&v))
        {
            Ok(_) => true,
            Err(0) => false,
            Err(i) => {
                let (start, len_m1) = self.runs[i - 1];
                u32::from(start) + u32::from(len_m1) >= v
            }
        }
    }

    /// Validates the invariants for a run-list.
    #[must_use]
    fn runs_are_valid(runs: &[(u16, u16)]) -> bool {
        for i in 0..runs.len() {
            let (start, len_m1) = runs[i];
            let end = u32::from(start) + u32::from(len_m1);
            if end > u32::from(u16::MAX) {
                return false;
            }
            if i > 0 {
                let prev_end = u32::from(runs[i - 1].0) + u32::from(runs[i - 1].1);
                if u32::from(start) <= prev_end + 1 {
                    // Adjacent or overlapping: should have been coalesced.
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

    use super::*;
    // `Vec` and `vec!` are not in the no-std prelude; alias them from
    // `alloc` for the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    #[test]
    fn bitmap_container_set_get_clear() {
        let mut bc = BitmapContainer::empty();
        assert_eq!(bc.cardinality(), 0);
        assert!(bc.is_empty());

        assert!(bc.insert(0));
        assert!(!bc.insert(0));
        assert!(bc.insert(63));
        assert!(bc.insert(64));
        assert!(bc.insert(65535));
        assert_eq!(bc.cardinality(), 4);
        assert!(bc.contains(0));
        assert!(bc.contains(63));
        assert!(bc.contains(64));
        assert!(bc.contains(65535));
        assert!(!bc.contains(1));
        assert!(!bc.contains(62));
        assert!(!bc.contains(65534));

        assert!(bc.remove(63));
        assert!(!bc.remove(63));
        assert_eq!(bc.cardinality(), 3);
    }

    #[test]
    fn bitmap_iter_in_order() {
        let mut bc = BitmapContainer::empty();
        for v in [0_u16, 5, 63, 64, 100, 4096, 8192, 65535] {
            bc.insert(v);
        }
        let collected: Vec<u16> = bc.iter().collect();
        assert_eq!(collected, [0, 5, 63, 64, 100, 4096, 8192, 65535]);
    }

    #[test]
    fn bitmap_to_array_round_trips() {
        let mut bc = BitmapContainer::empty();
        for v in (0_u32..1024).step_by(7).map(|v| v as u16) {
            bc.insert(v);
        }
        let array = bc.to_array();
        assert_eq!(array.len() as u32, bc.cardinality());
        for &v in &array {
            assert!(bc.contains(v));
        }
    }

    #[test]
    fn array_container_constructors() {
        let ac = ArrayContainer::from_sorted(vec![1, 3, 5, 65535]);
        assert_eq!(ac.cardinality(), 4);
        assert!(!ac.is_empty());
        assert!(ac.contains(3));
        assert!(ac.contains(65535));
        assert!(!ac.contains(2));
        assert!(!ac.contains(0));
    }

    #[test]
    fn run_container_cardinality_and_contains() {
        let rc = RunContainer::from_runs(vec![(0, 9), (100, 0), (1000, 99)]);
        // Run 1: 0..=9 = 10 values; run 2: 100..=100 = 1; run 3: 1000..=1099 = 100.
        assert_eq!(rc.cardinality(), 10 + 1 + 100);
        assert!(rc.contains(0));
        assert!(rc.contains(9));
        assert!(!rc.contains(10));
        assert!(rc.contains(100));
        assert!(!rc.contains(101));
        assert!(rc.contains(1000));
        assert!(rc.contains(1099));
        assert!(!rc.contains(1100));
    }

    #[test]
    fn run_container_validates_no_touching() {
        // (0, 0) covers value 0 only; (1, 0) covers value 1 — these
        // should have been coalesced to a single run.
        assert!(!RunContainer::runs_are_valid(&[(0, 0), (1, 0)]));
        // (0, 4) covers 0..=4; (5, 0) covers value 5 — adjacent, must
        // have been coalesced.
        assert!(!RunContainer::runs_are_valid(&[(0, 4), (5, 0)]));
        // (0, 4) then (6, 0) leaves a gap at value 5 — valid.
        assert!(RunContainer::runs_are_valid(&[(0, 4), (6, 0)]));
    }

    #[test]
    fn run_container_validates_max_value() {
        // (65530, 5) covers 65530..=65535 — fits.
        assert!(RunContainer::runs_are_valid(&[(65530, 5)]));
        // (65530, 6) would extend to 65536 which is out of u16 — invalid.
        assert!(!RunContainer::runs_are_valid(&[(65530, 6)]));
    }

    // ---- try_from_vec validating constructors --------------------------

    #[test]
    fn array_try_from_vec_happy_path() {
        let ac = ArrayContainer::try_from_vec(vec![1, 3, 5, 65535]).unwrap();
        assert_eq!(ac.cardinality(), 4);
        assert!(ac.contains(3));
        assert!(ac.contains(65535));
        assert!(!ac.contains(2));
    }

    #[test]
    fn array_try_from_vec_empty_is_ok() {
        let ac = ArrayContainer::try_from_vec(Vec::new()).unwrap();
        assert!(ac.is_empty());
    }

    #[test]
    fn array_try_from_vec_rejects_unsorted() {
        // 7 < 5 violates strictly ascending at index 2.
        let err = ArrayContainer::try_from_vec(vec![1, 5, 3, 9]).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::ArrayUnsorted { offending_index: 2 }
        );
    }

    #[test]
    fn array_try_from_vec_rejects_duplicate() {
        // Duplicate at index 1: 5 is not strictly greater than 5.
        let err = ArrayContainer::try_from_vec(vec![5, 5]).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::ArrayUnsorted { offending_index: 1 }
        );
    }

    #[test]
    fn array_try_from_vec_rejects_over_threshold() {
        // 4097 elements exceeds the 4096 promotion threshold; values can
        // be sorted yet still fail the cap.
        let too_big: Vec<u16> = (0..ARRAY_MAX_CARDINALITY as u32 + 1)
            .map(|v| v as u16)
            .collect();
        let len = too_big.len();
        let err = ArrayContainer::try_from_vec(too_big).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::ArrayOverThreshold {
                len,
                threshold: ARRAY_MAX_CARDINALITY,
            }
        );
    }

    #[test]
    fn array_try_from_vec_at_threshold_is_ok() {
        // Exactly 4096 elements (the cap) is allowed.
        let exact: Vec<u16> = (0..ARRAY_MAX_CARDINALITY as u32)
            .map(|v| v as u16)
            .collect();
        assert!(ArrayContainer::try_from_vec(exact).is_ok());
    }

    #[test]
    fn run_try_from_vec_happy_path() {
        let rc = RunContainer::try_from_vec(vec![(0, 9), (100, 0), (1000, 99)]).unwrap();
        assert_eq!(rc.cardinality(), 10 + 1 + 100);
        assert!(rc.contains(0));
        assert!(rc.contains(1099));
        assert!(!rc.contains(1100));
    }

    #[test]
    fn run_try_from_vec_empty_is_ok() {
        let rc = RunContainer::try_from_vec(Vec::new()).unwrap();
        assert!(rc.is_empty());
    }

    #[test]
    fn run_try_from_vec_rejects_overflow() {
        // (65530, 6) extends to 65536 which is one past u16::MAX.
        let err = RunContainer::try_from_vec(vec![(65530, 6)]).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::RunOverflowsValueSpace {
                offending_index: 0,
                start: 65530,
                length_minus_one: 6,
            }
        );
    }

    #[test]
    fn run_try_from_vec_rejects_unsorted() {
        // (200, 0) followed by (100, 0): start went backwards.
        let err = RunContainer::try_from_vec(vec![(200, 0), (100, 0)]).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::RunsUnsorted { offending_index: 1 }
        );
    }

    #[test]
    fn run_try_from_vec_rejects_duplicate_start() {
        // Two runs with the same start — fails the sorted check (not
        // strictly ascending by start).
        let err = RunContainer::try_from_vec(vec![(100, 0), (100, 0)]).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::RunsUnsorted { offending_index: 1 }
        );
    }

    #[test]
    fn run_try_from_vec_rejects_overlap() {
        // (0, 9) covers 0..=9; (5, 0) starts inside the previous run.
        let err = RunContainer::try_from_vec(vec![(0, 9), (5, 0)]).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::RunsNotCoalesced { offending_index: 1 }
        );
    }

    #[test]
    fn run_try_from_vec_rejects_adjacency() {
        // (0, 4) covers 0..=4; (5, 0) is immediately adjacent — should
        // have been coalesced into (0, 5).
        let err = RunContainer::try_from_vec(vec![(0, 4), (5, 0)]).unwrap_err();
        assert_eq!(
            err,
            ContainerInvariantError::RunsNotCoalesced { offending_index: 1 }
        );
    }

    #[test]
    fn run_try_from_vec_at_max_value_is_ok() {
        // (65530, 5) covers exactly 65530..=65535 — fits at the limit.
        assert!(RunContainer::try_from_vec(vec![(65530, 5)]).is_ok());
    }

    #[test]
    fn container_invariant_error_display_smoke() {
        // Smoke-coverage of the Display impl so the message format
        // doesn't drift silently.
        #[cfg(all(feature = "alloc", not(feature = "std")))]
        use alloc::format;
        let s = format!(
            "{}",
            ContainerInvariantError::ArrayUnsorted { offending_index: 7 }
        );
        assert!(s.contains("not strictly ascending"));
        assert!(s.contains("7"));
    }

    // ---- Read-only accessor contract -----------------------------------
    //
    // The `data` / `runs` fields became `pub(crate)` in the Layer 4
    // hardening pass; external callers must construct via
    // `try_from_vec` (or the trusted `from_sorted` / `from_runs`
    // helpers) and read via `data()` / `runs()`. The tests below pin
    // the accessor surface so a regression that re-widens visibility or
    // changes the slice shape would fail visibly.

    #[test]
    fn array_data_accessor_returns_constructor_input() {
        let input = vec![1_u16, 3, 5, 65535];
        let ac = ArrayContainer::try_from_vec(input.clone()).unwrap();
        // The accessor returns a borrow tracking the same elements that
        // were handed to `try_from_vec`; callers cannot mutate through
        // the borrow because it is `&[u16]`, not `&mut [u16]`.
        assert_eq!(ac.data(), input.as_slice());
        assert_eq!(ac.data().len(), 4);
    }

    #[test]
    fn array_data_accessor_matches_from_sorted_path() {
        // `from_sorted` and `try_from_vec` should produce identical
        // observable state over the accessor.
        let values = vec![10_u16, 20, 30, 40];
        let ac_sorted = ArrayContainer::from_sorted(values.clone());
        let ac_validated = ArrayContainer::try_from_vec(values.clone()).unwrap();
        assert_eq!(ac_sorted.data(), ac_validated.data());
        assert_eq!(ac_sorted.data(), values.as_slice());
    }

    #[test]
    fn array_data_accessor_on_empty_is_empty_slice() {
        let ac = ArrayContainer::empty();
        assert!(ac.data().is_empty());
    }

    #[test]
    fn run_runs_accessor_returns_constructor_input() {
        let input = vec![(0_u16, 9_u16), (100, 0), (1000, 99)];
        let rc = RunContainer::try_from_vec(input.clone()).unwrap();
        assert_eq!(rc.runs(), input.as_slice());
        assert_eq!(rc.runs().len(), 3);
    }

    #[test]
    fn run_runs_accessor_matches_from_runs_path() {
        let runs = vec![(0_u16, 4_u16), (10, 4)];
        let rc_trusted = RunContainer::from_runs(runs.clone());
        let rc_validated = RunContainer::try_from_vec(runs.clone()).unwrap();
        assert_eq!(rc_trusted.runs(), rc_validated.runs());
        assert_eq!(rc_trusted.runs(), runs.as_slice());
    }

    #[test]
    fn run_runs_accessor_on_empty_is_empty_slice() {
        let rc = RunContainer::empty();
        assert!(rc.runs().is_empty());
    }

    // The compiler-enforced part of the construction contract — that
    // `ArrayContainer { data: ... }` and `RunContainer { runs: ... }`
    // do not type-check from outside the crate — cannot be expressed as
    // a runtime test. The intra-crate tests in this module *do*
    // exercise the validating constructors on every interesting input
    // shape (above), so the contract is end-to-end exercised: external
    // callers get a compile error if they try the field form, and
    // internal kernels keep direct field access for the
    // invariants-by-construction emit paths.
}

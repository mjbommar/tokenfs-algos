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
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArrayContainer {
    /// Sorted `u16` values.
    pub data: Vec<u16>,
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
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RunContainer {
    /// Sorted, coalesced `(start, length_minus_one)` pairs.
    pub runs: Vec<(u16, u16)>,
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
    use super::*;

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
}

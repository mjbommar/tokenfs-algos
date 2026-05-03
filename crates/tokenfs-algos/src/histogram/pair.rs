//! Dense byte-pair histograms.
//!
//! This is the allocation-free exact H2 building block. It keeps all 65,536
//! byte-pair counters in fixed storage, so callers should reuse the struct for
//! hot streaming paths rather than repeatedly placing it on small stacks.
//!
//! ## Kernel-stack hazard
//!
//! [`BytePairHistogram`] occupies `256 * 256 * 4 + 8 = 262_152` bytes — about
//! **256 KiB**. That is well above the typical kernel stack budget (8-16 KiB,
//! see `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`) and unsafe to allocate
//! by value on a kernel/FUSE call frame. Kernel-adjacent callers should
//! either:
//!
//! - construct via [`BytePairHistogram::with_scratch`] using caller-owned
//!   `&mut BytePairCountsScratch` storage (heap-allocated `Box`, mmap,
//!   thread-local pool, postgres memory context, kernel `kmalloc`'d slab,
//!   etc.); or
//! - use the lazy-clear [`BytePairScratch`] form via
//!   [`crate::entropy::joint::h2_pairs_with_scratch`].

use core::fmt;

/// Exact dense histogram of adjacent byte pairs.
///
/// **Stack footprint**: `256 * 256 * 4 + 8` ≈ 256 KiB. Use
/// [`with_scratch`](Self::with_scratch) — paired with caller-owned
/// [`BytePairCountsScratch`] storage — on kernel-adjacent / shallow-stack
/// paths to keep the dense table off the call frame.
#[derive(Clone, Eq, PartialEq)]
pub struct BytePairHistogram {
    counts: [u32; 256 * 256],
    observations: u64,
}

/// Caller-provided scratch storage for [`BytePairHistogram::with_scratch`].
///
/// Wraps the dense `256 * 256` u32 counter table. Allocate this via a path
/// that does not stack-materialise the inner array (`Box::new_uninit`,
/// thread-local, postgres memory context, kernel `kmalloc`'d slab) and
/// reuse across calls; the stack-safe constructor will reset its contents
/// on every entry.
///
/// At 256 KiB the inner array is well above kernel stack limits — its
/// public field is intentionally exposed so callers can pre-allocate the
/// backing storage on the heap (e.g. `Box::<BytePairCountsScratch>::new_uninit()`)
/// and pass a `&mut` borrow to the constructor below without ever
/// materialising the array on a small stack.
#[repr(transparent)]
pub struct BytePairCountsScratch {
    /// Dense `256 * 256` u32 counter table. The outer wrapper exists so
    /// the type is named in public APIs and so any future fields can be
    /// added without breaking call sites.
    pub counts: [u32; 256 * 256],
}

impl BytePairCountsScratch {
    /// Returns a writable reference to the inner counter array. Useful
    /// when the caller needs to clear or inspect the raw counts.
    #[must_use]
    pub fn counts_mut(&mut self) -> &mut [u32; 256 * 256] {
        &mut self.counts
    }
}

/// Reusable exact byte-pair scratch space with lazy counter reset.
///
/// `BytePairScratch` is intended for hot paths that repeatedly compute exact
/// adjacent-pair statistics. It keeps dense counters, but reset is proportional
/// to the next call's active pairs rather than 65,536 counters. The struct is
/// large by design; allocate it once per worker/file/stream and reuse it.
#[derive(Clone, Eq, PartialEq)]
pub struct BytePairScratch {
    counts: [u32; 256 * 256],
    pair_stamps: [u32; 256 * 256],
    active_pairs: [u16; 256 * 256],
    active_pair_len: usize,
    predecessor_counts: [u32; 256],
    predecessor_stamps: [u32; 256],
    active_predecessors: [u8; 256],
    active_predecessor_len: usize,
    generation: u32,
    observations: u64,
}

impl BytePairHistogram {
    /// Creates an empty pair histogram.
    ///
    /// **Stack footprint**: returns `Self` (~256 KiB) by value.
    /// Available only with `feature = "userspace"` (audit-R9 #5).
    /// Kernel/FUSE callers should use
    /// [`with_scratch`](Self::with_scratch) — paired with caller-owned
    /// [`BytePairCountsScratch`] storage — to keep the dense table off
    /// the call frame entirely.
    #[cfg(feature = "userspace")]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; 256 * 256],
            observations: 0,
        }
    }

    /// Builds a pair histogram from adjacent byte pairs in `bytes`.
    ///
    /// **Stack footprint**: returns `Self` (~256 KiB) by value.
    /// Available only with `feature = "userspace"` (audit-R9 #5).
    /// Kernel/FUSE callers should use
    /// [`with_scratch`](Self::with_scratch) instead, paired with
    /// caller-owned [`BytePairCountsScratch`] storage; that path keeps
    /// the dense counter table off the call frame entirely.
    #[cfg(feature = "userspace")]
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut histogram = Self::new();
        histogram.add_bytes(bytes);
        histogram
    }

    /// Heap-free / kernel-safe constructor: counts adjacent byte pairs
    /// from `bytes` directly into `scratch`, then returns a thin
    /// `BytePairHistogramView<'_>` that borrows the populated counts.
    ///
    /// This is the kernel-adjacent counterpart to [`from_bytes`](Self::from_bytes):
    /// the dense `256 * 256` u32 counter table lives in caller-provided
    /// scratch (typically a heap `Box<BytePairCountsScratch>`, an mmap'd
    /// arena, a postgres memory context, or a kernel slab) so no large
    /// array is materialised on the call frame. The returned view exposes
    /// the same read-only API as [`BytePairHistogram`]
    /// (`count_pair` / `counts` / `observations` / `is_empty` /
    /// `iter_nonzero`) and is consumed by
    /// [`crate::entropy::joint::h2_from_pair_view`].
    ///
    /// `scratch` is fully overwritten on entry — its prior contents are
    /// discarded — so callers may reuse a single buffer across many calls
    /// without pre-clearing it.
    ///
    /// Mirrors the §156 caller-provided-scratch convention used by
    /// [`crate::similarity::kernels_gather::build_table_from_seeds_into`]
    /// (audit-R5).
    pub fn with_scratch<'a>(
        bytes: &[u8],
        scratch: &'a mut BytePairCountsScratch,
    ) -> BytePairHistogramView<'a> {
        // Clear the caller-provided counter table; we cannot trust prior
        // contents.
        for slot in scratch.counts.iter_mut() {
            *slot = 0;
        }
        let mut observations = 0_u64;
        for pair in bytes.windows(2) {
            let index = pair_index(pair[0], pair[1]);
            scratch.counts[index] += 1;
            observations += 1;
        }
        BytePairHistogramView {
            counts: &scratch.counts,
            observations,
        }
    }

    /// Clears all counters.
    pub fn clear(&mut self) {
        self.counts = [0; 256 * 256];
        self.observations = 0;
    }

    /// Adds adjacent byte pairs from `bytes`.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        for pair in bytes.windows(2) {
            self.add_pair(pair[0], pair[1]);
        }
    }

    /// Adds one byte pair.
    pub fn add_pair(&mut self, first: u8, second: u8) {
        self.counts[pair_index(first, second)] += 1;
        self.observations += 1;
    }

    /// Returns the count for one byte pair.
    #[must_use]
    pub const fn count_pair(&self, first: u8, second: u8) -> u32 {
        self.counts[pair_index(first, second)]
    }

    /// Returns all dense pair counts.
    #[must_use]
    pub const fn counts(&self) -> &[u32; 256 * 256] {
        &self.counts
    }

    /// Returns the number of observed adjacent pairs.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns true when no pairs were observed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.observations == 0
    }

    /// Iterates over non-zero pair counts as `(first, second, count)`.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (u8, u8, u32)> + '_ {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(index, count)| {
                let first = (index >> 8) as u8;
                let second = (index & 0xff) as u8;
                (first, second, *count)
            })
    }
}

// `Default` delegates to `new`, which is gated on `userspace` (audit-R9 #5).
// Trait impls cannot be cfg-conditional in a way that preserves bound code,
// so the impl itself is feature-gated and kernel/FUSE callers must use
// `BytePairHistogram::with_scratch` directly.
#[cfg(feature = "userspace")]
impl Default for BytePairHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed view over a populated dense byte-pair counter table.
///
/// Returned by [`BytePairHistogram::with_scratch`]; exposes the same
/// read-only surface as [`BytePairHistogram`] without owning the
/// underlying counter array. The lifetime parameter ties the view to the
/// caller-provided [`BytePairCountsScratch`] storage so the borrow
/// checker prevents dangling references.
#[derive(Clone, Copy)]
pub struct BytePairHistogramView<'a> {
    counts: &'a [u32; 256 * 256],
    observations: u64,
}

impl<'a> BytePairHistogramView<'a> {
    /// Returns the count for one byte pair.
    #[must_use]
    pub fn count_pair(&self, first: u8, second: u8) -> u32 {
        self.counts[pair_index(first, second)]
    }

    /// Returns all dense pair counts.
    #[must_use]
    pub fn counts(&self) -> &'a [u32; 256 * 256] {
        self.counts
    }

    /// Returns the number of observed adjacent pairs.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns true when no pairs were observed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.observations == 0
    }

    /// Iterates over non-zero pair counts as `(first, second, count)`.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (u8, u8, u32)> + '_ {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(index, count)| {
                let first = (index >> 8) as u8;
                let second = (index & 0xff) as u8;
                (first, second, *count)
            })
    }
}

impl<'a> fmt::Debug for BytePairHistogramView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytePairHistogramView")
            .field("observations", &self.observations)
            .field(
                "distinct_pairs",
                &self.counts.iter().filter(|count| **count != 0).count(),
            )
            .finish_non_exhaustive()
    }
}

impl BytePairScratch {
    /// Creates empty reusable scratch state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; 256 * 256],
            pair_stamps: [0; 256 * 256],
            active_pairs: [0; 256 * 256],
            active_pair_len: 0,
            predecessor_counts: [0; 256],
            predecessor_stamps: [0; 256],
            active_predecessors: [0; 256],
            active_predecessor_len: 0,
            generation: 1,
            observations: 0,
        }
    }

    /// Lazily clears all counters.
    ///
    /// This does not touch every pair counter unless the internal generation
    /// wraps, which is effectively never for normal process lifetimes.
    pub fn clear(&mut self) {
        self.active_pair_len = 0;
        self.active_predecessor_len = 0;
        self.observations = 0;
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.pair_stamps = [0; 256 * 256];
            self.predecessor_stamps = [0; 256];
            self.generation = 1;
        }
    }

    /// Clears and adds adjacent byte pairs from `bytes`.
    pub fn reset_and_add_bytes(&mut self, bytes: &[u8]) {
        self.clear();
        self.add_bytes(bytes);
    }

    /// Adds adjacent byte pairs from `bytes`.
    pub fn add_bytes(&mut self, bytes: &[u8]) {
        for pair in bytes.windows(2) {
            self.add_pair(pair[0], pair[1]);
        }
    }

    /// Adds one byte pair.
    pub fn add_pair(&mut self, first: u8, second: u8) {
        let index = pair_index(first, second);
        if self.pair_stamps[index] != self.generation {
            self.pair_stamps[index] = self.generation;
            self.counts[index] = 0;
            self.active_pairs[self.active_pair_len] = index as u16;
            self.active_pair_len += 1;
        }
        self.counts[index] += 1;

        let predecessor = first as usize;
        if self.predecessor_stamps[predecessor] != self.generation {
            self.predecessor_stamps[predecessor] = self.generation;
            self.predecessor_counts[predecessor] = 0;
            self.active_predecessors[self.active_predecessor_len] = first;
            self.active_predecessor_len += 1;
        }
        self.predecessor_counts[predecessor] += 1;
        self.observations += 1;
    }

    /// Returns the count for one byte pair in the current generation.
    #[must_use]
    pub fn count_pair(&self, first: u8, second: u8) -> u32 {
        let index = pair_index(first, second);
        if self.pair_stamps[index] == self.generation {
            self.counts[index]
        } else {
            0
        }
    }

    /// Returns the predecessor count for one first byte.
    #[must_use]
    pub fn predecessor_count(&self, first: u8) -> u32 {
        let index = first as usize;
        if self.predecessor_stamps[index] == self.generation {
            self.predecessor_counts[index]
        } else {
            0
        }
    }

    /// Returns the number of observed adjacent pairs.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns the number of distinct active pairs.
    #[must_use]
    pub const fn distinct_pairs(&self) -> usize {
        self.active_pair_len
    }

    /// Returns true when no pairs were observed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.observations == 0
    }

    /// Iterates over non-zero pair counts as `(first, second, count)`.
    pub fn iter_nonzero(&self) -> impl Iterator<Item = (u8, u8, u32)> + '_ {
        self.active_pairs[..self.active_pair_len]
            .iter()
            .copied()
            .map(|index| {
                let index = index as usize;
                let first = (index >> 8) as u8;
                let second = (index & 0xff) as u8;
                (first, second, self.counts[index])
            })
    }

    /// Iterates over non-zero predecessor counts as `(byte, count)`.
    pub fn iter_predecessors(&self) -> impl Iterator<Item = (u8, u32)> + '_ {
        self.active_predecessors[..self.active_predecessor_len]
            .iter()
            .copied()
            .map(|byte| (byte, self.predecessor_counts[byte as usize]))
    }
}

impl Default for BytePairScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for BytePairScratch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytePairScratch")
            .field("observations", &self.observations)
            .field("distinct_pairs", &self.active_pair_len)
            .field("distinct_predecessors", &self.active_predecessor_len)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for BytePairHistogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BytePairHistogram")
            .field("observations", &self.observations)
            .field(
                "distinct_pairs",
                &self.counts.iter().filter(|count| **count != 0).count(),
            )
            .finish_non_exhaustive()
    }
}

const fn pair_index(first: u8, second: u8) -> usize {
    ((first as usize) << 8) | second as usize
}

#[cfg(test)]
mod tests {
    use super::{BytePairCountsScratch, BytePairHistogram, BytePairScratch};
    // `Box`/`Vec` are not in the no-std prelude; alias them from `alloc`
    // for the alloc-only build (audit-R6 #164, audit-R8 followup).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    /// Heap-allocate a zeroed `BytePairCountsScratch` without ever
    /// materialising the 256 KiB inner array on the test stack.
    fn alloc_zeroed_pair_counts_scratch() -> Box<BytePairCountsScratch> {
        use core::mem::MaybeUninit;
        let mut uninit: Box<MaybeUninit<BytePairCountsScratch>> = Box::new_uninit();
        // SAFETY: `uninit.as_mut_ptr()` is a valid, properly-aligned
        // pointer to writable heap storage of `sizeof::<BytePairCountsScratch>()`
        // bytes. After `write_bytes(0)` every u32 is the bit-pattern 0
        // (a valid u32), so `assume_init` is sound.
        unsafe {
            core::ptr::write_bytes(uninit.as_mut_ptr().cast::<u32>(), 0, 256 * 256);
            uninit.assume_init()
        }
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn counts_adjacent_pairs() {
        let histogram = BytePairHistogram::from_bytes(b"ababa");
        assert_eq!(histogram.observations(), 4);
        assert_eq!(histogram.count_pair(b'a', b'b'), 2);
        assert_eq!(histogram.count_pair(b'b', b'a'), 2);
        assert_eq!(histogram.count_pair(b'a', b'a'), 0);
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn clear_resets_pair_histogram() {
        let mut histogram = BytePairHistogram::from_bytes(b"abcdef");
        histogram.clear();
        assert!(histogram.is_empty());
        assert_eq!(histogram.counts().iter().sum::<u32>(), 0);
    }

    #[test]
    fn scratch_reuses_storage_with_lazy_clear() {
        let mut scratch = Box::new(BytePairScratch::new());
        scratch.reset_and_add_bytes(b"ababa");
        assert_eq!(scratch.observations(), 4);
        assert_eq!(scratch.count_pair(b'a', b'b'), 2);
        assert_eq!(scratch.predecessor_count(b'a'), 2);

        scratch.reset_and_add_bytes(b"zzzz");
        assert_eq!(scratch.observations(), 3);
        assert_eq!(scratch.count_pair(b'a', b'b'), 0);
        assert_eq!(scratch.count_pair(b'z', b'z'), 3);
        assert_eq!(scratch.distinct_pairs(), 1);
    }

    /// `BytePairHistogram::with_scratch` produces a view whose pair
    /// counts and observation total match the by-value
    /// `BytePairHistogram::from_bytes` form bit-exactly.
    ///
    /// This guards audit-R8 #6a: the scratch path is the kernel-safe
    /// constructor and must not drift in semantics from the legacy
    /// by-value form.
    #[cfg(feature = "userspace")]
    #[test]
    fn with_scratch_view_matches_from_bytes() {
        let bytes = b"abacabadabacaba_zz_yyyy";
        let by_value = BytePairHistogram::from_bytes(bytes);

        let mut scratch = alloc_zeroed_pair_counts_scratch();
        let view = BytePairHistogram::with_scratch(bytes, &mut scratch);

        assert_eq!(view.observations(), by_value.observations());
        for first in 0_u16..=255 {
            for second in 0_u16..=255 {
                assert_eq!(
                    view.count_pair(first as u8, second as u8),
                    by_value.count_pair(first as u8, second as u8),
                    "first={first} second={second}",
                );
            }
        }
        assert_eq!(view.counts(), by_value.counts());
        assert_eq!(view.is_empty(), by_value.is_empty());

        // iter_nonzero matches as a multiset.
        let mut nz_view: Vec<(u8, u8, u32)> = view.iter_nonzero().collect();
        let mut nz_owned: Vec<(u8, u8, u32)> = by_value.iter_nonzero().collect();
        nz_view.sort();
        nz_owned.sort();
        assert_eq!(nz_view, nz_owned);
    }

    /// Empty / single-byte input is a no-op: zero observations, no
    /// populated pairs, no UB on the freshly cleared scratch.
    #[test]
    fn with_scratch_empty_input_is_noop() {
        let mut scratch = alloc_zeroed_pair_counts_scratch();
        let view = BytePairHistogram::with_scratch(b"", &mut scratch);
        assert_eq!(view.observations(), 0);
        assert!(view.is_empty());
        assert_eq!(view.iter_nonzero().count(), 0);

        // Single-byte input has no adjacent pairs either.
        let view = BytePairHistogram::with_scratch(b"a", &mut scratch);
        assert_eq!(view.observations(), 0);
        assert!(view.is_empty());
    }

    /// Reusing the same scratch across calls overwrites prior contents
    /// — the view from the second call sees only the second input's
    /// pairs.
    #[test]
    fn with_scratch_overwrites_prior_contents() {
        let mut scratch = alloc_zeroed_pair_counts_scratch();
        let _ = BytePairHistogram::with_scratch(b"zzzz", &mut scratch);
        let view = BytePairHistogram::with_scratch(b"ababa", &mut scratch);
        assert_eq!(view.observations(), 4);
        assert_eq!(view.count_pair(b'a', b'b'), 2);
        assert_eq!(view.count_pair(b'b', b'a'), 2);
        assert_eq!(view.count_pair(b'z', b'z'), 0);
    }
}

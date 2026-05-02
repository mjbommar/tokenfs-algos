//! Locality-improving orderings for graph- and point-shaped data.
//!
//! See `docs/v0.2_planning/14_PERMUTATION.md` for the spec. This module
//! ships the shared [`Permutation`] type plus the CSR adjacency input
//! type [`CsrGraph`], and the **build-time** ordering primitives that
//! produce a [`Permutation`] from such inputs.
//!
//! ## Sprint 11-13 / Sprint 47-49 status
//!
//! Phase B4 of `01_PHASES.md` lands [`rcm()`] (Reverse Cuthill-McKee).
//! Phase B5 lands `hilbert_2d` / `hilbert_nd` behind the
//! `permutation_hilbert` Cargo feature (vendor wrappers around the
//! `fast_hilbert` and `hilbert` crates per spec § 4). Sprint 47-49 of
//! Phase D1 lands [`rabbit_order()`] as a single-pass sequential
//! baseline; the SIMD modularity-gain inner loop (Sprint 50-52) and
//! concurrent merging (Sprint 53-55) are follow-on sprints.
//!
//! ## Deployment posture
//!
//! Per `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`:
//!
//! * **Permutation construction** ([`rcm()`], `hilbert_2d` /
//!   `hilbert_nd`, [`rabbit_order()`]): build-time only. These
//!   algorithms allocate large work buffers (BFS queue, key-array
//!   sort, dendrogram) that cannot be made stack-only. Never runs in
//!   kernel.
//! * **[`Permutation::apply`] / [`Permutation::apply_into`]**:
//!   kernel-safe. Stateless borrowed slices, allocation-free for the
//!   `_into` form, SIMD-friendly via gather.
//!
//! The `tokenfs_writer` constructs the permutation at image-build time
//! (userspace, ample memory) and writes the resulting `[u32; n]` array
//! into the sealed image. The `tokenfs_reader` (FUSE or kernel) loads
//! the array from the mmap'd image and applies it on demand.
//!
//! ## Public API
//!
//! ```
//! use tokenfs_algos::permutation::{CsrGraph, Permutation, rcm};
//!
//! // Star graph: vertex 0 connects to 1, 2, 3.
//! let offsets = [0_u32, 3, 4, 5, 6];
//! let neighbors = [1_u32, 2, 3, 0, 0, 0];
//! let graph = CsrGraph { n: 4, offsets: &offsets, neighbors: &neighbors };
//! let perm = rcm(graph);
//! assert_eq!(perm.len(), 4);
//! // The result is a valid permutation: each id 0..n appears exactly once.
//! let mut seen = [false; 4];
//! for &new_id in perm.as_slice() {
//!     seen[new_id as usize] = true;
//! }
//! assert!(seen.iter().all(|b| *b));
//! ```

// The whole `permutation` module is gated on `std OR alloc` in
// `lib.rs`. Pull `Vec` / `vec!` from the right namespace for both
// configurations: `std` re-exports them in the prelude already, and
// the no-std + alloc path needs an explicit import.
#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "permutation_hilbert")]
pub mod hilbert;
pub mod rabbit;
pub mod rcm;

#[cfg(feature = "permutation_hilbert")]
pub use hilbert::{hilbert_2d, hilbert_nd};
pub use rabbit::rabbit_order;
pub use rcm::rcm;

/// A permutation array.
///
/// `Permutation(perm)` represents the mapping `perm[old_id] = new_id`,
/// i.e. element `i` of the input is moved to position `perm[i]` in the
/// output. `perm` is a permutation of `0..n` where `n = perm.len()`.
///
/// The type is intentionally a thin wrapper around `Vec<u32>` so the
/// underlying buffer can be moved into / out of an image manifest with
/// no copy. All non-trivial operations are checked: see
/// [`Permutation::try_from_vec`] for the validating constructor.
///
/// ## Layout
///
/// `perm[i]` is always within `0..n`. The constructors enforce this
/// invariant; mutating accessors are not exposed. Two permutations
/// compose via `apply`: `b.apply(&a.apply(src))` applies `a` then `b`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Permutation(Vec<u32>);

impl Permutation {
    /// Returns the identity permutation of length `n` — `perm[i] = i`.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        // u32::try_from is preferred but the cast is safe for any
        // `n <= u32::MAX as usize`; permutations longer than that are
        // out of scope (vertex IDs are u32).
        assert!(
            n <= u32::MAX as usize,
            "permutation length exceeds u32 vertex space"
        );
        let mut perm = Vec::with_capacity(n);
        for i in 0..n {
            // SAFETY: i < n <= u32::MAX as usize, so the cast cannot overflow.
            perm.push(i as u32);
        }
        Self(perm)
    }

    /// Constructs a [`Permutation`] from a raw `Vec<u32>` WITHOUT
    /// validating that it contains every index `0..n` exactly once.
    ///
    /// # Safety
    ///
    /// The caller must ensure `perm` is a valid permutation:
    /// * `perm.len()` is the size of the underlying set and
    ///   `perm.len() <= u32::MAX as usize`.
    /// * Every index in `0..perm.len()` appears exactly once in `perm`.
    ///
    /// Constructing a [`Permutation`] that violates these invariants
    /// does not trigger undefined behaviour directly (later operations
    /// bounds-check) but causes panics, infinite loops in
    /// [`Permutation::inverse`], or silently wrong results in
    /// [`Permutation::apply`]. For untrusted input, prefer
    /// [`Permutation::try_from_vec`], which validates the invariant in
    /// O(n) time.
    #[must_use]
    pub unsafe fn from_vec_unchecked(perm: Vec<u32>) -> Self {
        Self(perm)
    }

    /// Constructs a [`Permutation`] from a raw `Vec<u32>`, checking that
    /// it is a valid permutation.
    ///
    /// Returns `None` if `perm` is not a valid permutation of
    /// `0..perm.len()` or if its length exceeds `u32::MAX as usize`.
    #[must_use]
    pub fn try_from_vec(perm: Vec<u32>) -> Option<Self> {
        if perm.len() > u32::MAX as usize {
            return None;
        }
        let n = perm.len();
        let mut seen = vec![false; n];
        for &id in &perm {
            let id = id as usize;
            if id >= n || seen[id] {
                return None;
            }
            seen[id] = true;
        }
        Some(Self(perm))
    }

    /// Returns the inverse permutation.
    ///
    /// If `p` maps `old_id -> new_id`, then `p.inverse()` maps
    /// `new_id -> old_id`. `p.apply(p.inverse().apply(src)) == src` and
    /// vice versa.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let n = self.0.len();
        let mut inv = vec![0_u32; n];
        for (old_id, &new_id) in self.0.iter().enumerate() {
            // The constructor invariants guarantee `new_id < n`; the
            // unchecked indexing is correct under those invariants but
            // we use the checked form so callers who broke the safety
            // contract of `from_vec_unchecked` still get a panic
            // instead of out-of-bounds writes.
            inv[new_id as usize] = old_id as u32;
        }
        Self(inv)
    }

    /// Applies the permutation, allocating the output.
    ///
    /// `out[perm[i]] = src[i]`. Returns a new `Vec<T>` of length
    /// `self.len()`. Every output slot is written exactly once because
    /// the permutation invariants guarantee every `new_id` in
    /// `0..self.len()` appears exactly once.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != self.len()`.
    ///
    /// For zero-length inputs, the empty `Vec` is returned without
    /// inspecting `src`.
    ///
    /// # Safety contract
    ///
    /// If `self` is a valid bijection on `0..n` (constructed via
    /// [`Permutation::try_from_vec`] or [`Permutation::identity`]), this
    /// writes every dst slot exactly once and is safe.
    ///
    /// If `self` was constructed via [`Permutation::from_vec_unchecked`]
    /// (an `unsafe` constructor) and is NOT a valid bijection:
    ///
    ///   * Duplicate destination indices: dst slots written multiple
    ///     times; OTHER dst slots remain unmodified, leaking whatever
    ///     was in `dst` on entry. Because [`Permutation::apply`] seeds
    ///     `dst` with `src[0]` before writing, the visible leak is
    ///     bounded to data that was already in `src`. Kernel/FUSE
    ///     consumers loading from untrusted storage MUST validate via
    ///     [`Permutation::validate_no_alloc`] or use
    ///     [`Permutation::try_apply_into_strict`] before trusting the
    ///     output.
    ///   * Out-of-range indices: panics in debug, undefined-but-bounded
    ///     behaviour in release (the `Vec` bounds-check still triggers).
    #[must_use]
    pub fn apply<T: Copy>(&self, src: &[T]) -> Vec<T> {
        assert_eq!(
            src.len(),
            self.0.len(),
            "Permutation::apply: src.len() ({}) != perm.len() ({})",
            src.len(),
            self.0.len()
        );
        let n = self.0.len();
        if n == 0 {
            return Vec::new();
        }
        // Initialise with a clone of `src[0]` to satisfy the `Vec`
        // invariant that all elements be valid `T`. Every slot is
        // overwritten by the permutation loop below — the bijection
        // invariant on `Permutation` guarantees that. Cloning is a
        // no-op for `Copy` types but stays inside the `T: Copy` bound
        // the spec calls out (no `T: Default` requirement).
        let mut dst = vec![src[0]; n];
        self.apply_into(src, &mut dst);
        dst
    }

    /// Applies the permutation into a caller-provided buffer.
    ///
    /// `dst[perm[i]] = src[i]`. The caller controls the allocation, so
    /// this form is kernel-safe (no internal Vec).
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != self.len()` or `dst.len() < self.len()`.
    ///
    /// # Safety contract
    ///
    /// If `self` is a valid bijection on `0..n` (constructed via
    /// [`Permutation::try_from_vec`] or [`Permutation::identity`]), this
    /// writes every dst slot exactly once and is safe.
    ///
    /// If `self` was constructed via [`Permutation::from_vec_unchecked`]
    /// (an `unsafe` constructor) and is NOT a valid bijection:
    ///
    ///   * Duplicate destination indices: dst slots written multiple
    ///     times; OTHER dst slots remain unmodified, leaking whatever
    ///     was in `dst` on entry. **Kernel/FUSE consumers loading from
    ///     untrusted storage MUST validate via
    ///     [`Permutation::validate_no_alloc`] or use
    ///     [`Permutation::try_apply_into_strict`] before relying on
    ///     `dst`.**
    ///   * Out-of-range indices: panics in debug, undefined-but-bounded
    ///     behaviour in release (the `Vec` bounds-check still triggers).
    pub fn apply_into<T: Copy>(&self, src: &[T], dst: &mut [T]) {
        assert_eq!(
            src.len(),
            self.0.len(),
            "Permutation::apply_into: src.len() ({}) != perm.len() ({})",
            src.len(),
            self.0.len()
        );
        assert!(
            dst.len() >= self.0.len(),
            "Permutation::apply_into: dst.len() ({}) < perm.len() ({})",
            dst.len(),
            self.0.len()
        );
        for (i, &new_id) in self.0.iter().enumerate() {
            dst[new_id as usize] = src[i];
        }
    }

    /// Validates that this [`Permutation`] is a valid bijection on
    /// `0..self.len()`, using caller-provided u64-bitset scratch
    /// instead of allocating a `Vec<bool>`.
    ///
    /// `scratch` must have at least `self.len().div_ceil(64)` u64
    /// words. The first `len().div_ceil(64)` words of `scratch` are
    /// zeroed on entry. Returns `true` if valid, `false` otherwise
    /// (length overflow, out-of-range index, or duplicate index).
    ///
    /// Use this from kernel-mode consumers that loaded a
    /// [`Permutation`] from untrusted on-disk data and need to
    /// validate before calling [`Permutation::apply_into`]. The
    /// bitset-based check matches [`Permutation::try_from_vec`] but
    /// avoids the heap allocation it performs.
    ///
    /// # Panics
    ///
    /// Panics if `scratch.len()` is below the required word count
    /// (`self.len().div_ceil(64)`). The kernel-safe sibling that
    /// returns an error instead of panicking is
    /// [`Permutation::try_apply_into_strict`], which also performs
    /// the apply.
    pub fn validate_no_alloc(&self, scratch: &mut [u64]) -> bool {
        let n = self.0.len();
        if n > u32::MAX as usize {
            return false;
        }
        let words_needed = n.div_ceil(64);
        assert!(
            scratch.len() >= words_needed,
            "Permutation::validate_no_alloc: scratch words ({}) < needed ({words_needed})",
            scratch.len(),
        );
        // Zero only the prefix we will use; leave any tail untouched
        // so callers can re-use a larger scratch buffer cheaply.
        for slot in scratch.iter_mut().take(words_needed) {
            *slot = 0;
        }
        for &id in &self.0 {
            let id = id as usize;
            if id >= n {
                return false;
            }
            let word = id >> 6;
            let bit = 1_u64 << (id & 63);
            // SAFETY: `id < n` and `word = id / 64 < n.div_ceil(64) =
            // words_needed <= scratch.len()`, so the index is in bounds.
            // We use checked indexing to keep the function panic-free
            // on the hot path (the outer assert already verified the
            // upper bound).
            let cell = &mut scratch[word];
            if *cell & bit != 0 {
                return false;
            }
            *cell |= bit;
        }
        true
    }

    /// Like [`Permutation::apply_into`] but verifies during apply that
    /// no destination slot is written twice. Uses caller-provided
    /// u64-bitset scratch instead of allocating.
    ///
    /// `scratch` must have at least `self.len().div_ceil(64)` u64
    /// words; the first `len().div_ceil(64)` words are zeroed on
    /// entry. On error the function returns early — `dst` may be
    /// partially written and must NOT be trusted by the caller.
    ///
    /// Kernel-mode consumers should prefer this over
    /// [`Permutation::apply_into`] when the source [`Permutation`]
    /// came from untrusted storage. It detects every failure mode of
    /// [`Permutation::from_vec_unchecked`] without allocating.
    ///
    /// # Errors
    ///
    /// Returns:
    ///
    ///   * [`PermutationApplyError::SrcLenMismatch`] when
    ///     `src.len() != self.len()`.
    ///   * [`PermutationApplyError::DstTooSmall`] when
    ///     `dst.len() < self.len()`.
    ///   * [`PermutationApplyError::ScratchTooSmall`] when `scratch`
    ///     has fewer than `self.len().div_ceil(64)` words.
    ///   * [`PermutationApplyError::OutOfRangeDst`] when an entry of
    ///     the permutation is `>= self.len()`.
    ///   * [`PermutationApplyError::DuplicateDst`] when two entries
    ///     of the permutation map to the same destination slot.
    pub fn try_apply_into_strict<T: Copy>(
        &self,
        src: &[T],
        dst: &mut [T],
        scratch: &mut [u64],
    ) -> Result<(), PermutationApplyError> {
        let n = self.0.len();
        if src.len() != n {
            return Err(PermutationApplyError::SrcLenMismatch {
                expected: n,
                actual: src.len(),
            });
        }
        if dst.len() < n {
            return Err(PermutationApplyError::DstTooSmall {
                needed: n,
                actual: dst.len(),
            });
        }
        let words_needed = n.div_ceil(64);
        if scratch.len() < words_needed {
            return Err(PermutationApplyError::ScratchTooSmall {
                needed_words: words_needed,
                actual_words: scratch.len(),
            });
        }
        for slot in scratch.iter_mut().take(words_needed) {
            *slot = 0;
        }
        for (i, &perm_i) in self.0.iter().enumerate() {
            let id = perm_i as usize;
            if id >= n {
                return Err(PermutationApplyError::OutOfRangeDst {
                    src_index: i,
                    dst_slot: perm_i,
                });
            }
            let word = id >> 6;
            let bit = 1_u64 << (id & 63);
            let cell = &mut scratch[word];
            if *cell & bit != 0 {
                return Err(PermutationApplyError::DuplicateDst {
                    src_index: i,
                    dst_slot: perm_i,
                });
            }
            *cell |= bit;
            dst[id] = src[i];
        }
        Ok(())
    }

    /// Returns the underlying permutation array.
    ///
    /// `perm[i]` is the new position of the item that was at position
    /// `i` in the input.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// Returns the length of the permutation (number of items).
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` when the permutation is empty (`len() == 0`).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Consumes the [`Permutation`] and returns the underlying `Vec<u32>`.
    ///
    /// Useful when serialising into an image manifest where the buffer
    /// can be moved without re-allocating.
    #[must_use]
    pub fn into_vec(self) -> Vec<u32> {
        self.0
    }
}

/// Failure modes returned by [`Permutation::try_apply_into_strict`].
///
/// All variants are non-allocating and `Copy` so they can be propagated
/// from kernel-mode call sites without touching the heap. They surface
/// the offending source index and destination slot when applicable so a
/// caller can log or telemetry-tag corrupted on-disk permutations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermutationApplyError {
    /// `src.len() != self.len()`. Source has the wrong number of
    /// elements for this permutation.
    SrcLenMismatch {
        /// Permutation length.
        expected: usize,
        /// Caller-supplied source length.
        actual: usize,
    },
    /// `dst.len() < self.len()`. Destination buffer cannot fit the
    /// full permuted output.
    DstTooSmall {
        /// Permutation length (minimum required `dst.len()`).
        needed: usize,
        /// Caller-supplied destination length.
        actual: usize,
    },
    /// `scratch.len() < self.len().div_ceil(64)`. Bitset scratch is
    /// too small to track destination occupancy.
    ScratchTooSmall {
        /// Required number of u64 words.
        needed_words: usize,
        /// Caller-supplied number of u64 words.
        actual_words: usize,
    },
    /// Duplicate destination index detected — perm is not a valid
    /// bijection. The destination slot was already written by an
    /// earlier source index.
    DuplicateDst {
        /// Source index whose mapping triggered the collision.
        src_index: usize,
        /// Destination slot that was already taken.
        dst_slot: u32,
    },
    /// Out-of-range destination index — perm contains an entry
    /// `>= self.len()`.
    OutOfRangeDst {
        /// Source index whose mapping is out of range.
        src_index: usize,
        /// The offending out-of-range destination slot.
        dst_slot: u32,
    },
}

impl core::fmt::Display for PermutationApplyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SrcLenMismatch { expected, actual } => write!(
                f,
                "Permutation::try_apply_into_strict: src.len() ({actual}) != perm.len() ({expected})"
            ),
            Self::DstTooSmall { needed, actual } => write!(
                f,
                "Permutation::try_apply_into_strict: dst.len() ({actual}) < perm.len() ({needed})"
            ),
            Self::ScratchTooSmall {
                needed_words,
                actual_words,
            } => write!(
                f,
                "Permutation::try_apply_into_strict: scratch words ({actual_words}) < needed ({needed_words})"
            ),
            Self::DuplicateDst {
                src_index,
                dst_slot,
            } => write!(
                f,
                "Permutation::try_apply_into_strict: duplicate dst slot {dst_slot} at src index {src_index}"
            ),
            Self::OutOfRangeDst {
                src_index,
                dst_slot,
            } => write!(
                f,
                "Permutation::try_apply_into_strict: out-of-range dst slot {dst_slot} at src index {src_index}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PermutationApplyError {}

/// A borrowed compressed sparse row (CSR) adjacency input for graph
/// permutations.
///
/// `offsets` has length `n + 1`; vertex `v`'s neighbour list lives in
/// `neighbors[offsets[v] as usize..offsets[v + 1] as usize]`.
///
/// The graph is treated as undirected: callers that want directed
/// semantics should symmetrise upstream. For [`rcm()`], duplicate edges
/// and self-loops are tolerated and contribute degree weight without
/// affecting reachability.
#[derive(Copy, Clone, Debug)]
pub struct CsrGraph<'a> {
    /// Number of vertices.
    pub n: u32,
    /// CSR offsets array of length `n + 1`. `offsets[i]` is the starting
    /// index in `neighbors` for vertex `i`'s neighbour list.
    pub offsets: &'a [u32],
    /// Concatenated neighbour lists for every vertex.
    pub neighbors: &'a [u32],
}

impl<'a> CsrGraph<'a> {
    /// Returns the neighbours of vertex `v`.
    ///
    /// # Panics
    ///
    /// Panics if `v >= self.n` or if the offsets/neighbors arrays are
    /// inconsistent (e.g. `offsets[v + 1] < offsets[v]`).
    #[must_use]
    pub fn neighbors_of(&self, v: u32) -> &'a [u32] {
        assert!(v < self.n, "vertex {v} out of range [0, {})", self.n);
        let start = self.offsets[v as usize] as usize;
        let end = self.offsets[v as usize + 1] as usize;
        &self.neighbors[start..end]
    }

    /// Returns the degree of vertex `v` (count of out-edges in CSR;
    /// includes any duplicates / self-loops).
    ///
    /// # Panics
    ///
    /// Panics under the same conditions as [`Self::neighbors_of`].
    #[must_use]
    pub fn degree(&self, v: u32) -> u32 {
        assert!(v < self.n, "vertex {v} out of range [0, {})", self.n);
        self.offsets[v as usize + 1] - self.offsets[v as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_round_trip() {
        for n in [0_usize, 1, 2, 5, 17, 256] {
            let perm = Permutation::identity(n);
            assert_eq!(perm.len(), n);
            assert_eq!(perm.is_empty(), n == 0);
            for (i, &v) in perm.as_slice().iter().enumerate() {
                assert_eq!(v as usize, i);
            }
            // Identity applied to any source returns the source unchanged.
            let src: Vec<u32> = (0..n as u32).collect();
            let dst = perm.apply(&src);
            assert_eq!(dst, src);
        }
    }

    #[test]
    fn inverse_round_trips_arbitrary_data() {
        // Hand-built permutation: 0->2, 1->0, 2->3, 3->1.
        let perm = Permutation::try_from_vec(vec![2, 0, 3, 1]).expect("valid perm");
        let src: Vec<i32> = vec![10, 20, 30, 40];
        let permuted = perm.apply(&src);
        // src[0]=10 lands at position 2; src[1]=20 at 0; src[2]=30 at 3; src[3]=40 at 1.
        assert_eq!(permuted, vec![20, 40, 10, 30]);
        // Inverse round-trips back to src.
        let inv = perm.inverse();
        let recovered = inv.apply(&permuted);
        assert_eq!(recovered, src);
    }

    #[test]
    fn try_from_vec_rejects_duplicates() {
        assert!(Permutation::try_from_vec(vec![0, 1, 1]).is_none());
        assert!(Permutation::try_from_vec(vec![0, 0]).is_none());
    }

    #[test]
    fn try_from_vec_rejects_out_of_range() {
        assert!(Permutation::try_from_vec(vec![0, 5]).is_none());
        assert!(Permutation::try_from_vec(vec![3]).is_none());
    }

    #[test]
    fn try_from_vec_accepts_identity_and_reverse() {
        assert!(Permutation::try_from_vec(vec![0, 1, 2, 3]).is_some());
        assert!(Permutation::try_from_vec(vec![3, 2, 1, 0]).is_some());
    }

    #[test]
    fn try_from_vec_rejects_length_mismatch() {
        // Index `n` (out of range for length `n`) — covers the "length
        // mismatch" branch where every entry is in-range for some larger
        // domain but exceeds `perm.len() - 1` for this particular vector.
        assert!(Permutation::try_from_vec(vec![1, 2, 3]).is_none());
        // Mixed: in-range and out-of-range together.
        assert!(Permutation::try_from_vec(vec![0, 1, 4]).is_none());
    }

    #[test]
    fn try_from_vec_accepts_empty() {
        // Empty Vec is the unique 0-length permutation (vacuous bijection).
        let perm = Permutation::try_from_vec(Vec::<u32>::new()).expect("empty perm valid");
        assert_eq!(perm.len(), 0);
        assert!(perm.is_empty());
    }

    #[test]
    fn from_vec_unchecked_round_trips_valid_input() {
        // SAFETY: `[2, 0, 3, 1]` is a permutation of `0..4`.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![2_u32, 0, 3, 1]) };
        assert_eq!(perm.as_slice(), &[2, 0, 3, 1]);
        // The same bytes pass `try_from_vec`, demonstrating the two
        // constructors agree on valid input.
        let checked = Permutation::try_from_vec(vec![2_u32, 0, 3, 1]).expect("valid");
        assert_eq!(perm, checked);
    }

    #[test]
    fn apply_into_writes_caller_buffer() {
        let perm = Permutation::try_from_vec(vec![1, 0, 2]).expect("valid perm");
        let src: Vec<u8> = vec![7, 8, 9];
        let mut dst = [0_u8; 3];
        perm.apply_into(&src, &mut dst);
        assert_eq!(dst, [8, 7, 9]);
    }

    #[test]
    fn csr_graph_degrees_match() {
        // Path: 0-1-2-3, undirected, so each interior node has degree 2.
        let offsets = [0_u32, 1, 3, 5, 6];
        let neighbors = [1_u32, 0, 2, 1, 3, 2];
        let g = CsrGraph {
            n: 4,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        assert_eq!(g.degree(0), 1);
        assert_eq!(g.degree(1), 2);
        assert_eq!(g.degree(2), 2);
        assert_eq!(g.degree(3), 1);
        assert_eq!(g.neighbors_of(2), &[1_u32, 3]);
    }

    #[test]
    fn into_vec_returns_underlying_storage() {
        let perm = Permutation::try_from_vec(vec![2, 0, 1]).expect("valid perm");
        assert_eq!(perm.into_vec(), vec![2, 0, 1]);
    }

    // ----- audit-R6 #161 — heap-free validation + strict apply -----

    #[test]
    fn validate_no_alloc_accepts_identity_and_arbitrary_valid_perms() {
        for n in [0_usize, 1, 2, 5, 17, 64, 65, 128, 129, 256] {
            let perm = Permutation::identity(n);
            let mut scratch = vec![0_u64; n.div_ceil(64).max(1)];
            assert!(
                perm.validate_no_alloc(&mut scratch),
                "identity of length {n} must validate"
            );
        }
        // Non-trivial valid permutation across the 64-bit word boundary.
        // Reverse permutation: 0 -> n-1, 1 -> n-2, ...
        for n in [1_usize, 2, 63, 64, 65, 128, 129] {
            let mut v: Vec<u32> = (0..n as u32).rev().collect();
            // Touch a couple of cross-word slots to be sure.
            if n >= 70 {
                v.swap(3, 70);
            }
            let perm = Permutation::try_from_vec(v).expect("valid");
            let mut scratch = vec![0_u64; n.div_ceil(64).max(1)];
            assert!(
                perm.validate_no_alloc(&mut scratch),
                "reverse perm of length {n} must validate"
            );
        }
    }

    #[test]
    fn validate_no_alloc_rejects_duplicates() {
        // SAFETY: deliberately invalid — used only to exercise the validator.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![0_u32, 1, 1]) };
        let mut scratch = [0_u64; 1];
        assert!(!perm.validate_no_alloc(&mut scratch));
    }

    #[test]
    fn validate_no_alloc_rejects_out_of_range() {
        // SAFETY: deliberately invalid — used only to exercise the validator.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![0_u32, 1, 5]) };
        let mut scratch = [0_u64; 1];
        assert!(!perm.validate_no_alloc(&mut scratch));
    }

    #[test]
    fn validate_no_alloc_rejects_dup_across_word_boundary() {
        // Length 70 — bit 65 lives in word 1. Duplicate the value 65 to
        // exercise a multi-word bitset write.
        let mut v: Vec<u32> = (0..70_u32).collect();
        v[3] = 65; // duplicates 65 (which is also at index 65)
        // SAFETY: deliberately invalid — used only to exercise the validator.
        let perm = unsafe { Permutation::from_vec_unchecked(v) };
        let mut scratch = [0_u64; 2];
        assert!(!perm.validate_no_alloc(&mut scratch));
    }

    #[test]
    #[should_panic(expected = "scratch words")]
    fn validate_no_alloc_panics_on_undersized_scratch() {
        let perm = Permutation::identity(65);
        let mut scratch = [0_u64; 1]; // need 2 words for n=65
        let _ = perm.validate_no_alloc(&mut scratch);
    }

    #[test]
    fn try_apply_into_strict_matches_apply_into_on_valid_perm() {
        // Mid-size permutation that crosses the 64-bit word boundary.
        let mut v: Vec<u32> = (0..70_u32).collect();
        v.swap(3, 65);
        v.swap(7, 12);
        let perm = Permutation::try_from_vec(v).expect("valid");
        let src: Vec<u32> = (1000..1070_u32).collect();

        let mut dst_strict = vec![0_u32; 70];
        let mut scratch = [0_u64; 2];
        perm.try_apply_into_strict(&src, &mut dst_strict, &mut scratch)
            .expect("valid perm must succeed");

        let mut dst_unchecked = vec![0_u32; 70];
        perm.apply_into(&src, &mut dst_unchecked);

        assert_eq!(dst_strict, dst_unchecked);
    }

    #[test]
    fn try_apply_into_strict_detects_duplicate_dst() {
        // SAFETY: deliberately invalid — perm[0]=1 and perm[2]=1 both
        // target slot 1.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![1_u32, 0, 1]) };
        let src = [10_u8, 20, 30];
        let mut dst = [0_u8; 3];
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_apply_into_strict(&src, &mut dst, &mut scratch)
            .expect_err("duplicate must error");
        assert_eq!(
            err,
            PermutationApplyError::DuplicateDst {
                src_index: 2,
                dst_slot: 1,
            }
        );
    }

    #[test]
    fn try_apply_into_strict_detects_out_of_range_dst() {
        // SAFETY: deliberately invalid — perm[1]=5 is out of range for n=3.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![0_u32, 5, 2]) };
        let src = [10_u8, 20, 30];
        let mut dst = [0_u8; 3];
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_apply_into_strict(&src, &mut dst, &mut scratch)
            .expect_err("out-of-range must error");
        assert_eq!(
            err,
            PermutationApplyError::OutOfRangeDst {
                src_index: 1,
                dst_slot: 5,
            }
        );
    }

    #[test]
    fn try_apply_into_strict_rejects_src_len_mismatch() {
        let perm = Permutation::identity(3);
        let src = [10_u8, 20];
        let mut dst = [0_u8; 3];
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_apply_into_strict(&src, &mut dst, &mut scratch)
            .expect_err("src length mismatch must error");
        assert_eq!(
            err,
            PermutationApplyError::SrcLenMismatch {
                expected: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn try_apply_into_strict_rejects_dst_too_small() {
        let perm = Permutation::identity(3);
        let src = [10_u8, 20, 30];
        let mut dst = [0_u8; 2];
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_apply_into_strict(&src, &mut dst, &mut scratch)
            .expect_err("dst too small must error");
        assert_eq!(
            err,
            PermutationApplyError::DstTooSmall {
                needed: 3,
                actual: 2,
            }
        );
    }

    #[test]
    fn try_apply_into_strict_rejects_scratch_too_small() {
        // n=65 needs 2 u64 words; pass 1.
        let perm = Permutation::identity(65);
        let src: Vec<u32> = (0..65_u32).collect();
        let mut dst = vec![0_u32; 65];
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_apply_into_strict(&src, &mut dst, &mut scratch)
            .expect_err("scratch too small must error");
        assert_eq!(
            err,
            PermutationApplyError::ScratchTooSmall {
                needed_words: 2,
                actual_words: 1,
            }
        );
    }

    #[test]
    fn apply_into_leaks_dst_sentinel_on_invalid_perm() {
        // audit-R6 #161 regression demonstration: when an invalid
        // permutation built via from_vec_unchecked has duplicate dst
        // indices, `apply_into` writes one slot twice and leaves
        // another slot at its prior value. A kernel that trusts the
        // result is exposed to whatever was previously in `dst`.
        //
        // Setup: perm = [1, 1, 2] is NOT a bijection. Slot 1 is
        // written twice (by src[0]=10 then src[1]=20); slot 0 is
        // never written and retains the sentinel.
        // SAFETY: deliberately invalid — pinning the failure mode for
        // the regression test.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![1_u32, 1, 2]) };
        let src = [10_u8, 20, 30];
        let mut dst = [0xAA_u8; 3];
        perm.apply_into(&src, &mut dst);
        // Failure mode: slot 0 retains the sentinel because it was
        // never targeted by any src index. This is the leak the strict
        // variant exists to defend against.
        assert_eq!(
            dst[0], 0xAA,
            "dst[0] should remain at the prior sentinel value (the leak)"
        );
        assert_eq!(dst[1], 20); // perm[1] = 1, so src[1]=20 overwrites src[0].
        assert_eq!(dst[2], 30);

        // The strict variant catches this immediately.
        let mut dst_strict = [0xAA_u8; 3];
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_apply_into_strict(&src, &mut dst_strict, &mut scratch)
            .expect_err("strict variant must reject the corrupted perm");
        assert_eq!(
            err,
            PermutationApplyError::DuplicateDst {
                src_index: 1,
                dst_slot: 1,
            }
        );
    }

    #[test]
    fn permutation_apply_error_display_renders_all_variants() {
        // Smoke test the Display impl so the rustdoc covers all arms.
        let cases = [
            PermutationApplyError::SrcLenMismatch {
                expected: 3,
                actual: 2,
            },
            PermutationApplyError::DstTooSmall {
                needed: 3,
                actual: 2,
            },
            PermutationApplyError::ScratchTooSmall {
                needed_words: 2,
                actual_words: 1,
            },
            PermutationApplyError::DuplicateDst {
                src_index: 2,
                dst_slot: 1,
            },
            PermutationApplyError::OutOfRangeDst {
                src_index: 1,
                dst_slot: 9,
            },
        ];
        for c in cases {
            let s = format!("{c}");
            assert!(s.starts_with("Permutation::try_apply_into_strict"));
        }
    }
}

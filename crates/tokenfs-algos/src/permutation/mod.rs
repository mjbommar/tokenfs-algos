//! Locality-improving orderings for graph- and point-shaped data.
//!
//! See `docs/v0.2_planning/14_PERMUTATION.md` for the spec. This module
//! ships the shared [`Permutation`] type plus the CSR adjacency input
//! type [`CsrGraph`], and the **build-time** ordering primitives that
//! produce a [`Permutation`] from such inputs.
//!
//! ## Available orderings
//!
//! Four orderings are exposed; pick by input shape and quality / cost
//! trade-off. See `docs/PHASE_D_RABBIT_ORDER.md` for the side-by-side
//! comparison and `docs/v0.2_planning/14_PERMUTATION.md` for the full
//! spec.
//!
//! | API | Input shape | Build cost | Quality | Use when |
//! |---|---|---|---|---|
//! | [`Permutation::identity`] | none (length only) | trivial | none | a permutation slot is required but no reordering is desired (placeholder, baseline, or fixture) |
//! | [`rcm()`] | sparse undirected graph ([`CsrGraph`]) | very cheap (`O(\|V\| + \|E\| log Δ)`, ~10 ms / 228 K vertices) | bandwidth-minimising; modest cache-locality win | sparse-matrix solvers, bandwidth-driven workloads, or any "good enough" graph reorder where build time matters |
//! | [`hilbert_2d`] / [`hilbert_nd`] | point cloud in 2D or N-D (`&[(f32,f32)]` / `&[Vec<f32>]`) | `O(n log n)` sort | preserves metric locality on the embedding | data has a true low-dimensional point embedding (PCA-projected fingerprints, t-SNE/UMAP outputs); spatial-locality scans |
//! | [`rabbit_order()`] / [`rabbit_order_par()`] | sparse undirected graph ([`CsrGraph`]) | heavy (`O(\|E\| log \|V\|)`, 1-5 s / 228 K vertices) | best published cache-locality; community-aware | community-structured graphs feeding locality-sensitive workloads (BFS, PageRank, neighbour scans, TokenFS dedup-cluster reads) |
//!
//! [`hilbert_2d`] / [`hilbert_nd`] are gated on the `permutation_hilbert`
//! Cargo feature; [`rabbit_order_par`] is gated on the `parallel` Cargo
//! feature.
//!
//! Identity vs the others: pick [`Permutation::identity`] only when the
//! permutation is a structural placeholder (the type's invariant is a
//! valid bijection on `0..n`, so callers cannot pass a raw `Vec<u32>`).
//! For real data, prefer one of the three reordering primitives.
//!
//! ## Sprint 11-13 / Sprint 47-49 / Sprint 53-55 status
//!
//! Phase B4 of `01_PHASES.md` lands [`rcm()`] (Reverse Cuthill-McKee).
//! Phase B5 lands `hilbert_2d` / `hilbert_nd` behind the
//! `permutation_hilbert` Cargo feature (vendor wrappers around the
//! `fast_hilbert` and `hilbert` crates per spec § 4). Sprint 47-49 of
//! Phase D1 lands [`rabbit_order()`] as a single-pass sequential
//! baseline; Sprint 50-52 lifts the modularity-gain inner loop into a
//! SIMD kernel module (see [`rabbit::kernels`]); Sprint 53-55 ships
//! [`rabbit::rabbit_order_par`] (gated on the `parallel` Cargo
//! feature), a round-based concurrent variant that parallelises the
//! per-vertex merge-proposal phase across rayon while keeping the
//! merge-application phase deterministic.
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
pub use rabbit::{rabbit_order, try_rabbit_order};
#[cfg(feature = "parallel")]
pub use rabbit::{rabbit_order_par, try_rabbit_order_par};
pub use rcm::{rcm, try_rcm};

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
    /// Kernel/FUSE callers that need a non-panicking shape check should
    /// prefer [`Permutation::try_apply`]. Callers that additionally want
    /// proof of permutation validity should prefer
    /// [`Permutation::try_apply_into_strict`].
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
    /// Kernel/FUSE callers that need a non-panicking shape check should
    /// prefer [`Permutation::try_apply_into`]. Callers that additionally
    /// want proof of permutation validity should prefer
    /// [`Permutation::try_apply_into_strict`].
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

    /// Like [`Permutation::apply`] but returns an error on
    /// shape-mismatch instead of panicking.
    ///
    /// Validates `src.len() == self.len()` upfront; on mismatch returns
    /// [`PermutationApplyError::SrcLenMismatch`] without inspecting
    /// `src` further. On success the body is identical to
    /// [`Permutation::apply`] — `dst[perm[i]] = src[i]` is written into
    /// a freshly allocated `Vec<T>` of length `self.len()`.
    ///
    /// This variant only checks the shape contract; it does NOT verify
    /// that the [`Permutation`] is a valid bijection. If `self` was
    /// constructed via [`Permutation::from_vec_unchecked`] from
    /// untrusted on-disk data, prefer
    /// [`Permutation::try_apply_into_strict`], which additionally
    /// detects out-of-range and duplicate destination indices via a
    /// caller-provided scratch bitset.
    ///
    /// For zero-length permutations, the empty `Vec` is returned
    /// without inspecting `src` (provided the length matches).
    ///
    /// # Errors
    ///
    /// * [`PermutationApplyError::SrcLenMismatch`] when
    ///   `src.len() != self.len()`.
    pub fn try_apply<T: Copy>(&self, src: &[T]) -> Result<Vec<T>, PermutationApplyError> {
        let n = self.0.len();
        if src.len() != n {
            return Err(PermutationApplyError::SrcLenMismatch {
                expected: n,
                actual: src.len(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        // Initialise dst with a clone of `src[0]` to satisfy the `Vec`
        // invariant; every slot is overwritten by `apply_into`. The
        // bound `T: Copy` makes the clone a no-op.
        let mut dst = vec![src[0]; n];
        // SAFETY-LIKE: `apply_into`'s shape contract is satisfied
        // because we just verified `src.len() == n` and built `dst` of
        // length `n`. The unchecked-panic-on-shape path inside
        // `apply_into` cannot trigger.
        self.apply_into(src, &mut dst);
        Ok(dst)
    }

    /// Like [`Permutation::apply_into`] but returns an error on
    /// shape-mismatch instead of panicking.
    ///
    /// Validates `src.len() == self.len()` and `dst.len() >= self.len()`
    /// upfront; on either mismatch returns the corresponding
    /// [`PermutationApplyError`] variant without writing to `dst`.
    ///
    /// This variant only checks the shape contract; it does NOT verify
    /// that the [`Permutation`] is a valid bijection. If `self` was
    /// constructed via [`Permutation::from_vec_unchecked`] from
    /// untrusted on-disk data, prefer
    /// [`Permutation::try_apply_into_strict`], which additionally
    /// detects out-of-range and duplicate destination indices via a
    /// caller-provided scratch bitset.
    ///
    /// On success the body is identical to
    /// [`Permutation::apply_into`]: `dst[perm[i]] = src[i]` for every
    /// `i in 0..self.len()`.
    ///
    /// # Errors
    ///
    /// * [`PermutationApplyError::SrcLenMismatch`] when
    ///   `src.len() != self.len()`.
    /// * [`PermutationApplyError::DstTooSmall`] when
    ///   `dst.len() < self.len()`.
    pub fn try_apply_into<T: Copy>(
        &self,
        src: &[T],
        dst: &mut [T],
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
        for (i, &new_id) in self.0.iter().enumerate() {
            dst[new_id as usize] = src[i];
        }
        Ok(())
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

    /// Like [`Permutation::validate_no_alloc`] but returns a structured
    /// error on failure instead of panicking on undersized scratch.
    ///
    /// `scratch` must have at least `self.len().div_ceil(64)` u64
    /// words. The first `len().div_ceil(64)` words of `scratch` are
    /// zeroed on entry. On a successful traversal, returns `Ok(())` —
    /// the [`Permutation`] is a valid bijection on `0..self.len()`.
    ///
    /// This is the kernel-safe sibling of
    /// [`Permutation::validate_no_alloc`]: it covers exactly the same
    /// failure modes (length overflow, undersized scratch, out-of-range
    /// index, duplicate index) but reports each via a non-panicking
    /// [`PermutationValidationError`] variant. Use it when validating a
    /// [`Permutation`] freshly loaded from untrusted on-disk data
    /// without committing to the apply step `try_apply_into_strict`
    /// performs.
    ///
    /// # Errors
    ///
    /// * [`PermutationValidationError::LengthOverflow`] when
    ///   `self.len() > u32::MAX as usize` (vertex IDs are u32 by
    ///   construction; should never trigger via the public
    ///   constructors but defends against `from_vec_unchecked` abuse).
    /// * [`PermutationValidationError::ScratchTooSmall`] when
    ///   `scratch.len() < self.len().div_ceil(64)`.
    /// * [`PermutationValidationError::OutOfRangeIndex`] when an entry
    ///   of the permutation is `>= self.len()`.
    /// * [`PermutationValidationError::DuplicateIndex`] when two
    ///   entries of the permutation map to the same destination slot.
    pub fn try_validate_no_alloc(
        &self,
        scratch: &mut [u64],
    ) -> Result<(), PermutationValidationError> {
        let n = self.0.len();
        if n > u32::MAX as usize {
            return Err(PermutationValidationError::LengthOverflow { len: n });
        }
        let words_needed = n.div_ceil(64);
        if scratch.len() < words_needed {
            return Err(PermutationValidationError::ScratchTooSmall {
                needed_words: words_needed,
                actual_words: scratch.len(),
            });
        }
        // Zero only the prefix we will use; leave any tail untouched
        // so callers can re-use a larger scratch buffer cheaply.
        for slot in scratch.iter_mut().take(words_needed) {
            *slot = 0;
        }
        for (i, &id_u32) in self.0.iter().enumerate() {
            let id = id_u32 as usize;
            if id >= n {
                return Err(PermutationValidationError::OutOfRangeIndex {
                    src_index: i,
                    value: id_u32,
                });
            }
            let word = id >> 6;
            let bit = 1_u64 << (id & 63);
            let cell = &mut scratch[word];
            if *cell & bit != 0 {
                return Err(PermutationValidationError::DuplicateIndex {
                    src_index: i,
                    value: id_u32,
                });
            }
            *cell |= bit;
        }
        Ok(())
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

/// Failure modes returned by [`Permutation::try_validate_no_alloc`].
///
/// All variants are non-allocating and `Copy` so they can be propagated
/// from kernel-mode call sites without touching the heap. They surface
/// the offending source index and value when applicable so a caller can
/// log or telemetry-tag corrupted on-disk permutations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermutationValidationError {
    /// Permutation length exceeds `u32::MAX as usize`. Vertex IDs are
    /// `u32` by construction so this should be unreachable through
    /// public constructors; the variant exists to defend against
    /// misuse of [`Permutation::from_vec_unchecked`].
    LengthOverflow {
        /// The over-large permutation length.
        len: usize,
    },
    /// `scratch.len() < self.len().div_ceil(64)`. Bitset scratch is
    /// too small to track destination occupancy.
    ScratchTooSmall {
        /// Required number of u64 words.
        needed_words: usize,
        /// Caller-supplied number of u64 words.
        actual_words: usize,
    },
    /// Out-of-range index — perm contains an entry `>= self.len()`.
    OutOfRangeIndex {
        /// Source index whose value is out of range.
        src_index: usize,
        /// The offending out-of-range value.
        value: u32,
    },
    /// Duplicate index detected — perm is not a valid bijection. The
    /// destination slot was already claimed by an earlier source index.
    DuplicateIndex {
        /// Source index whose mapping triggered the collision.
        src_index: usize,
        /// The duplicated value.
        value: u32,
    },
}

impl core::fmt::Display for PermutationValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthOverflow { len } => write!(
                f,
                "Permutation::try_validate_no_alloc: permutation length ({len}) exceeds u32::MAX"
            ),
            Self::ScratchTooSmall {
                needed_words,
                actual_words,
            } => write!(
                f,
                "Permutation::try_validate_no_alloc: scratch words ({actual_words}) < needed ({needed_words})"
            ),
            Self::OutOfRangeIndex { src_index, value } => write!(
                f,
                "Permutation::try_validate_no_alloc: out-of-range value {value} at src index {src_index}"
            ),
            Self::DuplicateIndex { src_index, value } => write!(
                f,
                "Permutation::try_validate_no_alloc: duplicate value {value} at src index {src_index}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PermutationValidationError {}

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
    /// inconsistent (e.g. `offsets[v + 1] < offsets[v]`,
    /// `offsets.len() < v + 2`, or `offsets[v + 1] > neighbors.len()`).
    /// Kernel/FUSE callers that need a non-panicking variant should
    /// prefer [`CsrGraph::try_neighbors_of`].
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
    /// Kernel/FUSE callers that need a non-panicking variant should
    /// prefer [`CsrGraph::try_degree_of`].
    #[must_use]
    pub fn degree(&self, v: u32) -> u32 {
        assert!(v < self.n, "vertex {v} out of range [0, {})", self.n);
        self.offsets[v as usize + 1] - self.offsets[v as usize]
    }

    /// Like [`CsrGraph::neighbors_of`] but returns an error on
    /// invalid input instead of panicking.
    ///
    /// Validates that `v < self.n`, that the offsets array is long
    /// enough to read both `offsets[v]` and `offsets[v + 1]`, that the
    /// offset pair is monotone (`offsets[v] <= offsets[v + 1]`), and
    /// that the slice bounds fit inside `self.neighbors`. On success
    /// returns the borrowed neighbour slice; on failure returns the
    /// matching [`CsrGraphError`] variant without touching memory
    /// outside the bounds the validator already checked.
    ///
    /// Use this from kernel-mode consumers that loaded a [`CsrGraph`]
    /// from an untrusted on-disk image and need to traverse
    /// adjacency lists without exposing a DoS hazard at the kernel
    /// boundary.
    ///
    /// # Errors
    ///
    /// * [`CsrGraphError::OutOfRange`] when `v >= self.n` or when
    ///   `self.offsets.len() < (v as usize) + 2`.
    /// * [`CsrGraphError::OffsetsNonMonotone`] when
    ///   `self.offsets[v] > self.offsets[v + 1]`.
    /// * [`CsrGraphError::NeighborsOutOfBounds`] when
    ///   `self.offsets[v + 1] as usize > self.neighbors.len()`.
    pub fn try_neighbors_of(&self, v: u32) -> Result<&'a [u32], CsrGraphError> {
        if v >= self.n {
            return Err(CsrGraphError::OutOfRange { v, n: self.n });
        }
        let v_idx = v as usize;
        // The offsets array must be long enough to address `v + 1`.
        // Treat that as another "out of range" failure mode — the
        // caller's CsrGraph header is internally inconsistent.
        if self.offsets.len() <= v_idx + 1 {
            return Err(CsrGraphError::OutOfRange { v, n: self.n });
        }
        let lo = self.offsets[v_idx];
        let hi = self.offsets[v_idx + 1];
        if lo > hi {
            return Err(CsrGraphError::OffsetsNonMonotone { i: v, lo, hi });
        }
        let hi_idx = hi as usize;
        if hi_idx > self.neighbors.len() {
            return Err(CsrGraphError::NeighborsOutOfBounds {
                offset: hi,
                neighbors_len: self.neighbors.len(),
            });
        }
        let lo_idx = lo as usize;
        // SAFETY-LIKE: lo <= hi (monotone check) and hi_idx <= len
        // (bounds check) together imply lo_idx <= hi_idx <= len, so
        // the slice index is in bounds.
        Ok(&self.neighbors[lo_idx..hi_idx])
    }

    /// Like [`CsrGraph::degree`] but returns an error on invalid
    /// input instead of panicking.
    ///
    /// Validates that `v < self.n`, that the offsets array is long
    /// enough to read both `offsets[v]` and `offsets[v + 1]`, and
    /// that the offset pair is monotone. On success returns
    /// `offsets[v + 1] - offsets[v]`. The neighbours array length is
    /// not consulted — degree is purely a property of the offsets
    /// array — so [`CsrGraphError::NeighborsOutOfBounds`] is never
    /// returned.
    ///
    /// # Errors
    ///
    /// * [`CsrGraphError::OutOfRange`] when `v >= self.n` or when
    ///   `self.offsets.len() < (v as usize) + 2`.
    /// * [`CsrGraphError::OffsetsNonMonotone`] when
    ///   `self.offsets[v] > self.offsets[v + 1]`.
    pub fn try_degree_of(&self, v: u32) -> Result<u32, CsrGraphError> {
        if v >= self.n {
            return Err(CsrGraphError::OutOfRange { v, n: self.n });
        }
        let v_idx = v as usize;
        if self.offsets.len() <= v_idx + 1 {
            return Err(CsrGraphError::OutOfRange { v, n: self.n });
        }
        let lo = self.offsets[v_idx];
        let hi = self.offsets[v_idx + 1];
        if lo > hi {
            return Err(CsrGraphError::OffsetsNonMonotone { i: v, lo, hi });
        }
        Ok(hi - lo)
    }

    /// Validates the entire CSR header in O(|V| + |E|) time, returning
    /// `Ok(())` only when every per-vertex offset pair is monotone and
    /// in-bounds against the neighbours array, and every neighbour ID
    /// is in range `0..self.n`.
    ///
    /// This is the upfront-validation routine the fallible permutation
    /// builders ([`try_rcm`], [`rabbit::try_rabbit_order`],
    /// [`rabbit::try_rabbit_order_par`] when the `parallel` feature is
    /// enabled) call before they touch any of the existing internal
    /// pipelines. It is cheap relative to the agglomeration / BFS
    /// passes themselves and runs entirely on borrowed slices —
    /// suitable for kernel/FUSE callers that need to gate untrusted
    /// on-disk CSR images before invoking the heavyweight builder.
    ///
    /// Note: the per-vertex `try_neighbors_of` / `try_degree_of`
    /// helpers do NOT call this. They validate only the offsets pair
    /// they read; this routine adds the global `offsets.len() == n + 1`
    /// and `offsets[n] == neighbors.len()` checks plus a full sweep
    /// over `neighbors` for in-range vertex IDs.
    ///
    /// # Errors
    ///
    /// Returns the first failure mode encountered:
    ///
    /// * [`CsrGraphError::OffsetsLengthMismatch`] when
    ///   `self.offsets.len() != self.n as usize + 1`.
    /// * [`CsrGraphError::OffsetsNonMonotone`] when any consecutive
    ///   pair of offsets is inverted.
    /// * [`CsrGraphError::NeighborsLengthMismatch`] when
    ///   `self.offsets[n] as usize != self.neighbors.len()`.
    /// * [`CsrGraphError::NeighborOutOfRange`] when any entry of
    ///   `self.neighbors` is `>= self.n`.
    pub fn try_validate(&self) -> Result<(), CsrGraphError> {
        let n_usize = self.n as usize;
        if self.offsets.len() != n_usize + 1 {
            return Err(CsrGraphError::OffsetsLengthMismatch {
                actual_len: self.offsets.len(),
                expected_len: n_usize + 1,
            });
        }
        // n == 0: offsets is `[0]`, neighbors is empty. Validate that.
        if self.n == 0 {
            if self.neighbors.is_empty() {
                return Ok(());
            }
            return Err(CsrGraphError::NeighborsLengthMismatch {
                offsets_tail: self.offsets[0],
                neighbors_len: self.neighbors.len(),
            });
        }
        // Monotone offsets check.
        for i in 0..n_usize {
            let lo = self.offsets[i];
            let hi = self.offsets[i + 1];
            if lo > hi {
                // SAFETY-LIKE: i < n <= u32::MAX as usize so the cast
                // cannot overflow.
                #[allow(clippy::cast_possible_truncation)]
                return Err(CsrGraphError::OffsetsNonMonotone {
                    i: i as u32,
                    lo,
                    hi,
                });
            }
        }
        // Tail consistency: offsets[n] must equal neighbors.len().
        let tail = self.offsets[n_usize];
        if tail as usize != self.neighbors.len() {
            return Err(CsrGraphError::NeighborsLengthMismatch {
                offsets_tail: tail,
                neighbors_len: self.neighbors.len(),
            });
        }
        // In-range neighbour IDs.
        for (k, &u) in self.neighbors.iter().enumerate() {
            if u >= self.n {
                return Err(CsrGraphError::NeighborOutOfRange {
                    neighbor: u,
                    n: self.n,
                    at_index: k,
                });
            }
        }
        Ok(())
    }
}

/// Failure modes returned by the fallible [`CsrGraph`] accessors
/// ([`CsrGraph::try_neighbors_of`], [`CsrGraph::try_degree_of`]) and by
/// the fallible permutation builders' upfront CSR validation
/// (`try_rcm`, `try_rabbit_order`, `try_rabbit_order_par` — added in
/// audit-R7-followup #11).
///
/// All variants are non-allocating and `Copy` so they propagate from
/// kernel-mode call sites without touching the heap. They surface
/// the offending vertex / offset values so a caller can log or
/// telemetry-tag corrupted on-disk CSR images.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CsrGraphError {
    /// Vertex `v` is out of range for the graph: `v >= n`, or the
    /// `offsets` array does not cover index `v + 1` (header
    /// inconsistency).
    OutOfRange {
        /// The offending vertex ID.
        v: u32,
        /// The graph's declared vertex count `self.n`.
        n: u32,
    },
    /// `offsets[i] > offsets[i + 1]` — the CSR offsets array is not
    /// monotonically non-decreasing.
    OffsetsNonMonotone {
        /// Index `i` whose offset pair is inverted.
        i: u32,
        /// `offsets[i]`.
        lo: u32,
        /// `offsets[i + 1]`.
        hi: u32,
    },
    /// `offsets[v + 1] as usize > neighbors.len()` — the offset
    /// claims to address past the end of the neighbours array.
    NeighborsOutOfBounds {
        /// The offending offset value.
        offset: u32,
        /// The actual length of `neighbors`.
        neighbors_len: usize,
    },
    /// `offsets.len() != (n as usize) + 1` — the offsets array does
    /// not have the required CSR length.
    OffsetsLengthMismatch {
        /// The actual `offsets.len()`.
        actual_len: usize,
        /// The required length `n + 1`.
        expected_len: usize,
    },
    /// `offsets[n] as usize != neighbors.len()` — the offsets tail
    /// does not match the neighbour array length.
    NeighborsLengthMismatch {
        /// The value of `offsets[n]`.
        offsets_tail: u32,
        /// The actual length of `neighbors`.
        neighbors_len: usize,
    },
    /// A neighbour ID is out of range: `neighbors[k] >= n` for some
    /// `k`. Reported by the permutation builders' upfront CSR
    /// validation; not produced by the per-vertex try_* accessors
    /// (which never read `neighbors[k]`'s value).
    NeighborOutOfRange {
        /// The offending neighbour ID.
        neighbor: u32,
        /// The graph's declared vertex count `self.n`.
        n: u32,
        /// Index in the `neighbors` array where the offending value
        /// lives, for diagnostics.
        at_index: usize,
    },
}

impl core::fmt::Display for CsrGraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfRange { v, n } => {
                write!(f, "CsrGraph: vertex {v} out of range [0, {n})")
            }
            Self::OffsetsNonMonotone { i, lo, hi } => write!(
                f,
                "CsrGraph: offsets non-monotone at index {i}: offsets[{i}]={lo} > offsets[{}]={hi}",
                i + 1
            ),
            Self::NeighborsOutOfBounds {
                offset,
                neighbors_len,
            } => write!(
                f,
                "CsrGraph: offset {offset} addresses past neighbors.len() ({neighbors_len})"
            ),
            Self::OffsetsLengthMismatch {
                actual_len,
                expected_len,
            } => write!(
                f,
                "CsrGraph: offsets.len() ({actual_len}) != n + 1 ({expected_len})"
            ),
            Self::NeighborsLengthMismatch {
                offsets_tail,
                neighbors_len,
            } => write!(
                f,
                "CsrGraph: offsets[n] ({offsets_tail}) != neighbors.len() ({neighbors_len})"
            ),
            Self::NeighborOutOfRange {
                neighbor,
                n,
                at_index,
            } => write!(
                f,
                "CsrGraph: neighbor {neighbor} at neighbors[{at_index}] out of range [0, {n})"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CsrGraphError {}

/// Failure modes returned by the fallible permutation builders
/// ([`try_rcm`], [`rabbit::try_rabbit_order`], and
/// [`rabbit::try_rabbit_order_par`] when the `parallel` Cargo feature
/// is enabled).
///
/// The builders validate the input CSR upfront via
/// [`CsrGraph::try_validate`]; any header / offset / neighbour-bounds
/// failure is reported via the [`Self::InvalidCsr`] variant. The
/// existing internal builder pipelines panic only on conditions the
/// upfront validation already covers, so the try_* path produces
/// every failure mode through this top-level enum without unwinding
/// the kernel boundary.
///
/// All variants are non-allocating and `Copy` so they propagate from
/// kernel-mode call sites without touching the heap.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PermutationConstructionError {
    /// The input [`CsrGraph`] is internally inconsistent. See the
    /// nested [`CsrGraphError`] for the precise failure mode.
    InvalidCsr(CsrGraphError),
}

impl From<CsrGraphError> for PermutationConstructionError {
    fn from(err: CsrGraphError) -> Self {
        Self::InvalidCsr(err)
    }
}

impl core::fmt::Display for PermutationConstructionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidCsr(err) => {
                write!(f, "permutation construction failed: invalid CSR: {err}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PermutationConstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidCsr(err) => Some(err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The `format!` macro is not in the no-std prelude; alias it from
    // `alloc` for the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::format;

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

    // ----- audit-R7-followup #7 — shape-safe try_apply / try_apply_into -----

    #[test]
    fn try_apply_matches_apply_on_valid_perm() {
        // Happy path: identical output to the panicking sibling on a
        // shape-correct call.
        let perm = Permutation::try_from_vec(vec![2, 0, 3, 1]).expect("valid");
        let src: Vec<i32> = vec![10, 20, 30, 40];
        let expected = perm.apply(&src);
        let actual = perm.try_apply(&src).expect("shape matches");
        assert_eq!(actual, expected);
    }

    #[test]
    fn try_apply_returns_src_len_mismatch_without_panicking() {
        let perm = Permutation::identity(4);
        let src: Vec<i32> = vec![10, 20, 30]; // wrong length
        let err = perm.try_apply(&src).expect_err("length mismatch");
        assert_eq!(
            err,
            PermutationApplyError::SrcLenMismatch {
                expected: 4,
                actual: 3,
            }
        );
    }

    #[test]
    fn try_apply_handles_empty_perm_without_inspecting_src() {
        // Boundary: zero-length perm with zero-length src returns
        // empty Vec without touching `src`.
        let perm = Permutation::identity(0);
        let src: Vec<u8> = Vec::new();
        let out = perm.try_apply(&src).expect("empty matches empty");
        assert!(out.is_empty());
    }

    #[test]
    fn try_apply_into_matches_apply_into_on_valid_perm() {
        // Happy path: identical output to the panicking sibling.
        let perm = Permutation::try_from_vec(vec![1, 0, 2]).expect("valid perm");
        let src: Vec<u8> = vec![7, 8, 9];
        let mut dst_panic = [0_u8; 3];
        let mut dst_try = [0_u8; 3];
        perm.apply_into(&src, &mut dst_panic);
        perm.try_apply_into(&src, &mut dst_try)
            .expect("shape matches");
        assert_eq!(dst_try, dst_panic);
    }

    #[test]
    fn try_apply_into_rejects_src_len_mismatch() {
        let perm = Permutation::identity(3);
        let src = [10_u8, 20]; // wrong length
        let mut dst = [0_u8; 3];
        let err = perm
            .try_apply_into(&src, &mut dst)
            .expect_err("src length mismatch must error");
        assert_eq!(
            err,
            PermutationApplyError::SrcLenMismatch {
                expected: 3,
                actual: 2,
            }
        );
        // dst must not have been touched.
        assert_eq!(dst, [0_u8; 3]);
    }

    #[test]
    fn try_apply_into_rejects_dst_too_small() {
        let perm = Permutation::identity(3);
        let src = [10_u8, 20, 30];
        let mut dst = [0_u8; 2]; // too small
        let err = perm
            .try_apply_into(&src, &mut dst)
            .expect_err("dst too small must error");
        assert_eq!(
            err,
            PermutationApplyError::DstTooSmall {
                needed: 3,
                actual: 2,
            }
        );
        // dst must not have been touched.
        assert_eq!(dst, [0_u8; 2]);
    }

    // ----- audit-R7-followup #9 — try_validate_no_alloc -----

    #[test]
    fn try_validate_no_alloc_accepts_valid_perms() {
        // Happy path matches `validate_no_alloc(...)` -> true on every
        // valid permutation we previously verified that way.
        for n in [0_usize, 1, 2, 5, 17, 64, 65, 128, 129, 256] {
            let perm = Permutation::identity(n);
            let mut scratch = vec![0_u64; n.div_ceil(64).max(1)];
            perm.try_validate_no_alloc(&mut scratch)
                .expect("identity must validate");
        }
        // Cross-word valid permutation.
        let mut v: Vec<u32> = (0..70_u32).rev().collect();
        v.swap(3, 65);
        let perm = Permutation::try_from_vec(v).expect("valid");
        let mut scratch = [0_u64; 2];
        perm.try_validate_no_alloc(&mut scratch)
            .expect("reverse-with-swap must validate");
    }

    #[test]
    fn try_validate_no_alloc_rejects_undersized_scratch() {
        // n=65 needs 2 u64 words; pass 1.
        let perm = Permutation::identity(65);
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_validate_no_alloc(&mut scratch)
            .expect_err("undersized scratch must error, not panic");
        assert_eq!(
            err,
            PermutationValidationError::ScratchTooSmall {
                needed_words: 2,
                actual_words: 1,
            }
        );
    }

    #[test]
    fn try_validate_no_alloc_rejects_duplicates() {
        // SAFETY: deliberately invalid — used only to exercise the validator.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![0_u32, 1, 1]) };
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_validate_no_alloc(&mut scratch)
            .expect_err("duplicate must error");
        assert_eq!(
            err,
            PermutationValidationError::DuplicateIndex {
                src_index: 2,
                value: 1,
            }
        );
    }

    #[test]
    fn try_validate_no_alloc_rejects_out_of_range() {
        // SAFETY: deliberately invalid — used only to exercise the validator.
        let perm = unsafe { Permutation::from_vec_unchecked(vec![0_u32, 1, 5]) };
        let mut scratch = [0_u64; 1];
        let err = perm
            .try_validate_no_alloc(&mut scratch)
            .expect_err("out-of-range must error");
        assert_eq!(
            err,
            PermutationValidationError::OutOfRangeIndex {
                src_index: 2,
                value: 5,
            }
        );
    }

    #[test]
    fn permutation_validation_error_display_renders_all_variants() {
        // Cover the Display impl so all arms are exercised.
        let cases = [
            PermutationValidationError::LengthOverflow { len: 1 << 33 },
            PermutationValidationError::ScratchTooSmall {
                needed_words: 2,
                actual_words: 1,
            },
            PermutationValidationError::OutOfRangeIndex {
                src_index: 1,
                value: 9,
            },
            PermutationValidationError::DuplicateIndex {
                src_index: 2,
                value: 1,
            },
        ];
        for c in cases {
            let s = format!("{c}");
            assert!(s.starts_with("Permutation::try_validate_no_alloc"));
        }
    }

    // ----- audit-R7-followup #10 — CsrGraph::try_neighbors_of / try_degree_of -----

    #[test]
    fn try_neighbors_of_matches_neighbors_of_on_well_formed_csr() {
        // Path: 0-1-2-3, undirected.
        let offsets = [0_u32, 1, 3, 5, 6];
        let neighbors = [1_u32, 0, 2, 1, 3, 2];
        let g = CsrGraph {
            n: 4,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        for v in 0_u32..4 {
            let panic_path = g.neighbors_of(v);
            let try_path = g
                .try_neighbors_of(v)
                .expect("well-formed CSR must succeed");
            assert_eq!(panic_path, try_path);
        }
    }

    #[test]
    fn try_degree_of_matches_degree_on_well_formed_csr() {
        let offsets = [0_u32, 1, 3, 5, 6];
        let neighbors = [1_u32, 0, 2, 1, 3, 2];
        let g = CsrGraph {
            n: 4,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        for v in 0_u32..4 {
            let panic_path = g.degree(v);
            let try_path = g.try_degree_of(v).expect("well-formed CSR must succeed");
            assert_eq!(panic_path, try_path);
        }
    }

    #[test]
    fn try_neighbors_of_rejects_v_out_of_range() {
        let offsets = [0_u32, 1, 3];
        let neighbors = [1_u32, 0, 2];
        let g = CsrGraph {
            n: 2,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g
            .try_neighbors_of(2)
            .expect_err("v == n must error, not panic");
        assert_eq!(err, CsrGraphError::OutOfRange { v: 2, n: 2 });
        let err = g.try_neighbors_of(99).expect_err("v >> n must error");
        assert_eq!(err, CsrGraphError::OutOfRange { v: 99, n: 2 });
    }

    #[test]
    fn try_neighbors_of_rejects_offsets_non_monotone() {
        // Inverted offset pair at index 1: offsets[1]=5 > offsets[2]=2.
        // (Both are within neighbors.len() so the bounds check
        // succeeds; the failure mode under test is the monotone
        // check.)
        let offsets = [0_u32, 5, 2, 6];
        let neighbors = [1_u32, 2, 3, 4, 5, 6];
        let g = CsrGraph {
            n: 3,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g
            .try_neighbors_of(1)
            .expect_err("non-monotone offsets must error");
        assert_eq!(
            err,
            CsrGraphError::OffsetsNonMonotone {
                i: 1,
                lo: 5,
                hi: 2,
            }
        );
    }

    #[test]
    fn try_neighbors_of_rejects_neighbors_out_of_bounds() {
        // offsets[1] = 99 addresses past the end of neighbors.
        let offsets = [0_u32, 99];
        let neighbors = [1_u32, 0];
        let g = CsrGraph {
            n: 1,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g
            .try_neighbors_of(0)
            .expect_err("offset > neighbors.len() must error");
        assert_eq!(
            err,
            CsrGraphError::NeighborsOutOfBounds {
                offset: 99,
                neighbors_len: 2,
            }
        );
    }

    #[test]
    fn try_degree_of_rejects_v_out_of_range_and_non_monotone() {
        let offsets = [0_u32, 5, 2];
        let neighbors = [1_u32, 2, 3, 4, 5, 6];
        let g = CsrGraph {
            n: 2,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g.try_degree_of(7).expect_err("v >> n must error");
        assert_eq!(err, CsrGraphError::OutOfRange { v: 7, n: 2 });
        let err = g
            .try_degree_of(1)
            .expect_err("non-monotone offsets must error");
        assert_eq!(
            err,
            CsrGraphError::OffsetsNonMonotone {
                i: 1,
                lo: 5,
                hi: 2,
            }
        );
    }

    #[test]
    fn try_neighbors_of_rejects_truncated_offsets_array() {
        // offsets array has only one entry but n=1 — there is no
        // offsets[1] to read.
        let offsets = [0_u32];
        let neighbors: [u32; 0] = [];
        let g = CsrGraph {
            n: 1,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g
            .try_neighbors_of(0)
            .expect_err("truncated offsets must error");
        assert_eq!(err, CsrGraphError::OutOfRange { v: 0, n: 1 });
    }

    #[test]
    fn csr_graph_error_display_renders_all_variants() {
        // Cover every Display arm.
        let cases = [
            CsrGraphError::OutOfRange { v: 5, n: 3 },
            CsrGraphError::OffsetsNonMonotone {
                i: 1,
                lo: 5,
                hi: 2,
            },
            CsrGraphError::NeighborsOutOfBounds {
                offset: 99,
                neighbors_len: 5,
            },
            CsrGraphError::OffsetsLengthMismatch {
                actual_len: 3,
                expected_len: 5,
            },
            CsrGraphError::NeighborsLengthMismatch {
                offsets_tail: 7,
                neighbors_len: 6,
            },
            CsrGraphError::NeighborOutOfRange {
                neighbor: 9,
                n: 4,
                at_index: 2,
            },
        ];
        for c in cases {
            let s = format!("{c}");
            assert!(s.starts_with("CsrGraph"));
        }
    }

    // ----- audit-R7-followup #11 — CsrGraph::try_validate + PermutationConstructionError -----

    #[test]
    fn try_validate_accepts_well_formed_csr() {
        // Path: 0-1-2-3, undirected.
        let offsets = [0_u32, 1, 3, 5, 6];
        let neighbors = [1_u32, 0, 2, 1, 3, 2];
        let g = CsrGraph {
            n: 4,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        g.try_validate().expect("well-formed CSR must validate");
    }

    #[test]
    fn try_validate_accepts_empty_graph() {
        let offsets = [0_u32];
        let neighbors: [u32; 0] = [];
        let g = CsrGraph {
            n: 0,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        g.try_validate().expect("empty graph must validate");
    }

    #[test]
    fn try_validate_rejects_offsets_length_mismatch() {
        let offsets = [0_u32, 1];
        let neighbors = [1_u32, 0];
        let g = CsrGraph {
            n: 3, // expects 4 offsets
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g.try_validate().expect_err("length mismatch must error");
        assert_eq!(
            err,
            CsrGraphError::OffsetsLengthMismatch {
                actual_len: 2,
                expected_len: 4,
            }
        );
    }

    #[test]
    fn try_validate_rejects_non_monotone_offsets() {
        let offsets = [0_u32, 5, 2, 6];
        let neighbors = [1_u32, 2, 3, 4, 5, 6];
        let g = CsrGraph {
            n: 3,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g
            .try_validate()
            .expect_err("non-monotone offsets must error");
        assert_eq!(
            err,
            CsrGraphError::OffsetsNonMonotone {
                i: 1,
                lo: 5,
                hi: 2,
            }
        );
    }

    #[test]
    fn try_validate_rejects_neighbors_length_mismatch() {
        // offsets[n]=2 but neighbors only has 1 element.
        let offsets = [0_u32, 2];
        let neighbors = [0_u32];
        let g = CsrGraph {
            n: 1,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g.try_validate().expect_err("tail mismatch must error");
        assert_eq!(
            err,
            CsrGraphError::NeighborsLengthMismatch {
                offsets_tail: 2,
                neighbors_len: 1,
            }
        );
    }

    #[test]
    fn try_validate_rejects_neighbor_out_of_range() {
        // Single neighbour with id 99 but n=2.
        let offsets = [0_u32, 1, 1];
        let neighbors = [99_u32];
        let g = CsrGraph {
            n: 2,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = g
            .try_validate()
            .expect_err("out-of-range neighbour must error");
        assert_eq!(
            err,
            CsrGraphError::NeighborOutOfRange {
                neighbor: 99,
                n: 2,
                at_index: 0,
            }
        );
    }

    #[test]
    fn permutation_construction_error_display_renders() {
        let err = PermutationConstructionError::InvalidCsr(CsrGraphError::OutOfRange {
            v: 5,
            n: 3,
        });
        let s = format!("{err}");
        assert!(s.starts_with("permutation construction failed"));
        assert!(s.contains("CsrGraph"));
    }

    #[test]
    fn permutation_construction_error_from_csr_graph_error() {
        // Verify the From impl that the try_* builders rely on.
        let csr_err = CsrGraphError::OffsetsLengthMismatch {
            actual_len: 2,
            expected_len: 5,
        };
        let wrapped: PermutationConstructionError = csr_err.into();
        assert_eq!(wrapped, PermutationConstructionError::InvalidCsr(csr_err));
    }
}

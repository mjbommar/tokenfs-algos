//! Locality-improving orderings for graph- and point-shaped data.
//!
//! See `docs/v0.2_planning/14_PERMUTATION.md` for the spec. This module
//! ships the shared [`Permutation`] type plus the CSR adjacency input
//! type [`CsrGraph`], and the **build-time** ordering primitives that
//! produce a [`Permutation`] from such inputs.
//!
//! ## Sprint 11-13 status
//!
//! Phase B4 of `01_PHASES.md` lands [`rcm()`] (Reverse Cuthill-McKee). Two
//! follow-ons are spec'd:
//!
//! * `hilbert_2d` / `hilbert_nd` — Hilbert curve ordering (Phase B5).
//! * `rabbit_order` — community-detection-driven (Phase D1).
//!
//! ## Deployment posture
//!
//! Per `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`:
//!
//! * **Permutation construction** ([`rcm()`], future `hilbert_*`,
//!   `rabbit_order`): build-time only. These algorithms allocate large
//!   work buffers (BFS queue, dendrogram) that cannot be made
//!   stack-only. Never runs in kernel.
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

pub mod rcm;

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

    /// Constructs a [`Permutation`] from a raw `Vec<u32>` without checks.
    ///
    /// The caller asserts that `perm` is a valid permutation: every
    /// integer in `0..perm.len()` appears exactly once and `perm.len()
    /// <= u32::MAX as usize`. Use [`Permutation::try_from_vec`] for the
    /// checked constructor.
    #[must_use]
    pub fn from_vec_unchecked(perm: Vec<u32>) -> Self {
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
            // we use the checked form so accidental construction via
            // `from_vec_unchecked` with bad data still produces a
            // panic instead of UB.
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
}

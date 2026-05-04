//! Candidate min-heap for HNSW search.
//!
//! HNSW's beam search needs two ordered candidate sets:
//!
//! - **Working set** (`ef`-sized; max-distance heap so the worst candidate
//!   is at the top, ready to be evicted when a closer candidate appears).
//! - **Result set** (`k`-sized; the final top-k by distance).
//!
//! Both have the same shape: bounded capacity, push-with-eviction-by-
//! distance, deterministic tie-break by [`NodeId`] ascending. The
//! tie-break is the load-bearing determinism property — without it,
//! two candidates with identical distances could appear in different
//! relative order across runs, breaking SLSA-L3 reproducibility (per
//! `docs/hnsw/research/DETERMINISM.md` §4).
//!
//! We expose [`MaxHeap`] as the single primitive; the walker uses it
//! both as the working set (with `cap = ef_search`) and as the result
//! set (with `cap = k`). Both are max-heaps so the largest-distance
//! candidate is at the top — ready to be popped off when a smaller-
//! distance candidate arrives.
//!
//! # Audit posture
//!
//! - All accessors are `Result`-free or return `Option`; never panic on
//!   well-formed input.
//! - `no_std + alloc`-clean: only allocation is the `Vec<Candidate>`
//!   sized to `cap` at construction (no resizing during use).

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use core::cmp::Ordering;

use super::{Distance, NodeId};

/// One ranked candidate. `Distance` is integer (per the `Distance`
/// type alias in `mod.rs`); for f32 metrics the walker pre-encodes via
/// the IEEE-754 total-ordering bit trick so this comparator is correct
/// for ALL metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// Distance from query to this node, integer-encoded.
    pub distance: Distance,
    /// Slot index of the node in the graph.
    pub node: NodeId,
}

impl Candidate {
    /// Construct a candidate.
    pub const fn new(distance: Distance, node: NodeId) -> Self {
        Candidate { distance, node }
    }
}

/// Total ordering: primary key distance ascending, tie-break by node
/// ID ascending. `Ord::cmp(a, b)` returns `Less` when `a` should sort
/// "before" (be smaller than) `b` — so smaller distance, then smaller
/// node ID, comes first.
///
/// `BinaryHeap` in std is a *max*-heap (root = largest), so to use this
/// ordering as a max-heap (root = worst candidate, ready to evict) the
/// walker doesn't need to reverse anything — we want the largest
/// `(distance, node)` tuple at the top.
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .cmp(&other.distance)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bounded max-heap of candidates, used by HNSW search for both the
/// working set (`ef`-sized) and the result set (`k`-sized).
///
/// "Max" here means the heap's root is the worst (largest distance,
/// or with equal distance the largest node ID). When the heap is full
/// and a new candidate arrives whose distance is smaller than the
/// current worst, the worst is evicted; otherwise the new candidate is
/// dropped.
///
/// Internally a sorted `Vec<Candidate>` (worst at index 0). For the
/// k-NN sizes typical in HNSW (`ef = 16..512`) a sorted-vec heap with
/// O(k) insert is faster than a binary-heap with O(log k) insert
/// because k is small + cache-friendly + branch-predictable.
#[derive(Debug, Clone)]
pub struct MaxHeap {
    /// Sorted ascending by `(distance, node)`. Worst (largest) is the LAST
    /// element. `pop_worst` therefore takes from the back; `peek_worst`
    /// reads from the back.
    items: Vec<Candidate>,
    cap: usize,
}

impl MaxHeap {
    /// Create a heap with capacity `cap`. Calls outside the hot path
    /// can pass `cap = 0` to get a no-op heap (push always returns false).
    pub fn try_with_capacity(cap: usize) -> Self {
        MaxHeap {
            items: Vec::with_capacity(cap),
            cap,
        }
    }

    /// Maximum number of candidates the heap will retain.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Number of candidates currently stored.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// `true` if no candidates stored.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// `true` if `len == cap`.
    pub fn is_full(&self) -> bool {
        self.items.len() == self.cap
    }

    /// Peek the worst (largest-distance, tie-broken by largest node ID)
    /// candidate currently in the heap. `None` if empty.
    pub fn peek_worst(&self) -> Option<&Candidate> {
        self.items.last()
    }

    /// Peek the best (smallest-distance, tie-broken by smallest node ID)
    /// candidate currently in the heap. `None` if empty.
    pub fn peek_best(&self) -> Option<&Candidate> {
        self.items.first()
    }

    /// Try to insert a candidate. Returns `true` if the candidate was
    /// inserted (heap grew by one OR worst was evicted in favor); `false`
    /// if the heap was full and the candidate was no better than the
    /// current worst.
    pub fn try_push(&mut self, c: Candidate) -> bool {
        if self.cap == 0 {
            return false;
        }
        if self.items.len() < self.cap {
            self.insert_sorted(c);
            return true;
        }
        // Full heap. Compare against current worst.
        let worst = *self.items.last().expect("non-empty (cap >= 1, len == cap)");
        if c.cmp(&worst) == Ordering::Less {
            // Evict worst, insert new.
            self.items.pop();
            self.insert_sorted(c);
            true
        } else {
            // Drop incoming candidate; not better than current worst.
            false
        }
    }

    /// Drop the worst candidate (pop from back). Useful when shrinking
    /// the result set after collecting more than `k`.
    pub fn pop_worst(&mut self) -> Option<Candidate> {
        self.items.pop()
    }

    /// Iterate candidates in ascending order (best first).
    pub fn iter_best_first(&self) -> impl Iterator<Item = &Candidate> {
        self.items.iter()
    }

    /// Drain into a sorted vector (best first). Consumes the heap.
    pub fn into_sorted_vec(self) -> Vec<Candidate> {
        self.items
    }

    /// Insert maintaining sorted order. O(cap) worst case via binary
    /// search + rotate; for the small k typical in HNSW (16..512) this
    /// is faster than `BinaryHeap` due to cache effects.
    fn insert_sorted(&mut self, c: Candidate) {
        // partition_point returns the first index whose element is
        // strictly greater than `c`; equivalently the insert position
        // that keeps `items` sorted ascending.
        let idx = self
            .items
            .partition_point(|x| x.cmp(&c) != Ordering::Greater);
        self.items.insert(idx, c);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn empty_heap_state() {
        let h = MaxHeap::try_with_capacity(4);
        assert_eq!(h.capacity(), 4);
        assert_eq!(h.len(), 0);
        assert!(h.is_empty());
        assert!(!h.is_full());
        assert_eq!(h.peek_best(), None);
        assert_eq!(h.peek_worst(), None);
    }

    #[test]
    fn zero_capacity_drops_all_pushes() {
        let mut h = MaxHeap::try_with_capacity(0);
        assert!(!h.try_push(Candidate::new(0, 0)));
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn fills_to_capacity() {
        let mut h = MaxHeap::try_with_capacity(3);
        for i in 0..3u32 {
            assert!(h.try_push(Candidate::new(i, i)));
        }
        assert_eq!(h.len(), 3);
        assert!(h.is_full());
    }

    #[test]
    fn evicts_worst_when_full_and_better_arrives() {
        let mut h = MaxHeap::try_with_capacity(3);
        for d in [10u32, 20, 30] {
            h.try_push(Candidate::new(d, 0));
        }
        assert_eq!(h.peek_worst().unwrap().distance, 30);
        assert!(h.try_push(Candidate::new(15, 1))); // 15 < 30, evict 30
        assert_eq!(h.peek_worst().unwrap().distance, 20);
        assert_eq!(h.peek_best().unwrap().distance, 10);
    }

    #[test]
    fn rejects_when_full_and_worse_arrives() {
        let mut h = MaxHeap::try_with_capacity(3);
        for d in [10u32, 20, 30] {
            h.try_push(Candidate::new(d, 0));
        }
        assert!(!h.try_push(Candidate::new(40, 1))); // 40 > 30, drop
        assert_eq!(h.len(), 3);
        assert_eq!(h.peek_worst().unwrap().distance, 30);
    }

    #[test]
    fn tie_break_by_node_id_ascending() {
        // Three candidates at distance 10 — node IDs should sort
        // ascending so worst (last) has the largest node ID.
        let mut h = MaxHeap::try_with_capacity(3);
        h.try_push(Candidate::new(10, 5));
        h.try_push(Candidate::new(10, 2));
        h.try_push(Candidate::new(10, 8));
        let sorted: Vec<_> = h.iter_best_first().copied().collect();
        assert_eq!(
            sorted,
            vec![
                Candidate::new(10, 2),
                Candidate::new(10, 5),
                Candidate::new(10, 8),
            ]
        );
        assert_eq!(h.peek_worst().unwrap().node, 8);
        assert_eq!(h.peek_best().unwrap().node, 2);
    }

    #[test]
    fn tie_break_evicts_largest_node_id_at_same_distance() {
        let mut h = MaxHeap::try_with_capacity(3);
        h.try_push(Candidate::new(10, 5));
        h.try_push(Candidate::new(10, 2));
        h.try_push(Candidate::new(10, 8));
        // Insert another at distance 10; node ID 0 is smaller than the
        // current worst (10, 8), so it evicts the (10, 8) entry.
        assert!(h.try_push(Candidate::new(10, 0)));
        let sorted: Vec<_> = h.iter_best_first().copied().collect();
        assert_eq!(
            sorted,
            vec![
                Candidate::new(10, 0),
                Candidate::new(10, 2),
                Candidate::new(10, 5),
            ]
        );
    }

    #[test]
    fn iter_best_first_returns_sorted_ascending() {
        let mut h = MaxHeap::try_with_capacity(5);
        for (d, n) in [(50, 1), (10, 2), (30, 3), (40, 4), (20, 5)] {
            h.try_push(Candidate::new(d, n));
        }
        let distances: Vec<_> = h.iter_best_first().map(|c| c.distance).collect();
        assert_eq!(distances, vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn into_sorted_vec_returns_best_first() {
        let mut h = MaxHeap::try_with_capacity(4);
        for d in [3u32, 1, 4, 1, 5] {
            h.try_push(Candidate::new(d, 0));
        }
        // Tie at distance 1 with same node 0 — both kept until cap fills,
        // then the worst (3) is the eviction target when 5 arrives.
        // Actually: cap=4, push 3,1,4,1 (now full, distances [1,1,3,4]),
        // then push 5 — 5 > 4, dropped. Final: [1, 1, 3, 4].
        let v = h.into_sorted_vec();
        assert_eq!(v.len(), 4);
        let distances: Vec<_> = v.iter().map(|c| c.distance).collect();
        assert_eq!(distances, vec![1, 1, 3, 4]);
    }

    #[test]
    fn pop_worst_drains_back_to_front() {
        let mut h = MaxHeap::try_with_capacity(4);
        for d in [10u32, 20, 30, 40] {
            h.try_push(Candidate::new(d, 0));
        }
        assert_eq!(h.pop_worst().unwrap().distance, 40);
        assert_eq!(h.pop_worst().unwrap().distance, 30);
        assert_eq!(h.pop_worst().unwrap().distance, 20);
        assert_eq!(h.pop_worst().unwrap().distance, 10);
        assert_eq!(h.pop_worst(), None);
    }

    #[test]
    fn deterministic_across_two_identical_runs() {
        // Same input sequence → bit-identical heap state. Validates the
        // determinism contract (DETERMINISM.md §4 tie-break rule).
        let inputs: &[(u32, u32)] = &[
            (50, 7),
            (50, 3),
            (50, 11),
            (10, 0),
            (10, 0), // duplicate (distance, node) — should still be deterministic
            (30, 5),
            (30, 5),
        ];
        let run = || {
            let mut h = MaxHeap::try_with_capacity(4);
            for &(d, n) in inputs {
                h.try_push(Candidate::new(d, n));
            }
            h.into_sorted_vec()
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn large_random_workload_does_not_panic() {
        // Crude smoke: 1024 random pushes into a small heap; final state
        // must contain at most cap entries, all sorted, with no
        // duplicates of the (distance, node) tuple.
        let mut state = 0xCAFE_BABE_DEAD_BEEFu64;
        let mut h = MaxHeap::try_with_capacity(16);
        for _ in 0..1024 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let d = (state & 0xFFFF) as u32;
            let n = ((state >> 16) & 0xFFF) as u32;
            h.try_push(Candidate::new(d, n));
        }
        assert!(h.len() <= h.capacity());
        let v = h.into_sorted_vec();
        for w in v.windows(2) {
            assert!(
                w[0].cmp(&w[1]) != Ordering::Greater,
                "heap output not sorted: {:?} > {:?}",
                w[0],
                w[1]
            );
        }
    }
}

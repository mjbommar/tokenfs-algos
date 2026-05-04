//! Visited-set tracking for HNSW search.
//!
//! Search requires marking each visited node exactly once per query so
//! the beam-search loop doesn't re-evaluate distance to the same
//! candidate. Two implementation choices, both common in production
//! HNSW:
//!
//! - **Bitset**: one bit per node. Cheap insert/contains; expensive
//!   `clear` (O(node_count) memset).
//! - **Generation counter**: `Vec<u32>` of generation tags;
//!   `clear` is just incrementing a single counter (O(1)). A node is
//!   considered "visited" if its tag matches the current generation.
//!
//! Walker hot path issues O(1) clears between queries (one search → next
//! search) so the generation-counter approach is the right call. We
//! still expose `mark` / `was_marked` semantically as a set; the
//! generation tag is an implementation detail.
//!
//! # Audit posture
//!
//! - All accessors return `Result` or `bool`; never panic.
//! - `no_std + alloc`-clean: only allocation is the `Vec<u32>` of
//!   generation tags, sized to `node_count` at construction.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use super::NodeId;

/// Generation-counter visited set.
///
/// `clear()` is O(1) (increments the generation counter). The
/// `Vec<u32>` only resizes when the generation counter wraps around
/// `u32::MAX` — at typical HNSW query rates that's after ~4 billion
/// queries between resets, so in practice never.
#[derive(Debug, Clone)]
pub struct VisitedSet {
    /// Per-node generation tag. A node is "marked" iff `tags[id] == generation`.
    tags: Vec<u32>,
    /// Current generation. Bumped by `clear()`.
    generation: u32,
}

impl VisitedSet {
    /// Create a new visited set sized for `node_count` nodes. Initial
    /// state: nothing marked.
    pub fn try_with_capacity(node_count: usize) -> Self {
        // Generation starts at 1 so the initial all-zero `tags`
        // correctly represents "nothing visited" without an initial
        // memset on a fresh allocation. Using `core::iter::repeat`
        // avoids both the `vec![]` macro (which needs `extern crate
        // alloc` in scope under no_std + alloc) and clippy's
        // `slow_vector_initialization` lint on the `with_capacity` +
        // `resize` pattern.
        let tags: Vec<u32> = core::iter::repeat_n(0u32, node_count).collect();
        VisitedSet {
            tags,
            generation: 1,
        }
    }

    /// Number of nodes this set can track. Equal to `node_count`
    /// supplied at construction.
    pub fn capacity(&self) -> usize {
        self.tags.len()
    }

    /// Reset the visited set in O(1). After this call no node is
    /// marked. If the generation counter would overflow, falls back to
    /// an O(n) memset and resets the counter to 1.
    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            // Wrapped past u32::MAX. Reset all tags to 0 and bump
            // generation to 1 so "tag == generation" still works.
            for tag in &mut self.tags {
                *tag = 0;
            }
            self.generation = 1;
        }
    }

    /// Mark `id`. Returns `true` if newly marked, `false` if already
    /// marked or `id` is out of range. Out-of-range silently no-ops to
    /// keep the hot path branch-free; callers must ensure `id` is in
    /// range via `HnswView::try_node`'s prior check.
    pub fn mark(&mut self, id: NodeId) -> bool {
        let idx = id as usize;
        if idx >= self.tags.len() {
            return false;
        }
        let was_unmarked = self.tags[idx] != self.generation;
        self.tags[idx] = self.generation;
        was_unmarked
    }

    /// Test whether `id` is marked in the current generation. Returns
    /// `false` for out-of-range IDs.
    pub fn was_marked(&self, id: NodeId) -> bool {
        let idx = id as usize;
        if idx >= self.tags.len() {
            return false;
        }
        self.tags[idx] == self.generation
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn empty_set_no_nodes_marked() {
        let set = VisitedSet::try_with_capacity(8);
        for id in 0..8u32 {
            assert!(!set.was_marked(id));
        }
    }

    #[test]
    fn mark_then_check_returns_true() {
        let mut set = VisitedSet::try_with_capacity(8);
        assert!(set.mark(3));
        assert!(set.was_marked(3));
        assert!(!set.was_marked(4));
    }

    #[test]
    fn double_mark_returns_false_second_time() {
        let mut set = VisitedSet::try_with_capacity(4);
        assert!(set.mark(0));
        assert!(!set.mark(0));
        assert!(set.was_marked(0));
    }

    #[test]
    fn clear_unmarks_everything_in_o1() {
        let mut set = VisitedSet::try_with_capacity(8);
        for id in 0..8u32 {
            set.mark(id);
        }
        for id in 0..8u32 {
            assert!(set.was_marked(id));
        }
        set.clear();
        for id in 0..8u32 {
            assert!(
                !set.was_marked(id),
                "node {id} should be unmarked after clear"
            );
        }
    }

    #[test]
    fn out_of_range_mark_is_no_op() {
        let mut set = VisitedSet::try_with_capacity(4);
        assert!(!set.mark(99));
        assert!(!set.was_marked(99));
        // Should not have affected in-range entries.
        for id in 0..4u32 {
            assert!(!set.was_marked(id));
        }
    }

    #[test]
    fn mark_persists_across_unrelated_marks() {
        let mut set = VisitedSet::try_with_capacity(16);
        set.mark(5);
        set.mark(7);
        set.mark(9);
        assert!(set.was_marked(5));
        assert!(set.was_marked(7));
        assert!(set.was_marked(9));
        assert!(!set.was_marked(6));
        assert!(!set.was_marked(8));
    }

    #[test]
    fn generation_wraparound_resets_via_o_n_path() {
        let mut set = VisitedSet::try_with_capacity(4);
        // Force generation to u32::MAX so the next clear() wraps to 0,
        // triggering the O(n) reset path.
        set.generation = u32::MAX;
        set.mark(2);
        assert!(set.was_marked(2));
        set.clear();
        // Should not be marked even though tags[2] still has the old
        // generation value (because the clear branch reset all tags).
        assert!(!set.was_marked(2));
        assert_eq!(set.generation, 1);
        // And new marks still work post-wraparound.
        set.mark(3);
        assert!(set.was_marked(3));
    }

    #[test]
    fn capacity_matches_construction() {
        let set = VisitedSet::try_with_capacity(42);
        assert_eq!(set.capacity(), 42);
    }
}

//! In-search filter primitives for HNSW.
//!
//! Lets the walker prune candidates by external membership during
//! beam-search rather than after, preserving the sub-linear search win
//! at low selectivity. Per `docs/hnsw/components/FILTER.md` and
//! `docs/hnsw/research/HNSW_ALGORITHM_NOTES.md` §7 (ACORN-shape).
//!
//! ## Why in-search vs post-filter
//!
//! Post-filter ("retrieve top-k unfiltered, then drop") destroys the
//! HNSW recall guarantee at low selectivity: most retrieved
//! candidates get dropped, and the rejected slots are lost capacity.
//! In-search pruning skips disallowed neighbors at the
//! result-acceptance step but **keeps them in the graph-hop
//! expansion** so the search can route through denied-but-connected
//! intermediate nodes to reach permitted ones.
//!
//! ## Brute-force fallback
//!
//! At very low selectivity (default <5%), the HNSW graph traversal
//! cost dominates the per-candidate distance compute, and a flat
//! brute-force scan over the permitted set is faster. The walker
//! falls back automatically per `SearchConfig::brute_force_threshold`
//! (added in this commit).
//!
//! ## Audit posture
//!
//! - `HnswFilter` is a borrowed view; no allocation, no panic.
//! - All accessors are kernel-safe (`no_std + alloc`).

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use super::NodeId;

/// Filter that controls which `NodeId`s the walker may include in
/// search results.
///
/// Currently backed by a sorted permitted-id `Vec<NodeId>` for O(log n)
/// membership; future revisions may swap in a Roaring bitmap when
/// the v0.7+ similarity workloads need very-large permitted sets at
/// reasonable per-check cost (Roaring's array+bitmap container model
/// is in `crate::bitmap`; we'd compose against it here).
#[derive(Debug, Clone)]
pub struct HnswFilter<'a> {
    permitted_sorted: &'a [NodeId],
    total_nodes: usize,
}

impl<'a> HnswFilter<'a> {
    /// Construct from a sorted slice of permitted node IDs and the
    /// view's total node count (for selectivity estimation).
    ///
    /// **Caller contract:** `permitted_sorted` MUST be ascending +
    /// duplicate-free. `total_nodes` MUST equal `view.node_count()`.
    /// Violations don't cause UB but produce wrong filter behavior.
    pub fn new(permitted_sorted: &'a [NodeId], total_nodes: usize) -> Self {
        debug_assert!(
            permitted_sorted.windows(2).all(|w| w[0] < w[1]),
            "permitted_sorted must be ascending + duplicate-free"
        );
        HnswFilter {
            permitted_sorted,
            total_nodes,
        }
    }

    /// O(log n) membership check via binary search.
    pub fn permits(&self, id: NodeId) -> bool {
        self.permitted_sorted.binary_search(&id).is_ok()
    }

    /// Number of permitted nodes.
    pub fn permitted_count(&self) -> usize {
        self.permitted_sorted.len()
    }

    /// Selectivity ratio in [0.0, 1.0]. Used by the walker to decide
    /// whether to fall back to brute-force.
    pub fn selectivity(&self) -> f32 {
        if self.total_nodes == 0 {
            return 0.0;
        }
        self.permitted_sorted.len() as f32 / self.total_nodes as f32
    }

    /// Iterate the permitted node IDs in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = NodeId> + 'a {
        self.permitted_sorted.iter().copied()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn empty_filter_permits_nothing() {
        let f = HnswFilter::new(&[], 100);
        assert_eq!(f.permitted_count(), 0);
        assert_eq!(f.selectivity(), 0.0);
        assert!(!f.permits(0));
        assert!(!f.permits(99));
    }

    #[test]
    fn full_filter_permits_everything() {
        let permitted: Vec<NodeId> = (0..10).collect();
        let f = HnswFilter::new(&permitted, 10);
        assert_eq!(f.permitted_count(), 10);
        assert_eq!(f.selectivity(), 1.0);
        for id in 0..10 {
            assert!(f.permits(id));
        }
        assert!(!f.permits(10));
    }

    #[test]
    fn partial_filter_permits_listed_only() {
        let permitted = [1, 3, 5, 7];
        let f = HnswFilter::new(&permitted, 10);
        assert_eq!(f.selectivity(), 0.4);
        for id in [1, 3, 5, 7] {
            assert!(f.permits(id));
        }
        for id in [0, 2, 4, 6, 8, 9, 10] {
            assert!(!f.permits(id));
        }
    }

    #[test]
    fn iter_preserves_order() {
        let permitted = [2, 4, 6, 8];
        let f = HnswFilter::new(&permitted, 100);
        let collected: Vec<NodeId> = f.iter().collect();
        assert_eq!(collected, permitted);
    }

    #[test]
    fn selectivity_handles_zero_total() {
        let f = HnswFilter::new(&[], 0);
        assert_eq!(f.selectivity(), 0.0);
    }
}

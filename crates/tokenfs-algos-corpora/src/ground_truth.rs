//! Ground-truth labels for generated corpora.
//!
//! The load-bearing addition over a normal data generator is
//! **constructed ground truth**. Tests assert
//! `assert!(recall >= 0.95)` instead of "looks plausible."

use crate::{ClusterId, NodeKey};

/// Constructed ground truth for a [`crate::Corpus`].
#[derive(Debug, Clone, Default)]
pub struct GroundTruth {
    /// Cluster definitions when the bytes layer assigns clusters
    /// (e.g. bit-flip-variants). Empty for unstructured generators.
    pub clusters: Vec<Cluster>,
    /// Pre-computed expected k-NN per item, when small-N generators
    /// can afford brute force at generation time. Sparse — only
    /// populated for `Scale::Tiny` / `Scale::Small`. Larger scales
    /// rely on cluster-membership recall instead of per-query k-NN.
    pub expected_knn: Option<ExpectedKnn>,
}

impl GroundTruth {
    /// Empty ground truth — no clusters, no expected k-NN. Used by
    /// `CorpusSpec::generate` for unstructured generators or the
    /// scaffolded zero-item corpora returned by Phase-1 stubs.
    pub fn empty() -> Self {
        GroundTruth::default()
    }
}

/// One cluster: the seed key (the canonical member) plus all variant
/// keys belonging to it.
#[derive(Debug, Clone)]
pub struct Cluster {
    /// Cluster identifier (sequential from 0).
    pub id: ClusterId,
    /// All `NodeKey`s belonging to this cluster, in generation order.
    pub members: Vec<NodeKey>,
}

/// Pre-computed expected k-NN for ground-truth recall measurement.
#[derive(Debug, Clone, Default)]
pub struct ExpectedKnn {
    /// `k` value used to compute the expected results.
    pub k: usize,
    /// One sorted ascending list of `(neighbor_key, distance)` per
    /// item (length == [`crate::Corpus::items.len()`]).
    pub per_item: Vec<Vec<(NodeKey, u32)>>,
}

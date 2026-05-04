//! Owned in-memory HNSW graph (Phase 4 builder substrate).
//!
//! Mutable counterpart to [`super::view::HnswView`]. Built by
//! [`super::build::Builder`] via per-vector `try_insert` calls;
//! serialized to usearch v2.25 wire format via
//! [`super::build::serialize::serialize_to_bytes`] (when that lands).
//!
//! Each node holds:
//! - external `NodeKey` (u64)
//! - `level: u8` (top layer this node participates in)
//! - per-level neighbor list `Vec<NodeId>` of length `[0..]`
//! - vector bytes (length == `bytes_per_vector`)
//!
//! Neighbor lists grow during insert + shrink during the
//! Algorithm-1 line-13-16 prune step. Capped at `M` (upper layers)
//! / `M0` (base) by the builder.

#![cfg(feature = "std")]

use super::{NodeId, NodeKey};

/// Owned mutable graph state used by the builder.
#[derive(Debug, Clone)]
pub struct Graph {
    /// One entry per inserted node. Indexed by `NodeId`.
    pub(super) nodes: Vec<Node>,
    /// Slot of the entry-point (top-level) node. `None` until first insert.
    pub(super) entry_point: Option<NodeId>,
    /// Top level present anywhere in the graph (== level of `entry_point`).
    pub(super) max_level: u8,
    /// Bytes per vector (constant, set at construction).
    pub(super) bytes_per_vector: usize,
}

/// One node in the owned graph.
#[derive(Debug, Clone)]
pub struct Node {
    /// External key (u64; matches `index_dense_t` default).
    pub(super) key: NodeKey,
    /// Top layer this node participates in. Layer 0 is base.
    pub(super) level: u8,
    /// Neighbor lists per layer. Length == `level + 1`.
    /// `neighbors[0]` is the base-layer slab (cap `M0`).
    /// `neighbors[i]` for i > 0 is the layer-i slab (cap `M`).
    pub(super) neighbors: Vec<Vec<NodeId>>,
    /// Raw vector bytes (length == `Graph::bytes_per_vector`).
    pub(super) vector: Vec<u8>,
}

impl Graph {
    /// Create an empty graph.
    pub fn new(bytes_per_vector: usize) -> Self {
        Graph {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            bytes_per_vector,
        }
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Top level of the graph (entry-point's level). 0 for an empty graph.
    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    /// Entry-point slot (None for an empty graph).
    pub fn entry_point(&self) -> Option<NodeId> {
        self.entry_point
    }

    /// Bytes per vector (constant).
    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
    }

    /// Get a node by slot. Returns `None` for out-of-range.
    pub fn try_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id as usize)
    }

    /// Get a mutable node by slot.
    pub fn try_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(id as usize)
    }

    /// Append a node, returning its slot.
    pub fn push_node(&mut self, key: NodeKey, level: u8, vector: Vec<u8>) -> NodeId {
        debug_assert_eq!(vector.len(), self.bytes_per_vector);
        let slot = self.nodes.len() as NodeId;
        self.nodes.push(Node {
            key,
            level,
            neighbors: (0..=level as usize).map(|_| Vec::new()).collect(),
            vector,
        });
        if self.entry_point.is_none() || level > self.max_level {
            self.entry_point = Some(slot);
            self.max_level = level;
        }
        slot
    }
}

impl Node {
    /// External key.
    pub fn key(&self) -> NodeKey {
        self.key
    }

    /// Top level.
    pub fn level(&self) -> u8 {
        self.level
    }

    /// Vector bytes.
    pub fn vector(&self) -> &[u8] {
        &self.vector
    }

    /// Neighbor list at the given layer.
    pub fn neighbors_at(&self, layer: u8) -> Option<&[NodeId]> {
        self.neighbors.get(layer as usize).map(Vec::as_slice)
    }

    /// Mutable neighbor list at the given layer.
    pub fn neighbors_at_mut(&mut self, layer: u8) -> Option<&mut Vec<NodeId>> {
        self.neighbors.get_mut(layer as usize)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn empty_graph_has_no_entry_point() {
        let g = Graph::new(8);
        assert!(g.is_empty());
        assert_eq!(g.entry_point(), None);
        assert_eq!(g.max_level(), 0);
    }

    #[test]
    fn push_node_assigns_sequential_slots() {
        let mut g = Graph::new(4);
        let s0 = g.push_node(100, 0, vec![1, 2, 3, 4]);
        let s1 = g.push_node(101, 0, vec![5, 6, 7, 8]);
        assert_eq!(s0, 0);
        assert_eq!(s1, 1);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.entry_point(), Some(0));
    }

    #[test]
    fn entry_point_promotes_when_higher_level_inserted() {
        let mut g = Graph::new(4);
        g.push_node(100, 0, vec![0; 4]);
        g.push_node(101, 0, vec![0; 4]);
        g.push_node(102, 3, vec![0; 4]); // higher level
        assert_eq!(g.entry_point(), Some(2));
        assert_eq!(g.max_level(), 3);
    }

    #[test]
    fn neighbors_at_returns_per_layer_slabs() {
        let mut g = Graph::new(4);
        let slot = g.push_node(100, 2, vec![0; 4]); // level 2 → 3 slabs
        let n = g.try_node(slot).unwrap();
        assert_eq!(n.neighbors_at(0).unwrap().len(), 0);
        assert_eq!(n.neighbors_at(1).unwrap().len(), 0);
        assert_eq!(n.neighbors_at(2).unwrap().len(), 0);
        assert!(n.neighbors_at(3).is_none());
    }

    #[test]
    fn neighbors_at_mut_allows_push() {
        let mut g = Graph::new(4);
        let slot = g.push_node(100, 1, vec![0; 4]);
        g.try_node_mut(slot)
            .unwrap()
            .neighbors_at_mut(0)
            .unwrap()
            .push(7);
        let n = g.try_node(slot).unwrap();
        assert_eq!(n.neighbors_at(0).unwrap(), &[7]);
    }
}

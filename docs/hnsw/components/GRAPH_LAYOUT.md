# Component: Graph layout (in-memory + zero-copy view)

**Status:** skeleton, 2026-05-03. Filled in across Phases 1 + 4.

**Lives in:** `crates/tokenfs-algos/src/similarity/hnsw/{view.rs, graph.rs}`

## Role

Two complementary representations of the HNSW graph:

- **`HnswView<'a>`** (Phase 1): zero-copy view over mmapped section bytes. Read-only. Used by the walker.
- **`Graph`** (Phase 4): owned in-memory representation with growable neighbor lists. Used by the builder during insert; serialized to wire format on `finish`.

Both representations agree on the same node-id space and the same neighbor-list layout per level. The wire format is the contract between them.

## Required research input

- [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md) §1 (header) + §2 (graph topology + addressing)
- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) §1 (Algorithm 1 — what the builder needs to mutate)
- [`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md) (`permutation::*` patterns for graph traversal)

## Sections to fill in

1. **NodeId vs NodeKey.** `NodeId` is the internal slot index (u32); `NodeKey` is the external key supplied by the caller (typically u64 for TokenFS extent IDs). The wire format stores both.
2. **Per-node tape format.** Header (key + level), vector blob, per-level neighbor list. Aligned per usearch's spec (see `WIRE_FORMAT.md`).
3. **Vector blob layout.** Whether vectors are interleaved with nodes or in a separate blob. (Per usearch v2.25: TBD pending `USEARCH_DEEP_DIVE.md`.)
4. **Neighbor list packing.** uint40 vs u32 vs variable-length. Trade-offs.
5. **Owned `Graph` mutation primitives.** `add_node`, `set_neighbors_at`, `prune_neighbors_at`. Bidirectional-edge invariants.
6. **Zero-copy `HnswView` access.** `try_node` returns `NodeRef<'a>` with bounds checks; `NodeRef::try_neighbors(level)` returns `&'a [NodeId]`; no allocation, no panics.
7. **Serialize/deserialize duality.** `Graph` → bytes via `build/serialize.rs`. Bytes → `HnswView` via `view.rs::try_new`. Round-trip test asserts equivalence.

## API skeleton

```rust
pub type NodeId = u32;
pub type NodeKey = u64;

// view.rs (no_std + alloc; zero-copy)
pub struct HnswView<'a> { /* opaque */ }
impl<'a> HnswView<'a> {
    pub fn try_new(bytes: &'a [u8]) -> Result<Self, HnswViewError>;
    pub fn node_count(&self) -> usize;
    pub fn dimensions(&self) -> usize;
    pub fn entry_point(&self) -> Option<NodeId>;
    pub fn try_node(&self, id: NodeId) -> Result<NodeRef<'a>, HnswViewError>;
}
pub struct NodeRef<'a> { /* opaque */ }
impl<'a> NodeRef<'a> {
    pub fn key(&self) -> NodeKey;
    pub fn level(&self) -> u8;
    pub fn try_neighbors(&self, level: u8) -> Result<&'a [NodeId], HnswViewError>;
    pub fn vector_bytes(&self) -> &'a [u8];
}

// graph.rs (cfg(feature = "std"); owned, mutable)
#[cfg(feature = "std")]
pub struct Graph { /* opaque */ }

#[cfg(feature = "std")]
impl Graph {
    pub fn try_new(config: &BuildConfig) -> Result<Self, HnswBuildError>;
    pub fn try_add_node(&mut self, key: NodeKey, vector: Vec<u8>, level: u8) -> Result<NodeId, HnswBuildError>;
    pub fn try_set_neighbors(&mut self, id: NodeId, level: u8, neighbors: &[NodeId]) -> Result<(), HnswBuildError>;
    pub fn neighbors_at(&self, id: NodeId, level: u8) -> &[NodeId];
    pub fn entry_point(&self) -> Option<NodeId>;
    pub fn set_entry_point(&mut self, id: NodeId);
    pub fn node_count(&self) -> usize;
    pub fn current_max_level(&self) -> u8;
}
```

## Cross-references

- Phases: [`../phases/PHASE_1.md`](../phases/PHASE_1.md) (HnswView + NodeRef), [`../phases/PHASE_4.md`](../phases/PHASE_4.md) (Graph + mutation)
- Related: [`WIRE_FORMAT.md`](WIRE_FORMAT.md) (the bytes both representations agree on), [`WALKER.md`](WALKER.md) (the consumer of HnswView), [`BUILDER.md`](BUILDER.md) (the consumer of Graph)
- Pattern reference: `src/permutation/*` (existing graph traversal patterns in the crate)

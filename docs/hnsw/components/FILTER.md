# Component: Filter primitives

**Status:** skeleton, 2026-05-03. Filled in during Phase 3.

**Lives in:** `crates/tokenfs-algos/src/similarity/hnsw/filter.rs`

## Role

In-search Roaring-bitmap pruning of HNSW candidates. Required for capability-aware search (per `NATIVE_HYBRID_SIMILARITY.md` N3 + `SIMILARITY_API_SURFACE.md`). Composes with existing `bitmap::*` SIMD intersect/cardinality kernels.

## Required research input

- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) §7 (filter integration: in-search vs post-filter)
- [`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md) (`bitmap::*` reuse map)

## Key findings to fold in

- **`bitmap::Container` directly satisfies the requirement.** The existing crate's `crates/tokenfs-algos/src/bitmap/` provides a Roaring-style `Container` with SIMD intersect/union/cardinality. `HnswFilter` is a thin newtype around `&Container`; no new SIMD work for filter primitives proper.
- **ACORN is the canonical filter-integration shape.** Per HNSW algorithm notes §7, ACORN keeps denied candidates in the graph-hop expansion but excludes them from the result heap. We follow ACORN's shape.
- **SeRF's edge-time annotation is out of scope for v1.** SeRF prebuilds per-predicate filter-aware indexes; we don't rebuild per filter.
- **Brute-force fallback threshold is empirical.** Default 5% selectivity; document in bench output.

## Sections to fill in

1. **Why in-search, not post-filter.** Post-filter reads top-k unfiltered then drops; for low-selectivity filters this destroys HNSW's sub-linear win because most retrieved nodes get discarded. In-search prunes during traversal.
2. **Why don't we just stop exploring denied subgraphs?** Because HNSW's neighbor structure means a permitted node may only be reachable through denied intermediate nodes. Denied nodes are NOT added to the result heap, but ARE explored as graph hops.
3. **Brute-force fallback at low selectivity.** When `filter.estimated_selectivity() < threshold` (default 5%), HNSW's sub-linear advantage is gone and a linear scan over the filter set is faster. Walker switches strategies at this cliff.
4. **Composition with `bitmap::*`.** `HnswFilter` wraps `&RoaringBitmap`; `permits(NodeId)` is O(1) (Roaring high-12-bits bucket lookup + low-16-bits bitmap test). Batch checks for the candidate-list expansion step use SIMD intersect from `bitmap::kernels`.
5. **Audit posture.** No new public surface beyond what's documented here. The filter passes through the walker's existing `try_*` discipline.

## API skeleton

```rust
pub struct HnswFilter<'a> {
    bitmap: &'a RoaringBitmap,
}

impl<'a> HnswFilter<'a> {
    pub fn new(bitmap: &'a RoaringBitmap) -> Self;
    pub fn permits(&self, id: NodeId) -> bool;
    pub fn permitted_count(&self) -> usize;
    pub fn estimated_selectivity(&self, total_nodes: usize) -> f32;
}

// Used by walker (see WALKER.md):
//   pub fn try_search_with_filter(view, query, config, &HnswFilter) -> Result<...>;
```

## Cross-references

- Phase: [`../phases/PHASE_3.md`](../phases/PHASE_3.md) (full implementation)
- Related: [`WALKER.md`](WALKER.md) (the search path that consumes the filter), `src/bitmap/*` (the SIMD primitives this composes with)
- External: `tokenfs-paper/docs/NATIVE_HYBRID_SIMILARITY.md` N3 (capability filtering as a multi-tenant requirement)
- External: `tokenfs-paper/docs/SIMILARITY_API_SURFACE.md` (the user-facing surface this enables)

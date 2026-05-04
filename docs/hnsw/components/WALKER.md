# Component: Walker (search)

**Status:** Phase 1 implementation landed 2026-05-04 (commits f985337..c1c4c98). Filter integration arrives in Phase 3; kernel-FPU bracketing in Phase 5.

**Lives in:** `crates/tokenfs-algos/src/similarity/hnsw/{walker.rs, visit.rs, candidates.rs}`

## Phase 1 implementation summary

The walker landed 2026-05-04 in `walker.rs` (~470 LOC). Implements
HNSW Algorithm 5 (K-NN-SEARCH) calling Algorithm 2 (SEARCH-LAYER)
via the existing `HnswView` zero-copy graph access plus three
freshly-landed support primitives:

- **`VisitedSet`** (`visit.rs`, ~100 LOC) — generation-counter
  bitset; O(1) `clear` between layer searches via incrementing a
  single `u32` counter.
- **`MaxHeap` + `Candidate`** (`candidates.rs`, ~200 LOC) — bounded
  sorted-vec heap with deterministic tie-break by `(distance, NodeId)`
  ascending. Used both as the working set (`ef`-sized) and the
  result set (`k`-sized).
- **`kernels::scalar::*`** (`kernels/scalar.rs`, ~250 LOC) — eight
  reference kernels covering `(L2² | Cosine | InnerProduct) × (F32 | I8 | U8)`
  plus `(Hamming | Jaccard) × B1x8`. Tanimoto on binary collapses to
  Jaccard.

`SearchCtx` keeps `search_layer`'s argument count under clippy's
`too-many-arguments` cap. Distance dispatch happens in
`compute_distance(query, target, metric, scalar)` — Phase 1 routes
all combos through scalar; SIMD dispatch lands Phase 2 via
`kernels::auto::*`.

Cross-arch reproducibility: integer / binary metrics produce
byte-identical distances across x86/AArch64; f32 metrics may differ
under FMA fusion (documented in `DETERMINISM.md` §9). Phase 5 adds
a `try_search_kernel_safe` variant that rejects f32 metrics at the
type level.

## Role

Search-only graph traversal over an `HnswView`. Returns top-k by distance, optionally filtered by a Roaring bitmap. `no_std + alloc`, kernel-reachable. Hot path; audit-R10 surface.

## Required research input

- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) §1 (Algorithm 2: SEARCH-LAYER, Algorithm 5: K-NN-SEARCH), §6 (implementation gotchas, including the published bug catalog), §7 (filter integration)
- [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md) §3 (search algorithm specifics)
- [`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md) (visit / candidate / distance primitives we reuse)

## Production bugs to defend against

The HNSW algorithm research catalogued bugs from production HNSW implementations:

- **CVE-2023-37365 (hnswlib).** Out-of-bounds read in graph traversal via crafted index file. Our walker's `HnswView::try_node` is bounds-checked at every access; fuzzer target in Phase 5 specifically attacks the parser + traversal with random bytes.
- **pgvector dead-tuple visibility.** pgvector's HNSW could return tuples that should be invisible to the current MVCC snapshot. We don't have MVCC, but the analogous risk for us is "search returns nodes whose extents have been garbage-collected." Sealed-image semantics rules this out at the format layer; document the assumption.
- **qdrant filter-recall drop.** Filtering at low selectivity dropped recall significantly because the filter was applied post-graph-traversal. Our in-search pruning + brute-force fallback at the selectivity cliff (Phase 3) is the canonical fix.
- **hnswlib `mark_deleted` consistency.** Deletion-tombstone bookkeeping had subtle bugs. We don't support delete in v1; document as "not supported" in the API surface.
- **`_MM_HINT_T0` prefetch pattern from hnswlib.** Worth adopting in our walker's inner loop — cite SIMD prior art for the implementation pattern.

## Sections to fill in

1. **Search algorithm.** Beam-search structure (per HNSW paper Algorithm 5 + Algorithm 2). Level descent rules. Entry-point handling. ef-search parameter mechanics.
2. **Visited tracking.** `VisitedSet` — bitset over node IDs with O(1) clear via generation counter. Reuses `bits::rank_select` / `bitmap` primitives where possible (per `PRIMITIVE_INVENTORY.md`).
3. **Candidate min-heap.** `CandidateHeap` for the result top-k. Tie-break by NodeId for determinism. Bounded allocation.
4. **Distance kernel dispatch.** Walker does not own kernels; calls into `kernels::auto::distance_*` (see `DISTANCE_KERNELS.md`).
5. **Kernel-FPU bracketing (Phase 5).** f32 paths use `kernel_fpu_guard!` once per query, not per candidate. Integer/binary metrics avoid FPU entirely.
6. **Filter integration (Phase 3).** In-search pruning vs post-filter. Brute-force fallback at low selectivity. See `FILTER.md`.
7. **Audit posture.** `try_search_inner` private; `try_search` kernel-safe entry; `search` userspace-gated panicking wrapper. `try_search_kernel_safe` rejects f32 metrics for callers explicitly in kernel context.

## API as shipped (Phase 1)

```rust
pub struct SearchConfig {
    pub k: usize,                  // top-K to return
    pub ef_search: usize,          // base-layer dynamic candidate list
                                   //   — internally bumped to max(k, ef_search)
}

impl SearchConfig {
    pub const fn new(k: usize, ef_search: usize) -> Self;
    /// k=16, ef_search=64 — suitable for 32-byte F22 / Hamming.
    pub const DEFAULT: SearchConfig;
}

pub enum HnswSearchError {
    InvalidK,
    QueryLengthMismatch { got: usize, expected: usize },
    UnsupportedMetricScalar { metric: MetricKind, scalar: ScalarKind },
    ViewCorruption(HnswViewError),
    KernelError(HnswKernelError),
}

/// Kernel-safe entry. Reachable in --no-default-features --features alloc.
pub fn try_search(
    view: &HnswView<'_>,
    query: &[u8],
    config: &SearchConfig,
) -> Result<Vec<(NodeKey, Distance)>, HnswSearchError>;
```

The `metric` and `scalar` selection comes from the view's
`HnswHeader` (whatever the index was built with). Phase 1 does not
need a `metric` knob in `SearchConfig` because the index already
encodes the answer — supplying a different metric at query time
would just mean "I built this index but want to misuse it."

## API additions across later phases

Phase 3 adds `HnswFilter` + `try_search_with_filter`. Phase 5 adds
`try_search_kernel_safe` (rejects f32 metrics) and the userspace
panicking `search` shim gated on `cfg(feature = "userspace")`.

## Cross-references

- Phases: [`../phases/PHASE_1.md`](../phases/PHASE_1.md) (skeleton + scalar), [`../phases/PHASE_3.md`](../phases/PHASE_3.md) (filter integration), [`../phases/PHASE_5.md`](../phases/PHASE_5.md) (kernel-FPU bracketing)
- Research: [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md), [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md)
- Related: [`DISTANCE_KERNELS.md`](DISTANCE_KERNELS.md), [`FILTER.md`](FILTER.md), [`GRAPH_LAYOUT.md`](GRAPH_LAYOUT.md)
- Audit: [`../../KERNEL_SAFETY.md`](../../KERNEL_SAFETY.md)

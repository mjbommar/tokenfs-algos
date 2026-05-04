# Component: Builder (insert)

**Status:** skeleton, 2026-05-03. Filled in during Phase 4.

**Lives in:** `crates/tokenfs-algos/src/similarity/hnsw/build/{mod.rs, insert.rs, level.rs, serialize.rs}` plus `graph.rs` and `select.rs` at the parent module level.

## Role

Construct an HNSW index from a sequence of `(NodeKey, vector)` insertions. Single-threaded for SLSA-L3 determinism. Output is byte-for-byte usearch v2.25 wire format. `std + alloc`-gated; userspace-only (no kernel-mode build path).

## Required research input

- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) §1 (Algorithms 1, 3, 4), §3 (level distribution), §4 (edge cases), §6 (implementation gotchas)
- [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md) §4 (insert algorithm specifics)
- [`../research/DETERMINISM.md`](../research/DETERMINISM.md) (the rules this component enforces)

## Determinism gaps in usearch we close

Per the determinism research deep dive:

- **No public seed API in usearch.** `level_generator` at `_references/usearch/include/usearch/index.hpp:2334` is default-constructed, no seed parameter exposed. Our `BuildConfig::seed` is a first-class field; deterministic by construction.
- **No tie-breaker on `candidate_t::operator<`.** `_references/usearch/include/usearch/index.hpp:2227` compares distance only. Our candidate-min-heap tie-breaks by `(distance, node_key, slot)` so identical-distance candidates have a stable, reproducible ordering.
- **`expansion_add` not in serialized header.** `index_serialized_header_t` at `index.hpp:1990` doesn't record this parameter. Our `BuildConfig` is recorded by the TokenFS image manifest out-of-band; reproducibility check is "build with same `BuildConfig` + same input → byte-identical bytes."
- **Use ChaCha8 from `image_salt`.** Per determinism research recommendation: `rand_chacha::ChaCha8Rng` seeded from the TokenFS `image_salt` field gives cross-arch byte-identical RNG output; deterministic across x86 / AArch64.
- **Cross-arch f32 caveat.** AVX2 vs AVX-512 vs NEON FMA fusion produces different f32 distance values; SLSA-L3 builds should constrain to integer metrics (Hamming on packed binary, L2² on i8) for cross-arch byte-identical output. Documented in DETERMINISM.md §9.

## Sections to fill in

1. **Algorithm 1 (INSERT) walkthrough.** Level assignment via seeded RNG; per-level search down to current entry point at `efConstruction` candidate set size; `select_neighbors_heuristic` at each level; bidirectional edge update with M-pruning.
2. **Algorithm 3 vs Algorithm 4 selection.** When `select_neighbors_simple` vs `select_neighbors_heuristic` is used. usearch's choices.
3. **Level assignment rule.** `l = floor(-ln(unif(0,1)) * mL)`, `mL = 1 / ln(M)`. Capped at `max_level`. Deterministic seeded RNG (no thread-local).
4. **Edge cases.** Empty graph (first insert becomes entry point). Single-node graph. Vector dim mismatch (returns Result, never panics).
5. **Determinism contract.** Single-thread by construction; sorted input is caller's responsibility; seeded RNG; tie-break by NodeKey ascending; integer metrics for cross-arch byte-identical output.
6. **Round-trip with libusearch v2.25.** Phase 4 round-trip test asserts byte-identical output given the same (input, seed, config) sequence.
7. **Audit posture.** Same try_*/_inner conventions; gated on `cfg(feature = "std")` (builder doesn't need to run in kernel mode). Panicking wrappers gated on `cfg(feature = "userspace")`.

## API skeleton

```rust
#[cfg(feature = "std")]
pub struct BuildConfig {
    pub dimensions: usize,
    pub scalar_kind: ScalarKind,
    pub metric: Metric,
    pub M: u32,                  // typical 16
    pub M_max: u32,              // typical == M
    pub M_max0: u32,             // typical == 2 * M
    pub ef_construction: u32,    // typical 64-256
    pub seed: u64,               // RNG seed for level assignment; default = 0xCAFE_BABE
    pub max_level: u8,           // hard cap to avoid pathological tail; default 16
}

#[cfg(feature = "std")]
pub enum HnswBuildError {
    InvalidConfig(/* details */),
    VectorDimMismatch { expected: usize, got: usize },
    NodeKeyDuplicate(NodeKey),
    GraphCorruption(/* internal invariant violation */),
}

#[cfg(feature = "std")]
pub struct Builder { /* opaque */ }

#[cfg(feature = "std")]
impl Builder {
    pub fn try_new(config: BuildConfig) -> Result<Self, HnswBuildError>;

    pub fn try_insert(
        &mut self,
        key: NodeKey,
        vector: &[u8],
    ) -> Result<(), HnswBuildError>;

    pub fn try_finish_to_bytes(self) -> Result<Vec<u8>, HnswBuildError>;

    pub fn node_count(&self) -> usize;
    pub fn current_max_level(&self) -> u8;
}

// Userspace ergonomic. Panicking wrappers per the shape-API convention.
#[cfg(feature = "userspace")]
impl Builder {
    pub fn new(config: BuildConfig) -> Self;
    pub fn insert(&mut self, key: NodeKey, vector: &[u8]);
    pub fn finish_to_bytes(self) -> Vec<u8>;
}
```

## Cross-references

- Phase: [`../phases/PHASE_4.md`](../phases/PHASE_4.md) (full implementation)
- Research: [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md), [`../research/DETERMINISM.md`](../research/DETERMINISM.md), [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md)
- Related: [`WIRE_FORMAT.md`](WIRE_FORMAT.md) (the format `serialize_to_bytes` writes), [`WALKER.md`](WALKER.md) (the search the builder calls during insert), [`DISTANCE_KERNELS.md`](DISTANCE_KERNELS.md) (the kernels the builder calls for distance computation), [`GRAPH_LAYOUT.md`](GRAPH_LAYOUT.md) (the in-memory representation the builder owns)
- Audit: [`../../KERNEL_SAFETY.md`](../../KERNEL_SAFETY.md)

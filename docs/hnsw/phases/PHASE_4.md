# Phase 4 — Native deterministic builder

**Status:** plan, 2026-05-03. **Week 5 of the v0.7.0 HNSW landing.**

**Goal:** end-of-week, our `Builder` produces a usearch v2.25 wire-format index that round-trips: build → serialize → walker → results match brute-force scan over the same vectors at >=95% recall. Single-threaded, deterministic, SLSA-L3-ready. Clustering-fuzz validates algorithmic correctness against constructed ground truth (per `../components/CLUSTERING_FUZZ.md`).

## Correctness gate (revised 2026-05-04)

Original plan called for "libusearch round-trip" (build with our builder, build with libusearch, assert byte-identical). Per the user-directed fixture-strategy redesign, the primary gate is now **brute-force + clustering-fuzz**, no external dependency. See [`../components/CLUSTERING_FUZZ.md`](../components/CLUSTERING_FUZZ.md) for the design.

The libusearch byte-identical check is preserved as an **optional** test gated on `--features hnsw-libusearch-compat`; runs only when the user has Python `usearch==2.25.x` available locally. Not in CI; manual sanity only.

## Deliverables

### Code

- `crates/tokenfs-algos/src/similarity/hnsw/graph.rs`:
  - `Graph` — owned in-memory representation
  - `Vec<Node>`, neighbor list per level per node, level assignments, entry-point pointer, max_level counter
  - `Graph::new(BuildConfig) -> Self`
  - `Graph::add_node(NodeKey, Vec<u8>, level) -> NodeId`
  - `Graph::neighbors_at(NodeId, level) -> &[NodeId]`
  - `Graph::add_edge(NodeId, NodeId, level)`, `prune_edges(NodeId, level, M_max)`

- `crates/tokenfs-algos/src/similarity/hnsw/select.rs`:
  - `select_neighbors_simple(candidates, M) -> Vec<NodeId>` — Algorithm 3 of HNSW paper
  - `select_neighbors_heuristic(candidates, M, lc, extend, keep_pruned) -> Vec<NodeId>` — Algorithm 4
  - Pure-functional given a candidate set; no Graph mutation
  - Tie-break by NodeKey ascending for determinism
  - Used by `build/insert.rs` after each per-level search returns candidates

- `crates/tokenfs-algos/src/similarity/hnsw/build/level.rs`:
  - `random_level(rng: &mut impl Rng, mL: f64, max_level: u8) -> u8`
  - Per HNSW paper §4: `l = floor(-ln(unif(0,1)) * mL)`, capped at max_level
  - Uses a deterministic seeded RNG from `rand_xoshiro` or `rand_chacha` (no thread-local, no OS random)

- `crates/tokenfs-algos/src/similarity/hnsw/build/insert.rs`:
  - `insert_one(graph: &mut Graph, key: NodeKey, vector: &[u8], rng: &mut Rng, config: &BuildConfig) -> Result<(), HnswBuildError>`
  - Implements Algorithm 1 of HNSW paper
  - Per-level: search down from current entry point at `efConstruction` candidate set size; run `select_neighbors_heuristic`; bidirectional edge update with M-pruning
  - Uses the same `walker::try_search_inner` for per-level search (search lives in walker; insert just orchestrates)

- `crates/tokenfs-algos/src/similarity/hnsw/build/serialize.rs`:
  - `serialize_to_bytes(&Graph, &BuildConfig) -> Result<Vec<u8>, HnswBuildError>`
  - Writes the usearch v2.25 wire format byte-for-byte
  - Header section (64 bytes); per-node tape (vector blob + per-level neighbor lists); endianness little-endian; alignment per spec
  - Round-trip test asserts: `Walker::view(serialized).search(q) == Walker::view(libusearch_serialized_same_input).search(q)`

- `crates/tokenfs-algos/src/similarity/hnsw/build/mod.rs`:
  - Public `Builder` struct
  - `Builder::try_new(config: BuildConfig) -> Result<Self, HnswBuildError>` — validates config (M >= 2, M_max0 >= M_max, etc.)
  - `Builder::try_insert(&mut self, key: NodeKey, vector: &[u8]) -> Result<(), HnswBuildError>` — vector length validated against config.dimensions
  - `Builder::try_finish_to_bytes(self) -> Result<Vec<u8>, HnswBuildError>`
  - `Builder::node_count() -> usize` — for progress reporting
  - All gated on `#[cfg(feature = "std")]` (no kernel-mode build path)
  - Userspace-gated panicking variants under `#[cfg(feature = "userspace")]` per established pattern

- `crates/tokenfs-algos/src/similarity/hnsw/mod.rs`:
  - Re-export `Builder`, `BuildConfig`, `HnswBuildError` under `cfg(feature = "std")`

### Tests

- `crates/tokenfs-algos/src/similarity/hnsw/build/tests.rs`:
  - `level_distribution_matches_paper()` — generate 10⁶ levels with seed=0; assert geometric distribution within tolerance
  - `level_distribution_deterministic_across_runs()` — same seed → identical level sequence
  - `select_neighbors_heuristic_matches_paper_example()` — small hand-crafted candidate set with known correct output
  - `select_neighbors_simple_matches_paper_example()`
  - `insert_one_into_empty_graph_succeeds()` — first node becomes entry point
  - `insert_one_into_single_node_graph_creates_edges()`
  - `insert_one_with_wrong_vector_length_returns_error()` — does NOT panic
  - `builder_finish_returns_valid_wire_format()` — output passes `HnswView::try_new` round-trip
  - `builder_determinism_same_seed_same_input_same_output()` — build twice, assert byte-identical

- `crates/tokenfs-algos/tests/hnsw_clustering.rs` (integration; primary correctness gate):
  - Generates a clustering corpus per `../components/CLUSTERING_FUZZ.md` at three operating points: (M=100, N=20, p=0.05), (M=50, N=10, p=0.10), (M=20, N=5, p=0.20)
  - Builds with our Builder; walks with our walker
  - Asserts average per-query recall >= the floor for each (p, T) pair documented in CLUSTERING_FUZZ.md §"Correctness assertions"
  - Brute-force overlap: walker results' set vs. brute-force top-N >= 0.95 at efSearch=64

- `crates/tokenfs-algos/tests/hnsw_round_trip.rs` (integration):
  - Build a 10⁴-vector clustering corpus
  - Builder serializes
  - Walker reads the bytes back, runs the same queries
  - Asserts walker results match brute-force scan at >=95% recall (no external libusearch oracle needed)

- `crates/tokenfs-algos/tests/hnsw_builder_determinism.rs`:
  - Build twice with the same input + same seed; assert byte-identical output
  - Build with seed=42 and seed=43; assert different output
  - Build with same seed but different sort order of input; assert different output (validates that caller-side sorting is the determinism contract)

- `crates/tokenfs-algos/tests/hnsw_libusearch_compat.rs` (optional, gated on `--features hnsw-libusearch-compat`):
  - Spawns Python subprocess with `usearch==2.25.x`
  - Asks libusearch to read our Builder's serialized bytes
  - Asserts libusearch's k-NN results agree with our walker
  - NOT in CI; manual sanity check only. Documented in `tests/README.md`.

### Benchmarks

- `crates/tokenfs-algos/benches/hnsw_build.rs` (new criterion bench):
  - Single-thread build throughput at N = 10³, 10⁴, 10⁵
  - For each (M, efConstruction) combination from {(8, 32), (16, 64), (32, 128)}
  - Compare against libusearch single-thread (informational; not a regression gate)

- `crates/tokenfs-algos/benches/iai_primitives.rs`:
  - `iai_hnsw_build_n1k_M16` — instruction-count for building a 1000-vector index. Stays in 1% IR regression gate.

### Docs

- Fill in `docs/hnsw/components/BUILDER.md` with the algorithm spec
- Fill in `docs/hnsw/components/GRAPH_LAYOUT.md` (the in-memory side; on-disk side already in `WIRE_FORMAT.md`)
- Update `docs/hnsw/research/DETERMINISM.md` (if research already landed) with concrete enforcement rules

### CI

- `cargo xtask check` — passes
- `cargo test -p tokenfs-algos --test hnsw_builder_round_trip --features arch-pinned-kernels` — passes
- `cargo test -p tokenfs-algos --test hnsw_builder_determinism --features arch-pinned-kernels` — passes (load-bearing for SLSA-L3)

## Acceptance criteria

```bash
$ cargo test -p tokenfs-algos --test hnsw_builder_round_trip --features arch-pinned-kernels
... build_matches_libusearch_byte_for_byte ... ok
... walker_parity_after_build ... ok

$ cargo test -p tokenfs-algos --test hnsw_builder_determinism --features arch-pinned-kernels
... same_seed_same_input_byte_identical ... ok
... different_seed_different_output ... ok

$ cargo bench -p tokenfs-algos --bench hnsw_build
... build N=10⁴ M=16 efC=64: ~2.3s (libusearch single-thread: ~1.8s)

$ cargo xtask check
xtask: panic-surface-lint: pub fn surface within allowlist (0 entries)
```

## Out of scope for Phase 4

- Multi-threaded builder (defer; SLSA-L3 doesn't need it; optional `parallel` feature can land later)
- Builder API for incremental update / delete (v1 is insert-only; sealed-image semantics)
- Kernel-FPU bracketing (Phase 5)
- tokenfs_writer integration (Phase 5)

## Risk + mitigations

| Risk | Mitigation |
|---|---|
| Builder produces graph topology that fails clustering recall on the corpus | The clustering-fuzz test (per `../components/CLUSTERING_FUZZ.md`) bisects: try (M=10, N=5, p=0.05) first; if recall is low at that scale, the bug is in select_neighbors / level assignment / edge pruning, not in serialization. Use the brute-force scan as ground truth — it cannot diverge from the inserted vectors. |
| Our builder produces different bytes than libusearch given identical insert order | This is the optional `hnsw-libusearch-compat` test (out of CI). Wire-format compat is the contract; bit-identical builds are nice-to-have. If the optional test fires, document the divergence and verify our walker still recovers the corpus (the load-bearing property). |
| Our builder is materially slower than a hypothetical libusearch reference | We don't have libusearch as a baseline at all. Target: within 2× of brute-force build for small N (where build dominates), faster than brute-force search for large N (where search dominates). iai-callgrind + criterion track absolute throughput; bench-history.yml plots regressions. |
| Float-point determinism breaks across CPU backends | Gate SLSA-L3 builds to integer metrics (Hamming on packed binary, L2² on i8). f32 metrics work but cross-arch reproducibility is documented as not guaranteed. See `research/DETERMINISM.md` for the recommendation. |
| Builder panics on edge cases (empty graph, vector dim mismatch) | All public entries are `try_*` returning `Result<_, HnswBuildError>`. Panic-surface lint catches any regression. |
| Builder's RNG choice affects portability | Use `rand_chacha::ChaCha8Rng` (deterministic across architectures, well-defined output). Document the choice in `research/DETERMINISM.md`. |

## Cross-references

- [`PHASE_3.md`](PHASE_3.md) — must complete before this phase starts
- [`../components/BUILDER.md`](../components/BUILDER.md) — fills out by end of phase
- [`../components/GRAPH_LAYOUT.md`](../components/GRAPH_LAYOUT.md) — adds owned-graph spec
- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) — Algorithm 1 / 3 / 4 references
- [`../research/DETERMINISM.md`](../research/DETERMINISM.md) — required reading; contains the rules this phase enforces

# Phase 4 — Native deterministic builder

**Status:** plan, 2026-05-03. **Week 5 of the v0.7.0 HNSW landing.**

**Goal:** end-of-week, our `Builder` produces a usearch v2.25 wire-format index that the libusearch reader consumes correctly; round-trip test confirms our builder + our walker = libusearch builder + libusearch walker on the same insertion sequence + RNG seed. Single-threaded, deterministic, SLSA-L3-ready.

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

- `crates/tokenfs-algos/tests/hnsw_builder_round_trip.rs` (integration):
  - Build a 10⁴-vector index with our Builder + seed=42 in single-threaded mode
  - Build the same input with libusearch v2.25 + seed=42 in single-threaded mode
  - Compare the two serialized bytes byte-for-byte; if not byte-identical, document the divergence and verify recall equivalence on 100 queries
  - Walker parity: load both indexes; same 100 queries; assert k-NN result lists match

- `crates/tokenfs-algos/tests/hnsw_builder_determinism.rs`:
  - Build twice with the same input; assert byte-identical output
  - Build with seed=42 and seed=43; assert different output
  - Build with same seed but different sort order of input; assert different output (validates that caller-side sorting is the determinism contract)

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
| Our builder produces different graph topology than libusearch given identical insert order | The HNSW paper is precise enough that a correct implementation should match libusearch on deterministic input. If we diverge, the round-trip test catches it; we then bisect by checking each algorithm's output against libusearch's intermediate state. |
| Our builder is materially slower than libusearch single-thread | Target: within 2× of libusearch single-thread. We use the same SIMD distance kernels; the algorithmic difference should be small. If slower, profile + tune. |
| Float-point determinism breaks across CPU backends | Gate SLSA-L3 builds to integer metrics (Hamming on packed binary, L2² on i8). f32 metrics work but cross-arch reproducibility is documented as not guaranteed. See `research/DETERMINISM.md` for the recommendation. |
| Builder panics on edge cases (empty graph, vector dim mismatch) | All public entries are `try_*` returning `Result<_, HnswBuildError>`. Panic-surface lint catches any regression. |
| Builder's RNG choice affects portability | Use `rand_chacha::ChaCha8Rng` (deterministic across architectures, well-defined output). Document the choice in `research/DETERMINISM.md`. |

## Cross-references

- [`PHASE_3.md`](PHASE_3.md) — must complete before this phase starts
- [`../components/BUILDER.md`](../components/BUILDER.md) — fills out by end of phase
- [`../components/GRAPH_LAYOUT.md`](../components/GRAPH_LAYOUT.md) — adds owned-graph spec
- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) — Algorithm 1 / 3 / 4 references
- [`../research/DETERMINISM.md`](../research/DETERMINISM.md) — required reading; contains the rules this phase enforces

# Phase 1 — Wire format parser + scalar walker skeleton

**Status:** plan, 2026-05-03 (fixture strategy revised 2026-05-04). **Week 1 of the v0.7.0 HNSW landing.**

**Goal:** end-of-week, `try_search` returns top-16 by Hamming distance over a hand-crafted toy index (4 nodes, 2 levels, 32-byte vectors), with byte-level header + view arithmetic validated and scalar distance kernels matching hand-computed reference values. Scalar-only — no SIMD yet. This phase locks the wire-format contract end-to-end before SIMD complexity enters the picture.

## Fixture strategy (revised 2026-05-04)

**No external oracle in Phase 1.** Survey of `_references/usearch/cpp/test.cpp`, `_references/usearch/c/test.c`, and `_references/usearch/python/scripts/test_*.py` confirmed that usearch ships no committed binary fixtures — every test builds an index at runtime, saves to `tmp.usearch`, reloads, and re-queries. We have no analogous "Apache-2.0-licensed canonical bytes" to copy.

Instead:

- **Phase 1 oracle: hand-crafted toy fixture in source.** A tiny 4-node, 2-level usearch v2.25 byte stream as `const TOY_INDEX_V2_25: &[u8] = &[...]` in `tests.rs`, constructed by hand from `USEARCH_DEEP_DIVE.md` §1's byte layout. Validates header parser + view byte arithmetic + scalar distance kernels. ~100 LOC.
- **Phase 4 oracle: round-trip + clustering-fuzz.** Once the Builder lands, the primary correctness gate becomes: build → serialize → walk → compare to a brute-force scan over the same vectors. Brute-force IS the oracle. The clustering-fuzz generator (per user's design — see [`../components/CLUSTERING_FUZZ.md`](../components/CLUSTERING_FUZZ.md)) generates M seed vectors with N bit-flipped variants each at probability p%, giving constructed ground truth: every variant of seed_i must be in the top-N nearest neighbors of seed_i. Catches algorithmic recall regressions a single golden fixture can't.
- **Optional libusearch wire-format compat test.** A `tests/hnsw_libusearch_compat.rs` integration test, gated on `--features hnsw-libusearch-compat`, runs only when the user has Python `usearch==2.25.x` available locally. Not in CI; rare manual sanity check. Out of scope for Phase 1.

## Deliverables

### Code

- `crates/tokenfs-algos/src/similarity/hnsw/mod.rs` — module skeleton, public API re-exports.
- `crates/tokenfs-algos/src/similarity/hnsw/header.rs`:
  - `HnswHeader` opaque type
  - `HnswHeaderError` enum
  - `try_parse_header(&[u8]) -> Result<HnswHeader, HnswHeaderError>` — validates magic, version range (v2.25.x), scalar_kind enum, metric_kind enum, dimensions, count_present
  - `header_bytes_len() -> usize` — constant 64 bytes
- `crates/tokenfs-algos/src/similarity/hnsw/view.rs`:
  - `HnswView<'a>` zero-copy view over `&'a [u8]`
  - `HnswView::try_new(bytes) -> Result<Self, ...>` — parses header + sets up node-table offsets
  - `HnswView::node_count()`, `dimensions()`, `entry_point()`, `max_level()`
  - `HnswView::try_node(NodeId) -> Result<NodeRef<'a>, ...>` — bounds-checked node lookup
  - `NodeRef::try_neighbors(level) -> Result<&[NodeId], ...>` — bounds-checked neighbor list
  - `NodeRef::vector_bytes() -> &[u8]` — raw vector blob slice
- `crates/tokenfs-algos/src/similarity/hnsw/visit.rs`:
  - `VisitedSet` — bitset over node IDs; alloc-backed Vec<u64>
  - `VisitedSet::try_with_capacity(n) -> Result<Self, ...>`
  - `VisitedSet::try_insert(NodeId) -> bool` (returns true if newly inserted)
  - `VisitedSet::clear()` — O(1) generation-counter trick to avoid re-allocating per query
- `crates/tokenfs-algos/src/similarity/hnsw/candidates.rs`:
  - `CandidateHeap` — bounded min-heap of (distance, NodeId), ordered for k-NN
  - `try_with_capacity(k)`, `try_push`, `pop_top`, `peek_top`
  - Tie-break by NodeId ascending for determinism
- `crates/tokenfs-algos/src/similarity/hnsw/walker.rs`:
  - `SearchConfig { k, ef_search, metric, scalar_kind }` (no f32 cross-arch concern this phase since we're scalar)
  - `try_search_inner(&HnswView, &[u8], &SearchConfig, &mut Vec<...>) -> Result<(), HnswSearchError>` — pre-validated body
  - `try_search(&HnswView, &[u8], &SearchConfig) -> Result<Vec<(NodeKey, Distance)>, HnswSearchError>` — kernel-safe entry, validates query length + k > 0
  - `#[cfg(feature = "userspace")] fn search(...)` — panicking wrapper that calls `try_search(...).expect(...)`
- `crates/tokenfs-algos/src/similarity/hnsw/kernels/mod.rs`:
  - Distance metric / scalar kind enums
  - `auto::distance(...)` runtime dispatcher (this phase: only scalar; SIMD lands Phase 2)
- `crates/tokenfs-algos/src/similarity/hnsw/kernels/scalar.rs`:
  - 8 scalar reference kernels (no SIMD): L2² / cosine / dot / Hamming for `f32 / i8 / u8 / binary` (8 combos)
  - Each one is `fn distance_<metric>_<scalar>(a: &[u8], b: &[u8]) -> Distance`
  - These ARE the parity oracle for every SIMD backend in Phase 2

### Tests

- `crates/tokenfs-algos/src/similarity/hnsw/tests.rs`:
  - `const TOY_INDEX_V2_25: &[u8] = &[...]` — hand-crafted 4-node, 2-level usearch v2.25 byte stream
  - `header_parses_canonical_v2_25()` — parses TOY_INDEX_V2_25's header; asserts every field
  - `header_rejects_wrong_magic()`
  - `header_rejects_unknown_scalar_kind()`
  - `header_rejects_unsupported_format_version()` — fail closed on v2.20 / v3.0 stub
  - `view_node_lookup_in_bounds()` — walks the 4 toy nodes
  - `view_node_lookup_out_of_bounds_returns_error()` — does NOT panic
  - `view_neighbor_iteration_per_level()`
  - `visited_set_insert_idempotent()`
  - `visited_set_clear_via_generation_counter()`
  - `candidates_top_k_with_ties_breaks_by_node_id()`
  - Scalar distance round-trip: every metric on small f32 / i8 / u8 / binary inputs against hand-computed reference values
  - `walker_returns_top_2_on_toy_index()` — runs try_search over TOY_INDEX_V2_25 for a hand-picked query, asserts the two known-correct nearest neighbors come back in the right order

**Deferred to Phase 4** (was here, moved):

- 10⁴-vector parity fixture with libusearch oracle → replaced by **clustering-fuzz + round-trip tests** in Phase 4, no external oracle dependency. See [`../components/CLUSTERING_FUZZ.md`](../components/CLUSTERING_FUZZ.md).

### Docs

- Fill in `docs/hnsw/components/WIRE_FORMAT.md` with the byte-level layout (using research from `research/USEARCH_DEEP_DIVE.md`)
- Fill in `docs/hnsw/components/WALKER.md` with the search algorithm spec (using `research/HNSW_ALGORITHM_NOTES.md`)
- Fill in `docs/hnsw/components/GRAPH_LAYOUT.md` with the in-memory + on-disk layouts

### CI

- `cargo xtask check` — passes (fmt, clippy, doc, no-std, security, panic-surface lint at 0 entries)
- `cargo test -p tokenfs-algos --lib similarity::hnsw::tests` — passes (toy-fixture unit tests)
- `cargo build -p tokenfs-algos --no-default-features --features alloc --lib` — passes (kernel-safe surface compiles)

## Acceptance criteria

A demo at end-of-week shows:

```bash
$ cargo test -p tokenfs-algos --lib similarity::hnsw
running 11 tests
test similarity::hnsw::tests::header_parses_canonical_v2_25 ... ok
test similarity::hnsw::tests::header_rejects_wrong_magic ... ok
test similarity::hnsw::tests::header_rejects_unknown_scalar_kind ... ok
test similarity::hnsw::tests::header_rejects_unsupported_format_version ... ok
test similarity::hnsw::tests::view_node_lookup_in_bounds ... ok
test similarity::hnsw::tests::view_node_lookup_out_of_bounds_returns_error ... ok
test similarity::hnsw::tests::view_neighbor_iteration_per_level ... ok
test similarity::hnsw::tests::visited_set_insert_idempotent ... ok
test similarity::hnsw::tests::candidates_top_k_with_ties_breaks_by_node_id ... ok
test similarity::hnsw::tests::scalar_distance_kernels_against_hand_computed ... ok
test similarity::hnsw::tests::walker_returns_top_2_on_toy_index ... ok

test result: ok. 11 passed; 0 failed
```

And:

```bash
$ cargo xtask check
xtask: panic-surface-lint: pub fn surface within allowlist (0 entries snapshotted)
```

## Out of scope for Phase 1

- Any SIMD kernel (Phase 2)
- AVX-512 (Phase 3)
- Filter primitives (Phase 3)
- The `Builder` (Phase 4)
- Kernel-FPU bracketing (Phase 5)
- iai-callgrind benches (Phase 2 — when the SIMD kernels exist to bench against scalar)

## Risk + mitigations

| Risk | Mitigation |
|---|---|
| Wire-format misparse → silent wrong results | Phase 1's hand-crafted toy fixture validates byte-format correctness; Phase 4's clustering-fuzz validates algorithmic correctness against constructed ground truth (see `../components/CLUSTERING_FUZZ.md`). Both must pass before declaring HNSW correct. |
| **CVE-2023-37365-class bounds-read in graph traversal** | Every `HnswView::try_*` accessor is bounds-checked at the byte level. Phase 5 adds a `cargo-fuzz` target (`fuzz_targets/hnsw_walker_fuzz.rs`) feeding random bytes through the parser + walker; must never panic and must reject malformed input cleanly. |
| Toy fixture only validates happy-path byte format | Acceptable for Phase 1 — the load-bearing correctness gate is Phase 4's clustering-fuzz with constructed ground truth. Phase 1 just proves "we can read what usearch v2.25 writes" via the byte-by-byte hand-crafted index. |
| Hand-crafted fixture bytes drift from real libusearch output over time | One-time risk: fixture is constructed from `USEARCH_DEEP_DIVE.md` §1's documented byte layout. If usearch v2.25.x patch releases change anything, a future patch updates the toy fixture's bytes. Never load-bearing for production correctness. |
| HNSW algorithm subtleties (entry-point-empty case, etc.) | Phase 1 walker handles only the non-empty case (the toy fixture has 4 nodes); empty/single-node edge cases land in Phase 4 alongside the Builder, where they actually matter. |

## Cross-references

- [`../00_ARCHITECTURE.md`](../00_ARCHITECTURE.md) — the layer map this phase populates
- [`../components/WIRE_FORMAT.md`](../components/WIRE_FORMAT.md) — fills out by end of phase
- [`../components/WALKER.md`](../components/WALKER.md) — fills out by end of phase
- [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md) — required input
- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) — required input
- [`../../KERNEL_SAFETY.md`](../../KERNEL_SAFETY.md) — `try_*` / `_unchecked` / `_inner` discipline

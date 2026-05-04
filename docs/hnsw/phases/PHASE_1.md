# Phase 1 — Wire format parser + scalar walker skeleton

**Status:** plan, 2026-05-03. **Week 1 of the v0.7.0 HNSW landing.**

**Goal:** end-of-week, `try_search` returns top-16 by Hamming distance over a 10⁴-node F22 fingerprint reference index, matching libusearch's k-NN result list bit-for-bit. Scalar-only — no SIMD yet. This phase locks the wire-format contract end-to-end before SIMD complexity enters the picture.

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
  - `header_parses_canonical_v2_25()` — feeds a known-good usearch v2.25 header, asserts every field
  - `header_rejects_wrong_magic()`
  - `header_rejects_unknown_scalar_kind()`
  - `view_node_lookup_in_bounds()`
  - `view_node_lookup_out_of_bounds_returns_error()` — does NOT panic
  - `visited_set_insert_idempotent()`
  - `candidates_top_k_with_ties_breaks_by_node_id()`
  - Scalar distance round-trip: every metric on small f32 / i8 / u8 / binary inputs against hand-computed reference values
- `crates/tokenfs-algos/tests/data/hnsw/f22_10k_v2_25.bin` (committed test fixture):
  - 10⁴ random F22 fingerprints (32 bytes each, deterministic seed)
  - Built with libusearch v2.25 in single-threaded mode
  - ~1-2 MB on disk
  - `tests/data/hnsw/README.md` documents how to regenerate from a known seed
- `crates/tokenfs-algos/tests/hnsw_walker_parity.rs` (integration test):
  - Loads the fixture
  - Runs `try_search` for 100 known queries
  - Asserts `result.iter().map(|(k,_)| k).eq(libusearch_reference_results)` for each query
  - This is the load-bearing parity test for the entire HNSW landing; gates every later phase

### Docs

- Fill in `docs/hnsw/components/WIRE_FORMAT.md` with the byte-level layout (using research from `research/USEARCH_DEEP_DIVE.md`)
- Fill in `docs/hnsw/components/WALKER.md` with the search algorithm spec (using `research/HNSW_ALGORITHM_NOTES.md`)
- Fill in `docs/hnsw/components/GRAPH_LAYOUT.md` with the in-memory + on-disk layouts

### CI

- `cargo xtask check` — passes (fmt, clippy, doc, no-std, security, panic-surface lint at 0 entries)
- `cargo test -p tokenfs-algos --test hnsw_walker_parity` — passes (the parity test against libusearch fixture)
- `cargo build -p tokenfs-algos --no-default-features --features alloc --lib` — passes (kernel-safe surface compiles)

## Acceptance criteria

A demo at end-of-week shows:

```bash
$ cargo test -p tokenfs-algos --test hnsw_walker_parity
running 100 tests
test parity_query_0 ... ok
test parity_query_1 ... ok
...
test parity_query_99 ... ok

test result: ok. 100 passed; 0 failed
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
| Wire-format misparse → silent wrong results | Phase 1's parity test against the libusearch fixture is the gate. Until that passes 100% on 100 queries, do not move to Phase 2. |
| **CVE-2023-37365-class bounds-read in graph traversal** | Every `HnswView::try_*` accessor is bounds-checked at the byte level. Phase 5 adds a `cargo-fuzz` target (`fuzz_targets/hnsw_walker_fuzz.rs`) feeding random bytes through the parser + walker; must never panic and must reject malformed input cleanly. |
| Test fixture commit size exceeds reasonable limit | 10⁴ vectors at ~250 B/node ≈ 2.5 MB. If too large, drop to 10³ (~250 KB). The test asserts correctness, not scale. |
| Dependency on libusearch to *generate* the fixture | One-shot dependency for fixture generation only. Document the libusearch version + command in `tests/data/hnsw/README.md`. |
| HNSW algorithm subtleties (entry-point-empty case, etc.) | Phase 1 walker handles only the non-empty case (the fixture has 10⁴ nodes); empty/single-node edge cases land in Phase 4 alongside the Builder, where they actually matter. |

## Cross-references

- [`../00_ARCHITECTURE.md`](../00_ARCHITECTURE.md) — the layer map this phase populates
- [`../components/WIRE_FORMAT.md`](../components/WIRE_FORMAT.md) — fills out by end of phase
- [`../components/WALKER.md`](../components/WALKER.md) — fills out by end of phase
- [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md) — required input
- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) — required input
- [`../../KERNEL_SAFETY.md`](../../KERNEL_SAFETY.md) — `try_*` / `_unchecked` / `_inner` discipline

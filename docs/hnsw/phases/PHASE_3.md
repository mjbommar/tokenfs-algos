# Phase 3 — Filter primitives + AVX-512

**Status:** plan, 2026-05-03. **Week 4 of the v0.7.0 HNSW landing.**

**Goal:** end-of-week, capability-aware HNSW search returns only files in the user's permitted cluster, with the sub-linear search win preserved (vs. post-filter dropping it). AVX-512 distance kernels exist for the hot metrics on **stable Rust** (per Rust 1.89, August 2025 — see scope update below), gated on `feature = "avx512"` and `is_x86_feature_detected!("avx512f")`.

## Scope updates from research

**1. AVX-512 no longer needs nightly.** The SIMD prior-art research ([`../research/SIMD_PRIOR_ART.md`](../research/SIMD_PRIOR_ART.md)) confirmed that AVX-512 intrinsics stabilized in Rust 1.89 (released 2025-08-07). The original phase plan (and `HNSW_PATH_DECISION.md` §6 / §10) called for nightly + `feature = "avx512"`. We can drop the nightly requirement: AVX-512 ships on stable. The compile-time `feature = "avx512"` flag stays (so consumers without AVX-512 hardware aren't paying the binary-size cost), but `nightly` does not.

**2. `bitmap::Container` directly satisfies the FILTER component.** The primitive inventory ([`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md)) confirmed that `crates/tokenfs-algos/src/bitmap/` already provides a Roaring-style `Container` with SIMD intersect/union/cardinality. `HnswFilter` is a thin newtype wrapper around `&Container` (or `&RoaringBitmap` once the bitmap crate name is settled); no new SIMD work in Phase 3 for the filter primitives proper.

**3. Filter design references ACORN + SeRF.** Per `HNSW_ALGORITHM_NOTES.md` §7, ACORN (single-stage filter) and SeRF (predicate-on-edges) are the canonical published filter-integration strategies. We follow the ACORN shape: predicate-checked candidates are still expanded as graph hops but not added to the result heap. SeRF's edge-time annotation is out of scope for v1 (we don't rebuild the index per filter).

## Deliverables

### Code

- `crates/tokenfs-algos/src/similarity/hnsw/filter.rs`:
  - `HnswFilter<'a>` — wrapper around `Option<&'a RoaringBitmap>` of permitted node IDs
  - `HnswFilter::permits(NodeId) -> bool` — O(1) membership test using existing `bitmap::*` primitives
  - `HnswFilter::estimated_selectivity()` — used by walker to decide whether to fall back to brute-force (when filter selectivity is too low for HNSW to be worthwhile)
  - Composes with `crates/tokenfs-algos/src/bitmap/` SIMD intersect kernels for batch filter checks

- `crates/tokenfs-algos/src/similarity/hnsw/walker.rs` extension:
  - `try_search_with_filter(&HnswView, &[u8], &SearchConfig, &HnswFilter) -> Result<Vec<(NodeKey, Distance)>, HnswSearchError>`
  - In-search pruning: candidates that fail the filter are NOT added to the result heap, but ARE explored as graph hops (so the search doesn't get trapped in a permitted-but-unreachable subgraph)
  - Brute-force fallback when `filter.estimated_selectivity() < threshold` — at very low selectivity, HNSW's sub-linear advantage is gone and a linear scan over the filter set is faster
  - Same `try_search_with_filter_inner` / `try_*` / `_unchecked` / `_inner` shape

- `crates/tokenfs-algos/src/similarity/hnsw/kernels/avx512.rs`:
  - Gated on `#[cfg(all(feature = "avx512", target_arch = "x86_64"))]` (stable Rust 1.89+)
  - `unsafe fn distance_l2_squared_f32_unchecked(...)` — uses `_mm512_fmadd_ps`
  - `unsafe fn distance_dot_f32_unchecked(...)` — VFMADD231PS
  - `unsafe fn distance_l2_squared_i8_unchecked(...)` — VPDPBUSD if VNNI, else PMADDUBSW + PMADDWD pattern
  - `unsafe fn distance_hamming_binary_unchecked(...)` — VPOPCNTQ if BITALG, else PSHUFB-popcount lifted to 512-bit lanes
  - All `_unchecked` siblings as established
  - All under `#[target_feature(enable = "avx512f,avx512vl")]` + relevant sub-features

- `crates/tokenfs-algos/src/similarity/hnsw/kernels/mod.rs`:
  - Extend `auto::distance(...)` to prefer AVX-512 when `is_x86_feature_detected!("avx512f")` AND the avx512 feature is compiled in
  - The `dispatch::planner` HNSW rules from Phase 2 add an AVX-512 priority row

### Tests

- `crates/tokenfs-algos/src/similarity/hnsw/tests.rs` extensions:
  - `filter_permits_in_search()` — verify in-search pruning returns only permitted nodes
  - `filter_keeps_subgraph_reachable()` — verify denied-node graph hops still happen (so we don't get stranded)
  - `filter_brute_force_fallback_at_low_selectivity()` — when filter permits <1% of nodes, the walker switches to brute force
  - `filter_zero_permitted_returns_empty_not_error()` — edge case
  - `filter_all_permitted_matches_unfiltered()` — when filter permits all nodes, results are identical to `try_search` without filter

- `crates/tokenfs-algos/tests/hnsw_walker_parity.rs`:
  - New section: filter parity. Run libusearch with predicate-based filter; run our walker with Roaring filter; assert k-NN result lists match.

- `crates/tokenfs-algos/tests/avx512_parity.rs` (new file or extension):
  - For every AVX-512 kernel, parity against scalar oracle on 1000 random pairs
  - Gated on `is_x86_feature_detected!("avx512f")`; skipped on hosts without AVX-512

- `crates/tokenfs-algos/.github/workflows/ci-avx512.yml`:
  - Already exists; runs the AVX-512 parity tests on a self-hosted AVX-512 runner
  - Phase 3 just adds the new HNSW kernel rows to the existing matrix

### Benchmarks

- `crates/tokenfs-algos/benches/iai_primitives.rs`:
  - `iai_hnsw_distance_l2_f32_avx512`, `iai_hnsw_distance_hamming_binary_avx512` (gated)
  - `iai_hnsw_filter_check_n10k` — measures filter check cost at scale

- `crates/tokenfs-algos/benches/hnsw_filter.rs` (new criterion bench):
  - Search throughput at filter selectivity 1% / 10% / 50% / 99%
  - Compares: (a) in-search pruning, (b) post-filter, (c) brute-force fallback
  - Demonstrates the cliff where each strategy wins

### CI

- `cargo xtask check` — passes
- AVX-512 job (`ci-avx512.yml`) — picks up new HNSW kernel rows automatically (now runs on stable Rust 1.89+)
- iai-callgrind regression gate covers new AVX-512 + filter rows

## Acceptance criteria

```bash
$ cargo test -p tokenfs-algos --test hnsw_walker_parity --features arch-pinned-kernels
... filter_permits_in_search ... ok
... filter_brute_force_fallback_at_low_selectivity ... ok

$ cargo test -p tokenfs-algos --features avx512,arch-pinned-kernels --test avx512_parity
... distance_l2_squared_f32_avx512 ... ok
... distance_hamming_binary_avx512 ... ok

$ cargo bench -p tokenfs-algos --bench hnsw_filter
... filter selectivity 50%: in-search 8x faster than post-filter
... filter selectivity 1%: brute-force 3x faster than in-search

$ cargo xtask check
xtask: panic-surface-lint: pub fn surface within allowlist (0 entries)
```

## Hardware backend matrix at end of Phase 3

| Metric | scalar | AVX2 | AVX-512 (nightly) | NEON | SSE4.1 | SSSE3 |
|---|---|---|---|---|---|---|
| L2² (f32) | ✅ | ✅ FMA | ✅ FMA + 512-bit lanes | ✅ FMLA | — | — |
| L2² (i8 / u8) | ✅ | ✅ | ✅ VPDPBUSD if VNNI | ✅ | ✅ | — |
| cosine (f32) | ✅ | ✅ | ✅ | ✅ | — | — |
| cosine (i8 / u8) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| dot (f32) | ✅ | ✅ FMA | ✅ VFMADD231PS | ✅ FMLA | — | — |
| Hamming (binary) | ✅ | ✅ | ✅ VPOPCNTQ if BITALG | ✅ VCNT | ✅ | ✅ |
| Jaccard (binary) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tanimoto (binary) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

## Out of scope for Phase 3

- The Builder (Phase 4)
- Kernel-FPU bracketing for f32 (Phase 5)
- SVE2 (deferred — no consumer yet)
- GPU paths (separate `tokenfs-gpu` crate)

## Risk + mitigations

| Risk | Mitigation |
|---|---|
| AVX-512 stable in Rust 1.89 — older Rust versions need updating | Workspace `rust-toolchain.toml` pins to >=1.89. CI matrix already runs >=1.89. Local devs on older toolchains see a clear compile error from the gating attribute. |
| In-search filtering breaks recall vs. post-filter | The phase test asserts in-search filter result == libusearch predicate-filter result. If recall diverges, document the tradeoff. The HNSW paper isn't prescriptive about filter integration; production implementations vary. |
| Filter brute-force threshold tuning | Default to a conservative threshold (e.g. <5% selectivity). Add a `SearchConfig::brute_force_threshold` knob for callers who want to override. Document the empirical curve in the bench output. |
| AVX-512 downclock biases benches | Already documented in `PROFILING.md` profiling caveats. iai-callgrind is deterministic so it's not affected; criterion benches need long warmup. |

## Cross-references

- [`PHASE_2.md`](PHASE_2.md) — must complete before this phase starts
- [`../components/FILTER.md`](../components/FILTER.md) — fills out by end of phase
- [`../components/WALKER.md`](../components/WALKER.md) — adds filter integration
- [`../components/DISTANCE_KERNELS.md`](../components/DISTANCE_KERNELS.md) — adds AVX-512 row
- [`../../AVX512_HARDWARE.md`](../../AVX512_HARDWARE.md) — nightly + feature-flag conventions
- [`../research/HNSW_ALGORITHM_NOTES.md`](../research/HNSW_ALGORITHM_NOTES.md) — filter integration section

# HNSW implementation planning

**Status:** planning tree, 2026-05-03. This directory holds the working
documents for the v0.7.0 native HNSW implementation under
`tokenfs-algos::similarity::hnsw`.

The top-level decision (commit to native walker + builder, usearch v2.25
wire format, no libusearch dependency) is recorded in
[`../HNSW_PATH_DECISION.md`](../HNSW_PATH_DECISION.md). This directory
breaks that decision down into phase plans, component specs, and the
research that backs each.

## Reading order

1. **[`SELF_PROMPT.md`](SELF_PROMPT.md)** — kickoff prompt for picking up work in a fresh session. Has the inviolable contracts, the work loop, the "don't do this" list. Read this first if you're starting a work session.
2. **[`00_ARCHITECTURE.md`](00_ARCHITECTURE.md)** — top-level architecture: the layer table, module structure, posture matrix.
3. **[`phases/`](phases/)** — week-by-week phase plans with deliverables, demos, and acceptance criteria.
4. **[`components/`](components/)** — per-component specs (wire format, walker, builder, distance kernels, filter, graph layout). Filled in as research lands.
5. **[`research/`](research/)** — read-only background: usearch source deep dive, our existing primitive inventory, HNSW algorithm notes, SIMD prior art, determinism constraints. Backs the component specs.

## Phase index

| Phase | Week | Theme | Doc |
|---|---|---|---|
| 1 | 1 | Wire format parser + scalar walker skeleton | [`phases/PHASE_1.md`](phases/PHASE_1.md) |
| 2 | 2-3 | AVX2 + NEON + SSE4.1 distance kernels + iai benches | [`phases/PHASE_2.md`](phases/PHASE_2.md) |
| 3 | 4 | Filter primitives + AVX-512 | [`phases/PHASE_3.md`](phases/PHASE_3.md) |
| 4 | 5 | Native deterministic builder | [`phases/PHASE_4.md`](phases/PHASE_4.md) |
| 5 | 6-7 | Kernel-FPU bracketing + tokenfs_writer integration | [`phases/PHASE_5.md`](phases/PHASE_5.md) |

## Component index

| Component | Lives in | Doc |
|---|---|---|
| Wire format (read + write) | `header.rs` + `view.rs` + `build/serialize.rs` | [`components/WIRE_FORMAT.md`](components/WIRE_FORMAT.md) |
| Walker (search) | `walker.rs` (no_std + alloc) | [`components/WALKER.md`](components/WALKER.md) |
| Builder (insert) | `build/` (std-gated) | [`components/BUILDER.md`](components/BUILDER.md) |
| Distance kernels | `kernels/{scalar,avx2,avx512,neon,sse41,ssse3}.rs` | [`components/DISTANCE_KERNELS.md`](components/DISTANCE_KERNELS.md) |
| Filter primitives | `filter.rs` | [`components/FILTER.md`](components/FILTER.md) |
| Graph layout | `graph.rs` (owned) + `view.rs` (zero-copy) | [`components/GRAPH_LAYOUT.md`](components/GRAPH_LAYOUT.md) |

## Research index (all complete)

| Topic | Method | Doc | Lines |
|---|---|---|---|
| usearch v2.25 source code deep dive | Read `_references/usearch/` | [`research/USEARCH_DEEP_DIVE.md`](research/USEARCH_DEEP_DIVE.md) | 1084 |
| Our existing primitives inventory | Read `crates/tokenfs-algos/src/` | [`research/PRIMITIVE_INVENTORY.md`](research/PRIMITIVE_INVENTORY.md) | 696 |
| HNSW algorithm + variants + known issues | Web search + paper | [`research/HNSW_ALGORITHM_NOTES.md`](research/HNSW_ALGORITHM_NOTES.md) | 869 |
| SIMD distance kernel prior art | Web search (NumKong, faiss, hnswlib) | [`research/SIMD_PRIOR_ART.md`](research/SIMD_PRIOR_ART.md) | 796 |
| Determinism + SLSA-L3 reproducibility | Web search + spec analysis | [`research/DETERMINISM.md`](research/DETERMINISM.md) | 541 |

### Headline findings folded into phase + component plans

- **f32 + binary-popcount kernels reusable verbatim** from existing `vector::*` and `bits::popcount`. Phase 2's actual new work narrows to i8/u8 + SSE4.1/SSSE3 fallbacks. (PRIMITIVE_INVENTORY)
- **`bitmap::Container` directly satisfies the FILTER component** — no new SIMD work for filter primitives. (PRIMITIVE_INVENTORY)
- **AVX-512 stabilized in Rust 1.89 (2025-08-07)** — Phase 3 doesn't need nightly. (SIMD_PRIOR_ART)
- **SimSIMD renamed to NumKong (March 2026)** — `_references/NumKong/` is the active up-to-date kernel-pattern reference; cite it instead of SimSIMD for kernel patterns. (SIMD_PRIOR_ART)
- **usearch has no public RNG seed API + no candidate tie-breaker** — both are determinism gaps we close in our Builder. (DETERMINISM)
- **Use `rand_chacha::ChaCha8Rng` seeded from `image_salt`** for cross-arch byte-identical RNG. (DETERMINISM)
- **CVE-2023-37365 + pgvector + qdrant + hnswlib bug catalog** — production HNSW bugs to defend against. Phase 1 + Phase 5 risk rows updated. (HNSW_ALGORITHM_NOTES)
- **Tanimoto on binary collapses to Jaccard** — saves a kernel slot. (SIMD_PRIOR_ART, USEARCH_DEEP_DIVE)
- **VPDPBUSD's "XOR 0x80 unsigned-bias trick"** for AVX-512 i8 dot. (SIMD_PRIOR_ART)
- **hnswlib's `_MM_HINT_T0` graph-traversal prefetch** — adopt in walker inner loop. (SIMD_PRIOR_ART)

## Out of scope (recorded here so it doesn't drift back in)

- **Wrapping libusearch.** Decision in `HNSW_PATH_DECISION.md` §2: fully native, no C++ FFI in `tokenfs-algos`. `tokenfs_writer` calls our builder directly.
- **Multi-threaded builder.** v1 is single-threaded by design (SLSA-L3 mandates determinism). Optional `parallel` feature can land later.
- **GPU walker.** Separate `tokenfs-gpu` crate per `HARDWARE_ACCELERATION_LANDSCAPE.md` §6.
- **Multi-modal hybrid scoring (G2).** `similarity::hybrid` is a separate v0.8.0 landing.
- **Updates / deletions.** v1 builder is insert-only; walker is read-only. Extensible later if a consumer asks.
- **Quantization scalar types beyond `f32 / i8 / u8 / binary`.** F22 is byte-quantized; embeddings can be quantized to i8 by the producer. f16/bf16/e5m2 etc. defer to v0.9+ if a consumer needs them.

## Cross-references

- [`docs/HNSW_PATH_DECISION.md`](../HNSW_PATH_DECISION.md) — the parent decision doc.
- [`docs/KERNEL_SAFETY.md`](../KERNEL_SAFETY.md) — kernel-safe-by-default contract; HNSW inherits it.
- [`docs/PRIMITIVE_CONTRACTS.md`](../PRIMITIVE_CONTRACTS.md) — primitive design discipline (queue-pruning gate; pinned-kernel layout).
- [`docs/PROCESSOR_AWARE_DISPATCH.md`](../PROCESSOR_AWARE_DISPATCH.md) — per-backend kernel buffet pattern.
- `tokenfs-paper/docs/USEARCH_INTEGRATION_ANALYSIS.md` — the strategic frame (format-as-contract).
- `tokenfs-paper/docs/NATIVE_HYBRID_SIMILARITY.md` — the multi-modal scoring layer that lands above this.
- `tokenfs-paper/docs/IMAGE_FORMAT_v0.3.md` — the image-format spec (HNSW is section `0x203`).

---

*All docs in this tree update as Phase 1 starts. Skeletal docs are intentional — each phase's component spec gets filled in as that phase begins, not in advance.*

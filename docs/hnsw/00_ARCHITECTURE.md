# HNSW architecture

**Status:** architectural spec, 2026-05-03. Operationalizes the decision in
[`../HNSW_PATH_DECISION.md`](../HNSW_PATH_DECISION.md) into a concrete
module layout, posture matrix, and dependency graph.

## Layer map

| Layer | Where | Posture | Notes |
|---|---|---|---|
| Walker (search) | `similarity::hnsw` | `no_std + alloc` | Kernel-reachable; audit-R10 surface; `try_*` / `_unchecked` / `_inner` siblings |
| Builder (insert) | `similarity::hnsw::build` | `std + alloc` (gated) | Userspace-only; same audit conventions; uses walker's distance kernels |
| Wire format | usearch v2.25 (read **and** write) | spec contract | Byte-for-byte compatible — consumable by DuckDB / ClickHouse / pgvector |
| Distance kernels | `kernels/{scalar,avx2,avx512,neon,sse41,ssse3}.rs` | per-backend `arch-pinned-kernels` | Shared between walker and builder; iai-callgrind gated |
| Filter primitives | `similarity::hnsw::filter` | same module | In-search Roaring-bitmap pruning, composes with existing `bitmap::*` |
| Hybrid scoring (G2) | `similarity::hybrid` | separate v0.8.0 landing | Depends on walker output |
| GPU walker | `tokenfs-gpu` (future) | separate workspace | Out of scope for `tokenfs-algos`'s no_std contract |

## Module tree

```
crates/tokenfs-algos/src/similarity/hnsw/
├── mod.rs                  # public API re-exports + opaque type definitions
├── header.rs               # parse + write usearch v2.25 wire-format header
├── view.rs                 # zero-copy view over mmapped section bytes
├── graph.rs                # owned in-memory graph (used by builder)
├── walker.rs               # search-only graph traversal (no_std + alloc)
├── visit.rs                # bitset-based visited tracking
├── candidates.rs           # min-heap for k-NN + ef-search dynamic candidates
├── filter.rs               # in-search pruning: Option<&RoaringBitmap>
├── select.rs               # neighbor-selection heuristic (Algorithm 3 / 4)
├── build/                  # builder submodule — std-gated
│   ├── mod.rs              # public Builder API
│   ├── insert.rs           # one-vector insert
│   ├── level.rs            # logarithmic level assignment
│   └── serialize.rs        # in-memory graph → usearch v2.25 wire format
├── kernels/                # arch-pinned distance kernels
│   ├── mod.rs              # auto-dispatch
│   ├── scalar.rs
│   ├── avx2.rs
│   ├── avx512.rs           # nightly + feature = "avx512"
│   ├── neon.rs
│   ├── sse41.rs
│   └── ssse3.rs
└── tests.rs                # parity + round-trip tests
```

## Posture: kernel-safe vs userspace

| Path | Compiles in `--no-default-features --features alloc`? | Reachable in kernel module? |
|---|---|---|
| `try_search` (walker) | yes | yes (with integer/binary metrics; f32 needs FPU bracket) |
| `search` (panicking userspace wrapper) | no — `cfg(feature = "userspace")` gated | no |
| `Builder::try_new` / `try_insert` / `try_finish_to_bytes` | no — `cfg(feature = "std")` gated | no |
| `kernels::*::*_unchecked` (per-backend) | yes | yes (caller upholds contract) |
| `kernels::*::*` (per-backend, asserting) | no — `cfg(feature = "userspace")` gated | no |
| `kernels::auto::*` (runtime dispatcher) | yes | yes (routes through `_unchecked`) |

The pattern matches `bits::popcount`, `bits::streamvbyte`, `vector::distance` — every per-backend kernel has a paired `_unchecked` sibling so the runtime dispatcher can route to it from kernel-default builds, and the asserting variant is kept as a userspace-only oracle.

## Dependency graph

```
                        ┌─────────────────────────────┐
                        │  similarity::hybrid (v0.8)  │
                        │  multi-modal scoring        │
                        └──────────────┬──────────────┘
                                       │ consumes ranked candidates
                                       ▼
   ┌────────────────────────────────────────────────────────────────┐
   │              similarity::hnsw (this v0.7.0 landing)             │
   │                                                                 │
   │   ┌──────────────────┐    ┌──────────────────────────────────┐  │
   │   │  build::Builder  │    │  walker::try_search              │  │
   │   │  (std-gated)     │───▶│  (no_std + alloc, kernel-safe)   │  │
   │   │  uses search     │    │                                  │  │
   │   │  during insert   │    │  Returns Vec<(NodeKey, Distance)> │  │
   │   └────────┬─────────┘    └──────────────┬───────────────────┘  │
   │            │                              │                      │
   │            │   both call                  │                      │
   │            ▼                              ▼                      │
   │   ┌──────────────────────────────────────────────────────────┐  │
   │   │  kernels::auto::distance_*  (runtime dispatch)            │  │
   │   │    routes to scalar / AVX2 / AVX-512 / NEON / SSE / SSSE3 │  │
   │   └────────────────┬─────────────────────────────────────────┘  │
   │                    │                                              │
   │                    ▼                                              │
   │   ┌──────────────────────────────────────────────────────────┐  │
   │   │  graph view + visit bitset + candidate heap + filter     │  │
   │   │     (composed from existing primitives below)             │  │
   │   └────────────────┬─────────────────────────────────────────┘  │
   └─────────────────────┼─────────────────────────────────────────┘
                          │ reuses primitives from
                          ▼
   ┌────────────────────────────────────────────────────────────────┐
   │              existing tokenfs-algos primitives                  │
   │                                                                 │
   │   vector::*           — distance kernels (f32 / u32 / i8 / u8)  │
   │   bits::popcount      — binary distance metrics + bitset ops    │
   │   bits::rank_select   — visited-bitmap underpinning             │
   │   bitmap::*           — Roaring intersect/union (filter)        │
   │   hash::set_membership — alternative visited tracking           │
   │   approx::misra_gries — sorted-buffer pattern (candidate heap)  │
   │   dispatch::planner   — runtime backend selection               │
   └────────────────────────────────────────────────────────────────┘
```

The map of which existing primitives we reuse vs. write fresh lives in
[`research/PRIMITIVE_INVENTORY.md`](research/PRIMITIVE_INVENTORY.md)
once that research lands.

## API shape (skeletal)

```rust
//! crates/tokenfs-algos/src/similarity/hnsw/mod.rs

pub use self::header::{HnswHeader, HnswHeaderError};
pub use self::view::{HnswView, HnswViewError};
pub use self::walker::{
    SearchConfig, SearchResult, HnswSearchError,
    try_search, try_search_with_filter,
};

#[cfg(feature = "userspace")]
pub use self::walker::{search, search_with_filter};

#[cfg(feature = "std")]
pub use self::build::{
    Builder, BuildConfig, HnswBuildError,
};

pub use self::filter::HnswFilter;

pub mod kernels;  // arch-pinned distance kernels (gated on arch-pinned-kernels)
```

The walker's hot entry (`try_search`) takes:

```rust
pub fn try_search(
    view: &HnswView<'_>,        // zero-copy over mmapped section bytes
    query: &[u8],               // query vector (length validated against view's dim)
    config: &SearchConfig,      // k, efSearch, metric, scalar_kind
    filter: Option<&HnswFilter>, // None = unfiltered; Some = Roaring bitmap of permitted node IDs
) -> Result<Vec<(NodeKey, Distance)>, HnswSearchError>
```

The builder's API (skeletal):

```rust
#[cfg(feature = "std")]
impl Builder {
    pub fn try_new(config: BuildConfig) -> Result<Self, HnswBuildError>;

    /// Insert one vector. O(M · efConstruction · log N) per insert.
    pub fn try_insert(
        &mut self,
        key: NodeKey,
        vector: &[u8],
    ) -> Result<(), HnswBuildError>;

    /// Serialize the in-memory graph to usearch v2.25 wire format.
    /// Determinism contract: byte-for-byte identical to libusearch v2.25
    /// given the same single-threaded insertion order + same RNG seed.
    pub fn try_finish_to_bytes(self) -> Result<Vec<u8>, HnswBuildError>;
}
```

Detailed per-component API specs live in [`components/`](components/).

## Cross-cuts

- **Audit-R10 discipline.** Every public entry has its `try_*` / `_unchecked` / `_inner` shape per [`KERNEL_SAFETY.md`](../KERNEL_SAFETY.md). The panic-surface lint must stay at 0 entries; no new ungated panic macros land.
- **iai-callgrind regression gate.** Per-(metric, scalar_kind) bench rows added to `benches/iai_primitives.rs` for the hot kernels at fixed (k=16, N=10⁵). 1% IR regression threshold per the v0.5.0 T3.4 contract.
- **Scalar parity tests.** Per-backend `tests/parity.rs` rows for every `(metric, scalar_kind, backend)` cell. Same shape as `bits/streamvbyte/kernels` parity.
- **no-std-smoke crate.** `tokenfs-algos-no-std-smoke::smoke()` extends to call `try_search` and a representative integer-metric kernel, exercising the kernel-safe path through a real `#![no_std]` consumer.
- **Wire-format round-trip.** A fixture corpus (10⁴-vector F22 fingerprint index, ~1-2 MB) committed under `crates/tokenfs-algos/tests/data/hnsw/`, built with libusearch v2.25 single-threaded. Tests assert: (a) our walker returns the same k-NN as libusearch on the same query, (b) our Builder produces byte-identical serialization given the same insertion order + seed, (c) our Builder's output is consumable by libusearch's reader.

# HNSW Path Decision

**Status:** decision doc, 2026-05-03. This commits `tokenfs-algos` v0.7.0 to a specific HNSW implementation path. Supersedes any earlier "decide later" framing in `PLAN.md` §13.

> See also:
> - [`KERNEL_SAFETY.md`](KERNEL_SAFETY.md) — the `try_*` / `_unchecked` / `_inner` discipline this design inherits.
> - [`PRIMITIVE_CONTRACTS.md`](PRIMITIVE_CONTRACTS.md) — queue-pruning gate (§ "Hot-Path Contract") and pinned-kernel layout.
> - [`PROCESSOR_AWARE_DISPATCH.md`](PROCESSOR_AWARE_DISPATCH.md) — per-backend kernel buffet pattern.
> - `tokenfs-paper/docs/USEARCH_INTEGRATION_ANALYSIS.md` — the strategic frame this doc operationalizes.
> - `tokenfs-paper/docs/NATIVE_HYBRID_SIMILARITY.md` — the multi-modal hybrid scoring layer that sits on top of this walker.
> - `tokenfs-paper/docs/HARDWARE_ACCELERATION_LANDSCAPE.md` — explains why GPU paths live in a separate `tokenfs-gpu` crate, not here.

---

## §1. Decision

`tokenfs-algos` will ship a **pure-Rust no_std HNSW *walker*** (search-only graph traversal) under `similarity::hnsw`. The walker:

- Reads the **usearch wire format** (`unum-cloud/usearch` v2.25 serialized index, byte-for-byte) — adopted as the on-disk contract.
- Is **search-only**. Index *construction* is delegated to libusearch in the userspace builder (`tokenfs_writer`, separate workspace member). Build is one-time, can be slow, runs in std + threads; search is hot-path, kernel-reachable, must be no_std + alloc + audited.
- Follows the existing audit-R10 patterns: `try_*` siblings + `_unchecked` / `_inner` split, per-backend kernel buffet under `arch-pinned-kernels`, scalar parity tests, iai-callgrind benches, `panic_surface_allowlist.txt = 0`.
- Distance metrics in v1: **L2², cosine, dot, Hamming, Jaccard, Tanimoto**, on `f32` / `i8` / `u8` / `binary` scalar types. The integer/binary metrics are the kernel-mode default (no FPU bracketing required).
- Filter primitives (Roaring bitmap of permitted node IDs) are co-designed with the walker — same module — because they are required for in-search pruning rather than post-filter (per `NATIVE_HYBRID_SIMILARITY.md` N3).

**What this is not:**

- Not a from-scratch HNSW implementation that invents its own format. The agent's market survey (every serious vector DB writes their own walker but few invent their own format) confirms the value of format alignment. Adopting usearch's format gets us multi-language reader compatibility (DuckDB, ClickHouse, pgvector) for free.
- Not a wrap of libusearch in `tokenfs-algos` itself. libusearch is C++17 + STL; can't compile no_std, can't link into a kernel module, requires `cxx`/`cxx-build` at the crate boundary. It belongs in `tokenfs_writer` for the build path, not in our kernel-safe primitive crate.
- Not a multi-modal hybrid scoring layer. That sits **above** the walker as `similarity::hybrid` (a separate gap, G2 in the gap review). The walker returns single-metric ranked candidates; the hybrid layer combines them.
- Not a GPU path. GPU work (per `HARDWARE_ACCELERATION_LANDSCAPE.md` §6) lives in a separate `tokenfs-gpu` crate so it doesn't pollute the no_std contract here.

---

## §2. Why this synthesis (not pure native, not pure wrap)

The two extreme positions are easy to defend in isolation but lose to the middle:

**Pure native (walker + builder both in pure Rust, invent our own format).** Tempting for purity. Cost: ~6 weeks of HNSW work *plus* ~2-3 months of catching up to libusearch's tuning *plus* losing the format-as-contract win. The agent's survey ([HNSW Rust ecosystem, 2026-05-03]) found that every serious vector DB (Qdrant, Weaviate, pgvector, Milvus) **writes its own walker** but **none invent their own wire format gratuitously** — they either adopt an existing one or define their own as the tail-end of an algorithmic specialization. We are not doing the latter.

**Pure wrap (libusearch in both userspace and kernel via cxx).** Doesn't exist. libusearch is C++17 + STL, cannot link into the kernel, and pulling cxx into `tokenfs-algos` would burn the no_std-clean property we just spent two weeks defending through audit-R10.

**The synthesis (walker native, builder wrapped, format shared)** matches what real Linux subsystems do. fs-verity, dm-verity, IMA all separate userspace-builders from kernel-readers via a shared on-disk format. We are taking the same shape.

| Layer | Implementation | Lives in | Reason |
|---|---|---|---|
| Wire format | usearch v2.25 (adopted byte-for-byte) | spec contract | Industry alignment; multi-language reader compat. |
| Builder | libusearch (C++17, STL, threads) | `tokenfs_writer` (separate crate) | Build-once code; well-tuned existing implementation. |
| Walker | pure Rust no_std + alloc | `tokenfs-algos::similarity::hnsw` | Hot path; needs to reach kernel module + audit posture. |
| Hybrid scoring | pure Rust (above walker) | `tokenfs-algos::similarity::hybrid` | Multi-modal score combination; depends on walker's output. |
| GPU walker (later) | CUDA/HIP | `tokenfs-gpu` (future, separate workspace) | Doesn't share constraints with `tokenfs-algos`'s no_std posture. |

---

## §3. Queue-pruning gate (per `PRIMITIVE_CONTRACTS.md` §"Queue-Pruning Gate")

Every new primitive answers four questions before joining the queue. For HNSW walker:

1. **What query/workload binds on this primitive being slow?**
   The "find files structurally + semantically + content-wise similar to my buffer" query in `NATIVE_HYBRID_SIMILARITY.md` §1 — without an HNSW walker this is brute-force linear scan over the fingerprint sidecar (O(N) per query at ~1-10 GiB/s scan throughput). HNSW takes the same query to O(log N) at similar per-step cost. At N=10⁶ extents (a typical TokenFS image), that's the difference between sub-millisecond and seconds per query. The agentic-FS feature catalog (`AI_AGENTIC_FS_FEATURES.md` S1 `find_similar`) names this as the load-bearing latency win.

2. **What's the cache-residency picture for the data this primitive operates on?**
   HNSW graph nodes are typically 64-256 B each (one cache line plus neighbor IDs). A 10⁶-node graph at 256 B/node is 256 MiB — fits in L3 on server-class hardware, mmap'd from the section payload on smaller hosts. Beam-search expansion touches O(M log N) neighbor lists per query (M = connectivity), so the working set per query is small (a few KiB) but spread across the whole graph randomly — bandwidth-sensitive, not latency-sensitive. This argues for: (a) packed neighbor representation that fits in fewer cache lines, (b) prefetch hints in the inner loop, (c) integer/byte distance metrics that don't pull FPU state.

3. **Which consumer environments can this primitive operate in?**
   Per `02b_DEPLOYMENT_MATRIX.md`: kernel-mode (no FPU outside `kernel_fpu_begin/end`, no rayon, ~8-16 KB stack), FUSE (full userspace), Postgres extension (must be cgo-friendly via the C ABI), CDN edge (no rayon, latency-bound), Python research (PyO3). The walker needs to satisfy all five. The integer/binary metrics path is kernel-safe by construction; the f32 path needs FPU bracketing (or stays out of kernel mode).

4. **What's the consumer surface — one consumer or many?**
   Many — every consumer named above will issue similarity queries. Multi-consumer means the walker gets its own module (`similarity::hnsw`), not parked inside one consumer's code path.

Gate passes. Walker queues for v0.7.0.

---

## §4. Module structure

Following the established `bits::popcount` / `vector::distance` / `permutation::rabbit` patterns:

```
crates/tokenfs-algos/src/similarity/hnsw/
├── mod.rs                  # public API: try_search, try_search_with_filter
│                           # opaque types: HnswView, HnswHeader, SearchConfig
├── header.rs               # parse usearch v2.25 wire-format header
│                           # validate magic / version / scalar_kind / metric_kind
├── view.rs                 # zero-copy view over mmapped section bytes
│                           # node lookup, neighbor iteration, vector access
├── walker.rs               # search-only graph traversal
│                           # internal _inner helper, public try_*, userspace-gated panicking sibling
├── visit.rs                # bitset-based visited tracking (reuses bits::rank_select primitives)
├── candidates.rs           # min-heap for k-NN candidates (top-K with bounded allocation)
├── filter.rs               # in-search pruning: Option<&RoaringBitmap> of permitted node IDs
│                           # composes with bitmap::* SIMD kernels
├── kernels/                # arch-pinned distance kernels for the metrics walker uses
│   ├── mod.rs              # auto-dispatch
│   ├── scalar.rs           # reference impl for every (metric, scalar_kind) pair
│   ├── avx2.rs             # x86 AVX2: i8 / u8 / f32 / binary
│   ├── avx512.rs           # nightly: f32 + i8 + binary; gated on feature = "avx512"
│   ├── neon.rs             # AArch64: i8 / u8 / f32 / binary
│   ├── sve2.rs             # AArch64 SVE2 (deferred; nightly-gated)
│   ├── sse41.rs            # x86 fallback: integer paths
│   └── ssse3.rs            # x86 fallback: binary popcount-based metrics
└── tests.rs                # parity tests against libusearch reference indexes
```

The walker is **`no_std + alloc`-clean**. No file I/O, no thread spawn, no FPU outside what `kernels::*` brackets when in kernel context. Inputs: `&[u8]` (mmapped section bytes), `&[u8]` (query vector), `SearchConfig`, `Option<&RoaringBitmap>`. Outputs: `Vec<(NodeKey, Distance)>` capped at `k`.

The builder is **NOT in this crate.** It lives in `tokenfs_writer` and wraps libusearch via the existing `usearch` Rust crate. Out of scope here.

---

## §5. Audit-R10 discipline applied

Every public function follows the contract from `KERNEL_SAFETY.md`:

```rust
// In `similarity::hnsw::walker`:

#[inline(never)]
fn try_search_inner(
    view: &HnswView<'_>,
    query: &[u8],
    config: &SearchConfig,
    filter: Option<&RoaringBitmap>,
    out: &mut Vec<(NodeKey, Distance)>,
) -> Result<(), HnswSearchError> {
    // Pre-validated body — header + dims + scalar match query, k > 0,
    // out has cap >= k. Shared between try_search and the panicking
    // userspace wrapper below.
    ...
}

// Kernel-safe entry. Reachable in --no-default-features --features alloc.
pub fn try_search(
    view: &HnswView<'_>,
    query: &[u8],
    config: &SearchConfig,
    filter: Option<&RoaringBitmap>,
) -> Result<Vec<(NodeKey, Distance)>, HnswSearchError> {
    if config.k == 0 {
        return Err(HnswSearchError::InvalidK);
    }
    if query.len() != view.header().vector_bytes() {
        return Err(HnswSearchError::QueryDimMismatch { ... });
    }
    let mut out = Vec::with_capacity(config.k);
    try_search_inner(view, query, config, filter, &mut out)?;
    Ok(out)
}

// Userspace ergonomic. Gated; never reachable in kernel-default builds.
#[cfg(feature = "userspace")]
pub fn search(
    view: &HnswView<'_>,
    query: &[u8],
    config: &SearchConfig,
    filter: Option<&RoaringBitmap>,
) -> Vec<(NodeKey, Distance)> {
    try_search(view, query, config, filter)
        .expect("search inputs satisfy contract; see try_search for fallible variant")
}
```

Backend kernels follow the `_unchecked` sibling pattern from `bits/streamvbyte/kernels` and `approx/bloom_kernels`:

```rust
// In `similarity::hnsw::kernels::avx2`:

#[target_feature(enable = "avx2")]
pub unsafe fn distance_l2_squared_i8_unchecked(
    a: *const i8,
    b: *const i8,
    len: usize,
) -> u32 { ... }

// Asserting variant only compiled in userspace. Same kernel, plus
// upfront assertions on alignment / length / CPU-feature presence.
#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
pub unsafe fn distance_l2_squared_i8(
    a: *const i8,
    b: *const i8,
    len: usize,
) -> u32 {
    debug_assert!(len > 0);
    debug_assert!(len.is_multiple_of(32));  // AVX2 lane width
    debug_assert!(is_x86_feature_detected!("avx2"));
    distance_l2_squared_i8_unchecked(a, b, len)
}
```

The `kernels::auto` dispatcher routes through `_unchecked` after CPU detection at the *outer* boundary; same as every other primitive in the crate.

**Lint expectations:**
- `cargo xtask panic-surface-lint` — must stay at 0 entries in `panic_surface_allowlist.txt`. The HNSW walker introduces no new gated-only entries.
- `cargo xtask security` — must build the no_std smoke crate calling `try_search` (kernel-safe path) — extends `tokenfs-algos-no-std-smoke::smoke()`.
- iai-callgrind — adds `iai_hnsw_search_*` benches (one per metric × scalar_kind combo, k=16) to `iai_primitives.rs`. Same 1% IR regression gate as the rest.

---

## §6. Hardware acceleration

The distance kernel is the hot inner loop — one call per beam-search candidate, ~M·efSearch·log(N) calls per query. Backend selection table:

| Backend | Metrics | Scalar kinds | Status | Why |
|---|---|---|---|---|
| **scalar** | all | all | required | reference oracle; portable |
| **AVX2** | L2² / cosine / dot / Hamming / Jaccard | i8 / u8 / f32 / binary | v1 — primary x86 target | full per-deployment-matrix coverage |
| **AVX-512** | L2² / cosine / dot / Hamming | f32 / i8 / binary | v1 — nightly + `avx512` feature | VPOPCNTQ for binary metrics; FMA for f32 |
| **NEON** | L2² / cosine / dot / Hamming / Jaccard | i8 / u8 / f32 / binary | v1 — Apple Silicon / Graviton primary | parity with AVX2 |
| **SSE4.1** | L2² / Hamming / Jaccard | i8 / u8 / binary | v1 fallback | older x86 |
| **SSSE3** | binary metrics only | binary | v1 fallback | popcount-via-PSHUFB |
| **SHA-NI / FEAT_SHA2** | n/a | n/a | not applicable | hashes, not distances |
| **SVE2** | all | all | deferred (nightly) | wait on `stdarch_aarch64_sve` stabilization or implement via inline asm; no immediate consumer |
| **GPU (CUDA/HIP)** | all | all | **out of scope** for this crate | lives in future `tokenfs-gpu` per `HARDWARE_ACCELERATION_LANDSCAPE.md` §6 |

**Kernel-mode FPU note.** Per `USEARCH_INTEGRATION_ANALYSIS.md` §4, `kernel_fpu_begin/end` brackets are nontrivial cost (~hundred-cycle save+restore) and the naive HNSW inner loop would call them per candidate. Two mitigations land together:

1. **Integer/binary metrics are the kernel default.** F22 fingerprints are byte-quantized (32 × u8); MinHash signatures are u32 hashes; SimHash is a single u64. None require FPU. The kernel-safe `try_search` path with these metrics never touches floating-point state.
2. **f32 metrics, when used, batch FPU brackets at the per-query (not per-candidate) level.** Same pattern as kernel ZFS Fletcher4. Documented in the f32 kernel module's docstring; enforced via a `kernel_fpu_guard!` macro that the walker uses for f32 paths only.

Userspace consumers don't pay either of these costs and can use any (metric, scalar) combination directly.

---

## §7. Wire format (the contract)

Adopt usearch v2.25.x serialized format byte-for-byte. The TokenFS section wrapper is an 8-byte prefix per `IMAGE_FORMAT_v0.3_SIMILARITY.md` (the sub-spec gets updated alongside this work):

```c
struct hnsw_section_header {
    u8     magic[4];                  // {'H','N','S','W'}
    u8     section_version;           // == 0x01 for the v0.3 spec
    u8     reserved[3];               // MUST be zero
    u8     usearch_blob[];            // entire usearch serialized index,
                                      // starting with usearch's own
                                      // 64-byte header at byte 8
};
```

The walker's API takes `&[u8]` pointing at the section payload. It validates the 8-byte TokenFS wrapper, then hands the rest to the usearch-format header parser at `byte 8`. The header parser is ~100 LOC of byte-offset arithmetic (per `_references/usearch/include/usearch/index_dense.hpp:42-79`) — no allocation, no I/O.

**Pinned format version:** v2.25.x. Future usearch format versions land in a *new* section ID (e.g., `0x213` per the sub-spec) so old images keep working with old walkers; we never silently shift the contract.

**Multi-consumer compatibility.** Because we adopted usearch's format byte-for-byte, the section payload (after the 8-byte wrapper) is consumable by:
- `usearch_view` in libusearch (C, Python, Go, Java, Rust, etc.)
- DuckDB's vector extension (which uses usearch internally)
- ClickHouse's vector index (which uses usearch internally)
- Future pgvector backends if they adopt usearch (some are trialing it)

This is a real win. We did not invent another HNSW serialization.

---

## §8. Determinism

Builds are reproducible per the SLSA-L3 contract in v0.3 §11. For HNSW that means:

- libusearch builds in `tokenfs_writer` MUST use single-threaded insertion (no `add()` from multiple threads concurrently).
- Vectors MUST be inserted in deterministic order (sorted by external key ascending).
- `expansion_add` parameter MUST be set to 1 (disable randomized expansion).

These are constraints on the *builder*, not the walker. The walker is deterministic by construction (no shared state, no thread-local RNG).

Cost: single-threaded HNSW build is ~10× slower than parallel. For typical TokenFS images (10⁵–10⁷ extents), single-threaded build runs in seconds-to-minutes — acceptable for a one-time build step. Documented in `IMAGE_FORMAT_v0.3_SIMILARITY.md` as a writer requirement.

---

## §9. Implementation phases

Total estimated effort: **6 weeks** of focused engineering. Each phase ends with a demo-able state.

### Phase 1 — wire format + walker skeleton (week 1)

- `header.rs` — parse usearch v2.25 wire-format header; magic + version validation; scalar / metric kind enum mapping.
- `view.rs` — zero-copy view over mmapped bytes; node offset arithmetic; bounds-checked neighbor iteration.
- `walker.rs::try_search_inner` — beam-search loop, scalar-only distance dispatch, no SIMD yet.
- `kernels::scalar` — L2² / cosine / dot / Hamming for f32 + i8 + u8 + binary (8 distance kernels).
- `tests.rs` — parity test against a libusearch-built reference index (committed as test fixture, ~50 KB).
- `panic_surface_allowlist.txt` stays at 0; `try_search` is the kernel-safe entry.

**Demo:** `try_search` returns top-16 by Hamming distance over a 10⁴-node F22 fingerprint index, matching libusearch reference output bit-for-bit.

### Phase 2 — SIMD distance kernels + iai benches (weeks 2-3)

- `kernels::avx2` — i8 / u8 / f32 / binary (8 kernels × `_unchecked` + asserting siblings = 16 entries).
- `kernels::neon` — same matrix on AArch64.
- `kernels::sse41` — integer fallback for older x86.
- Per-backend scalar parity tests (the `tests/parity.rs` pattern from `bits/streamvbyte`).
- Add `iai_hnsw_search_*` cases to `benches/iai_primitives.rs` for the hot metric × scalar combos at fixed k=16, N=10⁵.
- `dispatch::planner` adds an HNSW signal class — picks scalar / AVX2 / NEON based on cpu_features and graph dims.

**Demo:** AVX2 i8 Hamming search at >10× scalar throughput; iai-callgrind bench rows for popcount-style instruction-count gating.

### Phase 3 — filter primitives + AVX-512 (week 4)

- `filter.rs` — `Option<&RoaringBitmap>` of permitted node IDs; tested for in-search-pruning vs. post-filter equivalence.
- `kernels::avx512` (nightly) — f32 + i8 + binary; VPOPCNTQ for binary metrics.
- Filter primitive composes with `bitmap::*` SIMD kernels (already present in the crate).

**Demo:** capability-aware search returns only files in the user's permitted cluster; HNSW's sub-linear win is preserved (vs. post-filter dropping it).

### Phase 4 — kernel-FPU bracketing + integration (weeks 5-6)

- `kernel_fpu_guard!` macro for f32 paths; one bracket per query (not per candidate).
- Integer-only `try_search_kernel_safe` variant explicitly for kernel-module callers (rejects f32 metrics at the type level).
- `tokenfs-algos-no-std-smoke` extends `smoke()` to call `try_search` so the no-std-smoke CI step exercises the new entry.
- Wire format version compatibility test: old usearch v2.20 indexes should fail closed at the header parser, not silently produce wrong results.
- `dispatch::planner` and `processor profile` get HNSW-aware sizing rules.

**Demo:** `cargo xtask security` green. Allowlist still 0. iai-callgrind regression gate covers HNSW. README updated.

### Out-of-phase (parallel work)

- libusearch builder integration in `tokenfs_writer` — separate workspace member; ~50 LOC Rust binding around the existing `usearch` crate. Independent of this phase plan.
- `IMAGE_FORMAT_v0.3_SIMILARITY.md` — sub-spec writeup; lives in `tokenfs-paper`. Independent.
- `similarity::hybrid` (multi-modal scoring) — depends on this walker landing first; queued for v0.8.0.
- `content_class` module (G3) — paper-side dictionary lock first; then the module here.

---

## §10. Backend coverage matrix (target end-state at v0.7.0 release)

Per `PROCESSOR_AWARE_DISPATCH.md`'s buffet language. Every cell either ships or is explicitly marked deferred with rationale.

| Metric | scalar | AVX2 | AVX-512 (nightly) | NEON | SSE4.1 | SSSE3 | SVE2 |
|---|---|---|---|---|---|---|---|
| L2² (f32) | ✅ | ✅ | ✅ FMA | ✅ | — | — | deferred |
| L2² (i8 / u8) | ✅ | ✅ | ✅ | ✅ | ✅ | — | deferred |
| cosine (f32) | ✅ | ✅ | ✅ FMA | ✅ | — | — | deferred |
| cosine (i8 / u8) | ✅ | ✅ | ✅ | ✅ | ✅ | — | deferred |
| dot (f32) | ✅ | ✅ | ✅ FMA | ✅ | — | — | deferred |
| Hamming (binary) | ✅ | ✅ | ✅ VPOPCNTQ | ✅ VCNT | ✅ POPCNT | ✅ PSHUFB-popcount | deferred |
| Jaccard (binary) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | deferred |
| Tanimoto (binary) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | deferred |

This matches the per-deployment-matrix priority in `02b_DEPLOYMENT_MATRIX.md`: NEON high (Apple Silicon, Graviton), AVX2 universal x86, AVX-512 medium (Sapphire Rapids, Zen5), SVE2 deferred until a real consumer demands it.

---

## §11. Scope boundaries (what this is *not*)

Explicitly **out of scope** for the v0.7.0 walker landing:

- **Index construction.** Lives in `tokenfs_writer` as a libusearch wrapper.
- **Multi-modal hybrid scoring.** Lives in `similarity::hybrid`, separate landing.
- **GPU walker.** Separate `tokenfs-gpu` crate; cuFile + GPUDirect Storage flow per `HARDWARE_ACCELERATION_LANDSCAPE.md` §6. Out of scope for `tokenfs-algos`.
- **Embedding sidecar producer.** Out of crate per gap review G9. Lives in a separate workspace member (`tokenfs-embed-producer` or similar) with a designated owner.
- **Updates / deletions.** The walker is search-only. usearch supports incremental update, but our consumers (TokenFS sealed images) build once and read many times. If incremental update is needed later, it lands in `tokenfs_writer`, not in the walker.
- **Quantization training.** F22 fingerprints arrive pre-quantized to u8; the walker treats them as opaque scalars. Quantization choice lives in `fingerprint::*` and the F22 sub-spec.
- **Distance metrics beyond the table in §10.** Custom user metrics are an explicit non-goal — the walker dispatches on a fixed enum, not a function pointer. This is what lets us audit + iai-bench every kernel exhaustively.

---

## §12. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| usearch wire format changes in v3.0 | medium | Pin to v2.25; new format version → new section ID per v0.3 §11 forward-compat. Conformance tests against canonical reference indexes. |
| Walker drifts from libusearch's algorithm semantics | medium | Parity test in `tests.rs` against libusearch-built reference index + same query → bit-identical k-NN result list. Regression-gated. |
| FPU bracketing in kernel mode is wrong | high | Integer/binary metrics are the kernel default; f32 paths use the explicit `kernel_fpu_guard!` macro that brackets per-query. Integration test in kernel-module smoke crate. |
| SIMD distance kernels disagree with scalar oracle | high | Standard pattern — `tests/parity.rs` per backend, run in CI on every push. Same as every other primitive in the crate. |
| Audit posture regression | high (we just paid this debt) | `cargo xtask panic-surface-lint` gate at 0 entries; new walker entries follow `try_*` / `_unchecked` / `_inner`. CI blocks merges. |
| Multi-modal scoring layer (G2) gets entangled with walker | medium | Strict separation: walker returns single-metric ranked candidates; `similarity::hybrid` (separate module, separate landing) combines them. Walker has no `weights[]` parameter. |
| Builder determinism not enforced | medium (SLSA-L3 cost) | Documented in `IMAGE_FORMAT_v0.3_SIMILARITY.md` as a writer MUST. `tokenfs_writer` integration test re-builds a small reference image and asserts byte-identical section payload. |
| Kernel walker takes longer than estimated | low | The 6-week estimate is conservative. usearch wire format is well-documented; the algorithm (HNSW) is published. The non-trivial work is the per-backend SIMD kernels, not the graph walk. |

---

## §13. Cross-references and follow-on work

This decision unblocks (or is a prerequisite for):

- **G2** (multi-modal hybrid scoring layer, gap review §2) — depends on this walker. Queued v0.8.0.
- **G4** (filter primitives) — co-designed with the walker, in this same landing. No separate task.
- **G5** (Merkle binding for similarity sidecars) — orthogonal; can land before, with, or after the walker.
- **G6** (image_salt threading) — must land before any signed external image. Independent SemVer-major bump in v0.6.0.
- **`SIMILARITY_API_SURFACE.md` review pushback** (xattr namespace, query-shape xattrs, per-file overlay) — orthogonal API-surface concerns; documented separately.

This decision *supersedes*:

- The "decide later" framing in `PLAN.md` §13 item 4 (now resolved).
- The "Option A vs Option D" choice in `USEARCH_INTEGRATION_ANALYSIS.md` §7 — we are taking the synthesis: Option A's format-as-contract + a native walker that owns the search path. Neither pure-A nor pure-D.

This decision *does not* change:

- The kernel-safe-by-default narrative in `KERNEL_SAFETY.md`.
- The audit-R10 discipline (allowlist still 0, panic-surface-lint still gates).
- The benchmark contract in `PRIMITIVE_CONTRACTS.md` §"Benchmark Contract".
- The deployment matrix obligations in `v0.2_planning/02b_DEPLOYMENT_MATRIX.md`.

---

## §14. References

- `_references/usearch/include/usearch/index_dense.hpp` — wire format header struct (lines 42-79).
- `_references/usearch/include/usearch/index.hpp` — graph topology + search loop (line 1850 `memory_mapped_file_t`).
- `_references/usearch/c/usearch.h` — C99 ABI surface (used by the Rust `usearch` crate in `tokenfs_writer`).
- `_references/usearch/BENCHMARKS.md` — usearch performance numbers (vs. FAISS; reference for our walker's targets).
- `_references/README.md` — provenance of the reference subtree.
- [HNSW paper (Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320) — original algorithm.
- [fs-verity Documentation](https://www.kernel.org/doc/html/latest/filesystems/fsverity.html) — format-as-contract precedent (kernel reads, userspace builds).
- `tokenfs-paper/docs/USEARCH_INTEGRATION_ANALYSIS.md` — the strategic frame this doc operationalizes.
- `tokenfs-paper/docs/NATIVE_HYBRID_SIMILARITY.md` — the hybrid scoring layer that lands above this.
- `tokenfs-paper/docs/SIMILARITY_API_SURFACE.md` — consumer-facing API surface (xattr / magic-path / ioctl / MCP / CLI layers).
- `tokenfs-paper/docs/HARDWARE_ACCELERATION_LANDSCAPE.md` §6 — GPU-direct path; explains why it's a separate crate.
- `crates/tokenfs-algos/docs/KERNEL_SAFETY.md` — kernel-safe-by-default contract.
- `crates/tokenfs-algos/docs/PRIMITIVE_CONTRACTS.md` — primitive design discipline.
- `crates/tokenfs-algos/docs/PROCESSOR_AWARE_DISPATCH.md` — per-backend kernel buffet pattern.
- `crates/tokenfs-algos/docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md` — multi-consumer constraints.

---

*This decision is intended to be stable until the walker ships. Update only if: (a) usearch ships v3.0 with breaking format changes before we land, (b) a kernel-mode HNSW consumer emerges that requires a different scope split, (c) GPU work merges back into `tokenfs-algos` (it should not).*

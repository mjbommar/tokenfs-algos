# Phases & dependency tree

**Status:** plan, 2026-05-02. Phases are dependency-ordered; rate of work is contributor's choice.

## Dependency graph

```
                              ┌─────────────────────┐
                              │ Phase A — roots     │
                              │ (no module deps)    │
                              └──────────┬──────────┘
                                         │
        ┌────────────────────┬───────────┼──────────────┬─────────────────┐
        │                    │           │              │                 │
        ▼                    ▼           ▼              ▼                 ▼
   bits/popcount      bits/bit_pack   hash/batched    vector/distance  hash/membership
   (rank/select       (token decode,  (BLAKE3+SHA-256 (L2², dot,       (set membership
    foundation)        succinct DS)    batched API)    cosine, …)       SIMD)
        │                    │           │              │                 │
        │ depends on         │           │              │                 │
        ▼ popcount           │           │              │                 │
   bits/rank_select          │           │              │                 │
        │                    │           │              │                 │
        │ depends on rank/select         │              │                 │
        ▼                    │           │              │                 │
   bits/streamvbyte ─────────┘           │              │                 │
   (independent of                       │              │                 │
    rank/select; uses                    │              │                 │
    bit_pack scaffolding)                │              │                 │
        │                                │              │                 │
        ▼                                ▼              ▼                 ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Phase B — high-leverage primitives that build on Phase A                │
   └────────────────────┬────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼──────────────────┐
        │               │                  │
        ▼               ▼                  ▼
   bitmap/roaring   permutation/RCM    permutation/rabbit
   (uses popcount   (CSR walk,         (community detection,
    for cardinality)  argsort)          dendrogram DFS)
        │               │                  │
        │               │                  │
        ▼               ▼                  ▼
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Phase C — composition demonstrators                                     │
   └─────────────────────────────────────────────────────────────────────────┘
        │
        ▼
   examples: token n-gram inverted index (bitmap + bits/streamvbyte)
   examples: image build pipeline (hash batched + permutation)
   examples: similarity scan (vector/distance + bits/bit_pack)
```

## Phase A — roots (no inter-module deps)

These can ship in any order; nothing depends on anything else here.

| # | Module / primitive | Doc | Estimated complexity |
|---|---|---|---|
| A1 | `bits::popcount` (scalar + AVX2 + AVX-512 VPOPCNTQ + NEON VCNT) | [`10_BITS.md`](10_BITS.md) § 4 | small (1-2 days) |
| A2 | `bits::bit_pack` (arbitrary widths 1-32) | [`10_BITS.md`](10_BITS.md) § 2 | medium (3-5 days) |
| A3 | `hash::batched_blake3` + `hash::batched_sha256` | [`12_HASH_BATCHED.md`](12_HASH_BATCHED.md) § 2 | small (2-3 days) |
| A4 | `hash::set_membership_simd` | [`12_HASH_BATCHED.md`](12_HASH_BATCHED.md) § 3 | small (1-2 days) |
| A5 | `vector::distance` (5 metrics × scalar/AVX2/AVX-512/NEON) | [`13_VECTOR.md`](13_VECTOR.md) | medium (5-7 days) |

**Phase A ship gate:** all 5 primitives have scalar oracle + at least one SIMD backend with parity tests, dispatched via existing `dispatch::` infrastructure, with criterion benches showing measurable speedup over scalar on representative inputs.

## Phase B — high-leverage primitives that build on Phase A

| # | Module / primitive | Depends on | Doc |
|---|---|---|---|
| B1 | `bits::rank_select` | A1 popcount | [`10_BITS.md`](10_BITS.md) § 5 |
| B2 | `bits::streamvbyte` (encode + decode) | A2 bit_pack scaffolding | [`10_BITS.md`](10_BITS.md) § 3 |
| B3 | `bitmap::roaring` (intersection/union/difference/cardinality) | A1 popcount | [`11_BITMAP.md`](11_BITMAP.md) |
| B4 | `permutation::rcm` (Reverse Cuthill-McKee) | none (A1 popcount used by argsort optionally) | [`14_PERMUTATION.md`](14_PERMUTATION.md) § 2 |
| B5 | `permutation::hilbert` (2D/N-D Hilbert curve sort) | none | [`14_PERMUTATION.md`](14_PERMUTATION.md) § 4 |

**Phase B ship gate:** parity tests against scalar reference (or against a known-good external impl: `roaring`-rs scalar for B3; `sprs::reverse_cuthill_mckee` for B4); criterion benches; for B3, head-to-head numbers against the `roaring` crate's scalar inner loops on intersection of two ~10K-cardinality posting lists.

## Phase C — composition demonstrators

| # | Demonstrator | Composes |
|---|---|---|
| C1 | Token n-gram inverted index example | B3 (Roaring) + B2 (Stream-VByte) |
| C2 | Image build pipeline benchmark | A3 (batched BLAKE3) + B4 (RCM) |
| C3 | Similarity scan benchmark | A5 (vector distance) + A2 (bit_pack for fingerprint decode) |

**Phase C ship gate:** examples run from `cargo run --example`, produce real numbers reported in benchmark JSON, and inform the `tokenfs-paper` consumer specification for what's actually achievable.

## Phase D — opportunistic Tier-B finishing

| # | Module / primitive | Depends on | Doc |
|---|---|---|---|
| D1 | `permutation::rabbit_order` (community-detection DFS ordering) | nothing strictly required, but B4 RCM provides the interface template | [`14_PERMUTATION.md`](14_PERMUTATION.md) § 3 |

Rabbit Order is highest-quality but no Rust port exists, so this is a multi-week effort. Defer unless image-layout quality becomes a measured bottleneck. RCM (B4) covers the 80% case at 1% of the cost.

## What's NOT in this plan

See [`20_DEFERRED.md`](20_DEFERRED.md) for the full deferred list. Highlights:

- **CSR walk + BFS frontier kernels** — wait for a documented graph-traversal hot path. The `permutation` module gives you the layout that makes CSR walks fast scalarly first; SIMD on the walk itself is incremental and harder to justify.
- **MinHash SIMD signature kernel** — wait for "find similar files" to become a primary query.
- **Wavelet tree, FM-index** — Phase B (`bits::rank_select`) lands the foundation; the trees themselves wait for a consumer asking for token-stream rank/select queries.
- **HNSW** — almost certainly never for 6-D fingerprints; reconsider only when learned embeddings ship.

## Ordering notes

1. **A3 (batched hash) is small and high-value** — could ship first to unblock the `tokenfs_writer` build pipeline immediately.
2. **A5 (vector distance) parallels A2/A3** — independent skeleton; a different contributor could land it concurrently.
3. **B3 (Roaring) is the longest single piece** in Phase B — if there's only one contributor, scheduling B3 to start as soon as A1 lands gives the most parallelism.
4. **B4 (RCM) is the most "deliverable today" item** because `sprs::reverse_cuthill_mckee` exists as the oracle and the Rust port doesn't need novel SIMD.
5. **D1 (Rabbit Order) is the only "really hard" item** — original C++ uses Boost concurrent hash maps, dendrogram structures, and parallel agglomerative merging. Plan for a real engineering effort, not a port.

## Per-phase contributor sketch

If solo: A1 → A3 → A2 → A4 → A5 → B1 → B2 → B3 → B4 → B5 → C* → D1.

If two contributors:
- Track 1: A1 → A2 → B1 → B2 → C1
- Track 2: A3 → A4 → A5 → B3 → C3 (in parallel) → B4 → C2 → D1

Both tracks meet at Phase C demonstrators.

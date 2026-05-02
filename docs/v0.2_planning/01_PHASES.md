# Phases & dependency tree

**Status:** plan, 2026-05-02. Phases are dependency-ordered; rate of work is contributor's choice. Phase ordering is consumer-agnostic: the same dependency tree applies regardless of whether the immediate consumer is TokenFS reader, Postgres extension, or a CDN edge cache. Consumer-environment fitness per primitive is documented in [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md) and in each module doc's "Environment fitness" section.

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

## Milestone breakdown

Phase A as a single milestone is too big to ship cleanly in solo development (5 sub-modules × parity tests + benches + 4 SIMD backends each ≈ 2-3 weeks of focused work). Split into two production milestones:

### v0.1.x (stopgap — ships first, ~1 week solo)

Two small, independent, high-impact primitives that unblock immediate work without waiting for the rest of Phase A:

| # | Module / primitive | Why this slice first | Doc |
|---|---|---|---|
| **A1** | `bits::popcount` (scalar + AVX2 + AVX-512 VPOPCNTQ + NEON VCNT) | Foundation for B1 rank/select, B3 Roaring cardinality, vector hamming/jaccard. Smallest API; 1-2 days. | [`10_BITS.md`](10_BITS.md) § 4 |
| **A3** | `hash::sha256_batch_st` (+ userspace `blake3_batch_*` if blake3 feature on) | Unblocks `tokenfs_writer` build-time Merkle work; 200K-extent Merkle leaf hashing. Real production win. 2-3 days. | [`12_HASH_BATCHED.md`](12_HASH_BATCHED.md) § 2 |

**v0.1.x ship gate:** both primitives have scalar oracle + at least one SIMD backend with parity tests, dispatched via existing `dispatch::` infrastructure, with criterion benches showing measurable speedup over scalar. Ship as v0.1.x patch release.

### v0.2 Phase A continuation (~2 weeks solo)

The remaining Phase A primitives, dependency-independent of each other and of v0.1.x's A1/A3:

| # | Module / primitive | Doc | Estimated complexity | Kernel-safe? |
|---|---|---|---|---|
| A2 | `bits::bit_pack` (arbitrary widths 1-32) | [`10_BITS.md`](10_BITS.md) § 2 | medium (3-5 days) | ✅ |
| A4 | `hash::set_membership_simd` | [`12_HASH_BATCHED.md`](12_HASH_BATCHED.md) § 3 | small (1-2 days) | ✅ |
| A5 | `vector::distance` (5 metrics × scalar/AVX2/AVX-512/NEON) | [`13_VECTOR.md`](13_VECTOR.md) | medium (5-7 days) | ✅ |

**v0.2 Phase A ship gate:** A2 + A4 + A5 each have scalar oracle + at least one SIMD backend with parity tests, dispatched via existing `dispatch::` infrastructure, with criterion benches showing measurable speedup over scalar on representative inputs.

## Phase B — high-leverage primitives that build on Phase A

| # | Module / primitive | Depends on | Doc | Estimated complexity | Kernel-safe? |
|---|---|---|---|---|---|
| B1 | `bits::rank_select` | A1 popcount | [`10_BITS.md`](10_BITS.md) § 5 | **large (1-2 weeks)** — superblock + block + sample tables; multiple variants (RRR / Vigna broadword) | ✅ (queries); ⚠️ build allocates |
| B2 | `bits::streamvbyte` (encode + decode) | A2 bit_pack scaffolding | [`10_BITS.md`](10_BITS.md) § 3 | medium (4-6 days) | ✅ |
| B3 | `bitmap::roaring` (intersection/union/difference/cardinality) | A1 popcount | [`11_BITMAP.md`](11_BITMAP.md) | large (1-2 weeks) — pair-dispatch table × backends | ✅ (caller-provided output) |
| B4 | `permutation::rcm` (Reverse Cuthill-McKee) | none (A1 popcount used by argsort optionally) | [`14_PERMUTATION.md`](14_PERMUTATION.md) § 2 | small (3-4 days) | ❌ build-time only |
| B5 | `permutation::hilbert` (2D/N-D Hilbert curve sort) | none | [`14_PERMUTATION.md`](14_PERMUTATION.md) § 4 | small (1-2 days) | ❌ build-time only |

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

If solo (revised given milestone split):
- **v0.1.x ship**: A1 popcount → A3 batched hash. ~1 week. Two production wins.
- **v0.2 ship**: A2 bit_pack → A4 set_membership → A5 vector distance → B2 streamvbyte → B3 Roaring → B4 RCM → B5 Hilbert → B1 rank_select. **B1 sits last because it's the longest single piece (1-2 weeks)** and unblocks Tier-D items that aren't on the v0.2 path anyway.
- **Phase C**: composition demonstrators after relevant Bs.
- **Phase D**: D1 Rabbit Order opportunistically.

If two contributors:
- Track 1: A1 popcount → A3 batched hash (v0.1.x) → A2 bit_pack → B2 streamvbyte → B1 rank_select → C1
- Track 2: A4 → A5 → B3 → C3 (in parallel) → B4 → B5 → C2 → D1

Both tracks meet at Phase C demonstrators. Track 2 gets the heavy B3 (Roaring) so Track 1 can focus on the bits-module sequence ending in rank_select.

## Total time estimate (solo)

Honest budget given complexity revisions per critique:
- v0.1.x stopgap: ~1 week.
- v0.2 Phase A continuation (A2+A4+A5): ~2 weeks.
- v0.2 Phase B (B1-B5, with B1 and B3 being the long poles): ~5-6 weeks.
- Phase C demonstrators: ~1 week.
- **Total v0.1.x → v0.2 release: ~9-10 weeks solo.**

This is honest scoping. With two contributors, halve. With opportunistic Phase D Rabbit Order, add 2-4 weeks.

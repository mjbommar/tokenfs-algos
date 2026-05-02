# v0.2 planning

**Status:** working plan, 2026-05-02. Targets the v0.2 milestone of `tokenfs-algos` for **the full near-term consumer surface** — TokenFS (`tokenfs-paper`) FUSE/kernel/build, plus Postgres extensions, MinIO/Go services via cgo, CDN edge caches, forensics tools, Python research bindings.

This folder is a planning workbench, not a contract. Each doc is intended to be revised as primitives land and assumptions are validated against real benchmarks.

## How this folder is organized

| Doc | Purpose |
|---|---|
| [`00_BOTTOM_UP_ANALYSIS.md`](00_BOTTOM_UP_ANALYSIS.md) | Why the prior `FS_PRIMITIVES_GAP.md` ranking is wrong, what the actual binding constraints are, what the corrected ranking is. |
| [`01_PHASES.md`](01_PHASES.md) | Phase A → D ship order, dependency graph between modules, gating criteria. |
| [`02_CACHE_RESIDENCY.md`](02_CACHE_RESIDENCY.md) | Per-table cache-tier analysis across consumer scales (TokenFS image, Postgres column, CDN edge). |
| [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md) | Consumer environment × constraint matrix (kernel module, FUSE, Postgres, cgo, CDN edge, forensics, Python). |
| [`10_BITS.md`](10_BITS.md) | New `bits` module: bit-pack/unpack at arbitrary widths, Stream-VByte codec, bit-rank/select dictionary. |
| [`11_BITMAP.md`](11_BITMAP.md) | New `bitmap` module: Roaring-style SIMD container kernels. |
| [`12_HASH_BATCHED.md`](12_HASH_BATCHED.md) | Additions to `hash`: batched BLAKE3 + SHA-256 wrappers, hash-set membership SIMD. |
| [`13_VECTOR.md`](13_VECTOR.md) | New `vector` module: dense distance kernels (L2², dot, cosine, Hamming, Jaccard). |
| [`14_PERMUTATION.md`](14_PERMUTATION.md) | New `permutation` module: locality-improving orderings (RCM, Rabbit-order, Hilbert). The missing-from-prior-docs primitive class. |
| [`20_DEFERRED.md`](20_DEFERRED.md) | Items intentionally deferred to v0.3+, each with deferral rationale and trigger condition (across all consumer environments). |

## Reading order

If you have 5 minutes: read [`00_BOTTOM_UP_ANALYSIS.md`](00_BOTTOM_UP_ANALYSIS.md), the dependency graph at the top of [`01_PHASES.md`](01_PHASES.md), and skim the matrix in [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md). That tells you what's being built, why, and for whom.

If you have 30 minutes: also read [`02_CACHE_RESIDENCY.md`](02_CACHE_RESIDENCY.md) and the module doc for whichever module you're about to touch.

If you're committing to a phase: read the relevant module doc end-to-end (each ends with an "Environment fitness" section mapping its API surface to `02b_DEPLOYMENT_MATRIX.md`). Each spec covers algorithm, API surface sketch, per-backend SIMD plan, test plan, bench plan, known risks, and consumer-fit notes.

## Framing principle

Every primitive in this plan must justify its existence by answering four questions:

1. **What query/workload binds on this primitive being slow?** If the answer is hypothetical for *every* consumer in `02b_DEPLOYMENT_MATRIX.md`, the primitive is in `20_DEFERRED.md`.
2. **What's the cache-residency picture for the data this primitive operates on?** Cache-tier conclusions vary across consumers (TokenFS metadata fits in L3; Postgres indexes don't). L1-resident kernels can be branchy; DRAM-resident kernels must be bandwidth-aware.
3. **Which consumer environments can this primitive operate in?** Per `02b_DEPLOYMENT_MATRIX.md`, kernel-module use forbids rayon, blake3, large stack scratch; cgo-bridged use mandates batch APIs. Some primitives are kernel-safe (Tier-A `bits`, most of `bitmap`, `vector`); others are inherently userspace (`permutation`).
4. **What's the consumer surface — one consumer or many?** Single-consumer primitives ride in their consumer's module. Multi-consumer primitives get a dedicated module.

The Tier-A picks (`bits`, `bitmap`, `hash` batched additions, `vector`) all have multiple distinct consumers across multiple environments. The Tier-D items in `20_DEFERRED.md` typically fail one of these four tests for every documented consumer.

## Relationship to existing docs

This plan **supersedes the v0.2 + v0.3 ship lists in `FS_PRIMITIVES_GAP.md` § 4** for the reasons in `00_BOTTOM_UP_ANALYSIS.md`. The v1 manifest layout in `tokenfs-paper/docs/data-structures.md` § 7 is the consumer specification this plan supports.

The existing `PRIMITIVE_KERNEL_BUFFET.md`, `PRIMITIVE_CONTRACTS.md`, `PROCESSOR_AWARE_DISPATCH.md`, and `PLANNER_DESIGN.md` define the engineering conventions every new module here must follow (target_feature gating, runtime detection, scalar oracle, fallible constructors, dispatch metadata). These conventions are inputs to this plan, not subjects of it.

## What this plan is NOT

- Not a v1 paper outline. The paper claims live in `tokenfs-paper`.
- Not an image-format spec. That's `tokenfs-paper/docs/data-structures.md`.
- Not a calendar / Gantt chart. Phase boundaries are dependency-driven, not date-driven. The order the phases ship is fixed by their dependency graph; the rate they ship at is the contributor's choice.
- Not a commitment to vendor any specific external crate. Each module doc lists external crate options; vendoring is a per-primitive review.

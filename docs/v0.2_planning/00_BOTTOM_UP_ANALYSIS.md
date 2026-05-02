# Bottom-up analysis: what actually binds, what to build

**Status:** rationale doc, 2026-05-02. Companion to `README.md`. Argues why the prior `FS_PRIMITIVES_GAP.md` ranking should be reorganized.

## TL;DR

Current `FS_PRIMITIVES_GAP.md` ranks tier-1 primitives by "what fits in the v1 manifest layout." That's the *consumer's* perspective. If you instead ask "what physical bottleneck does each primitive remove, ranked by hardware cost per bottleneck," the ordering changes substantially. This doc walks the analysis layer by layer (hardware → cache hierarchy → workloads → primitives) and lands on a different tier-A.

## Layer 0 — what the hardware actually does

| Operation | Throughput / latency on a modern x86 P-core |
|---|---|
| DRAM streaming read | ~30 GB/s/core (DDR5-6400) |
| L3 random access | ~30 ns per cache line |
| L2 random access | ~12 ns |
| L1 random access | ~4 ns |
| DRAM cache miss | ~80 ns |
| AVX2 register op (in L1) | 1 cycle / 32 B = ~100+ GB/s |
| AVX2 op with L2 loads | ~30 GB/s (L2 bandwidth-capped) |
| AVX-512 register op | 1 cycle / 64 B |
| AVX-512 `VPOPCNTQ` | 1 cycle / 64 B (vs 4-cycle software popcount = ~10× speedup) |
| AVX-512 `VPCONFLICTD` | parallel histogram in 4-6 cycles for 16 lanes |
| AVX-512 `VPCOMPRESSQ` | branch-free pack/compress |
| Page fault (cold mmap) | ~5-200 µs depending on prefetch |

**The critical observation:** for kernels operating on already-mapped data, **memory bandwidth is the wall, not compute.** SIMD helps because it amortizes per-byte instruction overhead, not because the kernel is compute-bound. The right design objective is **minimize bytes touched per logical query**, not "go fast on the bytes you do touch." The prior docs talk about throughput (GB/s) but never make bytes-touched-per-query first-class.

## Layer 1 — what consumers actually move

The crate's named consumers (per `_references/README.md` and `AGENTS.md`) span multiple environments with different table cardinalities. See [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md) for the full constraint matrix.

### Workload examples by consumer

#### TokenFS image reader (FUSE, kernel module, build pipeline)

Data shapes from `tokenfs-paper/docs/data-structures.md`:

1. **Extent payloads** — variable-length (1KB–1MB), content-addressed.
2. **Inode records** — fixed ~64B each, indexed by inode number.
3. **Path strings** — variable-length, in a giant FST or sorted block.
4. **Token streams** — packed bit-fields of u11/u12/u16 token IDs per extent.
5. **Fingerprints** — 32B/extent, F22 cluster signature.
6. **Inverted-index posting lists** — sorted u32 extent IDs, Stream-VByte/Roaring encoded.

#### Postgres extension consumer (pgvector-class, GIN-bitmap-class)

1. **Index tuples** — 100M-1B rows × 16-32B index entries = GB-scale; never fits cache.
2. **Bitmap-scan posting lists** — Roaring-encoded result sets; mid-scan sizes 100K-10M elements.
3. **Vector embeddings** — pgvector-class, ~768D f32 ~3KB each; millions of rows.

#### MinIO / Go service via cgo

1. **Object metadata** — millions of objects × ~256B per metadata record = GB.
2. **Content fingerprints** — N × 32B SHA-256 entries; dedup pre-checks via Bloom.
3. **Cross-tenant analytics** — TB-scale aggregates.

#### CDN edge cache

1. **Per-edge cache directory** — ~10M entries at typical edge.
2. **MinHash similarity sketches** — for cross-edge dedup; ~256B per object.
3. **Bloom filter pre-checks** — sub-µs membership at request handling time.

### Cardinality / cache-residency comparison

| Workload | Hot working set | Cache tier (modern x86) |
|---|---|---|
| TokenFS-rootfs hot metadata (~130K paths, ~228K extents) | **~26 MB** | **L3** (fits in 32 MB) |
| TokenFS-large image (10M extents) | ~640 MB | DRAM |
| Postgres GIN bitmap-scan working set | 10s of MB to GB | L3-DRAM border |
| Postgres pgvector ~768D × 1M embeddings | ~3 GB | DRAM |
| MinIO metadata table (10M objects) | ~2.5 GB | DRAM |
| CDN edge cache directory (10M entries × 32B) | ~320 MB | DRAM |

**The TokenFS-image hot-metadata-fits-in-L3 fact** (the single most important observation that prior `data-structures.md` and `FS_PRIMITIVES_GAP.md` miss for TokenFS's typical scale) **does NOT generalize**. For Postgres-class or MinIO-class workloads, nothing fits in L3 and bandwidth dominates differently.

For TokenFS specifically, this means:
- "Save bytes in inode table" optimizations are wasted effort. The table fits anyway.
- The right design pressure is **"what query patterns can be answered without leaving L3?"**
- Random gathers within L3 cost ~30 ns each. SIMD doesn't help gathers; only locality does.

For Postgres / MinIO / CDN-class consumers:
- **Compression matters** — saving 30% of a 3 GB embedding table saves 1 GB of RAM that's otherwise pageable.
- **Streaming bandwidth dominates** at the DRAM tier (~30 GB/s/core); SIMD that doubles per-byte throughput halves wall time.
- **Locality permutations** still matter, but at a different granularity (page-level, not cache-line-level).

The same primitives serve both regimes; the **bench harness should report numbers at multiple working-set sizes** (in-L1, in-L2, in-L3, in-DRAM) so consumers can pick the right operating point. See [`02_CACHE_RESIDENCY.md`](02_CACHE_RESIDENCY.md) for the bench-axis convention.

## Layer 2 — what queries bind on what primitive

Cross-consumer mapping: same primitive families tend to serve multiple consumer environments, but the bottleneck shape differs.

### TokenFS reader paths

**Hot path (FUSE / kernel module runtime):**
- `lookup(path) → inode` — Path FST traversal in `fst` crate. Already optimized. Nothing for `tokenfs-algos` to do.
- `read(inode, offset, length) → bytes` — inode lookup + extent map binary search + payload mmap. Cache-miss-bound (~3-5 misses warm, ~80 ns each = ~few µs). Pure compute primitives don't help. **Layout primitives** (CSR co-location, locality-improving permutations) help.

**Cold path (analytics / AI ingestion / build pipeline):**
- `intersect token A AND token B posting lists` — Roaring container ops on bitmap×bitmap, array×array, array×bitmap pairs.
- `decode tokens for extent` — bit-unpack u11/u12 packed stream.
- `find files like X` — MinHash brute force OR vector distance over fingerprints.
- `verify image Merkle root` — hash all 200K extent leaves.
- `is content fingerprint Y in this image?` — Bloom pre-check, then content-addressed table lookup.

### Postgres extension paths

- `bitmap_scan(index)` predicate — Roaring-style intersection of two posting lists per WHERE clause.
- `pgvector(query, k)` ANN search — vector distance kernel applied to N database vectors.
- `bloom_index_check(value)` — Bloom membership pre-check before heap scan.
- `approx_count_distinct(column)` — HyperLogLog merge across partitions.

### MinIO/CDN paths

- `is_duplicate(content_hash)` — Bloom pre-check then content-addressed lookup.
- `find_similar(object)` — MinHash-LSH band-and-hash, vector distance for re-rank.
- `verify_etag(content)` — batched SHA-256 / BLAKE3.
- `dedup_metadata` — Roaring set intersection on per-tenant fingerprint sets.

### Workload → primitive mapping

| Workload | Real bottleneck | Right primitive | Consumers |
|---|---|---|---|
| `read(inode, off)` warm | log-N cache misses through extent map | Layout (CSR co-loc) — *not* SIMD | TokenFS |
| `read(inode, off)` cold | page faults (~80 µs) | Prefetch — *not* SIMD | TokenFS |
| `lookup(path)` | FST traversal in `fst` crate | *Nothing for us to add* | TokenFS |
| Build-time Merkle | hashing N small extents | **Batched BLAKE3 / SHA-256** | TokenFS, MinIO, CDN |
| Build-time CAS dedup | SHA-256 over payloads | **Batched SHA-256** | TokenFS, MinIO |
| Posting-list intersect | Roaring container ops | **Roaring SIMD** | TokenFS, Postgres, MinIO |
| Posting-list payload | Stream-VByte decode | **Stream-VByte SIMD** | TokenFS, Postgres, columnar DBs |
| Token decode for scan | bit-unpack u11/u12 stream | **Bit-pack/unpack SIMD** | TokenFS, columnar DBs |
| Similarity rank | dot / L2 / cosine over vectors | **Dense vector distances** | TokenFS, Postgres pgvector, MinIO, CDN |
| Vocab/set lookup | hash-set membership | **Hash-set membership SIMD** | TokenFS, Postgres, all CASes |
| Image-build layout | min-bandwidth permutation | **`permutation` (RCM/Rabbit/Hilbert)** | TokenFS only (build-time) |
| Bloom pre-check | per-element bit-test | **Bloom SIMD** (deferred — see `20_DEFERRED.md`) | Postgres, MinIO, CDN |
| Approx distinct rollup | HLL merge | **HyperLogLog SIMD merge** (deferred) | Postgres, OLAP DBs |
| MinHash signature | windowed hash | **MinHash SIMD** (deferred) | TokenFS, CDN edge dedup |

## Layer 3 — what the prior docs got wrong

`FS_PRIMITIVES_GAP.md` § 3-4 has these specific issues:

1. **F22 fingerprint sidecar listed as Tier-1 #1.** F22 is done. Listing finished work as priority work hides the real queue.
2. **Path FST listed Tier-1 #2 with "None we should build."** Don't list non-work as work.
3. **Tier 1 mixes structures (F22, Path FST, CSR adjacency) with primitives (Roaring, CHD).** Inconsistent abstraction level. Bottom-up: only primitives belong in `tokenfs-algos`'s queue. Structures are `tokenfs-paper` compositions.
4. **Stream-VByte and bit-unpack listed as dependencies of #3 but not as Tier-1 work themselves.** They're foundational for *three* unrelated consumers (posting lists, token streams, succinct DS). Move to Tier-A.
5. **Dense vector distance kernels ranked Tier 3 "for HNSW".** They're the inner loop of *any* vector similarity work — MinHash signature distance, F22 fingerprint comparison, brute-force ANN, **and Postgres pgvector**. Universal kernel mis-classified as speculative.
6. **`xxhash3 / wyhash SIMD` listed v0.2.** Premature. Current FNV/CRC32C path hasn't shown a throughput problem in any documented bench. Don't add a hash family without a documented bottleneck.
7. **Cache-residency analysis missing entirely.** The single most important design pressure for the 22 MB TokenFS working set isn't mentioned. Multi-consumer cache regimes aren't either.
8. **Physical-layout / permutation primitives missing.** TokenFS's read-only nature unlocks "given similarity graph + dedup hypergraph, emit inode ordering that minimizes spatial random access." This is pure-compute over u32/u64 arrays — clean `tokenfs-algos` material. **Not in either prior doc.** Adding `permutation` module here.
9. **Token-decoder throughput barely mentioned.** Every read of a tokenized extent runs the decoder. If decoder is 1 GB/s vs raw bytes at 30 GB/s, tokens are a slow path. The SIMD primitives that make decode fast (bit-unpack, table-lookup gather) are universal `tokenfs-algos` material — but `FS_PRIMITIVES_GAP.md` puts decode in `bbpe`'s lap and never asks what primitives make it fast.
10. **HNSW for 6-D fingerprints stays on the list** even though `data-structures.md` open question #5 admits this is unjustified at our scale. Cut. (Reconsider for Postgres pgvector consumers separately — different scale.)
11. **Multi-consumer constraints missing.** No mention of kernel module / FUSE / Postgres / cgo deployment differences. The `blake3_batch` spec used `rayon`, which is forbidden in kernel modules. Caller-provided-scratch pattern (§77 lesson) needed across more APIs than I initially called out. See [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md).

## Layer 4 — corrected ranking

**Tier A — load-bearing for both ingest and runtime, multiple consumers:**

| # | Primitive | Module | Consumers |
|---|---|---|---|
| 1 | bit-pack / bit-unpack / Stream-VByte / bit-rank-select | `bits` (new) | posting lists; token streams; succinct DS |
| 2 | batched BLAKE3 + SHA-256 | `hash` (extension) | Merkle integrity; CAS dedup; reproducibility |
| 3 | Roaring SIMD set ops (intersect/union/diff/cardinality) | `bitmap` (new) | every analytics query passes through this |

**Tier B — high-leverage, narrower scope:**

| # | Primitive | Module | Why |
|---|---|---|---|
| 4 | dense vector distance kernels (L2², dot, cosine, Hamming, Jaccard) | `vector` (new) | inner loop of any vector similarity work |
| 5 | hash-set membership SIMD (`is x in 256-element set?`) | `hash` (extension) | vocab lookup, content-class membership, Bloom pre-check |
| 6 | locality-improving permutation (RCM + Rabbit-order + Hilbert) | `permutation` (new) | read-only image's once-and-forever layout primitive; makes EVERYTHING faster downstream |

**Tier C — composition-driven, build when consumer demands:**

| # | Primitive | When to build |
|---|---|---|
| 7 | CSR walk + BFS frontier | when a real graph traversal hot path appears |
| 8 | MinHash SIMD signature kernel | when "find similar files" becomes a primary query (currently isn't) |
| 9 | CHD batched lookup | when CHD is in active use in `tokenfs-paper` |
| 10 | Levenshtein / Hamming SIMD-DP | when path-typo tolerance becomes a feature |

**Tier D — defer until a justifying signal exists:**

See [`20_DEFERRED.md`](20_DEFERRED.md). Items: wavelet tree, FM-index, HNSW, vectorized DFA, top-K SIMD heap, xxh3/wyhash SIMD, FFT/NTT, Reed-Solomon. Each has a deferral rationale and a documented trigger condition that would un-defer it.

## What this means for v0.2

The v0.2 ship target becomes: **`bits` + `bitmap` + batched-hash + `vector` + `permutation`**, in roughly that dependency order (see `01_PHASES.md`). This is 5 modules' worth of work, not 14. Each module has a small enough surface to land cleanly in 1-2 weeks of focused implementation; together they:
- Unblock essentially every Tier-1 + Tier-2 structure in the v1 TokenFS manifest layout.
- Provide a Roaring-SIMD path that's the missing piece for both TokenFS analytics AND Postgres GIN-bitmap-scan extension consumers.
- Provide vector-distance kernels that are pgvector-equivalent (and could become an upstream contribution).
- Provide batched cryptographic hash that fits both TokenFS Merkle and MinIO/CDN content-addressing.

Each module spec ends with an "Environment fitness" section per `02b_DEPLOYMENT_MATRIX.md`, calling out which APIs are kernel-safe, which require std/rayon, and which are inherently userspace.

Tier C items wait for documented consumer demand from **any** environment (not just TokenFS). Tier D items wait for a documented bottleneck. This is the discipline that keeps the crate's surface area aligned with what consumers actually bind on, instead of growing speculatively.

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

## Layer 1 — what TokenFS actually moves

The data shapes from `tokenfs-paper/docs/data-structures.md`:

1. **Extent payloads** — variable-length (1KB–1MB), content-addressed. Read pattern: "give me extent N's bytes." Cold-cache disk-bound; warm-cache mmap-bound.
2. **Inode records** — fixed ~64B each, indexed by inode number. Read: "give me inode N." 
3. **Path strings** — variable-length, in a giant FST or sorted block. Read: "is this path here, what's its inode?"
4. **Token streams** — packed bit-fields of u11/u12/u16 token IDs per extent. Read: "scan tokens for n-gram match" or "decode tokens to bytes."
5. **Fingerprints** — 32B/extent, F22 cluster signature. Read: "scan all fingerprints to find similar to this one."
6. **Inverted-index posting lists** — sorted u32 extent IDs, Stream-VByte/Roaring encoded. Read: "intersect posting list of token A with posting list of token B."

The **cardinalities** for an Ubuntu-rootfs class image (~130K paths, ~228K extents, ~2K vocab):

| Table | Cardinality × per-entry size | Total | Cache tier |
|---|---|---|---|
| Vocab table | ~2K × ~16B avg | ~32 KB | **L1** (32 KB) |
| Inode table | ~130K × 64B | ~8 MB | **L2/L3** |
| Extent metadata sidecar | ~228K × 32B | ~7 MB | **L2/L3** |
| Fingerprint array (F22) | ~228K × 32B | ~7 MB | **L2/L3** |
| **Hot metadata combined** | | **~22 MB** | **fits in 32 MB L3** |
| Token n-gram inverted index | varies | MB-GB | DRAM |
| Extent payloads | varies | GB | mmap / disk |

**The single most important fact in either prior doc, that neither states:** the entire hot-metadata working set fits in L3. A full metadata scan is ~10 ms warm. This means:

- "Save bytes in inode table" optimizations are wasted effort. The table fits anyway.
- The right design pressure is **"what query patterns can be answered without leaving L3?"**
- Random gathers within L3 cost ~30 ns each. SIMD doesn't help gathers; only locality does.

## Layer 2 — what queries bind on what primitive

Two traffic patterns:

**Hot path (FUSE-mounted runtime):**
- `lookup(path) → inode` — Path FST traversal in `fst` crate. Already optimized. Nothing for `tokenfs-algos` to do.
- `read(inode, offset, length) → bytes` — inode lookup + extent map binary search + payload mmap. Cache-miss-bound (~3-5 misses warm, ~80 ns each = ~few µs). Pure compute primitives don't help. **Layout primitives** (CSR co-location, locality-improving permutations) help.

**Cold path (analytics / AI ingestion):**
- `intersect token A AND token B posting lists` — Roaring container ops on bitmap×bitmap, array×array, array×bitmap pairs. **THIS is where SIMD pays off.** Documented 30-60 GB/s on AVX-512 for dense intersections.
- `decode tokens for extent` — bit-unpack u11/u12 packed stream. Every read of a tokenized extent runs this. Throughput dictates whether tokens are a fast or slow path.
- `find files like X` (MinHash brute force) — N × hash budget. SIMD MinHash 5-10 GB/s on AVX-512.
- `verify image Merkle root` — hash all 200K extent leaves. Batched BLAKE3 wrapper unblocks this.
- `is content fingerprint Y in this image?` — Bloom pre-check, then content-addressed table lookup.

| Workload | Real bottleneck | Right primitive |
|---|---|---|
| `read(inode, off)` warm | log-N cache misses through extent map | Layout (CSR co-loc) — *not* SIMD |
| `read(inode, off)` cold | page faults (~80 µs) | Prefetch — *not* SIMD |
| `lookup(path)` | FST traversal in `fst` crate | *Nothing for us to add* |
| Build-time Merkle | hashing 200K small extents | **Batched BLAKE3** |
| Build-time CAS dedup | SHA-256 over extent payloads | **Batched SHA-256** |
| Analytics: posting list intersect | Roaring container ops | **Roaring SIMD** |
| Analytics: posting list payload | Stream-VByte decode | **Stream-VByte SIMD** |
| Analytics: token decode for scan | bit-unpack u11/u12 stream | **Bit-pack/unpack SIMD** |
| Analytics: similarity rank | dot / L2 / cosine over fingerprints | **Dense vector distances** |
| Analytics: vocab lookup | hash-set membership in 2K entries | **Hash-set membership SIMD** |

## Layer 3 — what the prior docs got wrong

`FS_PRIMITIVES_GAP.md` § 3-4 has these specific issues:

1. **F22 fingerprint sidecar listed as Tier-1 #1.** F22 is done. Listing finished work as priority work hides the real queue.
2. **Path FST listed Tier-1 #2 with "None we should build."** Don't list non-work as work.
3. **Tier 1 mixes structures (F22, Path FST, CSR adjacency) with primitives (Roaring, CHD).** Inconsistent abstraction level. Bottom-up: only primitives belong in `tokenfs-algos`'s queue. Structures are `tokenfs-paper` compositions.
4. **Stream-VByte and bit-unpack listed as dependencies of #3 but not as Tier-1 work themselves.** They're foundational for *three* unrelated consumers (posting lists, token streams, succinct DS). Move to Tier-A.
5. **Dense vector distance kernels ranked Tier 3 "for HNSW".** They're the inner loop of *any* vector similarity work — MinHash signature distance, F22 fingerprint comparison, brute-force ANN. Universal kernel mis-classified as speculative.
6. **`xxhash3 / wyhash SIMD` listed v0.2.** Premature. Current FNV/CRC32C path hasn't shown a throughput problem in any documented bench. Don't add a hash family without a documented bottleneck.
7. **Cache-residency analysis missing entirely.** The single most important design pressure for a 22 MB working set isn't mentioned.
8. **Physical-layout / permutation primitives missing.** TokenFS's read-only nature unlocks "given similarity graph + dedup hypergraph, emit inode ordering that minimizes spatial random access." This is pure-compute over u32/u64 arrays — clean `tokenfs-algos` material. **Not in either prior doc.** Adding `permutation` module here.
9. **Token-decoder throughput barely mentioned.** Every read of a tokenized extent runs the decoder. If decoder is 1 GB/s vs raw bytes at 30 GB/s, tokens are a slow path. The SIMD primitives that make decode fast (bit-unpack, table-lookup gather) are universal `tokenfs-algos` material — but `FS_PRIMITIVES_GAP.md` puts decode in `bbpe`'s lap and never asks what primitives make it fast.
10. **HNSW for 6-D fingerprints stays on the list** even though `data-structures.md` open question #5 admits this is unjustified at our scale. Cut.

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

The v0.2 ship target becomes: **`bits` + `bitmap` + batched-hash + `vector` + `permutation`**, in roughly that dependency order (see `01_PHASES.md`). This is 5 modules' worth of work, not 14. Each module has a small enough surface to land cleanly in 1-2 weeks of focused implementation; together they unblock essentially every Tier-1 + Tier-2 structure in the v1 manifest layout.

Tier C items wait for documented consumer demand. Tier D items wait for a documented bottleneck. This is the discipline that keeps the crate's surface area aligned with what the consumers actually bind on, instead of growing speculatively.

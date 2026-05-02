# Deferred items

**Status:** intentional-deferral list, 2026-05-02. Items in `FS_PRIMITIVES_GAP.md` v0.2/v0.3/v0.4 lists that don't make the v0.2 cut, with rationale and trigger condition.

The deferral discipline: each item lists (a) **why deferred**, (b) **what would un-defer it** (a documented signal, not a guess).

## Tier C — composition-driven, deferred until consumer asks

### CSR walk + parallel BFS frontier

- **Why deferred**: Pointer-chase-bound. The `permutation` module gives the layout that makes CSR walks fast scalarly first; SIMD on the walk itself is incremental and harder to justify. None of the v1 manifest queries (`tokenfs-paper/docs/data-structures.md` § 4) currently list a graph-traversal hot path that can't be served by good layout + scalar BFS.
- **Trigger to un-defer**: a documented `tokenfs-paper` query that profiles bottlenecking on raw BFS/DFS throughput, not on the surrounding work. (Most likely candidate: live `find -type f -name '*.so' | xargs grep` analogue at FUSE layer — but that's I/O bound on `read()`, not BFS bound.)
- **If un-deferred**: lands as `graph` module (Phase C+).

### MinHash SIMD signature kernel

- **Why deferred**: Existing `similarity::minhash` is scalar but works. The SIMD win is real (~5x on AVX-512) but only matters when "compute MinHash sig over many byte windows in parallel" becomes a hot path. Currently isn't.
- **Trigger to un-defer**: "find files like X" or "image-wide near-dup detection" becomes a primary query. Or: image-build-time MinHash sidecar generation profiles bottlenecking.
- **If un-deferred**: extends `similarity::minhash` with `kernels::{avx2,avx512,neon}` modules. ~3-5 days of work.

### CHD batched lookup

- **Why deferred**: CHD itself is in `tokenfs-paper`'s consumption (`boomphf` crate or similar). The batched-lookup primitive only matters once CHD is in active use AND profiling shows CHD lookup is hot (it almost certainly won't, since CHD is a single-cache-line probe).
- **Trigger to un-defer**: CHD lookup on the FUSE hot path measured at >5% of total `lookup(path)` time.
- **If un-deferred**: small kernel in `hash::chd_lookup_batch`.

### Levenshtein / Hamming SIMD-DP

- **Why deferred**: No documented use case. Path-typo-tolerance for shell completion is conceivable but not on the v1 spec.
- **Trigger to un-defer**: a shell-completion or fuzzy-search consumer asks for it.
- **If un-deferred**: new `string` module. AVX2 16-lane SIMD-DP.

### HyperLogLog SIMD merge

- **Why deferred**: Existing `approx::HyperLogLog` is scalar but works. SIMD merge only matters when merging many HLL sketches in tight loops — which is only Image Build Pipeline aggregation, not a hot path.
- **Trigger to un-defer**: image build profiling shows HLL merge bottlenecking.
- **If un-deferred**: extends `approx::HyperLogLog` with merge kernel.

### Bloom / Cuckoo SIMD insert + query

- **Why deferred**: Existing `approx::BloomFilter` is scalar. SIMD on a Bloom probe is ~50 GB/s on AVX-512 vs ~5 GB/s scalar — real win — but Bloom probes are typically L1-resident few-cache-line operations where the constant factors don't dominate.
- **Trigger to un-defer**: a "bulk content-fingerprint pre-check across many images" workload.
- **If un-deferred**: extends `approx::BloomFilter`.

### Top-K via SIMD heap

- **Why deferred**: Existing `approx::SpaceSaving` and Misra-Gries cover heavy-hitters. Generic top-K isn't hot anywhere documented. Branch-free small-array sort (≤ 64 elements via AVX-512 VPCOMPRESSQ) is a clean SIMD win, but no consumer.
- **Trigger to un-defer**: a "top-K nearest-neighbor" (i.e., HNSW-adjacent) consumer asks.
- **If un-deferred**: new `sort` module.

### `xxh3` / `wyhash` SIMD batch

- **Why deferred**: No documented bottleneck on existing FNV-1a / mix64 / CRC32C. The existing benches don't show the hash families as the bottleneck on any hot path.
- **Trigger to un-defer**: a benchmark or consumer demonstrates the existing hash family is bottlenecking on >5% of total runtime.
- **If un-deferred**: extends `hash` module.

## Tier D — speculative, no near-term consumer

### Wavelet tree on token stream

- **Why deferred**: Beautiful succinct DS but no v1 query asks for `rank(token X, position i)` or `select(token X, k-th occurrence)`. The v1 inverted index covers token presence; the wavelet tree adds positional queries that aren't yet on the spec.
- **Foundation needed**: `bits::rank_select` (Phase B1). Once that lands, the wavelet tree itself is ~500 lines of glue.
- **Trigger to un-defer**: a token-stream positional-query consumer (`grep`-class, "find the third occurrence of token X in this extent").

### FM-index on token stream

- **Why deferred**: Same as wavelet tree but with substring search. Higher build cost; needs suffix array or BWT scaffolding.
- **Foundation needed**: wavelet tree (above) + Burrows-Wheeler transform.
- **Trigger to un-defer**: a "token n-gram backward search" consumer.

### HNSW over fingerprints

- **Why deferred**: F22 fingerprints are 6-D. Brute-force nearest-neighbor over 228K points is sub-second on AVX-512 (~250 µs per query, see `02_CACHE_RESIDENCY.md`). HNSW only pays off above ~20-D or millions of points.
- **Trigger to un-defer**: TokenFS adds learned embeddings (typical 256-768-D). Reconsider then.

### Vectorized regex DFA

- **Why deferred**: `aho-corasick` covers the immediate "byte-level multi-pattern search" need. Hyperscan-class vectorized DFA would extend this to richer regex classes, but no consumer asks.
- **Trigger to un-defer**: a feature-flagged regex engine in `tokenfs-paper`.

### FFT / NTT

- **Why deferred**: No use case. Unless content fingerprinting moves toward signal-processing-style summaries (it hasn't), FFT isn't a TokenFS primitive.
- **Trigger to un-defer**: signal-processing-class fingerprint research.

### Reed-Solomon erasure coding

- **Why deferred**: Useful for distributed/replicated TokenFS but not load-bearing for v1 (which is single-image read-only).
- **Trigger to un-defer**: distributed/replicated TokenFS deployment scenario.

### Sparse-bitmap segment-tree / Fenwick-tree

- **Why deferred**: Roaring covers most range-query needs. Fenwick is a different shape (range-sum) but no consumer asks.
- **Trigger to un-defer**: a range-sum / prefix-count consumer.

## Tier "won't ever ship in `tokenfs-algos`"

These are out of scope by the crate's charter (`tokenfs-algos` is content-agnostic byte-slice compute):

- **Path FST**: lives in `tokenfs-paper`, depends on `fst` crate. Per `00_BOTTOM_UP_ANALYSIS.md` §3 #2, the prior doc's listing of this as a Tier-1 item was a category error.
- **Manifest layout / section table / inode table**: `tokenfs-paper` layer.
- **Tokenizer (BPE / vocab build)**: `bbpe` crate, not algos.
- **Tokenfs FUSE adapter**: `tokenfs-paper` layer.

## Re-ranking summary

For ease of comparison: the v0.2 list from `FS_PRIMITIVES_GAP.md` § 4, mapped to this plan:

| `FS_PRIMITIVES_GAP.md` v0.2 item | Status here |
|---|---|
| Stream-VByte codec | **Phase B2** (`bits::streamvbyte`) — kept, in `10_BITS.md` |
| Roaring SIMD kernels | **Phase B3** (`bitmap`) — kept, in `11_BITMAP.md` |
| Bit-rank/bit-select dictionary | **Phase B1** (`bits::rank_select`) — kept, in `10_BITS.md` |
| Dense vector distance kernels | **Phase A5** (`vector`) — *promoted from v0.3* per `00_BOTTOM_UP_ANALYSIS.md` |
| xxhash3 / wyhash SIMD | **Deferred** (Tier C) — no documented bottleneck |
| Bit-packed varint streams | **Phase A2** (`bits::bit_pack`) — kept, in `10_BITS.md` |

| `FS_PRIMITIVES_GAP.md` v0.3 item | Status here |
|---|---|
| CSR adjacency walk + BFS frontier | **Deferred** (Tier C) — wait for graph hot path |
| Hash-set membership SIMD | **Phase A4** (`hash::set_membership_simd`) — *promoted from v0.3* |
| Bloom / Cuckoo SIMD | **Deferred** (Tier C) — no documented bottleneck |
| Top-K SIMD heap | **Deferred** (Tier D) — no consumer |
| Levenshtein / Hamming distance | **Deferred** (Tier C) — no consumer |
| HyperLogLog SIMD merge | **Deferred** (Tier C) — no documented bottleneck |
| Batched BLAKE3 | **Phase A3** (`hash::blake3_batch`) — *promoted from v0.3* |

Net additions to v0.2 in this plan: `permutation` module (RCM + Hilbert in v0.2; Rabbit in D1) — entirely new primitive class not in `FS_PRIMITIVES_GAP.md`.

Net deferrals from `FS_PRIMITIVES_GAP.md`'s v0.2/v0.3 lists: 7 items, each with documented trigger condition.

Net result: v0.2 surface is **smaller** than `FS_PRIMITIVES_GAP.md`'s combined v0.2 + v0.3 (5 modules vs 11 items) but **higher leverage per LOC** because each shipped primitive has multiple consumers and a documented bottleneck it removes.

# Deferred items

**Status:** intentional-deferral list, 2026-05-02. Items in `FS_PRIMITIVES_GAP.md` v0.2/v0.3/v0.4 lists that don't make the v0.2 cut, with rationale, trigger condition, and **per-consumer interest map**.

The deferral discipline: each item lists (a) **why deferred**, (b) **what would un-defer it** (a documented signal from *any* consumer per [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md), not a guess), (c) **which consumers might un-defer it**.

The trigger reframing: prior version of this doc wrote triggers as "TokenFS asks." Multi-consumer view: triggers become "any consumer asks." Several Tier-C items have likely non-TokenFS consumers waiting in the wings — the trigger is now broader.

## Tier C — composition-driven, deferred until consumer asks

### CSR walk + parallel BFS frontier

- **Why deferred**: Pointer-chase-bound. The `permutation` module gives the layout that makes CSR walks fast scalarly first; SIMD on the walk itself is incremental and harder to justify.
- **Plausible consumers**: Graph databases (Neo4j-class), recommendation systems with collaborative-filtering walks. TokenFS unlikely (FS-traversal is I/O-bound).
- **Trigger to un-defer**: any consumer profiles bottlenecking on raw BFS/DFS throughput.
- **If un-deferred**: lands as `graph` module (Phase C+).

### MinHash SIMD signature kernel

- **Why deferred**: Existing `similarity::minhash` is scalar but works. The SIMD win is real (~5-10x on AVX-512) but no current consumer drives the "compute MinHash sig over many byte windows in parallel" hot path.
- **Plausible consumers**: **CDN edge dedup is the strongest** (Cloudflare/Fastly-class — actively use MinHash for cross-edge dedup); MinIO bulk-object similarity scoring; TokenFS image-wide near-dup detection.
- **Trigger to un-defer**: any of the above lights up as a workload, or image-build-time MinHash sidecar generation profiles bottlenecking.
- **If un-deferred**: extends `similarity::minhash` with `kernels::{avx2,avx512,neon}` modules. ~3-5 days of work.

### CHD batched lookup

- **Why deferred**: CHD itself is `tokenfs-paper`'s consumer (`boomphf` crate or similar). The batched-lookup primitive only matters once CHD is in active use AND profiling shows CHD lookup is hot (it almost certainly won't, since CHD is a single-cache-line probe).
- **Plausible consumers**: any CHD-using key-value store. TokenFS first.
- **Trigger to un-defer**: CHD lookup measured at >5% of total `lookup(path)` time on any consumer.
- **If un-deferred**: small kernel in `hash::chd_lookup_batch`.

### Levenshtein / Hamming SIMD-DP

- **Why deferred**: No documented use case in TokenFS. Path-typo-tolerance for shell completion is conceivable but not on the v1 spec.
- **Plausible consumers**: **Postgres `pg_trgm` extension** for fuzzy text search (real, well-known); shell completion; Lucene/tantivy-class search engines.
- **Trigger to un-defer**: any fuzzy-search consumer asks.
- **If un-deferred**: new `string` module. AVX2 16-lane SIMD-DP.

### HyperLogLog SIMD merge

- **Why deferred**: Existing `approx::HyperLogLog` is scalar but works. SIMD merge only matters when merging many HLL sketches in tight loops.
- **Plausible consumers**: **Postgres `approx_count_distinct` extensions** (Citus's `hll` extension is widely deployed); OLAP databases (Druid, Pinot, ClickHouse all have HLL operators that vectorize); TokenFS image-build aggregation.
- **Trigger to un-defer**: a Postgres HLL extension or OLAP database asks; or image build profiling shows HLL merge bottlenecking.
- **If un-deferred**: extends `approx::HyperLogLog` with merge kernel.

### Bloom / Cuckoo SIMD insert + query

- **Why deferred**: Existing `approx::BloomFilter` is scalar. SIMD on a Bloom probe is ~50 GB/s on AVX-512 vs ~5 GB/s scalar — real win — but Bloom probes are typically L1-resident few-cache-line operations where the constant factors don't always dominate.
- **Plausible consumers**: **Postgres bloom filter index** (`bloom` extension is in core); **MinIO content-fingerprint pre-checks** (every dedup query starts with a Bloom probe); **CDN edge "have this object?" pre-checks** (every request); TokenFS cross-image dedup.
- **Trigger to un-defer**: any of those workloads documents Bloom-probe time as a meaningful fraction of total query time.
- **If un-deferred**: extends `approx::BloomFilter`. The Postgres / CDN consumers make this **probably the highest-impact deferred item across the matrix.**

### Top-K via SIMD heap

- **Why deferred**: Existing `approx::SpaceSaving` and Misra-Gries cover heavy-hitters. Generic top-K isn't hot anywhere documented.
- **Plausible consumers**: **Postgres `ORDER BY ... LIMIT` queries** (executor maintains a heap), **pgvector ANN top-K**, MinHash-LSH candidate top-K, and any nearest-neighbor system. Branch-free small-array sort (≤ 64 elements via AVX-512 VPCOMPRESSQ) is a clean SIMD win.
- **Trigger to un-defer**: pgvector or any "top-K nearest-neighbor" consumer asks.
- **If un-deferred**: new `sort` module.

### `xxh3` / `wyhash` SIMD batch

- **Why deferred**: No documented bottleneck on existing FNV-1a / mix64 / CRC32C in any consumer. Modern xxh3 / wyhash are 5-10x faster than FNV scalar.
- **Plausible consumers**: high-throughput hash table workloads, content-addressed systems, hash-join inner loops in OLAP databases. **Probably has consumers, just none documented yet.**
- **Trigger to un-defer**: a benchmark or consumer demonstrates the existing hash family is bottlenecking on >5% of total runtime.
- **If un-deferred**: extends `hash` module.

## Tier D — speculative, no near-term consumer in any environment

### Wavelet tree on token stream

- **Why deferred**: Beautiful succinct DS but no v1 query asks for `rank(token X, position i)` or `select(token X, k-th occurrence)`. The v1 inverted index covers token presence; the wavelet tree adds positional queries.
- **Plausible consumers**: succinct-DS research; specialized tokenized-text query engines.
- **Foundation needed**: `bits::rank_select` (Phase B1). Once that lands, the wavelet tree itself is ~500 lines of glue.
- **Trigger to un-defer**: a token-stream positional-query consumer (`grep`-class, "find the third occurrence of token X in this extent").

### FM-index on token stream

- **Why deferred**: Same as wavelet tree but with substring search. Higher build cost; needs suffix array or BWT scaffolding.
- **Plausible consumers**: bioinformatics (FM-index over genomic-token streams); substring-aware search systems.
- **Foundation needed**: wavelet tree (above) + Burrows-Wheeler transform.
- **Trigger to un-defer**: a "token n-gram backward search" consumer.

### HNSW over fingerprints

- **Why deferred for TokenFS**: F22 fingerprints are 6-D. Brute-force nearest-neighbor over 228K points is sub-second on AVX-512 (~250 µs per query, see `02_CACHE_RESIDENCY.md`). HNSW only pays off above ~20-D or millions of points.
- **Plausible non-TokenFS consumers**: **Postgres pgvector** uses HNSW for ANN; FAISS/Vamana/DiskANN-class consumers. The v0.2 `vector::distance` kernels are *exactly* the inner loop pgvector consumes — that satisfies the "vector primitive" need without needing HNSW the data structure.
- **Trigger to un-defer**: TokenFS adds learned embeddings (typical 256-768-D), OR a non-TokenFS consumer wants a vendored HNSW (rather than rolling their own with our distance kernels). The latter is the more likely path.

### Vectorized regex DFA

- **Why deferred**: `aho-corasick` covers the immediate "byte-level multi-pattern search" need. Hyperscan-class vectorized DFA would extend this to richer regex classes.
- **Plausible consumers**: Postgres `regexp_matches` extension; intrusion-detection / log analytics tools.
- **Trigger to un-defer**: a feature-flagged regex engine in `tokenfs-paper`, or a Postgres regex extension consumer.

### FFT / NTT

- **Why deferred**: No use case. Unless content fingerprinting moves toward signal-processing-style summaries (it hasn't), FFT isn't a near-term primitive.
- **Plausible consumers**: signal-processing-class fingerprinting research; cryptographic primitives needing NTT.
- **Trigger to un-defer**: any of those research directions activates.

### Reed-Solomon erasure coding

- **Why deferred**: Useful for distributed/replicated TokenFS but not load-bearing for v1 (single-image read-only). MinIO uses erasure coding internally but at a different layer.
- **Plausible consumers**: distributed/replicated TokenFS deployment; storage systems wanting library-grade erasure coding.
- **Trigger to un-defer**: distributed/replicated TokenFS deployment scenario, or a storage system asks.

### Sparse-bitmap segment-tree / Fenwick-tree

- **Why deferred**: Roaring covers most range-query needs. Fenwick is a different shape (range-sum) but no consumer asks.
- **Plausible consumers**: range-sum query systems (financial time-series, telemetry aggregation).
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

## Re-prioritization candidates by consumer signal

If non-TokenFS consumers materialize during the v0.2 implementation phase, the deferred-item ranking should re-shuffle. Most likely promotions:

1. **Bloom SIMD** — Postgres `bloom` index, MinIO content-fingerprint pre-check, CDN edge "have this object" probe are all real and immediate. **Highest deferred-item leverage across the matrix.**
2. **HyperLogLog SIMD merge** — Postgres `approx_count_distinct` extensions (Citus's `hll`), OLAP DB consumers. Real, immediate.
3. **MinHash SIMD** — CDN edge dedup (Cloudflare/Fastly-class). Real if a CDN consumer materializes.
4. **Top-K SIMD heap** — pgvector ANN, Postgres `ORDER BY LIMIT`. Real if pgvector consumer asks.
5. **Levenshtein / Hamming** — Postgres `pg_trgm` consumer. Real if asked.

The discipline still holds: don't build until a consumer documents a bottleneck. But the consumers in the matrix are **real and the asks are likely**, not hypothetical.

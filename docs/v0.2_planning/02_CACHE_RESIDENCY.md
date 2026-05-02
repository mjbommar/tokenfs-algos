# Cache residency analysis

**Status:** design-pressure doc, 2026-05-02. The piece both `data-structures.md` and `FS_PRIMITIVES_GAP.md` miss for TokenFS scale; *also* relevant at Postgres / CDN / MinIO scale where conclusions flip.

## Why this doc exists

The cache-tier picture varies dramatically across consumers. The same primitive may be in-L3 for TokenFS-rootfs scale and DRAM-bound for Postgres-scale. Module designs need to handle both regimes; benches need to report numbers at multiple working-set sizes.

For a TokenFS image at 130K paths / 228K extents / 2K vocab, the entire hot metadata working set (inode table + extent metadata + fingerprint array + vocab) is **~22 MB** — fits in a single L3 cache. Most modern P-cores have 24-32 MB L3. **This single fact reorders the primitive priority list for TokenFS-scale consumers** but does NOT generalize to larger consumers.

For a Postgres pgvector index over 1M ~768D embeddings, the working set is ~3 GB — DRAM-bound. The bottleneck shifts from L3-gather latency to DRAM streaming bandwidth.

For a MinIO content-fingerprint table over 10M objects, the working set is ~320 MB — partially DRAM-bound, with bloom-filter pre-checks ideally L1/L2-resident.

When the working set is L3-resident (TokenFS regime):
- **Compression / size optimization is wasted effort.** The table fits anyway.
- **Random-gather kernels (~30 ns per L3 hit) are the bottleneck**, not bandwidth (~30 GB/s of L2/L3 transfer).
- **SIMD doesn't help random gathers**; only locality-improving layouts do.
- **The right design objective is "what queries stay inside L3?"** not "how fast do you process bytes once L1-loaded?"

When the working set is DRAM-bound (Postgres / MinIO / CDN regime):
- **Compression matters** — saving 30% of a 3 GB embedding table saves 1 GB of RAM that's otherwise pageable.
- **Streaming bandwidth dominates** at the DRAM tier (~30 GB/s/core); SIMD that doubles per-byte throughput halves wall time.
- **NUMA-locality and prefetch matter** at the page granularity, not at the cache-line granularity.
- **The right design objective is "minimize bytes touched per query"** — bit-packing, succinct DS, Roaring containers all pay direct dividends.

The prior docs talk about throughput (GB/s) but never about cache tier. This doc fills that gap for both regimes.

## Hardware reference numbers

| Cache tier | Capacity (P-core, modern x86) | Latency | Bandwidth (per core) |
|---|---|---|---|
| L1d | 48-64 KB (Zen4 64 KB; Sapphire Rapids 48 KB) | ~4 cycles ≈ 1 ns | ~250 GB/s |
| L2 | 1-4 MB (Zen4 1 MB; SPR 2 MB) | ~12 cycles ≈ 3 ns | ~100 GB/s |
| L3 | 16-128 MB (Zen4 32 MB shared; SPR 48 MB shared) | ~30 cycles ≈ 8 ns / random ~30 ns | ~30-50 GB/s |
| DRAM | -- | ~80 ns random | ~30 GB/s/core (DDR5-6400) |
| NVMe (cold extent) | -- | ~50 µs | ~3-7 GB/s |

NEON-side reference (Apple M-series, Graviton):

| Tier | Apple M2 | Graviton 3 |
|---|---|---|
| L1d | 128 KB | 64 KB |
| L2 | 16 MB (perf cluster) | 1 MB |
| SLC / L3 | 24 MB | 32 MB |
| DRAM bandwidth | ~100 GB/s/core | ~50 GB/s |

NEON is bandwidth-rich on M-series; primitives that bottleneck on bandwidth on x86 won't on M.

## TokenFS table residency

For an Ubuntu-rootfs-class image (130K paths, 228K extents, 2K vocab tokens):

| Table | Per-entry | Cardinality | Total | Tier | Notes |
|---|---|---|---|---|---|
| Vocab table | ~16 B avg | 2K | **~32 KB** | **L1** | Hot, every detokenize touches it |
| L1d size budget consumed by vocab | | | | 50% of 64 KB L1d | |
| Extent rep+len header | 16 B | 228K | ~3.6 MB | L2/L3 | Touched on every read() |
| Extent metadata sidecar | 32 B | 228K | ~7.3 MB | L2/L3 | Touched in analytics scans |
| Inode table | 64 B | 130K | ~8.3 MB | L2/L3 | Touched on every lookup |
| Fingerprint array (F22) | 32 B | 228K | ~7.3 MB | L2/L3 | Touched in similarity scans |
| **All hot metadata combined** | | | **~26 MB** | **fits in 32 MB L3** | **Full scan ~10 ms warm** |
| Path FST | ~10 B/path avg | 130K | ~1.3 MB | L2/L3 | Per-path lookup ~log path-depth misses |
| 2-gram inverted index posting lists (Roaring-encoded) | varies | ~|V|² = 4M entries | 10s of MB | L3/DRAM border | |
| 4-gram inverted index posting lists | varies | huge | hundreds of MB to GB | DRAM | |
| Token streams (packed u11) | ~1.4 B/token | varies | MB to GB per image | DRAM / mmap | |
| Extent payloads | varies | varies | GB-scale | mmap / NVMe | Cold-path on first read |

## Other consumer cardinality regimes

### Postgres extension scale

For a Postgres database with 1M-100M-row tables:

| Table | Per-entry | Cardinality | Total | Tier | Notes |
|---|---|---|---|---|---|
| GIN posting list (per indexed term) | varies | 100K-10M tids | 1-100 MB | L3-DRAM border | Roaring-style intersection bound |
| pgvector embedding table | ~3 KB (~768D f32) | 100K-10M | 300 MB - 30 GB | DRAM | Vector distance scan |
| BRIN summary | ~32 B per range | 1K-100K ranges | 32 KB - 3 MB | L1-L2 | Lookup pre-filter |
| Bloom filter index | ~1 byte/element | per-page | 8 KB / page | L1 per probe | Membership test |

**Implication:** Roaring set ops at Postgres scale are full streaming DRAM ops, not in-cache. Stream-VByte for column compression matters because tables don't fit RAM, period. Vector distance is bandwidth-bound on the DRAM streaming of vectors.

### MinIO / object-store scale

| Table | Per-entry | Cardinality | Total | Tier | Notes |
|---|---|---|---|---|---|
| Object metadata | ~256 B | 10M objects | 2.5 GB | DRAM | ~one-cache-line per probe |
| Content-fingerprint table | 32 B (SHA-256) | 10M | 320 MB | DRAM | dedup pre-check |
| Bloom filter for "exists" pre-check | 8-16 bits/key | 10M | 10-20 MB | L3 | sub-µs probe |
| Per-tenant aggregate stats | ~64 B | 1K-10K tenants | < 1 MB | L1-L2 | hot |

**Implication:** Bloom SIMD matters here in a way it doesn't at TokenFS scale. Postgres "exists" probes are a real consumer. MinIO bulk dedup queries also.

### CDN edge scale

| Table | Per-entry | Cardinality | Total | Tier | Notes |
|---|---|---|---|---|---|
| Per-edge cache directory | ~32 B | 10M entries | 320 MB | DRAM | per-request lookup |
| MinHash sketches per object | 256 B | 10M | 2.5 GB | DRAM | similarity LSH |
| Bloom for "have this object" | 1 byte/key | 10M | 10 MB | L3 | per-request pre-check |
| Per-tenant LRU heads | ~8 B | 100K | < 1 MB | L1 | hot |

**Implication:** the CDN-edge primitive set is **Bloom + MinHash + Roaring**, all of which we either have skeletal versions of (Bloom, MinHash) or are planning (Roaring SIMD). The deferral-trigger reframing in `20_DEFERRED.md` reflects this.

### Build-time pipeline (TokenFS, MinIO, CDN edge ingest)

| Workload | Per-extent compute | Total work for 200K extents |
|---|---|---|
| BLAKE3 over extent payloads (avg 4 KB) | ~1.3 µs scalar / ~0.4 µs AVX-512 | ~250 ms / ~80 ms |
| SHA-256 with SHA-NI | ~1 µs | ~200 ms |
| F22 fingerprint per block | ~1.4 µs (current AVX2) | ~280 ms |
| Roaring-encode posting list | varies | 100s of ms |

**Implication:** the build pipeline is the canonical "batched cryptographic hash" workload. The same 200K-Merkle-leaves shape applies in TokenFS (extents), MinIO (objects), and CDN ingest. Cross-consumer batch API.

## Implications for primitive design

### 1. Inode + extent metadata + fingerprint = "L3-resident sweep zone"

Any analytics workload that scans these arrays is a pure ~10-30 ms full-table-walk operation, dominated by L3 latency for random hits. **SIMD on the per-element compute is a 2-3x win at most**, because the kernel is already memory-bound on L3 transfer (~30 GB/s) not on instruction throughput.

What helps these workloads:
- **Layout / locality:** if you scan in physical order (sequential cache lines), L3 streaming bandwidth dominates and you hit ~30 GB/s. If you scan in random order, you pay ~30 ns per random L3 hit and effective bandwidth drops 10-30x.
- **`permutation` module** is the primitive that makes scans physically sequential. **Highest leverage primitive in the entire crate**, in terms of "speedup × number of consumers."
- **Bit-packing the metadata** (saving 50% of the 26 MB) doesn't help: the table still fits L3 either way.
- **Software prefetch** at the right distance (~16 cache lines ahead for L3 walks) reclaims most of the loss when scan order is non-sequential — typical 1.5-3x win on random traversal patterns. Cheap to add (one `core::intrinsics::prefetch_read_data` per loop body).

### 2. Vocab table = "L1-resident lookup table"

Vocab table at 32 KB is **half of L1d**. Detokenization runs the same lookup every byte. The primitive design pressure here is:

- Keep the table cache-line aligned and contiguous.
- Lookups are u11/u12 → byte-string-pointer-and-length. Implementable as 2 packed arrays: u32 offsets[V] (8 KB) and the byte-payload arena (~24 KB).
- **No SIMD needed** for individual lookups (single L1 load is 4 cycles; SIMD doesn't beat that). SIMD wins when batching K lookups in a tight loop, where you can do K gathers in parallel via VPGATHERDD (~5 GB/s realistic on AVX2).

### 3. Posting lists = "the place SIMD pays off"

A 2-gram inverted index has |V|² entries, each pointing to a sorted u32 extent ID list. Across all 2-grams, total volume is in the tens of MB to low GB depending on coverage.

**The hot operation is intersection of two posting lists** for a boolean query. Both lists are typically sequentially streamed into L1 (good locality), the per-element compute is comparison + branch + advance, and SIMD wins big here:

- Roaring container ops are 5-30x faster SIMD vs scalar on dense data.
- Stream-VByte decode is 4-16 GB/s SIMD vs 200-500 MB/s scalar.

This is the workload class where the `bitmap` and `bits` modules pay direct dividends.

### 4. Token streams = "throughput-bound decode"

A typical extent's tokenized form is hundreds of KB to a few MB. Reading the extent runs the decoder over the entire token payload once. Decoder throughput therefore directly affects every read.

If decoder is 1 GB/s: a 2 MB extent decode takes 2 ms. At 10 GB/s: 200 µs. At 30 GB/s (memcpy-class): 65 µs. **The bit-unpack SIMD primitive is the difference between tokenized extents being a slow path and a fast path.**

### 5. Extent payloads = "I/O bound"

For cold reads: NVMe is ~50 µs and decompressed bandwidth is whatever the storage stack does. SIMD primitives don't help here. **Prefetch and layout matter** (which is the `permutation` story again, applied to extent placement order on disk).

For warm reads: the extent is mmap'd and page-cache-resident. Now decode (above) is the bottleneck.

## Bytes-touched-per-query budgets

Some illustrative budgets for designing primitives against:

| Query | Bytes touched | Time at L3 bw (~30 GB/s) | Time at DRAM bw (~30 GB/s) |
|---|---|---|---|
| `lookup(path)` warm | ~few KB (FST traversal) | <1 µs | n/a |
| `read(inode, off, 4096)` warm | ~few KB (inode + extent map) + 4 KB (page) | ~1 µs | n/a |
| Full inode-table scan (filter by mtime) | 8.3 MB | ~280 µs | ~280 µs |
| Full fingerprint scan (find similar) | 7.3 MB | ~250 µs | ~250 µs |
| Posting-list intersect (10K × 10K, both Roaring-sparse) | ~80 KB | ~3 µs | n/a |
| Posting-list intersect (100K × 100K, both bitmap-dense) | ~16 KB (two 8KB containers per range) | ~1 µs | n/a |
| Image-wide Merkle verify | ~7 MB extent hashes + work | ~5-50 ms compute-bound on hash | -- |
| Image-build BLAKE3 over extents | sum of extent payloads | I/O bound | -- |

The 250-µs full-fingerprint-scan is the most striking number: at modern hardware, the *entire* "find me the most similar extent in this image" workload is sub-millisecond if the fingerprint array is L3-resident and laid out sequentially. **The job of the primitive layer is to make sure the layout is sequential** (`permutation`) **and that the per-element distance compute is at least L3-bandwidth-fast** (`vector::distance`). Both of those primitives are in Phase A/B; together they make this query a single-digit-ms operation.

## Design pressure summary

| Primitive | Bound by | What helps |
|---|---|---|
| `bits::popcount` | compute (~10 cycles scalar; 1 cycle AVX-512) | AVX-512 VPOPCNTQ (10x) |
| `bits::bit_pack/unpack` | bandwidth (touched every detokenize) | SIMD shifts + table-lookup gather |
| `bits::streamvbyte` | bandwidth (posting list sweep) | AVX2 VPSHUFB lookup |
| `bits::rank_select` | latency (single rank query is one block lookup + one popcount) | small block size; prefetch |
| `bitmap::roaring intersect` | bandwidth (sequential streaming of two containers) | SIMD merge / branchless intersect |
| `hash::batched_blake3` | compute (per-byte hash work) | BLAKE3 internal SIMD; parallelism across messages |
| `vector::distance` | bandwidth (linear sweep of two vectors) | SIMD FMA; cache prefetch at sweep distance |
| `permutation::rcm` | latency (BFS over CSR — pointer chasing) | none; the kernel itself doesn't SIMD; the OUTPUT makes everything else fast |
| `permutation::rabbit` | mixed (modularity inner loop is dot-shaped; merging is concurrent-hash-bound) | SIMD on inner loop; concurrent-merge tricks |
| `hash::set_membership_simd` | compute (small set, hot kernel) | AVX2 VPCMPEQ broadcast + VPMOVMSKB |

The `permutation` row stands out: it's the only primitive whose *output*, not *throughput*, is the value. Once you have a good permutation, every downstream layout-aware operation gets faster — typically 1.5-3x on graph algorithms, similar on metadata scans. This is why it's Tier B / phase B even though the inner loop itself isn't a "SIMD win."

## What this means for benchmarking

Every Phase A/B primitive bench should report:
1. **In-L1 throughput** (warmup-dominant).
2. **In-L3 throughput** (~22 MB working set).
3. **DRAM throughput** (~256 MB working set).
4. **Cold-cache latency** (single op after `clflush` / equivalent on aarch64).

Most current `tokenfs-algos` benches only report (1). Adding (2-4) makes the SIMD-vs-scalar tradeoff honest.

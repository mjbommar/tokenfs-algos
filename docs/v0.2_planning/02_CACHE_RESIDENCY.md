# Cache residency analysis

**Status:** design-pressure doc, 2026-05-02. The piece both `data-structures.md` and `FS_PRIMITIVES_GAP.md` miss.

## Why this doc exists

For a TokenFS image at 130K paths / 228K extents / 2K vocab, the entire hot metadata working set (inode table + extent metadata + fingerprint array + vocab) is **~22 MB** — fits in a single L3 cache. Most modern P-cores have 24-32 MB L3.

This single fact reorders the primitive priority list. When everything is L3-resident:

- **Compression / size optimization is wasted effort.** The table fits anyway.
- **Random-gather kernels (~30 ns per L3 hit) are the bottleneck**, not bandwidth (~30 GB/s of L2/L3 transfer).
- **SIMD doesn't help random gathers**; only locality-improving layouts do.
- **The right design objective is "what queries stay inside L3?"** not "how fast do you process bytes once L1-loaded?"

The prior docs talk about throughput (GB/s) but never about cache tier. This doc fills that gap.

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

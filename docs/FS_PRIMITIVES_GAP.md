# tokenfs-algos × tokenfs-paper — primitive gap analysis

**Status:** working synthesis doc. 2026-05-01. Companion to `tokenfs-paper/docs/data-structures.md`. Maps the FS-level data structures TokenFS wants to ship to the `tokenfs-algos` primitives those structures need, identifies what already exists, and ranks the missing hardware-accelerated primitives by leverage.

**Audience:** `tokenfs-algos` contributors deciding what to add next, and `tokenfs-paper` contributors needing to know which compute primitives they can rely on.

---

## 1. The two projects, the seam between them

`tokenfs-algos` is a low-level, **content-agnostic** Rust crate. It owns hardware-accelerated algorithms over `&[u8]`: histograms, n-gram counters, entropy, run-length detection, byte classification, content-defined chunking, distribution distances, sketches, F22 fingerprints. It knows nothing about extents, inodes, manifests, tokenizers, or filesystem objects. AGENTS.md states this explicitly: *"The core library is not a tokenizer, filesystem, compression codec, file parser, or machine-learning layer. It is pure compute over byte slices."*

`tokenfs-paper` is the **systems** project: TokenFS image format, build-time oracle, FUSE reader, kernel module, paper claims. It owns the on-disk layout (manifest, sections, inode table, extent metadata), the build pipeline (`tokenfs_writer`), the read path (`tokenfs_reader`, `tokenfs_fuse`), the encoder/decoder (`tokenfs_algorithms`, `tokenfs_pybind`), and the experiments / paper artifacts.

The seam is clean: **tokenfs-paper depends on tokenfs-algos.** Never the other way around. tokenfs-algos has zero knowledge of TokenFS. This is the rule that lets the algorithms crate become useful to other consumers (kernel filesystem, FUSE, future Python data-pipeline tooling, compression-codec dispatchers, forensics tools, columnar databases).

The data-structures doc in `tokenfs-paper/docs/data-structures.md` lists 16 candidate FS data structures across four layers (storage, namespace graph, indexing, distance/matching). For each one, the question this doc answers is: **which of its compute primitives belong in tokenfs-algos?** The structure itself — the on-disk layout, the section pointer, the manifest binding — stays in `tokenfs-paper`. The pure-compute kernel underneath belongs here.

## 2. Current tokenfs-algos surface (as of 2026-05-01)

Public modules in `crates/tokenfs-algos/src/`:

```
approx          byteclass    chunk        dispatch     distribution
divergence      entropy      fingerprint  format       hash
histogram       math         paper        prelude      primitives
runlength       search       selector     similarity   sketch
sketch_p2       structure    windows
```

What exists (from `PLAN.md` § 0 and `PRIMITIVE_KERNEL_BUFFET.md`) at scalar / reference level, with x86 AVX2 backends for the canonical hot paths:

- Byte histograms (multi-counter), exact n-gram histograms, dense byte-pair histograms.
- Misra-Gries heavy-hitters, Count-Min Sketch, P² streaming quantile.
- CRC32C hash bins, FNV-1a, mix64.
- Byte-class counts, UTF-8 validation, run-length / structure summaries.
- Gear chunking, normalized FastCDC chunking.
- Distribution distances (KL, JS, Hellinger, χ², KS, TV — the `divergence` module).
- Calibrated byte-distribution lookup, selector signals, F21 selector.
- F22 block & extent fingerprints (scalar + AVX2/SSE4.2 pinned paths).
- Runtime processor detection, planner explanations, pinned histogram kernels, benchmark comparison and parity reports.

What's stubbed or aspirational: AVX-512, NEON, SVE/SVE2 backends are feature-shaped scalar fallbacks. AVX2 is the only optimized x86 target with real kernels.

## 3. Mapping FS data structures to tokenfs-algos primitives

For each candidate from `data-structures.md`, the table shows what tokenfs-algos already provides, what's missing, and where the missing pieces would land.

### Tier 1 (ship in v1)

#### #1 Per-extent fingerprint sidecar (F22)

| | Status |
|---|---|
| Where it lives | `tokenfs-algos::fingerprint` (in progress migration from `tokenfs-paper/tools/rust/entropy_primitives/`) |
| Existing primitives | byte hist, n-gram hist, run-length, byte-class, c·log₂(c) entropy LUT, CRC32C hash bins |
| Missing primitives | none — F22 is the canonical example of what tokenfs-algos already does |
| Hardware acceleration | AVX2 (1.41 µs/block), scalar fallback. AVX-512 + NEON are real targets for v0.2. |

#### #2 Path FST

| | Status |
|---|---|
| Where it lives | `tokenfs-paper` (uses BurntSushi `fst` crate as a dep) |
| Existing primitives | none in tokenfs-algos |
| Missing primitives | **None we should build.** The `fst` crate is mature, peer-of-`memchr` philosophy, MIT/Apache. Take as a dep in `tokenfs_writer`/`tokenfs_reader`. tokenfs-algos doesn't need to own this. |
| Hardware acceleration | The `fst` crate uses memchr internally for byte-search hot paths (already SIMD). |

#### #3 Token n-gram inverted index (Roaring + Stream-VByte)

| | Status |
|---|---|
| Where it lives | Composition is in `tokenfs-paper`; primitives in `tokenfs-algos` |
| Existing primitives | n-gram histogram, sketch (CMS / MG) |
| **Missing — high priority** | **Stream-VByte codec** (encode + decode), **Roaring bitmap SIMD intersection / union / difference kernels** |
| Hardware acceleration | Stream-VByte: AVX2 → ~2 GB/s decode (Lemire's reference). Roaring AVX-512 intersection: 30–60 GB/s on dense regions (CRoaring numbers). Both are clear tokenfs-algos primitives because they're domain-agnostic, SIMD-friendly, and currently absent from idiomatic Rust crates. |

#### #4 CSR forward adjacency for directory tree

| | Status |
|---|---|
| Where it lives | Composition in `tokenfs-paper`; graph primitive in `tokenfs-algos` |
| Existing primitives | none |
| **Missing — medium priority** | **CSR adjacency walk**, **vectorized BFS frontier expansion**, **DFS preorder iterator**. The graph-processing primitive set. |
| Hardware acceleration | CSR walks are pointer-chase-bound; AVX2 `VPGATHERDD` accelerates neighbor enumeration. Realistic ~1–5 GB/s. |

#### #5 CHD perfect hash for directory entries

| | Status |
|---|---|
| Where it lives | Construction in `tokenfs_writer`; lookup in `tokenfs_reader` |
| Existing primitives | hash (FNV-1a, mix64) |
| **Missing — medium priority** | **CHD or BBHash construction + lookup primitive**. Crates: `phf`/`boomphf` are mature deps. Question: vendor or take? Recommendation: take `boomphf` as a dep in tokenfs-paper; tokenfs-algos provides only the **batched lookup primitive** (look up K keys against a perfect hash in vectorized form). |
| Hardware acceleration | Batched CHD lookup: AVX2 hashes 4–8 keys in parallel. ~5 GB/s realistic. |

### Tier 2 (v1.1 / v1.2)

#### #6 Content-addressed extent table

| | Status |
|---|---|
| Where it lives | `tokenfs_writer` / `tokenfs_reader` (composition); `tokenfs-algos::hash` (batched hashing) |
| Existing primitives | hash module (FNV/mix64), CRC32C bins |
| **Missing — high priority** | **Batched BLAKE3 / SHA-256** for content addressing. The cryptographic hash itself comes from `blake3` / `sha2` crates (already SIMD-internal); tokenfs-algos provides the **batched-hash primitive** that processes N small inputs in parallel for Merkle-leaf and content-addressing throughput. |
| Hardware acceleration | BLAKE3 is internally AVX-512 / AVX2 / SSE4.1 / NEON optimized. Wrapping it for batched use in tokenfs-algos costs little but consolidates the hashing surface. |

#### #7 CSR reverse adjacency

Same primitives as #4. Reuses the graph-walk module.

#### #8 Subtree bitmaps (Roaring per directory)

Reuses #3's Roaring SIMD kernels. The composition adds nothing new.

#### #9 Merkle tree over extent hashes

| | Status |
|---|---|
| Where it lives | `tokenfs_writer` (build); `tokenfs_reader` (verify) |
| Existing primitives | hash (FNV/mix64). |
| **Missing** | **BLAKE3 tree-mode integration** (tokenfs-algos exposes a "Merkle tree builder" wrapper over `blake3` that's allocator-aware). Optional: **batched SHA-256** for compatibility with dm-verity / fs-verity. |
| Hardware acceleration | BLAKE3 tree mode is internally vectorized; ~3–5 GB/s on AVX2 for the leaf hashes, scaling near-linearly with cores. |

#### #10 Content-class index per inode

Trivial; no primitive needed beyond a sorted array. Reuses F21 cluster-id output.

### Tier 3 (research / speculative)

#### #11 Wavelet tree over token stream

| | Status |
|---|---|
| Where it lives | `tokenfs-algos` (succinct DS primitive) |
| Existing primitives | none |
| **Missing — research investment** | **Bit-rank / bit-select dictionary** with SIMD popcount. The fundamental operation in wavelet trees is `popcount` over bit ranges and `rank(i, b)` = "how many `b`s before position i". |
| Hardware acceleration | AVX-512 `VPOPCNTQ` gives 64-lane popcount; ~10–50 GB/s realistic. AVX2 software popcount ~5 GB/s. NEON `VCNT` similar. The popcount kernel is one of the cleanest SIMD wins in succinct data structures and would set the foundation for FM-index too. |

#### #12 MinHash + LSH

| | Status |
|---|---|
| Where it lives | `tokenfs-algos::similarity` (already has the module skeleton) |
| Existing primitives | hash families, sketch |
| **Missing — medium priority** | **SIMD MinHash signature kernel** (compute K min-hashes over windowed bytes in parallel), **LSH band-and-hash partitioner**. |
| Hardware acceleration | StringZilla's reference: ~5–10 GB/s on AVX-512. AVX2 ~3 GB/s. The hash family (xxh3 / wyhash / mum) needs to be vectorizable. |

#### #13 HNSW over fingerprints (or learned embeddings)

| | Status |
|---|---|
| Where it lives | Higher crate (`tokenfs-similarity`, future); tokenfs-algos provides distance kernels only |
| Existing primitives | divergence module covers KL/JS/Hellinger; no L2/dot/cosine yet |
| **Missing — medium priority** | **Dense vector distance kernels** (L2², dot product, cosine, Hamming, Jaccard) — single-pair and batched. These are the inner loop of every ANN method (HNSW, IVF, brute force). |
| Hardware acceleration | AVX-512 FMA: ~30 GB/s on f32. AVX2: ~15 GB/s. NEON: ~8 GB/s. The cleanest SIMD kernel family in vector search. **High priority even without HNSW** — they're load-bearing for any vector-similarity work. |

#### #14 FM-index on token stream

Reuses #11's bit-rank/select dictionary plus a wavelet tree. Build cost is high; lookup is the same primitives. Defer until #11 is shipped.

#### #15–16 Topo sort / connected components / MST

Mostly scalar graph algorithms. No SIMD wins worth pursuing. tokenfs-algos could ship a tiny `graph` module with reference scalar implementations + rayon parallelism, but these aren't the hardware-acceleration story.

## 4. Additional hardware-accelerated primitives worth implementing

Beyond the FS-data-structure-driven list above, here are primitives that don't appear in `data-structures.md` but should land in tokenfs-algos because they're (a) load-bearing for multiple use cases, (b) have clean SIMD wins, (c) absent from the Rust ecosystem in idiomatic form. Ranked by leverage.

### Ship in v0.2

**Stream-VByte codec.** Lemire's algorithm. AVX2 inner loop with VPSHUFB-driven byte permutations. Encodes integer streams at 1 GB/s, decodes at 2 GB/s. Used in inverted indices, columnar formats, anywhere delta-coded varints appear. No idiomatic Rust SIMD implementation exists.

**Roaring bitmap SIMD kernels.** Intersection / union / difference / cardinality / iteration. CRoaring (Lemire's C lib) has them; the Rust `roaring` crate uses scalar fallbacks. AVX-512 `VPCONFLICTQ` + `VPCMPEQQ` accelerate set intersection. Lifting just the SIMD inner loops gives tokenfs-algos a real edge.

**Bit-rank / bit-select dictionary primitives.** Foundation for wavelet trees, FM-index, succinct trees, RRR sequences. AVX-512 `VPOPCNTQ` makes this 10–50 GB/s. Even AVX2 software popcount hits ~5 GB/s. The single primitive that unlocks the most downstream succinct-DS work.

**Dense vector distance kernels.** L2², dot product, cosine, Hamming, Jaccard. Single-pair APIs and batched-many-vs-one APIs. AVX-512 FMA gives ~30 GB/s on f32 vectors. **High priority even without HNSW** — they're load-bearing for any future vector similarity work, and the AVX-512 FMA path is one of the cleanest SIMD wins in the whole crate.

**xxhash3 / wyhash SIMD.** Fast non-cryptographic hashing for hash tables, sketches, content-defined chunking. xxhash3 has AVX-512 in the C reference; Rust `xxh3` doesn't surface SIMD. wyhash is faster scalar but doesn't vectorize as cleanly. Worth a small SIMD batch wrapper.

**Bit-packed varint streams.** Pack/unpack arbitrary bit-widths (1, 2, 4, 8, 12, 16-bit values into byte-aligned streams) with SIMD bit-fiddling. Foundation for `token_packed`-style storage, succinct sequences, custom integer encodings.

### Ship in v0.3

**CSR adjacency walk + parallel BFS frontier.** Graph-processing primitive. Pregel/Ligra/GraphBLAS pattern. Used by every directory-tree query in TokenFS. AVX2 `VPGATHERDD` accelerates neighbor lookup. ~1–5 GB/s realistic.

**Hash-set membership SIMD.** "Is x in this 256-element set?" via `VPCMPEQ` broadcast + `VPMOVMSKB`. Used in dictionary lookups, Bloom-filter pre-checks, content-class membership. AVX2 hits ~30 GB/s.

**Bloom filter / Cuckoo filter SIMD insert + query.** Membership testing in batch. AVX-512 makes this trivially fast (~50 GB/s).

**Top-K via SIMD-accelerated heap.** Streaming top-K with AVX2 branch-free inner loop. Used in heavy-hitters consolidation, MinHash candidate update, partial sorts. Branch-free comparison sort over small arrays (≤ 64 elements) via AVX-512 `VPCOMPRESSQ` extends this.

**Levenshtein / Hamming distance vectors.** Hot in approximate string matching. AVX2 16-lane SIMD-DP gives 5× speedup on ~256-byte string pairs. Foundation for typo-tolerant search, near-duplicate detection in path namespaces.

**HyperLogLog merge / cardinality SIMD.** Leading-zero-count and min-merge ops vectorize. Useful for "how many distinct n-grams in this region?" estimation.

**Batched BLAKE3 for Merkle-leaf hashing.** Wraps the existing `blake3` crate but exposes a "hash N small inputs in parallel" API that consumes the BLAKE3 internal SIMD evenly. Critical for image-build performance when hashing 200K extents.

### Ship in v0.4 / research

**Substring search with multi-pattern.** `aho-corasick` already provides this for byte-level search; the question is whether tokenfs-algos exposes a token-level analogue (token-substring search) using the same automata pattern.

**Sparse-bitmap segment-tree / Fenwick-tree primitives.** Range queries over arbitrary alphabets. Useful when Roaring isn't the right shape (range-sum queries, prefix counts).

**Vectorized regex-like patterns.** Beyond `aho-corasick`'s automaton, vectorized DFA simulation for restricted regex classes. Hyperscan-style. Out of scope for v1 but tracked.

**FFT / NTT.** For convolutions and polynomial-evaluation-as-CRC. Probably out of scope unless we get into signal-processing-style content fingerprinting.

**Reed-Solomon erasure coding.** For error correction over extent payloads. Useful for distributed/replicated TokenFS but not load-bearing for v1.

## 5. Module organization for the new primitives

Mapping the new primitives to existing or new modules in `tokenfs-algos`:

| Primitive | Module |
|---|---|
| Stream-VByte codec | `bits` (new) |
| Bit-pack / unpack at arbitrary widths | `bits` (new) |
| Bit-rank / bit-select dictionary | `bits` (new) — succinct foundation |
| Roaring SIMD kernels | `bitmap` (new) |
| BLAKE3 batched leaves | `hash` (existing) |
| xxhash3 / wyhash SIMD | `hash` (existing) |
| Hash-set membership SIMD | `hash` (existing) |
| Bloom / Cuckoo SIMD | `sketch` (existing) — or split into `filter` |
| Top-K via SIMD heap | `sort` (new) — or fold into `sketch` |
| Branch-free small-array sort | `sort` (new) |
| Dense vector distance kernels | `vector` (new) — distinct from `divergence` (which is over distributions) |
| MinHash SIMD | `similarity` (existing) |
| LSH band-hash | `similarity` (existing) |
| CSR walk / BFS frontier | `graph` (new) |
| Levenshtein / Hamming dist | `string` (new) |
| HyperLogLog SIMD merge | `sketch` (existing) |

Final shape after additions:

```
approx     byteclass  bits      bitmap     chunk       dispatch
distribution  divergence  entropy   fingerprint  format    graph
hash       histogram  math      paper        prelude   primitives
runlength  search     selector  similarity   sketch    sort
string     structure  vector    windows
```

Six new top-level modules: `bits`, `bitmap`, `graph`, `sort`, `string`, `vector`. Each has a clear scalar reference and at least one SIMD backend. The existing modules stay untouched.

## 6. Dependency posture between projects

Concrete dependency flow:

```
tokenfs-paper (image format, FUSE, kernel module, paper)
    │
    ├── tokenfs_writer / tokenfs_reader / tokenfs_fuse
    │       │
    │       ├── bbpe (encoder/decoder/trainer)
    │       │       └── tokenfs-algos
    │       │
    │       └── tokenfs-algos
    │
    └── tokenfs-paper-experiments (Python, F-numbered scripts)
            └── tokenfs-algos (via pyo3 wrapper)
```

`tokenfs-algos` has zero deps on tokenfs-paper. tokenfs-paper has zero deps on tokenfs-algos's experiments or paper-lineage docs. This is the seam we're protecting.

External crate deps for the new primitives:

- `blake3` (for Merkle leaves) — already SIMD-internal, ~3–5 GB/s.
- `roaring` (for the bitmap representation) — but our SIMD kernels supplement, not replace.
- `fst` (for Path FST) — only consumed by tokenfs-paper, not by tokenfs-algos.
- `phf` or `boomphf` (for perfect hashing construction) — same: tokenfs-paper consumer.
- `sha2` / `xxh3` / `wyhash` (hash crates) — consumed inside tokenfs-algos, with SIMD batch wrappers exposed.

External structural references for design:
- **CRoaring** — the C library, for SIMD bitmap operations.
- **sdsl-lite** / **sucds** — for succinct data structures (wavelet tree, FM-index, bit-rank/select).
- **simdjson** — for stable-API + swappable-backend pattern.
- **simdutf** — for UTF-8 validation (already integrated in tokenfs-algos).
- **StringZilla** — for SIMD string kernels and capability reporting.

These are all `_references/`-friendly: take design ideas, vendor only with explicit license review.

## 7. Recommended sequence

This is the path from the current state to a v0.2 that supports the v1 image format from `data-structures.md` and the v0.2 research-flavored structures from there.

### Immediate (v0.1 → v0.1.x)

1. **Migrate F22 fingerprint kernel** from `tokenfs-paper/tools/rust/entropy_primitives/` into `tokenfs-algos::fingerprint`. (Already on the roadmap.)
2. **Land Stream-VByte codec** (`bits` module). AVX2 + scalar; criterion benches; parity tests.
3. **Land Roaring SIMD intersection / union / difference** (`bitmap` module). AVX2 first; AVX-512 stretch. Calibration test against the `roaring` crate's scalar output.
4. **Land bit-rank / bit-select dictionary primitive** (`bits` module). AVX-512 `VPOPCNTQ` where available; AVX2 software popcount fallback.

Together these unblock the v1 token n-gram inverted index (via #2 + #3) and the v0.2 wavelet tree (via #4).

### Short-term (v0.2)

5. **Dense vector distance kernels** (`vector` module). L2², dot, cosine, Hamming, Jaccard. AVX-512 FMA + AVX2 + scalar.
6. **xxhash3 / wyhash SIMD batch** (`hash` module).
7. **CSR adjacency + BFS frontier** (`graph` module). AVX2 `VPGATHERDD` for neighbor enumeration.
8. **CHD perfect-hash batched lookup primitive** (`hash` module).

### Medium-term (v0.3)

9. **MinHash SIMD signature kernel** (`similarity` module).
10. **Bloom / Cuckoo SIMD insert+query** (`sketch` module).
11. **Top-K SIMD heap + branch-free small-array sort** (`sort` module).
12. **Levenshtein / Hamming SIMD-DP** (`string` module).
13. **Batched BLAKE3 leaves wrapper** (`hash` module).

### Research / future (v0.4+)

14. **Wavelet tree over u16 token stream** (composition of `bits` + `bitmap`).
15. **FM-index on tokens** (composition of wavelet tree + suffix array).
16. **HNSW graph builder** (`vector`/`graph` composition; possibly its own crate `tokenfs-similarity`).
17. **Vectorized DFA simulation** for restricted regex (`string` module, stretch).

## 8. Open questions

1. **Should tokenfs-algos vendor `roaring` for SIMD acceleration, or contribute SIMD kernels upstream to `roaring-rs`?** Upstream contribution is cleaner ecosystem-wise but slower to land. Vendoring keeps us in control but forks the codebase. Recommended: contribute upstream when stable, ship our own kernels in the meantime under feature-gated paths.

2. **Where does the Path FST live — tokenfs-paper or tokenfs-algos?** I argued above that tokenfs-paper takes the `fst` crate as a dep directly. Counter-argument: tokenfs-algos could provide a thin "indexed-FST-with-hardware-SIMD-search" wrapper around `fst` for consistency. Probably not worth the indirection.

3. **NEON priority.** Apple Silicon and AWS Graviton are both real deployment targets. NEON for the popcount, byte-permute, and shift-and-compare primitives unlocks ARM. Recommended: NEON parity for the v0.2 set (Stream-VByte, Roaring, popcount, vector distance) before adding more x86 features.

4. **AVX-512 timeline.** Sapphire Rapids and Zen4/5 are widespread now. AVX-512 brings VPOPCNTQ (10× popcount win), VPCONFLICTD (parallel histogram), VPCOMPRESSQ (branch-free packing). Recommended: AVX-512 backend for popcount, vector distance, and Roaring intersection in v0.2; broader AVX-512 coverage in v0.3+.

5. **Where do graph algorithms live?** `tokenfs-algos::graph` is a clean home for CSR walk and BFS, but topo sort and connected components are only loosely SIMD-accelerable. Recommended: ship the graph traversal primitives (CSR walk, BFS frontier, DFS preorder iterator) with SIMD-friendly inner loops, but leave full graph algorithms (MST, PageRank, SCC) to consumer crates that can use rayon for parallelism.

6. **Do we expose a unified "image-build profile" API?** A consumer running through tokenfs_writer would benefit from a single call that streams an image's content, applies all the primitives (fingerprint, content-class, content-address, hash, statistics) in one pass, and returns aggregated results. Recommended: yes, but in tokenfs-paper's `tokenfs_writer`, not in tokenfs-algos. The crate provides primitives; the writer composes them.

7. **PyO3 binding shape.** AGENTS.md notes Python bindings are a future workspace member. The right shape: take large `numpy.ndarray` / `bytes` inputs, return aggregated stats / fingerprints / sketches. Do not expose per-block kernels at the Python boundary — the FFI overhead would dominate. Recommended: design the Python wrapper around batch APIs from day one; don't expose the internal primitive surface directly.

---

*This doc is a living plan. Update after each new primitive lands, particularly to record real benchmark numbers replacing the throughput estimates in §4. The estimates are conservative best-known-from-literature numbers; reality on the i9-12900K reference box may differ ±50%.*

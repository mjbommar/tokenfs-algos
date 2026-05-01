# Similarity And Approximation Roadmap

Date: 2026-05-01.

This roadmap extends `tokenfs-algos` from exact byte-stream primitives into
similarity search, fuzzy fingerprints, approximate indexes, and hardware-backed
dense distance kernels. The goal is not to add clever algorithms for their own
sake. The goal is to give TokenFS and downstream consumers a tested buffet of
low-level methods for answering:

- how similar are these byte distributions, sketches, or fingerprints?
- does this random block look like any calibrated profile?
- are these files or extents related even after insertion, deletion, or shift?
- can a cheap approximate method find candidates before exact comparison?
- which method wins on this CPU, this cache hierarchy, and this workload?

The rule stays the same as the rest of the crate: scalar truth first, pinned
optimized kernels second, planner defaults only after stable benchmark evidence.

## Current State

Implemented today:

- scalar/reference distribution distances in `divergence`;
- byte-distribution matching in `distribution`;
- dense `u32` sketch distance functions for cosine, L2, normalized L2, and
  Jensen-Shannon-style comparison;
- fixed-size CRC32C hash-bin sketches for 2-grams and 4-grams;
- approximate sketches: Misra-Gries, Count-Min Sketch, and CRC32C hash bins;
- sampled/adaptive strategies in histogram planning and large fingerprint
  extent processing;
- primitive benchmark labels for byte distribution, dense sketch distance, and
  n-gram sketch workloads.

Not implemented yet:

- true NumKong-style SIMD dense distance kernels;
- MinHash, b-bit MinHash, one-permutation MinHash, or LSH indexes;
- CTPH, ssdeep-like, TLSH-like, sdhash-style, or SimHash fingerprints;
- high-performance membership/cardinality structures such as Bloom, xor,
  quotient/cuckoo filters, and HyperLogLog;
- planner policies that choose approximate similarity methods;
- ARM NEON or SVE/SVE2 implementations for these families.

## Public Contract

The similarity layer should follow the same ladder as histogram and
fingerprint:

```rust
tokenfs_algos::similarity::distance::cosine_u32(a, b)
tokenfs_algos::similarity::distance::l2_u32(a, b)
tokenfs_algos::similarity::distance::jensen_shannon_u64(a, b)
tokenfs_algos::similarity::nearest::nearest_reference(query, refs, metric)

tokenfs_algos::similarity::kernels::scalar::cosine_u32(a, b)
tokenfs_algos::similarity::kernels::avx2::cosine_u32(a, b)
tokenfs_algos::similarity::kernels::neon::cosine_u32(a, b)
```

Default public functions may dispatch through a planner. Pinned kernel paths
must remain available for reproducibility, paper replication, forensic use, and
benchmark archaeology.

The contract for every hot primitive:

- pure function over slices or fixed-size structs;
- no heap allocation in hot paths;
- scalar baseline always available;
- optimized kernels match scalar exactly, or document numeric tolerance;
- stable benchmark labels;
- target-feature availability visible at the API boundary;
- planner explanation records method, backend, confidence, and fallback reason.

## Phase 1: Dense Distance Kernels

Dense distances are the foundation. They support byte distributions, n-gram
sketches, calibrated profile matching, fingerprint comparison, and approximate
candidate reranking.

Reference scalar functions:

- dot product;
- L1 / Manhattan distance;
- L2 and squared L2;
- normalized L2 over count distributions;
- cosine similarity and cosine distance;
- total variation;
- Hellinger distance;
- Jensen-Shannon distance;
- chi-squared and triangular discrimination;
- Kolmogorov-Smirnov statistic for ordered byte histograms.

Types to cover:

- `u8`;
- `u16`;
- `u32`;
- `u64` for byte histograms and exact counts;
- `f32` for normalized vectors and learned/calibrated profile embeddings.

Important fixed sizes:

- 256 bins for byte histograms;
- 1024 bins for compact n-gram sketches;
- 4096 bins for richer n-gram sketches;
- compact fixed-length fingerprint vectors once the weighted F22/F23 layout is
  formalized.

Pinned kernel families:

- `similarity::kernels::scalar`;
- `similarity::kernels::sse2`;
- `similarity::kernels::sse42` where CRC/hash related work benefits;
- `similarity::kernels::avx2`;
- `similarity::kernels::avx512`;
- `similarity::kernels::neon`;
- `similarity::kernels::sve`;
- `similarity::kernels::sve2`.

Implementation order:

1. scalar reference;
2. AVX2 for `u32` and `f32` dot/L2/cosine;
3. AVX2 fixed-size 256-bin paths for byte histograms;
4. NEON parity for `u32` and `f32` dot/L2/cosine;
5. AVX-512 once AVX2 has stable wins;
6. SVE/SVE2 only after we can run parity and benchmark gates on real hardware.

Correctness gates:

- identical vectors have zero distance where expected;
- symmetric metrics are symmetric;
- known tiny vectors match hand-computed values;
- randomized SIMD parity against scalar;
- tolerance is explicit for floating-point SIMD reductions;
- length mismatch behavior is stable;
- empty-vector behavior is stable and documented.

Benchmarks:

- `bench-similarity-distance`;
- `bench-sketch-distance`;
- `bench-distribution-distance`;
- `bench-nearest-reference`;
- fixed-size 256/1024/4096-bin cases;
- cold-cache and hot-cache modes;
- alignment sweep at offsets 0, 1, 3, 7, and 31;
- thread scaling where the caller compares many references.

## Phase 2: Fuzzy Fingerprints And CTPH

Fuzzy fingerprints serve file-level and region-level similarity. They are not a
replacement for byte histograms or F22 block fingerprints. They answer a
different question: "are these byte streams related despite shifts, insertions,
or local edits?"

Prototype families:

- ssdeep-like rolling CTPH;
- TLSH-like bucketed locality-sensitive digest;
- sdhash-style statistically selected features;
- SimHash over byte, n-gram, and structure features;
- compact F22/F23-derived weighted similarity fingerprints.

Expected fit:

- ssdeep-like CTPH: related files and shifted regions;
- TLSH-like digest: whole-file or large-region fuzzy identity;
- sdhash-style features: sparse feature matching across related content;
- SimHash: compact coarse similarity over feature vectors;
- F22/F23-derived fingerprint: TokenFS selector and planner decisions.

APIs should distinguish block, region, and stream:

```rust
similarity::fuzzy::ctph::digest(bytes, config)
similarity::fuzzy::tlsh_like::digest(bytes, config)
similarity::fuzzy::simhash::digest_features(features)
similarity::fuzzy::compare(left, right, metric)
```

Correctness and quality gates:

- stable digests for identical input;
- near-identical files rank closer than unrelated files;
- insertion/deletion/shift fixtures preserve expected similarity;
- random unrelated pairs have low false-positive rate;
- digest compare functions are deterministic and allocation-free;
- quality gates report recall, precision, false positives, and candidate
  reduction ratio rather than pretending approximate methods are exact.

Benchmarks:

- digest build throughput;
- digest compare latency;
- pairwise matrix comparison;
- shifted-file fixtures;
- source tree and tarball mutations;
- Ubuntu ISO slices;
- Magic-BPE samples;
- F21/F22 extents;
- streaming window update cost.

## Phase 3: MinHash, SimHash, And LSH

MinHash and LSH are the most promising approximate path for n-gram sets and
populated sketch bins. They are useful before dense reranking:

```text
bytes
  -> 2-gram or 4-gram features
  -> MinHash or SimHash signature
  -> LSH candidate lookup
  -> dense distance rerank
  -> planner/selector decision
```

MinHash variants:

- classic k-minimum MinHash;
- b-bit MinHash for compact signatures;
- one-permutation MinHash;
- densified one-permutation MinHash;
- weighted MinHash candidate for count sketches and byte distributions.

SimHash variants:

- bit-signature over n-gram hashes;
- weighted SimHash over count/sketch bins;
- F22 feature SimHash using entropy, run, byte-class, and top-K features.

LSH variants:

- banded LSH over MinHash signatures;
- Hamming-radius buckets for SimHash;
- populated-bin bitset prefilter;
- two-stage lookup: cheap approximate candidates, then exact/dense distance.

APIs:

```rust
similarity::minhash::Signature<K>
similarity::minhash::from_ngrams::<N, K>(bytes)
similarity::simhash::from_features(features)
similarity::lsh::Index::insert(id, signature)
similarity::lsh::Index::query(signature)
```

Correctness and quality gates:

- controlled Jaccard fixtures produce expected MinHash similarity;
- b-bit and one-permutation variants stay within documented error bounds;
- SimHash Hamming distance tracks controlled feature perturbations;
- LSH recall meets configured target at chosen thresholds;
- candidate reduction ratio is reported alongside recall;
- deterministic seeding for reproducible tests and papers.

Benchmarks:

- signature build latency;
- index build time;
- query latency;
- memory footprint;
- recall and precision by similarity threshold;
- candidate reduction ratio;
- dense rerank cost after LSH;
- streaming update where supported.

## Phase 4: Approximate Data Structures

These structures support fast "seen before?", "how many distinct?", "what are
the heavy features?", and "which references are possible candidates?"

Candidate structures:

- Bloom filter;
- blocked Bloom filter;
- xor filter;
- quotient filter;
- cuckoo filter;
- HyperLogLog;
- SpaceSaving heavy hitters;
- populated-bin bitsets;
- reservoir sampling for streams;
- optional ANN-style reference indexes in a sibling crate if they require
  allocation, persistence, or large mutable state.

Expected uses:

- Bloom/xor/filter family: quick membership and negative checks for n-gram
  features or known block signatures;
- HyperLogLog: approximate n-gram cardinality and uniqueness signals;
- SpaceSaving: stronger heavy-hitter alternative to Misra-Gries;
- populated-bin bitsets: SIMD-friendly prefilter before dense distance;
- reservoir sampling: bounded streaming summaries for large files.

Rules:

- fixed-size, allocation-free variants belong in `tokenfs-algos`;
- large mutable indexes probably belong in a sibling crate;
- every structure needs false-positive/error-bound tests;
- every structure needs memory-footprint reporting.

## Hardware Acceleration Plan

Hardware acceleration is required, but it must be honest. A backend is not
"implemented" until it has parity tests, benchmark labels, and hardware report
visibility.

### x86_64

First targets:

- SSE2 for baseline packed arithmetic where it is materially useful;
- SSE4.2 for CRC32C sketch paths;
- AVX2 for dense dot/L2/cosine, 256-bin histogram distance, and sketch
  distance;
- AVX-512 for wider dense reductions and mask-heavy comparisons after AVX2
  establishes the kernel shape.

Measurement requirements:

- cycles/byte;
- GiB/s;
- latency per comparison;
- speedup over scalar;
- speedup over compiler autovectorized scalar;
- effect of alignment;
- effect of 256/1024/4096-bin vector length;
- downclocking notes for AVX-512.

### ARM

First targets:

- NEON for dense dot/L2/cosine and histogram/sketch distance;
- CRC alternatives for hash-bin sketches where ARM CRC instructions are
  available;
- SVE/SVE2 only after real hardware or reliable CI runners are available.

Measurement requirements:

- same labels as x86_64;
- report runtime feature detection;
- report compiled feature support;
- keep scalar fallback available and tested.

### Portable SIMD

Portable SIMD or crates such as `wide` may be useful after concrete scalar,
AVX2, and NEON kernels establish the actual reduction patterns. They should not
replace direct target-feature kernels until benchmarks show that the abstraction
cost is acceptable.

## Benchmark And Report Requirements

New suites:

- `bench-similarity-distance`;
- `bench-sketch-distance`;
- `bench-fuzzy-fingerprint`;
- `bench-minhash`;
- `bench-lsh`;
- `bench-approx-structures`;
- `bench-similarity-real`;
- `bench-similarity-profile`.

Workloads:

- random high-entropy blocks;
- low-entropy zero/repeated blocks;
- medium-entropy structured binary;
- ASCII and UTF-8 text;
- source trees;
- logs;
- JSON/CSV;
- SQLite/parquet;
- compressed archives;
- Ubuntu ISO slices;
- Magic-BPE limited/shuffled samples;
- F21/F22 calibration fixtures;
- mutated file pairs with insertion/deletion/shift;
- mixed-region files at 4 KiB, 64 KiB, and 1 MiB boundaries.

Report dimensions:

- primitive family;
- algorithm;
- backend;
- workload;
- size;
- entropy class;
- alignment;
- thread count;
- CPU model;
- cache profile;
- compiled features;
- runtime features;
- cycles/byte;
- GiB/s;
- latency;
- memory footprint;
- recall/precision for approximate methods;
- candidate reduction ratio for indexes;
- planner choice, winner, and gap where applicable.

Visual outputs:

- throughput heatmaps;
- latency tables;
- winner-by-workload charts;
- planner win/miss/gap charts;
- recall/precision curves;
- candidate-reduction charts;
- memory-footprint charts;
- flamegraphs and timing tables when perf is available.

## Planner Integration

The planner should not use these methods until reports show durable wins.
Planner integration should happen in this order:

1. dense distance backend selection;
2. byte-distribution nearest-reference backend selection;
3. n-gram sketch distance backend selection;
4. MinHash/LSH prefilter for large reference sets;
5. fuzzy file/region digest selection;
6. consumer-specific defaults for FUSE, batch image building, Python, and
   kernel-adjacent callers.

Planner explanation must include:

- selected method;
- selected backend;
- input signals;
- estimated cost;
- confidence source;
- fallback reason;
- best observed benchmark kernel when available;
- gap between planner choice and best observed kernel.

Confidence sources:

- `static-rule`: deterministic rule from size/content/feature availability;
- `calibration-rule`: backed by benchmark or F21/F22/Magic-BPE calibration;
- `runtime-profile`: backed by local benchmark/autotuning data;
- `fallback`: conservative choice without enough evidence.

## Boundaries And Sibling Crates

Keep inside `tokenfs-algos`:

- pure functions over bytes, slices, fixed arrays, and compact structs;
- scalar and target-feature kernels;
- fixed-size sketches and approximate structures;
- no-std/alloc-compatible primitives where feasible;
- benchmark labels and planner explanation types.

Move to sibling crates when needed:

- large persistent indexes;
- file-system object models;
- profile databases;
- Python/Numpy bindings;
- FUSE or kernel-facing integration;
- CLI tools for building and querying large similarity indexes.

Likely future siblings:

- `tokenfs-algos-profiles`: calibrated MIME/content profile artifacts;
- `tokenfs-similarity-index`: persistent MinHash/LSH/reference indexes;
- `tokenfs-py`: PyO3/Numpy bindings over stable batch APIs;
- `tokenfs-fuse`: FUSE integration and read-pattern planning;
- `tokenfs-ffi`: C ABI for kernel-adjacent experiments.

## Acceptance Gates

A primitive family is not ready for planner use until it has:

- scalar reference implementation;
- pinned public kernel paths;
- parity or tolerance tests;
- randomized/property tests;
- benchmark labels;
- real-data benchmark rows;
- SVG/HTML report coverage;
- documented hardware support;
- documented fallback behavior;
- clear guidance on exact versus approximate semantics.

An approximate method additionally needs:

- recall/precision fixtures;
- false-positive or error-bound reporting;
- candidate-reduction reporting;
- deterministic seeds;
- quality-vs-latency charts.

Hardware backend acceptance:

- target-feature gate;
- runtime availability check where relevant;
- scalar parity tests;
- benchmark comparison against scalar and public default;
- no hidden backend selection in pinned paths;
- CPU/cache/features recorded in benchmark history.

## Suggested Implementation Order

1. Add `similarity` module with scalar dense distance APIs and tests.
2. Move or re-export existing dense distance functions under the new public
   contract while preserving old paths if needed.
3. Add AVX2 dot/L2/cosine kernels for `u32` and `f32`.
4. Add NEON dot/L2/cosine kernels behind real ARM feature gates.
5. Add fixed-size 256-bin byte-histogram distance kernels.
6. Add 1024/4096-bin sketch distance kernels.
7. Add `bench-similarity-distance` and report visuals.
8. Add MinHash and SimHash scalar prototypes with quality tests.
9. Add LSH index prototype for in-memory candidate lookup.
10. Add CTPH/TLSH-like/sdhash-like fuzzy digest prototypes.
11. Add approximate structures in priority order: SpaceSaving, Bloom/blocked
    Bloom, HyperLogLog, populated-bin bitsets, xor filter.
12. Run long real-data benchmark passes and only then update planner policy.

The first implementation target should be dense distance kernels. They are the
lowest-risk and highest-leverage foundation for everything else in this
roadmap.

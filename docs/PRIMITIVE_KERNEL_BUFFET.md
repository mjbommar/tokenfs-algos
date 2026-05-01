# Primitive Kernel Buffet

Date: 2026-04-30.

The library should expose stable high-level primitives while keeping a broad
internal buffet of kernels for benchmarking, dispatch, and future processor
specialization. The public API should not force users to know which kernel wins
on an i9 P-core, an ARM laptop, or a small VM.

## Primitive Families

| Family | Purpose | First kernels |
|---|---|---|
| Byte histogram | 256-bin byte counts for entropy and classification. | direct, local, striped, run-length, adaptive. |
| Entropy | H1 and windowed entropy over histograms. | scalar H1, min/collision/Renyi, dense exact H2, exact sparse H2..H8, hashed sketches for hot paths. |
| Run-length stats | Repetition and compressibility signals. | longest run, run count, adjacent-equal ratio. |
| Byte-class stats | ASCII/control/UTF-8/binary byte classes. | scalar classification and UTF-8 validation now, SIMD later. |
| Fingerprint | File/window fingerprints for TokenFS indexing. | F22 scalar block/extent and AVX2/SSE4.2 fused block fingerprint. |
| Divergence | Compare byte distributions across windows/files. | histogram distance, KL/JS candidates later. |
| Windows/chunks | Shared chunking and rolling state. | fixed-size windows, strided windows, Gear chunking, normalized FastCDC-style chunking. |
| Heavy hitters | Approximate frequent-byte or frequent-token summaries. | Misra-Gries top-K from F23a. |
| Hash-bin counters | Approximate n-gram distributions without full maps. | CRC32C 4-gram bins from F22/F23a; FNV-1a/mix64 scalar hashes for additional families. |
| Entropy reductions | Fast entropy from bounded integer counts. | `c * log2(c)` lookup table from F23a. |

The first production goal is still a trustworthy histogram and entropy base.
The buffet lets us test alternatives without exposing unstable internals.

## Histogram Buffet

Current and near-term histogram kernels:

| Kernel | Shape | Best observed use | Risk |
|---|---|---|---|
| `direct-u64` | One public table. | Simple baseline/reference. | Counter dependency chains. |
| `local-u32` | Local table reduced to public counts. | General high-entropy/text. | Extra reduction pass. |
| `stripe4-u32` | Four local tables. | Low-cardinality data. | Larger working set. |
| `stripe8-u32` | Eight local tables. | Low-cardinality and repeated data. | Larger L1 pressure. |
| `run-length-u64` | Count runs. | Huge runs/zero tails. | Poor high-entropy branch behavior. |
| `avx2-stripe4-u32` | **Planner placeholder, body is currently scalar four-stripe under an AVX2 target_feature gate.** | Bench-history continuity slot — preserved so a future genuine AVX2 implementation can replace it without reshaping the bench schema or planner enum. | **No active planner rule selects this** (see `planner_does_not_select_the_avx2_stripe4_placeholder` in `dispatch::tests`). Per published literature (Powturbo TurboHist shootout, Yann Collet FSE notes, Lemire's group), no AVX2 byte-histogram beats well-tuned scalar 4-/8-stripe on general inputs — VPGATHERDD, pshufb-16-bin nibble lookup, sort-then-RLE, and AVX-512 VPCONFLICTD all fall behind. Don't redirect planner rules into this kernel without bench evidence of a workload class where it dominates. |
| `adaptive-prefix-1k` | Sample first 1 KiB. | Best general stateless choice so far. | Prefix may miss later macro changes. |
| `adaptive-spread-4k` | Four 1 KiB samples. | Meso-structured blocks. | Too expensive for small reads. |
| `adaptive-chunked-64k` | Per-64 KiB prefix choice. | Macro-mixed files. | Extra per-region overhead. |
| `adaptive-sequential-online-64k` | Reconsiders choice at chunk boundaries. | Sequential reads with changing regions. | More classifier overhead. |
| `adaptive-file-cached-64k` | One spread sample reused across chunks. | Stable whole-file content. | Bad if first/global sample misses local shifts. |
| `adaptive-low-entropy-fast` | Aggressive low-cardinality/run promotion. | Zeros, sparse pages, repeated motifs. | Misclassification can hurt random/text. |
| `adaptive-ascii-fast` | Text dominance precheck. | Logs, JSON, source, CSV. | Needs SIMD byte-class backend to matter. |
| `adaptive-high-entropy-skip` | Random-looking sample skips special logic. | Compressed/encrypted/high-entropy blocks. | Sampling can hide later structure. |
| `adaptive-meso-detector` | Prefix-vs-spread disagreement detector. | Block-palette and mixed-region files. | Needs more calibration before promotion. |

The next production candidate should be a documented, non-`bench-internals`
variant of the `adaptive-prefix-1k` idea for block/stream chunks, plus a
file-level planner that can choose chunked adaptation.

The dispatch module now exposes a catalog for these histogram strategies via
`histogram_kernel_catalog()`. This is metadata only; the experimental kernel
implementations still live behind `bench-internals` until promotion criteria are
met.

## API Contexts

We should keep separate APIs for different usage contexts:

| Context | API direction | Dispatch needs |
|---|---|---|
| Block | `histogram_block(bytes)` | Cheap stateless plan. |
| File/extent | `histogram_file(bytes)` or reader-based variants. | Region sampling and chunk planning. |
| Sequential stream | `HistogramStreamState::add(bytes)`. | Reuse decisions and reclassify rarely. |
| Random small read | Specialized tiny-call path. | Avoid expensive classification. |
| Parallel file scan | Thread-local histograms plus reduction. | Chunk size, thread count, core topology. |

This split avoids forcing one function to cover contradictory performance
constraints.

## Benchmark-Driven Promotion

A kernel should not move from experimental to production until it has:

- reference parity tests against scalar public behavior;
- property tests over random and edge-length inputs;
- workload matrix coverage across content, entropy, scale, and access pattern;
- real-data coverage when available;
- clear failure modes in documentation;
- benchmark history showing why it is preferred.

This is the standard that keeps the library reliable enough for TokenFS,
FUSE/kernel integration, and Python bindings.

## F21/F22/F23a Candidates

The paper-linked primitives should move into this crate early, because they
anchor performance work to empirical claims rather than isolated throughput.
Per [Paper Lineage Naming](PAPER_LINEAGE_NAMING.md), these labels are retained
for reproducibility while public APIs use `selector`, `fingerprint`, and
`sketch`.
Per [Primitive Contracts](PRIMITIVE_CONTRACTS.md), each candidate needs a
portable scalar path, pinned kernel path, tests, and isolated benchmark rows
before it graduates from experiment to public primitive.

| Candidate | Source | Why it matters | Initial status |
|---|---|---|---|
| F22 block fingerprint | `../tokenfs-paper/tools/rust/entropy_primitives` | 8-byte per-256B block fingerprint used by paper calibration. | Scalar block/extent and AVX2/SSE4.2 fused block API migrated. |
| F21 parquet/sidecar extents | TokenFS paper corpus data | Ground truth for planner-oracle comparison over real extents. | Bench fixture hooks added; calibration data remains optional. |
| CRC32C 4-gram hash bins | F22/F23a | Approximate H4/n-gram entropy without large maps. | Scalar primitive added in `sketch`. |
| `c * log2(c)` LUT | F22/F23a | Removes repeated entropy `log2` calls for bounded counts. | Scalar API added; table specialization remains. |
| Misra-Gries top-K | F23a | Approximate heavy hitters for top-byte/token coverage. | Fixed-array primitive added in `sketch`. |

If any of these are intentionally left out, the reason should be documented
here so the crate does not silently drift away from the paper.

### Fingerprint extent bottleneck (2026-05-01)

Bench evidence (`primitive_matrix/fingerprint-extent-{auto,scalar}` on synthetic
1 MiB inputs):

| Path                | zeros | prng  | text  | runs  |
| ------------------- | ----- | ----- | ----- | ----- |
| `extent-auto` (SSE4.2 CRC32C) | 33.0 MiB/s | 32.4 MiB/s | 32.8 MiB/s | (similar) |
| `extent-scalar`     | 24.6 MiB/s | 24.2 MiB/s | 24.4 MiB/s | (similar) |

The auto path is already 33% faster than scalar from `_mm_crc32_u32` alone.
**The bottleneck is the CRC32C hash4 dependency chain**, not the histogram or
run-length passes — `_mm_crc32_u32` has 3-cycle latency and the existing code
feeds four stripes (c0/c1/c2/c3) sequentially per loop iteration.

Vectorizing histogram or transition counting in the extent body (the audit's
original intuition) would shift the bottleneck onto CRC32C anyway and add ~3×
cache traffic. The right next move is **PCLMULQDQ-parallel CRC32C** — Intel
whitepaper "Fast CRC Computation for iSCSI Polynomial Using CRC32 Instruction",
implemented in `intel-isa-l` and `folly`. Expected win: >1 GiB/s extent
throughput, which closes the F22 block-latency gap to the documented
`docs/CONSUMER_LATENCY_BUDGETS.md` 1 µs target.

Tracked separately as task #33 in the SIMD roadmap.

## Immediate Implementation Queue

1. Done: add processor profile and workload-shape types.
2. Done: add a simple histogram planner that chooses among named strategies.
3. Done: log processor profile and planner output with workload benchmarks.
4. Done: expose histogram kernel catalog metadata for dispatch and reports.
5. Done: migrate scalar F22 block/extent fingerprint surface.
6. Done: add F23a seed primitives for Misra-Gries, CRC32 hash bins, and
   entropy-from-counts.
7. Done: add run-length and byte-class primitives so classifiers can use real
   signals instead of only histogram-sample heuristics.
8. Done: add primitive contracts, pinned F22 scalar paths, paper compatibility
   aliases, structure detectors, selector signals, and isolated primitive
   benchmark/report artifacts.
9. Done: add explicit `auto` / `sse42` / `scalar` CRC32 hash-bin benchmark
   labels so runtime dispatch can be compared against pinned backends.
10. Done: move F22 AVX2 block dispatch after scalar parity and calibration
   stabilized.
11. Done: add exact H2..H8 entropy APIs for calibration and research use,
   while keeping hash-bin sketches as the allocation-free hot path.
12. Next: promote the best scalar adaptive strategy behind a public conservative
   API once the planner interface stabilizes.

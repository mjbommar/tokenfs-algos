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
| Entropy | H1 and windowed entropy over histograms. | scalar H1, chunk/window summaries. |
| Run-length stats | Repetition and compressibility signals. | longest run, run count, adjacent-equal ratio. |
| Byte-class stats | ASCII/control/UTF-8-ish/binary byte classes. | scalar classification table, SIMD later. |
| Fingerprint | File/window fingerprints for TokenFS indexing. | F22 migration, rolling/windowed later. |
| Divergence | Compare byte distributions across windows/files. | histogram distance, KL/JS candidates later. |
| Windows/chunks | Shared chunking and rolling state. | fixed-size windows, stream state. |
| Heavy hitters | Approximate frequent-byte or frequent-token summaries. | Misra-Gries top-K from F23a. |
| Hash-bin counters | Approximate n-gram distributions without full maps. | CRC32C 4-gram bins from F22/F23a. |
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
| `adaptive-prefix-1k` | Sample first 1 KiB. | Best general stateless choice so far. | Prefix may miss later macro changes. |
| `adaptive-spread-4k` | Four 1 KiB samples. | Meso-structured blocks. | Too expensive for small reads. |
| `adaptive-chunked-64k` | Per-64 KiB prefix choice. | Macro-mixed files. | Extra per-region overhead. |

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

| Candidate | Source | Why it matters | Initial status |
|---|---|---|---|
| F22 block fingerprint | `../tokenfs-paper/tools/rust/entropy_primitives` | 8-byte per-256B block fingerprint used by paper calibration. | Next migration target. |
| F21 parquet/sidecar extents | TokenFS paper corpus data | Ground truth for planner-oracle comparison over real extents. | Bench fixture support needed. |
| CRC32C 4-gram hash bins | F22/F23a | Approximate H4/n-gram entropy without large maps. | Candidate primitive. |
| `c * log2(c)` LUT | F22/F23a | Removes repeated entropy `log2` calls for bounded counts. | Candidate primitive. |
| Misra-Gries top-K | F23a | Approximate heavy hitters for top-byte/token coverage. | Candidate primitive. |

If any of these are intentionally left out, the reason should be documented
here so the crate does not silently drift away from the paper.

## Immediate Implementation Queue

1. Done: add processor profile and workload-shape types.
2. Done: add a simple histogram planner that chooses among named strategies.
3. Done: log processor profile and planner output with workload benchmarks.
4. Done: expose histogram kernel catalog metadata for dispatch and reports.
5. Next: promote the best scalar adaptive strategy behind a public conservative
   API once the planner interface stabilizes.
6. Next: add run-length and byte-class primitives so the classifier can use real
   signals instead of only histogram-sample heuristics.

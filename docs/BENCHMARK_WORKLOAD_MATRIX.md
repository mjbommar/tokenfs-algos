# Benchmark Workload Matrix

Date: 2026-04-30.

The workload matrix makes benchmark samples explicit instead of relying on
informal fixture names. Each row has axis metadata in both the Criterion ID and
the generated manifest:

- `source`: `synthetic` or `real`
- `content`: `text`, `binary`, or `mixed`
- `entropy`: declared class, such as `low`, `medium`, `high`, or `mixed`
- `scale`: `flat`, `micro`, `meso`, or `macro`
- `pattern`: generator or real-data slice label
- `access`: `block`, `sequential`, `readahead`, `random`,
  `zipfian-hot-cold`, `hot-repeat`, `cold-sweep`, `same-file-repeat`, or
  `parallel-sequential`
- `chunk`: bytes per call for chunked access patterns
- `threads`: worker count for parallel access patterns
- `bytes`: processed bytes for that benchmark row

The manifest also records measured global byte entropy and 4 KiB chunk entropy:
`h1_bits_per_byte`, `h1_4k_mean`, `h1_4k_min`, and `h1_4k_max`.
It also records the current dispatch planner recommendation as
`planned_kernel`, `planned_chunk_bytes`, `planned_sample_bytes`,
`planned_confidence_q8`, `planned_confidence_source`, and `plan_reason`.
The confidence source is one of `static-rule`, `calibration-rule`, or
`fallback`, so planner parity reports can distinguish durable calibrated rules
from conservative defaults.

## Synthetic Payloads

The quick matrix currently includes:

| Case | Content | Entropy | Scale | Purpose |
|---|---|---|---|---|
| `zeros` | binary | low | flat | Low-entropy best case for run detection and dependency-chain tests. |
| `prng` | binary | high | flat | High-entropy baseline with little local structure. |
| `ascii-text` | text | medium | micro | Text-like byte distribution and repeated syntax. |
| `repeated-random-256` | binary | medium | micro | A high-entropy-looking motif repeated across the file. |
| `block-palette-4k` | binary | mixed | meso | Per-4 KiB regions switch between zeros, PRNG, motif, and text. |
| `macro-regions` | mixed | mixed | macro | Large contiguous file regions with very different character. |
| `binary-words` | binary | medium | micro | Structured binary words with repeated byte lanes. |
| `json-lines` | text | medium | micro | Structured log/JSON-like text. |
| `csv-records` | text | medium | micro | Delimited numeric/text records. |
| `source-like` | text | medium | micro | Rust/C/Python-like code text. |
| `sqlite-like-pages` | binary | mixed | meso | Page-shaped binary records with headers and payloads. |
| `compressed-like` | binary | high | flat | High-entropy bytes with a compression-format-like header. |

Optional suites add:

- `TOKENFS_ALGOS_MIXED_REGION_SWEEP=1`: alternating zero/random/text/motif
  regions at 4 KiB, 64 KiB, and 1 MiB scales.
- `TOKENFS_ALGOS_MOTIF_SWEEP=1`: short motifs, long motifs, and periodic byte
  classes.
- `TOKENFS_ALGOS_ALIGNMENT_SWEEP=1`: equal payloads viewed at +0, +1, +3, +7,
  and +31 byte offsets.
- `TOKENFS_ALGOS_SIZE_SWEEP=1`: 64B, 256B, 1K, 4K, 8K, 16K, 64K, 1M, 16M,
  and 256M payloads. Set `TOKENFS_ALGOS_MAX_SWEEP_BYTES` to cap this locally.

When `TOKENFS_ALGOS_REAL_DATA` is set, the matrix also adds first, middle, and
last slices from that file. `TOKENFS_ALGOS_REAL_PATHS` accepts a path-list of
files or directories and samples representative source, log, JSON/CSV,
SQLite/parquet, archive, compressed, binary, and dictionary files. F21/F22 paper
fixtures can be passed with `TOKENFS_ALGOS_F21_DATA`, `TOKENFS_ALGOS_F22_DATA`,
or discovered under `../tokenfs-paper/data/`.

Magic-BPE processed-index samples are optional manual real workloads. Use
`cargo xtask bench-real-magic-bpe /nas4/data/training/magic-bpe/project/data`
and control scope with:

- `TOKENFS_ALGOS_MAGIC_BPE_LIMIT`: total sample cap;
- `TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT`: per-MIME cap;
- `TOKENFS_ALGOS_MAGIC_BPE_SAMPLE_BYTES`: bytes retained per sample, with `0`
  meaning no truncation;
- `TOKENFS_ALGOS_MAGIC_BPE_SHUFFLE` and `TOKENFS_ALGOS_MAGIC_BPE_SEED`:
  deterministic shuffle controls;
- `TOKENFS_ALGOS_WORKLOAD_MAX_INPUTS`: final payload-row cap.

Directory-backed real workloads can be limited with
`TOKENFS_ALGOS_REAL_DIR_LIMIT`.

## Access Patterns

The quick matrix runs:

| Access | Chunk | Processed bytes | Intent |
|---|---:|---:|---|
| `block` | 0 | full buffer | One whole-buffer call. |
| `sequential` | 4 KiB | full buffer | Small read loop, close to FUSE/page-sized reads. |
| `sequential` | 64 KiB | full buffer | Larger sequential read loop. |
| `random` | 1 byte | 16 KiB | Per-call overhead stress test. |
| `random` | 4 KiB | 256 KiB | Random page-like reads. |
| `readahead` | 64/128 KiB | full buffer | Larger read-ahead chunks. |
| `zipfian-hot-cold` | 4 KiB | sampled reads | Hot/cold random page reads. |
| `hot-repeat` | 64 KiB | repeated full buffer | Same bytes repeatedly, hot-cache leaning. |
| `cold-sweep` | 1 MiB | full buffer | Large sweep, cold-cache leaning. |
| `same-file-repeat` | 0 | repeated full buffer | Same file/planner-shape repeated. |

Set `TOKENFS_ALGOS_MATRIX_LEVEL=full` to increase buffer size and add 1 KiB,
8 KiB, and 16 KiB sequential reads. Set `TOKENFS_ALGOS_THREAD_SWEEP=1`,
`quick`, or `basic` to add 2-thread and 4-thread parallel sequential rows. Set
`TOKENFS_ALGOS_THREAD_SWEEP=full` or `topology` to add 2, 4, physical-core,
logical-processor, and saturated rows. The saturated point is `2 * logical`
threads and is intentionally an oversubscription stress case. Explicit numeric
lists are also accepted, for example `TOKENFS_ALGOS_THREAD_SWEEP=2,8,32`.
The full matrix level enables the topology sweep by default.

This is still a read-oriented algorithm benchmark. Write and mutation patterns
belong in the next layer once incremental update primitives exist.

## Commands

Synthetic public-default matrix:

```bash
cargo xtask bench-workloads
```

Synthetic adaptive-kernel matrix:

```bash
cargo xtask bench-workloads-adaptive
```

Synthetic plus real-data matrix:

```bash
cargo xtask bench-workloads-adaptive-real ~/ubuntu-26.04-desktop-amd64.iso
```

Short adaptive calibration matrix with thread rows:

```bash
cargo xtask bench-calibrate
```

`bench-calibrate` uses the topology thread sweep so the report can show small,
core-count, processor-count, and saturated rows on the current host.

Full matrix with threads:

```bash
TOKENFS_ALGOS_MATRIX_LEVEL=full \
  cargo xtask bench-workloads-adaptive-real ~/ubuntu-26.04-desktop-amd64.iso
```

Named suites:

```bash
cargo xtask bench-smoke
cargo xtask bench-synthetic-full
cargo xtask bench-real-iso ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-real-f21 /nas4/data/tokenfs-ubuntu/bench/cow/f22-extent-bytes.bin
cargo xtask bench-size-sweep
cargo xtask bench-alignment-sweep
cargo xtask bench-thread-topology
cargo xtask bench-planner-parity
cargo xtask bench-cache-hot-cold
cargo xtask bench-profile
```

`bench-planner-parity` intentionally measures all kernels, including
`direct-u64`, so the planner-vs-best chart has no hidden planned kernels.

Criterion filters still work, which is useful for tight iteration:

```bash
cargo xtask bench-workloads-adaptive -- \
  --sample-size 10 --warm-up-time 0.03 --measurement-time 0.03 \
  workload_matrix/adaptive/case=macro-regions
```

Compare two logged runs:

```bash
cargo xtask bench-compare \
  target/bench-history/runs/<old>.jsonl \
  target/bench-history/runs/<new>.jsonl
```

The comparison command writes:

- `summary.md`: matched/unmatched counts, median and mean change, top
  regressions/improvements, planner win/miss/gap summary, and grouped tables;
- `comparison.csv`: row-level old/new throughput and workload dimensions;
- `top-regressions.svg` and `top-improvements.svg`;
- `median-change-by-kernel.svg`, `median-change-by-entropy.svg`, and
  `median-change-by-thread.svg`.

Generate visual and tabular artifacts for a run:

```bash
cargo xtask bench-report target/bench-history/runs/<run>.jsonl
```

With no argument, `bench-report` uses the newest JSONL run. It writes:

- `summary.md`: report overview and top-throughput rows;
- `heatmap.html`: workload-by-kernel throughput heatmap, colored relative to
  the best kernel per workload, with links to the other report artifacts and an
  embedded visual gallery;
- `throughput-histogram.svg`: distribution of observed GiB/s values;
- `planner-vs-best.svg`: confusion matrix of static planner recommendation
  versus measured winner;
- `winner-counts.svg`: aggregate count of workload rows won by each kernel;
- `thread-scaling-by-kernel.svg`: median GiB/s versus thread count for
  `parallel-sequential` rows;
- `dimension-*-by-kernel.svg`: every logged workload dimension crossed against
  measured kernel;
- `dimension-*-by-thread.svg`: every logged workload dimension crossed against
  thread count for `parallel-sequential` rows;
- `timing.csv`: row-level timing table for scripts, spreadsheets, and paper
  plots.

## Logged Outputs

Every `bench-workloads*` command writes a history entry under:

```text
target/bench-history/
```

The important files are:

- `target/bench-history/latest.md`: latest human-readable summary.
- `target/bench-history/index.tsv`: append-only run index.
- `target/bench-history/runs/<timestamp>-<commit>.jsonl`: machine-readable rows.
- `target/bench-history/runs/<timestamp>-<commit>.md`: per-run Markdown summary.
- `target/bench-history/reports/<run>/`: generated report artifacts from
  `bench-report`.
- `target/tokenfs-algos/workload-manifest.jsonl`: latest workload manifest.

The JSONL rows include timestamp, git commit, dirty flag, rustc version, CPU
model, processor/cache profile, Criterion IDs, workload axes, manifest
metadata, planner recommendation, mean time, processed bytes, and GiB/s.

`target/bench-history` and `target/tokenfs-algos` are intentionally ignored by
git. Set `TOKENFS_ALGOS_BENCH_HISTORY` to write history to a persistent external
directory if you want to keep it outside `target`.

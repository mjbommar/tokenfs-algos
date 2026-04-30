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
- `access`: `block`, `sequential`, `random`, or `parallel-sequential`
- `chunk`: bytes per call for chunked access patterns
- `threads`: worker count for parallel access patterns
- `bytes`: processed bytes for that benchmark row

The manifest also records measured global byte entropy and 4 KiB chunk entropy:
`h1_bits_per_byte`, `h1_4k_mean`, `h1_4k_min`, and `h1_4k_max`.
It also records the current dispatch planner recommendation as
`planned_kernel`, `planned_chunk_bytes`, `planned_sample_bytes`, and
`plan_reason`.

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

When `TOKENFS_ALGOS_REAL_DATA` is set, the matrix also adds first, middle, and
last slices from that file.

## Access Patterns

The quick matrix runs:

| Access | Chunk | Processed bytes | Intent |
|---|---:|---:|---|
| `block` | 0 | full buffer | One whole-buffer call. |
| `sequential` | 4 KiB | full buffer | Small read loop, close to FUSE/page-sized reads. |
| `sequential` | 64 KiB | full buffer | Larger sequential read loop. |
| `random` | 1 byte | 16 KiB | Per-call overhead stress test. |
| `random` | 4 KiB | 256 KiB | Random page-like reads. |

Set `TOKENFS_ALGOS_MATRIX_LEVEL=full` to increase buffer size and add 1 KiB,
8 KiB, and 16 KiB sequential reads. Set `TOKENFS_ALGOS_THREAD_SWEEP=1` to add
2-thread and 4-thread parallel sequential rows. The full level also enables the
thread sweep.

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

Full matrix with threads:

```bash
TOKENFS_ALGOS_MATRIX_LEVEL=full \
  cargo xtask bench-workloads-adaptive-real ~/ubuntu-26.04-desktop-amd64.iso
```

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
- `target/tokenfs-algos/workload-manifest.jsonl`: latest workload manifest.

The JSONL rows include timestamp, git commit, dirty flag, rustc version, CPU
model, processor/cache profile, Criterion IDs, workload axes, manifest
metadata, planner recommendation, mean time, processed bytes, and GiB/s.

`target/bench-history` and `target/tokenfs-algos` are intentionally ignored by
git. Set `TOKENFS_ALGOS_BENCH_HISTORY` to write history to a persistent external
directory if you want to keep it outside `target`.

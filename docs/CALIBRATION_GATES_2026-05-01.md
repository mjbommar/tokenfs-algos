# Calibration Gates - 2026-05-01

`PLAN.md` now treats F21/F22 calibration as a hard gate, not an optional smoke
check.

## Commands

- `cargo test -p tokenfs-algos --features calibration --test fingerprint_f22 -- --nocapture`
- `cargo xtask bench-real-f21`
- `cargo xtask bench-real-f22`

## Test Gates

With `--features calibration`, missing paper artifacts fail the run. Without the
feature, the same tests print a skip message so regular contributor machines can
run the suite without the large local corpus.

The calibration test gates cover:

- F22 sidecar H1 drift against F21 sidecar values.
- F21-compatible feature drift for H1, H2, H3, top-16 4-gram coverage,
  run-length fraction, and byte-entropy skew.
- F21 analysis thresholds from `f21-analysis.json`:
  selector test accuracy and top-2 accuracy.
- F22 block and extent throughput sanity.

The current release-benchmark hard gates are:

- F22 block fingerprint: at most 1.8 us per 256 B block.
- F22 extent fingerprint: at least 0.93 GiB/s.

The roadmap optimization target remains 1.5 us per 256 B block once the next
block-kernel pass lands and a quiet-host baseline confirms it.

The `cargo test --features calibration` throughput check still uses relaxed
timing thresholds in debug builds because Rust test binaries are unoptimized by
default. The `xtask` benchmark gates verify Criterion release-benchmark output
and use release thresholds unless explicitly overridden by environment
variables.

## Benchmark Gates

`cargo xtask bench-real-f21` runs the paper/rootfs workload matrix, writes the
normal benchmark history/report artifacts, and fails if:

- no paper rows are present,
- the planner's selected kernel is missing from measured kernels, or
- the median planner gap exceeds `TOKENFS_ALGOS_F21_PLANNER_MAX_MEDIAN_GAP_PCT`
  (default 50%).

`cargo xtask bench-real-f22` runs the F22 fingerprint primitive matrix on the
sidecar bytes, writes report artifacts, and fails if:

- F22 real rows are absent,
- `fingerprint-block-auto` misses the block latency gate, or
- `fingerprint-extent-auto` misses the extent throughput gate.

Latest short gate runs from this implementation pass:

```text
target/bench-history/reports/1777645452-7b1846e04fb2-dirty/summary.md
F21 gate: paper_rows=252, workloads=14, wins=1, misses=13, median_gap=12.60%

target/bench-history/reports/1777645708-7b1846e04fb2-dirty/summary.md
F22 gate: best_ns_per_block=1563.7, best_extent_gib_s=1.195
```

The F22 block number is close to the 1.5 us optimization target but not stable
enough to use as a hard regression gate yet. The current hard gate is 1.8 us;
the next block-kernel optimization pass should aim to restore 1.5 us as the
default hard gate.

## Exact Reference vs Default Extent

`fingerprint::kernels::scalar::extent(bytes)` is the pinned exact reference. It
counts every CRC32C hash window for H4 and is the path used by F21/F22 drift
tests.

`fingerprint::extent(bytes)` is the product default. It computes exact H1,
run-length, top-16 coverage, and skew for every input. H4 is exact up to 64 KiB
and sampled every fourth hash window on larger extents. The current regression
bound for sampled H4 is 2.5 bits on a periodic-text fixture; this should be read
as an estimator contract, not exact parity. The split keeps FUSE and sequential
read latency bounded while preserving a reproducible scalar oracle for
calibration and forensic runs.

## Benchmark Semantics Fix

UTF-8 validation is now split into two benchmark labels:

- `byteclass-utf8-fullscan`: valid inputs only; throughput is full scanned bytes.
- `byteclass-utf8-reject-latency`: invalid inputs only; throughput bytes are the
  consumed prefix plus the error byte, so early exits no longer appear as
  impossible full-buffer GiB/s.

Chunk benchmarks now include quality labels:

- `chunk-gear-quality`
- `chunk-fastcdc-quality`

These compute `ChunkQuality` so reports can separate boundary quality work from
raw chunk iteration throughput.

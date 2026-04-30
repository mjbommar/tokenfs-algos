# Adaptive Histogram Experiments

Date: 2026-04-30.

This experiment tests cheap scalar classifiers for byte histograms. The goal is
not to expose these variants as the public API yet; it is to learn which
classification shape survives realistic file and sequential-read workloads.

All adaptive kernels are behind the `bench-internals` feature and are checked
against the public `ByteHistogram::from_block` reference in unit, property, and
release-mode tests.

## Candidates

- `adaptive-prefix-1k`: classify from the first 1 KiB, then use `local-u32`,
  `stripe8-u32`, or `run-length-u64`.
- `adaptive-prefix-4k`: same classifier, but with a first 4 KiB sample.
- `adaptive-spread-4k`: four 1 KiB samples spread across the block.
- `adaptive-run-sentinel-4k`: conservative diversion to run-length only for
  obvious huge-run inputs; otherwise use `local-u32`.
- `adaptive-chunked-64k`: process a block as 64 KiB chunks, applying the 1 KiB
  prefix classifier per chunk.

The classifier records sample length, distinct byte count, maximum byte count,
adjacent-equal count, and longest run. The current decision rule is deliberately
simple:

- run-length for huge runs or one byte dominating at least 90% of the sample;
- striped counting for low-cardinality or adjacent-equal-heavy samples;
- local `u32` counting otherwise.

## Three Levels Of Adaptation

The histogram family has three distinct adaptation levels. They should not be
collapsed into one idea, because each pays a different latency cost.

| Level | Shape | Current examples | Good fit | Main cost |
|---|---|---|---|---|
| Per-byte dynamic | Every byte updates the live counter path. | `direct-u64`, `local-u32`, striped kernels. | Tiny reads, scalar oracle, pinned reproducibility. | Counter dependencies and no global content signal. |
| Per-block sample then commit | Sample 1 KiB or 4 KiB, then pick one kernel for the whole block. | `adaptive-prefix-1k`, `adaptive-prefix-4k`, `adaptive-spread-4k`. | Stateless block APIs and homogeneous extents. | Classifier overhead and possible sample bias. |
| Per-region planning | Split into regions, sample each region, then reduce. | `adaptive-chunked-64k`, future file planner. | Macro-mixed files and large extents. | More planning calls, reduction cost, and chunk-size sensitivity. |

The default `histogram::block(bytes)` API now uses the planner. Pinned kernels
remain available under `histogram::kernels::*` for reproducible experiments and
paper calibration.

## Commands

The benchmark runs were pinned to CPU 8, a P-core on the local i9-12900K.

```bash
taskset -c 8 cargo xtask bench-adaptive -- \
  --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2 1048576

for where_ in first middle last; do
  taskset -c 8 cargo xtask bench-adaptive-real \
    ~/ubuntu-26.04-desktop-amd64.iso -- \
    --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2 \
    "${where_}/1048576"
done

taskset -c 8 cargo xtask bench-adaptive-contexts-real \
  ~/ubuntu-26.04-desktop-amd64.iso -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1 \
  file_histogram_64k

taskset -c 8 cargo xtask bench-adaptive-contexts-real \
  ~/ubuntu-26.04-desktop-amd64.iso -- \
  --sample-size 10 --warm-up-time 0.1 --measurement-time 0.1 \
  sequential_histogram_4k
```

These are directional measurements, not final release numbers.

## 1 MiB Block Results

Throughput is GiB/s.

| Case | prefix-1k | prefix-4k | spread-4k | run-sentinel | chunked-64k |
|---|---:|---:|---:|---:|---:|
| synthetic zeros | 4.33 | 4.60 | 4.61 | 4.14 | 3.08 |
| synthetic uniform cycle | 5.50 | 5.31 | 5.30 | 5.44 | 4.82 |
| synthetic PRNG | 4.24 | 4.31 | 4.25 | 4.42 | 3.77 |
| synthetic runs | 3.28 | 3.27 | 3.27 | 1.25 | 3.03 |
| synthetic text | 4.33 | 4.40 | 4.07 | 4.39 | 3.90 |
| ISO first | 3.31 | 2.10 | 3.28 | 2.03 | 2.49 |
| ISO middle | 4.44 | 4.33 | 4.28 | 4.39 | 3.97 |
| ISO last | 4.40 | 4.56 | 4.51 | 3.73 | 3.29 |

For a single block, `adaptive-prefix-1k` is the best balanced candidate. It is
near the fastest on high-entropy data and avoids the ISO-first regression seen
with the 4 KiB prefix and run-sentinel variants.

## File Context Results

The file context processes a whole input as 64 KiB chunks into one histogram.
Synthetic file cases are 4 MiB. ISO cases are 16 MiB slices.

| Case | prefix-1k | prefix-4k | spread-4k | run-sentinel | chunked-64k |
|---|---:|---:|---:|---:|---:|
| mixed-4m | 3.64 | 2.95 | 2.91 | 2.31 | 3.70 |
| prng-4m | 3.94 | 2.34 | 3.18 | 3.54 | 3.93 |
| zeros-4m | 3.24 | 2.95 | 2.95 | 3.14 | 2.90 |
| ISO first | 2.65 | 2.49 | 2.65 | 1.78 | 2.68 |
| ISO middle | 3.94 | 3.22 | 3.18 | 3.58 | 3.96 |
| ISO last | 3.50 | 2.85 | 2.92 | 2.64 | 3.50 |

For file-style work, `adaptive-chunked-64k` and `adaptive-prefix-1k` are close.
The chunked version helps when a file contains mixed regions, because each 64
KiB chunk can choose a different kernel. It is not clearly better on homogeneous
zero data.

## Sequential Read Results

The sequential context processes a whole input as 4 KiB reads into one
histogram. This approximates a FUSE or kernel caller issuing small sequential
reads.

| Case | prefix-1k | prefix-4k | spread-4k | run-sentinel | chunked-64k |
|---|---:|---:|---:|---:|---:|
| mixed-4m | 1.38 | 0.52 | 0.51 | 0.69 | 1.38 |
| prng-4m | 1.53 | 0.60 | 0.62 | 0.88 | 1.56 |
| zeros-4m | 1.07 | 0.42 | 0.42 | 0.53 | 1.10 |
| ISO first | 1.28 | 0.53 | 0.54 | 0.66 | 1.28 |
| ISO middle | 1.53 | 0.60 | 0.61 | 0.89 | 1.54 |
| ISO last | 1.47 | 0.56 | 0.56 | 0.79 | 1.42 |

For 4 KiB reads, the 4 KiB sample variants are too expensive because they
classify almost the entire read before counting it. `adaptive-prefix-1k` and
`adaptive-chunked-64k` dominate, and they behave almost identically because the
outer read size is already smaller than the chunk size.

## Recommendation

The next production candidate should be based on `adaptive-prefix-1k`.

For file-level APIs, add an explicit stateful mode that processes 64 KiB or
larger regions and permits per-region kernel choice. For sequential-read APIs,
avoid re-sampling a full small read. The right shape is likely a small state
object that observes the first few reads, caches a current kernel choice, and
only reclassifies at bounded intervals or when the recent-read signal changes
sharply.

# Primitive Benchmark Session

Date: 2026-04-30.

This session added the first isolated primitive benchmark matrix. The goal was
to measure low-level building blocks independently from the histogram workload
planner.

## Run

- run_id: `1777595436-05ecc61a82d4`
- commit: `05ecc61`
- JSONL: `target/bench-history/runs/1777595436-05ecc61a82d4.jsonl`
- report: `target/bench-history/reports/1777595436-05ecc61a82d4/`
- records: `300`
- primitive kernels: `15`
- synthetic cases: zeros, PRNG, ASCII text, long runs, repeated 64-byte motif
- sizes: 256 B, 4 KiB, 64 KiB, 1 MiB

Report artifacts:

- `heatmap.html`
- `timing.csv`
- `throughput-histogram.svg`
- `winner-counts.svg`
- `dimension-primitive-by-kernel.svg`
- `dimension-case-by-kernel.svg`
- `dimension-source-by-kernel.svg`
- `dimension-content-by-kernel.svg`
- `dimension-entropy-by-kernel.svg`
- `dimension-pattern-by-kernel.svg`
- `dimension-bytes-by-kernel.svg`

## Correctness Gates

Before this run:

- `cargo check --workspace --all-features` passed.
- `cargo test --workspace --all-features` passed.
- F22 sidecar calibration ran against the local sidecar and checked the first
  500 extents against the F21 H1 value with `< 0.05` absolute tolerance.
- Public F22 aliases match the productized `fingerprint` API.
- Pinned scalar fingerprint paths match the default path on generated blocks.
- SSE4.2 CRC32C matches the scalar CRC32C implementation when the CPU supports
  the feature.

## What The Data Teaches

Median throughput by primitive kernel:

| Kernel | Median GiB/s | Max GiB/s |
|---|---:|---:|
| `runlength-summarize` | 2.094 | 3.814 |
| `sketch-entropy-lut` | 1.126 | 4.541 |
| `sketch-crc32-hash4-auto` | 1.087 | 3.022 |
| `byteclass-classify` | 1.070 | 1.951 |
| `sketch-crc32-hash4-sse42` | 1.045 | 3.267 |
| `entropy-h1-from-histogram` | 0.969 | 4.112 |
| `structure-summarize` | 0.356 | 0.659 |
| `fingerprint-extent-auto` | 0.259 | 0.789 |
| `sketch-misra-gries-k16` | 0.230 | 2.240 |
| `sketch-count-min-4x1024` | 0.208 | 0.220 |
| `fingerprint-block-auto` | 0.168 | 0.208 |
| `selector-signals` | 0.133 | 0.344 |
| `sketch-crc32-hash4-scalar` | 0.086 | 0.092 |
| `fingerprint-extent-scalar` | 0.066 | 0.087 |
| `fingerprint-block-scalar` | 0.061 | 0.065 |

Key observations:

1. The SSE4.2 CRC32C path is load-bearing. At 1 MiB, `sketch-crc32-hash4-auto`
   was 7.7x to 33.6x faster than the scalar CRC32C path depending on content.
   The median explicit pinned `sketch-crc32-hash4-sse42` rows track `auto`
   closely, confirming that runtime dispatch is choosing the expected backend
   on this CPU. One clean-run 1 MiB text row for pinned SSE4.2 was noisy; use
   median and repeated runs for backend comparisons.
2. The fingerprint extent path inherits that win. At 1 MiB,
   `fingerprint-extent-auto` was 6.1x to 10.2x faster than pinned scalar.
3. Block fingerprints are intentionally expensive per byte because every 256 B
   block pays fixed histogram, RLE, top-K, byte-class, and hash-bin setup costs.
   Extent fingerprints amortize those costs and are the better default for
   file/extent-scale consumers.
4. `sketch-entropy-lut` and `entropy-h1-from-histogram` reach 4+ GiB/s on large
   motif/text/random cases. Entropy reduction itself is no longer the only
   budget problem; getting counts cheaply matters more.
5. `runlength-summarize` is extremely fast on low-entropy and long-run inputs,
   reaching roughly 3.6-3.7 GiB/s on large zero/run cases. It is a good planner
   sentinel.
6. `structure-summarize` and `selector-signals` are intentionally composite and
   slower. They should be used at extent/file planning boundaries, not on tiny
   FUSE read calls unless cached or sampled.
7. Tiny 256 B rows are dominated by fixed overhead. The planner should continue
   avoiding composite fingerprints and selector signals for tiny reads.

Selected 1 MiB speedups:

| Pair | Case | Auto GiB/s | SSE4.2 GiB/s | Scalar GiB/s | Speedup |
|---|---|---:|---:|---:|---:|
| CRC32 hash bins | zeros | 0.701 | 0.690 | 0.091 | 7.7x |
| CRC32 hash bins | PRNG | 2.781 | 3.267 | 0.092 | 30.1x |
| CRC32 hash bins | text | 2.950 | 0.261 | 0.092 | 32.1x |
| CRC32 hash bins | runs | 1.098 | 1.126 | 0.091 | 12.1x |
| CRC32 hash bins | motif-64 | 3.022 | 2.617 | 0.090 | 33.6x |
| fingerprint extent | zeros | 0.528 | n/a | 0.087 | 6.1x |
| fingerprint extent | PRNG | 0.785 | n/a | 0.082 | 9.6x |
| fingerprint extent | text | 0.766 | n/a | 0.081 | 9.5x |
| fingerprint extent | runs | 0.691 | n/a | 0.086 | 8.0x |
| fingerprint extent | motif-64 | 0.789 | n/a | 0.077 | 10.2x |

## Implications

- Keep scalar as the reproducibility baseline, but do not judge production
  fingerprint viability from scalar CRC32C numbers.
- Use block fingerprints for fixed 256 B calibration and academic parity.
- Use extent fingerprints or sampled extent fingerprints for file-scale
  planning.
- Use `selector::signals_from_parts` when callers already have precomputed
  `fingerprint` and `structure` summaries; this avoids recomputing composite
  signals.
- Add an AVX2 byte histogram/byteclass path before optimizing selector logic;
  many composite primitives are now bottlenecked by count/class passes.

## Next Primitive Work

1. Add AVX2 byte histogram and byteclass kernels.
2. Add real-data primitive rows from the Ubuntu ISO and F22 sidecar slices.
3. Add a primitive flamegraph profile for `selector-signals` and
   `fingerprint-extent-auto`.

# Primitive Benchmark Session

Date: 2026-04-30.

This session added the first isolated primitive benchmark matrix. The goal was
to measure low-level building blocks independently from the histogram workload
planner.

## Run

- run_id: `1777595055-d1ed85d12b10-dirty`
- JSONL: `target/bench-history/runs/1777595055-d1ed85d12b10-dirty.jsonl`
- report: `target/bench-history/reports/1777595055-d1ed85d12b10-dirty/`
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
| `runlength-summarize` | 2.075 | 3.739 |
| `sketch-entropy-lut` | 1.138 | 4.584 |
| `sketch-crc32-hash4-auto` | 1.094 | 3.274 |
| `sketch-crc32-hash4-sse42` | 1.091 | 3.377 |
| `byteclass-classify` | 1.042 | 1.662 |
| `entropy-h1-from-histogram` | 0.965 | 4.221 |
| `structure-summarize` | 0.368 | 0.656 |
| `fingerprint-extent-auto` | 0.285 | 0.782 |
| `sketch-misra-gries-k16` | 0.244 | 2.253 |
| `sketch-count-min-4x1024` | 0.208 | 0.224 |
| `fingerprint-block-auto` | 0.173 | 0.212 |
| `selector-signals` | 0.133 | 0.347 |
| `sketch-crc32-hash4-scalar` | 0.087 | 0.092 |
| `fingerprint-extent-scalar` | 0.067 | 0.086 |
| `fingerprint-block-scalar` | 0.061 | 0.066 |

Key observations:

1. The SSE4.2 CRC32C path is load-bearing. At 1 MiB, `sketch-crc32-hash4-auto`
   was 7.9x to 36.7x faster than the scalar CRC32C path depending on content.
   The explicit pinned `sketch-crc32-hash4-sse42` rows track `auto` closely,
   confirming that runtime dispatch is choosing the expected backend on this
   CPU.
2. The fingerprint extent path inherits that win. At 1 MiB,
   `fingerprint-extent-auto` was 6.1x to 9.3x faster than pinned scalar.
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
| CRC32 hash bins | zeros | 0.699 | 0.683 | 0.089 | 7.9x |
| CRC32 hash bins | PRNG | 2.757 | 3.377 | 0.092 | 29.9x |
| CRC32 hash bins | text | 3.274 | 3.041 | 0.089 | 36.7x |
| CRC32 hash bins | runs | 1.122 | 1.094 | 0.091 | 12.3x |
| CRC32 hash bins | motif-64 | 3.264 | 2.703 | 0.090 | 36.3x |
| fingerprint extent | zeros | 0.526 | n/a | 0.086 | 6.1x |
| fingerprint extent | PRNG | 0.777 | n/a | 0.082 | 9.5x |
| fingerprint extent | text | 0.782 | n/a | 0.084 | 9.3x |
| fingerprint extent | runs | 0.690 | n/a | 0.086 | 8.0x |
| fingerprint extent | motif-64 | 0.764 | n/a | 0.082 | 9.3x |

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

# Distribution And N-Gram Primitive Phase

Date: 2026-05-01.

This phase turns the Magic-BPE and MIME-distribution idea into first-class
library and benchmark surfaces. It also adds dense n-gram sketches so future
planner work can compare "what does this random block look like?" without
building maps or parsing file formats.

## Measurement Protocol

Short Criterion runs are useful for smoke tests, API regressions, and report
shape validation. They are not policy evidence by themselves. Planner tuning
should come from longer runs on a quiet host:

- pin CPU frequency or record that it was not pinned;
- run `cargo xtask bench-planner-parity-real` and `cargo xtask bench-real-magic-bpe`;
- run the primitive suites below with larger `--sample-size` and
  `--measurement-time`;
- capture `perf stat` and flamegraphs when `perf_event_paranoid` permits it;
- compare against the current baseline report
  `target/bench-history/reports/1777599655-d657bf8264a0-dirty/`;
- tune only repeated misses that survive reruns.

The report generator now includes planner confidence source in `timing.csv` and
`planner-parity.csv`. The SVG gallery already shows planner-vs-best,
planner-gap, winner-count, thread-scaling, and dimension cross-tabs.

## Byte Distribution API

New public module: `tokenfs_algos::distribution`.

Core shapes:

- `ByteDistribution`: exact 256-bin byte counts plus total bytes.
- `ByteDistributionMetric`: Jensen-Shannon, Hellinger, total variation, and KS.
- `ByteDistributionReference`: labeled calibrated reference, usually MIME/type.
- `nearest_byte_distribution`: nearest-reference lookup over calibrated
  references.

This is designed for two contexts:

- random block: compute one `ByteDistribution` and compare with references;
- stream/window: compute per-window distributions and keep nearest labels or
  distances as planner/cache metadata.

The hot API is allocation-free. Loading JSONL calibration data is kept in tools
and benchmarks, not in the core no-std library path.

## Calibration Pipeline

Existing command:

```bash
cargo xtask calibrate-magic-bpe /nas4/data/training/magic-bpe/project/data
```

Output:

```text
target/calibration/magic-bpe-byte-histograms.jsonl
```

Each JSONL row stores MIME type, sample count, byte count, H1, and 256 byte
counts. Primitive distribution benchmarks automatically load that file when
`TOKENFS_ALGOS_MAGIC_BPE_CALIBRATION` points at it; otherwise they use synthetic
reference profiles.

## N-Gram Sketches

New sketch primitives:

- `HashBinSketch<BINS>`: dense fixed-size CRC32C hash-bin sketch.
- `crc32_hash2_bins`: 2-gram hash-bin counter.
- `crc32_hash_ngram_bins<N, BINS>`: generic 1..4 gram counter.
- 256, 1024, and 4096 bin variants are benchmarked through the primitive matrix.

The 4-gram path reuses the SSE4.2 CRC32C dispatch when available, matching the
F22/F23a direction. The scalar path remains the correctness oracle.

Dense distance kernels were added for sketch comparison:

- raw L2 over `u32` dense counts;
- normalized L2 over count distributions;
- cosine distance;
- Jensen-Shannon distance over `u32` counts.

NumKong reinforces this shape: keep dense arrays, explicit metric kernels, and
clear runtime/compiled backend reporting. We are not copying its kernels, but
the API shape follows the same "metric over dense vectors" direction.

## Benchmark Suites

New xtask entries:

```bash
cargo xtask bench-distribution
cargo xtask bench-distribution-real
cargo xtask bench-ngram-sketch
cargo xtask bench-ngram-sketch-real
```

Manual Magic-BPE distribution run:

```bash
cargo xtask calibrate-magic-bpe /nas4/data/training/magic-bpe/project/data

TOKENFS_ALGOS_MAGIC_BPE_CALIBRATION=target/calibration/magic-bpe-byte-histograms.jsonl \
TOKENFS_ALGOS_PRIMITIVE_REAL=1 \
cargo xtask bench-distribution -- \
  --sample-size 20 --warm-up-time 0.05 --measurement-time 0.10 primitive_matrix
```

N-gram sketch run:

```bash
cargo xtask bench-ngram-sketch -- \
  --sample-size 20 --warm-up-time 0.05 --measurement-time 0.10 primitive_matrix
```

Generate visual reports from the latest run:

```bash
cargo xtask bench-report
```

## Planner Confidence

`HistogramPlan` now carries `confidence_source`:

- `static-rule`: rule from API/context and coarse workload metadata;
- `calibration-rule`: rule tied to benchmark or F21/F22/rootfs calibration;
- `fallback`: conservative low-confidence default.

This lets parity reports answer not only "which kernel won?" and "what did the
planner choose?", but also whether the planner choice came from a calibrated
claim or a safe default.

## SIMD Histogram Follow-Up

The AVX2 palette histogram now samples across the block instead of using only a
prefix sample. This preserves exactness and avoids a bad prefix-only case where
the first 4 KiB are low-cardinality but later regions are not. It remains a
palette kernel, not a general AVX2 byte histogram.

## Current Gates

Correctness gates added or extended:

- byte distribution zero-distance and nearest-reference tests;
- n-gram sketch observation/count invariants;
- generic 4-gram hash bins match the pinned 4-gram path;
- SSE4.2 n-gram bins match scalar when the CPU supports SSE4.2;
- dense distance kernels return zero for identical vectors and positive values
  for disjoint vectors;
- planner confidence source tests cover calibration and fallback rules.

## What Not To Tune Yet

Do not retune planner policy from the new short primitive rows alone. The next
policy changes should wait for longer parity reports that include:

- Magic-BPE references;
- F21/F22/rootfs extents;
- Ubuntu ISO slices;
- source/log/JSON/CSV/database/archive/compressed files;
- size, alignment, cache, and topology thread sweeps.

## Bounded Smoke Results

These runs validate the harness and give rough order-of-magnitude data. They
used very short measurement windows, so treat them as smoke data, not final
policy evidence.

Magic-BPE calibration:

- command: `TOKENFS_ALGOS_MAGIC_BPE_LIMIT=128
  TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT=8 cargo xtask calibrate-magic-bpe
  /nas4/data/training/magic-bpe/project/data`
- output: `target/calibration/magic-bpe-byte-histograms.jsonl`
- result: 56 MIME byte-histogram references from 128 samples.

Distribution primitive run:

- run: `target/bench-history/runs/1777602469-30d1e96a7888-dirty.jsonl`
- report: `target/bench-history/reports/1777602469-30d1e96a7888-dirty/`
- records: 60;
- workloads: 20;
- kernels: 3;
- fastest row: `distribution-nearest-byte-hellinger` on 1 MiB ASCII text at
  3.95 GiB/s.

Observed distribution takeaways:

- Hellinger is materially cheaper than Jensen-Shannon in this implementation.
  On 1 MiB text, Hellinger nearest-reference ran at 3.95 GiB/s while
  Jensen-Shannon nearest-reference ran at 3.71 GiB/s in the short run.
- JSD remains useful because it is smoother and familiar for probability
  distributions, but it should be reserved for calibration/offline comparison
  or after a cheaper prefilter when latency is tight.
- Long-run/zero-heavy distributions are slower in byte-distribution comparison
  because the count generation dominates and the useful work per byte is low.

N-gram sketch run:

- run: `target/bench-history/runs/1777602512-30d1e96a7888-dirty.jsonl`
- report: `target/bench-history/reports/1777602512-30d1e96a7888-dirty/`
- records: 160;
- workloads: 40;
- kernels: 8;
- fastest row: `ngram-hash2-bins1024` on 1 MiB PRNG at 2.06 GiB/s.

Observed n-gram takeaways:

- 2-gram sketches are currently much cheaper than 4-gram sketches. The best
  2-gram rows were around 2 GiB/s on 1 MiB PRNG/text/motif data.
- Bucket count is not monotonic in short runs. 1024 bins often beat 256 bins on
  larger inputs, while 4096 bins can lose to both because clearing and touching
  the larger dense table costs more.
- Dense cosine and normalized L2 are cheap enough to use after sketching.
  Dense JSD is substantially more expensive and should be treated like byte JSD:
  useful for calibrated comparison, not a first-pass prefilter.

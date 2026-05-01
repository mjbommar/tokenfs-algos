# Planner Parity, Magic-BPE, And Distribution Calibration

Date: 2026-04-30.

This session added planner-parity reports, the first AVX2 histogram kernel, a
public planner explanation path, optional Magic-BPE data support, and
byte-distribution divergence primitives.

## What Changed

- `histogram::explain_block(bytes, profile)` now returns the histogram, selected
  plan, and block signals. The signals include sample size, distinct byte count,
  top-byte count, ASCII/text bytes, adjacent equal pairs, longest run, content,
  entropy, and entropy scale.
- `histogram::block(bytes)` remains the ergonomic planned path. Pinned kernels
  remain available through `histogram::kernels::*`, including
  `avx2_palette_u32`.
- `xtask bench-report` now writes `planner-parity.csv` and SVG charts that show
  winner, planned kernel, planned-vs-best gap, thread scaling, winner counts,
  and every workload dimension cross-tab.
- `bench-real-magic-bpe` adds `/nas4/data/training/magic-bpe/project/data` as an
  optional manual workload source. It is limited and shuffled with
  `TOKENFS_ALGOS_MAGIC_BPE_LIMIT`, `TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT`,
  `TOKENFS_ALGOS_MAGIC_BPE_SHUFFLE`, and `TOKENFS_ALGOS_MAGIC_BPE_SEED`.
- `calibrate-magic-bpe` writes MIME-grouped byte histograms to
  `target/calibration/magic-bpe-byte-histograms.jsonl`.
- The divergence module now has byte-count distances for total variation, KS,
  KL, Jensen-Shannon distance, Hellinger distance, and triangular
  discrimination.

## AVX2 Histogram Kernel

The new `avx2-palette-u32` kernel is an exact low-cardinality fast path:

- sample up to 4 KiB;
- if the sample has no more than 16 distinct bytes, count each palette byte with
  AVX2 `cmpeq` + `movemask` + popcount;
- count tails and non-palette bytes exactly;
- fall back to the scalar local `u32` histogram when the sample has too many
  distinct bytes.

This is intentionally not a general AVX2 histogram. AVX2 lacks byte scatter, so
random/high-cardinality bytes still favor scalar direct/local/striped tables.
The palette path is useful for zeros, runs, motifs, low-cardinality binary, and
some text-like blocks.

## Planner Tuning

The first bounded planner-parity run on synthetic workloads showed large misses
on low-entropy and structured slices:

- run: `target/bench-history/runs/1777598502-d657bf8264a0-dirty.jsonl`
- report: `target/bench-history/reports/1777598502-d657bf8264a0-dirty/`
- largest miss: 656.204% on zero-filled random 4 KiB reads, where the planner
  chose `direct-u64` but `run-length-u64` won.

After policy tuning:

- run: `target/bench-history/runs/1777598919-d657bf8264a0-dirty.jsonl`
- report: `target/bench-history/reports/1777598919-d657bf8264a0-dirty/`
- same-case average gap improved from 144.136% to 0.535%;
- same-case max gap improved from 656.204% to 4.567%.

The first bounded Magic-BPE run found the old harness bug where random 4 KiB
reads over tiny files produced zero-byte workloads. The harness now drops any
access pattern whose effective processed byte count is zero.

Magic-BPE tuned report:

- run: `target/bench-history/runs/1777599655-d657bf8264a0-dirty.jsonl`
- report: `target/bench-history/reports/1777599655-d657bf8264a0-dirty/`
- records: 459;
- workloads: 27;
- kernels: 17;
- average planner gap: 4.870%;
- largest row in that short run: a noisy FLAC whole-block row where `stripe8`
  underperformed `stripe4`.

An isolated FLAC whole-block rerun put `stripe8-u32` back on top, so the policy
keeps high-entropy meso whole blocks on `stripe8-u32` for now. That row needs a
longer pinned-frequency run before finer tuning.

## Magic-BPE Commands

Calibration:

```bash
TOKENFS_ALGOS_MAGIC_BPE_LIMIT=24 \
TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT=3 \
cargo xtask calibrate-magic-bpe /nas4/data/training/magic-bpe/project/data
```

Bounded planner parity:

```bash
TOKENFS_ALGOS_MAGIC_BPE_LIMIT=6 \
TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT=1 \
TOKENFS_ALGOS_WORKLOAD_ACCESS='block,sequential-65536,random-4096' \
cargo xtask bench-real-magic-bpe /nas4/data/training/magic-bpe/project/data -- \
  --sample-size 10 --warm-up-time 0.005 --measurement-time 0.005 workload_matrix
```

Generate a report from a saved run:

```bash
cargo xtask bench-report target/bench-history/runs/<run>.jsonl
```

## Distribution Fingerprinting Direction

The byte-histogram calibration path is a first step toward fast MIME/type
comparison for random disk blocks or streaming bytes:

- byte histograms are cheap, exact, and work for small random blocks;
- Jensen-Shannon and Hellinger are better default distances than raw KL because
  they are symmetric and handle zero-heavy distributions with smoothing;
- KS is useful as a cheap shape check over cumulative byte distributions;
- total variation is cheap and easy to reason about.

For 2-grams and 4-grams, a full dense table becomes expensive. The next likely
primitive is a fixed-size hash-bin sketch:

- CRC32C or another hardware-friendly hash over n-grams;
- power-of-two bucket count, e.g. 256, 1024, or 4096 bins;
- compare sketches with SIMD-friendly dense distance kernels;
- optionally add MinHash/LSH over the set of populated bins for coarse nearest
  neighbor lookup before exact distance checks.

This gives a fast path for "what does this block look like?" without building
maps or parsing file formats. The calibrated MIME histograms become references;
random blocks or streaming windows become query fingerprints.

## Profiling Status

Profiling attempt:

```bash
cargo xtask profile-primitives -- \
  --sample-size 10 --warm-up-time 0.005 --measurement-time 0.005 \
  primitive_matrix/histogram-avx2-palette-u32
```

This wrote `target/profiles/1777599771-primitive-perf-stat.txt`, but hardware
cycle counters were reported as unsupported for this user/host setup.

Flamegraph attempt:

```bash
cargo xtask profile-primitives-flamegraph -- \
  --sample-size 10 --warm-up-time 0.005 --measurement-time 0.005 \
  primitive_matrix/histogram-avx2-palette-u32
```

This failed before sampling because `/proc/sys/kernel/perf_event_paranoid` is
`4`. No SVG flamegraph was generated. To capture flamegraphs on this machine,
rerun on a host/session with perf access, or lower the setting/capability for
the profiling process.

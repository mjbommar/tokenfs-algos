# Benchmark Baseline: a533d04

Date: 2026-05-01.
Commit: `a533d0404b81`.

This note records the first longer benchmark baseline for the histogram planner,
distribution matching, n-gram sketches, Magic-BPE workloads, and F22/rootfs
paper fixtures. It separates stable signals from noisy rows so planner policy is
tuned from durable misses only.

## Clean Baseline Runs

Magic-BPE calibration:

```bash
TOKENFS_ALGOS_MAGIC_BPE_LIMIT=512 \
TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT=16 \
TOKENFS_ALGOS_MAGIC_BPE_SHUFFLE=1 \
cargo xtask calibrate-magic-bpe /nas4/data/training/magic-bpe/project/data
```

Output:

```text
target/calibration/magic-bpe-byte-histograms.jsonl
```

Result: 97 MIME byte-histogram calibration rows from 512 processed-index
samples.

Histogram primitive real-data run:

```text
target/bench-history/runs/1777604152-a533d0404b81.jsonl
target/bench-history/reports/1777604152-a533d0404b81/
```

Planner parity reduced real/synthetic run:

```text
target/bench-history/runs/1777604419-a533d0404b81.jsonl
target/bench-history/reports/1777604419-a533d0404b81/
```

Distribution real-data run:

```text
target/bench-history/runs/1777604465-a533d0404b81.jsonl
target/bench-history/reports/1777604465-a533d0404b81/
```

N-gram sketch real-data run:

```text
target/bench-history/runs/1777604558-a533d0404b81.jsonl
target/bench-history/reports/1777604558-a533d0404b81/
```

Magic-BPE workload run:

```text
target/bench-history/runs/1777605284-a533d0404b81.jsonl
target/bench-history/reports/1777605284-a533d0404b81/
```

## Comparison Reports

The comparison reporter now writes a durable directory, not only terminal output.
It groups deltas by kernel, workload, size, entropy class, and thread count, and
adds a planner summary with wins, misses, missing planned kernels, and gap.

Clean planner comparison against the previous dirty baseline:

```bash
cargo xtask bench-compare \
  target/bench-history/runs/1777598502-d657bf8264a0-dirty.jsonl \
  target/bench-history/runs/1777604419-a533d0404b81.jsonl
```

Report:

```text
target/bench-history/comparisons/1777598502-d657bf8264a0-dirty--1777604419-a533d0404b81/
```

Clean Magic-BPE comparison against the previous dirty baseline:

```bash
cargo xtask bench-compare \
  target/bench-history/runs/1777599655-d657bf8264a0-dirty.jsonl \
  target/bench-history/runs/1777605284-a533d0404b81.jsonl
```

Report:

```text
target/bench-history/comparisons/1777599655-d657bf8264a0-dirty--1777605284-a533d0404b81/
```

Post-rule F22 comparison:

```text
target/bench-history/comparisons/1777605862-a533d0404b81-dirty--1777606215-a533d0404b81-dirty/
```

## F22 Planner Finding

The clean baseline preserved the broad paper-extent rule:

```text
paper/rootfs mixed or medium entropy, large extent -> stripe8-u32
```

That remains reasonable for whole-block and sequential large slices. The stable
miss was narrower: random 4 KiB reads from F22/rootfs paper extents should not
fall into the broad stripe8 rule.

Before the rule change, F22 random 4 KiB rows planned `stripe8-u32`:

| Slice | Old Planner | Winner | Gap |
|---|---|---|---:|
| first | stripe8-u32 | local-u32 | 7.65% |
| middle | stripe8-u32 | stripe4-u32 | 45.43% |
| last | stripe8-u32 | avx2-palette-u32 | 18.15% |

The planner now uses `direct-u64` for this narrow random-4K paper/rootfs case.
Post-rule report:

```text
target/bench-history/runs/1777606215-a533d0404b81-dirty.jsonl
target/bench-history/reports/1777606215-a533d0404b81-dirty/
```

Post-rule random rows:

| Slice | Planner | Winner | Gap |
|---|---|---|---:|
| first | direct-u64 | direct-u64 | 0.00% |
| middle | direct-u64 | stripe4-u32 | 21.36% |
| last | direct-u64 | local-u32 | 8.84% |

This is an improvement over routing random 4 KiB F22/rootfs reads to stripe8,
but not a final policy. The middle and last slices still show that the best
pinned kernel varies by local extent shape. The next durable policy needs either
more signal in the workload descriptor or a cheap per-region/random-read
classifier; it should not blindly switch all F22 random reads to stripe4.

## Stable Signals

- The report set now has usable visual artifacts for every major axis:
  planner-vs-best, largest planner gaps, winner counts, throughput histogram,
  thread scaling, and per-dimension cross-tabs.
- `bench-compare` is now suitable for commit-to-commit tracking because it
  reads full benchmark JSONL rows and writes CSV, Markdown, and SVG artifacts.
- Distribution nearest-reference lookup confirms that Hellinger is the cheaper
  first-pass byte-distribution metric in the current implementation.
- N-gram `hash2` sketches are substantially cheaper than `hash4` sketches in the
  current scalar/SSE4.2 setup.
- Dense JSD-style distance kernels are expensive enough on tiny rows that they
  should be an offline metric or second-stage metric after a cheaper prefilter.
- F22/rootfs random 4 KiB reads are a different policy problem than whole-block
  or sequential F22/rootfs extents.

## Noisy Or Incomplete Signals

- Criterion inline "change" lines compare against whatever local Criterion
  baseline exists under `target/criterion`; use benchmark-history comparisons
  for project-level interpretation.
- Short rows with 10 samples and 20-30 ms measurement windows still show large
  variance, especially tiny inputs and oversubscribed thread counts.
- Parallel F22/rootfs rows are not stable enough yet to tune broad topology
  policy. The winner changes across slice and thread count.
- Magic-BPE limited/shuffled runs are useful for coverage and report shape, but
  should be repeated with larger limits before making MIME-specific dispatch
  rules.
- The post-rule F22 report is dirty because it includes the new planner rule and
  benchmark tooling edits.

## Profiling Status

Initial profiling was blocked by the host kernel setting:

```text
/proc/sys/kernel/perf_event_paranoid = 4
```

On 2026-05-01, `sudo` was available for a temporary profiling window. The
setting was lowered to `1`, `perf stat` and flamegraph capture were run, and
the setting was restored to `4` afterward.

Short primitive profile artifacts:

```text
target/profiles/1777636510-primitive-perf-stat.txt
target/profiles/1777636643-primitive-flamegraph.svg
target/profiles/1777636643-primitive-direct-flamegraph.svg
target/profiles/1777636643-primitive-driver-flamegraph.svg
```

The Criterion flamegraphs are valid SVGs but still include harness/startup
frames. For primitive-level reading, use the direct driver:

```bash
TOKENFS_ALGOS_PROFILE_ITERS=500 \
cargo flamegraph \
  -o target/profiles/primitive-driver-flamegraph.svg \
  -p tokenfs-algos \
  --example profile_primitives \
  --features bench-internals \
  -- all
```

The direct-driver flamegraph spent most captured samples in
`add_block_stripe4_u32`; the all-kernel driver still showed some one-time buffer
setup. The next profiling refinement is to run one kernel per process with a
longer iteration count and ignore startup frames.

Perf stat for the short primitive matrix reported about 1.5 instructions per
cycle, roughly 2% branch-miss rate, and enough Criterion noise that the result
should be treated as a bottleneck locator, not a durable benchmark baseline.

Once profiling is enabled, profile these first:

- histogram planner parity on F22/rootfs random and parallel rows;
- n-gram `hash4` and CRC32C update loops;
- distribution nearest-reference lookup, especially dense JSD and Hellinger over
  calibrated MIME references.

## Workload Controls

Use these controls to keep real-data runs deliberate:

| Variable | Purpose |
|---|---|
| `TOKENFS_ALGOS_MAGIC_BPE_LIMIT` | Total processed-index samples to load. |
| `TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT` | Per-MIME cap for Magic-BPE samples. |
| `TOKENFS_ALGOS_MAGIC_BPE_SAMPLE_BYTES` | Bytes per Magic-BPE payload; `0` keeps full samples. |
| `TOKENFS_ALGOS_MAGIC_BPE_SHUFFLE` | Shuffle Magic-BPE samples before applying limits. |
| `TOKENFS_ALGOS_WORKLOAD_MAX_INPUTS` | Cap real workload payload count. |
| `TOKENFS_ALGOS_WORKLOAD_MAX_PAYLOADS` | Cap generated payload families. |
| `TOKENFS_ALGOS_WORKLOAD_CASES` | Select cases, for example `f22,rootfs,paper`. |
| `TOKENFS_ALGOS_WORKLOAD_ACCESS` | Select access modes, for example `block,random-4096`. |
| `TOKENFS_ALGOS_REAL_DIR_LIMIT` | Cap representative files collected from each directory. |
| `TOKENFS_ALGOS_MAX_SWEEP_BYTES` | Cap size-sweep payload size. |

The default Magic-BPE workload suite now caps payload rows at 128 unless
`TOKENFS_ALGOS_WORKLOAD_MAX_INPUTS` is explicitly set.

## Next Decisions

- Keep the F22 random 4 KiB direct rule unless longer reruns contradict it.
- Do not tune parallel F22/rootfs policy from the current data alone.
- Make the next optimized kernel only after flamegraphs or hardware counters
  point to a specific hot loop.
- Candidate bottlenecks remain AVX2/general histogram count generation,
  SSE4.2/CRC32C n-gram sketching, SIMD dense distance kernels, and calibrated
  nearest-reference lookup.

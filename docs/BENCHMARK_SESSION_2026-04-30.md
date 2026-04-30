# Benchmark Session: 2026-04-30

This records the first broad benchmark/profiling session after commit
`25b82ca` (`Expand benchmark suites and primitive variants`).

## Environment

- Host: `x86_64-unknown-linux-gnu`
- CPU: `12th Gen Intel(R) Core(TM) i9-12900K`
- Topology: 24 logical CPUs, 16 cores, 1 socket, SMT 2
- Cache: L1d 640 KiB total, L1i 768 KiB total, L2 14 MiB total, L3 30 MiB
- Relevant flags: `avx`, `avx2`, `fma`, `bmi1`, `bmi2`, `sse4_1`, `sse4_2`,
  `pclmulqdq`, `sha_ni`, `vaes`, `vpclmulqdq`, `avx_vnni`
- Rust: `rustc 1.97.0-nightly (bf4fbfb7a 2026-04-11)`, LLVM 22.1.2
- Cargo: `cargo 1.97.0-nightly (eb94155a9 2026-04-09)`

## Runs

All report paths are under `target/bench-history/reports/`.

| Suite | Run ID | Records | Workloads | Kernels | Notes |
|---|---:|---:|---:|---:|---|
| smoke matrix | `1777588591-25b82ca69e41` | 720 | 60 | 12 | Single-thread block/sequential/random sanity pass. |
| thread topology | `1777588849-25b82ca69e41` | 960 | 60 | 16 | `threads=2,4,16,24,48`, `parallel-sequential`. |
| real ISO | `1777589055-25b82ca69e41` | 816 | 48 | 16 | Ubuntu ISO first/middle/last 4 MiB slices. |
| F22 sidecar block | `1777589085-25b82ca69e41` | 48 | 3 | 16 | Uses `f22-extent-bytes.bin` as a raw real file, not parsed extents yet. |
| size sweep | `1777589256-25b82ca69e41` | 480 | 30 | 16 | 64 B through 256 MiB, block access. |
| alignment sweep | `1777589285-25b82ca69e41` | 80 | 5 | 16 | Offsets 0, 1, 3, 7, 31. |
| cache hot/cold | `1777589451-25b82ca69e41` | 576 | 36 | 16 | `hot-repeat`, `cold-sweep`, `same-file-repeat`. |
| planner parity | `1777589783-25b82ca69e41` | 848 | 53 | 16 | Full synthetic block parity, including 256 MiB cases. |

Each report contains `timing.csv`, `heatmap.html`, `throughput-histogram.svg`,
`planner-vs-best.svg`, `winner-counts.svg`, and dimension-specific SVG charts.
The thread and real-ISO reports also include `thread-scaling-by-kernel.svg` and
thread cross-tab charts.

## Profiling

- `cargo xtask bench-profile -- --sample-size 10 --warm-up-time 0.005 --measurement-time 0.005 'case=zeros.*access=block'`
  wrote `target/profiles/1777589793-perf-stat.txt`.
- The host reports `kernel.perf_event_paranoid=4`. Hardware and software perf
  events are blocked for this user, so `perf stat` only captured elapsed/user/sys
  time and no cycle/cache counters.
- Installed `flamegraph v0.6.12` and fixed `cargo xtask profile-flamegraph` to
  pass the required `bench-internals` feature. Flamegraph generation still failed
  because perf recording is blocked by `perf_event_paranoid=4`.

## What This Teaches Us

The main lesson is that the problem is not just "write the fastest histogram
kernel." The real product surface is a planner that selects among several
hardware-conscious primitives using workload context.

- There is no universal kernel winner. The best primitive changes with byte
  distribution, size, alignment, access pattern, and thread count.
- The planner needs first-class context: input length, access pattern, alignment,
  thread topology, cache/reuse expectations, and real-file/source hints.
- FUSE-like micro reads and random reads need a no-sampling path. Adaptive
  analysis can cost more than the actual histogram work for tiny reads.
- Large sequential file/image work can afford sampling and region planning, and
  benefits from specialized low-entropy, ASCII/text, and mixed-region paths.
- Thread count is not "more is better." The planner must choose thread policy by
  workload and hardware topology, and it needs a cap to avoid oversubscription.
- Benchmark visualizations are now part of the development loop. Every new
  primitive should answer: where does it win, where does it lose, and how should
  the planner learn that boundary?
- The current benchmark suite is a truth source for synthetic and real-slice
  throughput, but it is not yet the paper calibration suite until F21/F22 extents
  are parsed and compared against the oracle.

## Findings

1. Thread scaling is real but not monotonic. The thread-topology pass had median
   best throughput of 4.26 GiB/s at 2 threads, 5.75 GiB/s at 4 threads, 4.12
   GiB/s at 16 threads, 2.48 GiB/s at 24 threads, and 1.10 GiB/s at 48 threads.
   Oversubscription is actively harmful for these kernels.

2. Real ISO data benefits from parallel sequential reads. The Ubuntu ISO middle
   slice reached 12.73 GiB/s with `direct-u64` at 16 threads. Median best ISO
   throughput by access was roughly 3.39 GiB/s for single-thread block, 3.06
   GiB/s for sequential, 3.19 GiB/s for readahead, and 12.30 GiB/s for
   16-thread parallel sequential.

3. Small inputs need a direct fast path. In the size sweep, `direct-u64` won
   64 B, 256 B, and 1 KiB text/high-entropy cases. Sampling/adaptive overhead is
   too expensive for micro reads.

4. Low-entropy large buffers need their own primitive. For 256 MiB zero buffers,
   `direct-u64` was about 0.64 GiB/s, while low-entropy/run-sentinel style paths
   were around 4.5 to 4.7 GiB/s. This is a large enough gap to make low-entropy
   detection first-class in the planner.

5. Text shifts with size. `direct-u64` wins up to 4 KiB text in this run, `local-u32`
   wins many 8 KiB to 1 MiB text sizes, and `adaptive-ascii-fast` or
   `adaptive-file-cached-64k` wins the largest text cases.

6. Alignment matters. On 1 MiB high-entropy buffers, `direct-u64` dropped from
   3.98 GiB/s at offset 0 to 3.34 GiB/s at offset 31. The best kernel also
   changed by offset, so pointer alignment should be part of planner context.

7. Cache state changes winners more than raw throughput. Median best throughput
   was 3.85 GiB/s for `cold-sweep`, 3.81 GiB/s for `hot-repeat`, and 3.97 GiB/s
   for `same-file-repeat`, but winner counts changed across modes. Planner cache
   behavior should be benchmarked separately from raw kernel throughput.

8. The current planner is too coarse. Planner winner hit rates were 31.7% in the
   smoke pass, 15.1% in planner parity, 16.7% in size sweep, 3.3% in thread
   topology, 0% on ISO, and 0% on the raw F22 sidecar block run. The planned
   kernel is always present, so this is selection error, not missing coverage.

9. `adaptive-chunked-64k` is currently over-selected for threaded and real-file
   cases. On the thread-topology run it was planned for every workload but won
   only 2 of 60. On ISO it was planned for 39 of 48 workloads and won none.

10. The F22 sidecar run is useful but not yet the F21/F22 calibration test. It
    treats `f22-extent-bytes.bin` as three large raw file slices. The next step is
    to parse true extents and compare planner choices against the paper oracle.

## Planner Decisions

- Add a no-sampling micro path for small reads. Initial thresholds should be
  shaped by size sweep data: direct/simple below 4 KiB, mixed policy between
  4 KiB and 64 KiB, and adaptive/specialized policies above that.
- Add planner context for access pattern, thread count, input length, alignment,
  and cache/reuse mode. A single `bytes -> kernel` planner is not sufficient.
- Treat low-entropy detection as a core primitive for large buffers.
- Treat thread count as a policy dimension. Four worker threads beat higher
  topology settings on synthetic parallel sequential data in this run; 16 threads
  was best on real ISO slices, so file/source context matters.
- Keep kernel pinning public. The reports show enough winner churn that users
  need both ergonomic planner defaults and forensic control over exact kernels.

## Follow-Up Work

- Implement true F21/F22 extent parsing and calibration against the paper data.
- Add explicit planner tests for size thresholds, low-entropy selection, and
  threaded access selection.
- Add a profile permission check to `xtask` so flamegraph failures explain
  `perf_event_paranoid` before invoking perf.
- Run flamegraphs after enabling perf access on the host.
- Keep the full 256 MiB planner parity pass as an overnight/regression suite;
  use smaller filters for routine development.

## Planner V1 Follow-Up

After the session, planner v1 added first-class `PlanContext` fields for read
pattern, cache state, source hint, and alignment offset. It also made the
previously benchmark-only adaptive variants selectable through the public
`HistogramStrategy` enum.

Focused smoke rerun:

- Run ID: `1777591759-8dc2a97a3d43-dirty`
- Report: `target/bench-history/reports/1777591759-8dc2a97a3d43-dirty/`
- Scope: 720 records, 60 workloads, 12 smoke kernels
- Old smoke planner hit rate: 19/60 = 31.7%
- New smoke planner hit rate: 37/60 = 61.7%

The important improvement is not that the planner is done. It is that adding
context and allowing more public strategies immediately doubled smoke parity.
The remaining misses show the next planner work:

- `adaptive-prefix-4k` and `adaptive-ascii-fast` still win several cases where
  v1 chooses `direct-u64`, `local-u32`, or a broader adaptive path.
- Some medium mixed/binary motifs are still poorly modeled by simple
  entropy/content labels.
- Smoke parity improved, but the full planner-parity, thread-topology, real ISO,
  and true F21/F22 calibration suites still need reruns before treating v1 as a
  stable default policy.

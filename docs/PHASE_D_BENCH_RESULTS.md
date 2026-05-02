# Phase D Rabbit Order Bench Results (v0.3.0 baseline)

Validation run for Phase D (Rabbit Order build, runtime SIMD modularity-gain
kernel, round-based parallel build) ahead of the v0.3.0 release.

## Run metadata

- **Date**: 2026-05-02
- **Branch / commit**: `main` @ `5d28116` ("Harden calibration gates and primitive contracts")
- **Host arch**: `x86_64`
- **CPU**: 12th Gen Intel(R) Core(TM) i9-12900K (24 logical cores; P/E hybrid)
- **OS**: Linux 7.0.0-13-generic
- **Cargo command**:
  ```
  cargo bench -p tokenfs-algos --bench permutation_rcm \
      --features parallel,avx2 -- --quick rabbit
  ```
  (`--quick` reduces criterion sample count; the rabbit groups also pin
  `sample_size = 10` and `warm_up_time = 500 ms` from inside the bench
  source.)
- **Filter**: `rabbit` (excludes the RCM build and `apply_u32` groups)
- **Status**: all 16 distinct rabbit benches completed successfully; no
  panics, no aborts, no `target_feature` mismatches.

## Bench groups

### `permutation_rabbit/build` — sequential `rabbit_order` build cost

The sequential heap-driven dendrogram baseline (Sprint 47-49). One series
per `(n, avg_deg)` pair; throughput in vertices/sec.

| Workload (n, avg_deg, dir-edges) | Median wall time | Throughput | Meets <100ms@10K target? |
|---|---:|---:|---|
| n=10 000, deg=5, edges=99 932 | 17.72 ms | 564 Kelem/s | yes |
| n=10 000, deg=20, edges=399 178 | 98.02 ms | 102 Kelem/s | borderline (98 ms) |
| n=100 000, deg=5, edges=999 934 | 481.7 ms | 208 Kelem/s | n/a (above the 10K tier) |
| n=100 000, deg=20, edges=3 999 128 | 4.798 s | 20.8 Kelem/s | n/a (above the 10K tier) |

The 10 K node target (<100 ms) is met at the typical TokenFS sparse degree
(deg=5: 17.7 ms, comfortable headroom). At deg=20 the build sits right at
the 100 ms ceiling; this is consistent with the bench-file commentary that
the sequential dendrogram is roughly 100-500x heavier per vertex than RCM.

### `permutation_rabbit/modularity_gain` — runtime-dispatched SIMD inner loop

The per-neighbour modularity-gain score kernel, comparing the always-i128
`scalar` reference against the `auto` runtime dispatcher. With
`--features avx2` on this Alder Lake host the dispatcher selects AVX2.

| Neighbour batch n | scalar median | auto (AVX2) median | speedup (scalar / auto) |
|---:|---:|---:|---:|
| 10     | 18.28 ns  | 19.68 ns  | 0.93x  |
| 100    | 107.75 ns | 107.62 ns | 1.00x  |
| 1 000  | 954.00 ns | 969.60 ns | 0.98x  |
| 10 000 | 8.312 us  | 9.163 us  | 0.91x  |

The expected 2-3x AVX2 lane-width speedup is **not** observed in this run;
all four batch sizes show parity-or-slightly-slower auto vs scalar. See
the regression-candidates section below.

### `permutation_rabbit/par_build` — round-based parallel `rabbit_order_par`

The Sprint 53-55 round-based concurrent variant (`rabbit_order_par`)
against the sequential `rabbit_order` baseline at three vertex tiers,
fixed avg_deg=5.

| Workload (n, dir-edges) | Below `RABBIT_PARALLEL_EDGE_THRESHOLD` (200K)? | seq median | par median | par speedup (seq / par) |
|---|---|---:|---:|---:|
| n=10 000,    edges=99 932    | yes (fallback to seq) | 18.27 ms | 18.46 ms | 0.99x |
| n=100 000,   edges=999 934   | no                    | 441.4 ms | 500.0 ms | 0.88x |
| n=1 000 000, edges=9 999 932 | no                    | 20.63 s  | 22.70 s  | 0.91x |

The 10 K row sits below `RABBIT_PARALLEL_EDGE_THRESHOLD = 200_000`
directed edges, so the parallel routine delegates to the sequential
heap-based path. The two numbers agree to within noise (0.99x), confirming
the fallback is wired up correctly.

The 100 K and 1 M tiers run the genuine round-based parallel path, and
the parallel variant is **modestly slower** than the sequential baseline
(0.88x and 0.91x respectively). This matches the explicit performance
posture documented in the `rabbit_order_par` doc comment: the
sequential-apply phase per round bounds the achievable speedup, and on
TokenFS-typical sparse graphs at this scale "wall-clock parity or
modestly slower" is the expected outcome. The round-based variant exists
primarily as a deterministic API surface for rayon-driven pipelines, not
as a wall-clock win.

## Notable observations

The sequential `rabbit_order` baseline scales roughly linearly with edge
count at fixed n (4x more edges → 5.5x more time at n=10 000), but
super-linearly with vertex count at fixed degree (10x more vertices at
deg=5 → 27x more time, 17.7 ms → 481.7 ms). This is consistent with the
heap-update dominance noted in the implementation comments, and motivates
the upcoming SIMD inner-loop work. The `modularity_gain` kernel itself
clears 1 GElem/s at the n=1000 batch size, so per-pair score computation
is not the agglomeration-loop bottleneck on these inputs — the heap and
adjacency-fold bookkeeping is.

## Regression candidates

Two items in this run do not meet the speedup expectations called out in
the task brief and warrant a follow-up before tagging v0.3.0:

1. **AVX2 modularity-gain kernel does not beat scalar.** The brief
   anticipates a 2-3x AVX2 lane-width speedup; the measured `auto / scalar`
   ratio is parity at small/medium n (1.00x at n=100, 0.98x at n=1000) and
   slightly negative at the extremes (0.93x at n=10, 0.91x at n=10000).
   Likely root-causes worth investigating: (a) the i64 fast-path emits a
   final `i128::from(score)` widening per lane that materialises into a
   serially-dependent epilogue, eating the lane-parallel gain; (b) the
   AVX2 backend may be allocating its `Vec<i128>` output via the std
   allocator on every call and that allocator round-trip dominates at
   small n; (c) the runtime `is_x86_feature_detected!` check itself
   contributes per-call overhead that the scalar arm avoids. Suggest
   profiling with `cargo flamegraph` against the n=10 000 row first since
   it has the largest per-iteration budget. Note: `rabbit_order` itself
   uses `kernels::auto`, so the slow-by-9% kernel is also dragging down
   the `build` group — reclaiming the expected 2x would push the deg=5
   100 K build under 250 ms.

2. **Parallel `rabbit_order_par` is slower than sequential at every
   measured tier above the threshold.** At n=100 K (par 0.88x seq) and
   n=1 M (par 0.91x seq), the parallel path is a net loss on this 24-core
   host. The implementation's own doc comment explicitly anticipates
   "wall-clock parity or modestly slower" because the apply phase is
   sequential, so this is **acknowledged behaviour, not a bug**, but it is
   a regression candidate against the brief's "expect 2x+ on a 4-core
   machine". For v0.3.0 this is best left documented; the colouring-based
   conflict-free batching mentioned in the doc comment is the long-term
   fix and is out of scope for this validation run.

No bench panicked, returned NaN, or failed a target-feature check.

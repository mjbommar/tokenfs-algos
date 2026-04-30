# Histogram Kernel Experiments

Date: 2026-04-30.

These experiments compare exact byte-histogram kernels. All kernels are tested
against `ByteHistogram::from_block` before benchmark results are considered.

The benchmark command used short Criterion settings and pinned execution to CPU
8, a P-core on the local i9-12900K:

```bash
taskset -c 8 cargo xtask bench-kernels -- \
  --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2 1048576

taskset -c 8 cargo xtask bench-kernels-real ~/ubuntu-26.04-desktop-amd64.iso -- \
  --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2 first/1048576
taskset -c 8 cargo xtask bench-kernels-real ~/ubuntu-26.04-desktop-amd64.iso -- \
  --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2 middle/1048576
taskset -c 8 cargo xtask bench-kernels-real ~/ubuntu-26.04-desktop-amd64.iso -- \
  --sample-size 10 --warm-up-time 0.2 --measurement-time 0.2 last/1048576
```

These are directional results, not final release numbers.

## Kernels

- `direct-u64`: current direct scalar loop over `[u64; 256]`.
- `local-u32`: local `[u32; 256]`, reduced into `[u64; 256]`.
- `stripe4-u32`: four independent `[u32; 256]` tables, reduced at the end.
- `stripe8-u32`: eight independent `[u32; 256]` tables, reduced at the end.
- `run-length-u64`: scan runs and add whole run lengths to one counter.

## 1 MiB Throughput

| Case | direct-u64 | local-u32 | stripe4-u32 | stripe8-u32 | run-length-u64 |
|---|---:|---:|---:|---:|---:|
| synthetic zeros | 0.61 GiB/s | 0.64 GiB/s | 2.38 GiB/s | 3.41 GiB/s | 4.29 GiB/s |
| synthetic uniform cycle | 5.09 GiB/s | 5.40 GiB/s | 3.57 GiB/s | 3.61 GiB/s | 1.19 GiB/s |
| synthetic PRNG | 3.99 GiB/s | 4.47 GiB/s | 3.60 GiB/s | 3.69 GiB/s | 1.16 GiB/s |
| synthetic runs | 1.22 GiB/s | 1.24 GiB/s | 3.19 GiB/s | 3.52 GiB/s | 3.08 GiB/s |
| synthetic text | 4.01 GiB/s | 4.24 GiB/s | 3.67 GiB/s | 3.58 GiB/s | 1.17 GiB/s |
| ISO first | 0.89 GiB/s | 0.87 GiB/s | 2.80 GiB/s | 3.43 GiB/s | 2.01 GiB/s |
| ISO middle | 3.93 GiB/s | 4.26 GiB/s | 3.63 GiB/s | 3.64 GiB/s | 1.17 GiB/s |
| ISO last | 0.64 GiB/s | 0.65 GiB/s | 2.38 GiB/s | 2.91 GiB/s | 4.67 GiB/s |

## Initial Conclusions

- `local-u32` is the best general scalar candidate for high-entropy, uniform,
  PRNG, text-like, and ISO-middle inputs.
- `stripe8-u32` is much better than direct counting for low-entropy or
  low-cardinality inputs, because it breaks same-counter dependency chains.
- `run-length-u64` is excellent for huge runs, especially ISO tail data, but is
  poor on high-entropy data.
- A production scalar kernel should probably be adaptive:
  - quickly detect long runs or extremely low-cardinality regions;
  - use a run-length path for run-dominated data;
  - use `local-u32` for high-entropy data;
  - consider `stripe8-u32` for low-cardinality non-run-heavy data.

The follow-up adaptive classifier experiment is captured in
`docs/ADAPTIVE_HISTOGRAM_EXPERIMENTS.md`.

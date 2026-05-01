# Core Primitive Completion

Date: 2026-05-01.

This note records the first pass on the gaps called out after reviewing
`PLAN.md`: windows/chunking stubs, F22 AVX2 migration, exact H2..H8 entropy,
AVX2 histogram coverage, future-backend honesty, and profiling.

## Implemented

| Area | Status | Public Surface |
|---|---|---|
| Sliding windows | Implemented fixed-size overlapping n-grams, strided borrowed windows, little-endian n-gram pack/unpack, and Gear rolling hash. | `windows::ngrams`, `windows::strided`, `windows::pack_ngram_le`, `windows::GearHash64`. |
| Chunking | Implemented allocation-free Gear/FastCDC-style boundary detection with min/avg/max sizes. | `chunk::chunks`, `chunk::find_boundary`, `chunk::ChunkConfig`. |
| Exact n-gram histograms | Implemented exact sparse histograms for `1 <= N <= 8` behind `std` or `alloc`. | `histogram::ngram::NGramHistogram<N>`. |
| Exact H2..H8 | Implemented exact Shannon entropy over n-grams. | `entropy::ngram::{h2,h3,h4,h5,h6,h7,h8}`. |
| Dense H2 and conditional entropy | Implemented no-heap dense byte-pair histogram, joint H2, and conditional next-byte entropy. | `histogram::BytePairHistogram`, `entropy::joint::h2_pairs`, `entropy::conditional::h_next_given_prev`. |
| Min/Renyi entropy | Implemented min-entropy and collision/Renyi entropy reference paths. | `entropy::min`, `entropy::renyi`. |
| UTF-8 validation | Implemented scalar pinned UTF-8 validation with error offsets. | `byteclass::validate_utf8`, `byteclass::kernels::scalar::validate_utf8`. |
| Hash families | Added stable scalar FNV-1a and mix64 hash primitives. | `hash::fnv1a64`, `hash::mix64`. |
| F22 AVX2 block | Migrated a fused x86 AVX2/SSE4.2 block path and added scalar parity tests. | `fingerprint::kernels::avx2::block`. |
| AVX2 histogram candidate | Added a pinned AVX2 four-stripe histogram candidate and primitive benchmark rows. | `histogram::kernels::avx2_stripe4_u32::block`. |
| Backend honesty | Added support reporting so NEON/AVX-512/SVE/SVE2 are scalar fallback until real kernels exist. | `dispatch::backend_kernel_support`. |
| Primitive profiling | Added a direct profiling driver to reduce Criterion flamegraph noise. | `cargo run -p tokenfs-algos --example profile_primitives -- all`. |

## Interpretation

Exact H2..H8 is the right calibration and research API, but it is not the hot
path for high-cardinality data. On the short primitive run, exact H8 over random
64 KiB blocks fell to the tens of MiB/s range because every distinct n-gram
becomes sparse map work. That confirms the design split: exact entropy belongs
in calibration, tests, and offline analysis; hash-bin sketches remain the
allocation-free hot path for random disk blocks and streaming readers.

The F22 AVX2 block path is now a real fused path. It improves substantially over
scalar on 256-byte blocks and large synthetic slices, but the extent API still
uses default sub-primitives. A fully fused extent kernel is a separate piece of
work because it has different state, reduction, and chunking behavior.

The new AVX2 histogram path is deliberately exposed as a pinned kernel, not a
planner default. It is exact and reproducible, but benchmark snippets show that
specialized low-cardinality/palette paths can beat it on zeros, runs, and
motifs. x86 AVX2 does not provide a magic byte-histogram instruction; many
"SIMD histogram" designs still bottleneck on table increments, shuffles,
reductions, or conflict handling. The planner should only choose this path after
stable long-run evidence.

NEON, AVX-512, SVE, and SVE2 remain future backends. The crate can name them in
feature flags and catalog metadata, but `backend_kernel_support()` reports
scalar fallback so downstream users do not infer untested acceleration.

## Profile Artifacts

Short-run artifacts from this pass:

```text
target/profiles/1777636510-primitive-perf-stat.txt
target/profiles/1777636643-primitive-flamegraph.svg
target/profiles/1777636643-primitive-direct-flamegraph.svg
target/profiles/1777636643-primitive-driver-flamegraph.svg
```

The host setting was temporarily changed from `perf_event_paranoid=4` to `1`
for capture, then restored to `4`.

The most useful profile command is now the direct driver:

```bash
sudo sysctl kernel.perf_event_paranoid=1
TOKENFS_ALGOS_PROFILE_ITERS=500 \
cargo flamegraph \
  -o target/profiles/primitive-driver-flamegraph.svg \
  -p tokenfs-algos \
  --example profile_primitives \
  --features bench-internals \
  -- all
sudo sysctl kernel.perf_event_paranoid=4
```

For cleaner one-kernel SVGs, run the same example with one of:

```text
histogram-avx2-stripe4
fingerprint-avx2
entropy-h8-exact
```

## Clean Focused Benchmark

After committing the implementation, a focused primitive benchmark was run on
clean commit `8221a438846e`:

```text
target/bench-history/runs/1777637136-8221a438846e.jsonl
target/bench-history/reports/1777637136-8221a438846e/
```

Report artifacts include:

```text
heatmap.html
throughput-histogram.svg
winner-counts.svg
dimension-primitive-by-kernel.svg
dimension-case-by-kernel.svg
dimension-content-by-kernel.svg
dimension-entropy-by-kernel.svg
dimension-pattern-by-kernel.svg
dimension-bytes-by-kernel.svg
timing.csv
```

Short-run findings:

- F22 AVX2 beat scalar on all 20 focused fingerprint workloads, averaging about
  `2.8x` scalar throughput.
- On 1 MiB random data, exact entropy showed the expected high-cardinality cost:
  H1-from-histogram was about `4.0 GiB/s`, exact H2 about `15 MiB/s`, and exact
  H4/H8 about `7 MiB/s`.
- On 1 MiB histogram rows, `avx2-palette-u32` dominated zeros, text, runs, and
  random data in this short run; `avx2-stripe4-u32` won the repeated-motif row.
  This supports keeping the new stripe4 AVX2 path pinned and benchmarked rather
  than making it the planner default.

## Remaining Gaps

- Larger normalized FastCDC distribution tests over real files and entropy
  classes.
- Fully fused F22 extent AVX2 path.
- Public planner policy that can use the AVX2 histogram candidate only where it
  wins durably.
- AVX-512/NEON/SVE/SVE2 implementations on real hardware or CI runners.
- Longer benchmark-history runs after this commit so the new kernels appear in
  clean reports instead of short dirty profiling runs.

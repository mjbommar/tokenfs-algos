# Processor-Aware Dispatch

Date: 2026-04-30. Last revised: 2026-05-03.

> See also: [`KERNEL_SAFETY.md`](KERNEL_SAFETY.md) for the
> dispatcher's kernel-safety contract — the `auto::*` runtime
> dispatchers route through `_unchecked` per-backend siblings after
> the fallible `try_*` top-level entries validate upfront, so
> kernel-default builds reach the algorithms without any
> assertion-bearing code path.

`tokenfs-algos` should not treat CPU support as a single AVX2/NEON switch.
For byte-stream primitives, the relevant processor facts include instruction
sets, cache hierarchy, core topology, memory bandwidth, and the fixed cost of
calling small kernels repeatedly. The goal is a kernel buffet that can be
assembled at compile time or selected at runtime.

## What Matters

The current byte-histogram work showed that performance depends on:

- counter dependency chains;
- local table size and whether it fits in L1 data cache;
- chunk size relative to L2 and L3 cache;
- branch predictability for run-heavy inputs;
- reduction cost for striped and parallel histograms;
- per-call overhead for tiny random reads;
- P-core/E-core and NUMA topology for threaded file scans.

ISA extensions decide which kernels are legal. Cache and topology decide which
kernels are sensible.

## Kernel Catalog

Every implementation should advertise metadata:

| Field | Meaning |
|---|---|
| `primitive` | Histogram, entropy, byte-classification, fingerprint, etc. |
| `kernel` | Stable kernel identifier. |
| `isa` | Scalar, AVX2, AVX-512, NEON, SVE, SVE2. |
| `working_set` | Approximate hot data footprint. |
| `preferred_chunk` | Chunk-size range where the kernel is expected to work well. |
| `stateful` | Whether the kernel expects reusable stream/file state. |
| `good_for` | Workload classes where this kernel should be considered. |
| `bad_for` | Workload classes where dispatch should avoid it. |

Example:

```text
primitive: histogram
kernel: adaptive-prefix-1k
isa: scalar
working_set: small
preferred_chunk: 4 KiB+
stateful: false
good_for: high entropy, text, general block reads
bad_for: random 1-byte access
```

The catalog gives us a language for dispatch decisions, benchmark reports, and
eventual Python/FUSE diagnostics.

The first Rust catalog slice is now present in `tokenfs_algos::dispatch`:

- `histogram_kernel_catalog()` returns stable metadata for the current scalar
  histogram buffet;
- `HistogramStrategy::kernel_info()` maps a planner strategy back to catalog
  metadata;
- metadata records ISA class, working-set class, statefulness, preferred chunk
  size, classifier sample size, and private table footprint.

The catalog now includes pinned AVX2 entries where the implementation
and parity tests exist, including `avx2-stripe4-u32` for byte
histograms and the fused F22 block fingerprint path.

As of v0.4.6, real kernels exist for AVX2 (the primary x86 path),
AVX-512 (gated on `feature = "avx512"`, nightly), NEON (default on
AArch64), SSE4.1 (gated on `arch-pinned-kernels`), SSSE3 (likewise),
SHA-NI (x86 SHA hardware), and FEAT_SHA2 (AArch64 SHA hardware).
`backend_kernel_support()` distinguishes implemented backends from
feature-shaped fallbacks. SVE/SVE2 remain scalar fallback targets
until profiling evidence justifies real kernels.

The histogram planner that consumes this catalog is implemented as a rule
table in `dispatch::planner`. Architecture, named-constant convention,
trace-mode usage, and the recipe for adding a rule or threshold are
documented in `docs/PLANNER_DESIGN.md`. Read that before editing planner
rules — the audit-flagged brittlenesses (magic numbers, substring-matched
confidence sources, implicit precedence by source-line order) are
explicitly disallowed by the new design.

## Compile-Time Families

We should support multiple build families:

| Family | Purpose |
|---|---|
| `portable` | Scalar, deterministic, broad compatibility. |
| `native` | `target-cpu=native` for local research and deployment builds. |
| `x86_64-v3` | AVX2/BMI2/popcnt-era baseline for modern x86 wheels/binaries. |
| `x86_64-v4` | AVX-512-capable baseline where packaging allows it. |
| `aarch64-neon` | Common ARM baseline. |
| `aarch64-sve` / `sve2` | Nightly or specialized Linux builds. |

The Rust crate should remain portable by default. Release artifacts can choose
more aggressive families when the target environment is known.

## Runtime Profile

Runtime dispatch should use a processor profile:

- selected backend from feature detection;
- L1/L2/L3 sizes when available;
- cache line size;
- logical CPU count;
- optional physical/core-class details when available;
- benchmark calibration results when present.

The profile should be cheap to query and easy to print. The benchmark logger
should persist it with every run so results are tied to hardware facts instead
of a vague CPU model string.

Current implementation status:

- `ProcessorProfile::detect()` selects the runtime backend and logical CPU
  count;
- on Linux with `std`, it also reads cache line, L1D, L2, and L3 sizes from
  `/sys/devices/system/cpu/cpu0/cache`;
- `backend_kernel_support()` distinguishes native implemented backends from
  scalar fallback backends, so future ISA support is not implied by feature
  names alone;
- `cargo run -p tokenfs-algos --features userspace --example dispatch_explain`
  prints the detected profile, the catalog, and representative
  planner decisions. The `--features userspace` flag is required
  since v0.4.0 (audit-R9 #4).

## Autotuning

Static CPU metadata is useful, but benchmark probes are better. A future
autotuner should run a short calibration over:

- direct vs local vs striped histogram;
- cheap prefix sampling vs larger/spread sampling;
- 4 KiB, 64 KiB, and 1 MiB chunks;
- parallel per-core merge plans;
- random 1-byte and random 4 KiB call overhead.

The result can be cached by CPU model, backend, crate version, rustc version,
and binary hash. This is closer to BLAS/FFT planning than ordinary feature
detection.

## Dispatch Layers

Dispatch should happen in layers:

1. API context: block, file/extent, sequential stream, or random small reads.
2. Workload shape: text/binary/mixed, entropy class, and entropy scale.
3. Processor profile: backend, cache sizes, and core count.
4. Calibration table, when present.
5. Stable fallback rules.

That order matters. A 4 KiB sequential read should not use a 4 KiB classifier
just because AVX2 exists.

## Introspection

Callers should eventually be able to inspect a decision:

```text
primitive: histogram
backend: avx2
input: binary, mixed entropy, macro scale, sequential 64 KiB
chosen: adaptive-chunked-64k
reason: macro variation; per-region classification should beat one prefix sample
```

This makes performance debuggable across Rust, FUSE, kernel-adjacent code, and
Python bindings.

# Primitive Contracts

Date: 2026-04-30. Last revised: 2026-05-03.

> See also: [`KERNEL_SAFETY.md`](KERNEL_SAFETY.md) for the
> kernel-safe-by-default contract that overlays everything below
> (the `try_*` / `_unchecked` / `_inner` conventions, the
> panic-surface lint, and the empty allowlist policy).

This crate is a low-level primitive library. Every hot primitive should be
usable by TokenFS, FUSE, kernel-adjacent callers, Postgres extensions,
MinIO/Go consumers via cgo, CDN edge caches, Python bindings, benchmarks,
and paper-calibration code without changing semantics across backends.

## Queue-Pruning Gate

Every candidate primitive (whether new or being lifted from deferred status)
must justify itself by answering four questions before joining the
implementation queue. If the answer is hypothetical for *every* environment
in `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`, the primitive stays deferred.

1. **What query/workload binds on this primitive being slow?** A specific,
   documented consumer-side pattern — not a guess. If you can't name a
   consumer that profiles bottlenecking on its absence, defer.
2. **What's the cache-residency picture for the data this primitive operates
   on?** Cache-tier conclusions vary across consumers (TokenFS metadata fits
   in L3; Postgres indexes don't). L1-resident kernels can be branchy;
   DRAM-resident kernels must be bandwidth-aware. Pick benches accordingly.
3. **Which consumer environments can this primitive operate in?** Per the
   deployment matrix: kernel-module use forbids rayon, blake3, large stack
   scratch; cgo-bridged use mandates batch APIs. Some primitives are
   kernel-safe (bits, most of bitmap, vector); others are inherently
   userspace (permutation construction). Specify and verify.
4. **What's the consumer surface — one consumer or many?** Single-consumer
   primitives ride in their consumer's module. Multi-consumer primitives get
   a dedicated module. If only one consumer is real and the rest are
   speculative, push back on dedicated-module premise.

This gate is the primary discipline that keeps the crate's surface area
aligned with what consumers actually bind on, instead of growing
speculatively. It applies to every new primitive proposal, every Tier-D-to-C
promotion, and every "let's add this hash family / sort variant / regex
engine" suggestion.

The gate first appeared in `docs/v0.2_planning/README.md`'s framing principle.
Promoted here so it survives the v0.2 milestone as the canonical contract.

## Hot-Path Contract

Every hot primitive must satisfy these rules:

- The input is a byte slice or a fixed-size byte block.
- The function is pure with respect to the input bytes.
- The hot path does not allocate.
- A portable scalar implementation is always available.
- Optimized kernels match scalar exactly, or document a numeric tolerance.
- Benchmark labels are stable enough to compare across commits.
- The ergonomic public path may use planning or runtime dispatch.
- A pinned kernel path is available for reproducibility and forensic runs.

The intended API shape is:

```rust
tokenfs_algos::fingerprint::block(bytes);
tokenfs_algos::fingerprint::kernels::scalar::block(bytes);
tokenfs_algos::fingerprint::kernels::auto::block(bytes);
```

The first line is the normal product API. The pinned `kernels::*` paths are for
tests, benchmarks, paper replication, and users who need bit-for-bit backend
control.

When the normal product API intentionally uses an approximation for latency,
that approximation must be documented at the function boundary and paired with a
pinned exact scalar path. The current example is `fingerprint::extent`: H1,
run-length, top-16 coverage, and skew are exact, while large-extent H4 is sampled
unless callers choose `fingerprint::kernels::scalar::extent`. The current
large-extent sampled-H4 regression bound is 2.5 bits on a periodic-text fixture.

## Backend Order

Kernel families should land in this order:

1. `scalar`: safe, portable reference implementation.
2. `scalar-unrolled` or `u64-chunked`: still portable, lower overhead.
3. `sse4.2`: CRC32C and small x86 dispatch wins where relevant.
4. `avx2`: first wide x86 backend for histogram/fingerprint/byteclass.
5. `avx512`: later, only after AVX2 semantics are stable.
6. `neon`: AArch64 production path.
7. `sve` / `sve2`: later AArch64 wide-vector paths.

Backends may be present as documented candidates before they are implemented,
but public pinned modules should only expose kernels that have correctness tests.

## Benchmark Contract

Every primitive family should have isolated benchmarks:

- `bench-fingerprint`
- `bench-sketch`
- `bench-byteclass`
- `bench-runlength`
- `bench-entropy`
- `bench-selector`

The benchmark log must include:

- git commit and dirty state;
- rustc version;
- CPU model;
- detected CPU features;
- cache topology when available;
- primitive family;
- stable kernel label;
- workload case/source/content/entropy/pattern/bytes;
- throughput and timing.

Reports should generate:

- timing CSV;
- HTML heatmap;
- throughput histogram SVG;
- winner-count SVG;
- primitive-by-kernel SVG;
- dimension charts for size, content, entropy, source, pattern, and threads when
  the suite includes them.

## Paper Lineage

The paper labels remain calibration names:

- `F21` -> selector
- `F22` -> fingerprint
- `F23a` -> sketch
- `F23b` -> conditional dispatch

See [Paper Lineage Naming](PAPER_LINEAGE_NAMING.md) for naming rules. Normal
crate APIs use the product names; paper labels are retained in `paper::*`
compatibility namespaces and benchmark fixture names.

## v0.1 Gate

Before treating a primitive as v0.1-ready:

1. Scalar implementation exists.
2. Public default path exists (kernel-safe; never panics on caller input
   — see [`KERNEL_SAFETY.md`](KERNEL_SAFETY.md)).
3. Pinned scalar path exists.
4. Known-value tests exist.
5. Property/parity tests compare default vs scalar.
6. Benchmark rows appear in `primitive_matrix`.
7. Report artifacts show the primitive clearly.
8. Paper calibration either passes or is explicitly skipped with a missing-path
   message.
9. `cargo xtask panic-surface-lint` passes (no new ungated panic macros
   in the public surface).
10. The `tokenfs-algos-no-std-smoke` crate exercises the new kernel-safe
    entry point if one was added.

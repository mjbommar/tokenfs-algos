# AGENTS.md

Guidance for coding agents working in this repository.

## Current State (v0.4.6, 2026-05-03)

`tokenfs-algos` is a shipped Rust crate for hardware-accelerated byte-stream
compute primitives. As of v0.4.6:

- 979 lib tests pass with `--all-features`; 671 with
  `--no-default-features --features alloc` (the kernel-safe surface).
- 13 cargo-fuzz targets compile and run nightly under
  `.github/workflows/fuzz-nightly.yml`.
- 10 GitHub Actions workflows: per-push CI (correctness + clippy + bench),
  sanitizers (Miri + ASan + MSan), code coverage (cargo-llvm-cov),
  bench-regression, calibration (self-hosted), mutation testing
  (cargo-mutants), CodeQL, dependency security, and the panic-surface
  lint (`cargo xtask panic-surface-lint`).
- OSS-Fuzz integration files live in `oss-fuzz/`; PR submission to
  google/oss-fuzz is queued.
- Audit-R4 through R10 are closed. The kernel-safe-by-default narrative
  is structurally enforced: `tools/xtask/panic_surface_allowlist.txt`
  contains zero entries, and `cargo xtask panic-surface-lint` blocks
  any new `pub fn` that introduces a panicking macro without a
  `#[cfg(feature = "userspace")]` gate.

Read `PLAN.md` for the living roadmap, `CHANGELOG.md` for the audit-lineage
and per-release diff, and `docs/KERNEL_SAFETY.md` (when present) for the
kernel-safe-by-default contract.

## Where Things Live

```text
tokenfs-algos/
├── Cargo.toml                         # workspace
├── PLAN.md                            # living roadmap (read first)
├── CHANGELOG.md                       # per-release diff + audit lineage
├── README.md                          # consumer-facing entry point
├── AGENTS.md                          # this file
├── crates/
│   ├── tokenfs-algos/                 # the core crate
│   └── tokenfs-algos-no-std-smoke/    # no_std + alloc consumer smoke
├── tools/xtask/                       # workspace automation
│   ├── src/main.rs                    # `cargo xtask <task>` entries
│   └── panic_surface_allowlist.txt    # SHOULD STAY EMPTY (load-bearing)
├── fuzz/                              # 13 cargo-fuzz targets
├── oss-fuzz/                          # upstream submission mirror
├── docs/                              # planning + session logs
└── .github/workflows/                 # 10 CI workflows
```

`docs/` mixes living docs (PRIMITIVE_CONTRACTS.md, PLANNER_DESIGN.md,
NO_STD_POSTURE.md, PROCESSOR_AWARE_DISPATCH.md, FS_PRIMITIVES_GAP.md,
KERNEL_SAFETY.md) with dated session logs (BENCHMARK_SESSION_*.md,
CORE_PRIMITIVE_COMPLETION_*.md, etc.). Session logs are historical
record only — don't update them.

## Kernel-Safe-By-Default Contract

This is the load-bearing invariant of the crate. **Any new code must
maintain it.**

Public surface comes in two flavors:

- **Kernel-safe** (reachable in `--no-default-features --features alloc`
  builds): never panics on caller input. Returns `Result` for all
  shape / range / overflow errors. Backend kernels expose
  `pub (unsafe) fn <name>_unchecked` siblings the runtime dispatchers
  call after upfront validation.
- **Userspace** (gated on `feature = "userspace"`): the panicking
  ergonomic surface. Each panicking entry is a thin wrapper around the
  fallible `try_*` sibling or asserts then calls the `_unchecked`
  inner.

When adding a new `pub fn`:

1. If the function has caller-provided shape preconditions (length
   match, in-range index, non-zero divisor), add a `try_*` sibling
   that returns `Result<_, MyError>` for those preconditions.
2. The panicking version, if you keep one, must be
   `#[cfg(feature = "userspace")]`-gated and should be a thin
   `try_*(...).expect(...)` wrapper.
3. For runtime SIMD dispatchers, add a `_unchecked` sibling on each
   backend kernel and have the dispatcher call `_unchecked` after
   upfront validation. The asserting kernel stays as the
   userspace-only oracle.
4. Where the body is too large to duplicate, extract a private `_inner`
   helper that takes a pre-validated input. The userspace-gated entry
   asserts then calls `_inner`; the `try_*` sibling validates then
   calls `_inner`. See `permutation::rabbit::rabbit_order_inner` /
   `rcm_inner` for canonical examples.

`cargo xtask panic-surface-lint` enforces this. The allowlist
(`tools/xtask/panic_surface_allowlist.txt`) is empty by policy —
**extending it is almost never the correct response.** If the lint
catches you, fix the panic.

## Rust Toolchain

- `rust-toolchain.toml` selects nightly explicitly with `rustfmt`,
  `clippy`, `llvm-tools-preview`, and `miri` components.
- Stable Rust is a soft target: `cargo build` should succeed on stable
  for the default features. Nightly is required for fuzz, sanitizers,
  miri, and `bench-internals`-gated tests.
- Gate unstable functionality behind a `nightly` Cargo feature.
- Keep scalar reference implementations available for every public
  algorithm. Runtime-dispatched SIMD backends must be bit-exact with
  scalar.

## Core API Rules

- Operate on `&[u8]`, slices, iterators, and small value types.
- Avoid file I/O, process I/O, global mutable policy, and
  domain-specific types in the core crate.
- Never panic in kernel-safe `pub fn` bodies. Return small error types
  where errors are meaningful. See "Kernel-Safe-By-Default Contract"
  above.
- Keep outputs deterministic across architectures and feature sets.
- Prefer fixed-size arrays and plain data structs for hot-path outputs.
- Make data structures FFI-friendly where doing so does not distort
  the Rust API.

## Kernel and FUSE Readiness

The kernel-module use case means the core algorithms must avoid
unnecessary dependencies on `std`, allocation, threads, and OS
services.

- The default kernel-safe surface is
  `--no-default-features --features alloc`. The
  `tokenfs-algos-no-std-smoke` crate exercises this through a real
  `#![no_std]` consumer; its `smoke()` covers
  `bits::popcount_u64_slice`, `try_streamvbyte_{encode,decode}_u32`,
  `try_sha256_batch_st`, `hash::sha256::try_sha256`, `vector::dot_f32`,
  `vector::l2_squared_f32`, `Permutation::try_from_vec` +
  `try_apply_into`, `hash::contains_u32_simd`,
  `try_contains_u32_batch_simd`, and
  `search::packed_dfa::PackedDfa::try_new`.
- `cargo xtask security` builds + checks this crate; if a kernel-safe
  primitive moves out from under `--no-default-features --features alloc`,
  this build fails.
- Keep `rayon`, filesystem APIs, logging frameworks, and heap-heavy
  data structures behind feature gates.
- Keep unsafe code isolated, reviewed, documented, and covered by
  tests + Miri (when feasible) + ASan/MSan (`.github/workflows/sanitizers.yml`).

## Python Bindings (planned, not yet present)

When the bindings layer lands, it goes in a separate workspace
member (e.g. `bindings/python` or `tokenfs-algos-py`):

- PyO3 + maturin.
- Do not add PyO3 as a dependency of the core crate.
- Release the GIL around long-running compute kernels.
- Use zero-copy buffers where safe.
- Mirror only stable, useful high-level APIs into Python; do not
  expose raw SIMD primitives.

## Testing Requirements

Testing is part of the implementation, not a follow-up.

Minimum coverage for a new algorithm:

- Known-value unit tests (FIPS / NIST / paper references where
  available).
- Scalar reference tests.
- Backend parity tests for every available SIMD backend, asserting
  bit-exact equality with the scalar oracle.
- Property tests (proptest) for invariants — entropy bounds,
  divergence non-negativity, permutation bijection, etc.
- Fuzz target for parser-like or windowing code (add to
  `fuzz/fuzz_targets/` and the `fuzz/Cargo.toml` manifest).
- Regression test for any bug fixed during development.
- A `try_*` sibling test that exercises the error path explicitly.

Important invariants:

- Identical output for scalar, AVX2, AVX-512, NEON, and SVE backends
  when those backends are available.
- Correct handling of empty, tiny, unaligned, and non-power-of-two
  inputs.
- No out-of-bounds reads in vectorized tail handling — Miri + ASan
  catch these in CI.

## Benchmarking and Profiling

Every hot-path algorithm needs Criterion benchmarks before it is
considered done.

Bench targets are split:

- `cargo xtask bench-primitives` — isolated primitive matrix.
- `cargo xtask bench-workloads` — workload matrix.
- `cargo xtask bench-real-{f21,f22}` — calibration matrices on real
  paper data; opt-in via `TOKENFS_ALGOS_REAL_DATA`.
- `cargo xtask bench-compare <old> <new>` and
  `cargo xtask bench-regression-check <baseline> <current> <threshold_pct>`
  — drive the bench-regression CI workflow.

Use profiling tools as appropriate:

- `cargo xtask profile` / `profile-flamegraph` (perf wrappers).
- `valgrind` or sanitizer builds where useful.
- `miri` for unsafe-heavy code (run via `.github/workflows/sanitizers.yml`).

Do not optimize by changing algorithm semantics. If an approximation
is used, make it explicit, shared across backends, documented, and
tested.

## SIMD and Unsafe Code

- Write scalar first unless porting an already-proven SIMD prototype.
- Keep target-feature code small and local. The
  `arch-pinned-kernels` Cargo feature gates per-backend `pub mod`
  visibility — see `audit-R7 #17` / v0.4.1 for the rationale.
- Use runtime feature detection once and cache the selected backend.
- Keep lookup tables shared across backends to preserve bit-exact
  parity.
- Explain every unsafe block with the condition that makes it sound.
- Test unaligned buffers and tail lengths around vector widths.

## Dependencies

Keep dependencies tight.

Allowed baseline:

- `cfg-if`
- `bytemuck` for safe representation work
- `rayon` behind the `parallel` feature
- `blake3` behind the `blake3` feature
- `criterion`, `proptest`, and small test helpers as dev-dependencies

Avoid domain-specific dependencies in the core crate. Compression,
TokenFS integration, FUSE, Python, and profile catalogs live in
separate crates, features, examples, or downstream repositories.

`cargo xtask security` checks the no_std dependency tree and rejects
known-forbidden crates (blake3, criterion, proptest, rayon, serde_json)
from the kernel-safe surface.

## Code Quality Expectations

- `cargo xtask check` must pass before finalizing changes. It runs
  fmt, clippy, doc, no_std build matrix, no_std smoke crate, and the
  panic-surface lint.
- Prefer explicit, small APIs over clever macro-heavy surfaces.
- Keep comments useful and sparse. Don't restate what the code says;
  explain WHY when non-obvious (a workaround, a hidden constraint, a
  past incident).
- Keep public docs precise about units, bounds, determinism, and
  backend parity.
- Avoid unrelated refactors while implementing a focused task.
- For multi-file migrations (rename a function, gate it, update
  callers): always preview matches with `grep -n` first; explicitly
  guard regex with negative lookbehind for `try_` and `_unchecked`
  patterns; check the diff before running cargo. Sed/perl
  substitutions over the codebase repeatedly hit docstrings, string
  literals, and adjacent `try_*` calls — when in doubt, do the
  migration in surgical Edit calls.

## Definition Of Done For Substantial Changes

A substantial change is not done until:

- The implementation exists and follows the kernel-safe-by-default
  contract above.
- Tests cover scalar behavior, edge cases, and `try_*` error paths.
- SIMD parity is tested if SIMD is involved.
- Benchmarks exist for hot-path code.
- `cargo xtask check` is green (including the panic-surface lint).
- Documentation or examples are updated for public APIs.
- For audit-related work: the relevant `audit-RN #M` reference is in
  the commit message and CHANGELOG entry.

If a step cannot be completed in the current environment, state
exactly what was not run and why.

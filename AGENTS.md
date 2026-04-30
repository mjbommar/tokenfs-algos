# AGENTS.md

Guidance for coding agents working in this repository.

This repository is currently at the planning stage. Read `PLAN.md` before
writing code, and keep implementation choices aligned with that document unless
the user explicitly revises the plan.

## Project Goals

`tokenfs-algos` is intended to become a low-level Rust algorithms crate for
hardware-accelerated byte-stream analysis. The core crate must be usable by:

- `../tokenfs-paper/` and related TokenFS tooling.
- A future TokenFS kernel module or kernel-adjacent implementation.
- A future FUSE filesystem implementation.
- A Python package built with PyO3 and maturin on top of the Rust crate.

The core library is not a tokenizer, filesystem, compression codec, file parser,
or machine-learning layer. It is pure compute over byte slices.

## Repository Priorities

1. Build the Rust crate first.
2. Keep the core APIs content-agnostic and domain-neutral.
3. Keep kernel/FUSE viability in mind from the beginning.
4. Add Python bindings as a wrapper layer, not as a dependency of the core crate.
5. Benchmark, profile, fuzz, and parity-test aggressively.

## Rust Toolchain

- Prefer current Rust nightly for active development when it unlocks useful SIMD,
  const-generic, profiling, or benchmarking functionality.
- Gate unstable functionality behind a `nightly` Cargo feature.
- Keep scalar reference implementations available for every public algorithm.
- Keep runtime-dispatched SIMD backends bit-exact with scalar output.
- Use current stable Rust as a compatibility target where practical, but do not
  block important v0.x design work solely to preserve an old MSRV.

When adding a toolchain file later, prefer a checked-in `rust-toolchain.toml`
that selects nightly explicitly and lists required components such as
`rustfmt`, `clippy`, and profiler support.

## Crate Shape

Follow the module structure in `PLAN.md` unless there is a concrete reason to
change it:

- `windows`
- `histogram`
- `entropy`
- `divergence`
- `runlength`
- `byteclass`
- `chunk`
- `fingerprint`
- `primitives`
- `dispatch`
- `error`
- `prelude`

Keep raw SIMD and target-feature-specific functions in `primitives` and
`pub(crate)` by default. Public APIs should be safe, documented, and hard to
misuse.

## Core API Rules

- Operate on `&[u8]`, slices, iterators, and small value types.
- Avoid file I/O, process I/O, global mutable policy, and domain-specific types
  in the core crate.
- Avoid panics in public APIs; return small error types where errors are
  meaningful.
- Keep outputs deterministic across architectures and feature sets.
- Prefer fixed-size arrays and plain data structs for hot-path outputs.
- Make data structures FFI-friendly where doing so does not distort the Rust API.
- Document every public item with examples once implementation begins.

## Kernel And FUSE Readiness

The future kernel-module use case means the core algorithms should avoid
unnecessary dependencies on `std`, allocation, threads, and OS services.

- Keep a path toward `no_std` plus optional `alloc`.
- Do not make `rayon`, filesystem APIs, logging frameworks, or heap-heavy data
  structures mandatory for core algorithms.
- Keep parallelism feature-gated.
- Keep unsafe code isolated, reviewed, documented, and covered by tests.
- Do not assume x86_64. Maintain scalar fallback and plan for NEON/SVE.
- Do not let Python, FUSE, or CLI convenience concerns leak into the core crate.

FUSE-specific code, if added, should live in a separate crate or example binary
that depends on the core library.

## Python Binding Plan

Python bindings should be a separate workspace member or package layer, for
example `bindings/python` or `tokenfs-algos-py`.

- Use PyO3 and maturin.
- Do not add PyO3 as a dependency of the core Rust crate.
- Release the GIL around long-running compute kernels.
- Use zero-copy buffers where safe and practical.
- Keep NumPy support optional unless the user asks otherwise.
- Benchmark Python-call overhead separately from Rust kernel throughput.
- Mirror only stable, useful high-level APIs into Python; do not expose raw SIMD
  primitives.

## Testing Requirements

Testing is part of the implementation, not a follow-up.

Minimum expected coverage for new algorithms:

- Known-value unit tests.
- Scalar reference tests.
- Backend parity tests for every available SIMD backend.
- Property tests for invariants and edge cases.
- Fuzz or randomized tests for parser-like or windowing code.
- Regression tests for any bug fixed during development.

Important invariants include:

- Entropy bounds and exact values for constant/uniform inputs.
- Divergence non-negativity and identity properties.
- Identical output for scalar, AVX2, AVX-512, NEON, and SVE backends when those
  backends are available.
- Correct handling of empty, tiny, unaligned, and non-power-of-two inputs.
- No out-of-bounds reads in vectorized tail handling.

## Benchmarking And Profiling

Every hot-path algorithm needs Criterion benchmarks before it is considered
done.

Use benchmarks to track:

- Throughput in bytes/sec.
- Per-block latency for 256-byte and filesystem-scale blocks.
- Scalar vs. SIMD speedup.
- Runtime dispatch overhead.
- Python wrapper overhead when bindings exist.

Use profiling tools as appropriate:

- `cargo bench`
- `cargo test --release`
- `cargo flamegraph`
- `perf stat`
- `perf record`
- `valgrind` or sanitizer builds where useful
- `miri` for unsafe-heavy code when feasible

Do not optimize by changing algorithm semantics. If an approximation is used,
make it explicit, shared across backends, documented, and tested.

## SIMD And Unsafe Code

- Write scalar first unless porting an already-proven SIMD prototype.
- Keep target-feature code small and local.
- Use runtime feature detection once and cache the selected backend.
- Use `cfg-if` or a similarly clear pattern for backend selection.
- Keep lookup tables shared across backends to preserve bit-exact parity.
- Explain every unsafe block with the condition that makes it sound.
- Test unaligned buffers and tail lengths around vector widths.

## Dependencies

Keep dependencies tight.

Allowed baseline dependencies from the plan:

- `cfg-if`
- `bytemuck` when needed for safe representation work
- `rayon` behind a `parallel` feature
- `criterion`, `proptest`, and small test helpers as dev-dependencies

Avoid domain-specific dependencies in the core crate. Compression, TokenFS
integration, FUSE, Python, and profile catalogs should live in separate crates,
features, examples, or downstream repositories.

## Workspace Direction

Prefer a workspace layout from the first code commit, even if it initially
contains only the core crate. A likely future shape is:

```text
tokenfs-algos/
├── Cargo.toml
├── crates/
│   └── tokenfs-algos/
├── bindings/
│   └── python/
├── benches/
├── examples/
├── tests/
├── PLAN.md
└── AGENTS.md
```

If the user chooses a single-crate layout instead, keep paths simple and avoid
premature abstraction.

## Integration With tokenfs-paper

The F22 prototype is expected to migrate from:

```text
../tokenfs-paper/tools/rust/entropy_primitives/
```

into this crate.

Migration work must preserve:

- Existing semantics.
- Existing parity checks.
- Existing calibration behavior.
- Existing benchmark targets.

After migration, `../tokenfs-paper/` should consume this crate through a path
dependency or a published crate version. Do not duplicate algorithm code across
repositories once the migration is complete.

## Code Quality Expectations

- Run `cargo fmt` before finalizing Rust changes.
- Run `cargo clippy --all-targets --all-features` when practical.
- Prefer explicit, small APIs over clever macro-heavy surfaces.
- Keep comments useful and sparse.
- Keep public docs precise about units, bounds, determinism, and backend parity.
- Avoid unrelated refactors while implementing a focused task.

## Definition Of Done For Substantial Changes

A substantial change is not done until:

- The relevant implementation exists.
- Tests cover scalar behavior and edge cases.
- SIMD parity is tested if SIMD is involved.
- Benchmarks exist for hot-path code.
- Documentation or examples are updated for public APIs.
- The change has been checked against the goals in `PLAN.md`.

If a step cannot be completed in the current environment, state exactly what was
not run and why.

# tokenfs-algos

Low-level Rust algorithms for hardware-accelerated byte-stream analysis.

The core crate is content-agnostic, deterministic across backends, and
**kernel-safe by default**: every panicking public API is gated behind
`feature = "userspace"`, with fallible `try_*` siblings or `_unchecked`
runtime dispatchers reachable from kernel-default builds. The kernel-safe
contract is enforced by `cargo xtask panic-surface-lint` and documented
end-to-end in [`docs/KERNEL_SAFETY.md`](docs/KERNEL_SAFETY.md).

Designed for use from TokenFS tooling, kernel/FUSE integrations,
Postgres extensions, MinIO/CDN edge code, forensics tooling, and
(future) PyO3 Python bindings.

See [`PLAN.md`](PLAN.md) for the living roadmap and
[`CHANGELOG.md`](CHANGELOG.md) for per-release diffs + audit lineage.

## Development

Use the checked-in nightly toolchain:

```bash
cargo xtask check
cargo xtask test
cargo xtask bench
cargo xtask bench-kernels
cargo xtask bench-adaptive
cargo xtask bench-workloads
cargo xtask bench-workloads-adaptive
cargo xtask bench-calibrate
cargo xtask bench-real-f21
cargo xtask bench-real-f22
cargo xtask bench-compare target/bench-history/runs/<old>.jsonl target/bench-history/runs/<new>.jsonl
cargo xtask bench-report
cargo xtask profile
cargo xtask profile-flamegraph -- --profile-time 10 workload_matrix/adaptive-prefix-1k
cargo run -p tokenfs-algos --features userspace --example dispatch_explain
cargo run -p tokenfs-algos --features userspace --example classify_block
cargo run -p tokenfs-algos --features userspace --example cdc_chunking
cargo run -p tokenfs-algos --features userspace --example content_match
```

The `--features userspace` flag is required for the examples since the
v0.4 surface flip (audit-R9 #4); they reach panicking-shape APIs that
are not in the kernel-default surface. CI passes the flag explicitly
in `.github/workflows/ci.yml`.

Real-data benchmarks are opt-in and keep large corpora out of git:

```bash
cargo xtask bench-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-kernels-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-adaptive-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-adaptive-contexts-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-workloads-adaptive-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask profile-real ~/ubuntu-26.04-desktop-amd64.iso
```

If no path is provided, the real-data benchmark/profile tasks use
`TOKENFS_ALGOS_REAL_DATA`, then fall back to
`~/ubuntu-26.04-desktop-amd64.iso`.

Paper-data calibration is a hard gate when the `calibration` feature is enabled:

```bash
cargo test -p tokenfs-algos --features calibration --test fingerprint_f22 -- --nocapture
```

## Cross-arch testing (AArch64 / NEON)

NEON byte-class, run-length, and UTF-8 kernels live behind
`cfg(target_arch = "aarch64")` and are exercised by `tests/neon_parity.rs`. To
run them on an x86 dev host:

```bash
sudo apt install gcc-aarch64-linux-gnu libc6-dev-arm64-cross qemu-user
rustup target add aarch64-unknown-linux-gnu
cargo test -p tokenfs-algos --target aarch64-unknown-linux-gnu --test neon_parity
```

`.cargo/config.toml` wires `tools/cross-runner/qemu-aarch64.sh` as the test
runner so QEMU user-mode emulates the binaries transparently. See
`docs/PROFILING.md` for cache/perf/thread caveats.

## Fuzzing

Hot parser/windowing/dispatcher paths have cargo-fuzz harnesses in `fuzz/`:

```bash
cargo install cargo-fuzz   # one-time
cargo +nightly fuzz run chunk_gear_no_panic                    -- -max_total_time=60
cargo +nightly fuzz run chunk_fastcdc_no_panic                 -- -max_total_time=60
cargo +nightly fuzz run ngram_windows_no_panic                 -- -max_total_time=60
cargo +nightly fuzz run byteclass_utf8_dispatch_parity         -- -max_total_time=60
cargo +nightly fuzz run runlength_transitions_dispatch_parity  -- -max_total_time=60
```

The two `*_dispatch_parity` targets verify the runtime-dispatched SIMD path
matches the scalar reference byte-for-byte; the three `*_no_panic` targets
verify chunking and n-gram windowing tolerate arbitrary inputs without panic
or invariant violation.

The calibration feature requires the F21/F22 artifacts to exist locally or be
provided through `TOKENFS_ALGOS_F22_DATA` and `TOKENFS_ALGOS_F21_ANALYSIS`.

The current implementation includes scalar/reference byte histograms, entropy,
run-length and structure signals, byte classification and UTF-8 validation,
Gear and normalized FastCDC chunking, calibrated distribution lookup, selector
signals, and the F22/content fingerprint path.

The normal fingerprint API is `fingerprint::{block, extent}`. Pinned reference
paths live under `fingerprint::kernels::scalar::*`. The default extent path keeps
H1, run-length, top-16 coverage, and skew exact, and samples H4 on extents larger
than 64 KiB with a current 2.5-bit sampled-H4 regression bound on a
periodic-text fixture; use the scalar pinned extent when calibration or forensic
work needs the exact H4 hash-bin oracle.

The workload benchmark matrix and logged result format are documented in
`docs/BENCHMARK_WORKLOAD_MATRIX.md`. Processor-aware dispatch and kernel
promotion strategy are documented in `docs/PROCESSOR_AWARE_DISPATCH.md`,
`docs/PRIMITIVE_KERNEL_BUFFET.md`, and
`docs/AUTOTUNING_AND_BENCH_HISTORY.md`. The histogram planner architecture
(rule table, named constants with bench provenance, trace mode, and the
recipe for adding a rule or threshold) lives in `docs/PLANNER_DESIGN.md`.
Paper-linked primitive migration and consumer latency budgets are tracked
in `docs/PAPER_PRIMITIVE_MIGRATION.md` and `docs/CONSUMER_LATENCY_BUDGETS.md`.
Current calibration gates are summarized in
`docs/CALIBRATION_GATES_2026-05-01.md`.

## Cargo features

The crate ships several Cargo features that gate optional functionality
and tune the API surface for different deployment shapes:

- **`std`** (default) — pulls in the standard library. Required for any
  `OnceLock`-style globals, `std::error::Error` impls, and the
  parallel/blake3 paths.
- **`alloc`** — `Vec`/`Box`/`String` from the `alloc` crate. Implied by
  `std`. Kernel and other no_std consumers want this without `std`.
- **`avx2`** / **`avx512`** / **`neon`** / **`sve`** / **`sve2`** —
  enable the corresponding SIMD backend. Architecture-mismatched
  features compile to no-ops.
- **`parallel`** — pulls in `rayon` for batched parallel kernels
  (`*_batch_par`). Userspace only.
- **`blake3`** — pulls in the `blake3` crate for content-addressable
  hashing. Userspace only.
- **`userspace`** — umbrella feature for ergonomic userspace consumers.
  Pulls in `std` and `panicking-shape-apis`. **As of v0.4.0 the default
  features no longer include `panicking-shape-apis`** so kernel/FUSE
  consumers get the panic-free surface automatically; userspace
  consumers wanting the v0.3.x-equivalent ergonomic surface opt back
  in:

  ```toml
  tokenfs-algos = { version = "0.4", features = ["userspace"] }
  ```

  Kernel/FUSE consumers do nothing — the default is already kernel-safe.
- **`panicking-shape-apis`** (opt-in via `userspace` since v0.4.0) —
  exposes the panicking shape / length-validating wrappers
  (`BitPacker::encode_u32_slice`, `dot_f32_one_to_many`,
  `RankSelectDict::build`, `sha256_batch_st`, `signature_batch_simd`,
  etc.). Each has a fallible `try_*` parallel that returns a typed error
  (audit-R5 #157, audit-R7-followup #1/3/4/14).
- **`tunes-json`** / **`calibration`** / **`bench-internals`** /
  **`permutation_hilbert`** — auxiliary features for tooling, paper
  artifacts, micro-benches, and the Hilbert-curve permutation path.

## Kernel-safe surface

The kernel-safe surface is reached via:

```toml
tokenfs-algos = { default-features = false, features = ["alloc"] }
```

Under that shape:

- **No public function panics on caller input.** Every shape /
  range / overflow precondition is reachable through a `try_*`
  sibling that returns a typed error.
- **Backend kernels expose `_unchecked` siblings.** The runtime
  dispatchers (`kernels::auto::*_unchecked`) call them after
  upfront validation; the asserting versions stay as a userspace-only
  oracle for SIMD parity tests.
- **No allocation outside `alloc`.** No `std::thread`, no
  `/sys/devices/system/cpu/` reads, no rayon, no logging frameworks.
- **No file I/O, process I/O, or global mutable policy.**

The contract is enforced by `cargo xtask panic-surface-lint`, which
walks every `pub fn` body and rejects ungated panic macros. The
allowlist (`tools/xtask/panic_surface_allowlist.txt`) is empty by
policy — extending it is almost never the correct response when the
lint catches a new entry.

See [`docs/KERNEL_SAFETY.md`](docs/KERNEL_SAFETY.md) for the full
contract — `try_*` / `_unchecked` / `_inner` conventions with
codebase examples, the lint, and the audit-R4-through-R10 lineage.

## Audit posture

`tokenfs-algos` has been through external audit rounds R4 → R10. As
of v0.4.6:

- All HIGH and MEDIUM findings are closed; see `CHANGELOG.md` per
  release for the per-finding writeup.
- `tools/xtask/panic_surface_allowlist.txt` contains zero entries.
- 13 cargo-fuzz targets compile and run nightly under
  `.github/workflows/fuzz-nightly.yml`. cargo-mutants weekly. Miri /
  ASan / MSan on push + Sundays. cargo-llvm-cov per PR. OSS-Fuzz
  integration files live in `oss-fuzz/` ready for upstream submission.

# Reference Notes

Local reference repositories live under `_references/` and are ignored by git.
They are for design review only; do not vendor or copy implementation code from
them into this crate.

## Phase 1 Takeaways

### memchr

- Treat scalar behavior as the oracle.
- Test small inputs, unaligned ranges, feature-disabled builds, and unusual
  targets early.
- Keep benchmark harnesses buildable under CI even when full measurement is a
  local-only activity.
- Fuzz tiny byte-oriented APIs because those inputs are where tail bugs hide.

Action for `tokenfs-algos`: keep edge-length tests and property tests in the
core crate before adding AVX2. Add CI jobs later for no-default-features and
feature-disabled x86 builds once the scalar API stabilizes.

### simdutf and simdjson

- Compile multiple backends but expose one safe public API.
- Select the best supported implementation once, while preserving an explicit
  fallback.
- Keep backend-specific code in isolated translation units/modules.
- Provide a way to force a backend for tests and benchmarks.

Action for `tokenfs-algos`: keep `dispatch::Backend`, `detected_backend`, and
`force_backend` as the only public backend controls. SIMD primitives stay
`pub(crate)` and bit-exact against scalar.

### StringZilla and NumKong

- Multi-language packaging works best when the core kernels stay separate from
  language bindings.
- Capability reporting and benchmarks are user-facing documentation, not just
  developer tooling.
- Binding layers should wrap stable high-level APIs, not raw SIMD primitives.

Action for `tokenfs-algos`: keep PyO3/maturin in a later workspace member. Do
not add Python dependencies to the core crate.

### wide and safe_arch

- Portable wrappers can reduce raw intrinsic noise, but they can also constrain
  low-level kernel layout.
- Target-feature gating needs to remain visible and testable.

Action for `tokenfs-algos`: use scalar first. Re-evaluate wrappers when the
first AVX2 histogram kernel lands; do not choose an abstraction before the
kernel shape is clear.

### criterion.rs

- Criterion output directories and JSON estimates are useful for regression
  tracking.
- Short benchmark settings are useful for validating harnesses, but not for
  real measurements.
- Named baselines are worth adopting once scalar and AVX2 variants coexist.

Action for `tokenfs-algos`: keep quick benchmark commands for CI/build checks
and separate realistic local benchmark commands for ISO/source/binary corpora.

## Processor-Dispatch Review

A second focused pass through the local references reinforced a few design
rules for the processor-aware phase.

### memchr and aho-corasick

- Runtime feature detection should be paired with compile-time fallbacks. With
  `std`, `memchr` can use runtime AVX2 detection; without it, the build target
  controls what is legal.
- SIMD builders in `aho-corasick` and `memchr` treat `target_feature` as a
  safety boundary. Tests explicitly gate feature-specific paths before calling
  them.
- Packed/vector tests cover many tiny cases because short tails and alignment
  boundaries are where byte-oriented kernels fail.

Action for `tokenfs-algos`: keep scalar as the oracle, keep backend forcing for
tests, and add feature-disabled/forced-backend CI cases before exposing SIMD
kernels.

### StringZilla

- The benchmark harness asks cache questions directly: one cache line, SIMD
  width multiples, L1/L2/L3 residency, and CPU counters instead of only wall
  time.
- Its public C macros expose both runtime dispatch and cache constants such as
  cache-line width and combined cache size.
- Its contributing notes warn that dynamic dispatch should avoid tying the
  portable API to one OS-specific mechanism.

Action for `tokenfs-algos`: keep Linux sysfs cache detection as an optional
profile enhancement, not a requirement for correct behavior. Benchmark rows
must preserve cache metadata so cache-sensitive regressions can be explained.

### simdutf and simdjson

- Implementations advertise whether they are supported by the runtime system.
- Users can inspect and manually select active implementations for diagnostics.
- Dispatch happens behind one stable API, while implementation-specific code
  remains isolated.

Action for `tokenfs-algos`: expose catalog and explainability metadata, but keep
raw kernels internal until each has parity tests and workload evidence.

### NumKong

- Compile-time dispatch gives thinner binaries; runtime dispatch gives portable
  wheels/binaries at one indirection per call.
- Its benchmarks print compiled-vs-runtime capability indicators, which makes
  "compiled in but not usable" and "usable but not compiled in" obvious.
- It treats thread oversubscription and cache-blocking as first-class design
  issues, not afterthoughts.

Action for `tokenfs-algos`: benchmark reports should eventually show both
compiled features and runtime support, and parallel file scans should account
for caller-owned threads before adding internal parallelism.

### wide and safe_arch

- `wide` favors portable SIMD-shaped types selected by `cfg`.
- `safe_arch` keeps intrinsic availability visible through `target_feature`
  gates.

Action for `tokenfs-algos`: defer the wrapper choice until the first AVX2
histogram kernel lands. The scalar catalog and planner should not assume one
intrinsics abstraction.

### criterion.rs

- Criterion already stores `benchmark.json`, `estimates.json`, sample data, and
  baselines under `target/criterion`.
- Its baseline support is useful, but the crate still needs a separate
  processor/workload history because Criterion baselines do not encode our
  domain axes or cache profile.

Action for `tokenfs-algos`: keep mining Criterion JSON, but write our own JSONL
history rows and compare them with `cargo xtask bench-compare`.

## Phase 1 Local Data Policy

- Checked-in tests use deterministic generated inputs and small fixtures only.
- Large real data stays outside git.
- Real-data benchmarks are opt-in via `TOKENFS_ALGOS_REAL_DATA` or
  `cargo xtask bench-real <path>`.
- The current local ISO candidate is:

```text
/home/mjbommar/ubuntu-26.04-desktop-amd64.iso
```

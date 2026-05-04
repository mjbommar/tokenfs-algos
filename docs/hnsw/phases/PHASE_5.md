# Phase 5 — Kernel-FPU bracketing + tokenfs_writer integration

**Status:** plan, 2026-05-03. **Weeks 6-7 of the v0.7.0 HNSW landing.**

**Goal:** end-of-week-7, the kernel-safe HNSW search path is end-to-end exercised by `tokenfs-algos-no-std-smoke::smoke()`, the f32 paths cleanly bracket FPU state when called from a kernel context, and `tokenfs_writer` calls `tokenfs-algos::similarity::hnsw::build::Builder` directly without any libusearch dependency. Final `cargo xtask check` is green; allowlist still 0; iai-bench regression gate covers the full HNSW kernel set.

## Deliverables

### Code

- `crates/tokenfs-algos/src/similarity/hnsw/walker.rs`:
  - `kernel_fpu_guard!` macro (or equivalent helper) for f32 paths. Brackets FPU state save/restore at the per-query boundary, NOT per-candidate. (Same trick as kernel ZFS Fletcher4.)
  - `try_search_kernel_safe(...)` — explicit kernel-only entry point that REJECTS f32 metrics at the type level (returns `Err(HnswSearchError::FloatNotKernelSafe)`). Kernel callers must use integer/binary metrics.
  - `try_search` (the userspace-callable kernel-safe entry) keeps accepting f32 with the FPU guard wrapped around it; documented that this branch is not appropriate to call from in-kernel code without the explicit FPU bracketing macros from the kernel crate.
  - `walker.rs` module docstring: a clear table of which `try_*` to call from which context.

- `crates/tokenfs-algos/src/similarity/hnsw/header.rs`:
  - Forward-compatibility test: an artificial usearch v2.20 header byte stream should fail closed at `try_parse_header` with a clear `HnswHeaderError::UnsupportedFormatVersion`. Never silently produce wrong results.
  - Same forward-compat for v3.0 (when it lands) — fail closed; new format gets a new section ID per `IMAGE_FORMAT_v0.3` §11.

- `crates/tokenfs-algos/src/dispatch/planner/`:
  - HNSW-aware sizing rules: at fixed (k, efSearch), the planner picks scalar / SIMD / brute-force based on vector dim, scalar_kind, and detected backends.
  - `dispatch::ProcessorProfile` extended with HNSW signal: `hnsw_target_p99_us` (consumer-supplied; e.g. FUSE wants 50 µs, batch wants 5 ms).
  - Planner trace mode (existing infrastructure from `PLANNER_DESIGN.md`) now emits HNSW decisions.

- `crates/tokenfs-algos/src/lib.rs`:
  - Public re-exports: `pub use similarity::hnsw;` (with the right feature gates per `KERNEL_SAFETY.md`).

- `crates/tokenfs-algos-no-std-smoke/src/lib.rs`:
  - Extend `smoke()` to call `try_search` on a tiny in-memory test index built at compile-time (or via a `const fn` constructor)
  - This ensures the kernel-safe HNSW path stays reachable in `--no-default-features --features alloc` builds — if anyone breaks it, the no-std-smoke build fails

### tokenfs_writer integration (in the sibling crate)

This is in `tokenfs-paper`'s tokenfs_writer crate, NOT in tokenfs-algos. Documenting the integration here for the cross-team handoff:

- Remove any libusearch dependency from `tokenfs_writer/Cargo.toml` (if it was scaffolded).
- Replace any `usearch::Index::new(...)` / `index.add(...)` / `index.save_to_buffer()` calls with:
  ```rust
  use tokenfs_algos::similarity::hnsw::{Builder, BuildConfig};
  let mut builder = Builder::try_new(BuildConfig {
      dimensions: 32,                        // F22 fingerprint
      scalar_kind: ScalarKind::I8,
      metric: Metric::Hamming,
      M: 16,
      M_max: 16,
      M_max0: 32,
      ef_construction: 64,
      seed: 0xCAFE_BABE,                     // deterministic
      ..Default::default()
  })?;
  for (key, fingerprint) in fingerprints_sorted_by_key {
      builder.try_insert(key, &fingerprint)?;
  }
  let blob = builder.try_finish_to_bytes()?;
  // Wrap with TokenFS section header + write to disk
  ```
- Add an integration test: build an HNSW section, write the image, mmap it back, search via our walker. Assert recall on a held-out query set.

### Tests

- `crates/tokenfs-algos/src/similarity/hnsw/walker.rs`:
  - `try_search_kernel_safe_rejects_f32_metric()` — returns specific error variant
  - `kernel_fpu_guard_brackets_at_per_query_not_per_candidate()` — instrumentation test showing fpu_begin/end called once per query

- `crates/tokenfs-algos/src/similarity/hnsw/header.rs`:
  - `header_rejects_v2_20_format_closed()`
  - `header_rejects_future_v3_0_format_closed()`

- `crates/tokenfs-algos-no-std-smoke/src/lib.rs`:
  - The new `try_search` smoke call is part of `smoke()`; the workflow gate ensures any regression in kernel-safety reachability fails build

- `crates/tokenfs-algos/tests/hnsw_*` — re-run all earlier phase tests; nothing should regress

### Benchmarks

- iai-callgrind: full HNSW kernel set has bench rows; gate at 1% IR regression as established
- criterion: throughput tables for the demo

### Docs

- `docs/HNSW_PATH_DECISION.md` — final pass; mark all phases complete
- `docs/hnsw/README.md` — update phase table to mark complete
- All `docs/hnsw/components/*.md` — final pass to fill in any remaining sections
- `docs/PROFILING.md` — add a section on HNSW iai-bench rows to interpret
- README.md (top-level) — add a brief mention of `similarity::hnsw` to the primitive list
- CHANGELOG.md — full v0.7.0 entry

## Acceptance criteria

```bash
$ cargo xtask check
xtask: running `cargo fmt --all --check`
xtask: running `cargo clippy --workspace --all-targets --all-features -- -D warnings`
... [all green]
xtask: running `cargo check -p tokenfs-algos --no-default-features --lib`
xtask: running `cargo check -p tokenfs-algos --no-default-features --features alloc --lib`
xtask: running `cargo check -p tokenfs-algos-no-std-smoke`
xtask: running `cargo build -p tokenfs-algos-no-std-smoke --release`
xtask: no_std dependency tree only contains allowed core crates
xtask: panic-surface-lint: pub fn surface within allowlist (0 entries snapshotted)

$ cargo xtask bench-iai
... HNSW search/build kernel rows pass 1.0% IR regression gate

$ cargo test -p tokenfs-algos --features arch-pinned-kernels
running 1100+ tests
... [all pass; 100+ HNSW-specific test cases]

$ # In tokenfs_writer crate:
$ cargo test --test hnsw_section_round_trip
... build_image_with_hnsw_section ... ok
... mmap_and_search_via_tokenfs_algos_walker ... ok
```

## Out of scope for Phase 5 (or v0.7.0 entirely)

- Multi-threaded builder (`parallel` feature) — defer to v0.8+ if a non-deterministic-build consumer materializes
- Multi-modal hybrid scoring (`similarity::hybrid`) — separate v0.8.0 landing
- GPU walker (separate `tokenfs-gpu` crate)
- HNSW format version > 2.25 — out-of-band; new section ID per the format spec
- Production kernel-module deployment — this is the substrate; the kernel module itself is per `FUSE_TO_KERNEL_ARC.md` 2027+
- Extension to f16/bf16/e5m2/e4m3/e3m2/e2m3 scalar types — out of v1 scope; revisit in v0.9 if a consumer wants them

## Risk + mitigations

| Risk | Mitigation |
|---|---|
| Kernel-FPU bracketing is wrong (FPU state corruption, cycle waste) | The macro is small and grep-able; integration test in the no-std-smoke crate calls the f32 path through the bracket and asserts no FPU corruption visible to the smoke binary. |
| Forward-compat header test misses an attack vector | Add a fuzzer target (`fuzz_targets/hnsw_header_fuzz.rs`) per the v0.5.x cargo-fuzz convention. Random bytes through `try_parse_header` should never panic and never silently accept invalid format. |
| tokenfs_writer integration breaks reproducibility somewhere outside our crate | Round-trip test in tokenfs_writer asserts the same input + same seed produces byte-identical TokenFS section bytes across two runs. |
| Final regression gate fires on real change vs noise | iai-callgrind is deterministic per-binary; first-time gate fires get inspected manually. Tighten/loosen the 1.0% threshold based on observed variance per `PROFILING.md`. |
| AVX-512 nightly-only path slows v0.7.0 stable release | AVX-512 is gated; users who can't run nightly get scalar/AVX2 with no functional difference, just slower. Document in CHANGELOG. |

## v0.7.0 release checklist

After Phase 5 lands and CI is green:

- [ ] Bump `crates/tokenfs-algos/Cargo.toml` 0.6.x → 0.7.0
- [ ] Bump `crates/tokenfs-algos-no-std-smoke/Cargo.toml` to track
- [ ] `cargo update -p tokenfs-algos`
- [ ] Add `## [0.7.0]` CHANGELOG entry covering the full HNSW landing
- [ ] Update `PLAN.md` §0 release table
- [ ] Update `AGENTS.md` "Where Things Live" with `similarity::hnsw`
- [ ] Update README.md primitive list
- [ ] Tag commit; push
- [ ] Verify gh-pages bench-history publishes a new dot for HNSW kernels
- [ ] Verify iai-bench baseline auto-saves the new HNSW kernel rows on main

## Cross-references

- [`PHASE_4.md`](PHASE_4.md) — must complete before this phase starts
- [`../00_ARCHITECTURE.md`](../00_ARCHITECTURE.md) — final state matches what this phase ships
- [`../../KERNEL_SAFETY.md`](../../KERNEL_SAFETY.md) — final compliance check
- [`../../NO_STD_POSTURE.md`](../../NO_STD_POSTURE.md) — no-std-smoke crate updates
- `tokenfs-paper/docs/IMAGE_FORMAT_v0.3.md` — section 0x203 spec reference
- `tokenfs-paper/docs/FUSE_TO_KERNEL_ARC.md` — explains why kernel-mode is the v0.7.0 *substrate* not v0.7.0 *deployment*

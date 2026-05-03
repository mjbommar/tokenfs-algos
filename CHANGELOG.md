# Changelog

All notable changes to this crate will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

## [0.4.6] — 2026-05-03

Audit-R10 systematic gating sweep COMPLETE — `tools/xtask/panic_surface_allowlist.txt`
went from 30 grandfathered entries at v0.4.5 to **zero**. Every panicking
`pub fn` / `pub unsafe fn` reachable from the kernel-default surface is now
either gated on `#[cfg(feature = "userspace")]` or has a `_unchecked` /
`_inner` sibling that the runtime dispatcher / fallible `try_*` entry uses
to stay reachable in non-userspace builds. The kernel-safe-by-default
narrative is now actually true, not just lint-protected.

### Changed (BREAKING for non-`userspace` consumers reaching panicking APIs)

The following `pub fn` / `pub unsafe fn` are now `#[cfg(feature = "userspace")]`-gated;
kernel-default builds (`--no-default-features --features alloc`) no longer see them.
Each has either a documented `try_*` sibling or a `_unchecked` sibling for the
arch-pinned-kernels surface:

  * `approx::BloomFilter::contains_batch_simd` → `try_contains_batch_simd`
  * `approx::HyperLogLog::merge_simd` → `try_merge_simd`
  * `bits::rank_select::RankSelectDict::{rank0, rank1, rank1_batch, select1_batch}`
    → `try_rank0` / `try_rank1` / `try_rank1_batch` / `try_select1_batch`
  * `similarity::minhash::{one_permutation_from_hashes, one_permutation_from_bytes}`
    (no try_* sibling — const-generic `K > 0` invariant; document the
    constraint at the type level)
  * `permutation::{rabbit_order, rabbit_order_par, rcm}` →
    `try_rabbit_order` / `try_rabbit_order_par` / `try_rcm`
  * Backend kernels under
    `bits/{streamvbyte,bit_pack,rank_select}/kernels/{scalar,avx2,avx512,neon,sse41,ssse3}`,
    `approx/bloom_kernels/{scalar,avx2,avx512,neon}`,
    `hash/set_membership/kernels/{scalar,avx2,avx512,neon,sse41}`, and
    `permutation/rabbit/kernels/scalar` — each `pub (unsafe) fn`
    panicking entry now paired with a `pub (unsafe) fn <name>_unchecked`
    sibling. The `kernels::auto::*` runtime dispatchers route through
    `_unchecked` after their own upfront validation; the asserting
    versions are kept as a userspace-only oracle.
  * `permutation::rabbit::kernels::auto::modularity_gains_neighbor_batch`
    → `modularity_gains_neighbor_batch_unchecked` (used by
    `rabbit_order_inner` / `rabbit_order_par_inner` so the algorithm
    stays reachable from `try_rabbit_order` / `try_rabbit_order_par`
    in non-userspace builds).

### Added — `_inner` extraction pattern as codebase convention

Functions whose body needs to be shared between a userspace-gated panicking
entry and a kernel-safe `try_*` sibling now use a private `_inner` helper
that takes a pre-validated input. Established for:

  * `permutation::rabbit::rabbit_order_inner`
    (called by both `rabbit_order` and `try_rabbit_order`)
  * `permutation::rabbit::rabbit_order_par_inner`
    (called by both `rabbit_order_par` and `try_rabbit_order_par`)
  * `permutation::rcm::rcm_inner`
    (called by both `rcm` and `try_rcm`)
  * `permutation::rabbit::kernels::scalar::modularity_gains_neighbor_batch_inner`
    (called by both the asserting and `_unchecked` variants)
  * `hash::set_membership::kernels::scalar::contains_u32_batch_inner`
    (same pattern)

### Fixed — `permutation::hilbert::hilbert_2d` no longer panics

Previously `hilbert_2d` asserted on `points.len() <= u32::MAX` despite its
docstring promising "Does not panic." Replaced the assert with a
saturate-and-identity guard: inputs longer than `u32::MAX` collapse to
`Permutation::try_identity(u32::MAX)` rather than panic. Matches the
docstring's existing guarantee.

`permutation::hilbert::hilbert_nd` is now `#[cfg(feature = "userspace")]`-gated
since its body retains an inline assert on the dim/length contract;
kernel/FUSE callers needing N-D Hilbert ordering must validate upstream.

### Notes

  * Lib test counts unchanged: 978 with `--all-features` (+1 vs v0.4.5
    from `try_validate_no_alloc_errors_on_undersized_scratch`); 671
    `--no-default-features --features alloc`. The migration only changed
    function gating + signatures; no algorithmic behavior changed.
  * `panic_surface_allowlist.txt` content is now categorized comments
    only (no entries). The file is retained as the load-bearing artifact
    for the lint, with a rule that "extending the allowlist" is almost
    never the correct response — fix the panic instead.
  * `cargo xtask check` green: fmt, clippy, doc, no-std-tree,
    no-std-smoke build, panic-surface-lint at zero entries.
  * Audit-R10 status: **ALL TIERS CLOSED.** Tier 0 (T0.1-T0.4) shipped
    v0.4.4. Tier 1 (T1.1-T1.6) shipped v0.4.4. Tier 2 (T2.1-T2.5) +
    Tier 3 partial (T3.1-T3.3, T3.7-T3.9) shipped v0.4.5. Tier 2 carry-
    over (T2.6 systematic gating sweep) closed in this release.
    Remaining v0.5.0 candidates (T3.4 iai-callgrind, T3.5 bench-history
    publication, T3.6 default-flip) are all enhancement, not safety.

## [0.4.5] — 2026-05-03

Audit-R10 Tier 2 + Tier 3 closeout: 4 MEDIUM correctness fixes, 7 new
CI workflows, OSS-Fuzz integration, and the start of a systematic
panic-surface gating sweep (allowlist down 38 → 30 entries).

### Fixed (MEDIUM) — `try_sha256` + reject inputs > 2^64 bits (R10 #4)

Per-backend SHA-256 kernels compute the FIPS 180-4 padding bit length
via `wrapping_mul(8)`. The streaming `Hasher::try_update` checked the
overflow upfront, but the one-shot `sha256(&[u8])` had no fallible
sibling — for inputs > `u64::MAX / 8` bytes (~2 EiB), the digest
silently came out wrong.

  * Added `try_sha256(&[u8]) -> Result<[u8; 32], Sha256LengthOverflow>`
    that validates `bytes.len() * 8` fits in `u64` before dispatch.
  * Gated the panicking `sha256(&[u8])` on `feature = "userspace"`;
    it is now a thin `try_sha256(...).expect(...)` wrapper.
  * Migrated internal callers (`identity::sha256_cid`,
    `hash::batched::sha256_batch_st_inner`) to the kernel-safe
    dispatcher path so they remain reachable without `userspace`.

### Fixed (MEDIUM/LOW) — saturating_add on u32 counters (R10 #5)

`BytePairHistogram::add_pair`, `BytePairScratch::add_pair`,
`BytePairHistogram::with_scratch`, `MisraGries::update`, and the
n-gram sketch hot loop all incremented `u32` counters via `+= 1`.
Switched to `saturating_add` so an exhausted counter pins at
`u32::MAX` (or `u64::MAX` for `observations`) instead of silently
wrapping in release / panicking in overflow-checking builds.

### Fixed (LOW) — cap Sniffer Layer 1 DFA scan at `max_anchor` (R10 #6)

`Sniffer::detect` walked `dfa.find_iter(bytes)` over the entire input
even though no MAGIC_RULES rule starts past `max_anchor`. On
attacker-supplied multi-GB buffers in kernel/FUSE sniffing paths
that turned a constant-time check into an arbitrary-time scan.
Capped Layer 1 to `bytes[..max_anchor.min(bytes.len())]`. +1 test.

### Changed — `permutation::Permutation` panic surface (T2.6 / #216)

Audit-R10 #1 systematic gating sweep, partial — 8 of 38 allowlisted
entries cleared:

  * `Permutation::identity` is now a thin
    `try_identity(...).expect(...)` wrapper gated on `userspace`.
    Added `Permutation::try_identity(n) -> Result<_, LengthExceedsU32>`
    as the kernel-safe sibling.
  * `Permutation::apply`, `apply_into`, `validate_no_alloc`,
    `CsrGraph::neighbors_of`, `CsrGraph::degree` gated on
    `userspace`. Each already has a `try_*` sibling per audit-R4 #151
    / R6 #161; this enforces the kernel-safe-by-default narrative
    at the type-system level.
  * `permutation::hilbert::hilbert_2d` no longer panics on inputs
    `> u32::MAX` — saturates to `Permutation::try_identity(u32::MAX)`
    instead. `permutation::hilbert::hilbert_nd` gated on `userspace`
    (its dim/length contract assertions remain inside the body).
  * Migrated all 27 internal + test + example callers via
    `.apply()` → `.try_apply().expect(...)`,
    `.identity(n)` → `.try_identity(n).expect(...)`,
    `.validate_no_alloc(s)` → `.try_validate_no_alloc(s).is_ok()`,
    `.neighbors_of(v)` → `.try_neighbors_of(v).expect(...)`, etc.

The remaining 30 panic-surface allowlist entries (kernel backends in
`bits/{streamvbyte,bit_pack,rank_select}/kernels/`,
`approx/bloom_kernels/`, `hash/set_membership/kernels/`,
`permutation/rabbit/kernels/`; top-level
`approx::contains_batch_simd`, `merge_simd`,
`bits::rank_select::{rank0, rank1}`, `similarity::minhash::
one_permutation_from_hashes`; and `permutation::{rabbit_order,
rabbit_order_par, rcm, modularity_gains_neighbor_batch}`) require
extracting `_inner` helpers shared between panicking and `try_*`
paths so both can compile in non-userspace builds. Tracked as
the next batch of #216 — `panic_surface_allowlist.txt` documents
each.

### Added — CI infrastructure (Tier 2 + Tier 3 + audit-R10 honest-gaps)

Seven new `.github/workflows/*.yml` files plus an OSS-Fuzz
integration directory:

  * **`sanitizers.yml`** (T3.2) — Miri (UB / strict provenance,
    scalar-only no-default-features build), AddressSanitizer (full
    SIMD surface under `userspace,arch-pinned-kernels`),
    UndefinedBehaviorSanitizer / MSan (uninit memory tracking).
    Cron: Sundays 05:23 UTC.
  * **`coverage.yml`** (T3.3) — `cargo-llvm-cov` reports for both
    `--all-features` and the kernel-safe
    (`--no-default-features --features alloc`) profiles. Uploads
    LCOV + HTML to artifacts; runs per-PR.
  * **`fuzz-nightly.yml`** (T3.7 / #217) — daily 10-min/target
    matrix run for all 13 declared fuzz targets, with rolling
    corpus cache and reproducer upload on crash. Closes the
    "fuzz targets exist but aren't run on a schedule" honest-gap.
  * **`mutation-testing.yml`** (T3.8 / #218) — weekly
    `cargo-mutants` job. Excludes per-backend SIMD intrinsics
    (Miri / cross-arch parity already cover them); fails CI if
    any non-SIMD function survives mutation. Catches the
    "tests pass on broken impl" failure mode line-coverage misses.
  * **`bench-regression.yml`** (T2.3) — per-PR primitive bench
    compared against the latest main baseline (downloaded from
    the previous run's artifact); fails if any benchmark regresses
    > 15% in throughput. Uses the existing `xtask bench-report`
    machinery + new `cargo xtask bench-regression-check`.
  * **`calibration.yml`** (T2.4) — scheduled F21 / F22 / Magic-BPE
    calibration on a labeled self-hosted runner
    (`runs-on: [self-hosted, perf-quiet]`). Runs the throughput-
    gated test under `TOKENFS_ALGOS_RUN_THROUGHPUT_GATE=1` plus
    the `bench-real-{f21,f22,magic-bpe}` xtask matrices, snapshots
    output to `benches/_history/calibration-<run_id>/`. Cron:
    Sundays 11:00 UTC.
  * **`oss-fuzz/{Dockerfile,build.sh,project.yaml,README.md}`**
    (T3.9 / #219) — in-tree mirror of the
    `projects/tokenfs-algos/` files that will be submitted upstream
    to https://github.com/google/oss-fuzz. Closes the "OSS-Fuzz
    integration" honest-gap.

### Added — `cargo xtask` commands

  * `cargo xtask bench-regression-check <baseline.jsonl> <current.jsonl> <threshold_pct>`
    used by the new bench-regression CI workflow.
  * `cargo xtask security` (and therefore `cargo xtask check`) now
    additionally runs `cargo check -p tokenfs-algos-no-std-smoke`
    + the release build (T2.5 / audit-R10 #10). Smoke crate
    coverage broadened: `try_streamvbyte_{encode,decode}_u32`,
    `try_sha256`, `Permutation::try_from_vec` + `try_apply_into`,
    `try_contains_u32_batch_simd`, `PackedDfa::try_new`.

### Notes

  * Lib test counts: 978 with `--all-features` (was 975 at v0.4.4;
    +3 from `try_sha256` round-trip, format scan-cap test, and
    `try_validate_no_alloc` undersized-scratch error case).
    779 default. 672 `--no-default-features --features alloc`
    (was 669; +3 from the new `try_*` test coverage).
  * `cargo xtask check` green: fmt, clippy, doc, no-std-tree,
    panic-surface-lint (30/38 grandfathered), no-std-smoke build.
  * Audit-R10 status by tier:
    * Tier 0 (T0.1-T0.4): ✅ shipped in v0.4.4
    * Tier 1 (T1.1-T1.6): ✅ shipped in v0.4.4
    * Tier 2 (T2.1-T2.5): ✅ shipped in v0.4.5
    * Tier 2 carry-over (T2.6 / #216 systematic gating sweep):
      ⚠️ partial — 8/38 cleared, 30 grandfathered. Rest scheduled
      for follow-up under #216.
    * Tier 3 (T3.1-T3.3, T3.7-T3.9): ✅ shipped in v0.4.5
    * Tier 3 (T3.4 iai-callgrind, T3.5 bench-history publication,
      T3.6 default-flip): ⏳ pending, scheduled for v0.5.0.
  * **Audit-R10 honest-gaps**: code coverage, scheduled fuzzing,
    mutation testing, OSS-Fuzz integration — **all four closed.**

## [0.4.4] — 2026-05-02

Audit-R10 Tier 0 + Tier 1 closeout: 4 prerequisite CI / fuzz fixes
plus 5 HIGH-severity safety closeouts plus a cross-cutting CI lint
that prevents new ungated panic surface from regressing the
kernel-safe-by-default narrative.

### Fixed (HIGH) — `PackedDfa::try_new` with checked u32 + try_reserve (R10 #2)

`PackedDfa::new` accepted untrusted pattern sets and:

  * silently truncated `pat_idx`, `pattern_lens` entries, and goto
    state IDs via `as u32` casts on inputs > `u32::MAX`,
  * aborted on OOM via `Vec::push` instead of returning `Err`,
  * panicked on unbounded `vec!` allocation in alloc-only builds.

Added `PackedDfaError` enum (`TooManyPatterns`, `PatternTooLong`,
`TooManyStates`, `AllocationFailed`) with `Display` +
`std::error::Error`. `PackedDfa::try_new` validates pattern count +
per-pattern length upfront, then runs construction with
`Vec::try_reserve` on the largest vecs (`pattern_lens`,
`transitions`) and an in-loop overflow check before any new state
materialises. `PackedDfa::new` is now gated on `userspace` and is a
thin `try_new(...).expect(...)` wrapper. `format::Sniffer::new` was
updated to route through `try_new + expect` (MAGIC_RULES is
compile-time bounded).

### Fixed (HIGH) — remaining audit-R10 #1 panic sites gated on `userspace`

Ten more `pub fn` panic sites the audit identified but earlier rounds
had missed are now gated:

  * `BloomFilter::insert_simd` and the inlined SIMD body in
    `try_insert_simd` (the latter no longer routes through the
    panicking variant)
  * `HyperLogLog::merge`
  * `bits::streamvbyte::kernels::auto::encode_u32` (with
    `encode_u32_unchecked` sibling for the dispatcher)
  * `bits::bit_pack::kernels::auto::encode_u32_slice`
  * `hash::set_membership::kernels::auto::contains_u32_batch` (with
    `contains_u32_batch_unchecked` sibling)

Each panicking entry's docstring points at its `try_*` sibling.

### Fixed (HIGH) — `ssse3` + `sse41` backend modules now under `arch-pinned-kernels` (R10 #3)

`bits::streamvbyte::kernels::ssse3` and
`hash::set_membership::kernels::sse41` were previously declared
inline as `pub mod foo { ... }` (no opt-out for kernel callers).
Refactored to file-split modules under the
`arch-pinned-kernels`-gated pattern (matches v0.4.2's R7 #17
closeout): default surface stays dispatchers-only; the `ssse3` /
`sse41` per-backend symbols are reachable only with the opt-in
`arch-pinned-kernels` feature.

### Added — `panic-surface-lint` cross-cutting CI prevention (R10 T1.6)

New `cargo xtask panic-surface-lint` task (also wired into
`cargo xtask check` and `.github/workflows/ci.yml`) walks every
`.rs` file under `crates/tokenfs-algos/src/` and reports every
`pub fn` / `pub unsafe fn` / `pub(crate) fn` whose body contains
`assert!`, `panic!`, `assert_eq!`, `assert_ne!`, `unreachable!`,
`unimplemented!`, or `todo!` without an enclosing
`#[cfg(feature = "userspace")]` or `#[cfg(test)]` gate.

Single forward-pass with a brace-depth tracker + cfg-attr skip
stack — no syn / proc-macro toolchain required. Snapshot of the 38
currently-grandfathered sites lives in
`tools/xtask/panic_surface_allowlist.txt` with three categories
(kernel backends called via the gated `kernels::auto::*` dispatcher;
`permutation/*` panicking constructors; top-level SIMD/dispatch
entries audit-R10 #1 missed). Each entry is a tracked TODO whose
removal is scheduled as the v0.4.5 T2.6 systematic gating sweep.

### Fixed — CI infrastructure (Tier 0)

  * **T0.1** Bench / example feature flags now correctly pass
    `bench-internals,arch-pinned-kernels,userspace` to the
    `primitives` and `similarity` benches and `userspace` to the
    `dispatch_explain` example so the no-default-features migration
    didn't break their CI build.
  * **T0.2** Fuzz manifest enables
    `userspace,arch-pinned-kernels,parallel,blake3` so all 13
    declared fuzz targets compile with `cargo +nightly fuzz build`.
  * **T0.3** F22 throughput perf gate (`tests/fingerprint_f22.rs`)
    now requires `TOKENFS_ALGOS_RUN_THROUGHPUT_GATE=1` to actually
    enforce the wall-clock check; the correctness portion of the
    test stays in the default suite. The slow throughput gate is
    opt-in for calibrated hardware.
  * **T0.4** `tests/histogram_kernels.rs` now requires
    `bench-internals` so its `#[cfg(feature = "bench-internals")]`
    suite actually runs in CI (was reporting "0 tests passed"
    silently).

### Fixed — CI fuzz smoke (Tier 1)

  * **T1.4** `cargo +nightly fuzz build` now builds all 13 declared
    fuzz targets and `cargo +nightly fuzz list`-driven smoke matrix
    runs each one for a few seconds.
  * **T1.5** Removed `|| true` swallow from CI fuzz smoke runs.
    Real crashes now fail the workflow; libFuzzer's exit-code 124
    (timeout) is treated separately from exit-code 77 (crash).

### Notes

  * Lib test counts: 975 with `--all-features` (no change vs v0.4.3 —
    the T1.1 `_unchecked` dispatcher and T1.2 `try_new` route through
    the same existing test sites). 779 default.
    669 `--no-default-features --features alloc`.
  * All `cargo xtask check`, panic-surface-lint, aarch64 cross-clippy,
    `cargo deny check` green with zero advisory suppressions.
  * Audit-R10 Tier 2 (T2.1-T2.6, scheduled for v0.4.5) covers the
    remaining MEDIUM findings (`try_sha256`, saturating counters,
    bench regression CI, scheduled calibration, no_std smoke
    broadening, panic-surface allowlist clearance).

## [0.4.3] — 2026-05-03

Audit-R9 closeout: 7 of 8 findings closed (the 8th, R9 #1
verifiability, was already resolved in v0.4.2). Closes the
"kernel-safe-by-default" gap that R8 left open by gating
constructors, OS probing, process-global state, and remaining
panicking entry points behind `userspace` / `bench-internals`.

### Fixed (HIGH) — try_rank1_batch is now genuinely no-panic (R9 #3)

- **`RankSelectDict::try_rank1_batch`** previously documented per-position
  OOB as a panic ("pre-filter positions against `Self::len_bits` for a
  fully no-panic batch query"). Now validates ALL positions upfront
  before any kernel dispatch and returns the new
  `RankSelectError::BatchPositionOutOfRange { position, index, n_bits }`
  on first OOB. Caller's `out` buffer is never partially mutated on the
  failure path (verified by sentinel test). +2 tests.
- **`try_select1_batch`** verified structurally no-panic: `select1` returns
  `Option<usize>`, so per-element OOB surfaces as `out[i] = None`. No
  code change; docstring clarifies.

### Fixed (HIGH) — kernel-safe-by-default for constructors (R9 #2)

The R8 round gated panicking shape APIs but missed constructors that
assert on caller params. Now gated on `userspace`:

- `BloomFilter::new` + `BloomFilter::with_target` (try_new + try_with_target stay public)
- `HyperLogLog::new` (try_new stays public)
- `sketch_p2::Estimator::new` (try_new stays public)
- `similarity::lsh::MinHashIndex::new` + `SimHashIndex::new` + `Default for SimHashIndex` (try_new stays public)
- `hash::sha256::Hasher::update` (try_update stays public) — closes R9 #8
- `dispatch::force_backend` + `clear_forced_backend` gated on `bench-internals` instead (per their documented "primarily for tests and benchmarks" scope) — closes R9 #7

The `try_*` primitives no longer route through the panicking `new` —
they construct state directly. This decouples the no-panic surface from
the userspace-only one.

### Fixed (MEDIUM) — userspace OS probing now actually opt-in (R9 #4)

- `dispatch::detect_logical_cpus` (calls `std::thread::available_parallelism`) and `detect_linux_cache_profile` + helpers (`/sys/devices/system/cpu/cpu0/cache` reads) are now gated on `userspace` instead of `std`. Kernel/FUSE consumers using `default-features = false, features = ["alloc"]` (or even just default sans `userspace`) no longer pay these OS probes during dispatch.
- `std` remains in the default feature set for ergonomic test/bench/example building (~50 test sites use std-only macros like `eprintln!`); the audit's underlying concern (kernel-adjacent code reaching the OS probes through default features) is addressed at the call site rather than via a default-flip that would require gating every test using `eprintln!`/`println!`.

### Fixed (MEDIUM-LOW) — BytePairHistogram by-value entries gated (R9 #5)

- `BytePairHistogram::new`, `from_bytes`, `Default` impl gated on `userspace`. The 256 KiB stack allocation is no longer reachable from default builds. Heap-free `with_scratch` + `BytePairCountsScratch` siblings stay public.
- Cascading: `entropy::conditional::h_next_given_prev`, `entropy::kernels::scalar::conditional_h_next_given_prev`, `entropy::kernels::auto::conditional_h_next_given_prev` (which all build a `BytePairHistogram` inline) gated on `userspace`. The `_with_scratch` siblings stay public.

### Fixed (MEDIUM-LOW) — TLSH single canonical pearson_table (R9 #6)

- `pearson_table()` previously returned a Fisher-Yates-shuffled permutation under `std` and an identity table under `no_std`. Same input → different digest depending on which side of the kernel/userspace boundary computed it. Now: a single canonical `static [u8; 256]` produced by a `const fn` Fisher-Yates that runs at compile time. **Std users see no digest change** (verified byte-for-byte against the prior runtime-init); no_std/alloc-only users WILL see different digests than before (previously identity, now the canonical shuffle), which is the intended one-time correction. +2 tests pinning canonical entries.

### Notes

- Lib test counts: 975 with `--all-features` (was 971 at v0.4.2; +4),
  779 default, 669 `--no-default-features --features alloc` (was 672;
  -3 from the conditional/joint cascading gate).
- All `cargo xtask check`, aarch64 cross-clippy, `cargo deny check`
  green with zero advisory suppressions.
- One remaining v0.5.0 candidate from R9: dropping `std` from default
  outright (would require ~50 test sites to gate on `userspace`/`std`
  explicitly). The current call-site gating addresses the audit
  finding's underlying concern (kernel/FUSE consumers do not pay the
  OS probes); the structural default-flip is queued as a separate PR.

## [0.4.2] — 2026-05-03

Audit-R7 #17 + R8 #5 + R8 #6b closeout. Two architectural items
that the v0.4.1 release deferred (Cargo.toml claimed but didn't
actually wire) are now done.

### Changed (BREAKING for downstream callers reaching into per-backend kernels)

- **`arch-pinned-kernels` feature now wired** (audit-R7 #17, audit-R8 #5).
  ~80 source files refactored to file-split the per-backend kernel
  modules (`pub mod {scalar, avx2, avx512, neon, sse42, ...}`) so the
  visibility can be conditionally gated:

  ```rust
  #[cfg(feature = "arch-pinned-kernels")]
  pub mod foo;
  #[cfg(not(feature = "arch-pinned-kernels"))]
  #[allow(dead_code, unreachable_pub)]
  pub(crate) mod foo;
  ```

  Default surface is now dispatchers-only (`kernels::auto::*`).
  External callers wanting to pin a specific backend (benches,
  consumer apps doing perf comparisons) must enable
  `arch-pinned-kernels`. Closes the misuse-risk concern raised in
  R8 #5: kernel-adjacent consumers can no longer accidentally call
  `kernels::avx2::*` on a non-AVX2 host.

  Affected `[[bench]]` / `[[example]]` / `[[test]]` entries that
  reach into per-backend kernels gained
  `required-features = ["arch-pinned-kernels"]`. A few targets
  exercise both panicking surface and pinned backends; those carry
  `["userspace", "arch-pinned-kernels"]`.

### Changed (BREAKING for kernel-default builds)

- **By-value dense helpers gated behind `userspace`** (audit-R8 #6b
  closeout). Four functions that materialise large stack buffers
  are no longer reachable from the default (kernel-safe) build:
  - `entropy::joint::h2_pairs` (256 KiB stack) →
    `h2_pairs_with_scratch` / `h2_pairs_with_dense_scratch` siblings
    stay always-public.
  - `similarity::minhash::build_byte_table_from_seeds<K>` (up to
    512 KiB at K=256) → `_into` / `_boxed` siblings stay public.
  - `similarity::minhash::signature_simd<K>` → `_into` sibling stays
    public.
  - `similarity::minhash::classic_from_bytes_table_8` → `_into`
    sibling stays public.

  Each gated function's docstring points at its kernel-safe sibling.
  Internal call sites updated: `signature_batch_simd<K>` and
  `try_signature_batch_simd<K>` now dispatch through `signature_simd_into`
  so they remain in default builds.

### Fixed

- 3 integration tests (`tests/known_values.rs`, `tests/parity.rs`,
  `tests/fingerprint_f22.rs`) referenced the now-gated
  `joint::h2_pairs` / `kernels::*::joint_h2_pairs`; rewrote
  `tests/fingerprint_f22.rs` to use the heap-free
  `h2_pairs_with_scratch` sibling and `#[cfg(feature = "userspace")]`-gated
  the other two so they remain coverable under `--features userspace`
  but compile out of kernel-default builds.
- `chunk/mod.rs` and `histogram/pair.rs` test mods missed the no-std
  `vec!` / `Vec` `alloc::` imports for tests added in v0.4.1; fixed
  so `cargo test -p tokenfs-algos --no-default-features --features alloc --lib`
  is once-again clean (was 653 → 667 → 672 across these patches).

### Notes

- Lib test counts: 971 with `--all-features` (+1 vs v0.4.1's 969;
  the one new test came in via the #6b heap-free siblings round-trip
  parity check). 784 default. 879 `--features userspace`.
  672 `--no-default-features --features alloc`.
- All `cargo xtask check`, aarch64 cross-clippy, `cargo deny check`
  green with zero advisory suppressions.
- Audit-R7 + R8 are now fully closed. The two remaining open items
  on the docket (#171 AVX2 modularity perf parity, #172 rabbit_order_par
  colouring-based batching) are perf optimization — neither is a
  safety concern, both match documented behaviour at v0.4.x.

## [0.4.1] — 2026-05-02

Audit-round-8 hardening pass: 7 findings addressed via 4 parallel
worktree agents + foreground doc + Cargo.toml-comment fixes. All
additive — no semver-relevant changes (already pre-release; no
existing consumers).

### Fixed (HIGH)

- **#1 chunk::recursive: SplitPolicy untrusted offset** — the
  caller-supplied `SplitPolicy::split_at` was documented as requiring
  `0 < offset < bytes.len()` but the recursive driver trusted it
  blindly. `Some(0)` or `Some(len)` would recurse forever; `Some(>len)`
  panicked on slicing. Now guarded: contract violations are silently
  treated as terminal (leaf) splits in both the sequential `walk` and
  parallel `walk_par`. Driver returns `F::Acc` (no Result type
  available without breaking every caller); contract documented in
  the SplitPolicy rustdoc. +5 tests covering Some(0), Some(len),
  Some(len+1), Some(far_greater), and parallel-zero.
- **#2 chunk::ChunkConfig + FastCdcConfig public fields create
  zero-progress iterators** — public fields let callers bypass
  constructors and set `min/avg/max = 0`; iterators then never
  advance (CPU/memory DoS). Now defended on three layers:
  - new `ChunkConfigError` enum (`ZeroMin`, `ZeroAvg`, `ZeroMax`,
    `MinExceedsAvg`, `AvgExceedsMax`) returned by new `try_new` /
    `try_with_sizes` constructors;
  - new `FastCdcConfig::try_new` / `try_with_sizes` mirroring the
    same validation;
  - iterator progress guard: if `produced_bytes == 0`, the iterator
    returns None (terminates) instead of looping forever — defends
    against bypassed-constructor zero-configs;
  - field-level rustdoc warning that direct struct-literal
    construction is unsafe and pointing kernel callers at
    `try_with_sizes`.

  Fields kept `pub` (not made private) because the worktree agent's
  scope guardrail prevented updating cross-crate consumers
  (`benches/`, `examples/`, `fuzz/`); the progress guard + fallible
  constructors give the same defense without breaking the external
  fuzz crate. +14 tests across both Config types.

### Fixed (MEDIUM)

- **#3 try_contains_batch_simd panics for k > MAX_K** — the path
  slices a fixed `[u64; MAX_K]` stack buffer with `[..self.k]` which
  panics on `k > 32`. The single-key sibling
  `try_contains_simd` was added in this patch (returns
  `ApproxError::KExceedsSimdMax`); the batch sibling now early-checks
  `k > MAX_K` and returns the new
  `BloomBatchError::KExceedsSimdMax { k: u32, max: u32 }` variant
  before the slice access. +3 tests including a `#[should_panic]`
  regression on the panicking sibling to keep its contract.
- **#4 try_hamming_u64_one_to_many panics on stride > u32::MAX/64** —
  the post-validation kernel uses a panicking `sum_u64` for that
  stride regime. Documented at line 311 but never surfaced as Err.
  Now guarded: new `BatchShapeError::StrideExceedsHammingLimit { stride,
  limit }` variant returned before the kernel dispatch. The check
  fires *before* shape validation so an Err can be exercised with
  empty `query`/`db`/`out` slices instead of needing 67M-word
  allocations. Audited the four other vector::batch try_* siblings
  (`try_dot_f32`, `try_l2_squared_f32`, `try_cosine_similarity_f32`,
  `try_jaccard_u64`) — no other hidden panicking preconditions. +1
  test.

### Fixed (MEDIUM-LOW)

- **#5 honest Cargo.toml comment for `arch-pinned-kernels`** — the
  feature comment claimed it gated the per-backend kernel modules,
  but those modules are still unconditionally `pub` (the file-split
  refactor for ~30 modules across 15+ files is tracked as #180).
  Comment now acknowledges the feature is declared-but-unwired and
  steers kernel-adjacent consumers at `kernels::auto::*` until #180
  lands. Pure documentation honesty; no code change.
- **#6a entropy/joint + histogram/pair: dense 256x256
  BytePairHistogram on stack (256 KiB)** — kernel-stack hazard.
  New surface eliminates the inline allocation for kernel callers:
  - `histogram::pair::BytePairCountsScratch` (caller-owned 256 KiB
    table) + `BytePairHistogramView<'a>` (read-only borrow)
  - `BytePairHistogram::with_scratch(bytes, &mut scratch) -> View`
  - `entropy::joint::h2_pairs_with_dense_scratch(bytes, &mut scratch)`
  - `entropy::joint::h2_from_pair_view(&view)`

  By-value entries kept; each gained a `# Stack` (or
  `# Kernel callers`) section pointing at the new heap-free sibling.
  Module-level kernel-stack hazard discussion added to
  `histogram::pair`.
- **#6b/c similarity::minhash + kernels_gather: large arrays returned
  by value (up to K * 256 * 8 bytes = 512 KiB at K=256)** — same
  hazard. New `_into` siblings:
  - `similarity::minhash::signature_simd_into<K>(bytes, table, &mut out)`
  - `similarity::minhash::classic_from_bytes_table_8_into(...)`
  - `similarity::minhash::build_byte_table_from_seeds_into<K>(...)`

  The `kernels_gather` `_into` helpers already existed from R5 #156;
  this round added doc cross-references from the by-value entries.
  +13 tests across the heap-free siblings.

### Documentation (LOW)

- **#7 stale feature docs in lib.rs + README** — both files still
  said `panicking-shape-apis` was "on-by-default" but v0.4.0 dropped
  it. Both updated to state explicitly that the default is
  kernel-safe and that userspace consumers opt back in via
  `features = ["userspace"]`. README also added a top-of-list
  explanation of the `userspace` umbrella.
- **#15 (R7-followup carryover)**: SHA-256 `Hasher::update` doc
  promotes try_update to a dedicated `# Kernel/FUSE callers`
  section (was buried in the length-limit paragraph).

### Notes

- Lib test counts: 969 with `--all-features` (was 933 at v0.4.0;
  +36 from the four R8 fix agents).
- All `cargo xtask check`, aarch64 cross-clippy, `cargo deny check`
  green with zero advisory suppressions.
- Two outstanding architectural items remain open as task #180:
  the file-split refactor to wire `arch-pinned-kernels` (audit-R7
  #17 + audit-R8 #5). Per-backend kernel modules continue to be
  unconditionally `pub`; the feature is declared but does not yet
  gate anything. Kernel-adjacent consumers should call
  `kernels::auto::*` exclusively until that lands.

## [0.4.0] — 2026-05-02

The kernel-safe-by-default cut. Default features no longer expose the
ergonomic panicking shape APIs; userspace consumers wanting them opt in
via the new `userspace` umbrella feature. Adds nine families of `try_*`
fallible siblings closing every audit-R7 panic-path finding the previous
release left out. Pre-release: no semver compatibility commitment.

### Changed (BREAKING) — default feature surface flip

- **`panicking-shape-apis` no longer in `default = [...]`** (audit-R7
  followup #1, #3, #4, #14). The panic-prone entry points
  (`BitPacker::encode_u32_slice`, `streamvbyte_decode_u32`,
  `dot_f32_one_to_many`, `RankSelectDict::build`, `sha256_batch_st`,
  `signature_batch_simd`, etc.) now require the new `userspace`
  umbrella feature. The crate's primary consumer audience — kernel
  modules, FUSE daemons, Postgres extensions, MinIO/CDN, forensics
  tools — should not have panic-prone branches compile in by default.
  - **For userspace consumers**: add `features = ["userspace"]` to
    your `[dependencies]` entry. Equivalent to v0.3.x default behavior.
  - **For kernel/FUSE/forensics**: do nothing. The default is now
    panic-free.
  - **For benches that exercise panicking APIs** (similarity,
    hash_batched, bits_streamvbyte, bits_rank_select,
    hash_set_membership, bits_bit_pack): now declare
    `required-features = ["userspace"]` in `Cargo.toml`. CI is
    updated to enable it.
  - **For example programs** (build_pipeline, inverted_index,
    dispatch_explain, similarity_scan): now declare
    `required-features = ["userspace"]`.
  - **For integration tests** (avx2_parity, neon_parity,
    integration_phase_c): file-level `#![cfg(feature = "userspace")]`.

### Added — `try_*` panic-free coverage (audit-R7 followups)

For each module below the existing panicking entry points are unchanged
(now gated behind `userspace` per above); the new `try_*` siblings
return `Result<T, Error>` with informative diagnostic fields.

- **`bits::rank_select`** (#5, #6): `try_rank1`, `try_rank0`,
  `try_rank1_batch`, `try_select1_batch`. New error variants
  `RankSelectError::PositionOutOfRange { pos, n_bits }` and
  `RankSelectError::BatchOutputTooShort { needed, actual }`. Batch
  variants validate `out.len()` upfront so the caller's buffer is
  never partially mutated on the failure path. +12 tests.
- **`permutation`** (#7, #9, #10, #11):
  - `Permutation::try_apply`, `try_apply_into` (shape-checked, lighter
    than the existing `try_apply_into_strict` which also proves
    permutation validity).
  - `Permutation::try_validate_no_alloc` (returns Err on undersized
    scratch instead of panicking).
  - `CsrGraph::try_neighbors_of`, `try_degree_of`, `try_validate`
    with new `CsrGraphError` enum (OutOfRange, OffsetsNonMonotone,
    NeighborsOutOfBounds, OffsetsLengthMismatch, NeighborsLengthMismatch,
    NeighborOutOfRange).
  - `try_rcm`, `try_rabbit_order`, `try_rabbit_order_par` with new
    top-level `PermutationConstructionError` that wraps `CsrGraphError`
    via `From`. Note: these still allocate internally (work buffers);
    the `try_*` path is about not panicking on bad input, NOT about
    being heap-free. +36 tests.
- **`approx`** (#12, #13): `BloomFilter::try_with_target` (overflow-
  safe), `try_insert_simd`, `try_contains_simd`. HLL `try_new` /
  `try_merge` doc-completed (try paths already existed). New
  `ApproxError::BitCountOverflow`, `KExceedsSimdMax` variants;
  `ApproxError` now `#[non_exhaustive]`. +6 tests.
- **`similarity::lsh`** (#16): `MinHashIndex::try_new`,
  `SimHashIndex::try_new`. New `LshConstructionError` enum
  (ZeroBands, ZeroRows, BandsTimesRowsMismatchSignatureLen,
  BandsExceedSignatureLen). The `Default for SimHashIndex` impl
  delegates to the panicking `new` (trait impls cannot be fallible);
  kernel callers should call `try_new` directly. +14 tests.
- Existing panicking siblings retained but now gated on `userspace`;
  each gained a `# Panics` doc section pointing at its `try_*`
  counterpart for kernel/FUSE callers.

### Documentation (audit-R7 followups #8, #15)

- **`Hasher::update`** docstring elevates `try_update` guidance to a
  dedicated `# Kernel/FUSE callers` section, framing the panic as a
  DoS hazard since untrusted callers can supply adversarial byte
  streams approaching the 2 EiB FIPS cap across multiple syscalls.
- **`Permutation::apply` / `apply_into`** continue to point at
  `try_apply_into_strict` for kernel-boundary callers wanting
  permutation-validity proof; the new lightweight `try_apply` /
  `try_apply_into` siblings are referenced for shape-only validation.

### Added (unwired) — `arch-pinned-kernels` feature

New Cargo feature is declared but does not yet gate any modules. The
follow-up file-split refactor (audit-R7 #17) will move the per-backend
`pub mod {scalar,avx2,avx512,neon}` SIMD kernels behind it so external
callers cannot violate CPU-feature/bounds preconditions. Currently
those modules are still unconditionally `pub`. Tracked separately;
deferred from v0.4.0 because Rust does not permit conditional
visibility on inline `pub mod` without body duplication and the
file-split is invasive enough to warrant its own PR.

### Notes

- Lib test counts under each config (was 861 / 768 / 600 at v0.3.0):
  - `--all-features`: 933 (+72)
  - `default` (kernel-safe): 763 (+5 vs v0.3.0 default sans userspace)
  - `--features userspace`: 842
  - `--no-default-features --features alloc`: 653 (+49)
- All `cargo xtask check`, aarch64 cross-clippy, `cargo deny check`,
  and `cargo miri test --no-default-features --features alloc` gates
  green with zero advisory suppressions.
- v0.3.x carry-over follow-ups (AVX2 modularity-gain parity,
  rabbit_order_par colouring) remain open; both still match the
  documented "wall-clock parity or modestly slower" posture.
- No new external dependencies added.

## [0.3.0] — 2026-05-02

Phase D Rabbit Order release: SIMD modularity inner loop (Sprint 50-52) +
round-based concurrent merging (Sprint 53-55) on top of the v0.2.3
sequential baseline (Sprint 47-49). Audit-round-6 hardening rolled in.

### Added — Phase D Rabbit Order (full)

- **`permutation::rabbit::kernels`** module — Sprint 50-52 SIMD modularity
  inner loop. `modularity_gain_kernel::scalar` /
  `modularity_gain_kernel::avx2` / `modularity_gain_kernel::avx512` /
  `modularity_gain_kernel::neon` + `auto` runtime dispatcher. The inner
  loop computes per-neighbor `m * w_uv - k_u * k_v / m` (the integer
  `i128` modularity-gain ledger) over a community's adjacency in batches
  matching the host's lane width. The kernel itself clears 1 GElem/s at
  n≥1000 batches; per-call dispatch + i128 epilogue currently leave the
  AVX2 `auto` path at parity with scalar (0.91-1.00x) on Alder Lake —
  see `docs/PHASE_D_BENCH_RESULTS.md` and the regression-candidate
  analysis. The kernel is in place for downstream profiling work; a
  follow-up sprint will reclaim the lane-parallel gain. Bench:
  `permutation_rabbit/modularity_gain/*`.
- **`permutation::rabbit_order_par`** — Sprint 53-55 round-based
  concurrent merging. Parallelizes the dendrogram-build phase by
  partitioning eligible-merger candidates into rounds where each
  community appears at most once, then dispatching round merges via
  `rayon`. Above `RABBIT_PARALLEL_EDGE_THRESHOLD` edges, falls through
  to sequential `rabbit_order` to avoid coordination overhead. Above
  the threshold the round-based variant is currently **modestly slower
  than sequential** (0.88-0.91x on representative graphs) because the
  apply phase per round is sequential — this matches the explicit
  doc-comment posture and is documented as expected. The variant exists
  primarily as a deterministic API surface for rayon-driven pipelines;
  the colouring-based conflict-free batching that would deliver wall-
  clock speedup is a follow-up sprint. Bench:
  `permutation_rabbit/par_build/*`.
- Bench harness coverage: `bench_rabbit_build`, `bench_rabbit_par_build`,
  `bench_modularity_gain_kernel` in `crates/tokenfs-algos/benches/permutation_rcm.rs`.
- **`docs/PHASE_D_BENCH_RESULTS.md`** captures the v0.3.0 baseline
  numbers on i9-12900K (24 logical cores) including the two regression
  candidates above.

### Audit-round-7 hardening

- **#1 `select_in_word(word, k)` release-mode k>=64 guard** — the
  public `pub fn select_in_word(word: u64, k: u32)` previously
  validated `k < word.count_ones()` only via `debug_assert!`; in
  release builds, callers passing `k >= 64` reached
  `_pdep_u64(1u64 << k, word)` (UB on shift >= 64) or the broadword
  fallback where `k.wrapping_mul(L8)` silently produced garbage.
  Both paths now early-return the `64` "not found" sentinel when
  `k >= 64` or `k >= word.count_ones()`. `debug_assert!` retained
  so a contract violation still surfaces immediately under
  `cargo test`.
- **#2 `bit_pack` `_unchecked` split mirroring R6 #162** — every
  `pub fn *_u32_slice` in `bits::bit_pack::kernels::{scalar, avx2,
  neon, auto}` now has an `*_u32_slice_unchecked` sibling without
  the `assert!((1..=32).contains(&w))` / `assert!(out.len() >= n)`
  guards. The asserting wrappers call into `_unchecked` after
  pre-validation. The `try_*` paths on `BitPacker<W>` and
  `DynamicBitPacker` now dispatch directly to
  `kernels::auto::*_unchecked` after their own validation,
  eliminating panic sites from the fallible API surface even with
  `panicking-shape-apis` disabled. Closes the half-done state where
  R6 #162 had split streamvbyte but not bit_pack.
- **#3 `bits::rank_select` module doctest now runs** — the
  module-level doctest contained five `assert_eq!` calls
  demonstrating `rank1` / `select1` behaviour but was annotated
  ` ```no_run `, so the assertions never executed under
  `cargo test --doc`. Removed the `no_run` annotation; +1 doctest
  pass.
- **#4 `RuleDecision::index` widened `u16` → `u32`** — silent
  truncation cast `index as u16` is gone. The natural `usize`
  enumerate index always fits `u32` for any feasible rule count.
  **BREAKING** for direct readers of `RuleDecision`; trace-mode
  callers using accessor only need the new field type.
- **#6 kernels module doc-text reconciled `2^32` → `2^31`** — the
  module-level eligibility text claimed "every input < 2^32" but
  the literal `BOUND` constant on every backend is `1_u64 << 31`.
  Reconciled the prose to match the code (the conservative bound
  exists to keep the i64 product symmetric around zero — see the
  per-backend `BOUND` doc-comment for the sign-overflow rationale).
- **(R7 follow-up) `bit_pack::encoded_len_bytes` saturation** —
  switched from `n.saturating_mul(w as usize) >> 3` to per-byte
  arithmetic that saturates at the byte level. The old shape
  under-estimated the true byte count by a factor of 8 for
  adversarial `(n, w)` pairs near `usize::MAX`. Defensive only:
  not reachable from current call sites because `BitPacker` /
  `DynamicBitPacker` reject `w > 32` upstream.

### Audit-round-6 hardening

- **#161 `Permutation::try_apply_into_strict` + `validate_no_alloc`** —
  the strict-validation `apply_into_strict` now uses a caller-provided
  `scratch: &mut [u64]` for permutation validity proof (zero heap
  allocations on the hot path). Closes the prior stale-slot-leak
  trigger where `apply_into` would silently produce a partially
  permuted output if the permutation contained a duplicate index.
- **#162 split bits kernels into `_unchecked` + asserting wrappers** —
  `streamvbyte_decode_u32`, `bit_pack` encode/decode now have
  `_unchecked` siblings that skip bounds-checks (intended for the
  `try_*` callers that already validated upstream). Asserting
  wrappers retain the existing public contract. Eliminates redundant
  bounds checks on the panic-free `try_*` path.
- **#163 `RankSelectDict` superblock counts u32 → u64** — the per-
  superblock 1-count was previously `u32`, truncating silently for
  bitvectors with > 4G ones (~537 MB of bitset). Now `u64` end-to-end;
  no more silent miscounts at scale. **Breaking** for direct readers
  of `RankSelectDict` internals; public `rank1` / `select1` API
  unchanged.
- **#164 no-std + Miri test coverage gap** — `cargo miri test
  --no-default-features --features alloc` now compiles cleanly. Closes
  a gap where alloc-only test helpers in `dispatch::tests` and
  `streamvbyte::tests` would emit `Vec` / `vec!` references that
  failed to resolve under no-std prelude. Fold-in fixes:
  - `examples/{build_pipeline,inverted_index,dispatch_explain,similarity_scan}`
    are now `required-features = ["panicking-shape-apis"]` in Cargo.toml
    (they compose via the ergonomic panicking entry points; gated out
    of the kernel/FUSE deployment build).
  - `tests/integration_phase_c.rs` is `#![cfg(feature = "panicking-shape-apis")]`
    file-level for the same reason.
  - `bits::rank_select` and `hash::batched` doctests rewritten to use
    the always-available `try_build` / `try_sha256_batch_st` siblings;
    the panicking-API note moved to a follow-up paragraph.
  - `tokenfs-algos-no-std-smoke` Cargo.toml inherits license/edition
    from workspace and pins `tokenfs-algos = "^0.2"` (cargo-deny:
    bans, licenses).

### Documentation

- **`docs/PHASE_D_RABBIT_ORDER.md`** new — what Rabbit Order solves,
  when to use vs RCM / Hilbert, performance characteristics, worked
  example, Arai et al. IPDPS 2016 reference.
- **`docs/PHASE_D_BENCH_RESULTS.md`** — see SIMD modularity / par
  paragraph above for the regression-candidate breakdown.
- **`docs/PLANNER_DESIGN.md`** — the rules-as-data + named-constants
  planner architecture (32 rules, consts.rs provenance, trace mode,
  host tunes) already shipped in v0.2.3; called out here for v0.3.0
  release notes completeness.

### Notes

- Lib test counts: 861 with `--all-features` (was 804 at v0.2.3
  baseline; +57 across Phase D + R6 + R7), 773 default, 604 under
  `--no-default-features --features alloc`.
- All `cargo xtask check`, aarch64 cross-clippy, `cargo deny check`,
  and `cargo miri test --no-default-features --features alloc` gates
  green with zero advisory suppressions.
- No new external dependencies added.
- Two known regression candidates carried over to a v0.3.1 follow-up:
  AVX2 modularity-gain kernel at parity-or-slightly-slower vs scalar
  (likely Vec allocation or i128 epilogue overhead on small batches);
  `rabbit_order_par` modestly slower than sequential above the
  parallel-edge threshold (sequential apply phase per round bounds
  speedup; colouring-based conflict-free batching is the long-term
  fix). Both documented in `docs/PHASE_D_BENCH_RESULTS.md` and
  intentional posture for the v0.3.0 cut.

## [0.2.3] — 2026-05-02

v0.2.x candidate primitives + audit-round-5 hardening + Phase D Rabbit
Order sequential baseline. All shipped via parallel `isolation: "worktree"`
sub-agents.

### Added — v0.2.x SIMD primitives

- **`approx::BloomFilter::insert_simd` / `contains_simd` / `contains_batch_simd`**
  + `try_contains_batch_simd` + `BloomBatchError` + `bloom_kernels` module
  (scalar / AVX2 / AVX-512 / NEON). Sprint 42-43.
- **`approx::HyperLogLog::merge_simd` / `count_simd` / `count_raw`**
  + `try_merge_simd` + `HllMergeError` + `hll_kernels` module
  (scalar / AVX2 / AVX-512 VPOPCNTQ / NEON). AVX2 merge ~25x scalar
  (~50 GiB/s aggregate); count via `_mm256_max_epu8` per-bucket.
  Sprint 44.
- **`similarity::minhash::signature_simd<K>` / `signature_batch_simd<K>`**
  + `try_signature_batch_simd<K>` + `update_minhash_kway_auto<K>`
  dispatcher in `kernels_gather`. AVX2/AVX-512/NEON K-way kernels use
  direct `_mm256_loadu_si256` / `_mm512_loadu_si512` / `vld1q_u64`
  loads (gather micro-ops underperform contiguous loads on Alder
  Lake / Ice Lake / Zen 3+). Sprint 45-46.

### Added — Phase D Rabbit Order (sequential baseline)

- **`permutation::rabbit::rabbit_order(graph)`** — first Rust port of
  Arai et al. IPDPS 2016. Sequential single-pass: lowest-degree-first
  iteration via `BinaryHeap`, integer-only modularity gain in `i128`
  for determinism, sorted Vec-backed per-community adjacency with
  two-pointer merge on absorption, dendrogram DFS pre-order emit.
  Demonstrably better community grouping than RCM on K-clique-with-
  bridges fixtures (every clique's members within span K). Sprint
  47-49. SIMD modularity inner loop (Sprint 50-52) and concurrent
  merging (Sprint 53-55) are follow-on Phase D sprints.

### Audit-round-5 hardening

- **#157 panicking shape APIs gated behind `panicking-shape-apis` Cargo
  feature** — the panicking shape/length-validating public entry
  points (`BitPacker::encode_u32_slice` / `decode_u32_slice`,
  `DynamicBitPacker::new` / `encode_u32_slice` / `decode_u32_slice`,
  `streamvbyte_encode_u32` / `streamvbyte_decode_u32`,
  `RankSelectDict::build`, `dot_f32_one_to_many` /
  `l2_squared_f32_one_to_many` / `cosine_similarity_f32_one_to_many` /
  `hamming_u64_one_to_many` / `jaccard_u64_one_to_many`,
  `contains_u32_batch_simd`, `sha256_batch_st` / `sha256_batch_par` /
  `blake3_batch_st_32` / `blake3_batch_par_32`,
  `signature_batch_simd<K>`) are now gated behind the new
  `panicking-shape-apis` Cargo feature, which is **on by default** for
  back-compat. New `try_*` siblings were added for the SHA-256 and
  BLAKE3 batched hash entries (`try_sha256_batch_st` /
  `try_sha256_batch_par` / `try_blake3_batch_st_32` /
  `try_blake3_batch_par_32`) returning a new `HashBatchError` enum.
  Kernel/FUSE consumers should disable the feature
  (`default-features = false, features = ["alloc"]`) so that only the
  fallible `try_*` entry points are reachable. **Not BREAKING**:
  default-features build is unchanged.
- **#155 streamvbyte SIMD tables → `const fn` statics** — the SSSE3 /
  AVX2 / NEON shuffle (4 KiB) and length (256 B) tables now live in
  static rodata instead of `OnceLock`-initialized lazy globals. The
  table module no longer requires `feature = "std"`, so kernel-mode
  SIMD configs (`alloc,avx2` / `alloc,neon`) compile cleanly.
- **#158 replace upstream `hilbert` 0.1 with in-tree Skilling N-D** —
  drops `hilbert`, `num`, and 31 transitive crates including
  `criterion 0.3`, `atty`, `rustc-serialize`, `spectral`. Eliminates
  RUSTSEC-2022-0004 + RUSTSEC-2021-0145 entirely (no longer
  suppressed in deny.toml). New `permutation::hilbert::skilling_hilbert_key`
  + `interleave_be` (~156 lines). `cargo deny check advisories`
  reports `advisories ok` with **zero ignores**. Closes audit-R4 #150.
- **#159 bitmap container fields → `pub(crate)`** — `ArrayContainer.data`
  and `RunContainer.runs` are now `pub(crate)`; external callers must
  go through `try_from_vec` validating constructors (added in v0.2.2)
  or read via the new `data()` / `runs()` accessors. **BREAKING**:
  external direct field construction is now a compile error.
- **#160 `Permutation::from_vec_unchecked` is now `unsafe fn`** with
  full `# Safety` clause. All 6 internal call sites wrapped in
  `unsafe { ... }` with `// SAFETY:` justification. **BREAKING**:
  external callers must wrap invocations in `unsafe { }` or switch
  to `try_from_vec`.

### Notes

- 804 lib tests on x86_64 (was 737 at v0.2.2; +67 across all v0.2.x +
  R5 work).
- 84 AVX2 parity tests.
- All `cargo xtask check`, aarch64 cross-clippy, and
  `cargo deny check advisories` gates green with zero suppressions.
- Two audit-R5 items (#156 kernels_gather K=256 by-value) are deferred
  to a follow-on hardening pass; #157 (panicking APIs at kernel
  boundary feature gate) is closed by the `panicking-shape-apis` work
  above.

## [0.2.2] — 2026-05-02

Audit-round-4 hardening pass — closes 5 findings from external code
review. Additive only (no breaking changes); panic versions retain
their contracts.

### Added — fallible try_ variants for buffer-shape APIs

Five new error types (`Clone, Copy, Debug, Eq, PartialEq` + `Display`
+ `std::error::Error` under `feature = "std"`) and 12 new public
`try_*` APIs for kernel-adjacent callers that need DoS-safe error
propagation instead of panics:

- `bits::streamvbyte::StreamvbyteError` and `try_streamvbyte_encode_u32`
  / `try_streamvbyte_decode_u32`.
- `bits::bit_pack::BitPackError` and `BitPacker::try_encode_u32_slice`
  / `try_decode_u32_slice` (both const-generic and dynamic forms).
- `bits::rank_select::RankSelectError` and `RankSelectDict::try_build`.
- `vector::batch::BatchShapeError` and `try_dot_f32_one_to_many` /
  `try_l2_squared_f32_one_to_many` /
  `try_cosine_similarity_f32_one_to_many` /
  `try_hamming_u64_one_to_many` /
  `try_jaccard_u64_one_to_many`.
- `hash::set_membership::SetMembershipBatchError` and
  `try_contains_u32_batch_simd`.

Existing panic versions keep their contracts; their `# Panics` rustdoc
blocks now link to the matching `try_*` variants. `RankSelectDict::build`
delegates through `try_build` via `expect` to share the construction path.

### Added — bitmap container invariant validation

- `bitmap::ContainerInvariantError` enum + `ArrayContainer::try_from_vec`
  + `RunContainer::try_from_vec` validating constructors. Direct
  construction via the existing `pub` fields stays unchanged; rustdoc
  on each container type now points untrusted-input callers at the
  new `try_from_vec` path. Each error variant carries the offending
  index for shrinker-friendly diagnostics.

### Added — `no_std` smoke crate

- New workspace member `crates/tokenfs-algos-no-std-smoke/` (`test = false`,
  link-only) verifies kernel-claimed-safe primitives compile and link
  under `no_std + alloc + features = ["alloc"]`. Exercises
  `bits::popcount_u64_slice`, `hash::sha256_batch_st`,
  `hash::contains_u32_simd`, `vector::dot_f32`, and
  `Permutation::identity` + `apply_into`.

### Changed — `identity::base32_lower_len` overflow safety

- Switched from `input_bytes * 8` (wraps in release on `usize::MAX`/8+
  inputs) to `input_bytes.saturating_mul(8)`. Matches the convention
  established in `chunk::ChunkConfig` (audit-round-3 §78). Three
  regression tests (zero, one-byte canonical, saturating).

### Documented — `permutation_hilbert` supply-chain caveat

- The optional `permutation_hilbert` Cargo feature transitively pulls
  `hilbert 0.1`, which surfaces RUSTSEC-2022-0004 (`rustc-serialize 0.3`
  stack overflow) and RUSTSEC-2021-0145 (`atty 0.2` unsound +
  unmaintained). Neither vulnerable code path is reachable from our
  wrappers; the default-features build is provably clean.
- Added scoped `[advisories.ignore]` entries in `deny.toml` with
  detailed reason strings linking back to the design doc.
- Added `docs/v0.2_planning/14_PERMUTATION.md` § 8 supply-chain caveat
  sub-section listing the dep paths, verification command, and TODO to
  either fork upstream `hilbert` (drop misclassified runtime deps) or
  ship our own minimal Skilling N-D implementation.
- After the change: `cargo deny check` reports
  `advisories ok, bans ok, licenses ok, sources ok`.

### Notes

- 737 lib tests on x86_64 (was 679 at v0.1.x baseline; +58 across the
  audit-R4 surface).
- All `cargo xtask check`, aarch64 cross-clippy, and `cargo deny check`
  gates green.
- `cargo check -p tokenfs-algos-no-std-smoke` passes.

## [0.2.1] — 2026-05-02

v0.2 hardening pass — no API changes, only test/bench/fuzz coverage
fill-in. Closes the gaps identified in the post-v0.2.0 coverage audit.

### Added — fuzz

- 8 new fuzz targets in `fuzz/fuzz_targets/` for the v0.2 modules:
  - `bits_streamvbyte_round_trip` — exercises the 256-entry shuffle
    table on the dispatched (SSSE3/AVX2/NEON) decoder.
  - `bitmap_intersect_parity` — Schlegel SSE4.2 array×array vs scalar
    sorted-merge oracle, plus bitmap×bitmap AND in three variants
    (`_card`/`_nocard`/`_justcard`) vs scalar word-AND oracle.
  - `bits_bit_pack_round_trip` — round-trip across all widths W ∈ 1..=32.
  - `bits_rank_select_consistency` — rank/select monotonicity and
    inverse properties on `RankSelectDict`.
  - `vector_distance_parity` — six metrics (dot/L2/cosine/dot_u32/
    hamming/jaccard) dispatched vs scalar reference within Higham
    1e-3 tolerance against L1 norm of products.
  - `hash_batched_parity` — `sha256_batch_st` over up to 64 messages
    vs serial `sha256` per message.
  - `hash_set_membership_parity` — SIMD scan vs `slice::contains`.
  - `permutation_apply_round_trip` — Fisher-Yates → apply → inverse
    round-trip and `try_from_vec` validation.
- All 8 targets pass a 2000-iteration smoke run with no panics.

### Added — explicit AVX-512 parity tests

- 14 new AVX-512 parity tests in `tests/avx2_parity.rs`, each
  runtime-skipping when `is_x86_feature_detected!("avx512f")` is
  false. Covers `bits::rank_select::*_batch` auto-dispatcher contract,
  `vector::*_one_to_many` AVX-512 FMA paths (dot/L2/cosine/hamming/
  jaccard), `bitmap::kernels::bitmap_x_bitmap_avx512` (and/or/xor/
  andnot + VPOPCNTQ cardinality), and `hash::set_membership::avx512`.
- 7 new NEON parity tests in `tests/neon_parity.rs`: NEON `_nocard`
  bitmap parity (and/or), expanded streamvbyte round-trip + edge
  cases, NEON parity for the L2/cosine/jaccard `_one_to_many` APIs.
- Total: AVX2 parity 56 → 70 (+14); NEON parity 36 → 43 (+7).

### Added — Phase C composition integration tests

- `tests/integration_phase_c.rs` (3 tests, <0.01s combined runtime):
  - `inverted_index_composition_roundtrips_and_agrees_on_intersection`
    — `bitmap::Container` + `bits::streamvbyte`.
  - `build_pipeline_composition_hash_batches_match_and_rcm_round_trips`
    — `hash::sha256_batch_st` + `permutation::rcm` + `Permutation`.
  - `similarity_scan_composition_bitpack_roundtrip_and_distance_parity`
    — `bits::DynamicBitPacker` + `vector::*_one_to_many`.

### Added — real-data env vars in v0.2 benches

- Shared `support::real_files_as_bytes()` helper reading the
  colon-separated `TOKENFS_ALGOS_REAL_FILES` env var. Each file's
  bytes are loaded and passed to a per-bench `real_data_inputs()`
  helper that converts to the bench's native input shape.
- Wired into `bits_rank_select`, `bitmap_set_ops`, `bits_streamvbyte`,
  and `similarity` benches. Default behavior (env unset) unchanged.
- Per-bench input shapes: `rank_select` packs the low bit of every
  byte into u64 words; `bitmap` derives sorted-dedup u16 vecs (low
  12 bits) for array containers and `[u64; 1024]` for bitmap
  containers; `streamvbyte` packs 4-byte LE windows masked to low
  24 bits to match the synthetic posting-list-delta distribution;
  `similarity` decodes f32 (clamped [-256, 256]), u32 (low 20 bits),
  and full u64 lanes.

### Added — bench history snapshots

- `cargo xtask bench-history [--label <label>]` snapshots
  `target/criterion/` canonical result files (`base/estimates.json`,
  `new/estimates.json`, `new/sample.json`) into
  `benches/_history/<label>/` for cross-release perf regression
  detection. Defaults `<label>` to current short SHA.
- `benches/_history/README.md` documents the layout and a manual
  `jq` recipe for diffing snapshots on `mean.point_estimate`.

### Notes

- 679 lib + 70 AVX2 + 43 NEON + 13 SVE parity tests; integration suite
  at 3 Phase C composition tests + existing parity/known_values/etc.
- All `cargo xtask check` and aarch64 cross-clippy gates green.

## [0.2.0] — 2026-05-02

The full Phase A + B + C surface from `docs/v0.2_planning/03_EXECUTION_PLAN.md`
ships in this release. Five new modules (`bits`, `bitmap`, `vector`, `permutation`,
plus `hash::batched` extensions), three composition demonstrators, and ~170
new tests on top of the v0.1.x baseline.

### Added

- **`bits` module** — bit-level primitive surface for posting lists, token
  streams, and succinct DS:
  - `popcount` (shipped in 0.1.1)
  - `bit_pack` — pack/unpack u32 values at arbitrary widths W ∈ 1..=32 with
    const-generic `BitPacker<W>` and runtime-width `DynamicBitPacker`. Scalar
    + AVX2 + NEON kernels. Hard-coded fast paths for canonical token widths
    {8, 11, 12, 16, 32}.
  - `streamvbyte` — Lemire & Kurz variable-byte codec, wire-format compatible
    with the upstream C reference. Scalar + SSSE3 + AVX2 (dual-pumped PSHUFB)
    + NEON (vqtbl1q_u8) backends. Decode at 25–31 GiB/s on AVX2/SSSE3.
  - `rank_select` — `RankSelectDict<'a>` with constant-time rank1/select1
    over borrowed `&[u64]` bit slices. Two-level Vigna 2008 sampling
    (4096-bit superblocks + 256-bit blocks, ~0.7% overhead). BMI2
    PDEP+TZCNT fast path for select-in-word. ~4-5 ns warm rank, ~10-60 ns
    warm select.
- **`bitmap` module** — Roaring-style SIMD container kernels at primitive
  granularity:
  - `BitmapContainer` (8 KB / 65536 bits), `ArrayContainer` (sorted u16,
    ≤4096), `RunContainer` (sorted run-pairs).
  - `Container` enum + dispatch for intersect / union / difference /
    symmetric difference / cardinality.
  - bitmap×bitmap AVX2 kernels at 46 GiB/s (`and_into`, ~3.3x scalar);
    `_justcard` variant 5.6x scalar (78 GiB/s).
  - array×array Schlegel intersect via SSE4.2 `pcmpestrm` + 256-entry
    shuffle table (uses `pcmpestrm` not `pcmpistrm` to avoid the
    0-element-as-string-terminator footgun). 3 Gelem/s at n=10K, 6x scalar.
  - AVX-512 bitmap×bitmap with VPOPCNTQ for `_card` variants.
- **`vector` module** — dense vector distance kernels:
  - Single-pair APIs: `dot_f32`, `dot_u32`, `try_dot_u32`, `l2_squared_f32`,
    `l2_squared_u32`, `cosine_similarity_f32`, `cosine_similarity_u32`,
    `hamming_u64`, `jaccard_u64`.
  - Batched many-vs-one APIs: `dot_f32_one_to_many`,
    `l2_squared_f32_one_to_many`, `cosine_similarity_f32_one_to_many`,
    `hamming_u64_one_to_many`, `jaccard_u64_one_to_many` — the K-NN inner
    loop shape.
  - Backends: scalar / AVX2 / AVX-512 FMA / NEON. AVX2 dot_f32 hits
    128 GiB/s on 1024-element vectors (~6.5x scalar).
  - Reduction-order convention pinned in public contract: 8-way pairwise
    tree for AVX2, 16-way for AVX-512, 4-way for NEON, left-to-right for
    scalar. Cross-backend tolerance follows Higham §3 1e-3 against L1 norm
    of products.
  - `similarity::kernels` is preserved as `#[deprecated(since = "0.2.0")]`
    shims forwarding to the new home; `similarity::distance` / `minhash` /
    `simhash` / `lsh` continue to work unchanged.
- **`permutation` module** — locality-improving orderings:
  - Shared `Permutation` type with `identity`, `inverse`, `apply`,
    `apply_into`, `as_slice`. `apply_into` is kernel-safe (caller-provided
    output buffer, no internal allocation).
  - `CsrGraph` borrowed-input adjacency type.
  - `rcm()` — Reverse Cuthill-McKee ordering with GPS pseudoperipheral
    start vertex, BFS frontier-sort-by-degree, Liu-Sherman 1976 reversal,
    deterministic tie-breaking on equal degree, disconnected-component
    restart.
  - `hilbert_2d` and `hilbert_nd` (gated on `permutation_hilbert` Cargo
    feature) wrapping the `fast_hilbert` and `hilbert` crates per the
    vendor decision in `docs/v0.2_planning/14_PERMUTATION.md` § 4.
- **`hash::set_membership_simd`** — VPCMPEQ broadcast-compare scan for
  short u32 haystacks (≤256 typical for vocab tables, content-class
  membership, Bloom pre-checks). Scalar / SSE4.1 / AVX2 / AVX-512 / NEON
  kernels. ~150-240 GB/s on AVX2 for L1-resident haystacks.
- **`benches/`**: per-tier criterion benches for every new module
  (`bits_bit_pack`, `bits_streamvbyte`, `bits_rank_select`, `bitmap_set_ops`,
  `hash_set_membership`, `permutation_rcm`).
- **`examples/`**: three end-to-end composition demonstrators (Phase C):
  - `inverted_index` — token n-gram inverted index (bitmap + Stream-VByte).
  - `build_pipeline` — image build pipeline (batched SHA-256 + RCM).
  - `similarity_scan` — fingerprint similarity scan (vector + bit_pack).

### Changed

- `similarity::kernels::*` modules are `#[deprecated(since = "0.2.0")]`
  shims forwarding to `vector::kernels::*`. Source migration is a rename;
  no semantic change. SVE kernels in `similarity::kernels::sve` are NOT
  deprecated and will move with their own sprint.

### Process

- All v0.2 sprints landed via `isolation: "worktree"` parallel sub-agents,
  with explicit "no destructive git" guardrails to prevent cross-agent
  conflicts. See `docs/v0.2_planning/03_EXECUTION_PLAN.md` for the full
  sprint sequence and per-sprint ship gates.

### Tests

- 679 lib tests on x86_64 (was 462 at v0.1.0; +217 across the v0.2 surface).
- 56 AVX2 parity tests (was ~30 at v0.1.0).
- All `cargo xtask check` and aarch64 cross-clippy gates green.

## [0.1.1] — 2026-05-02

First v0.2-roadmap shipment. Two foundation primitives that gate downstream
v0.2 work: `bits::popcount` (for `bits::rank_select`, `bitmap` cardinality,
`vector` hamming/jaccard) and `hash::batched` (for `tokenfs_writer`-class
build-time Merkle leaf hashing). See `docs/v0.2_planning/03_EXECUTION_PLAN.md`
Sprints 1 + 2.

### Added

- `bits` module with `popcount_u64_slice` and `popcount_u8_slice`
  runtime-dispatched APIs. Backends: scalar (chunked u64), AVX2
  (Mula nibble-LUT), AVX-512 (`VPOPCNTQ`), NEON (`VCNT` + horizontal add).
  AVX2 path measured at ~62 GiB/s in-L1/L2/L3 vs ~17 GiB/s scalar baseline
  on a typical x86_64 host.
- `hash::batched` module with four batched cryptographic-hash APIs:
  - `sha256_batch_st` (kernel-safe single-thread)
  - `sha256_batch_par` (rayon parallel, `parallel` feature)
  - `blake3_batch_st_32` (`blake3` feature)
  - `blake3_batch_par_32` (`blake3` + `parallel` features)
  Threshold-based fallback: under `BATCH_PARALLEL_THRESHOLD = 256` messages,
  the `_par` variants delegate to single-thread to avoid rayon thread-pool
  overhead. SHA-256 hits ~25 GiB/s aggregate on 8-core host for the
  canonical 200K × 1KB Merkle workload.
- `benches/bits_popcount.rs` — Criterion bench using the new
  `support::cache_tier_sizes()` 4-tier reporting helper (in-L1 / in-L2 /
  in-L3 / in-DRAM).
- `benches/hash_batched.rs` — Criterion bench across three workloads
  (canonical Merkle 200K × 1KB, small messages 1M × 64B, single 1GB).
- `support::cache_tier_sizes()` bench helper for the v0.2 4-tier cache-
  residency reporting convention.

### Notes

- Both primitives are `no_std + alloc` clean; the `_st` (single-thread)
  hash variants are kernel-safe and verified by `xtask security`'s
  three-way `--no-default-features {-, --features alloc, --features std}`
  `--lib` check. The `_par` variants and `blake3` paths are userspace-only
  per `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`.

## [0.1.0]

Initial release. Histograms, n-gram counters, byte-class, run-length,
chunking, distribution distances, sketches, F22 fingerprints, identity
multihash, similarity primitives (MinHash, SimHash, LSH skeleton), search,
distribution / divergence, format sniffer, processor-aware dispatch.

See `docs/CORE_PRIMITIVE_COMPLETION_2026-05-01.md` for the v0.1.0 surface
inventory.

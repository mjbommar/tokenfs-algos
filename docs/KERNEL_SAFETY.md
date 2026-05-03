# Kernel-Safe-By-Default

> **Load-bearing invariant.** Maintain it.

`tokenfs-algos` exists to be reachable from kernel modules, FUSE
daemons, Postgres extensions, MinIO/CDN edge code, and forensics
tooling — environments where a panic is a kernel oops, a broken
filesystem mount, a hung query backend, or a data-loss event.

This document defines the contract that keeps the crate safe to
embed in those environments. It is the synthesis of audit rounds
R4 through R10 (see `CHANGELOG.md` for per-round detail).

## Feature Shapes

The kernel-safe surface is reached by:

```toml
tokenfs-algos = { default-features = false, features = ["alloc"] }
```

Feature flags layer up:

| Feature | What it adds | Kernel-safe? |
|---|---|---|
| `alloc` | `extern crate alloc;` | ✅ yes — the kernel-safe baseline |
| `std` | `extern crate std;`; OS probes (`std::thread::available_parallelism`, `/sys/.../cache`); `std::error::Error` impls | ⚠️ usually no |
| `userspace` | Panicking-shape APIs reachable: every `pub fn` whose body asserts on caller input is gated on this | ❌ no |
| `panicking-shape-apis` | Legacy alias retained for compatibility; same effect as `userspace` for panic-gating purposes | ❌ no |
| `arch-pinned-kernels` | Per-backend kernel modules (`scalar`, `avx2`, `avx512`, `neon`, `sse41`, `ssse3`, `x86_shani`, `aarch64_sha2`) become `pub mod` instead of `pub(crate) mod` | ⚠️ exposes `pub unsafe fn` surface; intended for benches / consumer perf comparisons |
| `parallel` | Pulls in `rayon`; `*_par` siblings light up | ❌ no — rayon is forbidden in the kernel-safe dep tree |
| `blake3` | Pulls in `blake3` crate; `blake3` hash entries light up | ❌ no — blake3 is forbidden in the kernel-safe dep tree |
| `bench-internals` | Exposes private metrics for benchmarking | ❌ no — gates `dispatch::force_backend` etc |

The default feature set is `["std", "avx2", "neon"]` for ergonomic
test/bench/example builds. **`userspace` is not in the default set.**
That is the architectural decision behind audit-R9 #4 + the v0.4
surface flip: kernel-default builds reach `try_*` siblings; userspace
opt-in unlocks the `expect`-style ergonomic surface.

`cargo xtask security` enforces this by checking the no_std dependency
tree against a forbidden list (`blake3`, `criterion`, `proptest`,
`rayon`, `serde_json`).

## The `try_*` / `_unchecked` Convention

For every public API that has a caller-supplied precondition (length
match, in-range index, non-zero divisor, finite bit-length) one of the
following must be true:

### Pattern A — `try_*` sibling

The fallible version is the kernel-safe surface, returns `Result<T, E>`,
and never panics on caller input. The panicking version is a thin
`#[cfg(feature = "userspace")]`-gated wrapper:

```rust
// Always available (kernel-safe).
pub fn try_sha256(bytes: &[u8]) -> Result<[u8; 32], Sha256LengthOverflow> {
    if (bytes.len() as u64).checked_mul(8).is_none() {
        return Err(Sha256LengthOverflow { /* ... */ });
    }
    Ok(kernels::auto::sha256(bytes))
}

// Only when userspace is enabled.
#[cfg(feature = "userspace")]
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    try_sha256(bytes).expect("sha256: input length exceeds 2^64 bits")
}
```

Examples in the codebase: `sha256` / `try_sha256`,
`PackedDfa::new` / `try_new`, `Permutation::identity` / `try_identity`,
`BloomFilter::new` / `try_new`, `HyperLogLog::new` / `try_new`.

### Pattern B — `_unchecked` sibling on backend kernels

For runtime-dispatched SIMD kernels, the runtime dispatcher
(`kernels::auto::*`) needs to call into per-backend kernels without
inheriting the asserts that the userspace-gated `auto::*` performs.

Each backend kernel exposes:
- `pub (unsafe) fn <name>` — the asserting version, gated on
  `#[cfg(feature = "userspace")]`. Reference oracle for the SIMD
  parity tests.
- `pub (unsafe) fn <name>_unchecked` — same body without asserts.
  Caller must uphold the precondition (documented in the `# Safety`
  section).

The runtime dispatcher has matching pairs:
- `auto::<name>` — userspace-gated; asserts then calls `_unchecked`.
- `auto::<name>_unchecked` — always available; runtime feature
  detection then calls a per-backend `_unchecked` after the SIMD
  availability check.

The fallible top-level entry (`try_*`) validates upfront, then calls
`auto::*_unchecked`. Kernel-default builds reach `auto::*_unchecked`
without ever entering an asserting code path.

Examples: every kernel under
`bits/{streamvbyte,bit_pack,rank_select}/kernels/`,
`approx/bloom_kernels/`,
`hash/set_membership/kernels/`,
`permutation/rabbit/kernels/`.

### Pattern C — `_inner` helper for shared bodies

When the algorithm body is too large to duplicate between the
panicking entry and the fallible sibling, extract a private `_inner`
helper that takes a pre-validated input. The userspace-gated entry
asserts then calls `_inner`; the `try_*` sibling validates then calls
`_inner`.

```rust
#[cfg(feature = "userspace")]
pub fn rabbit_order(graph: CsrGraph<'_>) -> Permutation {
    /* asserts on graph.offsets shape + monotonicity */
    rabbit_order_inner(graph)
}

pub fn try_rabbit_order(graph: CsrGraph<'_>) -> Result<Permutation, _> {
    graph.try_validate()?;
    Ok(rabbit_order_inner(graph))
}

// Private — no assertions; caller upholds the validation invariant.
fn rabbit_order_inner(graph: CsrGraph<'_>) -> Permutation { /* ... */ }
```

Codebase examples: `permutation::rabbit::rabbit_order_inner`,
`permutation::rabbit::rabbit_order_par_inner`,
`permutation::rcm::rcm_inner`,
`permutation::rabbit::kernels::scalar::modularity_gains_neighbor_batch_inner`,
`hash::set_membership::kernels::scalar::contains_u32_batch_inner`.

## The Lint

`cargo xtask panic-surface-lint` walks every `.rs` file under
`crates/tokenfs-algos/src/` and reports every `pub fn` /
`pub unsafe fn` / `pub(crate) fn` whose body contains
`assert!`, `panic!`, `assert_eq!`, `assert_ne!`,
`unreachable!`, `unimplemented!`, or `todo!` without an enclosing
`#[cfg(feature = "userspace")]` or `#[cfg(test)]` gate.

Single forward-pass, brace-depth + cfg-stack tracking, no syn /
proc-macro toolchain required. Wired into:

- `cargo xtask panic-surface-lint` — standalone task.
- `cargo xtask security` — runs as part of the security suite.
- `cargo xtask check` — the canonical "is the codebase healthy?" gate.
- `.github/workflows/ci.yml` — fails per-PR.

## The Allowlist

`tools/xtask/panic_surface_allowlist.txt` is the load-bearing artifact
for the lint. **It contains zero entries by policy.**

The file format is `<path>:<decl_line>:<fn_name>:<macro!>` with
`#`-prefixed comments. The lint reads it, suppresses any matching
violation, and warns about stale entries (allowlisted but no longer
matching).

When the lint catches you, **the correct response is almost never
"add an entry to the allowlist."** The correct responses, in order:

1. Add a `try_*` sibling and gate the panicking version on
   `#[cfg(feature = "userspace")]`.
2. Add a `_unchecked` sibling and route the runtime dispatcher
   through it.
3. Extract an `_inner` helper shared between the panicking and
   fallible paths.
4. Replace the assert with an algorithmic guard (e.g. saturate to
   the type max, return `Permutation::try_identity(0)` on empty
   input, etc.) — see `permutation::hilbert::hilbert_2d` for the
   canonical example of this approach.

If none of those work — e.g. the panic guards a const-generic
invariant the type system doesn't yet enforce — discuss the
addition with a maintainer before adding the allowlist entry, and
include the rationale in the file as a `# `-prefixed comment.

The lint also surfaces dead allowlist entries — entries that match
nothing. These should be cleaned up promptly.

## Audit Lineage

| Round | Focus | Key closeouts |
|---|---|---|
| R4 | First external audit | `try_*` coverage scaffolding (#151), hilbert dep risk (#150), bitmap container invariants (#153), no_std test build (#154) |
| R5 | Kernel-safe API gaps | streamvbyte no_std (#155), kernels_gather stack hazard (#156), panicking entry points (#157), hilbert dep replacement (#158), bitmap raw-field validation (#159), Permutation::from_vec_unchecked (#160) |
| R6 | Permutation correctness | apply_into stale-slot leak (#161), no-panic facade for streamvbyte/bit_pack (#162), RankSelectDict u32 truncation (#163), no-std + Miri test coverage (#164) |
| R7 | Phase D + arch-pinned-kernels | rank_select select_in_word + bit_pack _unchecked (#170), arch-pinned-kernels feature wiring (#180) |
| R8 | Audit follow-up | chunk recursive + config validation (#181), try_* still-panicky paths (#182), stack hazards in entropy + similarity (#183) |
| R9 | Default feature flip | gate panicking constructors (#187), actually-no-panic try_rank1_batch / try_select1_batch (#188), gate by-value MinHash + entropy (#189), TLSH canonical pearson_table (#190), drop std from default — partial (#191) |
| R10 | Comprehensive panic surface | Tier 0: CI + fuzz + bench fixes. Tier 1: HIGH-severity panic gating + PackedDfa::try_new + ssse3/sse41 backends + panic-surface lint. Tier 2: try_sha256 + saturating counters + bench regression CI + scheduled calibration + no_std smoke. Tier 3: Miri/ASan/MSan + coverage + sanitizers + scheduled fuzz + mutation testing + OSS-Fuzz. Tier 2 carry-over (T2.6 / #216): systematic gating sweep — allowlist 38 → 0 across v0.4.5 / v0.4.6. |

## See Also

- `AGENTS.md` — agent-specific guidance, including the kernel-safe
  contract section that mirrors this doc.
- `CHANGELOG.md` — per-release diff with audit lineage.
- `docs/PRIMITIVE_CONTRACTS.md` — the per-primitive contract (pure
  `&[u8]`, scalar baseline, pinned kernels, etc.).
- `docs/NO_STD_POSTURE.md` — the no_std + alloc story.
- `docs/PROCESSOR_AWARE_DISPATCH.md` — how `kernels::auto::*` works.
- `tools/xtask/panic_surface_allowlist.txt` — the (empty by policy)
  load-bearing artifact.

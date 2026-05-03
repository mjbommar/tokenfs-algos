# `no_std` posture

Verified by CI on every push (`.github/workflows/ci.yml`) and by
`cargo xtask check` locally (the `security` step). Since v0.4.5
(audit-R10 T2.5 / #10), `cargo xtask security` also builds the
`tokenfs-algos-no-std-smoke` crate which exercises the kernel-safe
surface through a real `#![no_std]` consumer — if any kernel-claimed-
safe primitive moves out from under `--no-default-features --features alloc`,
that build fails.

> See also: [`KERNEL_SAFETY.md`](KERNEL_SAFETY.md) for the
> kernel-safe-by-default contract that overlays this posture
> (the `try_*` / `_unchecked` / `_inner` conventions and the
> panic-surface lint).

| Configuration | Lib `cargo check --lib` | Lib `cargo build --lib` | Tests | Examples / Benches |
|---|---|---|---|---|
| `--no-default-features` | ✅ | ✅ | ❌ (intentional) | ❌ (intentional) |
| `--no-default-features --features alloc` | ✅ | ✅ | ❌ (intentional) | ❌ (intentional) |
| `--no-default-features --features std` (= same as above for our purposes) | ✅ | ✅ | ✅ | ✅ |
| `--all-features` | ✅ | ✅ | ✅ | ✅ |

## Why tests / examples / benches are gated on `std`

The library code is rigorously `no_std + alloc`-clean — every primitive
(histogram, fingerprint, byteclass, sketch, search, similarity,
identity, format, hash families) compiles cleanly without `std`.

Tests, examples, and benches deliberately are **not** rebuilt under
`no_std`. Three concrete reasons:

1. **`vec!` / `format!` macros** are std-only macros. Test fixtures
   use them pervasively (>200 call sites). Replacing each with
   `alloc::vec![…]` / `alloc::format!(…)` would clutter the test
   surface for zero downstream benefit — no caller runs tests inside
   a kernel.
2. **Dev-dep crates** (`proptest`, `criterion`, `serde_json`,
   `hex-literal`) are not no_std-friendly themselves. Any test or
   bench that uses one of them inherits std.
3. **Verification value is in lib coverage, not test coverage.** A
   kernel-adjacent caller links against the library, never against
   `cargo test`. The lib check posture above is exactly what
   matters.

If you need to verify a specific primitive's behaviour under
`no_std + alloc`, add a tiny standalone test crate (e.g. under
`crates/tokenfs-algos-no-std-smoke/`) with `default-features = false`
on the dependency line and a single `#[no_std]` `lib.rs`. The test
crates that exist today are intentionally `std`-flavored.

## What "no_std-clean" means in this crate

- The lib uses `extern crate alloc;` only when the `alloc` cargo
  feature is on AND `std` is off. Library types like `Vec`, `String`,
  `Box` are imported via `alloc::*` rather than `std::*` so the same
  code compiles in both feature configurations.
- Every public function that allocates is gated on
  `#[cfg(any(feature = "std", feature = "alloc"))]`.
- Float ops that aren't in `core` (sqrt, ln, cos, etc.) route through
  `crate::math::*` which falls back to `libm` when `std` is off.
- Hardware intrinsics that need runtime detection (SIMD, SHA, CRC) are
  cfg-gated on `feature = "std"` because `is_*_feature_detected!`
  requires std. Without std the dispatch routes through scalar.

## Audit findings related to this posture

- **#81** (audit-round-3): "no_std library builds, but no_std tests do
  not." Resolved by codifying the matrix above + adding the
  `--features alloc` lib check to CI. The discrepancy is intentional
  and documented; the lib coverage CI runs are the contract.

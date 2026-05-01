# Paper Primitive Migration

Date: 2026-04-30.

This crate should not become detached from the TokenFS paper results. The
histogram benchmark matrix tells us raw kernel throughput; F21/F22/F23a tell us
whether the primitives preserve the empirical behavior used in the paper.

Naming follows [Paper Lineage Naming](PAPER_LINEAGE_NAMING.md): `F21`, `F22`,
and `F23` remain paper/calibration labels, while the crate APIs use product names
such as `selector`, `fingerprint`, and `sketch`.
Primitive implementation rules follow [Primitive Contracts](PRIMITIVE_CONTRACTS.md).

## Current Source

The F22 implementation currently lives at:

```text
../tokenfs-paper/tools/rust/entropy_primitives/
```

Important pieces:

| File | Contents |
|---|---|
| `src/lib.rs` | `BlockFingerprint`, `ExtentFingerprint`, `fingerprint_block`, `fingerprint_extent`. |
| `src/scalar.rs` | Scalar byte histogram, nibble histogram, CRC32C 4-gram bins, run-length, top-K, byte-class, entropy LUT. |
| `src/dispatch.rs` | Runtime scalar/AVX2 dispatch for F22 primitives. |
| `src/avx2.rs` | AVX2 implementation for F22 block and hash-bin work. |
| `tests/calibration.rs` | F21 sidecar calibration test. |

## Migration Order

1. Done: move scalar F22 primitives into `tokenfs-algos::fingerprint` behind
   stable module names.
2. Done: add parity tests against copied scalar fixtures and deterministic generated
   extents.
3. Done: add `paper::f22` compatibility aliases and pinned
   `fingerprint::kernels::scalar` paths.
4. Done: add optional F21/F22 sidecar calibration tests that skip when data is
   unavailable.
5. Done: move the F22 AVX2/SSE4.2 fused block path after scalar parity is stable.
6. Add planner-oracle benchmark rows: for each F21 extent, compare the planner
   selection against the F21/F22 oracle outcome.

## Ground Truth Data

The current F22 calibration sidecar path in `tokenfs-paper` is:

```text
/nas4/data/tokenfs-ubuntu/bench/cow/f22-extent-bytes.bin
```

The requested F21 parquet fixture is the preferred v0.1 bench-real ground truth
when available because it is small enough to keep iteration fast while still
representing real corpus extents. It should be used to answer a different
question than the Ubuntu ISO:

| Input | Question |
|---|---|
| Ubuntu ISO slices | How fast are kernels on large binary regions? |
| F21 parquet extents | Does the planner choose the same kernel/feature path as the paper oracle on real extents? |

The benchmark harness now accepts:

- `TOKENFS_ALGOS_F21_DATA`
- `TOKENFS_ALGOS_F22_DATA`
- `TOKENFS_ALGOS_PAPER_ROOT`
- `cargo xtask bench-real-f21 [path]`

When those files are missing, the suite reports the missing path instead of
silently fabricating paper calibration data.

## v0.1 vs v0.2

For v0.1, keep dispatch static and transparent:

- scalar always available;
- AVX2 selected by runtime feature detection for implemented x86 kernels;
- NEON/AVX-512/SVE/SVE2 are reported as scalar fallback until parity-tested
  kernels exist on those backends;
- planner rules are documented and inspectable;
- calibration tests are opt-in when real sidecar/parquet data exists.

Defer persistent autotuning cache and generated dispatch tables to v0.2. The
kernel set needs to stabilize before a cache-backed planner can be trusted.

# Paper Lineage Naming

Date: 2026-04-30.

The `F21`, `F22`, and `F23` labels are paper lineage names. They are useful for
reproducibility, calibration, and historical traceability, but they should not be
the primary product or crate API names. Public APIs should use descriptive names
that make sense to TokenFS, FUSE, kernel-adjacent code, and Python users who have
not read the paper.

## Naming Rule

- Use descriptive module names for normal crate APIs.
- Keep paper labels in docs, benchmark suite names, fixture names, and explicit
  compatibility aliases.
- When a paper primitive is productized, document both its paper name and its
  project name.

## Canonical Mapping

| Paper label | Project name | Main Rust surface | Meaning |
|---|---|---|---|
| `F21` | extent selector / representation selector | `selector`, later `content_selector` if it becomes a downstream crate | Uses extent features to predict the best representation, compression, or storage path. |
| `F22` | fingerprint / content fingerprint | `fingerprint::block`, `fingerprint::extent`, `BlockFingerprint`, `ExtentFingerprint` | Extracts compact per-block and per-extent byte features. |
| `F23a` | sketch primitives | `sketch::MisraGries`, `sketch::crc32_hash4_bins`, entropy LUT APIs | Low-memory approximate counters, heavy hitters, hash-bin counts, and entropy reductions. |
| `F23b` | conditional encoder dispatch | Downstream encoder integration, probably outside `tokenfs-algos` | Uses fingerprints and sketches to choose or skip encoder work. |

## Public API Shape

Normal user code should prefer:

```rust
let fp = tokenfs_algos::fingerprint::block(bytes);
let extent_fp = tokenfs_algos::fingerprint::extent(bytes);
let plan = tokenfs_algos::histogram::plan_block(bytes);
```

Avoid making paper labels the ergonomic path:

```rust
// Avoid as the primary API.
let fp = tokenfs_algos::f22::fingerprint_block(bytes);
```

If paper-compatibility aliases are useful, put them behind an explicit namespace:

```rust
tokenfs_algos::paper::f22::fingerprint_block(bytes);
tokenfs_algos::paper::f21::selector_features(extent);
tokenfs_algos::paper::f23a::crc32_hash_bins(bytes);
```

That keeps the default surface readable while preserving exact paper lineage for
calibration, academic replication, and forensic benchmark runs.

## Benchmark And Fixture Names

Use paper labels when the benchmark is answering a paper-reproducibility
question:

- `bench-real-f21`
- `bench-f22-calibration`
- `bench-f23a-sketch`
- `bench-planner-oracle`

Use product names when the benchmark is a general library or consumer benchmark:

- `bench-fingerprint`
- `bench-sketch`
- `bench-selector`
- `bench-dispatch`
- `bench-thread-topology`

## Documentation Convention

When introducing a paper-derived primitive, write it as:

```text
F22 fingerprint
```

on first use, then use the project name after that:

```text
fingerprint
```

When the distinction matters, spell it out:

```text
The fingerprint API is the productized F22 block/extent primitive.
```

This avoids hiding the research lineage while keeping the crate language stable
for downstream users.

## Current Working Vocabulary

- `F21` -> selector
- `F22` -> fingerprint
- `F23a` -> sketch
- `F23b` -> conditional dispatch

The near-term implementation order should use those names:

1. Finish `fingerprint` calibration against F22 fixtures.
2. Add `selector` fixtures from F21 parquet/sidecar data.
3. Expand `sketch` with the F23a counters and entropy reductions.
4. Leave `conditional dispatch` as downstream integration until the primitive
   APIs are stable.

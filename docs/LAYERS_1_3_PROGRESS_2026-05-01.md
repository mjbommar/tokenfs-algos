# Layers 1-3 Progress

Date: 2026-05-01.

This note tracks the bottom of the public-contract ladder: tiny bricks,
trusted scalar references, and pinned kernels.

## Layer 1: Tiny Bricks

New bricks added in this pass:

| Brick | Public Surface | Notes |
|---|---|---|
| Dense byte-pair histogram | `histogram::BytePairHistogram` | Exact adjacent-pair counts in fixed 65,536-bin storage; no heap allocation. |
| Reusable byte-pair scratch | `histogram::BytePairScratch` | Caller-owned exact pair/predecessor counts with lazy reset for repeated hot calls. |
| Joint byte-pair entropy | `entropy::joint::h2_pairs` | Dense exact H2 path for hot/reference use without sparse maps. |
| Conditional entropy | `entropy::conditional::h_next_given_prev` | Computes `H(X_next | X_prev)` from dense pair/predecessor counts. |
| Min-entropy | `entropy::min::h1` | Byte min-entropy from `ByteHistogram`. |
| Renyi/collision entropy | `entropy::renyi::{h1_alpha, collision_h1}` | Scalar reference path with `libm` support in no-std builds. |
| UTF-8 validation | `byteclass::validate_utf8` | Scalar reference returns validity, valid prefix, and error length. |
| Hash families | `hash::{fnv1a64, mix64}` | Stable non-cryptographic scalar hashes for sketching/bucketing. |
| Normalized FastCDC | `chunk::{fastcdc_chunks, fastcdc_find_boundary}` | Separate normalized CDC path alongside the simpler Gear chunker. |
| Chunk quality | `chunk::summarize_chunk_quality` | Reports chunk count, min/max/mean length, and boundary-limit violations. |

The existing sparse exact n-gram histogram remains useful for H2..H8
calibration/research. The dense byte-pair path gives us a no-allocation exact H2
alternative, and reusable scratch avoids clearing the full pair table on
repeated kernel-adjacent and FUSE-style calls.

## Layer 2: Trusted Reference

Scalar/reference coverage added:

- Known values for min entropy, collision entropy, adjacent-pair entropy, and
  FNV-1a.
- Public/default-vs-pinned scalar parity tests for entropy kernels, UTF-8
  validation, hash kernels, and chunk boundary finders.
- Fingerprint block parity and small-extent parity between the public/default
  path and pinned scalar reference.
- `no_std` math wrapper backed by `libm`, used by entropy, divergence, sketch,
  and fingerprint quantization.

The following now pass:

```bash
cargo check -p tokenfs-algos --no-default-features
cargo check -p tokenfs-algos --no-default-features --features alloc
```

This does not mean every future API is kernel-ready, but the current crate
surface no longer assumes `std` for floating-point math.

## Layer 3: Pinned Kernels

Pinned/default shape added or clarified:

| Family | Default | Pinned Scalar |
|---|---|---|
| Entropy | `entropy::kernels::auto::*` | `entropy::kernels::scalar::*` |
| Fingerprint | `fingerprint::{block,extent}` | `fingerprint::kernels::scalar::{block,extent}` |
| Byteclass UTF-8 | `byteclass::validate_utf8` | `byteclass::kernels::scalar::validate_utf8` |
| Hash | `hash::{fnv1a64,mix64}` | `hash::kernels::scalar::*` |
| Chunking | `chunk::{chunks,fastcdc_chunks}` | `chunk::kernels::{gear,fastcdc}` |

For fingerprint extents, the pinned scalar path is the exact reference. The
public/default path is exact for H1, run-length, top-16 coverage, and skew, and
uses exact H4 up to 64 KiB before switching to a sampled H4 estimator. That is
the first deliberately documented tolerance in the public default path; the
current regression bound is 2.5 bits on a periodic-text fixture.

New primitive benchmark labels:

```text
byteclass-utf8-fullscan
byteclass-utf8-reject-latency
entropy-min-h1
entropy-collision-h1
entropy-joint-h2-dense
entropy-conditional-next-prev
hash-fnv1a64
hash-mix64
chunk-gear
chunk-fastcdc
chunk-gear-quality
chunk-fastcdc-quality
```

## Remaining Layer 1-3 Gaps

- Exact dense H3+ is intentionally not implemented because full dense state
  explodes; high-cardinality H3+ should use sketches or offline sparse maps.
- UTF-8 validation is scalar only; SIMD validation should follow a known
  simdutf/simdjson-style design with strict parity tests.
- Hash family work is deliberately conservative. Complete wyhash/rapidhash
  ports should be added only with upstream test vectors.
- Normalized FastCDC has bounds/coverage tests now; it still needs larger
  distribution tests over real files and synthetic entropy classes.
- NEON/AVX-512/SVE remain scalar fallback backends until tested kernels exist.

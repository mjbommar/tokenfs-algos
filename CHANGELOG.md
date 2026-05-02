# Changelog

All notable changes to this crate will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

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

# Changelog

All notable changes to this crate will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

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

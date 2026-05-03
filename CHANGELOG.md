# Changelog

All notable changes to this crate will be documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning
follows [Semantic Versioning](https://semver.org/).

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

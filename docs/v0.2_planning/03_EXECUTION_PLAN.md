# Execution plan — full v0.1.x → v0.2 → v0.2.x → Phase D

**Status:** concrete sprint-level execution plan, 2026-05-02. Ships everything in `01_PHASES.md` plus the v0.2.x next-up candidates from `20_DEFERRED.md` plus Phase D Rabbit Order.

This doc is sprint-granular. Each sprint has a scope, ship gate (verifiable commands), task list ready to feed into `TaskCreate`, and risk notes. Sprint **boundaries are dependency-driven, not calendar-driven**; the rate is the contributor's choice.

## Total scope and honest budget

| Milestone | Scope | Solo budget | Cumulative |
|---|---|---|---|
| **Sprint 0** | Pre-flight (CI, bench harness extension) | 1-2 days | 2 days |
| **v0.1.x release** | A1 popcount + A3 batched hash | ~1 week | ~9 days |
| **v0.2 Phase A continuation** | A2 + A4 + A5 | ~2 weeks | ~3 weeks |
| **v0.2 Phase B** | B5 + B4 + B2 + B3 + B1 (in that order) | ~5-6 weeks | ~9 weeks |
| **v0.2 Phase C** | C1 + C2 + C3 demonstrators | ~1 week | ~10 weeks |
| **v0.2 release** | tag, changelog, docs sweep | 2-3 days | ~10.5 weeks |
| **v0.2.x candidates** (contingent) | Bloom + HLL merge + MinHash | ~2 weeks | ~12.5 weeks |
| **Phase D Rabbit Order** | D1 | ~3-4 weeks | ~16 weeks |
| **Total v0.1.x → end of D** | | **~14-16 weeks solo** | |

With two contributors on parallel tracks: ~half. The two-contributor split is in `01_PHASES.md`; this doc walks the solo sequence.

## Branching strategy

- All work lands on `main` via PRs.
- v0.1.x patch releases: tag `v0.1.1`, `v0.1.2`, `v0.1.3` as A1 / A3 / batched-API-revision land. Lightweight tags, regenerated changelog.
- v0.2 release: cut from `main` after Phase C demonstrators land + bench numbers settle. Tag `v0.2.0`. Major release; version bump + changelog migration.
- v0.2.x patches (Bloom/HLL/MinHash): `v0.2.1`, `v0.2.2`, `v0.2.3` per-feature, gated on documented consumer signal.
- Phase D Rabbit Order: long-running feature branch `feature/rabbit-order` because of multi-week scope and concurrent-data-structure complexity. Merge to main as `v0.3.0` once parity-validated.

No release branches; main + tags is sufficient at this scale.

---

## Sprint 0 — Pre-flight (1-2 days)

**Scope:** infrastructure to support multi-tier benches and stricter kernel-safety verification. Land before any new primitive work.

### Tasks

1. **Bench harness 4-tier reporting**: extend `criterion` benches to report per-primitive numbers at in-L1 / in-L2 / in-L3 / in-DRAM working-set sizes. Per `02_CACHE_RESIDENCY.md` § 8 convention. Implement as sized inputs threaded through `support.rs`.
2. **Extend `xtask security`** to verify:
   - `cargo check -p tokenfs-algos --no-default-features --features alloc --lib` (already in place)
   - **NEW**: `cargo check -p tokenfs-algos --no-default-features --features std --lib` (verifies the no-rayon, no-blake3 std path that kernel-safe `_st` variants will live in).
   - **NEW**: assertion that no public `pub fn` in `lib.rs` reaches for `rayon::*` outside `cfg(feature = "parallel")` blocks. Either via `cargo expand` parsing or a regex grep in xtask.
3. **CI matrix sanity**: confirm `Security` workflow is now passing on PRs (was the dependabot blocker). Add `xtask security` invocation to `Security` workflow's `kernel core guard` job (replace inline cargo-check sequence — `xtask security` already does it).

### Ship gate

- `cargo xtask check` passes (includes new std-without-parallel check).
- Benches produce 4-tier output rows in their JSON.
- CI Security workflow green on a sentinel PR.

### Risk

Low. Pure tooling; no algorithmic work.

---

## v0.1.x ship — Sprint 1 + 2 (~1 week)

### Sprint 1 — A1 popcount (1-2 days)

**Scope:** `bits::popcount` foundation kernel. See `10_BITS.md` § 4.

**Tasks:**
1. Create `crates/tokenfs-algos/src/bits/mod.rs` with module skeleton.
2. `bits::popcount::scalar::popcount_u64_slice` + `popcount_u8_slice` — `u64::count_ones` per word, sum.
3. `bits::popcount::kernels::avx2` — Mula nibble-LUT method (Mula/Kurz/Lemire 2018, Computer Journal). 256-byte unrolled.
4. `bits::popcount::kernels::avx512` — `_mm512_popcnt_epi64` (VPOPCNTQ) when feature available.
5. `bits::popcount::kernels::neon` — `vcntq_u8` per byte + horizontal `vaddvq_u8`.
6. `dispatch::*` registration for runtime backend selection.
7. Parity tests: `bits/tests.rs` with property-test against scalar oracle; SIMD parity bit-exact.
8. Bench: `benches/bits_popcount.rs` at L1/L2/L3/DRAM tiers.

**Ship gate (Sprint 1 → continue to Sprint 2):**
- `cargo xtask check` passes.
- `cargo test --features avx2 -- bits::popcount` green.
- AVX-512 self-hosted CI green on the new tests.
- NEON parity tests green via QEMU + via real macos-aarch64.
- Bench shows ≥5x AVX2 speedup, ≥10x AVX-512 speedup vs scalar at L2 working set.

**Risk:** very low. Mula's algorithm is well-known.

### Sprint 2 — A3 batched hash (2-3 days)

**Scope:** `hash::sha256_batch_st` (kernel-safe), `hash::blake3_batch_*` (userspace). See `12_HASH_BATCHED.md` § 2.

**Tasks:**
1. `hash::sha256_batch_st(messages: &[&[u8]], out: &mut [[u8; 32]])` — single-thread loop calling `sha2::Sha256::digest`. Kernel-safe path.
2. `hash::sha256_batch_par` — rayon parallel form. `#[cfg(feature = "parallel")]`.
3. `hash::blake3_batch_st_32` — single-thread loop calling `blake3::hash`. Userspace-only (`#[cfg(feature = "blake3")]`).
4. `hash::blake3_batch_par_32` — rayon parallel. `#[cfg(all(feature = "blake3", feature = "parallel"))]`.
5. Parity tests: batch result equals serial-iteration result.
6. Bench: `benches/hash_batched.rs` — 200K × 1KB messages (canonical Merkle workload), 1M × 64B (small), 1 × 1GB (large single).

**Ship gate (Sprint 2 → v0.1.x release):**
- `cargo xtask check` passes (verifies all `_st` paths build without `parallel` feature).
- Batch parity tests green.
- Bench shows: serial baseline ~80 ms for the canonical 200K × 1KB; rayon-parallel hits ~10-15 ms on 8-core machines.
- Document the SHA-NI dependency for kernel use (`sha2` crate's `cpufeatures` integration).

**Risk:** low. Existing crates (`sha2`, `blake3`) do the heavy lifting; we wrap.

### v0.1.x release

**Tag and ship as `v0.1.1`** (since `v0.1.0` is the implicit current state). Changelog entry: "Add `bits::popcount` SIMD kernels and `hash::*_batch_*` parallel hashing helpers."

Update `CHANGELOG.md` (create if absent). Push tag.

---

## v0.2 Phase A continuation — Sprints 3-9 (~2 weeks)

### Sprint 3-5 — A2 bit_pack (3-5 days)

**Scope:** `bits::bit_pack` arbitrary-width pack/unpack 1-32 bits. See `10_BITS.md` § 2.

**Tasks:**
1. `bits::bit_pack::scalar` — shift-and-OR pack/unpack for arbitrary W ∈ {1..32}.
2. `BitPacker<const W: u32>` const-generic specialization for compile-time-known widths.
3. `DynamicBitPacker { width: u32 }` runtime-width form.
4. AVX2 decode kernel (W ≤ 8 path: VPSHUFB nibble-shuffle; 8 < W ≤ 32 path: VPSRLVD per-lane shifts + mask).
5. AVX-512 decode kernel (VPMULTISHIFTQB for one-shot extraction).
6. NEON decode kernel (TBL byte permute + shifts).
7. Hard-code fast paths for W ∈ {8, 11, 12, 16} (canonical token widths).
8. Parity tests: round-trip every W ∈ {1..32}, lengths ∈ {0, 1, 7, 8, 33, 1024}; SIMD bit-exact vs scalar.
9. Bench: per W ∈ {1, 4, 8, 11, 12, 16, 32}, encode + decode, three sizes.

**Ship gate:** all parity tests green; bench shows ≥3x AVX2 speedup at common token widths (W=11, W=12).

**Risk:** medium. Const-generic Rust + multiple SIMD lane-width regimes is fiddly.

### Sprint 6 — A4 set_membership (1-2 days)

**Scope:** `hash::contains_u32_simd` and `contains_u32_batch_simd`. See `12_HASH_BATCHED.md` § 3.

**Tasks:**
1. `hash::contains_u32_simd(haystack, needle)` — VPCMPEQ broadcast + VPMOVMSKB scan.
2. AVX2 / AVX-512 / SSE4.1 / NEON kernels.
3. `hash::contains_u32_batch_simd(haystack, needles, out)` — broadcast each needle in turn; aggregate masks.
4. Parity tests; bench at haystack sizes 16/64/256/1024.

**Ship gate:** parity tests green; bench shows ≥10x SIMD vs `slice::contains` at haystack ≤ 256.

**Risk:** very low. Well-known kernel.

### Sprint 7-9 — A5 vector distance (5-7 days)

**Scope:** `vector` module — promote distance kernels from `similarity::kernels`. See `13_VECTOR.md`.

**Tasks:**
1. Create `vector` module skeleton.
2. Migrate existing scalar/AVX2/NEON kernels for f32/u32 dot/L2/cosine (rename + re-export with `#[deprecated]` shims on `similarity::kernels::*`).
3. **NEW**: AVX-512 FMA backends for f32 dot/L2/cosine.
4. **NEW**: `hamming_u64`, `jaccard_u64` for binary signatures.
5. **NEW**: batched many-vs-one APIs: `dot_f32_one_to_many`, `l2_squared_f32_one_to_many`, `cosine_similarity_f32_one_to_many`, `hamming_u64_one_to_many`, `jaccard_u64_one_to_many`.
6. Update existing `similarity::tests` for migrated paths.
7. New bench: AVX-512 row, batched many-vs-one row, hamming/jaccard rows.

**Ship gate:** all migration parity green (deprecated shims forward correctly); new AVX-512 backend matches scalar within Higham 1e-3 tolerance; bench shows AVX-512 FMA ~2x AVX2 on f32.

**Risk:** medium. Migration touches existing tests; careful with `#[deprecated]` warnings vs `-D warnings`.

---

## v0.2 Phase B — Sprints 10-30 (~5-6 weeks)

Sequenced **shortest first** to maintain momentum; longest items (B1 rank_select, B3 Roaring) come last.

### Sprint 10 — B5 Hilbert (1-2 days)

**Scope:** `permutation::hilbert` 2D + N-D. See `14_PERMUTATION.md` § 4.

**Tasks:**
1. Add `permutation_hilbert = ["dep:fast_hilbert", "dep:hilbert"]` feature flag.
2. `permutation::hilbert_2d(points: &[(f32, f32)]) -> Permutation` wrapper around `fast_hilbert`.
3. `permutation::hilbert_nd(points: &[Vec<f32>], dim: usize) -> Permutation` wrapper around `hilbert`.
4. `Permutation` type (shared across module): `pub struct Permutation(Vec<u32>)` with `identity`, `inverse`, `apply`, `apply_into`, `as_slice`.
5. Tests: parity vs `fast_hilbert` directly; locality property (points within bounding box have contiguous keys).

**Ship gate:** parity tests green; doc-test for `Permutation::apply` shows the expected use.

**Risk:** very low. Wrapping mature crates.

### Sprint 11-13 — B4 RCM (3-4 days)

**Scope:** `permutation::rcm` Reverse Cuthill-McKee. See `14_PERMUTATION.md` § 2.

**Tasks:**
1. `CsrGraph` borrowed-input type.
2. GPS pseudoperipheral start vertex (BFS-twice-for-deepest-then-from-deepest).
3. BFS with frontier-sort-by-degree.
4. Reverse the visit order → permutation array.
5. Parity tests: bandwidth-reduction property; `sprs::reverse_cuthill_mckee` parity on small graphs.
6. Bench: 10K, 100K, 1M vertex synthetic graphs.

**Ship gate:** parity vs `sprs` on small graphs; bandwidth strictly ≤ original on adversarial graphs; bench reports build cost and downstream BFS cache-miss reduction (perf events).

**Risk:** medium. The pseudoperipheral start-vertex algorithm has edge cases (disconnected graphs, isolated vertices).

### Sprint 14-19 — B2 Stream-VByte (4-6 days)

**Scope:** `bits::streamvbyte` Lemire codec. See `10_BITS.md` § 3.

**Tasks:**
1. Generate 256-entry shuffle table at build time (`build.rs` const fn).
2. Scalar encode + decode (Lemire spec, byte-by-byte).
3. SSSE3+SSE4.1 decode kernel (PSHUFB-based, Lemire C reference).
4. AVX2 decode kernel (dual-pumped PSHUFB).
5. NEON decode kernel (vqtbl1q_u8).
6. Encode kernels: per-int byte-length compute (`max(1, (32 - lzcnt) / 8)`) + pack 4 lengths to control byte + write data bytes.
7. Tail handling: pad final group with `00` codes; scalar tail decoder.
8. Parity tests: round-trip per N; bit-exact SIMD vs scalar.
9. Bench: encode + decode at 256, 1K, 16K, 256K, 4M elements.

**Ship gate:** round-trip property green; SIMD parity bit-exact; bench shows AVX2 decode ≥6 GB/s, NEON decode ≥3 GB/s.

**Risk:** medium-high. The 256-entry shuffle table construction has subtle indexing; any bug surfaces only on specific control-byte patterns.

### Sprint 20-29 — B3 Roaring SIMD (1-2 weeks)

**Scope:** `bitmap` module — three container types, SIMD set ops. See `11_BITMAP.md`.

**Tasks:**
1. Container types: `BitmapContainer`, `ArrayContainer`, `RunContainer`.
2. `Container` enum + dispatch.
3. **bitmap × bitmap kernels** (AND/OR/XOR/ANDNOT, AVX2 + AVX-512):
   - `_card`, `_nocard`, `_justcard` variants per op.
   - AVX-512 VPOPCNTQ for cardinality.
4. **array × array intersect** (Schlegel pcmpistrm + shuffle-table, SSE4.2):
   - 256-entry shuffle table (separate from Stream-VByte's, different format).
   - Galloping search fallback when one side is much smaller.
5. **array × array union/diff/xor** (SIMD merge-sort).
6. **array × bitmap intersect** (gather + bit-test, AVX2 PSHUFB output materialization; AVX-512 VPCOMPRESSD).
7. **run × * scalar** interval-merge implementations.
8. Cardinality kernels (popcount-of-AND for arrays; VPOPCNTQ for bitmaps).
9. Parity tests against `roaring-rs` scalar inner loops on every (pair, op) combination.
10. Bench: posting list pairs of 100, 10K, 100K, 1M elements.

**Ship gate:** every (pair, op) parity-tested vs `roaring-rs`; bench shows bitmap×bitmap AND ≥25 GB/s on AVX-512; array×array intersect ≥0.8 billion u16/sec.

**Risk:** high. Schlegel's algorithm has known SIMD-bug-prone edge cases. AVX-512 emulation for VP2INTERSECT (Diez-Canas) is non-trivial. Container-pair dispatch table is wide. Budget extra time for parity test debugging.

### Sprint 30-37 — B1 rank_select (1-2 weeks)

**Scope:** `bits::rank_select` plain bitvector dictionary. See `10_BITS.md` § 5.

**Tasks:**
1. `RankSelectDict<'a>` borrowed type.
2. `build` constructor: superblock counts (u32 per 4096 bits) + block counts (u16 per 256 bits).
3. `rank1(i)` query: superblock + block + popcount partial word.
4. `select1(k)` query: binary search through superblock counts + within-block scan via Vigna broadword select.
5. AVX-512 VPOPCNTQ-accelerated rank-batch.
6. AVX-512 parallel-scan select-batch.
7. NEON variants.
8. Parity vs `sucds` on test bitvectors.
9. Bench: build cost per million bits; per-query latency warm + cold; batch rank-K / select-K throughput.

**Ship gate:** parity vs `sucds` green on dense + sparse + alternating bitvectors; rank query ≤ 30 ns warm; select query ≤ 50 ns warm; AVX-512 batch rank ≥30 GB/s.

**Risk:** high. This is the longest single-piece work in v0.2; multiple variants (RRR, SDArray) waiting in the wings if requirements shift. Vigna broadword select is bit-twiddly.

---

## v0.2 Phase C — Sprints 38-40 (~1 week)

Composition demonstrators that exercise the Phase A+B primitives end-to-end on realistic data.

### Sprint 38 — C1 token n-gram inverted index demonstrator (~3 days)

**Composes:** B3 Roaring + B2 Stream-VByte.

**Tasks:**
1. `examples/inverted_index.rs`: build a token n-gram inverted index over a sample corpus (Ubuntu rootfs vocab dump, or synthetic).
2. Encode posting lists with Stream-VByte for delta-coded extent IDs; Roaring for membership-only postings.
3. Demonstrate boolean query: "extents containing tokens A AND B" as a Roaring intersection.
4. Bench: build time vs scalar baseline; query time per intersection.

### Sprint 39 — C2 image build pipeline benchmark (~2 days)

**Composes:** A3 batched hash + B4 RCM.

**Tasks:**
1. `examples/build_pipeline.rs`: simulate image build for 200K extents.
2. Compute SHA-256 batched over all extent payloads (Merkle leaves).
3. Build extent-similarity graph from F22 fingerprints.
4. Apply RCM to get inode/extent ordering.
5. Apply permutation to all per-extent metadata.
6. Bench: total wall time; peak memory.

### Sprint 40 — C3 similarity scan benchmark (~2 days)

**Composes:** A5 vector distance + A2 bit_pack (for fingerprint decode).

**Tasks:**
1. `examples/similarity_scan.rs`: load 200K F22 fingerprints (32B each).
2. Decode any bit-packed quantization via `bit_pack::decode`.
3. For a query fingerprint, compute distance to all 200K database fingerprints in batched form.
4. Sort top-K by distance (scalar; SIMD top-K is deferred).
5. Bench: per-query latency; throughput in queries/sec.

**Ship gate (Phase C):** all 3 demonstrators run from `cargo run --example`; produce JSON benchmark output; numbers feed into the v0.2 release notes.

---

## v0.2 release — Sprint 41 (~2-3 days)

**Tasks:**
1. CHANGELOG entry summarizing Phase A + B + C.
2. Documentation pass:
   - Update `PRIMITIVE_KERNEL_BUFFET.md` with the new modules (`bits`, `bitmap`, `vector`, `permutation`).
   - Update `PRIMITIVE_CONTRACTS.md` with examples of the new APIs (queue-pruning gate already added).
   - Update `PROCESSOR_AWARE_DISPATCH.md` with the new dispatch entries.
   - Cross-reference v0.2_planning docs from FS_PRIMITIVES_GAP.md (already done).
3. Bench numbers refresh: run full bench suite, commit results to `target/criterion` history (or a benches/ historical folder).
4. Tag `v0.2.0`, push.

**Ship gate:** all CI green; CHANGELOG complete; tag pushed.

---

## v0.2.x candidates — Sprints 42-46 (~2 weeks, contingent on consumer signal)

Each sprint contingent on documented consumer signal per `20_DEFERRED.md`'s v0.2.x candidates section. Order of preference: Bloom > HLL merge > MinHash by likely consumer count.

### Sprint 42-43 — Bloom SIMD (~3-5 days, IF Postgres/MinIO/CDN consumer asks)

**Tasks:**
1. Extend `approx::BloomFilter` with `kernels::{avx2, avx512, neon}`.
2. SIMD insert: VPMUL/VPSHUF for K hash positions in parallel.
3. SIMD query: K AND operations + reduce.
4. Batched query API: query N keys in parallel.
5. Parity vs scalar; bench.

**Ship as `v0.2.1`.**

### Sprint 44 — HLL merge SIMD (~2-4 days, IF Postgres/OLAP consumer asks)

**Tasks:**
1. Extend `approx::HyperLogLog` with `kernels::merge_*`.
2. SIMD merge: per-bucket max via `_mm256_max_epu8`.
3. SIMD cardinality estimate: vectorized harmonic mean.
4. Parity vs scalar; bench.

**Ship as `v0.2.2`.**

### Sprint 45-46 — MinHash SIMD (~3-5 days, IF CDN consumer asks)

**Tasks:**
1. Extend `similarity::minhash` with `kernels::{avx2, avx512, neon}`.
2. SIMD signature kernel: K min-hashes over windowed bytes in parallel via gather + reduce.
3. Parity vs scalar; bench.

**Ship as `v0.2.3`.**

---

## Phase D — Sprints 47-58 (~3-4 weeks)

### D1 — Rabbit Order

**Scope:** `permutation::rabbit_order` per Arai et al. IPDPS 2016. See `14_PERMUTATION.md` § 3. **First Rust port.**

**Strategy:** long-running feature branch `feature/rabbit-order`. Multi-step plan:

#### Sprint 47-49 — Sequential dendrogram baseline (~5 days)

1. Sequential agglomerative merging: low-degree-first vertex ordering.
2. Modularity-gain inner loop (scalar): `dQ(u, v) = w(u,v)/m - deg(u)*deg(v)/(2*m^2)`.
3. Sequential dendrogram construction.
4. DFS visit-order assignment.
5. Parity test: produce same permutation as `araij/rabbit_order` C++ reference on shared input fixtures (ship a Python script that generates fixtures both impls can consume).
6. Bench: build cost on small synthetic graphs.

**Ship gate:** sequential parity vs C++ reference on test graphs.

#### Sprint 50-52 — SIMD modularity-gain inner loop (~3-5 days)

1. AVX2/AVX-512 FMA for the modularity-gain dot-product inner loop.
2. Parity vs scalar.
3. Bench.

#### Sprint 53-55 — Concurrent merging (~5-7 days)

1. Choose concurrent data structure: `dashmap` vs hand-rolled atomic-bucket.
2. Parallel low-degree iteration with rayon.
3. Conflict resolution on concurrent merges (per-thread merge buffer + sequence-then-merge).
4. Parity vs sequential.
5. Bench: parallel speedup on multi-core.

#### Sprint 56-58 — Polish + ship D1 (~3-5 days)

1. Documentation (algorithm details).
2. Examples: compare RCM vs Rabbit Order layout quality on a real graph.
3. CHANGELOG.
4. Merge `feature/rabbit-order` to main.
5. Tag `v0.3.0`.

**Ship gate (D1):** parity vs C++ reference; concurrent merging produces same permutation as sequential modulo tie-breaking; benchmark shows quality improvement (PageRank/BFS speedup) over RCM on test graphs.

**Risk:** very high. Multi-week scope; first Rust port; concurrent data structure complexity. Mitigation: ship the sequential baseline first as a usable artifact even if concurrent variant takes longer; document the perf gap honestly.

---

## Risk register

| Sprint | Risk | Mitigation |
|---|---|---|
| 14-19 | Stream-VByte 256-entry shuffle table indexing bugs | Property tests with N up to 4096; visual diff vs Lemire's C reference output on canonical inputs |
| 20-29 | Roaring container × container parity gaps | Use both `roaring-rs` and `croaring-sys` as oracle; automated parity tests on every commit |
| 20-29 | AVX-512 VP2INTERSECT availability narrow (Tiger Lake+ only) | Implement Diez-Canas emulation as primary AVX-512 path; native VP2INTERSECT as opportunistic fast-path on detection |
| 30-37 | rank_select Vigna broadword select bit-twiddling bugs | Property tests on dense/sparse/alternating bitvectors; parity vs `sucds` |
| 47-58 | Rabbit Order concurrent merging complexity | Ship sequential baseline first; document if concurrent variant slips |
| 47-58 | C++ reference parity hard to validate without running both | Ship a fixture-generator script that both impls consume; CI runs the script and diffs outputs |

---

## Per-sprint task creation discipline

When starting a sprint:

1. Open this doc, find the sprint section.
2. For each task in the "Tasks" list, create a `TaskCreate` entry with status pending.
3. Mark first task `in_progress`; work through; mark `completed` as each lands.
4. When sprint ship gate passes, commit + push (and tag if applicable).
5. Mark all sprint tasks `completed`.
6. Open this doc; find the next sprint.

This converts the plan into the in-context task tracker on demand, so the project state is always: this doc + `TaskList` for the current sprint. No separate tracker.

## Out-of-band work woven through

Things to handle as they come up across the whole timeline:

- **Audit findings round 4+**: each audit round historically yields ~5-10 findings. Slot in between sprints as needed; treat as "audit-round-N" patches.
- **Dependabot weekly bumps**: drop in 5 minutes per week to triage; merge trivial bumps; defer major-version bumps to dedicated mini-sprints.
- **Documentation pass after each phase**: don't let docs lag more than one phase behind code. The `PRIMITIVE_KERNEL_BUFFET.md` table must list everything shipped.
- **Bench history archival**: after each release tag, snapshot `target/criterion` JSON to a versioned archive so cross-release perf regressions are catchable.
- **CI green discipline**: never push past a red main; if a PR breaks CI, revert + reland with the fix.

## Summary

**16 weeks of solo work to ship everything.** Realistic estimate. Halve with two contributors. Add 4 more weeks if v0.2.x candidates all materialize.

The plan is **dependency-driven**, not calendar-driven. Sprint boundaries are work-shaped; the rate is yours.

The plan is **release-anchored**: v0.1.x ships at week ~1.5 (2 production wins), v0.2 ships at ~10.5 weeks (the substantive primitive surface), v0.3 ships at ~16 weeks (Rabbit Order). Three real release events, not one big-bang.

**Now-action**: start Sprint 0 (pre-flight). 1-2 days of CI/bench harness work. Then Sprint 1 (A1 popcount).

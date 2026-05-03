# tokenfs-algos — Plan

**Status:** living roadmap. Created 2026-04-30; last updated 2026-05-03 (post-v0.4.6 release). The workspace, core crate, and 0.4.x release train all exist; this document records both the founding intent and the gates that must stay true as implementation continues.

**Audience:** future readers (you, me, contributors) who want to understand why this crate exists separately from `binary-bpe`, `tokenfs_writer`, and `tokenfs_reader`, what it does, and how the API and modules fit together.

---

## 0. Living Roadmap

### Released

| Version | Date | Theme |
|---|---|---|
| 0.1.x | 2026-04-30 → 05-01 | Phase A: popcount, batched hash, bit_pack |
| 0.2.x | 2026-05-01 → 05-02 | Phase B + C: streamvbyte, bitmap, rank_select, demonstrators |
| 0.3.x | 2026-05-02 | Phase D: Rabbit Order |
| 0.4.0 | 2026-05-02 | Surface flip: try_* coverage + userspace umbrella |
| 0.4.1-0.4.3 | 2026-05-02 → 05-03 | Audit-R7/R8/R9 closeouts |
| 0.4.4 | 2026-05-02 | Audit-R10 Tier 0 + Tier 1 |
| 0.4.5 | 2026-05-03 | Audit-R10 Tier 2 + Tier 3 + honest-gap closure |
| 0.4.6 | 2026-05-03 | Audit-R10 systematic gating sweep COMPLETE (allowlist 30 → 0) |

See `CHANGELOG.md` for per-release diffs and audit lineage.

### Current State (as of v0.4.6)

- 979 lib tests pass with `--all-features`; 671 with `--no-default-features --features alloc` (the kernel-safe surface).
- Every module in §4 below exists with at least a scalar reference; AVX2/AVX-512/NEON/SSE4.1/SHA-NI/FEAT_SHA2 backends exist for the modules where benchmarks justify them (popcount, bit_pack, streamvbyte, rank_select, set_membership, bloom_kernels, hll/kernels, sha256, byteclass, runlength, vector distance, similarity::minhash::kernels_gather, permutation::rabbit::kernels). `arch-pinned-kernels` controls per-backend `pub mod` visibility.
- 13 cargo-fuzz targets compile and run nightly under `.github/workflows/fuzz-nightly.yml`. cargo-mutants weekly. Miri / ASan / MSan on push + Sundays.
- 10 GitHub Actions workflows are in place; OSS-Fuzz integration files live in `oss-fuzz/` ready for upstream submission.
- **Audit-R4 through R10 are closed.** The kernel-safe-by-default narrative is structurally enforced: `tools/xtask/panic_surface_allowlist.txt` contains zero entries, and `cargo xtask panic-surface-lint` blocks any new `pub fn` that introduces a panicking macro without a `#[cfg(feature = "userspace")]` gate.
- Public primitive contracts are documented in `docs/PRIMITIVE_CONTRACTS.md`. Kernel-safety contract is documented in `docs/KERNEL_SAFETY.md` (covers the `try_*` / `_unchecked` / `_inner` conventions, the lint, and the allowlist policy).
- F22 block fingerprint has scalar + x86 AVX2/SSE4.2 pinned paths. F22 extent fingerprint has a fused scalar/runtime-dispatched accumulator; pinned scalar is exact, public default samples large-extent H4 by design.
- `no_std + alloc` is the kernel-safe surface; math goes through the crate-local `math` wrapper and `libm` outside `std`.

### Next Acceptance Gates (v0.5.0)

The remaining work for v0.5.0 is enhancement, not safety. Each item is a separate task in the project tracker:

- **T3.4 iai-callgrind** — deterministic hardware-counter benches, complementing the wall-clock criterion benches. Decide: required for v0.5.0, or push to v0.6?
- **T3.5 bench-history publication** — gh-pages or bencher.dev. Decide: which target, what cadence.
- **T3.6 std default-flip** — drop `std` from default features (audit-R9 #4 carry-over). Breaking change; decide: do this in v0.5.0 or queue for v1.0?

The rest of v0.5.0 is whatever the next audit round (R11) surfaces.

### Hardware Backend Status

- AVX2 is the primary optimized x86 target with full coverage.
- AVX-512 backends exist for popcount, hll/kernels, set_membership; gated on `feature = "avx512"` and require nightly.
- NEON backends exist for popcount, hll/kernels, set_membership, bloom_kernels, streamvbyte, bit_pack, vector distance.
- SSE4.1 backend exists for set_membership; SSSE3 backend exists for streamvbyte. Both file-split + `arch-pinned-kernels`-gated per audit-R10 T1.3.
- SHA-NI (x86) and FEAT_SHA2 (AArch64) hardware kernels exist for sha256.
- SVE / SVE2 remain feature-shaped scalar fallbacks until profiling evidence justifies real kernels.

### Consumer-Facing Milestones

- FUSE/read-path users need bounded-latency 4 KiB and 64 KiB paths, cached file/region plans, and no surprise allocation.
- Kernel-adjacent consumers need fixed-size structs, no hidden heap allocation, scalar fallback, and a future C-friendly ABI layer.
- Batch/image-building consumers need richer exact features, calibration profiles, planner parity reports, and reproducible pinned kernels.
- Python/PyO3 consumers should receive batch APIs over large buffers, not one tiny call per 256-byte block.

---

## 1. Purpose

`tokenfs-algos` is a Rust crate that provides **low-level, hardware-accelerated, content-agnostic algorithms** for reasoning about byte streams. It is the "compute primitives" layer that sits underneath higher-level systems (tokenizers, filesystem builders, compression dispatchers, dedup engines, forensic tools) which need to ask cheap questions about chunks of bytes:

- *How random does this look?*
- *Is this a known content type?*
- *How similar is this distribution to that distribution?*
- *Where should I split this stream?*
- *What's the most-frequent N-gram in this window?*

These questions arise in many domains. Bundling the answers inside `binary-bpe` was the wrong call — they're not BPE-specific. This crate carves them out as a standalone, citable, narrow-scope library.

### Design principles (carried over from `docs/encoder-substrate-axioms.md` and `docs/entropy-simd-primitives.md`)

- **No domain assumptions.** No mention of tokens, merges, extents, or filesystem objects. Pure compute over `&[u8]`.
- **Hardware acceleration is the default direction, not a marketing label.** Every hot algorithm starts with a scalar reference and only exposes optimized backends after parity tests and benchmark labels exist. AVX2 is the first production target; AVX-512, NEON, SVE, and SVE2 remain scalar fallbacks until real kernels land.
- **One-pass kernels where useful.** When several features can be computed in a single pass over the bytes, expose a fused kernel that does so — the per-block fingerprint is the canonical example.
- **Reference parity with documented tolerances.** `cargo test` enforces that optimized backends match scalar exactly where the API is exact. Approximate or sampled estimators, such as large-extent fingerprint H4, must document their tolerance and keep a pinned exact scalar path.
- **Cargo conventions.** Kebab-case crate name. Public modules. Doc comments on every public item. Examples directory. Criterion benchmarks.
- **Pre-1.0 API.** Expect breaking changes between v0.1 → v0.2 as use cases shape the surface; stabilize at v1.0 once `tokenfs_writer`, `binary-bpe`, and a handful of external consumers have shipped against it.

---

## 2. Use cases that motivated this

### Tokenfs / bbpe direct consumers

Paper labels such as F21/F22/F23b are historical lineage names. Public crate
APIs should use the product names defined in
`docs/PAPER_LINEAGE_NAMING.md` (`selector`, `fingerprint`, `sketch`,
`conditional dispatch`) while docs and calibration benches retain the paper
labels.
Hot primitive APIs follow `docs/PRIMITIVE_CONTRACTS.md`: ergonomic public path,
pinned scalar path, no hot-path allocation, scalar parity tests, and stable
benchmark labels.

1. **F21 selector** — per-extent entropy fingerprint → predict best representation.
2. **F22 fingerprint kernel** — per-256-byte-block fingerprint for the predictive selector and skip-compression fast-path.
3. **F23b ConditionalSimdEncoder** — per-window dispatch in the bbpe SIMD encoder hot loop. Skip the merge trie on high-entropy regions.
4. **F18 compression-objective BPE training** — score merge candidates by expected bit-savings under an entropy coder. Needs entropy and divergence measures during training.
5. **F14 CDC + content-addressed token IDs** — rolling hash over byte windows for content-defined chunking.
6. **TokenFS oracle's skip-compression fast-path** — H₁ > 7.9 → store as raw, skip zstd.
7. **S5 per-content-family encoder dispatch** — k-means on fingerprints; classify each extent by content family (ELF, HTML, JSON, image, text, random).
8. **S4 cross-encoder distance** — similarity between fingerprints under different encoders, for routing decisions.
9. **F11 V4 per-extent BPE** — lightweight 2-gram histogram + top-K for per-extent dictionaries.
10. **bbpe trainer compression-aware merges** — folded into the trainer's existing per-chunk scan, gives the trainer per-chunk fingerprints for free.

### Adjacent uses (the reason for separating from bbpe)

11. **Compression codec dispatchers** (zstd / lz4 / xz / raw selection per block).
12. **Filesystem dedup heuristics** (skip CDC on high-entropy chunks; tune chunk-size by content class).
13. **Forensics / packed-binary detection** (entropy + n-gram fingerprint signatures of encrypted, compressed, or packed regions).
14. **Database columnar storage** (per-column dictionary-encoding decisions; Parquet/Arrow/DuckDB-style codec policies).
15. **Network DPI / packet classification** at line rate.
16. **LLM training-data curation** (filter random/encrypted/duplicated blobs; cluster by content family).
17. **Storage-tier placement** (hot/cold/archive class from content fingerprint).
18. **Approximate string matching / set membership** (Bloom filter + count-min sketch primitives).
19. **Streaming analytics over high-throughput byte sources** (online entropy + top-K).

The first ten justify the crate; the second ten justify spinning it out from `binary-bpe`.

---

## 3. Scope — in vs out

### In scope

- Sliding/rolling windows over byte streams (1, 2, 4, 8, 16-byte n-grams)
- Histograms (byte, n-byte, hash-binned for high-cardinality)
- Streaming top-K (Misra-Gries heavy-hitters) + Count-Min Sketch
- Entropy estimators (Shannon H₁..H₈, conditional, joint, Rényi, min-entropy)
- Distribution divergence (KL, Jensen-Shannon, Hellinger, KS, χ², total variation)
- Run-length detection and statistics
- Byte classification (ASCII, control, printable, high-bit, UTF-8 validity)
- Recursive binary chunking primitives (split, accumulate, tree-fold)
- Content-defined chunking (FastCDC and Gear-hash variants)
- Composite block / extent fingerprints (the F22 kernel, productized)
- Content profile matching (does this histogram match this stored profile?)
- Hash primitives (CRC32, wyhash, rapidhash) — the SIMD-friendly ones
- Hardware backends: AVX2, AVX-512, NEON, SVE/SVE2, scalar fallback
- Runtime feature dispatch + compile-time feature flags

### Out of scope

- Tokenization, BPE merge logic, vocabulary management → `binary-bpe`
- File format parsing, image/extent abstractions → `tokenfs_writer`/`tokenfs_reader`
- ML models, decision trees, clustering algorithms → a higher-level `content-fingerprint` crate that depends on this one
- General string algorithms (regex, search, Levenshtein) → use `aho-corasick`, `regex`, `stringzilla`
- Compression codecs → use `zstd`, `lz4`, etc.; this crate only helps decide *whether/how* to compress
- Cryptographic hashes → use `sha2`, `blake3`; CRC32 here is for hashing-into-bins, not integrity
- File I/O — pure compute over slices

---

## 4. Module layout

```
tokenfs-algos/
├── Cargo.toml
├── README.md
├── PLAN.md                     # this document
├── ARCHITECTURE.md             # deeper design docs (added after v0.2)
├── src/
│   ├── lib.rs                  # public API surface, module re-exports, prelude
│   ├── prelude.rs              # convenient `use tokenfs_algos::prelude::*`
│   ├── dispatch.rs             # backend selection (init-once atomic)
│   │
│   ├── windows/
│   │   ├── mod.rs              # public: NgramWindow, RollingHash, StrideIter
│   │   ├── ngram.rs            # u8/u16/u32/u64 n-gram extraction (zero-copy iter)
│   │   ├── rolling.rs          # rolling hashes: Rabin-Karp, Gear (FastCDC), Adler32
│   │   └── stride.rs           # power-of-2 chunk iteration helpers
│   │
│   ├── histogram/
│   │   ├── mod.rs              # public: ByteHistogram, NgramHistogram<N>, ContentHistogram
│   │   ├── byte.rs             # 256-bin byte histogram (multi-counter SIMD)
│   │   ├── ngram.rs            # 2/4/8-byte n-gram histograms (sparse hash for high N)
│   │   ├── topk.rs             # Misra-Gries heavy-hitters, top-K streaming
│   │   └── sketch.rs           # Count-Min Sketch for approx counts
│   │
│   ├── entropy/
│   │   ├── mod.rs              # public: shannon, renyi, min_entropy, joint, conditional
│   │   ├── shannon.rs          # H_1, H_2, H_3, H_4, H_8 in bits/byte
│   │   ├── conditional.rs      # H(X|Y)
│   │   ├── joint.rs            # H(X,Y)
│   │   ├── renyi.rs            # Rényi entropy (collision, min-entropy)
│   │   └── tables.rs           # c·log₂(c) lookup tables, cached
│   │
│   ├── divergence/
│   │   ├── mod.rs              # public: kl, js, hellinger, ks, chi_squared, total_variation
│   │   ├── kl.rs               # Kullback-Leibler (asymmetric)
│   │   ├── js.rs               # Jensen-Shannon (symmetric, bounded)
│   │   ├── hellinger.rs        # Hellinger distance
│   │   ├── ks.rs               # Kolmogorov-Smirnov (CDF-based)
│   │   ├── chi2.rs             # chi-squared
│   │   └── tv.rs               # total variation
│   │
│   ├── runlength/
│   │   ├── mod.rs              # public: detect_runs, run_length_stats
│   │   ├── detect.rs           # SIMD run-length detection (VPCMPEQB + VPMOVMSKB)
│   │   └── stats.rs            # run-length distribution / histogram
│   │
│   ├── byteclass/
│   │   ├── mod.rs              # public: ByteClassMask, classify_byte, classify_block
│   │   ├── ascii.rs            # ASCII / printable / control
│   │   ├── utf8.rs             # UTF-8 validity (vectorised, ranged-table lookup)
│   │   └── classes.rs          # composable byte-class predicates
│   │
│   ├── chunk/
│   │   ├── mod.rs              # public: ChunkTree, BinarySplit, FastCdc, GearChunker
│   │   ├── split.rs            # recursive binary split, tree-fold reductions
│   │   ├── accumulate.rs       # parallel tree reductions over chunks
│   │   └── cdc.rs              # FastCDC + Gear-hash content-defined chunking
│   │
│   ├── fingerprint/
│   │   ├── mod.rs              # public: BlockFingerprint, ExtentFingerprint, ContentProfile
│   │   ├── block.rs            # F22 kernel — per-256-byte composite fingerprint
│   │   ├── extent.rs           # aggregate fingerprint across an extent
│   │   ├── profile.rs          # ContentProfile (stored, comparable)
│   │   └── matcher.rs          # match_score(histogram, profile) — distance-based
│   │
│   ├── primitives/             # raw SIMD primitives — pub(crate) by default
│   │   ├── mod.rs
│   │   ├── histogram_avx2.rs
│   │   ├── histogram_avx512.rs
│   │   ├── histogram_neon.rs
│   │   ├── histogram_sve.rs
│   │   ├── histogram_scalar.rs
│   │   ├── compare_avx2.rs     # SIMD compare (RL detection, n-gram match)
│   │   ├── compare_neon.rs
│   │   ├── shuffle_avx2.rs     # SIMD shuffle (nibble extract, byte permute)
│   │   ├── crc32.rs            # SSE4.2 CRC32, hash-into-bin helper
│   │   └── gather.rs           # safe wrappers around gather/scatter (where supported)
│   │
│   └── error.rs                # AlgoError type, kept tiny
│
├── benches/
│   ├── histogram.rs
│   ├── entropy.rs
│   ├── divergence.rs
│   ├── windows.rs
│   ├── runlength.rs
│   ├── chunk.rs
│   └── fingerprint.rs
│
├── examples/
│   ├── classify_block.rs       # 30-line: load a block, print fingerprint
│   ├── content_match.rs        # match a block against a stored profile
│   ├── streaming_entropy.rs    # online H_1 over a stream
│   ├── cdc_chunking.rs         # FastCDC chunking demo
│   ├── cross_encoder_distance.rs
│   └── compression_dispatcher.rs # toy "should I zstd this?" logic
│
└── tests/
    ├── parity.rs               # all SIMD backends == scalar reference
    ├── known_values.rs         # H(uniform) ≈ 8, H(constant) = 0, KL(P, P) = 0, etc.
    ├── property.rs             # proptest-based fuzzing
    └── calibration.rs          # F21 / F22 reproducibility (against fixed snapshot data)
```

The module layout is deeper than typical Rust crates because each axis (windows / histograms / entropy / divergence / chunking / fingerprint / SIMD primitives) genuinely is a separate concern, and a flat surface would obscure this. The `prelude` re-export keeps imports terse for downstream consumers.

---

## 5. Public API sketch

This is the surface that downstream consumers (`binary-bpe`, `tokenfs_writer`, future external crates) actually touch. Internal SIMD primitives stay `pub(crate)`.

```rust
use tokenfs_algos::prelude::*;

// ── Histograms ───────────────────────────────────────────────────────
let mut h = ByteHistogram::new();
h.add_block(&block);
let counts: &[u32; 256] = h.counts();
let total: u64 = h.total();

let mut h2 = NgramHistogram::<2>::new();   // const-generic n-gram order
h2.add_block(&block);

// Top-K streaming
let mut tk = HeavyHitters::<16>::new();   // Misra-Gries, K=16
for window in NgramWindow::<u32, 4>::iter(&bytes) {
    tk.observe(window);
}
let top16: &[(u32, u64)] = tk.top_k();

// ── Entropy ──────────────────────────────────────────────────────────
let h1: f32 = entropy::shannon::h_n(&h, 1);   // bits/byte
let h4: f32 = entropy::shannon::h_n(&h2, 4);
let min_h: f32 = entropy::min_entropy(&h);
let collision_h: f32 = entropy::renyi::collision(&h);

// ── Divergence ──────────────────────────────────────────────────────
let kl = divergence::kl(&hist_a, &hist_b);
let js = divergence::js(&hist_a, &hist_b);   // symmetric, bounded [0, 1]
let dks = divergence::ks(&cdf_a, &cdf_b);

// ── Run-length & byte-class ─────────────────────────────────────────
let runs = runlength::stats(&block);   // {n_runs, max_run, fraction_in_runs_ge_4, ...}
let class_mask = byteclass::classify_block(&block);
if class_mask.contains(ByteClass::ASCII_PRINTABLE) { /* ... */ }

// ── Sliding windows ─────────────────────────────────────────────────
for ngram in NgramWindow::<u32, 4>::iter(&bytes) { /* ... */ }
for chunk in StrideIter::<256>::new(&bytes) { /* ... */ }

// Rolling hash for CDC
let mut gear = GearHash::new(seed);
for byte in bytes {
    let h = gear.update(*byte);
    if h & mask == 0 { /* boundary */ }
}

// ── Recursive split / fold ──────────────────────────────────────────
let tree = ChunkTree::binary_split(&bytes, 4096 /* leaf size */);
let combined: ByteHistogram = tree.fold(
    /* leaf  */ |chunk| ByteHistogram::from_block(chunk),
    /* merge */ |left, right| left + right,
);

// ── Content-defined chunking ────────────────────────────────────────
for chunk in FastCdc::new(&bytes)
    .target_size(8192)
    .min_size(2048)
    .max_size(65536)
    .build()
{
    /* chunk: &[u8] */
}

// ── Composite fingerprints (F22) ────────────────────────────────────
let fp: BlockFingerprint = fingerprint::block(&block);
fp.is_high_entropy();         // h1_q4 >= H1_RANDOM_THRESHOLD
fp.skip_compression_likely(); // composite predicate

let efp: ExtentFingerprint = fingerprint::extent(&extent_bytes);
efp.h1();        // f32 bits/byte
efp.h4();
efp.rl_fraction();

// ── Content profile matching ────────────────────────────────────────
let profile: ContentProfile = ContentProfile::load("elf64-amd64.profile")?;
let score: f32 = profile.match_score(&fp);    // 0..1, higher = better match
let class: Option<ContentClass> = profile.classify_block(&block);
```

The shape: **one module per axis, with a small public API per module, plus a `prelude` for ergonomic imports.** Most users will only touch `histogram`, `entropy`, and `fingerprint`; advanced users reach into `divergence`, `chunk`, and `windows`.

---

## 6. Backend / dispatch strategy

### Compile-time

`Cargo.toml` features:

```toml
[features]
default = ["std"]
std = []                 # always on for now; could shrink to no_std later
avx2 = []                # opt-out, default-on for x86_64
avx512 = []              # opt-in, requires nightly until stable intrinsics
neon = []                # opt-out, default-on for aarch64
sve = []                 # opt-in
sve2 = []                # opt-in
parallel = ["dep:rayon"] # parallel kernels via rayon
nightly = []             # gates AVX-512 and other unstable features
```

### Runtime

A single `Backend` enum is set once on init:

```rust
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Backend {
    Avx512,
    Avx2,
    Sve2,
    Sve,
    Neon,
    Scalar,
}

pub fn detected_backend() -> Backend { /* atomic-cached, init-once */ }
pub fn force_backend(backend: Backend) { /* for testing/benchmarks */ }
```

Each algorithm dispatches via `cfg-if` + `is_x86_feature_detected!`/`is_aarch64_feature_detected!` macros, with `pub(crate)` per-backend implementations. Pattern is well-established (see `bytemuck`, `wide`, `simdeez` for prior art on how to keep this clean).

### Correctness

- **Parity tests**: every exact public function has tests that run on every
  implemented backend (including scalar) and assert identical output on the same
  input. Approximate/sampled public defaults assert documented tolerances and
  keep a pinned exact scalar path.
- **Property-based tests**: proptest over random inputs, ensuring invariants hold (entropy is in [0, log₂ N], divergence is non-negative, etc.).
- **Calibration tests**: long-running, gated behind `--features=calibration`, that reproduce the F21/F22 selector accuracy on a fixed snapshot of the rootfs sample. Catches accidental algorithm drift between revisions.

---

## 7. Phasing roadmap

### v0.1 — "core kernels" (target: 2 weeks of focused work)

- byte histogram + n-gram histogram (n ∈ {1, 2, 4, 8})
- Shannon entropy H_1..H_8 with `c·log₂(c)` LUT
- run-length detection + stats
- byte classification (ASCII / control / high)
- Misra-Gries top-K
- Block + extent fingerprint (F22 kernel productized)
- AVX2 + scalar backends only
- Public API stable enough that `binary-bpe` can depend on it
- README, examples for `classify_block` and `streaming_entropy`

**Acceptance criteria**: F21 selector accuracy reproduces within 1% using this crate's primitives. F22 microbench gates are checked by `cargo xtask bench-real-f22` (current hard gate ≤1.8 µs/block AVX2 and ≥0.93 GiB/s extent throughput; optimization target ≤1.5 µs/block and quiet-host extent goal ≥1 GB/s).

### v0.2 — "distribution analysis" (target: +2 weeks)

- All divergence measures (KL, JS, Hellinger, KS, χ², TV)
- ContentProfile + match_score
- Recursive binary chunking (split, fold, accumulate)
- FastCDC + Gear-hash rolling hash
- Count-Min Sketch
- AVX-512 backend (nightly-gated initially)
- More examples: `content_match`, `cross_encoder_distance`, `cdc_chunking`

**Acceptance criteria**: a `content-fingerprint` crate (separate, downstream) can be written against this API to do the F21 decision-tree selector in ~200 LOC. CDC at line rate (target: ≥1.5 GB/s).

### v0.3 — "Arm + advanced entropy"

- NEON backend (Apple Silicon, AWS Graviton)
- Joint and conditional entropy
- Rényi entropy, min-entropy
- UTF-8 validation (vectorised, follows `simdjson`/`fast-utf8` patterns)
- Online streaming estimators (P² for quantiles)

### v0.4 — "advanced SIMD"

- SVE / SVE2 backend (Graviton 3/4, Cortex-X3+, Apple M4)
- AVX-10 (early)
- AVX-512 stable on stable Rust (whenever the intrinsics stabilize)
- Composite fingerprint diff/distance for cross-encoder mapping (axiom S4)

### v1.0 — "stable"

- API stable, semver from here on
- Published to crates.io
- ARCHITECTURE.md walks through the codebase
- 3+ external consumers cited in README

---

## 8. Dependencies (kept tight)

```toml
[dependencies]
cfg-if = "1"
bytemuck = { version = "1", optional = true }     # safe transmute
rayon = { version = "1", optional = true }        # parallel kernels (feature-gated)

[dev-dependencies]
criterion = "0.5"
proptest = "1"
hex-literal = "0.4"
```

Explicitly NOT depending on:
- `tokenizers` / `bbpe` (it's the other way around)
- ML libraries (those go in higher layer)
- Anything domain-specific

`stringzilla` could be a *target* of FFI from this crate for advanced primitives (per the F23a research finding that it has SIMD min-hash and count-min-sketch primitives), but isn't a hard dep — its inclusion is feature-gated and only for backends not implementable cleanly in pure Rust.

---

## 9. Migration plan from F22 prototype

The F22 work currently lives at `/home/mjbommar/projects/personal/tokenfs-paper/tools/rust/entropy_primitives/`. To productize:

1. Create the new crate skeleton at `/home/mjbommar/projects/personal/tokenfs-algos/`. `cargo init --lib`. Set up the module structure per §4.
2. Move F22's `BlockFingerprint`, `fingerprint_block`, `fingerprint_extent` into `src/fingerprint/`.
3. Move F22's primitives (`scalar.rs`, `avx2.rs`, dispatch) into `src/primitives/` with the per-backend file split.
4. Refactor F22's monolithic kernel into the new module structure: byte histogram → `histogram/byte.rs`, run-length → `runlength/`, entropy reduction → `entropy/shannon.rs`, etc.
5. Re-run F22's parity, calibration, and benchmark tests in the new location. They must pass before we move on.
6. Delete `tools/rust/entropy_primitives/`. The tokenfs-paper repo gets a new `Cargo.toml` workspace dependency on `tokenfs-algos = { path = "../tokenfs-algos" }` (or a published version once we publish).
7. Have `binary-bpe` add `tokenfs-algos` as an optional dep, gated behind a feature flag, ready for F23b integration.

After this migration the F22 work is no longer scattered "scratch" code; it's the v0.1.0 of a real crate.

---

## 10. Naming bikeshed

Crate name candidates:

- `tokenfs-algos` (the user's choice, what this plan assumes) — clear, owned, but ties the crate to TokenFS branding even though scope is broader
- `byte-algos` — describes scope honestly
- `entropy-toolkit` / `histogram-rs` — narrower than reality
- `simd-content` / `content-primitives` — describes use case

Recommendation: **stay with `tokenfs-algos`** despite the scope being broader. The TokenFS framing is a useful organizing identity; people will understand what it does from the README, and the alternative names sound either too narrow (`histogram-rs`) or too generic (`content-primitives`). If the crate later spawns a sibling that focuses on a non-TokenFS use case, that one gets its own name.

The crate has its own GitHub repo at `mjbommar/tokenfs-algos`. The TokenFS paper cites it as `\bibcite{bommarito2026tokenfsalgos}`.

---

## 11. Open questions

1. **Workspace layout.** Single crate at `/home/mjbommar/projects/personal/tokenfs-algos/`, or workspace with future `content-fingerprint` and `bbpe-fingerprint` siblings? **Recommended: workspace from day one**, even if only `tokenfs-algos` exists initially. Saves migration pain later.

2. **Initial Rust MSRV.** Target stable Rust as far back as is reasonable — probably 1.74 (matches `binary-bpe`). AVX-512 features are nightly-gated until intrinsics stabilize.

3. **`no_std` support.** Probably yes eventually. v0.1 stays `std` for ease; v0.3+ adds `no_std` feature.

4. **Public surface for SIMD primitives.** Are `histogram_byte_avx2(...)` and friends `pub` or `pub(crate)`? **Recommended: `pub(crate)` for v0.1** — too easy to misuse (alignment, pointer provenance, target-feature requirements). Re-evaluate at v0.3+.

5. **Versioning policy.** Pre-1.0 expected breaking changes; semver-strict from v1.0. **Recommended: bump minor on any public-API change pre-1.0**, so consumers can pin `~0.1.0` if they want stability.

6. **Benchmarks vs. tests.** `cargo bench` runs criterion benchmarks; `cargo test` runs parity + property + calibration. The calibration tests need a small snapshot of F21 fingerprint data (5,000 rows, ~340 KB parquet) — should this live in the repo, or be downloaded from a release artifact? **Recommended: in the repo**, in `tests/data/`. ~340 KB is fine.

7. **Cross-platform CI.** GitHub Actions matrix: ubuntu-latest (x86_64 AVX2), ubuntu-latest-arm (aarch64 NEON), macos-latest (Apple Silicon NEON). Each runs `cargo test --release`. AVX-512 + SVE need self-hosted runners or skipping.

8. **Public profile catalog.** ContentProfile is a serialised distribution. Do we ship a catalog of common profiles (ELF, PNG, gzip, JSON, plain-text, etc.) with the crate, or as a separate optional asset crate? **Recommended: separate asset crate `tokenfs-algos-profiles`** — profiles are large and update independently of the algorithms.

9. **Documentation site.** docs.rs is automatic. Do we need a separate mdBook? Probably not for v0.1; revisit at v0.3 once the API surface is large.

---

## 12. Definition of done — v0.1.0 release

- All §4 modules implemented at least at scalar level; AVX2 backend for histogram, byte-class, RLE, fingerprint kernel.
- All §5 public API items have doc comments with at least one usage example.
- Parity tests pass on AVX2 + scalar.
- Calibration test reproduces F21 selector accuracy within 1%.
- `cargo xtask bench-real-f22` checks F22 hard gates (1.8 µs/block, 0.93 GiB/s current extent gate), with 1.5 µs/block and 1 GB/s extent retained as quiet-host optimization targets.
- README has: motivation, quick-start, backend matrix, performance numbers, links to PLAN.md / ARCHITECTURE.md.
- 5+ examples runnable via `cargo run --example`.
- Published to crates.io (or held back behind a soft-launch period if we want to iterate without yanking).

---

## 13. Active Next Steps

As of v0.4.6, in priority order:

1. **Triage v0.5.0 scope.** The remaining items are T3.4 (iai-callgrind), T3.5 (bench-history publication), T3.6 (drop `std` from default features). Decide which are in-scope for v0.5.0 vs deferred. T3.6 is breaking; either v0.5.0 absorbs the breaking change or it waits for v1.0.
2. **Submit OSS-Fuzz integration.** The `oss-fuzz/` directory has the upstream files ready (`Dockerfile`, `build.sh`, `project.yaml`). Open a PR against `google/oss-fuzz`. Once merged, ClusterFuzz coverage tracking + automatic regression bisection come for free.
3. **Bake the new CI workflows.** `.github/workflows/{sanitizers,coverage,fuzz-nightly,mutation-testing,bench-regression,calibration}.yml` are all new in v0.4.5. Watch the first few weeks of runs, tune thresholds, and adjust the bench-regression threshold (currently 15%) once we have a noise baseline from GitHub-hosted runners.
4. **Set up the calibration host.** `.github/workflows/calibration.yml` requires a `[self-hosted, perf-quiet]` labeled runner. Until that exists, the workflow queues forever then times out. Either provision the host or rewrite the workflow to opt-in via `workflow_dispatch` only.
5. **Profile before new SIMD.** Add AVX-512 / NEON / SVE kernels only after flamegraphs or timing tables identify the next bottleneck. iai-callgrind (T3.4) will help here.
6. **Downstream integration.** Wire `binary-bpe` and TokenFS consumers against `fingerprint`, `selector`, `chunk`, `distribution`, `permutation`, `similarity`, and `format` APIs. The kernel-safe-by-default surface is now stable enough for kernel-mode consumers to design against.
7. **Audit-R11.** Schedule the next external audit pass once v0.5.0 ships. The previous audit cadence (R4 → R10) ran in parallel with implementation; R11 should focus on the consumer-facing API ergonomics and the no-panic surface from a fresh reviewer.

---

*This plan is intended to keep changing as code lands. Every implementation phase should either satisfy an acceptance gate above or update the gate with a concrete reason.*

# tokenfs-algos — Plan

**Status:** founding plan document. 2026-04-30. The crate does not yet exist; this document describes what it will be, scoped against the use cases that motivated it.

**Audience:** future readers (you, me, contributors) who want to understand why this crate exists separately from `binary-bpe`, `tokenfs_writer`, and `tokenfs_reader`, what it does, and how the API and modules fit together before any code is written.

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
- **Hardware acceleration is the default, not the exception.** Every algorithm has at least an AVX2 + scalar backend; key algorithms add AVX-512 and NEON. Runtime feature dispatch is set once on init.
- **One-pass kernels where useful.** When several features can be computed in a single pass over the bytes, expose a fused kernel that does so — the per-block fingerprint is the canonical example.
- **Bit-exact parity across backends.** `cargo test` enforces that the AVX2, AVX-512, NEON, and scalar backends produce identical results. Any backend-specific approximation (e.g., `c·log₂(c)` lookup tables) is shared across all backends.
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

- **Bit-exact parity tests**: every public function has a `tests/parity.rs` test that runs on every backend (including scalar) and asserts identical output on the same input. CI matrices over backends.
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

**Acceptance criteria**: F21 selector accuracy reproduces within 1% using this crate's primitives. F22 microbench targets met (≤1.5 µs/block AVX2, ≥1 GB/s extent throughput per-block-off).

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
- `cargo bench` shows F22 targets hit (1.5 µs/block, 1 GB/s extent).
- README has: motivation, quick-start, backend matrix, performance numbers, links to PLAN.md / ARCHITECTURE.md.
- 5+ examples runnable via `cargo run --example`.
- Published to crates.io (or held back behind a soft-launch period if we want to iterate without yanking).

---

## 13. Next steps from this plan

In the order they should happen:

1. **Approve / revise this plan.** Read through and flag any module that should be added / removed / renamed before code is written. Especially §4 and §5.
2. **Create the crate skeleton.** `cargo init --lib`, module stubs per §4, `Cargo.toml` per §8.
3. **Migrate F22.** §9 step-by-step. Verify parity + calibration + bench targets in the new location.
4. **Build out v0.1 modules** in priority order: `histogram` → `entropy` → `runlength` → `byteclass` → `windows` → `chunk` → `fingerprint`.
5. **Wire `binary-bpe` to depend on `tokenfs-algos`** — this is F23b. The `ConditionalSimdEncoder` calls into the new crate's `fingerprint::block` API.
6. **Publish v0.1.0** to crates.io once F23b lands and the API has held stable for a week of iteration.
7. **Begin v0.2** work (divergence, CDC, AVX-512). At this point the upper layer (`content-fingerprint`) starts to become reasonable to build.

---

*This plan is intended to be revised before any code lands. Open a follow-up doc or PR comment if a section needs to change.*

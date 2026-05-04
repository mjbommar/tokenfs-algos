# HNSW Builder Determinism and Reproducibility

Status: research draft for `tokenfs-algos` HNSW kernel
Audience: implementers of `Builder`, reviewers of the SLSA-L3 image profile
Companion specs:
- `/home/mjbommar/projects/personal/tokenfs-paper/docs/IMAGE_FORMAT_v0.3.md` §11
- `/home/mjbommar/projects/personal/tokenfs-paper/docs/IMAGE_FORMAT_v0.2.md` §11 (still load-bearing — v0.3 §11 only adds deltas)

This document enumerates every source of non-determinism in an HNSW build and the rules a Rust `Builder` MUST follow so that byte-identical input plus a fixed set of writer-side parameters yields a byte-identical serialized HNSW section. The HNSW section is one of several content-addressed sections inside a sealed TokenFS image; its bytes flow into the image's Merkle tree (§8.9) and ultimately into `image_uuid`. Any non-determinism in the HNSW builder propagates into the image hash and breaks SLSA-L3 verifiable rebuild.

---

## 1. SLSA-L3 reproducibility, precisely

### 1.1 What SLSA-L3 actually requires

SLSA v1.0 defines four Build levels (L0–L3). Build L3 is the highest defined level and adds two requirements over Build L2 ("Hosted, signed provenance"):

- **Provenance Unforgeable.** Every field in the provenance MUST be generated or verified by the build platform in a trusted control plane. Secret material (signing keys, attestation keys) MUST NOT be accessible to the user-defined build steps. (https://slsa.dev/spec/v1.0/requirements §"Build L3", §"Provenance unforgeable")
- **Isolation Strength.** It MUST NOT be possible for one build to persist or influence the build environment of a subsequent build; an ephemeral build environment MUST be provisioned for each build. It MUST NOT be possible for two overlapping builds to influence one another. (Same source, §"Isolated")

### 1.2 Reproducible builds vs SLSA — the distinction

SLSA does not, by itself, require bit-identical reproducible builds at any level. The v1.1 FAQ is explicit: "SLSA does not require verified reproducible builds directly. Instead, verified reproducible builds are one option for implementing the requirements." (https://slsa.dev/spec/v1.1/faq)

The reproducible-builds.org project defines reproducible separately: "A build is reproducible if given the same source code, build environment and build instructions, any party can recreate bit-by-bit identical copies of all specified artifacts." (https://reproducible-builds.org/docs/definition/)

### 1.3 How TokenFS combines them

TokenFS images are content-addressed and sealed; `image_uuid` is computed from a Merkle root over all sections. The image-format spec elects to make verified reproducibility a profile-level MUST: IMAGE_FORMAT_v0.2.md:765 reads "A SLSA Level 3+ build pipeline MUST hold all of the above" — "the above" being the v0.2 §11 determinism rules, including IMAGE_FORMAT_v0.2.md:763 "MinHash / HNSW (when present) MUST seed RNGs from `image_salt`."

In other words: SLSA-L3 alone is silent on reproducibility; the TokenFS image profile layers a verified-reproducible-build requirement on top of SLSA-L3, and the HNSW builder is one of the artifacts that requirement binds. The HNSW section MUST be byte-identical for byte-identical inputs, where "inputs" includes the canonicalized vector list, the `image_salt`-derived RNG seed, and the writer parameters (`connectivity`, `expansion_add`, metric kind, distance dtype).

### 1.4 What "byte-identical" applies to here

The serialized object MUST be byte-identical at the granularity of a single contiguous HNSW section. This includes:

- The graph header (size, connectivity, connectivity_base, max_level, entry_slot — see usearch's `index_serialized_header_t` at `_references/usearch/include/usearch/index.hpp:1990`).
- Per-node level array.
- Per-node neighbor lists at every level, in the exact order the builder linked them.
- Reserved/padding bytes (MUST be zero — IMAGE_FORMAT_v0.2.md:757).

It does NOT (yet) extend cross-architecture for f32 metrics; see §9.

---

## 2. Sources of HNSW build non-determinism

Each of the following is a known channel by which two runs of the same builder can diverge.

### 2.1 Random level assignment (RNG state)

HNSW assigns each new node a "top level" drawn from a geometric distribution. usearch implements this at `_references/usearch/include/usearch/index.hpp:4055`:

```
level_t choose_random_level_(std::default_random_engine& level_generator) const noexcept {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -std::log(distribution(level_generator)) * pre_.inverse_log_connectivity;
    return (level_t)r;
}
```

The RNG is `std::default_random_engine` per-context (per-thread), constructed without explicit seeding (see `index.hpp:2334`). With its default constructor `std::default_random_engine` uses an implementation-defined seed; usearch does not call `seed()` on it. This is one of the largest single sources of non-reproducibility in usearch's default mode and is a known footgun reported across implementations (see Faiss issue #1073, Milvus issue #25908, https://github.com/facebookresearch/faiss/issues/1073, https://github.com/milvus-io/milvus/issues/25908).

Our builder MUST use a seeded RNG per §3.3.

### 2.2 Concurrent insert (thread interleaving)

When two threads insert simultaneously, several outcomes depend on interleaving:
- Order in which nodes claim slots (`nodes_count_.fetch_add(1)` at `_references/usearch/include/usearch/index.hpp:2945`).
- Order in which neighbor lists on a shared node observe the new edge (the per-node `node_lock_` at `_references/usearch/include/usearch/index.hpp:2989` serializes individual writes but does not serialize the pair "search graph then write back").
- Which of two simultaneous inserts becomes the new top-level entry point when both pick a `new_target_level > max_level_copy` (`_references/usearch/include/usearch/index.hpp:3006`).

The hnswlib README and the Java port both note that multi-threaded `add_items` is correctness-safe but produces a non-reproducible graph. Pegasus and the bug threads at https://github.com/milvus-io/milvus/issues/25908 confirm: "If `full_speed` is enabled (multi-thread construction), the kNN results are not reproducible."

Our SLSA-L3 path BANS concurrent insert. See §3.1.

### 2.3 Floating-point summation order

Distance functions (Euclidean, cosine) reduce a per-element product across the vector. The reduction order changes the f32 value because addition is non-associative in IEEE-754:
- Scalar left-to-right vs SIMD partial sums + final reduction give different bit patterns.
- A 256-bit SIMD reduction tree differs from a 512-bit one (different number of accumulators).
- An FMA-based dot product differs from separate-multiply-then-add (one rounding vs two; see §9).

For pure-integer metrics (Hamming popcount, Jaccard on bitsets, Tanimoto on u8/i8) reduction order does not affect the value because integer addition is associative.

### 2.4 Tie-breaking in candidate lists

When two candidates have the same distance, the order in which they enter or leave the working min-heap depends on heap implementation details and on the order they were discovered. usearch's `candidate_t` (`_references/usearch/include/usearch/index.hpp:2224`) compares only on `distance`:

```
inline bool operator<(candidate_t other) const noexcept { return distance < other.distance; }
```

There is no secondary key. With ties, the heap's siftdown/siftup behavior depends on insertion order, which depends on the order neighbors were discovered, which depends on the order edges were laid down — i.e. the whole build history. This is a real divergence channel, not a theoretical one, especially for binary/integer metrics where ties are common.

Our builder MUST add a deterministic secondary key. See §3.5.

### 2.5 HashMap iteration order

Rust's `std::collections::HashMap` uses `RandomState`, which seeds from OS entropy; iteration order changes per process. (https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html, https://internals.rust-lang.org/t/support-turning-off-hashmap-randomness/19234)

Anywhere the builder iterates over a `HashMap` to write something or to make a decision, the output depends on RNG state set at process start. This is the canonical reason https://github.com/rust-lang/rust/issues/34902 ("Bit-for-bit deterministic / reproducible builds") cites for rustc's own reproducibility issues.

Our builder MUST avoid `HashMap` for any iteration that affects output. Options: `BTreeMap`, sorted `Vec<(K, V)>`, or `IndexMap` (insertion-order-preserving, but only deterministic if insertion order is itself deterministic). See §8.

### 2.6 Allocator behavior

In pure compute code with no I/O and no cross-allocation pointer-arithmetic decisions, allocator behavior is invisible to the output. The HNSW builder allocates node tapes and neighbor buffers via `Vec`/`Box`, but only writes byte content into them, never reads addresses. Verify this remains true: forbid any `as usize` from a pointer that flows into output, and forbid any `Hash` impl that uses `*const T` keys.

### 2.7 System time / nanos

The builder takes no clock readings. The image-level `mkfs_epoch` is set by the writer, not the HNSW kernel (IMAGE_FORMAT_v0.2.md:749). Verify: grep `Instant::now`, `SystemTime::now`, `chrono::Utc::now`, `time::OffsetDateTime::now_utc` in the HNSW crate at PR-time.

### 2.8 Runtime CPU feature detection

`std::is_x86_feature_detected!` and `std::arch::is_aarch64_feature_detected!` return different values on different CPUs, and the builder may then dispatch to a different distance kernel. For integer metrics (Hamming, Jaccard) the answer is identical across kernels because integer reductions are associative.

For f32 metrics the answer differs because:
- AVX2 dot product uses `vmulps` + `vaddps` (two roundings).
- AVX2 + FMA enables `vfmadd231ps` (one rounding).
- AVX-512 widens accumulators (different reduction tree shape).
- NEON in ARMv8 always emits FMA — there is no scalar multiply-then-add path on ARMv8 SIMD (https://www.kdab.com/fma-woes/).

This is the cross-arch problem of §9.

---

## 3. Our determinism contract for v1

### 3.1 Single-threaded insert

The SLSA-L3 builder path runs `Builder::try_insert` strictly single-threaded. No `rayon`, no `std::thread`, no `tokio::spawn` inside the build. `Builder` is `!Sync` to enforce this at the type level.

A non-deterministic `parallel` cargo feature MAY be added later for speed-only builds that do not flow into a sealed image, but it MUST NOT be enabled in the SLSA-L3 image-build profile, and the resulting graph MUST NOT be admitted into a content-addressed image section.

### 3.2 Sorted input

Determinism is a contract between caller and builder. The contract requires the caller to pre-sort input vectors by `NodeKey` ascending (lex-byte order) before the first `try_insert`. Insertion order determines which nodes pick the top entry point and which neighbor edges form first; with sorted input, a given vector list always inserts in the same order regardless of how the caller stored or iterated over it.

This is a caller-side rule, documented at the `Builder::try_insert` rustdoc. The builder does NOT validate sort order at runtime in release builds (cost), but a `debug_assert!(key > self.last_key)` SHOULD guard each insert in debug builds.

### 3.3 Seeded RNG for level assignment

The level-assignment RNG is one of:
- `rand_chacha::ChaCha8Rng` (recommended; simpler, slightly slower but cryptographically strong, well-tested across versions).
- `rand_xoshiro::Xoshiro256PlusPlus` (faster; equally portable).

Both implement `rand::SeedableRng` and are explicitly recommended for reproducibility by the rand book (https://rust-random.github.io/book/guide-seeding.html: "for a reproducible generator, use a named PRNG from an external crate, e.g. `rand_xoshiro` or `rand_chacha`"). `rand::rngs::StdRng` is BANNED — its underlying algorithm is unspecified across `rand` versions.

Seed source:
- The default seed is a fixed crate-wide constant (e.g. the bytes of `b"tokenfs-hnsw-v1\0"` zero-padded to 32 bytes).
- The caller MAY supply a seed via `BuilderConfig::level_seed: [u8; 32]`.
- In the TokenFS image-build path, the seed is derived from `image_salt`: `Blake3KDF(image_salt, "tokenfs-hnsw-level-rng-v1")`. This satisfies IMAGE_FORMAT_v0.2.md:763.

### 3.4 Floating-point handling

The default integer-metric path (`hamming_u64`, `jaccard_u64`, `tanimoto_u8`) is fully reproducible across architectures because every operation is a deterministic integer reduction.

For f32 metrics (`l2sq_f32`, `cos_f32`, `dot_f32`):
- Order of reduction is fixed in code: scalar left-to-right for the canonical kernel; SIMD kernels reduce by a fixed pattern documented in `KERNELS.md`.
- The builder's f32 distance kernel is selected at build time via cargo feature, not at runtime via CPU detection, in the SLSA-L3 build profile. Default feature is `kernel-portable-f32` which uses a single scalar implementation.
- Runtime SIMD dispatch is OFF for SLSA-L3 builds (see §9 for the reasoning).

The canonical SLSA-L3 image-build configuration uses integer metrics only; f32 metrics are flagged as "non-canonical for sealed images" in the `BuilderConfig` rustdoc.

### 3.5 Tie-breaking

When two candidates have equal distance, the candidate with the lower `NodeKey` wins (sorts first in the heap, gets selected first as a neighbor). Encode this in the candidate type:

```
struct Candidate {
    distance: D,        // primary: lower wins
    node_key: NodeKey,  // secondary: lower wins (tie-break)
    slot: Slot,         // tertiary: lower wins (paranoia tie-break)
}
```

with a manual `Ord` impl that compares `(distance, node_key, slot)` lexicographically. This is enforced in:
- The min-heap used for the working candidate set in `search_to_insert`.
- The bounded result-set used during `form_links_to_closest`.
- The neighbor-list write that materializes the chosen edges (sort the final list before writing, so insertion order into the heap cannot leak).

This rule fixes the divergence channel of §2.4. Document it in the `Candidate` rustdoc and in this file.

### 3.6 No mid-build mutation of writer-side parameters

`connectivity`, `connectivity_base`, `expansion_add`, metric kind, distance dtype, and `level_seed` are frozen at `Builder::new` time. There is no `change_expansion_add`-style hook (usearch exposes one at `_references/usearch/include/usearch/index_dense.hpp:718`; we deliberately do not). Mutating these mid-build would create a build that depends on call ordering not visible from the input.

---

## 4. Reproducibility test strategy

### 4.1 Unit-level

For every supported metric, a `#[test]`:
1. Build twice from the same input vector list with the same config.
2. Serialize each to bytes.
3. `assert_eq!(bytes_a, bytes_b)`.

Run this in CI on every push.

### 4.2 CI-level

A dedicated `builder-determinism` job:
- Runs the unit-level test on Linux x86_64, Linux aarch64 (qemu or native), and macOS aarch64.
- For integer metrics, asserts the bytes are identical across all three.
- For f32 metrics, asserts the bytes are identical within an architecture but does NOT assert across architectures (see §9).

A separate `builder-determinism-cross-version` job:
- Pinned to a specific `rustc` version (the minimum supported `rust-toolchain.toml`).
- Re-runs the deterministic build and asserts the bytes match a reference fixture committed to the repo.
- This catches accidental dependency on `rustc` codegen non-determinism (https://github.com/rust-lang/rust/issues/34902).

### 4.3 Cross-platform

- **Integer metrics (Hamming, Jaccard, Tanimoto on i8/u8/binary):** byte-identical across x86_64, aarch64, and any tier-1 Rust target. This is the canonical SLSA-L3 path.
- **f32 metrics:** byte-identical within an architecture-and-feature-flag pair. Documented as known cross-arch divergence; not admitted into the SLSA-L3 path. See §9 for options.

### 4.4 Cross-version (vs libusearch v2.25)

We need round-trip-compatibility tests against libusearch v2.25 in single-thread mode:
1. Run libusearch v2.25 `add()` with `index_update_config_t::thread = 0` for every input vector, with seeded `level_generator` (requires patching usearch — its `level_generator` at `_references/usearch/include/usearch/index.hpp:2334` is not seeded by the public API).
2. Save via `save_to_stream` (`_references/usearch/include/usearch/index.hpp:3412`).
3. Load that file with our reader, walk the graph, assert structure matches what our builder produces.

For the reverse direction (we build, libusearch reads):
1. Our builder emits a usearch-compatible byte layout (header `index_serialized_header_t` at `_references/usearch/include/usearch/index.hpp:1990`, then the level array, then the per-node tapes).
2. `usearch_load` succeeds.
3. Searches against the loaded index match our in-memory searches against the same graph.

Caveat: libusearch's `level_generator` is not seeded by its public API. The cross-version test fixture is built with a one-line patch to expose seeding, applied via a `_references/usearch/patches/seed-level-generator.patch` file. Document this in the test fixture's README.

---

## 5. usearch's own determinism story

Read `_references/usearch/include/usearch/index_dense.hpp` and `index.hpp` for what usearch does today.

### 5.1 `expansion_add` parameter

`expansion_add` (`_references/usearch/include/usearch/index_dense.hpp:103`, default 128 from `default_expansion_add()` at `_references/usearch/include/usearch/index.hpp:1472`) sets the size of the working candidate set (`top` and `next` heaps) during insertion. Larger means more candidates considered before pruning to the final neighbor list. It is the construction-time analog of `efSearch`.

`expansion_add` is part of the determinism contract: same value in, same graph out; different value in, different graph out. It MUST be frozen at `Builder::new` time and MUST appear in the build manifest.

### 5.2 Single-thread mode

usearch supports per-call `thread` parameter (`_references/usearch/include/usearch/index_dense.hpp:788`); passing `0` (or `any_thread()` which then resolves to a specific thread) routes the call through a single per-thread context. With `threads = 1` at index construction time and `thread = 0` on every `add()` call, the build is single-threaded.

But the level generator at `_references/usearch/include/usearch/index.hpp:2334` is constructed default and never seeded by the public API — single-threaded usearch is still non-reproducible across processes unless you patch this. There is no public seed parameter.

### 5.3 usearch caveats and known non-determinism

- Multi-threaded inserts produce non-deterministic graphs. The `node_lock_t` at `_references/usearch/include/usearch/index.hpp:3932` and the striped locks at `_references/usearch/include/usearch/index.hpp:606` provide correctness, not reproducibility.
- The `add()` path uses `nodes_count_.fetch_add(1)` (`_references/usearch/include/usearch/index.hpp:2945`) for slot allocation — slot N's identity depends on which thread won the race.
- `change_expansion_add` (`_references/usearch/include/usearch/index_dense.hpp:718`) lets you mutate the expansion factor mid-build, breaking the "config in → graph out" contract; do not use.
- The serialized header (`_references/usearch/include/usearch/index.hpp:1990`) does NOT record `expansion_add`. So even a byte-identical save does not guarantee that the reproducer used the same construction parameters; you must out-of-band-record `expansion_add` if you want third-party verification.

---

## 6. Other systems' determinism stories

### 6.1 faiss

Faiss design doc (https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls): "by default random seeds are fixed so that computations are deterministic. When computations are multi-threaded, this is true as well, since most computations can be balanced statically (`omp schedule static`)."

In practice, users still report non-reproducible HNSW builds (https://github.com/facebookresearch/faiss/issues/1073: three runs of the same data yielded recall 0.973862, 0.972478, 0.972676). The Faiss FAQ acknowledges that "even with a single thread, MKL does not by default guarantee bit-exact reproducible operations like matrix multiplication, which varies depending on the processor, but this can be overridden by setting the `MKL_CBWR` environment variable."

So Faiss has a documented determinism intent but does not deliver bit-exact reproducibility on f32 paths because of MKL non-determinism. Integer metrics (Faiss's `IndexBinaryHNSW`) are reproducible.

### 6.2 hnswlib (nmslib)

`init_index(random_seed=100)`. The seed controls the level-generator. README and DeepWiki (https://deepwiki.com/nmslib/hnswlib/2.1-algorithm-parameters) confirm:
- Single-threaded build with fixed `random_seed` → reproducible.
- Multi-threaded build (`add_items` with `num_threads > 1`) → non-reproducible. Documented in README as a known limitation.
- Saving and loading round-trips: the seed is written into the index and restored on load.

This is the closest existing precedent for our model.

### 6.3 pgvector

pgvector inherits PostgreSQL's WAL-based replication. Each replica replays the same insert sequence. But pgvector's HNSW build uses parallel workers by default (`max_parallel_maintenance_workers`), which makes the index non-deterministic across replicas of the same database. The pgvector README does not document a reproducible-build mode. (https://github.com/pgvector/pgvector — search README for "deterministic" returns no hits.) For pgvector, a logical replica re-builds the index locally; the index is not part of the replicated payload.

### 6.4 qdrant

qdrant uses the geometric distribution for level assignment (DeepWiki at https://deepwiki.com/qdrant/qdrant). Qdrant exposes no documented `random_seed` parameter on collection creation. Their HNSW builder is not designed for cross-process bit-identical builds; it is designed for re-buildable indexes from a deterministic point sequence in a single process.

### 6.5 Summary table

| System    | Single-thread reproducible? | Seeded RNG? | Cross-arch f32 bit-identical? |
|-----------|-----------------------------|-------------|-------------------------------|
| usearch   | No (level_generator unseeded by public API) | No public param | No |
| faiss     | Yes for binary HNSW; NO for f32 (MKL) | Yes by design | No |
| hnswlib   | Yes | Yes (`random_seed`) | No |
| pgvector  | Not the design goal | No | No |
| qdrant    | Not the design goal | No | No |

We are not aware of any production HNSW implementation that delivers bit-identical cross-architecture f32 reproducibility. Our v1 design accepts this and constrains the SLSA-L3 path to integer metrics.

---

## 7. Test fixture strategy

### 7.1 What we commit

For each SLSA-L3-eligible metric:
- `vectors.bin` — the input vector list, sorted by `NodeKey`, stored in our canonical wire format (the same bytes a TokenFS section would contain).
- `config.toml` — the `BuilderConfig` (connectivity, connectivity_base, expansion_add, metric, level_seed).
- `expected.bin` — the expected serialized HNSW section bytes, produced by the SLSA-L3 reference builder.
- `expected.sha256` — the hash of `expected.bin`.

`expected.bin` is produced by the patched libusearch v2.25 (see §4.4) for cross-version verification, and the expectation is that our Rust builder produces the exact same bytes.

### 7.2 Test corpora

| Name | Vectors | Dim | Bytes/vec | Total raw | With graph (~3x) | Cumulative repo cost |
|------|---------|-----|-----------|-----------|-------------------|----------------------|
| `tiny`   | 100   | 32 (binary, 4 u64) | 32 | 3.2 KB | 10 KB | 10 KB |
| `small`  | 1 000 | 32 (binary, 4 u64) | 32 | 32 KB  | 100 KB | 110 KB |
| `medium` | 10 000 | 32 (binary, 4 u64) | 32 | 320 KB | 1.0 MB | ~1.1 MB |
| `wide`   | 1 000 | 256 (Tanimoto u8) | 256 | 256 KB | 800 KB | ~1.9 MB |

Total committed test fixture: ~2 MB. Acceptable for a Rust crate.

### 7.3 Where in the repo

`crates/tokenfs-algos/tests/data/hnsw/`

with one subdirectory per corpus:

```
tests/data/hnsw/
  tiny-binary-h32/
    vectors.bin
    config.toml
    expected.bin
    expected.sha256
    README.md   # provenance: which libusearch SHA built this
  small-binary-h32/
    ...
```

`README.md` per corpus records: libusearch git SHA, the seed-patch applied, host architecture (irrelevant for integer metrics but recorded), build date.

### 7.4 What the test asserts

```
let cfg = read_config("tests/data/hnsw/tiny-binary-h32/config.toml");
let vectors = read_vectors("tests/data/hnsw/tiny-binary-h32/vectors.bin");
let mut builder = Builder::new(cfg);
for v in &vectors { builder.try_insert(v.key, &v.bytes)?; }
let bytes = builder.try_finalize()?.into_section_bytes();
let expected = read_bytes("tests/data/hnsw/tiny-binary-h32/expected.bin");
assert_eq!(bytes, expected);
```

A second test re-runs `Builder` and asserts `assert_eq!(bytes_run_a, bytes_run_b)` independent of the fixture; this catches in-process non-determinism even before cross-version.

---

## 8. Rust-specific determinism gotchas

### 8.1 Use `BTreeMap`, not `HashMap`

Anywhere the builder iterates over a key-value map and the iteration order influences output, use `BTreeMap`. Stdlib `HashMap` uses `RandomState` seeded from OS entropy (https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html). Iteration order changes per process. (https://doc.rust-lang.org/std/collections/struct.BTreeMap.html: "iterators obtained from functions such as `BTreeMap::iter` ... produce their items in key order".)

In practice the HNSW builder mostly uses `Vec`-backed adjacency lists; review at PR-time that no `HashMap` is on the output path.

### 8.2 Avoid `DefaultHasher`

`std::collections::hash_map::DefaultHasher` has the same OS-entropy-seeded `RandomState` problem. If any structure in the builder needs hashing (e.g. a "visited set" during graph traversal), use a deterministic hasher — `ahash::AHasher` with a fixed key, or `siphasher::sip128::SipHasher24` with a known key derived from `image_salt`.

The visited-set is a candidate site: usearch uses `growing_hash_set_gt` (`_references/usearch/include/usearch/index.hpp:2216`) which is its own implementation and doesn't depend on stdlib hashing. In Rust we have two choices:
- `Vec<bool>` of length `nodes_count` — trivially deterministic, allocates more, but the cost is acceptable for SLSA-L3 builds.
- `HashSet<Slot, BuildHasherDefault<SipHasher24>>` with a fixed key — small, fast, deterministic.

Pick the `Vec<bool>` for v1 unless profiling shows it dominates.

### 8.3 `sort_unstable` vs `sort`

`Vec::sort` is stable; `Vec::sort_unstable` is faster but does not preserve original order of equal elements. Stable matters only when two elements compare equal under our `Ord`. With the `(distance, node_key, slot)` total order from §3.5, no two elements compare equal — `node_key` and `slot` are unique per node. So either sort is acceptable. Recommend `sort_unstable` for speed, with an `assert!` in debug that no two elements have equal keys.

### 8.4 `rand::SeedableRng` and the recommended deterministic RNGs

The rand book is explicit (https://rust-random.github.io/book/guide-seeding.html, https://rust-random.github.io/book/guide-gen.html):
- `StdRng` is BANNED for reproducibility — its underlying algorithm may change across `rand` versions.
- `rand_chacha::ChaCha8Rng`, `ChaCha12Rng`, `ChaCha20Rng` are tested against reference vectors and stable across versions.
- `rand_xoshiro::Xoshiro256PlusPlus` is similarly stable.

We pin `rand_chacha = "=0.3.x"` (or `=0.9.x` per current MSRV) in `Cargo.toml` with `=` to forbid silent algorithm changes through the lockfile. Cargo `[patch]` overrides for these crates are forbidden in the SLSA-L3 build profile.

### 8.5 `core::sync::atomic` ordering

The HNSW builder is single-threaded in the SLSA-L3 path (§3.1), so atomic orderings on shared memory cannot affect output. But: atomics used as plain counters (e.g. an `AtomicUsize` for slot allocation) MUST be loaded with `Ordering::Relaxed` only — never `SeqCst`-with-side-effects-elsewhere — and the builder MUST NOT spawn threads. If we accidentally introduce parallelism later (e.g. to parallelize distance batch evaluation), `Relaxed` ordering on a shared counter would let two threads observe out-of-order values across runs, breaking determinism.

Rule: if it's an atomic, document why it's atomic and assert single-threaded use in the rustdoc.

### 8.6 Iteration over `&HashSet`, `&HashMap` literals

`HashMap::from([...])` and `HashSet::from([...])` use the same `RandomState`. If anywhere in the builder we have a small literal map for, say, opcode dispatch, use `[(K, V); N]` as a literal slice and iterate, or use `phf::Map` (compile-time perfect hash, deterministic).

### 8.7 `BuildHasherDefault` is your friend

`std::collections::HashMap<K, V, BuildHasherDefault<SipHasher24>>` is allowed — it has a fixed hasher. The non-determinism of `HashMap` lives in `RandomState`, not in the map structure itself.

---

## 9. The cross-arch f32 problem (deep dive)

### 9.1 Why the same algorithm gives different f32 results across CPUs

Three sources, in decreasing order of severity for HNSW:

**FMA fusion.** A dot product `sum = a0*b0 + a1*b1 + ... + an*bn` performed with separate-multiply-add does two roundings per term (round the product, then round the sum). The same dot product performed with `fmadd` does one rounding per term. Different bit patterns. https://www.kdab.com/fma-woes/ documents the divergence with examples. ARMv8 NEON always emits FMA (https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation) — there is no non-FMA path on ARMv8 SIMD. x86-64 only emits FMA when `target-feature=+fma` (https://rust-lang.github.io/packed_simd/perf-guide/target-feature/rustflags.html: "Setting `-C target-feature=+avx2` will not enable fma, even though all CPUs which support AVX2 also support FMA"). So an f32 dot product on aarch64 differs from the same product on x86_64 by default — same source, different bytes.

**SIMD reduction tree shape.** A 256-bit AVX2 register holds 8 f32; reduction is a tree of 8 → 4 → 2 → 1 with three pair-add stages. A 512-bit AVX-512 register holds 16 f32; reduction tree is 16 → 8 → 4 → 2 → 1. Different number of intermediate roundings, different bit patterns. https://acl.inf.ethz.ch/teaching/fastcode/2021/slides/07-simd-avx.pdf covers SIMD reduction patterns.

**Per-instruction rounding modes.** AVX-512 EVEX prefix supports per-instruction rounding control; AVX2 does not. This is a smaller issue in practice because we don't change rounding modes mid-kernel, but it is one more reason a portable f32 result is not free.

### 9.2 What this means for HNSW

The f32 distance kernel produces a slightly different distance value on a different CPU. With ties broken by `(distance, node_key, slot)` (§3.5), a vector that scored exactly `0.250000` on x86 might score `0.249999` on aarch64. The tie-break path is not entered; instead a different candidate wins outright. The graph diverges. The serialized bytes diverge.

This is not a defect-in-implementation; it is a property of f32 + SIMD + cross-architecture portability.

### 9.3 Options

**(a) Constrain SLSA-L3 builds to integer metrics.**
- Pros: zero divergence cost; integer Hamming/Jaccard/Tanimoto are byte-identical across all targets at all SIMD widths because integer addition is associative.
- Cons: loses cosine/L2 for f32-embedded sealed images.
- Precedent: Faiss documents that binary HNSW is reproducible while f32 HNSW is not (because of MKL).

**(b) Software FMA matching one canonical reference.**
- Force the f32 distance kernel to call a portable software implementation of FMA (e.g. `fma::fma_f32` from `core::intrinsics::fmaf32` which on most targets calls libm's `fmaf`).
- libm's `fmaf` is required by C99 to compute the fused multiply-add to infinite precision and then round once.
- Cons: ~5–20x slower than AVX2 fused FMA; still requires we avoid SIMD reductions, so we lose all SIMD speedup; effectively the canonical kernel runs in scalar.
- Pros: fully portable; one set of fixture bytes.

**(c) Document f32 cross-arch variance as part of the threat model.**
- Sealed images built on architecture X are valid only when verified on architecture X (or with an emulator that mimics X's f32).
- The image's `mkfs_arch` field records the architecture; `image_uuid` includes it; verifiers cross-check.
- Cons: a verifier on aarch64 cannot independently verify an x86 image's HNSW section.

### 9.4 Recommendation

For v1, recommend **option (a) with option (b) reserved as opt-in**:

- The canonical SLSA-L3 image-build profile only admits integer metrics for the HNSW section.
- A `BuilderConfig::canonical_kernel: bool` flag selects between SIMD-dispatched and `fma_f32`-only software paths for f32 metrics. When `canonical_kernel = true`, the builder produces cross-arch byte-identical output for f32 at the documented performance cost.
- The image format gains a `hnsw_section_arch: u8` field in v0.4 if we ever ship f32 metrics in canonical SLSA-L3 images without canonical_kernel; until then, f32 sections are flagged "non-canonical" in the image manifest.

Precedent: Debian's reproducible-builds program (https://reproducible-builds.org/docs/definition/) treats architecture as a documented input; their definition includes "the same source code, build environment and build instructions" — different architecture means different build environment, different output is expected. We follow the same precedent for f32.

---

## 10. Recommendations summary

### 10.1 Builder MUST enforce

- Single-threaded insert; `Builder: !Sync`. (§3.1)
- Inputs sorted by `NodeKey` ascending; `debug_assert!` guards each insert. (§3.2)
- RNG is `rand_chacha::ChaCha8Rng` or `rand_xoshiro::Xoshiro256PlusPlus`, version-pinned with `=`. Seeded from `BuilderConfig::level_seed`. (§3.3)
- `BuilderConfig` parameters frozen at `Builder::new`; no mid-build mutation. (§3.6)
- Candidate ordering uses `(distance, node_key, slot)` lex order. (§3.5)
- Final neighbor lists sorted before write. (§3.5)
- No `std::collections::HashMap` or `HashSet` on the output path. (§8.1)
- No `Instant::now`, `SystemTime::now`, `chrono::Utc::now`, `time::OffsetDateTime::now_utc`. (§2.7)
- No `std::is_x86_feature_detected!` runtime dispatch in the SLSA-L3 path; kernel selected at build time. (§3.4, §9)
- No pointer-address values flow into output (no `*const T as usize` writes). (§2.6)
- Padding/reserved bytes in the serialized section MUST be zero. (IMAGE_FORMAT_v0.2.md:757–758)

### 10.2 CI MUST run

- `cargo test --test builder_determinism` on x86_64-unknown-linux-gnu, aarch64-unknown-linux-gnu (qemu or native), aarch64-apple-darwin.
- For each test corpus in `tests/data/hnsw/`, assert byte-identical output between two in-process runs.
- For integer-metric corpora, assert byte-identical output across all three architectures.
- For each test corpus, assert byte-identical output against the committed `expected.bin` fixture.
- Lint: `cargo clippy -- -D clippy::disallowed_types` with a `clippy.toml` that bans `std::collections::HashMap`, `std::collections::HashSet`, `rand::rngs::StdRng`, `rand::rngs::ThreadRng`, `rand::thread_rng` from the HNSW crate.
- Pin `rust-toolchain.toml` to a specific version; CI fails if the local rustc differs.
- `cargo deny` allowlist for `rand_chacha = "=X.Y.Z"` style version pinning.

### 10.3 Caller MUST guarantee

- Input vectors pre-sorted by `NodeKey` ascending. (§3.2)
- Provide an explicit `level_seed: [u8; 32]` derived from the image's `image_salt`, OR accept the crate-default constant and document this in the image manifest.
- Use the integer-metric path (`hamming_u64`, `jaccard_u64`, `tanimoto_u8`) for SLSA-L3 sealed images, OR set `BuilderConfig::canonical_kernel = true` and accept the performance cost.
- Record `connectivity`, `connectivity_base`, `expansion_add`, metric kind, distance dtype, `level_seed` in the image manifest (these are not in the HNSW serialized header; see §5.3 caveat).
- Build with the `slsa-l3` cargo feature enabled (which compiles out `parallel`, `runtime-simd-dispatch`, and other non-deterministic paths at the type level).

---

## Sources

SLSA spec:
- https://slsa.dev/spec/v1.0/levels — Build levels overview
- https://slsa.dev/spec/v1.0/requirements — "Producing artifacts" requirements (Build L3: Provenance Unforgeable, Isolated)
- https://slsa.dev/spec/v1.1/faq — explicit clarification that SLSA does not require reproducible builds

Reproducible builds:
- https://reproducible-builds.org/docs/definition/ — bit-by-bit identical recreation definition
- https://wiki.debian.org/ReproducibleBuilds — Debian implementation
- https://www.kdab.com/fma-woes/ — FMA fusion issues across architectures
- https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation — ARMv8 always-FMA, x86 opt-in
- https://en.wikipedia.org/wiki/FMA_instruction_set — x86 FMA history

Rust:
- https://rust-random.github.io/book/guide-seeding.html — "use a named PRNG ... `rand_xoshiro` or `rand_chacha`"
- https://rust-random.github.io/book/guide-gen.html — types of generators
- https://doc.rust-lang.org/std/collections/struct.BTreeMap.html — deterministic iteration order
- https://doc.rust-lang.org/std/collections/hash_map/struct.RandomState.html — OS-entropy default seeding
- https://internals.rust-lang.org/t/support-turning-off-hashmap-randomness/19234 — discussion of deterministic HashMap
- https://github.com/rust-lang/rust/issues/34902 — Bit-for-bit deterministic / reproducible builds
- https://rust-lang.github.io/packed_simd/perf-guide/target-feature/rustflags.html — `+avx2` does not enable FMA

Other systems:
- https://github.com/nmslib/hnswlib — `random_seed` parameter; multi-thread non-determinism
- https://deepwiki.com/nmslib/hnswlib/2.1-algorithm-parameters — algorithm parameter doc
- https://github.com/facebookresearch/faiss/issues/1073 — non-reproducible HNSW recall report
- https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls — Faiss determinism design statement
- https://github.com/milvus-io/milvus/issues/25908 — Milvus HNSW reproducibility complaint
- https://github.com/pgvector/pgvector — pgvector source (no documented seed mode)
- https://deepwiki.com/qdrant/qdrant/3.2-payload-indexing-and-filtering — qdrant HNSW design

usearch source (in this repo):
- `_references/usearch/include/usearch/index.hpp:1472` — `default_expansion_add`
- `_references/usearch/include/usearch/index.hpp:1990` — `index_serialized_header_t`
- `_references/usearch/include/usearch/index.hpp:2218` — `inverse_log_connectivity` precomputed constant
- `_references/usearch/include/usearch/index.hpp:2227` — `candidate_t::operator<` — distance-only compare
- `_references/usearch/include/usearch/index.hpp:2334` — `level_generator{}` (default-constructed, unseeded)
- `_references/usearch/include/usearch/index.hpp:2945` — `nodes_count_.fetch_add(1)` slot allocation
- `_references/usearch/include/usearch/index.hpp:3412` — `save_to_stream`
- `_references/usearch/include/usearch/index.hpp:4055` — `choose_random_level_`
- `_references/usearch/include/usearch/index_dense.hpp:103` — `expansion_add` config field
- `_references/usearch/include/usearch/index_dense.hpp:716` — `expansion_add()` getter
- `_references/usearch/include/usearch/index_dense.hpp:718` — `change_expansion_add` (BANNED in our model)

TokenFS image format:
- `/home/mjbommar/projects/personal/tokenfs-paper/docs/IMAGE_FORMAT_v0.3.md:713` — §11 v0.3 deltas
- `/home/mjbommar/projects/personal/tokenfs-paper/docs/IMAGE_FORMAT_v0.2.md:745` — §11 base determinism profile
- `/home/mjbommar/projects/personal/tokenfs-paper/docs/IMAGE_FORMAT_v0.2.md:763` — "MinHash / HNSW (when present) MUST seed RNGs from `image_salt`"
- `/home/mjbommar/projects/personal/tokenfs-paper/docs/IMAGE_FORMAT_v0.2.md:765` — "A SLSA Level 3+ build pipeline MUST hold all of the above"

# `vector` module — dense vector distance kernels

**Status:** spec, 2026-05-02. Phase A5 of `01_PHASES.md`.

Promoted from "Tier 3 for HNSW" (`FS_PRIMITIVES_GAP.md`) to **Tier B / Phase A5** here. These are the inner loop of any vector similarity work — MinHash signature distance, F22 fingerprint comparison, brute-force ANN, future learned-embedding similarity. Universal kernel mis-classified as speculative in the prior doc.

## Goal & scope

Five distance metrics × two query shapes × four backends.

**Metrics:**
1. `dot` (inner product)
2. `l2_squared` (squared Euclidean distance)
3. `cosine_similarity` (normalized dot product)
4. `hamming` (XOR popcount, for binary signatures)
5. `jaccard` (popcount-of-AND / popcount-of-OR, for binary signatures)

**Query shapes:**
- **Single-pair**: `f(a: &[T], b: &[T]) -> R`. Two vectors, one number out.
- **Many-vs-one (batch)**: `f_batch_one_to_many(query: &[T], db: &[T], stride: usize, out: &mut [R])`. One query against N database vectors, all the same dimension `stride`. The K-nearest-neighbor inner loop.

**Element types:**
- `f32`: typical for fingerprints, learned embeddings.
- `u32`: counts, binned fingerprints.
- `u64` packed bitvectors (for hamming/jaccard).

## § 1 Module surface

```
similarity/                   // existing
├── mod.rs
├── distance.rs               // existing — runtime dispatch wrappers
├── kernels.rs                // existing — scalar + AVX2 + NEON kernels for f32/u32 dot/l2/cosine
├── kernels_gather.rs         // existing — gather-based MinHash kernels
└── …

vector/                       // NEW
├── mod.rs                    // public API — promotes the kernels to a domain-named module
├── distance.rs               // dispatched single-pair API
├── batch.rs                  // many-vs-one batched API
└── kernels/
    ├── scalar.rs             // (mostly re-exports from similarity::kernels::scalar)
    ├── avx2.rs
    ├── avx512.rs
    └── neon.rs
```

**Question: a new module, or extend `similarity`?** The existing `similarity` module already has dot/L2/cosine for f32/u32. The cleaner factoring is:

- `similarity` continues to own the *content-similarity* primitives (MinHash, SimHash, LSH bands).
- `vector` owns the *general-purpose distance kernels* — the inner-loop building blocks.
- `similarity::kernels` re-exports / depends on `vector::kernels`.

This makes `vector` consumable independently (e.g., for HNSW or for non-similarity uses) without dragging in the MinHash/SimHash machinery.

**Tentative: `vector` is a new module that absorbs the distance kernels. `similarity::kernels::scalar::dot_f32` becomes `vector::scalar::dot_f32` (or just `vector::dot_f32` with backend-suffix variants).** The migration is a rename + re-export with `#[deprecated]` shims for one release.

## § 2 Algorithms per metric

### dot, l2_squared, cosine_similarity (f32)

Standard SIMD reduction. The current `similarity::kernels` already implements these competently for AVX2 + NEON. Improvements to bring forward:

- **AVX-512 FMA** path. AVX-512 FMA gives 16-lane f32 fused-multiply-add at 1 cycle = ~30 GB/s on f32 vectors. Currently no AVX-512 backend in `similarity::kernels`. **Add for v0.2.**
- **Tail handling at the bench-honest level**: the `1e-3` Higham tolerance fix (just landed in `7eb0621`) applies here.
- **Batched many-vs-one**: load query into AVX2 registers once, sweep N database vectors past it, accumulating distances. Beats sequential single-pair calls because the query stays in L1 across all N comparisons.

### dot_u32, l1_u32, l2_squared_u32

Already implemented in `similarity::kernels`. Migration only.

### hamming (u8/u64 packed)

XOR + popcount over packed bits. Two regimes:

- `hamming_u8(a: &[u8], b: &[u8])` — per-byte XOR + popcount, sum.
- `hamming_u64(a: &[u64], b: &[u64])` — packed 64-bit XOR + popcount.

Inner loop is essentially `popcount` (see `bits::popcount`) applied to the XOR. Reuses the popcount kernel.

### jaccard (u64 packed)

`popcount(a AND b) / popcount(a OR b)`. Returns f32 ∈ [0, 1]. Two passes over the input (or one fused pass if SIMD register pressure allows).

For 256-bit MinHash signatures (4 u64s per signature), the per-pair compute is 4 ANDs + 4 ORs + 8 popcounts + 1 division. Negligible per-pair cost; the bottleneck for K-pair Jaccard is the streaming load.

## § 3 API surface

```rust
pub mod vector {
    /// Single-pair dot product.
    pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32>;
    pub fn dot_u32(a: &[u32], b: &[u32]) -> u64;  // wrap semantics; see try_dot_u32
    pub fn try_dot_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>>;

    /// Single-pair L2 squared distance.
    pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32>;
    pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64;

    /// Single-pair cosine similarity.
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32>;
    pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64>;

    /// Single-pair Hamming distance over packed u64 bitvectors.
    pub fn hamming_u64(a: &[u64], b: &[u64]) -> Option<u64>;

    /// Single-pair Jaccard similarity over packed u64 bitvectors.
    pub fn jaccard_u64(a: &[u64], b: &[u64]) -> Option<f64>;

    /// Many-vs-one batched form. `db` is a flat array of N vectors of length
    /// `stride`. Output length must equal N.
    pub fn dot_f32_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]);
    pub fn l2_squared_f32_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]);
    pub fn cosine_similarity_f32_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]);

    pub fn hamming_u64_one_to_many(query: &[u64], db: &[u64], stride: usize, out: &mut [u32]);
    pub fn jaccard_u64_one_to_many(query: &[u64], db: &[u64], stride: usize, out: &mut [f64]);
}
```

## § 4 Hardware acceleration plan

For f32 vectors of length 1024 (typical fingerprint or compact embedding):

| Backend | dot | l2² | cosine | hamming u64 | jaccard u64 |
|---|---|---|---|---|---|
| Scalar | ~3 GB/s | ~3 GB/s | ~3 GB/s | ~3 GB/s | ~2 GB/s |
| AVX2 (8-lane FMA) | ~12-15 GB/s | ~12-15 GB/s | ~10-12 GB/s | ~8 GB/s | ~6 GB/s |
| AVX-512 (16-lane FMA + VPOPCNTQ) | ~25-30 GB/s | ~25-30 GB/s | ~20-25 GB/s | ~30-50 GB/s | ~20-30 GB/s |
| NEON (4-lane FMA) | ~6-8 GB/s | ~6-8 GB/s | ~5-7 GB/s | ~5-7 GB/s | ~4-5 GB/s |

The Hamming/Jaccard rows assume `bits::popcount` is in place; AVX-512 VPOPCNTQ is *the* single largest SIMD win in the whole crate, ~10x over AVX2 software popcount.

## § 5 Reduction-order numerics

The f32 dot/L2/cosine kernels share a numerical hazard: SIMD pairwise tree summation differs from scalar left-to-right by ULP-level rounding, and adversarial cancellation can amplify this to 5-10% relative error against `|dot|`.

**Required tolerance model** (already shipped in `7eb0621`): scale by L1 norm of products `Σ |a_i * b_i|`, not by `|dot|`. Bound at `1e-3` relative to L1 scale catches genuine kernel divergence (which would be >>1e-3) without flaking on cancellation seeds.

**Reduction order convention:**
- 8-way pairwise tree for AVX2 (8 lanes → 4 → 2 → 1).
- 16-way pairwise tree for AVX-512.
- 4-way for NEON.

Document the exact reduction tree in each kernel's `# Safety / # Numerics` block so callers who care about reproducibility know the boundary.

## § 6 Test plan

For each (metric, type, backend):
- **Scalar oracle**: kernel `scalar::*` is the reference.
- **Property test**: random inputs in realistic range (`-256..256` for f32; `0..2^20` for u32 to avoid overflow); 1000 cases.
- **Parity**: SIMD result within tolerance of scalar (Higham L1-norm scale, 1e-3).
- **Edge cases**: empty vector (returns Some(0)/None per type), length 1, length not a multiple of SIMD width, alignment offsets {0, 1, 7, 31}.
- **Length-mismatch handling**: `a.len() != b.len()` returns `None` for f32; `try_*` variants return `None` for u32.

The existing `similarity/tests.rs` already covers this for the migrated kernels. New tests needed for AVX-512 backend.

## § 7 Bench plan

Existing `benches/similarity.rs` covers most of this. Add:

- **AVX-512 backend** rows.
- **Many-vs-one batched** rows: query=1, db ∈ {16, 256, 4K, 64K} vectors of stride=1024. Report aggregate throughput (db vectors / sec).
- **Hamming/Jaccard** rows: signatures of 256, 1024, 4096 bits. Report comparisons / sec.
- **In-L1 vs in-L2 vs in-L3 vs in-DRAM** working-set rows per the `02_CACHE_RESIDENCY.md` cache-tier convention.

## § 8 Open questions

1. **Naming**: the existing `similarity::kernels::scalar::dot_f32` vs the proposed `vector::dot_f32`. The migration is a public-API rename — semver question. **Tentative: hold the rename for v0.2 release boundary; ship both names with `#[deprecated]` on the old one.**

2. **i8 / i16 quantized vectors**: ANN systems quantize embeddings to int8 for memory. Worth supporting? **Tentative: skip in v0.2; add in v0.3 if a learned-embedding consumer asks.** The compute kernel is straightforward (i8 multiply-accumulate via VPMADDUBSW on AVX2 or i8 MMA on AVX-512-VNNI / SVE).

3. **f16 / bf16**: Sapphire Rapids has bf16 FMA; modern Apple Silicon has f16 NEON. **Tentative: defer; high engineering cost, narrow consumer base.**

4. **Cosine in u32 form** — the existing `cosine_similarity_u32` returns `Option<f64>` because the magnitude calc requires sqrt. Worth offering a "no-sqrt" variant returning the raw `dot² / (|a|² × |b|²)` for ranking-only use? **Tentative: not yet; revisit if a consumer asks.**

5. **K-NN top-K helper** that fuses the batched distance computation with a streaming top-K heap update. **Tentative: defer to v0.3 because it crosses with `sort` module decisions; ship the batched distances first as the building block.**

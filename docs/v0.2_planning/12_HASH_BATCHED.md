# `hash` module — batched additions

**Status:** spec, 2026-05-02. Phase A3 + A4 of `01_PHASES.md`.

Two additions to the existing `hash` module:

1. **§ 2 Batched cryptographic hash** — wrappers around `blake3` and `sha2` crates exposing "hash N small inputs in parallel" APIs that consume the BLAKE3 internal SIMD evenly across many messages. Used by Merkle leaves, content-addressed dedup, reproducibility hashes.
2. **§ 3 Hash-set membership SIMD** — "is `x` in this 256-element set of u32s?" via VPCMPEQ broadcast + VPMOVMSKB. Used by vocab lookup, content-class membership, Bloom pre-checks.

Neither is novel work — both are well-understood patterns. The leverage is in shipping them cleanly with the rest of `tokenfs-algos`'s dispatch + parity-test conventions, so consumers don't reach for ad-hoc wrappers.

## § 1 Module surface

Existing `hash` module surface (already shipped):
- `hash::fnv1a_64`
- `hash::mix64`
- `hash::sha256` (incremental hasher with `try_update` / overflow-safe API)
- `hash::blake3` (when `blake3` feature is on)
- `hash::Crc32cHasher` (incremental)

New surface:

```rust
pub mod hash {
    // §2: batched crypto hash
    #[cfg(feature = "blake3")]
    pub fn blake3_batch_32(messages: &[&[u8]], out: &mut [[u8; 32]]);
    #[cfg(feature = "blake3")]
    pub fn blake3_batch_n_par<const N: usize>(messages: &[&[u8]], out: &mut [[u8; N]]);
    pub fn sha256_batch(messages: &[&[u8]], out: &mut [[u8; 32]]);

    // §3: hash-set membership SIMD
    pub fn contains_u32_simd(haystack: &[u32], needle: u32) -> bool;
    pub fn contains_u32_batch_simd(haystack: &[u32], needles: &[u32], out: &mut [bool]);
}
```

## § 2 Batched cryptographic hash

### Why "batched"

`blake3` already SIMD-internalizes hashing for a single large message. But for many small messages — e.g., 200K Merkle-leaf hashes over chunks of ~1 KB each — calling `blake3::hash(msg)` 200K times leaves the internal SIMD parallelism idle, because each individual `hash()` call has a small enough input that BLAKE3 can't fill its 16-way SIMD pipeline.

The win from batching: feed BLAKE3's `Hasher::update` from N parallel input streams, so the internal SIMD packs N messages' chunks into a single SIMD register cycle. For N small messages, this is a 4-8x throughput win over naive per-message hashing.

The Rust `blake3` crate's `Hasher` API doesn't expose this directly. Two implementation options:

**Option A: parallel iteration via rayon.** Distribute message slices across threads; each thread runs `blake3::hash` per message. Memory bandwidth-bound on multi-core; gets near hash-per-thread throughput.

**Option B: explicit SIMD multi-stream hashing using `blake3::guts::ChunkState`.** BLAKE3's internal API exposes the chunk-state machinery; you can construct N parallel chunk states, feed them in lockstep, and finalize each. This is the "true" batched-internal-SIMD path.

**Tentative: Option A for v0.2.** Cleaner API, no internal-API dependency, near-optimal on multi-core. Option B is a v0.3 optimization if profiling shows it pays.

### SHA-256 batched

SHA-256 has no useful internal cross-message SIMD (it's intrinsically serial within a message; SHA-NI accelerates the per-message inner loop). The batched API is just rayon-parallel `sha2::Sha256::digest` calls.

The reason to expose `sha256_batch` is **API consistency** with `blake3_batch`, so consumers don't have to reach for rayon themselves. Also: shipping a batched API future-proofs us for the day we add proper SHA-NI multi-stream support (Intel SHA-NI's `SHA256RNDS2` is single-stream, but at the kernel level you can interleave 4-way streams in SSE registers; the gain is real but ~3-5x, less than BLAKE3).

### API surface

```rust
/// Hash N messages in parallel. Output buffer must have length == messages.len().
/// Internally distributes work via rayon when message count > THRESHOLD; falls
/// back to serial iteration for small batches.
#[cfg(feature = "blake3")]
pub fn blake3_batch_32(messages: &[&[u8]], out: &mut [[u8; 32]]) {
    assert_eq!(messages.len(), out.len(), "batch length mismatch");
    if messages.len() < BATCH_THRESHOLD {
        for (msg, dst) in messages.iter().zip(out.iter_mut()) {
            *dst = *blake3::hash(msg).as_bytes();
        }
    } else {
        use rayon::prelude::*;
        messages.par_iter()
            .zip(out.par_iter_mut())
            .for_each(|(msg, dst)| {
                *dst = *blake3::hash(msg).as_bytes();
            });
    }
}
```

### Performance expectations

For 200K extents averaging 1 KB each (~200 MB total):

| Approach | Expected wall-clock | Notes |
|---|---|---|
| Serial naive `blake3::hash` per extent | ~80 ms | 1 thread, ~3 GB/s |
| `blake3_batch_32` rayon-parallel | ~10-15 ms on 8 cores | ~16-20 GB/s aggregate |
| `sha256_batch` (multi-thread) | ~80 ms | SHA-NI hits ~3-5 GB/s; rayon scales 4-8x on cores |
| `sha256_batch` no SHA-NI | ~400 ms | software SHA-256 is ~600 MB/s/core |

### Test plan

- Parity: `blake3_batch_32(&[msg], &mut out)` matches `blake3::hash(msg)`.
- Property test: arbitrary message sets, batch result equals serial result.
- Edge cases: empty messages, single very-large message in batch, mixed sizes.

### Bench plan

- 200K × 1 KB messages (the canonical Merkle workload).
- 1 × 1 GB message (single-message large): batched API should match the unbatched directly.
- 1M × 64 B messages (very small, batching dominates).

## § 3 Hash-set membership SIMD

### Algorithm

Test whether `needle: u32` appears in `haystack: &[u32]`. For small haystacks (≤ 256 elements typical for vocab tables, content-class membership tables), the SIMD approach is:

```
for each 8-element chunk:
    let chunk = _mm256_loadu_si256(ptr);
    let cmp = _mm256_cmpeq_epi32(chunk, _mm256_set1_epi32(needle));
    let mask = _mm256_movemask_epi8(cmp);
    if mask != 0 { return true; }
```

For batched membership (N needles), broadcast each needle, run the same scan; aggregate masks across needles via OR.

### Throughput

AVX2 hits ~30 GB/s on a typical haystack-resident-in-L1 case. For a 2K-vocab haystack at 8 KB, this is ~250 ns per single membership query — basically L1-load-bound.

### When this is faster than a hashmap

For ≤ ~100 elements: linear SIMD scan beats hashmap because hashmap has 1-2 cache misses per lookup and the haystack fits in L1.

For ≥ ~1000 elements: hashmap O(1) wins.

For 100-1000: linear SIMD scan and hashmap are comparable; cache behavior dominates.

For TokenFS: vocab is ~2K. We ship the SIMD path for ≤ 256 elements (covers content-class membership and small set checks) and let larger consumers reach for `hashbrown` directly.

### API

```rust
/// Returns true if any element of `haystack` equals `needle`. Optimized for
/// short haystacks (≤ 256 elements). For longer haystacks, prefer a hashset.
pub fn contains_u32_simd(haystack: &[u32], needle: u32) -> bool;

/// Batched form: writes `out[i] = haystack.contains(&needles[i])` for each i.
/// `out.len()` must equal `needles.len()`.
pub fn contains_u32_batch_simd(haystack: &[u32], needles: &[u32], out: &mut [bool]);
```

### Hardware acceleration plan

| Backend | Approach | Throughput (haystack=64, needle=1) |
|---|---|---|
| Scalar | linear `iter().any(|&x| x == needle)` | ~12 GB/s (L1) |
| SSE4.1 | PCMPEQD + PMOVMSKB on 4-element chunks | ~25 GB/s |
| AVX2 | VPCMPEQD + VPMOVMSKB on 8-element chunks | ~30 GB/s |
| AVX-512 | VPCMPEQD + KMOVW on 16-element chunks; mask | ~50 GB/s |
| NEON | VCEQ + horizontal-OR | ~15-20 GB/s |

### Test plan

- Property test: `contains_u32_simd(h, n)` matches `h.contains(&n)`.
- Edge cases: empty haystack, single-element haystack, needle at first/last/middle/absent.
- Batched parity.

### Bench plan

- Haystack sizes 16, 64, 256, 1024.
- Per-query latency.
- Compare against `slice::contains`, `HashSet::contains`.

## § 4 Open questions

1. **Where do `xxh3` / `wyhash` SIMD batch wrappers go?** Per `00_BOTTOM_UP_ANALYSIS.md`, no documented bottleneck justifies them yet — current FNV/CRC32C path hasn't shown a problem. **Tentative: defer to `20_DEFERRED.md`.** Trigger condition: a benchmark or a consumer demonstrates the existing hash families are bottlenecking on >5% of total runtime.

2. **Should we add SHA-1 batched?** SHA-1 is broken cryptographically and we don't use it. Skip.

3. **Multihash codec primitives** — `tokenfs-algos::identity` already covers multihash encode/decode. Don't duplicate here.

4. **Whether to expose BLAKE3's internal `ChunkState` directly** — the Rust `blake3` crate's `guts` module is unstable. We'd be coupling to a non-public API. **Tentative: stay on the public API for v0.2.**

5. **AArch64 SHA-2 extension** (`FEAT_SHA2`): the `sha2` crate already uses these where available. We don't need to wrap it; `sha256_batch` rayon-parallel just composes correctly.

## § 5 Environment fitness — IMPORTANT REVISION

Per [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md), the original `blake3_batch` spec used `rayon` which is forbidden in kernel modules. **Revised API:**

```rust
// Single-thread, internal-multi-stream-SIMD via blake3::guts machinery.
// Kernel-safe (modulo blake3 needing std — see notes).
#[cfg(feature = "blake3")]
pub fn blake3_batch_st_32(messages: &[&[u8]], out: &mut [[u8; 32]]);

// Rayon-parallel convenience wrapper. Userspace only.
#[cfg(all(feature = "blake3", feature = "parallel"))]
pub fn blake3_batch_par_32(messages: &[&[u8]], out: &mut [[u8; 32]]);

// Same pattern for SHA-256.
pub fn sha256_batch_st(messages: &[&[u8]], out: &mut [[u8; 32]]);
#[cfg(feature = "parallel")]
pub fn sha256_batch_par(messages: &[&[u8]], out: &mut [[u8; 32]]);
```

| API | Kernel module | FUSE | Userspace | Postgres ext | cgo (Go) | Python (PyO3) |
|---|---|---|---|---|---|---|
| `blake3_batch_st_32` | ❌ blake3 needs std | ✅ | ✅ | ✅ | ✅ | ✅ |
| `blake3_batch_par_32` | ❌ rayon | ✅ | ✅ | ⚠️ multi-thread within bg worker only | ✅ | ✅ |
| `sha256_batch_st` | ✅ `sha2` is no_std-clean | ✅ | ✅ | ✅ | ✅ | ✅ |
| `sha256_batch_par` | ❌ rayon | ✅ | ✅ | ✅ | ✅ | ✅ |
| `contains_u32_simd` | ✅ | ✅ | ✅ | ✅ | ⚠️ batch | ✅ |
| `contains_u32_batch_simd` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Critical reframings:**

1. **`blake3` requires std** — per `Cargo.toml`'s `blake3 = ["dep:blake3", "std"]`, the blake3 feature gate already implies std. **The kernel content-addressing answer is `sha256_batch_st` only.** The `sha256_batch_st` variant is no_std + alloc + kernel-safe via the `sha2` crate's no_std support and ARMv8 SHA-2 acceleration via `cpufeatures`.

2. **Single-thread vs parallel naming convention** — `_st` suffix for single-thread (kernel-safe), `_par` suffix for rayon-parallel (userspace). The default function name (no suffix) is reserved for the highest-quality single-thread variant, so kernel consumers pick `blake3_batch_32` and get the safe path.

3. **Per-element cgo cost** — Go calling `contains_u32_simd` for one needle has ~200 ns cgo overhead. Always use `contains_u32_batch_simd` from cgo. Document this clearly.

4. **Postgres parallel-query mechanism** — Postgres bg workers parallelize at the query level; the SIMD primitives don't need to do their own rayon. **Use `_st` variants in Postgres extensions** unless the extension explicitly fans out across cores in user code.

**Verification action**: extend `xtask security` to assert that the no_std + alloc lib build doesn't reach for `blake3` (already in forbidden list) and that no_default + alloc + std (no parallel) compiles every public API.

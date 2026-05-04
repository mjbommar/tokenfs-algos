# HNSW Primitive Inventory

> Status: research note, 2026-05-03. Inventories what already lives in
> `tokenfs-algos` so the `similarity::hnsw` walker + builder design only
> writes what's actually missing. Companion to `docs/HNSW_PATH_DECISION.md`.

**Convention used below.**

- Path citations are relative to `crates/tokenfs-algos/src/`.
- "Kernel-safe?" answers two questions: (1) is the panicking entry behind `userspace`/`panicking-shape-apis`, and (2) is the no-panic `try_*` entry reachable in `--no-default-features --features alloc`?
- Reuse labels:
  - **Direct reuse** — the HNSW walker / builder calls the existing function unchanged.
  - **Pattern reuse** — the *shape* is right; HNSW writes a sibling adapted to the HNSW data layout.
  - **Inspiration** — useful prior art only; HNSW will own its own variant.

---

## §1. `vector::*` — dense distance kernels (HIGHEST PRIORITY)

This is the load-bearing module. Module entry: `vector/mod.rs:1`. The crate-level docstring (`vector/mod.rs:1-44`) names this as the canonical "two vectors → one number" surface and pins the cross-backend reduction-order tolerance at `1e-3 * sum(|a*b|)`. The whole module is documented as "kernel-safe across the board" (`vector/mod.rs:42`).

### 1.1 Single-pair entries (`vector/distance.rs`)

| Function | Signature | Site |
|---|---|---|
| `dot_f32` | `&[f32], &[f32] -> Option<f32>` | `vector/distance.rs:22` |
| `dot_u32` | `&[u32], &[u32] -> u64` (wraps on overflow) | `vector/distance.rs:38` |
| `try_dot_u32` | `&[u32], &[u32] -> Option<Option<u64>>` | `vector/distance.rs:47` |
| `l2_squared_f32` | `&[f32], &[f32] -> Option<f32>` | `vector/distance.rs:53` |
| `l2_squared_u32` | `&[u32], &[u32] -> u64` | `vector/distance.rs:61` |
| `cosine_similarity_f32` | `&[f32], &[f32] -> Option<f32>` | `vector/distance.rs:70` |
| `cosine_similarity_u32` | `&[u32], &[u32] -> Option<f64>` | `vector/distance.rs:80` |
| `hamming_u64` | `&[u64], &[u64] -> Option<u64>` | `vector/distance.rs:93` |
| `jaccard_u64` | `&[u64], &[u64] -> Option<f64>` | `vector/distance.rs:103` |

These are thin shims over `vector::kernels::auto::*` (`vector/distance.rs:12`). Length mismatch → `None`; no panic on inputs.

### 1.2 Batched many-vs-one entries (`vector/batch.rs`)

| Function | Signature | Site |
|---|---|---|
| `try_dot_f32_one_to_many` | `(query, db, stride, out) -> Result<(), BatchShapeError>` | `vector/batch.rs:148` |
| `try_l2_squared_f32_one_to_many` | same shape | `vector/batch.rs:195` |
| `try_cosine_similarity_f32_one_to_many` | same shape | `vector/batch.rs:271` |
| `try_hamming_u64_one_to_many` | `(query, db, stride, out: &mut [u32])` | `vector/batch.rs:346` |
| `try_jaccard_u64_one_to_many` | `(query, db, stride, out: &mut [f64])` | `vector/batch.rs:395` |
| panicking siblings | `dot_f32_one_to_many`, etc., `#[cfg(feature = "panicking-shape-apis")]` | `vector/batch.rs:127`, `:173`, `:225`, `:300`, `:373` |

Error type: `BatchShapeError` enum at `vector/batch.rs:46`. The `try_*` variants return `Err` on `out.len() * stride != db.len()` instead of panicking.

### 1.3 Pinned per-backend kernels (`vector/kernels/`)

Backend availability is feature-gated; visibility is gated on `arch-pinned-kernels` (per-backend modules go from `pub(crate)` to `pub`). See `vector/kernels/mod.rs:20-66` for the cfg matrix.

#### Scalar (`vector/kernels/scalar.rs`)
Reference oracle, always present, strictly left-to-right reduction (`vector/kernels/scalar.rs:8-12`).

| Function | Site |
|---|---|
| `dot_u32` | `:38` |
| `try_dot_u32` | `:55` |
| `l1_u32` | `:80` |
| `try_l1_u32` | `:96` |
| `l2_squared_u32` | `:120` |
| `try_l2_squared_u32` | `:137` |
| `cosine_similarity_u32` | `:161` |
| `dot_f32` | `:189` |
| `l2_squared_f32` | `:208` |
| `cosine_similarity_f32` | `:225` |
| `hamming_u64` | `:248` |
| `jaccard_u64` | `:265` |

#### AVX2 (`vector/kernels/avx2.rs`, `feature = "avx2"`)

| Function | Site |
|---|---|
| `is_available` | `:50` |
| `dot_u32` (`pub unsafe fn`) | `:125` |
| `l1_u32` | `:150` |
| `l2_squared_u32` | `:185` |
| `cosine_similarity_u32` | `:214` |
| `dot_f32` (8-way pairwise) | `:243` |
| `l2_squared_f32` | `:301` |
| `cosine_similarity_f32` | `:326` |
| `hamming_u64` | `:350` |
| `jaccard_u64` | `:436` |

#### AVX-512 (`vector/kernels/avx512.rs`, `feature = "avx512"`, nightly)

| Function | Site |
|---|---|
| `is_available` (AVX512F) | `:59` |
| `is_popcnt_available` (AVX512F + VPOPCNTDQ) | `:74` |
| `dot_f32` (16-way pairwise, FMA) | `:125` |
| `l2_squared_f32` | `:151` |
| `cosine_similarity_f32` | `:176` |
| `hamming_u64` (VPOPCNTQ) | `:203` |
| `jaccard_u64` (VPOPCNTQ) | `:268` |

Note: AVX-512 ships float kernels + binary popcount kernels only. Integer (`u32`) kernels stop at AVX2 — `vector/kernels/mod.rs:73-99` documents the rationale ("AVX2 already saturates the integer multiply-accumulate units").

#### NEON (`vector/kernels/neon.rs`, `feature = "neon"`)

| Function | Site |
|---|---|
| `is_available` (always true on AArch64) | `:37` |
| `dot_u32` | `:49` |
| `l1_u32` | `:78` |
| `l2_squared_u32` | `:110` |
| `cosine_similarity_u32` | `:139` |
| `dot_f32` (4-way pairwise via `vaddvq_f32`) | `:164` |
| `l2_squared_f32` | `:189` |
| `cosine_similarity_f32` | `:213` |
| `hamming_u64` (VCNT) | `:237` |
| `jaccard_u64` | `:289` |

#### Runtime dispatcher (`vector/kernels/mod.rs::auto`)

`vector/kernels/mod.rs:69-388` — one `pub fn` per (metric × scalar) pair. Each does:
1. Length check → `None` on mismatch (caller-input validation, not panic).
2. Try AVX-512 → AVX2 → NEON → scalar in availability order, gated by `cfg(feature = "...")` blocks.
3. Hot dispatch resolves once per call; the chosen `_unchecked` (well, here the `pub unsafe fn` siblings — these kernels predate the `_unchecked`-suffix convention) runs over the whole slice.

### 1.4 HNSW backend coverage matrix

Mapping the existing kernels onto the HNSW v1 metric table (`docs/HNSW_PATH_DECISION.md` §10):

| HNSW metric | scalar | AVX2 | AVX-512 | NEON | SSE4.1 | SSSE3 |
|---|---|---|---|---|---|---|
| L2² f32 | yes | yes | yes (FMA) | yes | gap | gap |
| L2² i8 / u8 | gap (only u32) | gap | gap | gap | gap | gap |
| cosine f32 | yes | yes | yes (FMA) | yes | gap | gap |
| cosine i8 / u8 | gap (only u32) | gap | gap | gap | gap | gap |
| dot f32 | yes | yes | yes (FMA) | yes | gap | gap |
| Hamming binary | yes (`hamming_u64`) | yes | yes (VPOPCNTQ) | yes (VCNT) | gap (no dedicated SSE4.1 kernel) | gap |
| Jaccard binary | yes | yes | yes | yes | gap | gap |
| Tanimoto binary | gap (would compose `popcount(a&b) / popcount(a|b)`) | gap | gap | gap | gap | gap |

**Gaps the HNSW kernels module needs to fill.** All `i8` / `u8` paths (the F22 fingerprint path) — there is no `dot_i8`, `l2_squared_i8`, `cosine_similarity_i8`, etc. anywhere. Tanimoto is one composition step beyond Jaccard but doesn't have a dedicated kernel today. SSE4.1 / SSSE3 fallbacks are not present in `vector/kernels/`. The f32 / binary paths already exist on every backend the HNSW table requires.

**Reuse posture.**

- **Direct reuse** for the f32 path (L2², cosine, dot) — the walker calls `vector::auto::*` for f32 metrics across scalar / AVX2 / AVX-512 / NEON without modification.
- **Direct reuse** for the binary path — `hamming_u64` and `jaccard_u64` over `&[u64]` packed bitvectors land Hamming and Jaccard for free on all four backends.
- **Pattern reuse** for the missing `i8` / `u8` kernels — the HNSW kernel files copy the AVX2 / NEON inner-loop shape from `vector/kernels/avx2.rs::dot_u32` (`:125`) / `::l1_u32` (`:150`) but reduce into a tighter accumulator (i32 or u16) per the F22 byte-quantized regime.

---

## §2. `bits::popcount::*` — popcount kernels

Module: `bits/popcount.rs`. Module docstring at `bits/popcount.rs:1-14`.

### 2.1 Top-level entries

| Function | Signature | Site |
|---|---|---|
| `popcount_u64_slice` | `&[u64] -> u64` | `bits/popcount.rs:24` |
| `popcount_u8_slice` | `&[u8] -> u64` | `bits/popcount.rs:33` |

Both are runtime-dispatched, no panic on input, kernel-safe (no `userspace` gate).

### 2.2 Per-backend kernels

| Backend | `popcount_u64_slice` | `popcount_u8_slice` | `is_available` |
|---|---|---|---|
| scalar | `bits/popcount/kernels/scalar.rs:3` | `:19` | always |
| AVX2 (Mula nibble-LUT) | `kernels/avx2.rs:46` (`pub unsafe fn`) | `:68` | `:28` |
| AVX-512 (VPOPCNTQ) | `kernels/avx512.rs:50` (target_feature `avx512f,avx512vpopcntdq`) | `:69` | `:31` |
| NEON (VCNT + horizontal add) | `kernels/neon.rs:32` | `:48` | `:21` |

### 2.3 Dispatcher

`bits/popcount.rs:38-115` — `kernels::auto::popcount_u64_slice` / `popcount_u8_slice`. Same shape as `vector::kernels::auto::*`: `cfg`-gated `is_available()` checks in AVX-512 → AVX2 → NEON → scalar order.

### 2.4 HNSW reuse

- **Direct reuse** when binary metrics (Hamming, Jaccard, Tanimoto) operate over `&[u64]` packed bitvectors. `vector::hamming_u64` and `jaccard_u64` already compose with `bits::popcount` (`vector/distance.rs:90` calls this out explicitly); the HNSW walker can call `vector::*` and inherit the popcount win transitively.
- **Direct reuse** for visited-bitset cardinality reporting. If the walker tracks `visited` as a `&[u64]` bitset (see §5), `bits::popcount_u64_slice` answers "how many candidates examined?" on every backend.
- **Pattern reuse** for the binary-vector inner loop in HNSW's binary distance kernels: AVX-512's VPOPCNTQ kernel structure (`bits/popcount/kernels/avx512.rs:42-58`) is exactly what we want to lift into `similarity::hnsw::kernels::avx512` for binary L2² (which equals Hamming on binary).

---

## §3. `bitmap::*` — Roaring container kernels (FILTER PRIMITIVES)

Module: `bitmap/mod.rs`. Spec: `bitmap/mod.rs:1-86`. The module ships container-level kernels (Schlegel intersect, AVX-512 AND + VPOPCNTQ, VPCOMPRESSD output) and the three Roaring container shapes; it explicitly does **not** ship a high-level `Bitmap` type — that's `roaring-rs`'s job (`bitmap/mod.rs:8-15`).

### 3.1 Container types (`bitmap/containers.rs`)

| Type | Site | Memory |
|---|---|---|
| `BitmapContainer` (1024 × u64 = 8 KiB) | `containers.rs:141` | dense |
| `ArrayContainer` (sorted u16, ≤ 4096) | `containers.rs:315` | sparse |
| `RunContainer` (sorted (start, len-1) pairs) | `containers.rs:462` | run-encoded |
| `Container` enum (dispatches over the three) | `containers.rs:423` | — |
| `BitmapIter` (set-bit iteration in ascending order) | `containers.rs:257` | — |

Validating constructors: `ArrayContainer::try_from_vec` (`:363`), `RunContainer::try_from_vec` (~`:489`). Direct field writes are `pub(crate)`-only.

Container constants: `BITMAP_WORDS = 1024` (`:117`), `BITMAP_BITS = 65536` (`:120`), `ARRAY_MAX_CARDINALITY = 4096` (`:128`).

### 3.2 Pairwise operations on the `Container` enum

| Method | Site | What |
|---|---|---|
| `intersect` | `bitmap/mod.rs:104` (delegates to `intersect.rs:32`) | a ∩ b |
| `union` | `mod.rs:110` (`union.rs:17`) | a ∪ b |
| `difference` | `mod.rs:116` (`difference.rs:13`) | a ∖ b |
| `symmetric_difference` | `mod.rs:122` (`xor.rs:13`) | a △ b |
| `cardinality` | `mod.rs:128` (`cardinality.rs:16`) | \|a\| |
| `intersect_cardinality` | `mod.rs:138` (`intersect.rs:48`) | \|a ∩ b\| (no materialization) |

### 3.3 BitmapContainer fast paths (CRoaring `_card` / `_nocard` / `_justcard` discipline)

Three variants per op (`bitmap/mod.rs:60-86`):

| Operation | `_into` (materialize + return card) | `_into_nocard` (materialize only) | `_cardinality` (just count) |
|---|---|---|---|
| AND | `mod.rs:151` | `:156` | `:162` |
| OR | `:167` | `:172` | `:178` |
| XOR | `:183` | `:188` | `:194` |
| ANDNOT | `:199` | `:204` | `:210` |

All three variants on every operation route through the `kernels_dispatch` module's macros (`bitmap/mod.rs:220-439`) which fan out to AVX-512 → AVX2 → NEON → scalar.

### 3.4 Per-backend kernels

| Backend | Kernel file | Site (`is_available`) |
|---|---|---|
| scalar bitmap×bitmap | `bitmap/kernels/bitmap_x_bitmap_scalar.rs` | always |
| AVX2 bitmap×bitmap | `bitmap_x_bitmap_avx2.rs:40` | `:40` |
| AVX-512 bitmap×bitmap (VPOPCNTQ) | `bitmap_x_bitmap_avx512.rs:37` | `:37` |
| NEON bitmap×bitmap | `bitmap_x_bitmap_neon.rs` | always on AArch64 |
| scalar array×array | `array_x_array_scalar.rs` | always |
| SSE4.2 array×array | `array_x_array_sse42.rs` | (Schlegel intersect) |
| scalar array×bitmap | `array_x_bitmap_scalar.rs` | always |
| AVX2 array×bitmap | `array_x_bitmap_avx2.rs` | (`is_available` per-file) |

`BitmapContainer::cardinality` (`containers.rs:190`) routes through `bits::popcount_u64_slice` so cardinality counts inherit VPOPCNTQ / VCNT for free.

### 3.5 HNSW reuse

- **Direct reuse** for the filter primitive. The HNSW walker takes `Option<&Container>` (or some bag of containers, depending on cardinality regime) of permitted node IDs; "is this candidate allowed?" composes with `Container`'s `contains` / `intersect_cardinality`. The "in-search vs. post-filter" requirement in `HNSW_PATH_DECISION.md` §1 is satisfied by composing per-candidate membership tests with `BitmapContainer::contains` (`containers.rs:163`) inside the beam-search loop.
- **Direct reuse** for cluster-AND / cluster-OR rule composition (consumers building the filter from "files in cluster X" + "files tagged Y" upstream).
- **Pattern reuse** for the visited bitset (§5). The HNSW visited-set is logically a single `BitmapContainer` (or sparse equivalent for small searches); we don't pull in the full Roaring container choice machine — but the 1024-word layout and `BitmapIter` shape transfer directly to the HNSW visited-tracking module.

---

## §4. `hash::set_membership::*` — small-set membership

Module: `hash/set_membership.rs`. Module docstring at `:1-23`.

> Optimized for short haystacks (≤ 256 elements) — vocab tables, content-class membership tables, Bloom pre-checks.

### 4.1 Top-level entries

| Function | Signature | Site |
|---|---|---|
| `contains_u32_simd` | `&[u32], u32 -> bool` | `hash/set_membership.rs:67` |
| `try_contains_u32_batch_simd` | `(haystack, needles, &mut [bool]) -> Result<(), SetMembershipBatchError>` | `:100` |
| `contains_u32_batch_simd` | panicking sibling, `#[cfg(feature = "panicking-shape-apis")]` | `:86` |
| `SetMembershipBatchError` enum | `LengthMismatch` only | `:31` |

### 4.2 Per-backend kernels

| Backend | `contains_u32` | `_batch_unchecked` | `is_available` |
|---|---|---|---|
| scalar | `set_membership/kernels/scalar.rs:7` | `:29` | always |
| SSE4.1 (PCMPEQD + PTEST) | `kernels/sse41.rs` | (per-file) | (per-file) |
| AVX2 (VPCMPEQD + VPTEST) | `kernels/avx2.rs:43` | `:125` | `:25` |
| AVX-512 (VPCMPEQD-mask + KORTESTW) | `kernels/avx512.rs:34` | `:77` | `:16` |
| NEON (VCEQQ + horizontal max) | `kernels/neon.rs` | (per-file) | (per-file) |

### 4.3 Dispatcher

`hash/set_membership.rs:121-257`. The module is one of the canonical examples of the `_unchecked` sibling pattern (`set_membership.rs:188-256`):
- `auto::contains_u32_batch` (gated on `userspace`) asserts then calls...
- `auto::contains_u32_batch_unchecked` (always available) does runtime dispatch then calls per-backend `_unchecked`.

The `try_*` top-level entry validates lengths (`:104-110`) and then calls `auto::*_unchecked` directly, never touching the asserting code path even in kernel builds.

### 4.4 HNSW reuse

This is the single most important "is this primitive a fit for visited-set tracking?" question in the inventory. Tradeoffs:

| Backing | Insert | Lookup | Memory at N visited | Iteration cost |
|---|---|---|---|---|
| `hash::set_membership` over a `Vec<u32>` of visited IDs | O(1) push | O(N) linear (SIMD broadcast-compare) | 4N bytes | grows linearly per query |
| Bitset (`Vec<u64>` indexed by node ID) | O(1) bit-set | O(1) bit-test | N_total / 8 bytes | constant per query (size of graph) |

For HNSW with N_visited ≪ N_total (typical: efSearch ≈ 200 ≪ 10⁶ nodes), the linear-scan break-even of `set_membership` (≤ 256-element haystack per its docstring) sits right around HNSW's typical visited-set size at ef=200. But the HNSW visited-set must be reset per query. With a bitset, "reset" is `O(N_total)` zero-fill; with `Vec<u32>` + `set_membership`, "reset" is `vec.clear()` — O(N_visited). The bitset wins on per-step lookup latency but loses on per-query reset for large graphs unless we use the "version-stamp" trick (compare against a per-query epoch instead of zeroing).

**Posture.** **Inspiration**, not direct reuse. The HNSW visited-set should likely be a hybrid: bitset for the top-layer beam (small, frequently consulted) and the version-stamp pattern for the bottom-layer walk. `hash::set_membership` is the right primitive *if* we choose to keep the visited-set as a sorted-then-scanned `Vec<u32>` for very small ef; otherwise we should compose with `bits::rank_select` (§5) for the bitset path.

---

## §5. `bits::rank_select::*` — bitset rank/select

Module: `bits/rank_select.rs`. Spec: `bits/rank_select.rs:1-60`.

### 5.1 Top-level type and constants

| Item | Site |
|---|---|
| `RankSelectDict<'a>` | `bits/rank_select.rs:211` |
| `RankSelectError` enum (`BitsTooShort`, `PositionOutOfRange`) | `:78` |
| `BLOCK_BITS = 256` | `:177` |
| `SUPERBLOCK_BITS = 4096` | `:183` |
| `WORDS_PER_BLOCK = 4` | `:186` |
| `WORDS_PER_SUPERBLOCK = 64` | `:189` |
| `BLOCKS_PER_SUPERBLOCK = 16` | `:192` |

### 5.2 Public methods (kernel-safe try_* discipline applied)

| Method | Site | Kernel-safe? |
|---|---|---|
| `try_build` | `:258` | yes (always available) |
| `build` (`#[cfg(feature = "panicking-shape-apis")]`) | `:250` | userspace-gated |
| `len_bits` | `:327` | const fn |
| `count_ones` | `:333` | const fn |
| `try_rank1` | `:368` | yes |
| `rank1` (userspace-gated) | `:352` | gated |
| `try_rank0` / `rank0` | `:445`, `:429` | rank0 gated |
| `select1` | `:459` | always (returns `Option`) |
| `select0` | `:469` | always (returns `Option`) |
| `memory_bytes` | `:487` | always |
| `try_rank1_batch` | `:536` | yes (validates `out.len() == positions.len()`) |
| `rank1_batch` (userspace-gated) | `:512` | gated |
| `try_select1_batch` | `:595` | yes |
| `select1_batch` (gated) | `:575` | gated |

### 5.3 Free helper functions

| Function | Site | What |
|---|---|---|
| `select_in_word` | `:762` | runtime-dispatched select-in-64-bit-word |
| `select_in_word_broadword` | `:829` | Vigna broadword variant |
| `kernels` submodule (auto / scalar) | `:924`, `:993` | dispatcher + scalar reference |

### 5.4 HNSW reuse

- **Direct reuse** as the visited-set backing if we pick the bitset path. The HNSW walker would carry `&mut [u64]` of `n_bits = graph.n_nodes`; per-step `is_visited` test is a single bit-load (no need to construct a `RankSelectDict` for that). However, if the walker also wants `rank` queries (e.g. "how many candidates of ID < x have been visited?", useful for some priority-queue variants), `RankSelectDict::try_rank1` lands O(1) rank.
- **Direct reuse** for the `select_in_word` broadword routine (`:829`) if HNSW ever needs to enumerate set bits of the visited mask without iterating linearly.
- **Pattern reuse** for the two-level sampling layout (superblock + block + per-word popcount) — useful conceptually if HNSW ever ships a static per-graph "permitted nodes" filter that needs O(1) `is_member` + O(1) rank.

---

## §6. `approx::*` — Misra-Gries, Bloom, HLL

Module: `approx.rs` (~1620 LOC, all in one file). Top-level docstring at `:1-30`.

### 6.1 Misra-Gries / SpaceSaving

The two implementations:

| Type | Site | Items |
|---|---|---|
| `SpaceSaving<const K>` (in `approx.rs`) | `approx.rs:194` | `u64` items, `[(u64,u32); K]` snapshot |
| `MisraGries<const K>` (in `histogram/topk.rs`) | `histogram/topk.rs:56` | `u8` (byte) items, `[u8;K] + [u64;K]` |
| `MisraGries` (in `sketch.rs`) | `sketch.rs:116` | `u32` items |

All three are bounded `[T; K]`-sized const-generic structures that *count* heavy hitters; none of them is shaped like the HNSW candidate min-heap (top-K *by score*, not top-K *by frequency*).

**HNSW reuse.** **Inspiration only.** The HNSW candidate min-heap stores `(distance, node_id)` pairs with a fixed cap `ef` (or `k` for the top-result heap) and pops the *worst* on overflow. None of the three Misra-Gries variants do that. The right shape is closer to `std::collections::BinaryHeap<Reverse<(distance, id)>>` with an explicit cap — exactly what `permutation/rabbit.rs:369` already uses. See §8.

### 6.2 Bloom (`approx::BloomFilter`)

Re-exported as `approx::BloomFilter` (`approx.rs:796`).

| Method | Site | Kernel-safe? |
|---|---|---|
| `BloomFilter::new` | `:381` | userspace-gated |
| `BloomFilter::try_new` | `:435` | yes |
| `BloomFilter::with_target` | `:411` | userspace-gated |
| `BloomFilter::try_with_target` | `:467` | yes |
| `insert` (`&[u8]`) | `:541` | always |
| `contains` (`&[u8]`) | `:555` | always |
| `insert_simd` (`u64`) | `:624` | userspace-gated |
| `try_insert_simd` (`u64`) | `:648` | yes |
| `contains_simd` (`u64`) | `:690` | userspace-gated |
| `try_contains_simd` (`u64`) | `:715` | yes |
| `contains_batch_simd` | `:743` | userspace-gated |
| `try_contains_batch_simd` | `:768` | yes |

The Bloom kernels (`approx::bloom_kernels`) at `approx.rs:821` use the canonical `_unchecked` dispatcher pattern — see §10.

**HNSW reuse.** **Inspiration only.** Bloom is the wrong primitive for the visited-set (false positives → walker drops valid candidates); use a bitset (§5) or sorted scan (§4) instead. Bloom *might* be useful for a probabilistic "have I seen a vector with this fingerprint before?" early-out in the builder, but that's a v0.8+ concern.

### 6.3 HyperLogLog (`approx::HyperLogLog`, also `approx::hll`)

Public surface at `approx.rs:1157`. Kernels at `approx::hll::kernels` (auto / scalar / avx2 / avx512 / neon).

| Method | Site |
|---|---|
| `HyperLogLog::new` | `:1175` (userspace-gated) |
| `HyperLogLog::try_new` | `:1194` |
| `insert` | `:1240` |
| `insert_hash` | `:1245` |
| `count_simd` | `:1297` |
| `merge` / `try_merge` | `:1340`, `:1353` |
| `merge_simd` / `try_merge_simd` (SIMD register max) | `:1381`, `:1394` |

Per-backend `merge` kernels: scalar (`approx/hll/kernels/scalar.rs:8`), AVX2 (`avx2.rs:40`), AVX-512 (`avx512.rs:43`), NEON (`neon.rs:24`). All `pub unsafe fn` with `is_available`.

**HNSW reuse.** **Inspiration only** for the HLL primitive itself (HNSW doesn't estimate cardinalities). But the **HLL merge SIMD pattern** (per-byte register-wise `max` via `_mm512_max_epu8` / `vmaxq_u8`) is the same SIMD shape we'd use for any per-byte SIMD reduction the HNSW walker might want — the HLL kernels are good reference for "byte-wise max across two 16K-register tables" if we ever need it.

---

## §7. `dispatch::planner::*` — runtime backend selection (rules-as-data)

Module: `dispatch/planner/mod.rs`. Architecture explained at `:1-46`.

### 7.1 Architecture

The planner is rules-as-data: every decision is a small `Rule` (`dispatch/planner/rule.rs:24`) declared as a `pub(crate) const` item, registered in source order in `rules.rs`'s `RULES` registry, and matched in priority order.

```rust
pub struct Rule {
    pub name: &'static str,
    pub reason: &'static str,
    pub source: PlannerConfidenceSource,
    pub predicate: for<'t> fn(&ProcessorProfile, &WorkloadShape, &Signals<'t>) -> bool,
    pub builder: for<'t> fn(&ProcessorProfile, &WorkloadShape, &Signals<'t>) -> HistogramPlan,
}
```

(`dispatch/planner/rule.rs:24-45`)

### 7.2 Inputs / outputs

| Type | Site | Role |
|---|---|---|
| `ProcessorProfile` | `dispatch/mod.rs:118` | host capability snapshot (backend, cache, logical CPUs, accelerators) |
| `Backend` enum | `dispatch/mod.rs:14` | scalar / AVX2 / AVX-512 / NEON / SVE / SVE2 |
| `CacheProfile` | `dispatch/mod.rs:46` | L1/L2/L3 sizes |
| `WorkloadShape` (= `PlanContext`) | `dispatch/mod.rs:543`, `:569` | per-call workload predicates |
| `Signals<'tunes>` | `dispatch/planner/signals.rs:25` | derived predicates: `random_like`, `sequential_like`, `mixedish_entropy`, etc. |
| `Tunes` | `dispatch/planner/tunes.rs` | overridable numeric thresholds |
| `HistogramPlan` (today's only output) | `dispatch/mod.rs:915` | strategy + chunk + sample sizes + confidence |
| `PlannerConfidenceSource` | `dispatch/mod.rs:932` | provenance of the confidence value |

### 7.3 Backend / kernel coverage tables

`dispatch/mod.rs:238-393` — `BackendKernelSupport` per `Backend`, including `similarity` field (`:252`):

> Similarity f32 distance kernels (dot, L2², cosine).

Today AVX2 / NEON / AVX-512 all report `similarity: KernelAvailability::Native` (see `:294`, `:315`, ~`:335` for AVX-512). HNSW will broaden the meaning of this row (it's now distance kernels for *all* metric × scalar combos, not just f32).

### 7.4 HNSW integration

The planner itself is currently histogram-specific (`plan_histogram`, `dispatch/mod.rs:959`). HNSW does not need a planner output until walker performance regressions show up; the existing rules-as-data architecture is reusable verbatim once the walker has tuneable knobs (e.g. "switch to scalar if vector_dim < 16" or "prefer SSE4.1 over AVX2 below N=10⁴ candidates").

**Reuse posture.**

- **Direct reuse** of `Backend`, `ProcessorProfile`, `CacheProfile`, the `is_*_available()` runtime probes — HNSW dispatchers call these to pick a backend.
- **Pattern reuse** for any future HNSW signal class. When we want a "plan an HNSW search" rule (efSearch sizing per cache profile, or "use binary metric path even if user picked f32 because the data fits"), we add a new `RULES` registry in `dispatch/planner/rules.rs` and a `Rule` per decision — no architectural change needed.
- **Pattern reuse** of the `BackendKernelSupport::similarity` field convention. HNSW will likely add an `hnsw_walker: KernelAvailability` field to `BackendKernelSupport` once the walker lands, mirroring the existing pattern.

---

## §8. `permutation::*` — graph traversal patterns (Rabbit Order)

Module: `permutation/mod.rs`. Top-level types and the `CsrGraph` view at `:858-993`.

### 8.1 CSR graph view

| Item | Site |
|---|---|
| `CsrGraph<'a>` (n, offsets, neighbors borrowed) | `permutation/mod.rs:859` |
| `CsrGraph::neighbors_of` (`#[cfg(feature = "userspace")]`) | `:883` |
| `CsrGraph::degree` (gated) | `:902` |
| `CsrGraph::try_neighbors_of` (kernel-safe) | `:931` |
| `CsrGraph::try_degree_of` | `:978` |
| `CsrGraph::try_validate` (full O(V+E) header validation) | (~`:1000`) |
| `CsrGraphError` enum | `:1092` |
| `Permutation` (validated `Vec<u32>`) | `:130` |
| `PermutationApplyError` / `PermutationValidationError` / `PermutationConstructionError` | `:695`, `:785`, `:1214` |

This is the right shape for "validated borrowed graph data structure with kernel-safe access", which the HNSW walker also needs.

### 8.2 Rabbit Order builder (`permutation/rabbit.rs`)

`rabbit_order_inner` at `permutation/rabbit.rs:317` is the single-pass community-merge loop that the HNSW builder pattern shares. Notable shapes:

- **Min-heap with stale-entry filtering.** `BinaryHeap<Reverse<(u64, u32)>>` constructed at `permutation/rabbit.rs:369` with `.with_capacity(n)`; the heap holds `(weighted_degree, vertex_id)`. Pop loop at `:391-442` uses an `alive: Vec<bool>` to filter absorbed vertices and a "rec_deg vs current degree" comparison to filter stale entries. **This is exactly the HNSW candidate-heap shape** (see §6.1 — Misra-Gries is wrong, this is right). The HNSW walker's beam-search heap stores `(distance, node_id)` and similarly accumulates stale entries when the bound tightens; the same "skip if stale" pattern works.
- **Scratch-buffer reuse across iterations.** `scratch_weights` / `scratch_degrees` allocated once at `:388-389` with `Vec::with_capacity(64)` and reused per heap pop. The HNSW walker should follow the same pattern for any per-step neighbor-list scratch (avoid the millions of per-step `Vec::new` calls).
- **Explicit `_inner` helper.** `rabbit_order_inner` at `:317` is the shared body between the panicking `rabbit_order` (`:282`, userspace-gated) and the kernel-safe `try_rabbit_order` (`:475`). Same convention HNSW must follow for `try_search` / `search` / `try_search_inner`.

### 8.3 HNSW reuse

- **Direct reuse** of `CsrGraph::try_neighbors_of` + `try_degree_of` if the HNSW graph is laid out as CSR (which is the natural mmap-friendly representation, and likely how the wire format will land). The HNSW walker traverses `graph.try_neighbors_of(node)?` to get a node's neighbors at the current layer; bounds-checked, kernel-safe, no panic.
- **Pattern reuse** of the min-heap-with-stale-filter loop (`rabbit.rs:391-442`) — the candidate heap and the dynamic-list heap in HNSW follow the same outer shape: pop, validate (alive? not stale?), expand, push.
- **Pattern reuse** of the scratch-buffer-reuse-across-iterations idiom for the per-step neighbor candidate batch.
- **Pattern reuse** of the `_inner` extraction discipline (`rabbit.rs:317`) — the HNSW design (`HNSW_PATH_DECISION.md` §5) calls this out explicitly.

---

## §9. `similarity::*` — existing similarity primitives (module shape)

Module: `similarity/mod.rs`. Docstring at `:1-27`.

The module shape HNSW should match:

```text
similarity/
├── mod.rs                  # public API; deprecated forwarders to crate::vector
├── kernels.rs              # backward-compat kernel forwarders (going away)
├── kernels_gather/         # gather-style (hash-driven) SIMD kernels
│   ├── mod.rs (compiles into kernels.rs in present layout)
│   ├── avx2.rs
│   ├── avx512.rs
│   └── neon.rs
├── minhash.rs              # MinHash signature builder + estimators
├── simhash.rs              # SimHash signature
├── lsh.rs                  # LSH banding
└── fuzzy/                  # CTPH, TLSH-like
    ├── ctph.rs
    └── tlsh_like.rs
```

### 9.1 MinHash (`similarity/minhash.rs`)

| Item | Site |
|---|---|
| `Signature<const K>` | `:32` |
| `classic_from_hashes` | `:104` |
| `classic_from_bytes` | `:129` |
| `signature_simd` (gather kernels) | `:415` |
| `signature_simd_into` | `:442` |
| `signature_batch_simd` | `:467` |
| `try_signature_batch_simd` | `:516` |
| `IncrementalSignature<'a, const K>` | `:575` |
| `one_permutation_from_hashes` | `:701` |
| `one_permutation_from_bytes` | `:726` |
| `densified_one_permutation` | `:746` |
| `jaccard_similarity` (over `Signature<K>`) | `:775` |
| `b_bit_jaccard_similarity` | `:806` |

### 9.2 SimHash (`similarity/simhash.rs`)

| Item | Site |
|---|---|
| `Signature64` | `:24` |
| `from_weighted_hashes` | `:63` |
| `from_unweighted_bytes` | `:93` |
| `from_weighted_bytes` | `:102` |

### 9.3 LSH (`similarity/lsh.rs`, `cfg(feature = "std")`)

| Item | Site |
|---|---|
| `LshConstructionError` enum | `:45` |
| `QueryStats` | `:103` |
| `MinHashIndex<Id, const K>` | `:141` |
| `SimHashIndex<Id, const BANDS>` | `:280` |

### 9.4 HNSW reuse

- **Direct reuse** of `MinHashIndex<Id, K>` / `SimHashIndex<Id, BANDS>` — these are *the existing similarity-search-with-an-index types* in the crate. They're the reference for "what does an `Index` API look like in this crate?" The HNSW walker's `try_search` shape mirrors `MinHashIndex::query`-like APIs.
- **Direct reuse** of the `try_signature_batch_simd` / `BatchShapeError` discipline — HNSW walker batched query APIs (`try_search_batch`?) follow the same `try_*` + `BatchShapeError`-style fallible pattern.
- **Pattern reuse** of the module layout: `mod.rs` (public API), `kernels.rs` + per-backend submodules, dedicated submodules per algorithm. HNSW lands as `similarity::hnsw` with `mod.rs`, `walker.rs`, `view.rs`, `kernels/{scalar,avx2,avx512,neon}.rs` per `HNSW_PATH_DECISION.md` §4 — the same shape.
- **Inspiration** for MinHash batching APIs: how the MinHash signature builder amortizes hash computation across many signatures (`signature_batch_simd` at `:467`) is similar to how the HNSW builder might amortize distance computation across all neighbor candidates of an inserting vector.

---

## §10. The `dispatch::auto::*` and `kernels::auto::*` pattern (the canonical playbook)

Walking through `bits::popcount` end-to-end as the reference HNSW kernels should match.

### 10.1 Public top-level entry (`bits/popcount.rs:24`)

```text
popcount_u64_slice(words: &[u64]) -> u64
└── kernels::auto::popcount_u64_slice(words)
```

No panic, kernel-safe. The function is `#[must_use]` and returns the count directly because there are no caller-input preconditions to fail (length 0 is fine, returns 0).

### 10.2 The `auto` dispatcher (`bits/popcount.rs:38-115`)

```text
fn auto::popcount_u64_slice(words: &[u64]) -> u64 {
    cfg(std + avx512 + x86) {
        if super::avx512::is_available() {
            return unsafe { super::avx512::popcount_u64_slice(words) };
        }
    }
    cfg(std + avx2 + x86) {
        if super::avx2::is_available() {
            return unsafe { super::avx2::popcount_u64_slice(words) };
        }
    }
    cfg(neon + aarch64) {
        if super::neon::is_available() {
            return unsafe { super::neon::popcount_u64_slice(words) };
        }
    }
    super::scalar::popcount_u64_slice(words)
}
```

Order: AVX-512 → AVX2 → NEON → scalar. Each `is_available()` is a compile-time `pub const fn` returning `false` in `--no-default-features` builds (see `bits/popcount/kernels/avx512.rs:36-40`) and a runtime `is_x86_feature_detected!` call when `std` is on (`:29-33`). This means the kernel-default build is `cfg`-pruned down to the scalar fallback — no panic, no SIMD entry point ever instantiated.

### 10.3 Per-backend kernel + visibility

`bits/popcount/kernels/avx512.rs:42-58` ships `pub unsafe fn popcount_u64_slice` with `#[target_feature(enable = "avx512f,avx512vpopcntdq")]`. `# Safety` documents the precondition (CPU support); the dispatcher's `is_available()` discharges it before the `unsafe { ... }`.

Visibility cfg per `bits/popcount.rs:119-189`: each backend module is `pub mod` under `arch-pinned-kernels` and `pub(crate) mod` otherwise. Feature-on exposes the per-backend kernels for bench / consumer comparison; feature-off keeps them internal but still compiled.

### 10.4 `_unchecked` siblings (`approx::bloom_kernels::scalar`)

Where the kernel has caller-input preconditions beyond CPU support, each per-backend module ships two `pub unsafe fn`s:

- `positions(h1, h2, k, bits, out)` — asserts `out.len() >= k` and `bits > 0` then calls `_unchecked`; `#[cfg(feature = "userspace")]`-gated (`approx/bloom_kernels/scalar.rs:13`).
- `positions_unchecked(h1, h2, k, bits, out)` — same body, no asserts; `# Safety` clause documents the precondition; always available (`approx/bloom_kernels/scalar.rs:30`).

The `auto::positions` dispatcher (`approx.rs:840-883`) always calls `_unchecked` after CPU detection — kernel-default builds reach the SIMD path without ever entering an asserting code path.

### 10.5 What HNSW kernels must do (cheat-sheet)

For each `(metric, scalar_kind)` pair (e.g. `l2_squared_i8`):

1. `similarity::hnsw::kernels::scalar::l2_squared_i8(a: &[i8], b: &[i8]) -> i32` — reference oracle, always present.
2. Per-backend module (`avx2.rs`, `avx512.rs`, `neon.rs`, `sse41.rs`): `pub fn is_available() -> bool`; `#[target_feature(...)] pub unsafe fn l2_squared_i8_unchecked(...)`; `#[cfg(feature = "userspace")] pub unsafe fn l2_squared_i8(...)` (asserts then calls `_unchecked`).
3. `similarity::hnsw::kernels::auto::l2_squared_i8(...) -> Option<i32>` — length check, then AVX-512 → AVX2 → NEON → SSE4.1 → scalar dispatch via `_unchecked` siblings.
4. Walker entry: `try_search_inner(...)` calls `auto::l2_squared_i8` — no panic on input.

---

## §11. Other adjacent primitives worth knowing about

### 11.1 `fingerprint::*` (`fingerprint/mod.rs`)

| Type | Site |
|---|---|
| `BlockFingerprint` (8-byte `repr(C)`) | `fingerprint/mod.rs:42` |
| `ExtentFingerprint` | `:61` |
| `block(bytes: &[u8; 256]) -> BlockFingerprint` | `:217` |
| `extent(bytes: &[u8]) -> ExtentFingerprint` | `:447` |

These are the F22-fingerprint producers — they sit *above* the HNSW walker (the walker reads pre-built F22 fingerprint vectors out of the index). Direct relevance to HNSW: the F22 path is "byte-quantized to u8", which is why the HNSW i8/u8 distance kernels (currently a gap, see §1.4) are the kernel-default path per `HNSW_PATH_DECISION.md` §1.

### 11.2 `search::*` (`search/mod.rs`)

Substring matchers (Bitap, TwoWay, RabinKarp, ShiftOr, PackedDfa, PackedPair) per `search/mod.rs:20-27`. Not composable with HNSW but a useful precedent for "many small algorithms in one module, all no_std-clean, all named alike" — every algorithm is its own submodule with the same constructor + `find` / `find_iter` shape.

### 11.3 `hash::mix64` / `hash::fnv1a64`

Cheap-mix functions used for MinHash seeding (`similarity/minhash.rs:25`); candidate seed-mix for HNSW's deterministic level-assignment RNG (`HNSW_PATH_DECISION.md` §4 `build/level.rs`: "deterministic RNG, no thread-local"). Both are no-std-clean.

### 11.4 `bits::streamvbyte` / `bits::bit_pack`, `divergence::*`

StreamVByte (`bits/streamvbyte.rs:1-79`) and dynamic bit-packing (`bits/bit_pack.rs:111`, `:232`) — out of scope for v1 HNSW (neighbor lists are fixed-width u32 / u64 IDs); candidates for a future v2 wire format that delta-encodes neighbors. `divergence::*` ships KL / Jensen-Shannon / Hellinger / total-variation count-vector distances re-exported via `similarity::distance::counts::*` (`similarity/mod.rs:50-57`); not on HNSW's metric list but candidates for the future `similarity::hybrid` layer (`HNSW_PATH_DECISION.md` §11). Inspiration only.

---

## §12. Summary — what HNSW reuses vs. what HNSW writes

### Direct reuse (zero new code in `similarity::hnsw`)

- `vector::dot_f32`, `l2_squared_f32`, `cosine_similarity_f32` — all backends.
- `vector::hamming_u64`, `vector::jaccard_u64` — all backends, kernel-safe binary distance.
- `bits::popcount_u64_slice`, `popcount_u8_slice` — VPOPCNTQ / VCNT for binary metrics + visited-set cardinality reporting.
- `bitmap::Container::*` (`intersect`, `intersect_cardinality`, `contains`) — filter primitive composition.
- `bitmap::BitmapContainer` (1024-word dense form) and the AVX-512 / AVX2 / NEON `_into_nocard` / `_cardinality` variants — filter set-algebra inside the walker.
- `permutation::CsrGraph::try_neighbors_of` / `try_degree_of` — graph adjacency access if HNSW lays the graph out as CSR.
- `bits::rank_select::*` — visited-bitset support if we pick the bitset path.
- `dispatch::Backend`, `ProcessorProfile`, `is_*_available()` — backend selection at the dispatcher.
- `hash::mix64` — deterministic level-assignment RNG seed mixing.

### Pattern reuse (HNSW writes the kernel; the SHAPE is borrowed)

- `_inner` / `try_*` / `_unchecked` discipline — the core audit-R10 contract; `permutation::rabbit::rabbit_order_inner` (`rabbit.rs:317`), `bits::rank_select::RankSelectDict::rank1_inner` (`:382`), and `hash::set_membership::auto::contains_u32_batch_unchecked` (`:197`) are the canonical examples.
- The `auto::*_unchecked` dispatcher pattern with `cfg`-gated `is_available()` chains — every `kernels/mod.rs` follows this; the HNSW kernels module copies the shape.
- The `_card` / `_nocard` / `_justcard` discipline from `bitmap::BitmapContainer` — applied to HNSW it would be e.g. "compute distances and return the heap" vs "compute distances, push if better" vs "just count how many candidates passed the threshold".
- The min-heap-with-stale-filter loop from `permutation/rabbit.rs:391-442` — the HNSW beam-search expansion loop has the same shape.
- Scratch-buffer reuse across iterations (`rabbit.rs:388-389`).
- Per-backend module visibility gated on `arch-pinned-kernels` (`vector/kernels/mod.rs:20-66`, `bits/popcount.rs:119-189`).
- The `BatchShapeError` enum convention and the `try_*_batch` / `*_batch` (gated) split — see `vector::batch::BatchShapeError` (`:46`), `set_membership::SetMembershipBatchError` (`:31`).

### Inspiration only (HNSW writes its own variant for HNSW-specific reasons)

- The candidate top-K / dynamic-list heap — Misra-Gries / SpaceSaving (`approx::SpaceSaving`, `histogram::topk::MisraGries`) are *frequency* heaps, not *score* heaps; the HNSW heap is closer to the `BinaryHeap<Reverse<(distance, id)>>` cap pattern from `permutation/rabbit.rs:369`, but with a fixed cap and explicit overflow eviction.
- The visited-set tracking — neither `hash::set_membership` (linear scan, breaks at >256 elements) nor a naive `Vec<bool>` (no compaction) nor `bits::rank_select` alone is the answer; the HNSW visited-set wants a hybrid bitset + version-stamp scheme. `set_membership` and `rank_select` are useful inputs, not the answer.
- HLL register-merge SIMD (`approx/hll/kernels/{avx2,avx512,neon}.rs`) — same SIMD shape as a future "batch min-update across K candidate distances" kernel if HNSW ever wants one. Not v1.
- `fingerprint::block` / `extent` — the producer of the F22 vectors HNSW indexes; not part of the HNSW path itself but defines the i8 / u8 byte-quantized regime that the HNSW kernel-default metrics target.

### Net new HNSW code (the things this inventory confirms are gaps)

1. **i8 / u8 distance kernels.** The entire `(metric × {i8, u8})` matrix from `HNSW_PATH_DECISION.md` §10 is missing — `vector::*` only ships `u32` integer kernels and `f32` / `u64`-binary float / popcount kernels. The HNSW kernels module writes `dot_i8`, `l2_squared_i8`, `cosine_similarity_i8`, plus `_u8` siblings on every backend (scalar / AVX2 / AVX-512 / NEON / SSE4.1 / SSSE3).
2. **Tanimoto.** Composition step beyond Jaccard; needs its own kernel or is a tiny wrapper over `popcount(a & b)` and `popcount(a | b)` that the HNSW kernel module owns.
3. **SSE4.1 / SSSE3 fallbacks.** None of `vector/kernels/` or `bits/popcount/kernels/` ship SSE4.1 / SSSE3 paths today; HNSW table says SSE4.1 must cover integer + binary metrics, SSSE3 must cover binary popcount via PSHUFB.
4. **HNSW data layout** — `header.rs`, `view.rs`, `graph.rs`, `walker.rs`, `visit.rs`, `candidates.rs`, `filter.rs`, `select.rs`, `build/{insert,level,serialize,mod}.rs` per `HNSW_PATH_DECISION.md` §4. None of these exist in the crate today; all of them have well-defined primitive consumers per the citations above.
5. **Wire-format parser** for usearch v2.25 — none in the crate (the `_references/usearch/include/usearch/index_dense.hpp` reference is checked in but no Rust parser exists).

The arithmetic the HNSW walker / builder needs to perform is largely already present in `tokenfs-algos` for the f32 + binary paths. The HNSW landing's net-new arithmetic is the i8 / u8 kernel matrix and the SSE4.1 / SSSE3 fallbacks. Everything else — the validation discipline, the dispatcher shape, the heap pattern, the scratch-buffer reuse, the bitmap filter primitive, the popcount foundation — is already in the crate and the HNSW module composes on top.

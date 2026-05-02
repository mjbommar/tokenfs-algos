# `bitmap` module — Roaring SIMD set kernels

**Status:** spec, 2026-05-02. Phase B3 of `01_PHASES.md`.

This module fills a real Rust ecosystem gap: no current crate exposes container-level SIMD kernels (`intersect_vector16`, bitmap AND with cardinality, etc.) as separately callable primitives. `roaring-rs` is scalar in its inner loops; `croaring-sys` exposes only the high-level `Bitmap` API. We ship the inner kernels at primitive granularity so callers can compose them into their own bitmap machinery.

## Goal & scope

Provide SIMD kernels for the **inner loops** of Roaring set algebra, parameterized over the three container types Roaring uses:

- **bitmap container**: 8 KB = 65,536 bits = 1024 u64s, dense.
- **array container**: sorted u16 array (cardinality ≤ 4096), sparse.
- **run container**: sorted run-pairs `(start, length)`, very sparse or very long-run-dominated.

The five operations: **intersect**, **union**, **difference**, **symmetric difference (XOR)**, **cardinality**.

## § 1 Module surface

```
bitmap/
├── mod.rs                          // public API
├── containers.rs                   // shared types: BitmapContainer, ArrayContainer, RunContainer
├── intersect.rs                    // dispatch by container-pair
├── union.rs
├── difference.rs
├── xor.rs
├── cardinality.rs
└── kernels/
    ├── bitmap_x_bitmap_avx2.rs     // 256-bit AND/OR/XOR/ANDNOT loops
    ├── bitmap_x_bitmap_avx512.rs   // 512-bit + VPOPCNTQ
    ├── bitmap_x_bitmap_neon.rs     // 128-bit AND/OR/XOR/ANDNOT
    ├── array_x_array_sse42.rs      // pcmpistrm-based intersect (Schlegel)
    ├── array_x_array_avx512.rs     // VP2INTERSECT (or its AVX-512F emulation)
    ├── array_x_bitmap_avx2.rs      // gather + bit-test
    └── array_x_bitmap_avx512.rs    // VPCOMPRESSD-based output materialization
```

Public types:

```rust
pub mod bitmap {
    /// Dense 65536-bit bitmap.
    pub struct BitmapContainer {
        pub words: Box<[u64; 1024]>,
    }

    /// Sorted u16 array, ≤ 4096 entries.
    pub struct ArrayContainer {
        pub data: Vec<u16>,
    }

    /// Sorted run-pairs.
    pub struct RunContainer {
        pub runs: Vec<(u16, u16)>,  // (start, length)
    }

    /// Container enum for dispatch.
    pub enum Container {
        Bitmap(BitmapContainer),
        Array(ArrayContainer),
        Run(RunContainer),
    }

    /// Set algebra ops (in-place and out-of-place variants).
    impl Container {
        pub fn intersect(&self, other: &Container) -> Container;
        pub fn union(&self, other: &Container) -> Container;
        pub fn difference(&self, other: &Container) -> Container;
        pub fn symmetric_difference(&self, other: &Container) -> Container;
        pub fn cardinality(&self) -> u32;
        pub fn intersect_cardinality(&self, other: &Container) -> u32;
    }
}
```

## § 2 Container-pair dispatch table

Per CRoaring's posture (research summarized below):

| pair | intersect | union | difference | xor | cardinality |
|---|---|---|---|---|---|
| **bitmap × bitmap** | **AVX-512 AND** + VPOPCNTQ | AVX-512 OR | AVX-512 ANDNOT | AVX-512 XOR | VPOPCNTQ |
| **array × array** | **SSE4.2 pcmpistrm + shuffle (Schlegel)** or AVX-512 VP2INTERSECT-emulated | merge-sort SIMD | merge-sort SIMD | merge-sort SIMD | popcount of intersect bitmask |
| **array × bitmap** | bit-test loop, **AVX-512 VPCOMPRESSD** for output | bitmap-side AND-NOT then add | array-side filter | scalar | scalar bit-test |
| **bitmap × array** | (symmetric to above) | (symmetric) | (symmetric) | (symmetric) | (symmetric) |
| **run × run** | **scalar interval merge** | scalar | scalar | scalar | sum(length) |
| **run × array** | scalar interval walk | scalar | scalar | scalar | scalar |
| **run × bitmap** | scalar | scalar | scalar | scalar | scalar |

The SIMD-heavy pairs are **bitmap×bitmap, array×array, and the bitmap-side of array×bitmap**. Plan the dispatch around these three; the scalar pairs ride on simple iterator machinery.

## § 3 bitmap × bitmap kernels (the easy big win)

### Algorithm

1024 u64 words AND/OR/XOR/ANDNOT, unrolled 8x using AVX-512 (`_mm512_loadu_si512`, `_mm512_and_si512`) or AVX2 (`_mm256_loadu_si256`, `_mm256_and_si256`).

For *cardinality-while-doing-the-op* variants (e.g., `intersect_cardinality`): accumulate a 512-bit popcount via `VPOPCNTQ` (AVX-512) or Mula's nibble-LUT method (AVX2, see `bits::popcount`).

### Three variants per op

CRoaring exposes `_card`, `_nocard`, `_justcard` — output the result *and* cardinality / output without cardinality / cardinality only. We ship the same three:

```rust
impl BitmapContainer {
    pub fn and_into(&self, other: &Self, out: &mut Self) -> u32;  // returns cardinality
    pub fn and_into_nocard(&self, other: &Self, out: &mut Self);
    pub fn and_cardinality(&self, other: &Self) -> u32;
}
```

The `_justcard` variant is critical because most boolean queries care about "is the result empty" or "does the result have ≥ k items" — knowing cardinality without materializing the result is a 2-3x win.

### Throughput

CRoaring numbers (Lemire et al. 2018): ~0.6-0.7 cycles per u64 word for `and_cardinality` on Skylake, memory-bandwidth-bound. ~25-30 GB/s effective. Pure AND without cardinality: ~0.1-0.2 cycles/word.

For our purposes: bitmap×bitmap operations on 8KB containers run in ~600-1500 cycles each. Sub-microsecond per op.

## § 4 array × array intersect (the SIMD set intersection paper)

### Algorithm: Schlegel et al. 2011

The canonical SSE4.2 algorithm:

```
load 8 u16 from A into Va
load 8 u16 from B into Vb
mask = pcmpistrm(Va, Vb, _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK)
shuf = shuffle_mask16[mask]
out  = pshufb(Va, shuf)         // compact survivors to front
store out, advance by popcount(mask)
advance Va or Vb based on which had the smaller max element
```

The 256-entry `shuffle_mask16` table maps a u8 bitmask to the corresponding `[u8; 16]` shuffle pattern. Precomputed at compile time.

Lemire's later switch from `pcmpestrm` to `pcmpistrm` (no length operand) is faster and is the current CRoaring kernel.

### Galloping fallback

When one side is much smaller than the other (e.g., 10 u16s vs 4000), the SIMD pcmpistrm wastes most of its lanes. Use **galloping search** (binary doubling) for the small side. Threshold: smaller side < 4 × log2(larger side) is the Schlegel paper's heuristic; CRoaring uses similar.

### AVX-512 considerations

CRoaring **does not** have AVX-512 array×array intersect implemented (open issue #454). Two options for our port:

1. **Native AVX-512 VP2INTERSECT** (Intel-only, Tiger Lake+). Direct hardware support but limited deployment.
2. **AVX-512 emulation** per Diez-Canas (arXiv 2112.06342) — beats native VP2INTERSECT on Tiger Lake using shuffle-and-compare patterns. Wider deployment.

**Tentative**: ship SSE4.2 (Schlegel) as the primary; experiment with AVX-512 emulation as opt-in for v0.3.

### Han et al. SIGMOD 2018 (QFilter)

QFilter is a byte-prefilter merge that's faster than Schlegel for graph workloads. CRoaring hasn't adopted it.

**Tentative**: stick with Schlegel for v0.2 (well-understood, well-tested); evaluate QFilter as a v0.3 experiment if a graph-workload bench shows it dominates.

### Throughput

CRoaring reports ~3-4 cycles per output element for vectorized vs ~7-8 for scalar. ~2x speedup at moderate selectivity. At 3 GHz: ~0.8-1.0 billion u16 outputs/sec.

## § 5 array × bitmap

### Algorithm

Iterate the sorted u16 array; for each element, bit-test the bitmap. Output the elements that hit. The bit-test is `(bitmap[v >> 6] >> (v & 63)) & 1`.

The SIMD win comes from *materializing the output array* via VPCOMPRESSD (AVX-512) — gather a chunk of array elements, broadcast their indices into the bitmap, AND, compress non-zero results to the front of the output register.

For AVX2 (no VPCOMPRESSD), use the same precomputed shuffle-mask table from § 4.

### Throughput

Bandwidth-bound on the bitmap reads. ~10 GB/s on AVX2, ~20 GB/s on AVX-512.

## § 6 run × * operations

CRoaring's run × * operations are scalar interval merges. The TODO in CRoaring's tree says "this could be a lot more efficient." Our v0.2 ships the same scalar interval-merge approach. Run containers are rare enough in TokenFS workloads (sorted u32 extent IDs are typically array-class, not run-class) that this isn't on the hot path.

**Trigger to revisit**: a workload profile shows run × run operations dominating.

## § 7 Cardinality (popcount)

For bitmap container: AVX-512 `VPOPCNTQ` is one cycle per u64. AVX2 falls back to Mula/Kurz/Lemire Harley-Seal CSA tree (Mula, Kurz, Lemire 2018, arXiv 1611.07612) — ~5 GB/s.

For array container: cardinality is just `data.len()`.

For run container: sum of `length+1` across all runs.

For "intersect cardinality" of array × array: same pcmpistrm kernel as § 4 but replace shuffle/store with `popcount(mask)`. Strictly cheaper than full intersect.

## § 8 API: stay separate from `roaring-rs`

We are NOT building a competing high-level Roaring `Bitmap` type. The `roaring-rs` crate already provides the high-level container-of-containers, build, ser/de, and iteration machinery. Our `bitmap` module exposes the SIMD container-level kernels as primitives that:

1. `roaring-rs` could vendor as an inner-loop replacement (when their `simd` feature stabilizes).
2. `tokenfs-paper` or other consumers can use directly when they need raw container ops without the full Roaring overhead.

This is the same pattern as `simdjson`/`simdutf` — primitives exposed for composition, not a competing parser.

## § 9 Hardware acceleration plan

| Operation | Scalar | SSE4.2 | AVX2 | AVX-512 | NEON |
|---|---|---|---|---|---|
| bitmap×bitmap AND | ~10 GB/s | ~25 GB/s | ~25-30 GB/s | ~30-50 GB/s | ~10-15 GB/s |
| bitmap cardinality (popcount) | ~3 GB/s | -- | ~5 GB/s | ~30-50 GB/s (VPOPCNTQ) | ~5-8 GB/s |
| array×array intersect (Schlegel) | 100M elem/sec | 200-400M | 200-400M | 600M+ (VP2INTERSECT or emulation) | (NEON variant exists, ~150-300M) |
| array×bitmap intersect | 50M elem/sec | -- | 200M | 400M (VPCOMPRESSD) | 100M |
| run × any | scalar only | -- | -- | -- | -- |

## § 10 Test plan

- **Scalar oracle** for every container pair × op.
- **Property tests**: random container pairs, every (op, pair) — result matches scalar oracle bit-exactly.
- **roaring-rs parity**: where `roaring-rs` exposes equivalent scalar APIs, our SIMD kernels match its output bit-exactly.
- **Edge cases**: empty containers, single-element containers, identical containers, disjoint containers, container at boundary (4096 array threshold).
- **AVX-512 specific**: cycle through CPU feature combinations (AVX-512F only; AVX-512F + VPOPCNTQ; AVX-512F + VBMI; AVX-512F + VP2INTERSECT).

## § 11 Bench plan

For each (pair, op):

- **Two posting lists of 100 entries each (sparse)** — array×array hot.
- **Two posting lists of 10K entries each (medium)** — likely array×bitmap or both bitmap.
- **Two posting lists of 100K entries each (dense)** — bitmap×bitmap hot.
- **Two posting lists of 1M entries each (very dense)** — bitmap×bitmap, multiple containers.

Report:
- Throughput in elements/sec.
- Cardinality-only variant separately.
- Comparison to `roaring-rs` scalar inner loops.
- Comparison to `croaring-sys` (FFI to CRoaring).

## § 12 Open questions

1. **Should we own the high-level `Bitmap` type**, or stay at container-kernel granularity? **Tentative: stay at container granularity.** Composability over completeness; let consumers compose.

2. **Run container support priority**: skip in v0.2 entirely vs ship scalar? **Tentative: ship scalar interval-merge.** Run containers are rare but appear in real Roaring workloads; an unimplemented `run` path would surprise consumers.

3. **VP2INTERSECT vs emulation**: Diez-Canas shows emulation beats native on some Intel parts. Native availability is narrow (Tiger Lake+, missing on Zen4). **Tentative: implement emulation as the AVX-512 path; native VP2INTERSECT as optional fast path on detection.**

4. **QFilter (Han et al. 2018)**: ~2x faster than Schlegel for graph workloads. Worth implementing? **Tentative: defer to v0.3; ship Schlegel first, evaluate QFilter when a graph workload is available to bench.**

5. **256-entry vs 65K-entry shuffle table**: AVX-512 with VBMI VPERMB could process 16 elements per iteration with a 65K-entry shuffle table (4 MiB). Cache-unfriendly. **Tentative: skip; the 256-entry table fits L1.**

6. **Where do `_card`, `_nocard`, `_justcard` API variants land?** Three function names per op gets noisy. Alternative: const-generic on a `WithCardinality` flag. **Tentative: explicit names; const-generics for `out.is_some()` adds API surface complexity for marginal gain.**

7. **Where does serialization live?** Our kernels operate on already-deserialized containers. Roaring's portable serialization format lives in `roaring-rs` / CRoaring; we don't reimplement it. **Tentative: out of scope.**

## § 13 Reference impls and citations

- **CRoaring** (C/C++): https://github.com/RoaringBitmap/CRoaring — primary reference.
- **roaring-rs** (Rust scalar): https://github.com/RoaringBitmap/roaring-rs — parity oracle.
- **croaring-sys** (Rust FFI): https://crates.io/crates/croaring — for ground-truth cross-checks.
- Chambi, Lemire, Kaser, Godin, "Better bitmap performance with Roaring bitmaps", SPE 46(5):709-719, 2016 (arXiv 1402.6407).
- Lemire, Ssi-Yan-Kai, Kaser, "Consistently faster and smaller compressed bitmaps with Roaring", SPE 46(11):1547-1569, 2016.
- Lemire et al., "Roaring Bitmaps: Implementation of an Optimized Software Library", SPE 48(4), 2018 (arXiv 1709.07821).
- Schlegel, Willhalm, Lehner, "Fast Sorted-Set Intersection using SIMD Instructions", ADMS@VLDB 2011.
- Inoue, Ohara, Taura, "Faster Set Intersection with SIMD instructions by Reducing Branch Mispredictions" (BMiss), VLDB 2014.
- Mula, Kurz, Lemire, "Faster Population Counts using AVX2 Instructions", Computer Journal 2018 (arXiv 1611.07612).
- Han, Zou, Yu, "Speeding Up Set Intersections in Graph Algorithms using SIMD Instructions", SIGMOD 2018.
- Zhang et al., "FESIA: A Fast and SIMD-Efficient Set Intersection Approach on Modern CPUs", ICDE 2020.
- Diez-Canas, "Faster-Than-Native Alternatives for x86 VP2INTERSECT Instructions", arXiv 2112.06342, 2021.
- CRoaring AVX-512 gap list: https://github.com/RoaringBitmap/CRoaring/issues/454.

## § 14 Environment fitness

Per [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md):

| API | Kernel module | FUSE | Userspace | Postgres ext | cgo (Go) | Python (PyO3) |
|---|---|---|---|---|---|---|
| `BitmapContainer` AND/OR/XOR/ANDNOT in-place | ✅ 8 KB stack OK | ✅ | ✅ | ✅ | ✅ batched | ✅ |
| `BitmapContainer::*_cardinality` | ✅ AVX-512 VPOPCNTQ is *the* kernel SIMD win | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ArrayContainer` intersect (Schlegel pcmpistrm) | ✅ 16 B SIMD lanes; 256-entry shuffle table is rodata | ✅ | ✅ | ✅ | ✅ | ✅ |
| `ArrayContainer` allocating output | ⚠️ caller-provided output Vec | ✅ | ✅ | ⚠️ palloc-friendly variant | ✅ | ✅ |
| `Container` enum dispatch | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `RunContainer` interval ops | ✅ scalar; small footprint | ✅ | ✅ | ✅ | ✅ | ✅ |

**Notes:**
- Bitmap containers are 8 KB — fit on kernel stack but only just. Two-input ops total 16 KB temp; safe with caller-provided output (`out: &mut BitmapContainer`).
- `intersect_vector16`'s 256-entry shuffle table is 4 KiB of static rodata — kernel-safe, fits L1.
- The `_card`, `_nocard`, `_justcard` variants exist specifically because *most* boolean queries want cardinality without materializing the result (Postgres "EXISTS"-style queries, MinIO "is overlap nonempty"). All three variants are kernel-safe.
- Postgres GIN bitmap-scan executor could consume these primitives directly; this is potentially the highest-leverage non-TokenFS adoption path.
- AVX-512 VPOPCNTQ in kernel SIMD section: real ~10x win over AVX2 software popcount. The `kernel_fpu_begin` overhead is amortized over a typical 8 KB bitmap operation.

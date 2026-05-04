# Component: Distance kernels

**Status:** skeleton, 2026-05-03. Filled in across Phases 1 → 2 → 3.

**Lives in:** `crates/tokenfs-algos/src/similarity/hnsw/kernels/{mod.rs, scalar.rs, avx2.rs, avx512.rs, neon.rs, sse41.rs, ssse3.rs}`

## Role

Per-backend distance kernels for the metrics × scalar-types combinations the walker and builder use. Same per-backend kernel buffet as `bits::popcount` / `vector::distance` — pinned modules under `arch-pinned-kernels`, runtime dispatcher via `kernels::auto`, scalar parity oracle.

## Required research input

- [`../research/SIMD_PRIOR_ART.md`](../research/SIMD_PRIOR_ART.md) (full file — patterns for every backend × every metric)
- [`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md) (which existing kernels we can directly reuse from `vector::*`, `bits::popcount`, etc.)
- [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md) §5 (usearch's distance kernel organization, simsimd integration)

## Key findings to fold in

- **f32 + binary-popcount kernels reusable.** Per primitive inventory: `vector::dot_f32`, `vector::l2_squared_f32`, `vector::cosine_f32` (scalar / AVX2 / AVX-512 / NEON), and `bits::popcount` (per-backend) cover most of the matrix without new code. HNSW's distance kernels for these cells are thin shims over the existing primitives.
- **Net new work: i8 + u8 quantized kernels + SSE4.1/SSSE3 fallbacks.** These are the matrix cells the existing crate doesn't already cover.
- **NumKong is the active prior art for kernel patterns.** SimSIMD was renamed and re-launched as NumKong in March 2026 (same author, same Apache-2.0/BSD-3 dual license). The local checkout `_references/NumKong/` is the up-to-date kernel source. Cite kernel patterns from `_references/NumKong/`; cite older published throughput tables from SimSIMD documentation.
- **AVX-512 stabilized in Rust 1.89 (2025-08-07).** No nightly required. Workspace `rust-toolchain.toml` pins to >=1.89.
- **Tanimoto on binary collapses to Jaccard.** usearch's runtime dispatcher maps `tanimoto_k → nk_kernel_jaccard_k`. Save backend matrix slots: don't duplicate kernels.
- **VPDPBUSD's "XOR 0x80 unsigned-bias trick"** — for signed×signed i8 dot product without VNNI's signed-only variant. AVX-512 i8 dot kernel uses this pattern.
- **hnswlib's `_MM_HINT_T0` graph-traversal prefetch.** Walker prefetches the next graph node's vector blob before computing distance to the current candidate. ~10-15% speedup on cold-cache workloads.

## Sections to fill in

1. **Kernel matrix.** The `(metric, scalar_kind, backend)` cells implemented and their target throughput.
2. **Per-backend implementation notes.** Specific SIMD instructions used per kernel. Tail handling. Horizontal reduction patterns.
3. **Kernel-safety pattern.** `_unchecked` siblings (callable from runtime dispatcher in kernel-default builds); asserting variants gated on `userspace`.
4. **Reuse map.** Which existing crate primitives this layer composes vs writes fresh. Specifically: do we delegate i8 dot product to `vector::dot_i8` if it exists, or write a new HNSW-specific kernel? (Decision per `PRIMITIVE_INVENTORY.md`.)
5. **Auto dispatcher.** `kernels::auto::distance(metric, scalar_kind, a, b, dim)` — runtime dispatch via `is_x86_feature_detected!` / equivalents. Routes to `_unchecked` after CPU detection.
6. **Cross-arch f32 caveat.** AVX2 vs AVX-512 vs NEON FMA fusion produces slightly different f32 results. Documented; integer metrics are cross-arch byte-identical.
7. **iai-callgrind benches.** Per-kernel bench rows for the regression gate.

## Final backend matrix (target end-state at v0.7.0 release)

| Metric | scalar | AVX2 | AVX-512 (nightly) | NEON | SSE4.1 | SSSE3 |
|---|---|---|---|---|---|---|
| L2² (f32) | ✅ | ✅ FMA | ✅ FMA + 512-bit | ✅ FMLA | — | — |
| L2² (i8 / u8) | ✅ | ✅ PMADDUBSW | ✅ VPDPBUSD if VNNI | ✅ SDOT/VMULL | ✅ | — |
| cosine (f32) | ✅ | ✅ | ✅ | ✅ | — | — |
| cosine (i8 / u8) | ✅ | ✅ | ✅ | ✅ | ✅ | — |
| dot (f32) | ✅ | ✅ FMA | ✅ VFMADD231PS | ✅ FMLA | — | — |
| Hamming (binary) | ✅ | ✅ PSHUFB-popcount | ✅ VPOPCNTQ if BITALG | ✅ VCNT | ✅ POPCNT | ✅ PSHUFB |
| Jaccard (binary) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tanimoto (binary) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

SVE2 deferred. SSE4.2 native POPCNT used opportunistically when host has it but no AVX2.

## API skeleton

```rust
pub enum Metric { L2Squared, Cosine, Dot, Hamming, Jaccard, Tanimoto }
pub enum ScalarKind { F32, I8, U8, Binary /* packed bits */ }
pub type Distance = u32;  // for integer/binary; f32 metrics use Distance::from_bits transparently

// Per-backend module pattern (e.g. avx2.rs):
#[target_feature(enable = "avx2")]
pub unsafe fn distance_l2_squared_i8_unchecked(
    a: *const i8,
    b: *const i8,
    len: usize,
) -> u32;

#[cfg(feature = "userspace")]
#[target_feature(enable = "avx2")]
pub unsafe fn distance_l2_squared_i8(
    a: *const i8,
    b: *const i8,
    len: usize,
) -> u32 {
    debug_assert!(len > 0);
    debug_assert!(len.is_multiple_of(32));
    debug_assert!(is_x86_feature_detected!("avx2"));
    distance_l2_squared_i8_unchecked(a, b, len)
}

// Auto dispatcher (kernels::auto::distance).
pub fn distance(
    metric: Metric,
    scalar: ScalarKind,
    a: &[u8],
    b: &[u8],
) -> Distance;
```

## Cross-references

- Phases: [`../phases/PHASE_1.md`](../phases/PHASE_1.md) (scalar oracle), [`../phases/PHASE_2.md`](../phases/PHASE_2.md) (AVX2/NEON/SSE4.1), [`../phases/PHASE_3.md`](../phases/PHASE_3.md) (AVX-512)
- Research: [`../research/SIMD_PRIOR_ART.md`](../research/SIMD_PRIOR_ART.md), [`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md)
- Pattern reference: `src/bits/popcount/kernels/*` and `src/vector/distance/kernels/*` (canonical per-backend kernel buffet)
- Audit: [`../../PROCESSOR_AWARE_DISPATCH.md`](../../PROCESSOR_AWARE_DISPATCH.md), [`../../KERNEL_SAFETY.md`](../../KERNEL_SAFETY.md)

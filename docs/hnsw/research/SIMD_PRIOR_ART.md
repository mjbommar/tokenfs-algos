# SIMD Distance Kernels: Prior Art Survey

Scope: state of the art in SIMD distance kernels (the inner loop of every vector-search system) so the `tokenfs-algos::similarity::hnsw::kernels` per-backend implementations can adopt the right patterns. Backends planned: scalar, AVX2, AVX-512 (nightly), NEON (AArch64), SSE4.1, SSSE3. SVE2 deferred. Metrics: L2², cosine, dot, Hamming, Jaccard, Tanimoto. Scalar types: f32, i8, u8, binary (packed bits).

Important note on naming: in March 2026 Ash Vardanian renamed and re-launched SimSIMD as **NumKong**. The two are the same lineage of code, with NumKong being the actively maintained successor (same author, same Apache-2.0 / 3-Clause BSD dual license, same dispatch model, expanded scope to ~2,000 kernels and 15+ numeric types). Throughout this document we cite the NumKong source tree under `_references/NumKong/` for the up-to-date kernel patterns and the older SimSIMD repository for historical published throughput tables. (Sources: [NumKong: 2'000 Mixed Precision Kernels](https://ashvardanian.com/posts/numkong/), [usearch issue #100555 on the rename](https://github.com/clickhouse/clickhouse/issues/100555).)

USearch is the canonical consumer: prior to v2.20 it embedded SimSIMD; after v2.20 it embeds NumKong via the optional `USEARCH_USE_NUMKONG=1` CMake flag. USearch's distance kernels are essentially a thin `nk_*` wrapper and a metric-kind dispatch. We treat NumKong/SimSIMD as the canonical kernel reference and USearch as the canonical integration reference.

---

## 1. NumKong / SimSIMD — the canonical reference

URL: https://github.com/ashvardanian/NumKong (formerly https://github.com/ashvardanian/SimSIMD).
License: Apache-2.0 OR BSD-3-Clause (dual license). Both are permissive — patterns can be studied, copied, or vendored with attribution. (Source: `_references/NumKong/LICENSE`.)

### 1.1 File / kernel organization

NumKong organizes kernels by `<operation>/<microarch>.h` instead of `<microarch>/<operation>.h`. From `_references/NumKong/include/numkong/`:

```
spatial.h            -- public API: nk_euclidean_*, nk_sqeuclidean_*, nk_angular_*
spatial/             -- per-microarch kernels (referenced from spatial.h includes)
dot.h                -- public API: nk_dot_*
dot/
  serial.h           -- portable scalar reference
  haswell.h          -- AVX2 + FMA + F16C
  skylake.h          -- AVX-512F + DQ + BW + VL
  icelake.h          -- + VNNI + VPOPCNTDQ + VBMI + GFNI + BITALG
  alder.h            -- AVX2 + AVX-VNNI (256-bit VPDPBUSD)
  sierra.h           -- AVX-VNNI-INT8 (signed×signed and unsigned×unsigned i8)
  genoa.h            -- AVX-512 + BF16
  sapphire.h         -- + AVX-512 FP16
  diamond.h          -- AVX10.2 FP8/FP16 VNNI
  neon.h             -- ARMv8-A baseline NEON
  neonsdot.h         -- + ARMv8.4 dotprod (SDOT/UDOT)
  neonbfdot.h        -- + bf16 dot (BFDOT)
  neonfhm.h          -- + FMLAL/FMLAL2 (f16 widening FMA)
  neonfp8.h          -- + FP8 (Armv9.2)
  neonhalf.h         -- + native FP16 vector
  sve.h, svehalf.h, svesdot.h, svebfdot.h
  rvv.h, rvvbf16.h, rvvhalf.h, rvvbb.h
  loongsonasx.h, powervsx.h, v128relaxed.h
set.h, set/<microarch>.h     -- binary Hamming/Jaccard
sets.h, sets/<microarch>.h   -- many-to-many binary metrics
cast.h, cast/<microarch>.h   -- numeric type widening (f16→f32, bf16→f32, etc.)
reduce.h, reduce/<microarch>.h  -- horizontal-reduction primitives
capabilities.h        -- runtime CPU detection + dispatch table
```

Each per-microarch header is wrapped in `#if NK_TARGET_<arch>_` / `#if NK_TARGET_<microarch>` guards with the relevant compiler `target` pragma:

```c
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,f16c,fma,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "f16c", "fma", "bmi", "bmi2")
#endif
```
(Source: `_references/NumKong/include/numkong/dot/haswell.h:101-106`.)

So every function in the file is compiled with that target without per-function `__attribute__((target(...)))` annotation. Rust's equivalent is `#[target_feature(enable = "...")]` on each function — there is no module-level attribute, so each kernel function ends up with its own attribute.

### 1.2 Per-backend, per-metric instructions

Below is the cross-product NumKong actually implements. Cycle counts come from comments in the relevant headers (Haswell/Genoa/Ice Lake/Sapphire/M1 Firestorm/A76 numbers).

| Metric | Type  | Haswell (AVX2)                           | Skylake-X / Cascade Lake                | Ice Lake (VNNI)                            | NEON                                            | NEON+SDOT                       | SVE / SVE2                       |
|--------|-------|------------------------------------------|------------------------------------------|--------------------------------------------|-------------------------------------------------|---------------------------------|----------------------------------|
| Dot f32 | f32  | `_mm256_fmadd_ps` (5cy@p01)              | `_mm512_fmadd_ps` (4cy@p05)              | same as Skylake                            | `vfmaq_f32` (4c)                                | same                            | `svmla_f32_x` FMLA               |
| Dot f64 | f64  | `_mm256_fmadd_pd` + Dot2 compensation    | `_mm512_fmadd_pd` + Dot2                 | same                                       | `vfmaq_f64`                                     | same                            | `svmla_f64_x`                    |
| Dot i8 | i8   | `_mm256_madd_epi16` (after `cvtepi8_epi16`) — VPMADDWD  | same widened path, 512-bit               | `_mm512_dpbusd_epi32` (VPDPBUSD) + `xor 0x80` bias trick | widening `vmlal_s8` + `vpaddl_s16` | `vdotq_s32` (SDOT, 3cy 2/cy)    | `svdot_s32` (SVDOT)              |
| L2² f32 | f32 | `_mm256_sub_ps` + `_mm256_fmadd_ps`      | `_mm512_sub_ps` + FMA                    | same                                       | `vsubq_f32` + `vfmaq_f32`                       | same                            | `svsub_f32_x` + FMLA             |
| L2² i8 | i8   | widen → i16 → square → `_mm256_madd_epi16` | same (512-bit)                          | unsigned-bias + VPDPBUSD                    | `vsubl_s8` (i8→i16) + `vmlal_s16`                | SDOT after subtract             | SVDOT after subtract             |
| Cos f32 | f32 | three FMA accumulators (a·b, a², b²) + `_mm_rsqrt_ps` finalize | same (512-bit) + VRSQRT14 | same | three FMLA accumulators + `vrsqrteq_f32` + 2 NR | same | three FMLA + svrsqrte           |
| Hamming u1 | bits | `_mm_popcnt_u64` per 8-byte chunk after extracting 4 64-bit lanes from a YMM XOR | `VPOPCNTQ` if VPOPCNTDQ enabled — else PSHUFB nibble LUT | native `_mm512_popcnt_epi64` | `vcntq_u8` (CNT) + accumulate up to 31 iterations into u8 lanes before widening to u32 | same | `svcnt_u8` predicated |
| Jaccard u1 | bits | XOR/AND/OR + same popcount path | VPOPCNTQ on AND and OR streams | native VPOPCNTQ on AND and OR | same as Hamming, two parallel accumulators | same | same |

References for each of those:
- Haswell f32 dot/L2/cos: `_references/NumKong/include/numkong/dot/haswell.h:107-200` (FMA fold + Dot2 compensation tree).
- Haswell i8 dot via `VPMADDWD`: `_references/NumKong/include/numkong/dot/haswell.h:30-40` and `_references/NumKong/include/numkong/spatial.h:101-105` ("AVX2 lacks signed 8-bit dot products, so Haswell widens to i16 and uses VPMADDWD").
- Ice Lake i8 dot via VPDPBUSD with the `XOR 0x80` bias: `_references/NumKong/include/numkong/dot/icelake.h:120-160`.
- Haswell binary metrics via scalar `_mm_popcnt_u64`: `_references/NumKong/include/numkong/set/haswell.h:60-80`.
- Ice Lake binary metrics via `_mm512_popcnt_epi64` and masked load tail: `_references/NumKong/include/numkong/set/icelake.h:60-110`.
- NEON binary via `vcntq_u8` accumulated into u8 lanes for up to 31 iterations: `_references/NumKong/include/numkong/set/neon.h:60-105`.
- NEON SDOT for i8: `_references/NumKong/include/numkong/dot/neonsdot.h:1-90`.

### 1.3 i8 / u8 / f16 / bf16 / binary in detail

**i8 with VPDPBUSD (Ice Lake / Genoa / Sapphire / Alder).** VPDPBUSD multiplies 4 unsigned bytes by 4 signed bytes and accumulates into one `i32` lane — four 8-bit MACs per 32-bit lane per instruction, 16 lanes per ZMM, totaling 64 8-bit MACs per VPDPBUSD. Ice Lake throughput is 1cy@p0 (~64 MAC/cy); Genoa is 0.5cy@p01 (~128 MAC/cy). The instruction is "unsigned×signed", so signed×signed has to be re-derived. NumKong does an algebraic shift: write `a = (a + 128) − 128`, then `a · b = (a+128)·b − 128·Σb`. The first term is now unsigned×signed and fits VPDPBUSD; the bias correction is folded into one SAD-based reduction. Snippet from `dot/icelake.h:135-156`:

```c
__m512i a_biased_u8x64 = _mm512_xor_si512(a_i8x64, xor_mask_u8x64);
sum_ab_i32x16 = _mm512_dpbusd_epi32(sum_ab_i32x16, a_biased_u8x64, b_i8x64);
__m512i b_biased_u8x64 = _mm512_xor_si512(b_i8x64, xor_mask_u8x64);
sum_b_biased_i64x8 = _mm512_add_epi64(sum_b_biased_i64x8, _mm512_sad_epu8(b_biased_u8x64, zeros_u8x64));
...
nk_i64_t correction = 128LL * sum_b_biased - 16384LL * (nk_i64_t)count_rounded;
*result = (nk_i32_t)(ab_sum - correction);
```

**i8 without VNNI (Haswell / Skylake-X / NEON pre-A65).** Use `VPMADDWD` (unsigned×signed → i16, horizontally added → i32) twice: first `VPMADDUBSW` produces i16 with possible overflow only when both half-products are near +max simultaneously, then `VPMADDWD` against a `vec(1)` adds adjacent i16 pairs into i32 lanes. NumKong widens 8-bit operands to 16-bit using `_mm256_cvtepi8_epi16` (3cy@p5 on Haswell, 4cy@p1+p2 on Genoa) and feeds `_mm256_madd_epi16` (5cy@p01 on Haswell, 3cy@p01 on Genoa). The naive PMADDUBSW path is slightly faster but has an overflow corner that needs an "all-zero second element of pair" guard (see fgiesen blog [Why those particular integer multiplies?](https://fgiesen.wordpress.com/2024/10/26/why-those-particular-integer-multiplies/)). For a search-engine kernel the widen-then-VPMADDWD path is the safer choice.

**f16 / bf16.** NumKong uses ISA-specific load-and-widen primitives. On Haswell, F16C `VCVTPH2PS` (5cy@p01) widens 8 f16 values to f32, then accumulates with FMA. On Skylake/Genoa, `_mm512_cvtepi16_epi32` + bit-shift gives bf16; on Sapphire Rapids, `_mm512_dpbf16_ps` (VDPBF16PS, 6cy@p01) and `_mm512_dpfp16_ps` (FP16-VNNI) hit native widening MAC throughput. On NEON, `vcvt_f32_f16` widens 4 f16 values to f32 in one cycle; ARMv8.4 `FMLAL2` (f16 widen-multiply-accumulate into f32) is ~20-48% faster than convert-then-FMA per the NumKong README §FP16. (Source: `_references/NumKong/README.md:393`.)

**Binary (packed bits / u1).** Hamming = `popcount(a XOR b)`; Jaccard = `1 − popcount(a AND b) / popcount(a OR b)`; Tanimoto is the same formula as Jaccard for binary inputs (NumKong/USearch maps `tanimoto_k → nk_kernel_jaccard_k` — see `_references/usearch/include/usearch/index_plugins.hpp:3019`). On x86 with AVX-512_VPOPCNTDQ, `_mm512_popcnt_epi64` is a single instruction per 8 u64 lanes. On AVX-512 without VPOPCNTDQ (Skylake-X without Cascade-Lake-AP), the Mula PSHUFB nibble lookup is the fallback. On AVX2 / SSE4.2, the canonical pattern is to extract 4 u64 lanes from each YMM and do scalar `_mm_popcnt_u64` (POPCNT is a separate uop on port 1, latency 3cy/throughput 1cy on Haswell; on Genoa POPCNT is 1cy@p0123). On NEON, `vcntq_u8` (CNT) gives per-byte popcount in 2cy; you accumulate into u8 lanes for up to 31 iterations (max u8 value 255 = 8 bits per byte × 31 iter < 256) before widening with `vaddlvq_u8` to avoid lane overflow.

### 1.4 Runtime dispatch shape

NumKong builds every kernel into the same binary and uses a runtime dispatcher. The capability bitmap is in `_references/NumKong/include/numkong/capabilities.h:261-294`:

```c
#define nk_cap_serial_k      ((nk_capability_t)1)
#define nk_cap_neon_k        ((nk_capability_t)1 << 1)
#define nk_cap_haswell_k     ((nk_capability_t)1 << 2)
#define nk_cap_skylake_k     ((nk_capability_t)1 << 3)
#define nk_cap_neonhalf_k    ((nk_capability_t)1 << 4)
#define nk_cap_neonsdot_k    ((nk_capability_t)1 << 5)
#define nk_cap_neonfhm_k     ((nk_capability_t)1 << 6)
#define nk_cap_icelake_k     ((nk_capability_t)1 << 7)
#define nk_cap_genoa_k       ((nk_capability_t)1 << 8)
#define nk_cap_neonbfdot_k   ((nk_capability_t)1 << 9)
#define nk_cap_sve_k         ((nk_capability_t)1 << 10)
#define nk_cap_alder_k       ((nk_capability_t)1 << 13)
#define nk_cap_sve2_k        ((nk_capability_t)1 << 15)
#define nk_cap_sapphire_k    ((nk_capability_t)1 << 17)
#define nk_cap_sapphireamx_k ((nk_capability_t)1 << 18)
...
```

USearch's integration with NumKong is `configure_with_numkong()` at `_references/usearch/include/usearch/index_plugins.hpp:3012-3055`. It calls `nk_find_kernel_punned(metric_kind, datatype, simd_caps, &simd_metric, &simd_kind)` and stores a function pointer in `metric_punned_t`. The dispatcher cost is one indirect call per distance evaluation. For HNSW search this is dwarfed by memory latency on graph traversal, so the pattern works well.

Per the NumKong README: *"The run-time path compiles every supported kernel into the binary and picks the best one on the target machine via `nk_capabilities()` — one pointer indirection per call, but a single binary runs everywhere."* (Source: `_references/NumKong/README.md`, §Runtime Dispatch.)

### 1.5 Published throughput

From `_references/NumKong/README.md:14-22` (single 2048-d dot product, single-threaded, Intel Sapphire Rapids):

| Input | NumPy + OpenBLAS | PyTorch + MKL | JAX | NumKong |
| :---- | ---------------: | ------------: | --: | ------: |
| `f64`  | 2.0 gso/s, 1e-15 err | 0.6 gso/s, 1e-15 err | 0.4 gso/s, 1e-14 err |  5.8 gso/s, 1e-16 err |
| `f32`  | 1.5 gso/s, 2e-6 err  | 0.6 gso/s, 2e-6 err  | 0.4 gso/s, 5e-6 err  |  7.1 gso/s, 2e-7 err  |
| `bf16` |                    — | 0.5 gso/s, 1.9% err  | 0.5 gso/s, 1.9% err  |  9.7 gso/s, 1.8% err  |
| `f16`  | 0.2 gso/s, 0.25% err | 0.5 gso/s, 0.25% err | 0.4 gso/s, 0.25% err | 11.5 gso/s, 0.24% err |
| `e5m2` |                    — | 0.7 gso/s, 4.6% err  | 0.5 gso/s, 4.6% err  |  7.1 gso/s, 0% err    |
| `i8`   | 1.1 gso/s, overflow  | 0.5 gso/s, overflow  | 0.5 gso/s, overflow  | 14.8 gso/s, 0% err    |

(`gso/s` = giga scalar operations per second, counting both integer and floating-point work uniformly.)

The older SimSIMD blog post [SciPy distances... up to 200x faster with AVX-512 & SVE](https://ashvardanian.com/posts/simsimd-faster-scipy/) reports for a 1536-d cosine distance (the OpenAI Ada embedding size) on Intel Sapphire Rapids:
- f32: 2.84 M ops/s (49.9× SciPy)
- f16: 4.14 M ops/s (242× SciPy)
- i8:  7.63 M ops/s (106× SciPy)

On AWS Graviton 3 (NEON + SVE):
- f32: 2.69 M ops/s (86× SciPy)
- f16: 3.03 M ops/s (206× SciPy)

On Apple M2 Pro (NEON only):
- i8 cosine: 15.4 M ops/s (190× SciPy)

The "200×" headline comes from the smaller-dtype paths where SciPy's reference is unvectorized Python+f32 and SimSIMD compares against a native f16/i8 widened path.

---

## 2. FAISS distance kernels

URL: https://github.com/facebookresearch/faiss.
License: MIT (Source: README in `facebookresearch/faiss`).

### 2.1 `distances_simd.cpp` organization

FAISS now uses a **template-based dispatch** pattern centered on `faiss/impl/simd_dispatch.h`:

```cpp
enum class SIMDLevel { NONE, AVX2, AVX512, AVX512_SPR, ARM_NEON, ARM_SVE, RISCV_RVV, ... };

// distances_simd.cpp:
#include <faiss/impl/simdlib/simdlib_dispatch.h>
#define THE_SIMD_LEVEL SIMDLevel::NONE
#include <faiss/utils/simd_impl/distances_autovec-inl.h>
#include <faiss/utils/simd_impl/distances_simdlib256.h>
```

The same source is included once per `THE_SIMD_LEVEL`, producing scalar/AVX2/AVX-512/AVX-512-SPR/NEON/SVE/RVV variants of every kernel. Then `distances_dispatch.h` defines a `with_simd_level<F>(F&& f)` higher-order function that resolves to the best available variant at runtime via `SIMDConfig::level`. Build option `FAISS_ENABLE_DD` (dynamic dispatch) toggles between runtime selection and compile-time selection of a single `SINGLE_SIMD_LEVEL`.

Subdirectory layout (from the GitHub directory listing):

```
faiss/utils/
  distances.cpp / .h                -- public fvec_L2sqr, fvec_inner_product, etc.
  distances_simd.cpp                -- per-SIMDLevel implementations (single TU, multi-include)
  distances_dispatch.h              -- runtime dispatcher
  extra_distances.cpp / .h
  distances_fused/                  -- fused L2sqr + reservoir-pick (search hot paths)
    avx512.h, ...
  simd_impl/                        -- per-target headers included by distances_simd.cpp
    distances_autovec-inl.h
    distances_simdlib256.h          -- 256-bit code path (used by AVX2 + 256-bit-AVX-512)
faiss/impl/
  simd_dispatch.h                   -- SIMDLevel enum, dispatcher
  simdlib/                          -- 128/256/512-bit wrapper types
```

### 2.2 `fvec_L2sqr` and `fvec_inner_product` SIMD variants

The pattern across FAISS variants is: load two SIMD vectors → subtract (L2) or skip (IP) → multiply + accumulate via FMA → at the end perform horizontal reduction. For a 256-bit AVX2 L2 squared:

```cpp
// pseudocode of the template body
__m256 acc = _mm256_setzero_ps();
for (size_t i = 0; i + 8 <= d; i += 8) {
    __m256 mx = _mm256_loadu_ps(x + i);
    __m256 my = _mm256_loadu_ps(y + i);
    __m256 diff = _mm256_sub_ps(mx, my);
    acc = _mm256_fmadd_ps(diff, diff, acc);
}
// horizontal reduce: hadd or extract-then-add
```

For the inner product variant, the body is `acc = _mm256_fmadd_ps(mx, my, acc);` — exactly the same shape with no subtract.

### 2.3 The "manual loop unrolling + horizontal reduction" pattern

FAISS unrolls 2–4× in the hot path (`fvec_L2sqr_ny_y_transposed_D` processes 8 vectors in parallel for some D values). The horizontal reduction at the end uses the standard 256-bit reduce: `_mm256_extractf128_ps(acc, 1)` + `_mm_add_ps(low, high)` → 128-bit `_mm_hadd_ps` × 2 → scalar. For AVX-512 the equivalent is `_mm512_reduce_add_ps` (which the compiler emits as a tree of extract+add).

The autovec backend (`SIMDLevel::NONE`) uses `#pragma omp simd reduction(+:s)` on a scalar loop and lets the compiler vectorize, which is the fallback when no SIMD level matches.

---

## 3. hnswlib distance functions

URL: https://github.com/nmslib/hnswlib.
License: Apache-2.0.

### 3.1 `hnswalg.h` graph traversal — distance is the inner loop

The relevant files are `hnswlib/space_l2.h`, `hnswlib/space_ip.h` (inner product), and `hnswlib/hnswalg.h` (graph traversal). The algorithm calls `fstdistfunc_(query, candidate, dist_func_param_)` once per candidate examined during `searchBaseLayerST`. Per the hnswlib README the per-distance cost is the dominant cost during search.

### 3.2 Template + intrinsic dispatch pattern

hnswlib selects the implementation **once at index construction time**, based on dimensionality alignment. From the `L2Space` constructor:

```cpp
if (dim % 16 == 0)
    fstdistfunc_ = L2SqrSIMD16Ext;
else if (dim % 4 == 0)
    fstdistfunc_ = L2SqrSIMD4Ext;
else if (dim > 16)
    fstdistfunc_ = L2SqrSIMD16ExtResiduals;
else if (dim > 4)
    fstdistfunc_ = L2SqrSIMD4ExtResiduals;
```

Within each variant, the build picks the widest available SIMD via macros `USE_AVX512`, `USE_AVX`, `USE_SSE` evaluated in `space_l2.h`. The `*Residuals` functions handle `dim % chunk != 0` by computing the aligned chunk via SIMD then scalar-processing the remainder.

Representative AVX2 inner loop body for `L2SqrSIMD16ExtAVX` (snippet via WebFetch of `space_l2.h`):

```c
v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
diff = _mm256_sub_ps(v1, v2);
sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
```

The block executes twice per outer iteration (16 floats per iteration). Note this is `mul+add`, not FMA — hnswlib still ships pre-FMA-aware code paths because some early HNSW deployments target older Intel chips. A modern Rust kernel should always use `_mm256_fmadd_ps` / `vfmaq_f32` instead.

Horizontal reduction in hnswlib is *not* `_mm256_reduce`-style; it does:

```c
_mm256_store_ps(TmpRes, sum);
return TmpRes[0] + TmpRes[1] + ... + TmpRes[7];
```

i.e. spill to stack and scalar-add. This is suboptimal — `_mm256_extractf128_ps` + `_mm_hadd_ps` × 2 is faster on every modern CPU — but for HNSW search this overhead is amortized across the per-call 6-8 cache-miss latency for fetching the candidate vector, so it does not show up at the system level.

### 3.3 What backends are covered

- AVX-512 (`USE_AVX512`): `L2SqrSIMD16ExtAVX512`, `InnerProductSIMD16ExtAVX512`, etc.
- AVX (`USE_AVX`): 256-bit variants (NOT FMA — `_mm256_mul_ps + _mm256_add_ps`)
- SSE (`USE_SSE`): 128-bit variants

NEON support is community-PR'd but not in the canonical release. There is no `USE_NEON` macro in upstream `space_l2.h`. For a Rust HNSW we cover NEON natively, so we are ahead of hnswlib here.

Inner product distance is `1 - InnerProduct(...)` (cosine on already-normalized vectors). hnswlib does not normalize at index time; the user must normalize externally if cosine semantics are wanted.

---

## 4. AVX-512 specifics for distance metrics

### 4.1 VPDPBUSD (AVX-512 VNNI) — the i8 dot product instruction

VPDPBUSD multiplies 4 unsigned bytes by 4 signed bytes per 32-bit lane and accumulates into the same 32-bit lane. Per Intel: *"VPDPBUSD multiplies the individual bytes (8-bit) of the first source operand by the corresponding bytes (8-bit) of the second source operand, producing intermediate word (16-bit) results which are summed and accumulated in the double word (32-bit) of the destination operand."* It fuses what previously took VPMADDUBSW + VPMADDWD + VPADDD into one instruction, so a 64-MAC kernel goes from ~3 instructions per 64 ops to ~1. (Source: [Intel: Deep Learning with AVX-512 and DL Boost](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html), [WikiChip AVX-512 VNNI](https://en.wikichip.org/wiki/x86/avx512_vnni).)

Latencies (from NumKong `dot/icelake.h:11-15`):

```
Intrinsic            Instruction               Icelake    Genoa
_mm512_dpbusd_epi32  VPDPBUSD (ZMM, ZMM, ZMM)  5cy @ p0   4cy @ p01
_mm512_dpwssd_epi32  VPDPWSSD (ZMM, ZMM, ZMM)  5cy @ p0   4cy @ p01
_mm512_madd_epi16    VPMADDWD (ZMM, ZMM, ZMM)  5cy @ p05  3cy @ p01
```

On Ice Lake VPDPBUSD bottlenecks on port 0 (1/cy throughput). On Genoa it dual-issues on ports 0+1 (0.5/cy throughput).

The signed×signed reformulation (because VPDPBUSD is unsigned×signed) is the Mula/Vardanian "XOR 0x80 bias" trick described in §1.3 above. Required CPUID: `AVX512_VNNI` (CPUID.7.1:ECX.AVX-VNNI[bit 4] for the 256-bit AVX-VNNI variant on Alder Lake; CPUID.7.0:ECX.AVX512_VNNI[bit 11] for the AVX-512 variant on Ice Lake+).

The `_ds` / `VPDPBUSDS` (saturating) variant clamps overflow to INT32_MAX/MIN instead of wrapping. For HNSW i8 distance it is irrelevant — for 1024-dim signed-byte vectors with values in [-127, +127], the maximum dot product is `1024 × 127 × 127 ≈ 1.65e7`, far below INT32_MAX, so non-saturating VPDPBUSD is the correct pick. Saturating is needed only for sequences > ~130k dim.

### 4.2 VPOPCNTQ (AVX-512_VPOPCNTDQ)

VPOPCNTDQ adds two instructions, `VPOPCNTD` (16 32-bit popcounts in parallel per ZMM) and `VPOPCNTQ` (8 64-bit popcounts in parallel). It was introduced in Knights Mill and Ice Lake server. Detection: `CPUID.7.0:ECX.AVX512_VPOPCNTDQ[bit 14]`. (Source: [Wikipedia: AVX-512](https://en.wikipedia.org/wiki/AVX-512), [WikiChip AVX-512 BITALG](https://en.wikichip.org/wiki/x86/avx512_bitalg).)

NumKong `set/icelake.h:11-13` reports:
```
_mm512_popcnt_epi64  VPOPCNTQ (ZMM, ZMM)  3cy @ p5   (Ice Lake)
                                          2cy @ p01  (Genoa)
```

For a 4096-bit binary vector (64 bytes = 1 ZMM) Hamming = `popcnt(a^b)` is a 1-instruction operation plus reduce.

For very large binary vectors (>64KiB), the **Harley-Seal carry-save adder** algorithm is the canonical optimization. It uses 3-input CSA (one instruction with `_mm512_ternarylogic_epi64`) to compress N popcounts down by a factor of ~3×, then runs VPOPCNTQ on the compressed stream. Per NumKong `set.h:55-70` (cycles per byte):

```
Method               Buffer    Ice Lake    Sapphire    Genoa
Native VPOPCNTQ      any       ~0.12       ~0.10       ~0.06
Harley-Seal CSA      1 KB       0.107       0.095       0.08
Harley-Seal CSA      4 KB       0.056       0.052       0.05
VPSHUFB lookup       4 KB       0.063       0.058       0.07
```

For HNSW binary vectors ≤4096 bits (the typical embedding-quantization size), native VPOPCNTQ wins; Harley-Seal only pays off for ≥1KB binary vectors.

### 4.3 VPDPBUSDS (saturating variant)

Same opcode family as VPDPBUSD; result is saturated to int32 instead of wrapping. For HNSW distance kernels: not relevant unless someone is trying to do dim > ~130k i8 dot product, which never happens in real embeddings. Skip in v1.

### 4.4 GFNI (Galois Field New Instructions)

GFNI adds three instructions: `VGF2P8AFFINEQB`, `VGF2P8AFFINEINVQB`, `VGF2P8MULB`. They're in AVX-512 GFNI (CPUID.7.0:ECX.GFNI[bit 8]) on Ice Lake / Tiger Lake / Genoa, and also in 128/256-bit variants on Tremont (Goldmont Plus). (Source: [Intel GFNI Technology Guide](https://builders.intel.com/docs/networkbuilders/galois-field-new-instructions-gfni-technology-guide-1-1639042826.pdf).)

For binary distance kernels GFNI is **not directly useful**: it operates in GF(2⁸), not GF(2). The affine transform `VGF2P8AFFINEQB` *can* be repurposed for arbitrary 8×8-bit permutations within a byte (bit reversal, byte rotation, etc.), which is occasionally used for SIMD bit-tricks like efficient byte-bit-transpose, but it does not accelerate Hamming/Jaccard. NumKong tracks `_mm512_gf2p8mul_epi8` as 5cy@p0 on Ice Lake / 3cy@p01 on Genoa (`set.h:49`) but does not use it in the binary-distance hot paths. Defer GFNI; revisit only if we add a binary-fingerprint inverse-permutation use case.

### 4.5 Reduce instructions and horizontal sum patterns

AVX-512 provides `_mm512_reduce_add_ps`, `_mm512_reduce_add_pd`, `_mm512_reduce_add_epi32`, `_mm512_reduce_add_epi64` as portable horizontal-reduce intrinsics. Compilers emit the standard tree: `_mm512_extracti64x4_epi64` to split into two 256-bit halves, `_mm256_add_*`, then 256→128 extract+add, then SSE3 `_mm_hadd_*` × 2.

For per-call kernels (HNSW search hot path), do the reduction once at the end. Do NOT reduce inside the loop. NumKong follows this pattern: accumulate into a single ZMM (or two for 2-way unroll, four for 4-way) and reduce at finalize time.

### 4.6 Mask register usage for tail handling

AVX-512 has 8 mask registers (`k0`-`k7`, where `k0` is implicit "no mask"). `_bzhi_u64(0xFFFFFFFFFFFFFFFF, n)` produces a "first n bits set" mask in O(1), which feeds `_mm512_maskz_loadu_epi8` (or `_ps`, `_pd`, etc.) for tail handling without a separate scalar fallback. NumKong uses this aggressively; from `set/icelake.h:62`:

```c
__mmask64 mask = (__mmask64)_bzhi_u64(0xFFFFFFFFFFFFFFFF, n_bytes);
__m512i a_u8x64 = _mm512_maskz_loadu_epi8(mask, a);
__m512i b_u8x64 = _mm512_maskz_loadu_epi8(mask, b);
```

This gives a branch-free tail handler for any size from 1 to 64 bytes. The cost is one extra `BZHI` (1cy@p15) per kernel call. The corresponding pre-AVX-512 pattern is harder: either a separate scalar tail loop, or a `_mm256_maskload_ps` with a manually-constructed mask (which costs more than the AVX-512 path).

---

## 5. AVX2 specifics

### 5.1 PMADDUBSW + PMADDWD chain for i8 dot

Without VNNI, the canonical i8 dot product on AVX2 is:

1. Widen i8 → i16 with `_mm256_cvtepi8_epi16` (or treat operands as PMADDUBSW-friendly directly with care)
2. Multiply pairs and horizontally add adjacent pairs into i16 with `_mm256_maddubs_epi16` (PMADDUBSW), or
3. Widen with `cvtepi8` then `_mm256_madd_epi16` (PMADDWD) which produces i32 lanes safely

The PMADDUBSW path has a saturation pitfall: each PMADDUBSW lane is `clip(a₀·b₀ + a₁·b₁, INT16)`, and `255·-128 = -32640` while INT16_MIN = -32768, so two near-extreme products in the same pair can saturate. The fix is to interleave operands so one byte of each pair is always zero, halving throughput. The widen-then-PMADDWD path (PMADDWD lanes are i16×i16 → i32, no saturation possible for 8-bit inputs) is safer at one extra widening cost. (Source: [fgiesen "Why those particular integer multiplies?"](https://fgiesen.wordpress.com/2024/10/26/why-those-particular-integer-multiplies/).)

NumKong's `dot/haswell.h` chooses the widen-then-PMADDWD path for safety, accepting a slight throughput penalty.

### 5.2 PMADDWD: i16×i16 → i32 horizontally added

`_mm256_madd_epi16` (PMADDWD) takes two i16x16 vectors and produces 8 i32 lanes, each containing `a₂ᵢ·b₂ᵢ + a₂ᵢ₊₁·b₂ᵢ₊₁`. This is the workhorse for sub-VNNI integer dot products: each instruction does 16 8-bit-equivalent MACs into 8 i32 accumulators.

Throughput: 5cy@p01 on Haswell, 3cy@p01 on Genoa.

### 5.3 No native VPOPCNTQ on AVX2 — Mula PSHUFB algorithm

AVX2 has no per-lane popcount. The canonical workaround is the **Wojciech Mula nibble-LUT algorithm** (`_mm256_shuffle_epi8` + bitwise AND + shift). Per the algorithm:

1. Load a precomputed 16-byte LUT containing popcount of nibbles 0..15: `[0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4]`
2. Mask low nibble of each byte (`AND 0x0F`), shuffle into LUT → low-nibble popcount
3. Shift right 4, mask, shuffle → high-nibble popcount
4. Add → per-byte popcount
5. Reduce per-byte counts via `_mm256_sad_epu8` (SAD against zero) → per-64-bit sums
6. Sum across lanes

Per Mula et al. *"Faster Population Counts Using AVX2 Instructions"* (arXiv:1611.07612), this beats scalar POPCNT by ~2× on large arrays (0.52 vs 1.02 cycles per 8 bytes). At least Clang's libc++ uses this pattern internally. Implementations in WojciechMula/sse-popcount (https://github.com/WojciechMula/sse-popcount) and libpopcnt (https://github.com/kimwalisch/libpopcnt). For HNSW binary vectors of dim ≤ ~2048, the simpler scalar-extract-and-POPCNT approach NumKong uses on Haswell is competitive because the vector fits in 32 bytes and the loop overhead dominates the popcount cost. Switch to the Mula PSHUFB pattern only if profiling shows popcount-port saturation.

### 5.4 L2² distance: PMULLD + PADDD vs FMA-based on f32

For f32 L2², the FMA-based path is best: `acc = fma(diff, diff, acc)` in one instruction (5cy@p01 Haswell, 4cy@p01 Genoa). Without FMA, separate `_mm256_mul_ps + _mm256_add_ps` is two instructions and breaks the dependency chain on the accumulator. hnswlib's pre-FMA AVX path uses the latter; we should always emit FMA (it's been baseline since Haswell 2013, so requiring `target_feature(enable = "avx2,fma")` is fine).

For i32 L2² (rare — used only when the source is u8/i8 and we choose to widen all the way to i32 instead of staying in i16), `_mm256_mullo_epi32` (PMULLD) is 10cy@p0 on Haswell — slow. Prefer to stay in i16 (PMULLW is 5cy@p0) or use the widen + PMADDWD trick.

### 5.5 Horizontal reduce patterns

The standard AVX2 horizontal sum of 8 f32 lanes:

```c
__m128 lo = _mm256_castps256_ps128(acc);
__m128 hi = _mm256_extractf128_ps(acc, 1);
__m128 sum128 = _mm_add_ps(lo, hi);
sum128 = _mm_hadd_ps(sum128, sum128);
sum128 = _mm_hadd_ps(sum128, sum128);
float result = _mm_cvtss_f32(sum128);
```

Or using shuffle+add for Skylake+ (where HADDPS is slow):
```c
__m128 shuf = _mm_movehdup_ps(sum128);
sum128 = _mm_add_ps(sum128, shuf);
shuf = _mm_movehl_ps(shuf, sum128);
sum128 = _mm_add_ss(sum128, shuf);
```

Avoid hnswlib's "store + scalar add" pattern — it's slower on every CPU.

---

## 6. NEON specifics for AArch64

### 6.1 VCNT — 8-bit per-lane popcount

`vcntq_u8` (CNT) gives popcount per byte, 2cy throughput on Cortex-A76 / Apple M-series. The maximum per-byte count is 8. To accumulate across many vectors without per-byte overflow, NumKong's pattern (`set/neon.h:65-80`) is:

1. Maintain a `uint8x16_t` accumulator initialized to 0
2. Inner loop runs up to **31 iterations** (31 × 8 = 248 < 255), adding 16 byte-popcounts per iteration
3. After 31 iterations, widen with `vaddlvq_u8` (ADDLV — 4cy on A76) into a single u32 and add to outer accumulator
4. Continue

For a 4096-bit (512-byte) Hamming, that's 32 outer loops × 16 bytes = 512 bytes, fitting in two outer iterations.

### 6.2 SDOT / UDOT (FEAT_DotProd, ARMv8.4-A)

`vdotq_s32` (SDOT) and `vdotq_u32` (UDOT) compute 4 dot products of 4-element 8-bit vectors, accumulating into 4 i32 lanes — i.e. 16 8-bit MACs per instruction. Required feature flag: `+dotprod`. CPUs that ship it: Cortex-A75/A76+, Cortex-A55 (Armv8.2 with DotProd extension), Apple M1+, AWS Graviton2+. (Source: [Arm Developer: Exploring the Arm dot product instructions](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/exploring-the-arm-dot-product-instructions), [SDOT (vector) AArch64](https://developer.arm.com/documentation/100069/0609/A64-SIMD-Vector-Instructions/SDOT--vector-).)

Latency: 3cy at 2/cy throughput on Cortex-A76, 3cy at 4/cy on Apple M5 (NumKong `dot/neonsdot.h:11-12`).

NEON SDOT, unlike VPDPBUSD, is symmetric (signed×signed and unsigned×unsigned variants exist directly) — no XOR-0x80 trick needed.

### 6.3 VMULL_S8 / VMLAL_S8 — i8 multiply with i16 accumulation (pre-DotProd fallback)

For ARMv8 NEON without DotProd (Cortex-A53, A55 base, etc.), the canonical i8 dot is:

1. `vmull_s8` (SMULL) widens 8 i8×i8 multiplications to 8 i16
2. `vmlal_s8` (SMLAL) does the next batch, accumulating to existing i16 lanes
3. Periodically widen the i16 accumulator to i32 with `vpaddlq_s16` to avoid overflow
4. Final reduce with `vaddvq_s32`

This matches NumKong's pre-DotProd `dot/neon.h` pattern. The widening cadence depends on operand range: for full-range i8 in [-128, +127], `i16` saturates after `32768 / (127·127) ≈ 2` MACs per lane, so widen every 2 iterations; for typical embedding scales of ±64 quantized, widen every ~8.

### 6.4 FMLA — f32 fused multiply-accumulate

`vfmaq_f32` (FMLA.4S) is the f32 FMA. Latency 4cy on Apple M1 Firestorm and most Cortex-A. Throughput typically 2/cy. This is the workhorse for f32 dot product, L2², and cosine. Use 2-way to 4-way accumulator unroll to hide the 4cy latency.

### 6.5 SVE / SVE2 / SVDOT (deferred for v1, useful to plan for)

SVE is vector-length-agnostic; the vector width depends on the implementation (128-bit on Apple M4+, 256-bit on Graviton 3, 512-bit on Fugaku). SVE provides `svdot_s32` / `svdot_u32` (SDOT/UDOT predicated for SVE), `svmla_f32_x` (FMLA), and `svaddv_f32` (FADDV horizontal reduce, 6cy on Neoverse V1).

The SVE programming idiom is **predicate-based loop control**: `svwhilelt_b32(i, n)` produces a "first (n-i) lanes" predicate, and every load/op can be predicate-masked, eliminating the tail handler entirely. From NumKong's `dot/sve.h:47-50`:

```c
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
```

For Rust, SVE intrinsics in `core::arch::aarch64` are still mostly nightly. SVE2 broadens the integer instruction set further. Per the spec we defer SVE2; the AArch64 path covers NEON+SDOT, which is enough for Apple M-series and Graviton 2+.

---

## 7. SSE4.1 / SSSE3 fallbacks

### 7.1 PSHUFB-based popcount (Mula's algorithm, SSSE3 origin)

The Mula nibble-LUT popcount algorithm was originally an SSSE3 algorithm — `_mm_shuffle_epi8` (PSHUFB) is SSSE3, and the entire algorithm fits in SSSE3+SSE2. The AVX2 version is a straight 256-bit lift. References: [Wojciech Mula's original 2008 page](https://github.com/WojciechMula/sse-popcount), [Faster Population Counts paper (arXiv:1611.07612)](https://arxiv.org/pdf/1611.07612).

### 7.2 PMADDUBSW for i8 dot — same pattern as AVX2

`_mm_maddubs_epi16` is the SSSE3 version. Same overflow caveat as the AVX2 PMADDUBSW path. For SSE4.1 i8 dot, use the same widen-then-PMADDWD pattern.

### 7.3 PMULLD for i32 multiply

`_mm_mullo_epi32` is SSE4.1 (introduced with Penryn in 2007). Without it (SSSE3 only), you have to combine `_mm_mul_epu32` (which only does even lanes) twice with shuffles — slow. So SSE4.1 is the realistic floor for any new code; SSSE3 should be reserved for binary popcount paths only.

### 7.4 POPCNT on SSE4.2 — should we use it for binary metrics?

`_mm_popcnt_u64` (POPCNT) is SSE4.2 (Nehalem 2008, AMD Bulldozer 2011). On Haswell it's 3cy@p1 throughput 1cy (port-1 bottleneck). On Genoa it's 1cy@p0123 — much faster. Dispatch policy:

- AVX-512 + VPOPCNTDQ available → use VPOPCNTQ
- AVX2 + POPCNT available, vector ≤ ~256 bits → scalar 4× POPCNT (NumKong's Haswell pattern)
- AVX2 + POPCNT available, vector ≥ ~512 bits → Mula PSHUFB or Harley-Seal (port-1 saturation matters at scale)
- SSE4.2 only → scalar POPCNT
- SSSE3 only (very old hardware) → Mula PSHUFB

For our SSE4.1 backend (no SSE4.2 implied), assume scalar `usize::count_ones()` on Rust will compile down to POPCNT if `target_feature = "popcnt"` is enabled separately. We should *gate* the SSE4.1 binary path to also require `popcnt` (POPCNT is widely available; Intel/AMD CPUs without it are pre-2008).

---

## 8. Quantization distance tricks

### 8.1 i8 L2² without i32 overflow

For i8 vectors, `(a - b)²` per lane is in `[0, 65025]` (=255²). Accumulating `dim` such squares fits in i32 if `dim < 2³¹ / 65025 ≈ 33,000`. So for any embedding dim ≤ ~33k (which covers everything we'll ever encode), an i32 accumulator is safe per the NumKong README §Numerical Stability.

Pattern:
1. Widen i8 → i16 with `cvtepi8_epi16` (or `vmovl_s8`)
2. Subtract: `i16 - i16 → i16` (range [-510, +510])
3. PMADDWD-style: square and pairwise-add → i32

NumKong's i8 L2² returns u32 (`nk_sqeuclidean_i8` signature: `nk_u32_t *result`) because the result is non-negative and fits in u32 for any reasonable dim. The cosine path returns f32 because of the rsqrt finalize.

### 8.2 Cosine via dot + magnitude precomputation

The HNSW canonical optimization: at index build time, precompute and store each vector's L2 norm (or its inverse). At query time, distance is `1 − (a · b) / (‖a‖·‖b‖)`. With both norms baked in, the per-call cost is one dot product + one fmul + one fdiv + one subtract — vs three accumulator passes for a naive cosine.

Even better: store inverse norm `1/‖a‖`, then `cosine_dist = 1 − (a · b) · inv_norm_a · inv_norm_b`, replacing the divide with two multiplies.

Better still for HNSW: pre-normalize all vectors at index build time to unit length. Then cosine collapses to `1 − (a · b)` and is identical to `IP` distance with the sign inverted. USearch follows this convention (see `metric_kind_t::cos_k` and `metric_kind_t::ip_k` in `_references/usearch/include/usearch/index_plugins.hpp:114-128`).

Caveat: pre-normalization loses per-vector magnitude, which is information-bearing for some embedding models (e.g. some E5 / BGE variants). Provide both modes; default to "user-provides-normalized" for IP/cosine.

### 8.3 Binary Tanimoto vs Jaccard — formula difference

For binary vectors:

- **Jaccard distance** = 1 − |A ∩ B| / |A ∪ B| = 1 − popcount(A & B) / popcount(A | B)
- **Tanimoto coefficient** (chemistry / molecular fingerprints) is identical to Jaccard for binary sets. They are the same formula. (Source: USearch maps `tanimoto_k → nk_kernel_jaccard_k` in `_references/usearch/include/usearch/index_plugins.hpp:3019`. NumKong also reuses Jaccard kernels for Tanimoto inputs.)
- **Sorensen-Dice** distance = 1 − 2·|A ∩ B| / (|A| + |B|) = 1 − 2·popcount(A & B) / (popcount(A) + popcount(B)) — different denominator (sum of cardinalities, not union). USearch implements it separately as `metric_sorensen_gt` at `_references/usearch/include/usearch/index_plugins.hpp:2362`.

For non-binary vectors, "Tanimoto" can mean a real-valued generalization: `T(a, b) = (a·b) / (‖a‖² + ‖b‖² − a·b)`. We implement only the binary variant in v1.

The norm-precomputation trick from §1.3 applies: with `popcount(A)` and `popcount(B)` precomputed at index build,
```
|A ∪ B| = |A| + |B| − |A ∩ B|
```
so Jaccard reduces to **one** popcount call (the intersection) per distance evaluation instead of two. NumKong's "streaming API" exists exactly for this — `nk_jaccard_u1x512_init/update/finalize` accumulates only `popcount(A & B)` and finalizes with the precomputed norms.

### 8.4 Scalar quantization vs product quantization (PQ)

Linear/scalar quantization (LSQ) maps each f32 component to i8/u8 via a per-axis or per-vector scale. Distance kernels can run directly on the i8 representation; the per-element error is bounded by the scale. This is what we do for our `i8`/`u8` HNSW path.

Product quantization (PQ) splits the vector into M sub-vectors, quantizes each into a codebook of K codes (typically K=256), then stores M bytes per vector. Distance is computed via per-sub-vector lookup tables (LUTs). PQ kernels on AVX-512 use `_mm512_permutexvar_epi8` (VPERMB) to do simultaneous LUT lookups. FAISS implements PQ + IVF as its primary backend; USearch added PQ support in v2.x.

We are NOT doing PQ in v1. Documenting for context: if/when we add PQ, the per-backend kernel is "8 byte indices → 8 LUT lookups → sum of 8 floats", which on AVX-512 maps cleanly to VPERMB-of-LUT + VFMADD reduce. The AVX2 fallback uses a 4-byte-at-a-time gather or PSHUFB-LUT. The NEON path uses TBL/TBX. For binary fingerprints, "binary PQ" is not commonly used; the bit-packed format already achieves ~16× compression vs f32.

---

## 9. Cache-aware patterns

### 9.1 Streaming loads (NT prefetch) vs cached loads

Non-temporal loads (`_mm256_stream_load_si256`, `_mm512_stream_load_si512`, `vldnt1q_*` on NEON) bypass the cache, used when the data won't be reused soon. For HNSW search:

- The **query vector** is reused across every candidate evaluation in a search. **Use cached loads.**
- Each **graph node's vector** is loaded once per search (typically), but is likely to be loaded again across queries (HNSW's small-world property means popular nodes are visited often). **Use cached loads, with software prefetch.**
- For **batch index build / brute-force one-pass scan**, candidate vectors are touched once and never again. **Use streaming loads** — see NumKong `set.h:91`: *"For large-scale batch processing where vectors won't be reused, consider non-temporal loads (`_mm512_stream_load_si512`) to bypass the cache and avoid pollution."*

### 9.2 Software prefetch for the next graph node

This is the *defining* optimization in hnswlib search and is the main reason hnswlib's search beats a naive HNSW. From hnswlib `hnswalg.h::searchBaseLayerST`:

```c
#ifdef USE_SSE
_mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
_mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
_mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_
             + offsetData_, _MM_HINT_T0);
#endif
```

The pattern: while computing distance to the current candidate (`data`), issue a `_MM_HINT_T0` prefetch to the **next candidate** (`*(data + 1)`)'s vector. Specifically:
- Prefetch the `visited_array` byte for the next candidate (1 cache line)
- Prefetch the next 64 bytes of `visited_array` (dispute: per [issue #397](https://github.com/nmslib/hnswlib/issues/397) this is partially redundant)
- Prefetch the next candidate's vector data

The hint is `_MM_HINT_T0` (load to all cache levels = L1+L2+L3). This relies on the distance-kernel runtime being just longer than the cache fetch latency (~30-200ns from L3, ~50-150ns from DRAM).

For our Rust HNSW kernels: expose a separate `distance_with_prefetch_next(query, current, next)` API where `next` is a hint for prefetch but is not loaded synchronously. The kernel issues the prefetch as its first instructions, then runs the actual distance computation against `current`. This pattern is cited in VSAG (VLDB 2025) and in *Compass* (USENIX OSDI 2025) as table-stakes for graph ANN.

The Rust intrinsic is `core::arch::x86_64::_mm_prefetch` (stable; x86 only) and `core::arch::aarch64::__pld` / `__pldl1keep` etc. (still mostly nightly under `stdarch_arm`); for AArch64 stable, the trick is to use `core::intrinsics::prefetch_read_data` (also nightly) or write inline-asm `prfm pldl1keep, [...]`.

### 9.3 Chunk size for the distance kernel

For HNSW, the vector dim is fixed per index, so the kernel knows `dim` at build time. There is no "chunk size" choice in the v1 sense — the kernel processes the full vector in one call and the loop trip count is `dim/SIMD_WIDTH` which is small (e.g. 1024 dim / 16 lanes = 64 iterations).

The relevant tuning question is **inner-unroll depth**. Per NumKong `dot/skylake.h`:

> *"To build memory-optimal tiled algorithms, this file defines... `nk_dot_f64x8_state_first, state_second, state_third, state_fourth`"*

i.e. 4-way independent accumulators, processing 4 (query, target) pairs in parallel per iteration. This hides the FMA's 4-cycle latency on Sapphire Rapids' single-port FMA (Skylake-X has dual-port, so 2-way is enough; client Skylake/Ice Lake has single-port, so 4-way is needed).

For **per-call HNSW search** (one query × one target at a time), 4-way unroll is over-spec — there is no second target available without changing the API. The right pattern is **`(unroll_factor) accumulators within the single (query, target) pair`**, hiding FMA latency by issuing 4 independent FMAs against 4 separate accumulators per loop iteration:

```c
acc0 = fma(q[i+0..7], t[i+0..7], acc0)
acc1 = fma(q[i+8..15], t[i+8..15], acc1)
acc2 = fma(q[i+16..23], t[i+16..23], acc2)
acc3 = fma(q[i+24..31], t[i+24..31], acc3)
i += 32
```

This requires `dim ≥ 4 × SIMD_WIDTH` to be worthwhile. For 128-bit SIMD (4 f32 lanes), `dim ≥ 16`; for 256-bit AVX2 (8 lanes), `dim ≥ 32`; for 512-bit AVX-512 (16 lanes), `dim ≥ 64`. All practical embeddings (≥ 128 dim) easily satisfy this.

### 9.4 Vector blob layout: interleaved vs separate

HNSW typically stores per-node `(vector, neighbor_list)` consecutively in `data_level0_memory_` (one contiguous arena). hnswlib's `size_data_per_element_` is `vector_size + sizeof(linklist)`, and the layout is `[neighbor_count(4B), neighbor_ids(M*4B), label(8B), vector(dim*sizeof(scalar))]` per node. Prefetch hits the vector once and the neighbor list once per node.

Alternative: store vectors in a separate flat array indexed by node ID (struct-of-arrays). This trades one extra indirection per fetch for better cache density when only the neighbor list is needed (e.g. during graph construction / pruning). For pure search-only workloads, the AoS layout (hnswlib default) is faster because both the vector and its neighbors arrive in adjacent cache lines.

For binary HNSW (where vectors are tiny — e.g. 256 bits = 32 bytes = 0.5 cache line), interleaving can pack the vector inline with the neighbor list with no padding cost, halving the cache lines touched per node.

---

## 10. State-of-the-art benchmarks

### 10.1 NumKong / SimSIMD published throughput

See §1.5 above.

For binary metrics specifically: NumKong's `set.h:62-66` (cycles per byte processed):
```
Method               1 KB        4 KB        Sapphire Rapids
Native VPOPCNTQ      ~0.10       ~0.10       ~0.10
Harley-Seal CSA      0.107       0.052       (best for ≥4KB)
VPSHUFB lookup       0.063       0.058
```

These translate to throughput ranges of 10-20 GB/s for binary distance on a single core of a recent Intel/AMD chip — i.e. ~ 1 binary-Hamming distance per 50-100 ns for 2048-bit vectors.

### 10.2 USearch BENCHMARKS.md numbers

From `_references/usearch/BENCHMARKS.md:30-50` (`f32` × 256-d, AWS `c7g.metal` Graviton 3, 64 cores, default M=16 / efC=128 / efS=64):

| Vectors      | Connectivity | EF @ A | EF @ S | Add QPS  | Search QPS | Recall @1 |
| :----------- | -----------: | -----: | -----: | -------: | ---------: | --------: |
| `f32`  x256  |           16 |    128 |     64 |   75,640 |    131,654 |     99.3% |
| `f32`  x256  |           12 |    128 |     64 |   81,747 |    149,728 |     99.0% |
| `f32`  x256  |           32 |    128 |     64 |   64,368 |    104,050 |     99.4% |

Quantization comparison (same vectors, same graph params):

| Vectors      | Add QPS | Search QPS | Recall @1 |
| :----------- | ------: | ---------: | --------: |
| `f32`  x256  |  87,995 |    171,856 |     99.1% |
| `f16`  x256  |  87,270 |    153,788 |     98.4% |
| `i8`   x256  | 115,923 |    274,653 |     98.9% |

Key observation: i8 search throughput is **1.6× higher** than f32 with **0.2% recall loss** — that's the headline argument for our i8 path.

### 10.3 FAISS / hnswlib comparison

FAISS's IVFFlat / IVFPQ benchmarks are documented in the FAISS wiki; for HNSW specifically, FAISS's HNSW backend (`HNSWFlat`) uses the same hnswlib-derived kernels and achieves ~the same per-distance throughput as hnswlib, with FAISS adding multithreaded query-batching on top. The hnswlib README claims ~95% recall@10 in 1-2ms/query on SIFT1M for f32 vectors. (Source: ANN-Benchmarks results page at https://ann-benchmarks.com/.)

ANN-Benchmarks rankings (recurring): hnswlib and USearch are typically Pareto-optimal across recall ranges; FAISS HNSW is competitive but slightly slower for high-recall regimes due to extra dispatch overhead.

### 10.4 Real ANN-Benchmarks results

Per ANN-Benchmarks (https://ann-benchmarks.com/), for SIFT1M (128-d u8, 1M vectors):
- hnswlib (M=12, efC=200, efS=10..400 sweep): ~95% recall@10 at 50,000 QPS single-thread
- USearch (M=16, efC=128, efS=64): ~99% recall@10 at ~30,000 QPS single-thread
- FAISS HNSW (M=32, efC=40, efS=16): ~92% recall@10 at ~40,000 QPS single-thread

Higher-dimensional datasets (DEEP1B 96-d, GIST 960-d, GloVe-100): all three converge to ~10,000-40,000 QPS at 95%+ recall.

For our v1 HNSW, hitting **~50% of hnswlib QPS at the same recall** would be a strong launch number; matching it on Apple M-series (where hnswlib lacks NEON in upstream) is the realistic stretch goal.

---

## 11. Rust-specific notes

### 11.1 `#[target_feature(enable = "...")]`

Per-function attribute that tells the compiler "this function may use these CPU features". The function body can then use intrinsics that require those features. The function becomes `unsafe` to call (callers must guarantee the runtime CPU has the features), unless the calling function also has the same `#[target_feature]` annotation. (Source: [Rust reference on attributes for code generation](https://doc.rust-lang.org/reference/attributes/codegen.html).)

Pattern for a kernel:

```rust
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_sqr_f32_avx2(a: &[f32], b: &[f32]) -> f32 { ... }

#[target_feature(enable = "avx512f")]
unsafe fn l2_sqr_f32_avx512(a: &[f32], b: &[f32]) -> f32 { ... }
```

NumKong's C/C++ equivalent is the file-scope `#pragma clang attribute push(__attribute__((target("..."))), apply_to = function)` — same effect, but applied to all functions in the file. In Rust we apply to each function manually, or use a build-system trick to compile separate `cfg!`-guarded modules with different target features.

### 11.2 `core::arch::x86_64::*` intrinsic naming

Rust's intrinsics mirror Intel's exactly: `_mm256_loadu_ps`, `_mm512_dpbusd_epi32`, `_mm_popcnt_u64`, etc. They live in `core::arch::x86_64` (and `core::arch::x86` for the shared subset). NEON intrinsics live in `core::arch::aarch64`: `vfmaq_f32`, `vdotq_s32`, `vcntq_u8`, etc.

### 11.3 `std::simd` vs `core::arch::*` in Rust 2026

- `core::arch::*` (intrinsics): **stable** for SSE/SSE2/SSE3/SSSE3/SSE4.1/SSE4.2/AVX/AVX2/FMA/POPCNT and most NEON. AVX-512 stabilized in **Rust 1.89 (2025-08-07)** under feature `stdarch_x86_avx512` — now stable. Per [Phoronix on Rust 1.89](https://www.phoronix.com/news/Rust-1.89-Released): *"Rust 1.89 stabilizes AVX-512 and several x86 crypto/target features."*
- `std::simd` (portable SIMD wrapper, formerly `packed_simd`): **still nightly only** as of Rust 1.89, with no near-term stabilization. Per Shnatsel's [State of SIMD in Rust in 2025](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d): *"std::simd remains nightly-only and will remain such for the foreseeable future."*
- Stable alternatives for portable SIMD: the `wide` crate (zero-cost portable f32x8/f64x4/etc. — depends on `safe_arch`); the `pulp` crate (with built-in multiversioning, used by `faer`); `simdeez`. None cover AVX-512 fully yet.

For our HNSW kernels, the recommendation is:

- Write each kernel directly against `core::arch::*` intrinsics (no portable abstraction)
- Mark kernel functions with `#[target_feature(enable = "...")]`
- Add a runtime dispatcher using `is_x86_feature_detected!` / `std::arch::is_aarch64_feature_detected!`
- Keep all kernels in the binary; the dispatcher picks the best at startup and stores a function pointer

This mirrors NumKong's pattern exactly and avoids depending on nightly Rust. The single nightly need is **AVX-512 intrinsics** before Rust 1.89 — but post-1.89 they are fully stable.

### 11.4 `#![feature(stdarch_*)]` requirements for nightly features

Pre-1.89 (Rust ≤ 1.88), each AVX-512 family had its own feature gate:
- `#![feature(stdarch_x86_avx512)]` — most AVX-512 (F, BW, DQ, VL, VNNI, BITALG, VPOPCNTDQ, etc.)
- `#![feature(stdarch_arm_neon_intrinsics)]` — newer NEON intrinsics

Post-1.89, the AVX-512 feature gate is gone for the stabilized subset. Some AVX-512 sub-extensions added later (AVX-512_FP16, AVX-512_BF16) may still need feature gates depending on rust version. Check the [rust-lang/rust issue #111137 tracking issue for AVX-512 intrinsics](https://github.com/rust-lang/rust/issues/111137) for the current cutoff.

For ARM SVE, intrinsics are still under `#![feature(stdarch_aarch64_sve)]` (nightly only in 2026). Our spec defers SVE2; the AArch64 NEON path covers SDOT/UDOT (stable) and FMLA/CNT (long-stable).

### 11.5 `is_x86_feature_detected!` and runtime dispatch

`is_x86_feature_detected!` is the canonical x86 runtime detector; it expands at compile time to either a constant `true` (if the feature is in the build's `-Ctarget-feature` set, allowing the compiler to elide the dispatch) or a `cpuid` runtime check on first call (cached after).

Supported feature literals include the relevant ones for distance kernels: `"avx2"`, `"fma"`, `"avx512f"`, `"avx512vnni"`, `"avx512vpopcntdq"`, `"avx512bitalg"`, `"gfni"`, `"sse4.1"`, `"ssse3"`, `"sse4.2"`, `"popcnt"`. (Source: [Rust std::arch macro is_x86_feature_detected](https://doc.rust-lang.org/std/arch/macro.is_x86_feature_detected.html).)

The AArch64 equivalent is `std::arch::is_aarch64_feature_detected!` with literals like `"neon"`, `"dotprod"`, `"fp16"`, `"sve"`, `"sve2"`, `"i8mm"`. On non-Linux AArch64 platforms (some macOS/iOS), runtime detection requires platform-specific `sysctl` calls; on Linux it uses `getauxval(AT_HWCAP)` / `AT_HWCAP2`. The macro abstracts this.

Dispatcher pattern (idiomatic Rust):

```rust
type DistFn = unsafe fn(&[f32], &[f32]) -> f32;

static DISPATCH: once_cell::sync::OnceCell<DistFn> = OnceCell::new();

fn pick_l2_f32() -> DistFn {
    #[cfg(target_arch = "x86_64")] {
        if is_x86_feature_detected!("avx512f") { return l2_sqr_f32_avx512; }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return l2_sqr_f32_avx2;
        }
        if is_x86_feature_detected!("sse4.1") { return l2_sqr_f32_sse41; }
    }
    #[cfg(target_arch = "aarch64")] {
        if std::arch::is_aarch64_feature_detected!("neon") { return l2_sqr_f32_neon; }
    }
    l2_sqr_f32_scalar
}

pub fn l2_sqr_f32(a: &[f32], b: &[f32]) -> f32 {
    let f = DISPATCH.get_or_init(pick_l2_f32);
    unsafe { f(a, b) }
}
```

Cost: one atomic load + one indirect call per distance evaluation. Dwarfed by the kernel's own runtime for any vector with dim ≥ 16.

For the **per-metric per-dtype matrix** (6 metrics × 4 dtypes × 6 backends = 144 functions), keep them organized like NumKong's `<op>/<arch>.rs` layout. The dispatcher then picks `(metric, dtype) → (best backend)`, returning a `DistFn`.

---

## Sources

- [Wojciech Mula et al., *Faster Population Counts Using AVX2 Instructions* (arXiv:1611.07612)](https://arxiv.org/pdf/1611.07612)
- [WojciechMula/sse-popcount on GitHub](https://github.com/WojciechMula/sse-popcount)
- [kimwalisch/libpopcnt](https://github.com/kimwalisch/libpopcnt)
- [Intel: Deep Learning with AVX-512 and DL Boost](https://www.intel.com/content/www/us/en/developer/articles/guide/deep-learning-with-avx512-and-dl-boost.html)
- [WikiChip x86 AVX-512 VNNI](https://en.wikichip.org/wiki/x86/avx512_vnni)
- [WikiChip x86 AVX-512 BITALG](https://en.wikichip.org/wiki/x86/avx512_bitalg)
- [Wikipedia AVX-512 (VPOPCNTDQ section)](https://en.wikipedia.org/wiki/AVX-512)
- [Intel GFNI Technology Guide](https://builders.intel.com/docs/networkbuilders/galois-field-new-instructions-gfni-technology-guide-1-1639042826.pdf)
- [ARM Developer: SDOT (vector) AArch64](https://developer.arm.com/documentation/100069/0609/A64-SIMD-Vector-Instructions/SDOT--vector-)
- [ARM Developer: Exploring the ARM dot product instructions](https://developer.arm.com/community/arm-community-blogs/b/tools-software-ides-blog/posts/exploring-the-arm-dot-product-instructions)
- [Fabian Giesen: *Why those particular integer multiplies?*](https://fgiesen.wordpress.com/2024/10/26/why-those-particular-integer-multiplies/)
- [Ash Vardanian: *NumKong — 2'000 Mixed Precision Kernels*](https://ashvardanian.com/posts/numkong/)
- [Ash Vardanian: *SciPy distances... up to 200x faster with AVX-512 & SVE*](https://ashvardanian.com/posts/simsimd-faster-scipy/)
- [GitHub: ashvardanian/NumKong](https://github.com/ashvardanian/NumKong) — Apache-2.0 OR BSD-3-Clause
- [GitHub: ashvardanian/SimSIMD](https://github.com/ashvardanian/SimSIMD) — predecessor project, same author, same license
- [GitHub: facebookresearch/faiss](https://github.com/facebookresearch/faiss) — MIT
- [GitHub: nmslib/hnswlib](https://github.com/nmslib/hnswlib) — Apache-2.0
- [GitHub: nmslib/hnswlib issue #397 — prefetch logic discussion](https://github.com/nmslib/hnswlib/issues/397)
- [Rust std::arch::is_x86_feature_detected](https://doc.rust-lang.org/std/arch/macro.is_x86_feature_detected.html)
- [Rust 1.89 release notes — AVX-512 stabilization](https://releases.rs/docs/1.89.0/)
- [Phoronix: Rust 1.89 Released With More AVX-512 Intrinsics](https://www.phoronix.com/news/Rust-1.89-Released)
- [Sergey Davidoff: *The state of SIMD in Rust in 2025*](https://shnatsel.medium.com/the-state-of-simd-in-rust-in-2025-32c263e5f53d)
- [ANN-Benchmarks](https://ann-benchmarks.com/)
- Local references in `_references/`:
  - `_references/NumKong/include/numkong/spatial.h` — spatial kernel public API
  - `_references/NumKong/include/numkong/dot/{haswell,skylake,icelake,sapphire,neon,neonsdot,sve}.h` — per-microarch dot kernels
  - `_references/NumKong/include/numkong/set/{haswell,icelake,neon}.h` — per-microarch binary set kernels
  - `_references/NumKong/include/numkong/set.h` — public binary set API and popcount strategy notes
  - `_references/NumKong/include/numkong/capabilities.h` — runtime capability bitmap
  - `_references/usearch/include/usearch/index_plugins.hpp` — metric kinds, NumKong dispatch glue
  - `_references/usearch/BENCHMARKS.md` — USearch benchmark numbers (Graviton 3)

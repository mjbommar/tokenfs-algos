# Phase 2 — SIMD distance kernels (AVX2 + NEON + SSE4.1) + iai benches

**Status:** plan, 2026-05-03. **Weeks 2-3 of the v0.7.0 HNSW landing.**

**Goal:** end-of-week-3, AVX2 i8 Hamming distance is at least 10× faster than the scalar reference, and iai-callgrind has per-kernel instruction-count bench rows in CI's regression gate. NEON parity with AVX2 on the integer/binary metrics. SSE4.1 fallback for older x86. Scalar parity tests pass on every backend × every (metric, scalar_kind) cell.

## Scope narrowing per primitive inventory

The primitive inventory ([`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md)) found that the **f32 distance kernels and binary-popcount kernels already exist in this crate** with full backend coverage:

- **`vector::dot_f32` / `vector::l2_squared_f32` / `vector::cosine_f32`** — scalar + AVX2 + AVX-512 + NEON already present and audit-R10-disciplined. HNSW reuses them via direct call (no wrapper, no copy). See `crates/tokenfs-algos/src/vector/`.
- **`bits::popcount_u64_slice` + per-backend `bits::popcount::kernels::*`** — covers the binary Hamming/Jaccard/Tanimoto inner loop. HNSW's `kernels::*::distance_hamming_binary_*` is a thin shim over these.

**What's actually new in Phase 2 is the i8/u8 quantized distance kernels and the SSE4.1/SSSE3 x86 fallbacks** for environments without AVX2. Those are the cells in the matrix below that aren't already covered by reuse. Phase 2's effort estimate stays at 2 weeks because the i8/u8 work + per-backend parity tests + iai bench rows is still the bulk of the work, but the design is materially simpler than implementing every cell from scratch.

## Deliverables

### Code

- `crates/tokenfs-algos/src/similarity/hnsw/kernels/avx2.rs`:
  - `unsafe fn distance_l2_squared_i8_unchecked(a, b, len) -> u32` (and `_u8`, `_f32`, `_binary`)
  - `unsafe fn distance_cosine_i8_unchecked(a, b, len) -> i32`
  - `unsafe fn distance_dot_f32_unchecked(a, b, len) -> f32` (FMA-based)
  - `unsafe fn distance_hamming_binary_unchecked(a, b, len_bits) -> u32` (PSHUFB-based popcount per Mula's algorithm)
  - `unsafe fn distance_jaccard_binary_unchecked(...)`, `distance_tanimoto_binary_unchecked(...)`
  - Each `_unchecked` paired with `#[cfg(feature = "userspace")] unsafe fn distance_<metric>_<scalar>(...)` asserting variant
  - All under `#[target_feature(enable = "avx2")]`

- `crates/tokenfs-algos/src/similarity/hnsw/kernels/neon.rs`:
  - Same matrix as AVX2 but via NEON intrinsics (`std::arch::aarch64::*`)
  - Use VCNT for binary popcount; fallback table-lookup for the same
  - SDOT (FEAT_DotProd) for i8 dot if available — otherwise VMULL_S8 + VMLAL_S8
  - All under `#[target_feature(enable = "neon")]`

- `crates/tokenfs-algos/src/similarity/hnsw/kernels/sse41.rs`:
  - Integer paths only (i8 / u8 / binary via PSHUFB-popcount)
  - Fallback for x86 hosts without AVX2
  - All under `#[target_feature(enable = "sse4.1")]`

- `crates/tokenfs-algos/src/similarity/hnsw/kernels/ssse3.rs`:
  - Just binary metrics via PSHUFB-popcount
  - Fallback for very old x86 (pre-SSE4.1)

- `crates/tokenfs-algos/src/similarity/hnsw/kernels/mod.rs`:
  - Update `auto::distance(...)` runtime dispatcher to route through `is_x86_feature_detected!` / `std::arch::is_aarch64_feature_detected!`
  - Routes to `_unchecked` siblings after CPU detection (kernel-safe path)
  - Asserting variants reachable only with `userspace` feature

- `crates/tokenfs-algos/src/dispatch/planner/`:
  - Add `HnswSignals` struct: dimensions, scalar_kind, metric, expected node count
  - Rule: scalar / AVX2 / NEON / SSE4.1 / SSSE3 / scalar-fallback selection
  - Append HNSW rules to `RULES` in `dispatch/planner/rules.rs` per the `PLANNER_DESIGN.md` rules-as-data convention

### Tests

- `crates/tokenfs-algos/tests/parity.rs` extension:
  - For every (metric, scalar_kind, backend) cell, generate 1000 random pairs
  - Assert `kernels::<backend>::distance_<metric>_<scalar>` matches `kernels::scalar::distance_<metric>_<scalar>` exactly (for integer metrics) or within 1 ULP (for f32 metrics)
  - Per-backend test gated on the corresponding `is_*_feature_detected!`

- `crates/tokenfs-algos/tests/hnsw_walker_parity.rs`:
  - Re-run the Phase 1 parity test on each backend
  - Assert: switching the walker's distance kernel from scalar to AVX2 produces the same k-NN result list (modulo the f32 cross-arch caveat for f32 metrics, which is documented and gated to integer-only for parity tests)

- `crates/tokenfs-algos/tests/avx2_parity.rs` extension:
  - Add HNSW kernel parity rows
- `crates/tokenfs-algos/tests/neon_parity.rs` extension:
  - Add HNSW kernel parity rows

### Benchmarks

- `crates/tokenfs-algos/benches/iai_primitives.rs`:
  - `iai_hnsw_distance_l2_i8_avx2` (and matching NEON / scalar)
  - `iai_hnsw_distance_hamming_binary_avx2` (and matching NEON / scalar)
  - `iai_hnsw_distance_dot_f32_avx2` (and matching NEON / scalar)
  - `iai_hnsw_search_k16_n100k` — full search benchmark at fixed (k=16, N=10⁵) over the test fixture
  - Each bench has a `_scalar` companion so the iai regression gate compares apples-to-apples within a backend, not across

- `crates/tokenfs-algos/benches/hnsw_distance.rs` (new criterion bench):
  - Per-backend throughput at multiple vector dims (32, 128, 768)
  - Per-metric × per-scalar matrix
  - Used by `bench-regression.yml` and bench-history.yml

### CI

- `cargo xtask check` — passes (clippy must accept `unsafe fn ... _unchecked` per established pattern)
- `cargo xtask bench-iai` (locally with valgrind) — runs and produces baseline numbers
- iai-callgrind workflow already in CI from v0.5.0 — auto-gates the new bench rows at 1% IR regression
- Per-backend parity tests run on push (CI matrix already covers x86 + AArch64 via QEMU)

## Acceptance criteria

```bash
$ cargo test -p tokenfs-algos --features arch-pinned-kernels --test parity
running ~80 tests (8 metric/scalar combos × multiple backends)
...
test result: ok. 80 passed; 0 failed

$ cargo xtask bench-iai
... AVX2 i8 Hamming distance: ~10x faster than scalar
... AVX2 i8 L2² distance: ~6x faster than scalar
... NEON parity within 20% of AVX2

$ cargo xtask check
xtask: panic-surface-lint: pub fn surface within allowlist (0 entries snapshotted)
```

## Hardware backend matrix at end of Phase 2

| Metric | scalar | AVX2 | NEON | SSE4.1 | SSSE3 |
|---|---|---|---|---|---|
| L2² (f32) | ✅ | ✅ FMA | ✅ FMLA | — | — |
| L2² (i8 / u8) | ✅ | ✅ PMADDUBSW | ✅ SDOT/VMULL | ✅ | — |
| cosine (f32) | ✅ | ✅ | ✅ | — | — |
| cosine (i8 / u8) | ✅ | ✅ | ✅ | ✅ | — |
| dot (f32) | ✅ | ✅ FMA | ✅ FMLA | — | — |
| Hamming (binary) | ✅ | ✅ PSHUFB-popcount | ✅ VCNT | ✅ POPCNT-or-PSHUFB | ✅ PSHUFB |
| Jaccard (binary) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tanimoto (binary) | ✅ | ✅ | ✅ | ✅ | ✅ |

(AVX-512 row added in Phase 3.)

## Out of scope for Phase 2

- AVX-512 backend (Phase 3 — keeps nightly-feature complexity off this phase)
- Filter primitives (Phase 3)
- The Builder (Phase 4)
- Kernel-FPU bracketing for f32 in kernel mode (Phase 5)
- SVE2 (deferred entirely)

## Risk + mitigations

| Risk | Mitigation |
|---|---|
| SIMD kernel disagrees with scalar oracle | Per-backend `tests/parity.rs` rows on every push. Same shape as `bits/streamvbyte/kernels` parity. Block PR if any cell fails. |
| f32 cross-arch divergence (FMA fusion differences) breaks parity tests | Document in `research/DETERMINISM.md` and in the f32 kernel module; integer parity is exact, f32 parity is "within 1 ULP" or "match the scalar reference operating on the same FMA-fused path." |
| iai-callgrind regression gate false-positives on first run | Phase 1 already has the iai infra; we're just adding rows. The gate calibrates against the first main-branch baseline. |
| AVX2 popcount via Mula's algorithm slower than expected | Document in `SIMD_PRIOR_ART.md`. If <10× scalar, profile to see if the Mula constants are right. |
| NEON SDOT not available on QEMU CI runner | Use SDOT only if `is_aarch64_feature_detected!("dotprod")`; otherwise fall back to VMULL_S8 + VMLAL_S8. Both paths tested. |

## Cross-references

- [`PHASE_1.md`](PHASE_1.md) — must complete before this phase starts
- [`../components/DISTANCE_KERNELS.md`](../components/DISTANCE_KERNELS.md) — fills out by end of phase
- [`../research/SIMD_PRIOR_ART.md`](../research/SIMD_PRIOR_ART.md) — required input
- [`../research/PRIMITIVE_INVENTORY.md`](../research/PRIMITIVE_INVENTORY.md) — confirms which existing kernels we reuse via direct call
- [`../../PROCESSOR_AWARE_DISPATCH.md`](../../PROCESSOR_AWARE_DISPATCH.md) — per-backend kernel buffet pattern
- [`../../PLANNER_DESIGN.md`](../../PLANNER_DESIGN.md) — rules-as-data architecture for HNSW signals

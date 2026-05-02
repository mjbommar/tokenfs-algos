//! Vectorized gather-based hash for K-min `MinHash` signatures.
//!
//! Given a precomputed permutation table `T : [u8 -> [u64; K]]` (i.e. one
//! row per possible byte value, each row is `K` independent random
//! permutations of that byte), the per-byte K-min update degenerates to
//!
//! ```text
//! for each input byte b:
//!     for k in 0..K:
//!         sig[k] = min(sig[k], T[b][k])
//! ```
//!
//! The expensive piece is the per-byte K independent hash lookups. With
//! the table laid out so that all K hash values for a single byte are
//! contiguous (`[u64; K]`), AVX2's `_mm256_i32gather_epi64` evaluates 4
//! lookups per gather and AVX-512's `_mm512_i32gather_epi64` evaluates 8.
//!
//! ## State footprint
//!
//! - `K = 8`:  256 * 8 * 8  = **16 KiB** (fits L1d on every modern x86)
//! - `K = 16`: 256 * 16 * 8 = **32 KiB** (exactly L1d on Skylake-class)
//! - `K = 32`: 256 * 32 * 8 = **64 KiB** (exceeds L1d, hits L2)
//! - `K = 64`: 256 * 64 * 8 = **128 KiB**
//! - `K = 128`: 256 * 128 * 8 = **256 KiB** (exceeds L2 on most desktops)
//!
//! Past K=16 the working set leaves L1, so the gather path's effective
//! speedup over scalar drops sharply; benchmarks should explicitly cover
//! K=8 / K=16 / K=32 to surface this.
//!
//! ## Microarchitecture caveat (honest signal)
//!
//! `vpgatherqq` (`_mm256_i32gather_epi64` / `_mm512_i32gather_epi64`) is
//! historically slow on Zen 3 / Zen 4 — micro-op throughput is far below
//! the equivalent scalar loads on those parts. The benchmark in
//! `examples/bench_compare.rs` reports both scalar and gather paths so
//! the negative finding is visible per host. On Intel Skylake-X / Ice
//! Lake / Sapphire Rapids the AVX-512 gather variant is the one most
//! likely to beat scalar for a wide K.

#![allow(clippy::cast_possible_truncation)]

/// Total number of byte values addressable by the gather table.
pub const TABLE_ROWS: usize = 256;

/// Width of the gather lane (number of independent hash slots updated
/// in a single 256-bit AVX2 gather).
pub const AVX2_LANE_WIDTH: usize = 4;

/// Width of the gather lane for AVX-512 (8 × u64 in a single `__m512i`).
pub const AVX512_LANE_WIDTH: usize = 8;

/// Largest `K` for which [`build_table_from_seeds`] is stack-safe in a
/// kernel-adjacent context.
///
/// The table footprint is `K * 256 * 8` bytes. At `K = 8` that is 16 KiB
/// — at the upper end of the typical kernel stack budget (8-16 KiB, see
/// `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`). For larger `K`
/// (`K ∈ {16, 32, 64, 128, 256}`) callers in kernel-adjacent paths must
/// use [`build_table_from_seeds_into`] with caller-provided heap or
/// thread-local scratch.
pub const MINHASH_TABLE_BY_VALUE_SAFE_K_MAX: usize = 8;

/// Builds a permutation table compatible with the gather kernels.
///
/// `seeds[k]` parameterizes the hash for slot `k`; the result satisfies
/// `table[b][k] == crate::hash::mix_word(b as u64 ^ seeds[k])`. The table
/// occupies `K * 256 * 8` bytes — see the module-level state-footprint
/// table.
///
/// **WARNING (kernel stack)**: returns `[[u64; K]; 256]` *by value*. At
/// `K = 64` that is 128 KiB; at `K = 256` (the new MinHash-bench width)
/// it is 512 KiB. Neither is safe to allocate on a kernel stack
/// (typical 8-16 KiB budget, see
/// `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`). Kernel-adjacent
/// callers and any path that crosses an FFI/cgo boundary with a
/// constrained stack should call [`build_table_from_seeds_into`]
/// instead, which takes caller-provided storage.
///
/// For `K <= MINHASH_TABLE_BY_VALUE_SAFE_K_MAX` (256 * K * 8 bytes
/// stays at or under the 16 KiB upper bound) the by-value form is
/// stack-safe. Beyond that, prefer the `_into` variant.
#[must_use]
pub fn build_table_from_seeds<const K: usize>(seeds: &[u64; K]) -> [[u64; K]; TABLE_ROWS] {
    let mut table = [[0_u64; K]; TABLE_ROWS];
    build_table_from_seeds_into::<K>(seeds, &mut table);
    table
}

/// Kernel-safe variant of [`build_table_from_seeds`]: writes the
/// `K`-row × 256-byte minhash table into caller-provided scratch
/// instead of returning the (potentially large) array by value.
///
/// Same semantics as [`build_table_from_seeds`] — the resulting table
/// satisfies `scratch[b][k] == crate::hash::mix_word(b as u64 ^ seeds[k])`
/// — but no large stack allocation happens at the call site. The
/// caller decides where the table lives (heap `Box`, mmap, thread-local
/// pool, postgres memory context, kernel `kmalloc`'d slab, etc.) so the
/// kernel-stack hazard documented on [`build_table_from_seeds`] does
/// not apply.
///
/// `scratch` is fully overwritten — its prior contents are discarded —
/// so the caller may reuse a single buffer across many calls without
/// pre-clearing it.
///
/// For the safe-by-value `K` threshold, see
/// [`MINHASH_TABLE_BY_VALUE_SAFE_K_MAX`].
pub fn build_table_from_seeds_into<const K: usize>(
    seeds: &[u64; K],
    scratch: &mut [[u64; K]; TABLE_ROWS],
) {
    let mut byte = 0_usize;
    while byte < TABLE_ROWS {
        let mut k = 0;
        while k < K {
            scratch[byte][k] = crate::hash::mix_word((byte as u64) ^ seeds[k]);
            k += 1;
        }
        byte += 1;
    }
}

/// Reference scalar K-min update over a byte slice using a precomputed
/// gather table. Bit-exact with both the AVX2 and AVX-512 gather paths.
pub fn update_minhash_scalar<const K: usize>(
    bytes: &[u8],
    table: &[[u64; K]; TABLE_ROWS],
    sig: &mut [u64; K],
) {
    for &b in bytes {
        let row = &table[b as usize];
        for k in 0..K {
            if row[k] < sig[k] {
                sig[k] = row[k];
            }
        }
    }
}

/// AVX2 gather-based K-min `MinHash` update for `x86` / `x86_64`.
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2 {
    use super::TABLE_ROWS;

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m128i, __m256i, _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpgt_epi64,
        _mm256_i32gather_epi64, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_setr_epi32,
        _mm256_storeu_si256, _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_blendv_epi8, _mm256_castsi256_si128, _mm256_cmpgt_epi64,
        _mm256_i32gather_epi64, _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_setr_epi32,
        _mm256_storeu_si256, _mm256_xor_si256,
    };

    /// Returns true when AVX2 is available. The gather + unsigned-min
    /// synthesis below stays inside the AVX2 ISA — no AVX-512VL needed.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    /// Returns true when AVX2 is available.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Lane-wise unsigned 64-bit minimum on AVX2 (no `_mm256_min_epu64`).
    ///
    /// AVX2 only ships `_mm256_cmpgt_epi64` (signed). We flip the sign
    /// bit on both operands before the compare so the unsigned
    /// ordering is preserved, then blend.
    ///
    /// # Safety
    ///
    /// Caller must enter via a function with `target_feature = "avx2"`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn min_epu64_avx2(a: __m256i, b: __m256i) -> __m256i {
        // SAFETY: AVX2 enabled.
        let sign = _mm256_set1_epi64x(i64::MIN);
        let aa = _mm256_xor_si256(a, sign);
        let bb = _mm256_xor_si256(b, sign);
        // mask = 0xFFFF... where aa > bb (i.e. a > b unsigned).
        let mask = _mm256_cmpgt_epi64(aa, bb);
        // Where mask is set, take b; else take a.
        _mm256_blendv_epi8(a, b, mask)
    }

    /// Vectorized K-min `MinHash` update for `K = 8` using AVX2 gather.
    ///
    /// Performs two `vpgatherqq` lookups per input byte: one over the
    /// low-4 hash slots, one over the high-4. Each gather loads four
    /// 64-bit hashes from `table[b]` and the running signature halves
    /// are reduced via a synthesized unsigned-64 minimum (AVX2 has no
    /// native `_mm256_min_epu64`; we fall back to sign-flipped
    /// `cmpgt + blendv`).
    ///
    /// # Safety
    ///
    /// The caller must ensure all of:
    ///  - AVX2 is available at runtime;
    ///  - `table` is at least 8-byte aligned (`[[u64; 8]; 256]` is
    ///    naturally so when allocated by the compiler);
    ///  - `sig` and `table` outlive the call.
    #[target_feature(enable = "avx2")]
    pub unsafe fn update_minhash_8way(
        bytes: &[u8],
        table: &[[u64; 8]; TABLE_ROWS],
        sig: &mut [u64; 8],
    ) {
        // Two halves of the running signature, each holding 4×u64.
        // SAFETY: `sig` is &mut [u64; 8] (64 bytes), AVX2/VL enabled.
        let mut sig_lo = unsafe { _mm256_loadu_si256(sig.as_ptr().cast::<__m256i>()) };
        let mut sig_hi = unsafe { _mm256_loadu_si256(sig.as_ptr().add(4).cast::<__m256i>()) };

        // Index lanes for the two gather instructions.
        // `_mm256_i32gather_epi64` reads 4 indices from a 128-bit lane
        // (Vindex), each scaled by `scale` bytes. With scale = 8, the
        // indices map directly to u64 slots inside the table row.
        // `_mm256_setr_epi32` and `_mm256_castsi256_si128` are safe in
        // both `core::arch::x86` and `core::arch::x86_64` — they don't
        // execute SIMD instructions, just construct constants.
        let idx_lo: __m128i = {
            // We only need the low 4 i32 lanes of the __m256i for
            // `_mm256_i32gather_epi64`; encode (0,1,2,3).
            let v: __m256i = _mm256_setr_epi32(0, 1, 2, 3, 0, 0, 0, 0);
            _mm256_castsi256_si128(v)
        };
        let idx_hi: __m128i = {
            let v: __m256i = _mm256_setr_epi32(4, 5, 6, 7, 0, 0, 0, 0);
            _mm256_castsi256_si128(v)
        };

        let row_stride_bytes = core::mem::size_of::<[u64; 8]>();
        debug_assert_eq!(row_stride_bytes, 64);

        let table_base = table.as_ptr().cast::<i64>();

        for &b in bytes {
            // SAFETY: 0 <= b < 256, so the base + b * 64 byte offset is
            // inside the table allocation. Gather scale = 8 then walks
            // the per-row u64 array.
            let row_ptr = unsafe {
                table_base.add((b as usize) * 8 /* u64s per row */)
            };

            // Two 4-wide gathers cover slots [0..4) and [4..8).
            // SAFETY: AVX2 enabled; row_ptr + scale*indices stays
            // within `table[b]` (indices are 0..8, scale = 8 bytes).
            let lo = unsafe { _mm256_i32gather_epi64::<8>(row_ptr, idx_lo) };
            let hi = unsafe { _mm256_i32gather_epi64::<8>(row_ptr, idx_hi) };

            // Unsigned 64-bit lane-wise minimum, synthesized for AVX2.
            // SAFETY: AVX2 enabled by target_feature.
            sig_lo = unsafe { min_epu64_avx2(sig_lo, lo) };
            sig_hi = unsafe { min_epu64_avx2(sig_hi, hi) };
        }

        // SAFETY: `sig` is &mut [u64; 8] (64 bytes writable).
        unsafe { _mm256_storeu_si256(sig.as_mut_ptr().cast::<__m256i>(), sig_lo) };
        unsafe { _mm256_storeu_si256(sig.as_mut_ptr().add(4).cast::<__m256i>(), sig_hi) };
    }

    /// Vectorized K-min `MinHash` update for general `K` on AVX2.
    ///
    /// Iterates the `K` slots in groups of 4 — each group is one
    /// 32-byte `_mm256_loadu_si256` (loading four contiguous u64
    /// hashes from `table[b][group_start..group_start+4]`) followed by
    /// an unsigned-64 lane-wise min reduction into the running
    /// signature vector for that group. Tail slots (when `K % 4 != 0`)
    /// fall back to scalar updates per byte.
    ///
    /// **Why direct loads instead of `vpgatherqq`:** the per-byte K
    /// slot indices are always sequential (0, 1, 2, ..., K-1), so a
    /// scalar gather degenerates to a contiguous load. `vpgatherqq` is
    /// micro-op-bound on Alder Lake, Zen 3, and Zen 4 — direct loads
    /// retire ~2-4× faster on those parts. The 8-way kernel
    /// [`update_minhash_8way`] is kept gather-based for ABI continuity
    /// (it's been shipped) and for the parity test it provides on
    /// CPUs where gather is competitive.
    ///
    /// `K` is a const generic so the compiler unrolls the per-group
    /// loop and propagates the per-row stride constant.
    ///
    /// # Safety
    ///
    /// The caller must ensure AVX2 is available at runtime.
    #[target_feature(enable = "avx2")]
    pub unsafe fn update_minhash_kway<const K: usize>(
        bytes: &[u8],
        table: &[[u64; K]; TABLE_ROWS],
        sig: &mut [u64; K],
    ) {
        // Process slots 4 at a time.
        let groups = K / 4;
        let tail = K - groups * 4;

        for &b in bytes {
            // SAFETY: 0 <= b < 256; row stays inside the table allocation.
            let row_ptr = unsafe { table.as_ptr().add(b as usize).cast::<u64>() };

            // Vectorised groups of 4 — direct loads.
            for g in 0..groups {
                let base = g * 4;
                // SAFETY: row is K × u64 = 8K bytes; load 32 bytes at
                // `base * 8` byte offset, which is within the row.
                let v = unsafe { _mm256_loadu_si256(row_ptr.add(base).cast::<__m256i>()) };
                // SAFETY: 4 u64 = 32 writable bytes at sig[base..base+4].
                let cur = unsafe { _mm256_loadu_si256(sig.as_ptr().add(base).cast::<__m256i>()) };
                // SAFETY: AVX2 enabled.
                let merged = unsafe { min_epu64_avx2(cur, v) };
                // SAFETY: as above.
                unsafe {
                    _mm256_storeu_si256(sig.as_mut_ptr().add(base).cast::<__m256i>(), merged);
                };
            }

            // Scalar tail for the up-to-3 leftover slots when K is not
            // a multiple of 4.
            if tail > 0 {
                let row = unsafe { &*table.as_ptr().add(b as usize) };
                let base = groups * 4;
                for k in base..K {
                    let h = row[k];
                    if h < sig[k] {
                        sig[k] = h;
                    }
                }
            }
        }
    }
}

/// AVX-512 gather-based K-min `MinHash` update.
#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx512 {
    use super::TABLE_ROWS;

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256i, __m512i, _mm256_setr_epi32, _mm512_i32gather_epi64, _mm512_loadu_si512,
        _mm512_min_epu64, _mm512_storeu_si512,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256i, __m512i, _mm256_setr_epi32, _mm512_i32gather_epi64, _mm512_loadu_si512,
        _mm512_min_epu64, _mm512_storeu_si512,
    };

    /// Returns true when AVX-512F is available (gather + min are part of
    /// the AVX-512F base ISA).
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx512f")
    }

    /// Returns true when AVX-512F is available.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Vectorized K-min `MinHash` update for `K = 8` using AVX-512 gather.
    ///
    /// One `vpgatherqq` (`_mm512_i32gather_epi64`) reads all 8 hash
    /// values for a single input byte. The running signature is
    /// reduced via `_mm512_min_epu64`. ~3 instructions per byte (load
    /// indices once, gather, min) instead of 8 separate scalar hashes.
    ///
    /// # Safety
    ///
    /// The caller must ensure AVX-512F is available at runtime.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn update_minhash_8way(
        bytes: &[u8],
        table: &[[u64; 8]; TABLE_ROWS],
        sig: &mut [u64; 8],
    ) {
        // SAFETY: AVX-512F enabled; `sig` is 64 bytes (8 × u64).
        let mut acc = unsafe { _mm512_loadu_si512(sig.as_ptr().cast::<__m512i>()) };

        // 8-lane index vector encoding [0, 1, 2, 3, 4, 5, 6, 7].
        // `_mm256_setr_epi32` is safe — it's just a constant constructor.
        let idx: __m256i = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

        let table_base = table.as_ptr().cast::<i64>();
        let _ = TABLE_ROWS; // silence unused-const warning when feature gates differ

        for &b in bytes {
            // SAFETY: b ∈ 0..256; row_ptr stays inside the table allocation
            // (TABLE_ROWS rows of 8 u64). `_mm512_i32gather_epi64` reads
            // 8 × u64 starting at row_ptr with `scale = 8` — all 8 indices
            // (0..8) are inside the row.
            let row_ptr = unsafe { table_base.add((b as usize) * 8) };
            let v = unsafe { _mm512_i32gather_epi64::<8>(idx, row_ptr) };
            // `_mm512_min_epu64` is safe in `core::arch` once
            // `target_feature = "avx512f"` is enabled.
            acc = _mm512_min_epu64(acc, v);
        }

        // SAFETY: AVX-512F enabled; `sig` is 64 writable bytes.
        unsafe { _mm512_storeu_si512(sig.as_mut_ptr().cast::<__m512i>(), acc) };
    }

    /// Vectorized K-min `MinHash` update for general `K` on AVX-512.
    ///
    /// Iterates the `K` slots in groups of 8 — each group is one
    /// 64-byte `_mm512_loadu_si512` (loading eight contiguous u64
    /// hashes from `table[b][group_start..group_start+8]`) followed by
    /// `_mm512_min_epu64`. Tail slots (when `K % 8 != 0`) fall back to
    /// scalar updates.
    ///
    /// **Why direct loads instead of `vpgatherqq`:** the per-byte K
    /// slot indices are always sequential (0, 1, 2, ..., K-1), so a
    /// scalar gather degenerates to a contiguous 64-byte load.
    /// `vpgatherqq` micro-op throughput is far below a contiguous
    /// `vmovdqu64` on every shipping AVX-512 micro-arch (Ice Lake,
    /// Sapphire Rapids, Tiger Lake, Zen 4) — the direct load is
    /// strictly faster.
    ///
    /// `K` is a const generic so the compiler unrolls the per-group
    /// loop and propagates the per-row stride constant. For
    /// `K ∈ {16, 32, 64, 128, 256}` the unrolled inner loop is two,
    /// four, eight, sixteen, or thirty-two load + min pairs per byte.
    ///
    /// # Safety
    ///
    /// The caller must ensure AVX-512F is available at runtime.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn update_minhash_kway<const K: usize>(
        bytes: &[u8],
        table: &[[u64; K]; TABLE_ROWS],
        sig: &mut [u64; K],
    ) {
        let groups = K / 8;
        let tail = K - groups * 8;

        for &b in bytes {
            // SAFETY: 0 <= b < 256; row stays inside the table allocation.
            let row_ptr = unsafe { table.as_ptr().add(b as usize).cast::<u64>() };

            for g in 0..groups {
                let base = g * 8;
                // SAFETY: row is K × u64 (8K bytes); load 64 bytes at
                // `base * 8` byte offset, which is within the row.
                let v = unsafe { _mm512_loadu_si512(row_ptr.add(base).cast::<__m512i>()) };
                // SAFETY: 8 u64 = 64 writable bytes at sig[base..base+8].
                let cur = unsafe { _mm512_loadu_si512(sig.as_ptr().add(base).cast::<__m512i>()) };
                let merged = _mm512_min_epu64(cur, v);
                // SAFETY: as above.
                unsafe {
                    _mm512_storeu_si512(sig.as_mut_ptr().add(base).cast::<__m512i>(), merged);
                };
            }

            // Scalar tail for the up-to-7 leftover slots.
            if tail > 0 {
                let row = unsafe { &*table.as_ptr().add(b as usize) };
                let base = groups * 8;
                for k in base..K {
                    let h = row[k];
                    if h < sig[k] {
                        sig[k] = h;
                    }
                }
            }
        }
    }
}

/// AArch64 NEON gather-based K-min `MinHash` update.
///
/// NEON has no scatter-gather instruction; instead we lean on the row
/// being a contiguous `[u64; K]` and load it in 128-bit chunks
/// (`vld1q_u64` — 2 × u64 per load) followed by a synthesized
/// lane-wise unsigned-64 minimum. The result is bit-identical with
/// the AVX2 / AVX-512 paths.
///
/// AArch64 has no `vminq_u64` intrinsic in stable Rust; we synthesize
/// it from `vcgtq_u64` (lane-wise unsigned greater-than → mask) and
/// `vbslq_u64` (bitwise select) — two cycles per pair of u64 lanes,
/// the same shape as the AVX2 fallback that emulates `_mm256_min_epu64`.
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub mod neon {
    use super::TABLE_ROWS;

    use core::arch::aarch64::{uint64x2_t, vbslq_u64, vcgtq_u64, vld1q_u64, vst1q_u64};

    /// Returns true when NEON is available at runtime.
    ///
    /// NEON is mandatory on AArch64; this exists for API symmetry with
    /// the x86 `is_available` helpers.
    #[must_use]
    pub const fn is_available() -> bool {
        true
    }

    /// Synthesized lane-wise unsigned 64-bit minimum on AArch64.
    ///
    /// AArch64 stable Rust exposes `vminq_u32` but not `vminq_u64`.
    /// We use `vcgtq_u64(a, b)` which produces 0xFFFF... per lane where
    /// `a > b` (unsigned), then `vbslq_u64(mask, b, a)` selects `b`
    /// where the mask is set (the smaller of the pair).
    ///
    /// # Safety
    ///
    /// NEON is mandatory on AArch64; the helper is `#[inline]` so the
    /// caller's `target_feature` propagates.
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn min_u64x2(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
        // mask lanes where a > b unsigned.
        let mask = vcgtq_u64(a, b);
        // bsl: if mask bit set, take b; else take a.
        vbslq_u64(mask, b, a)
    }

    /// Vectorized K-min `MinHash` update for `K = 8` using NEON.
    ///
    /// Each input byte triggers four 128-bit `vld1q_u64` loads from
    /// `table[b]` (covering all 8 u64 slots) and four lane-wise
    /// minimum reductions into the running signature.
    ///
    /// # Safety
    ///
    /// The caller must ensure NEON is available (always true on
    /// AArch64) and that `sig` and `table` outlive the call.
    #[target_feature(enable = "neon")]
    pub unsafe fn update_minhash_8way(
        bytes: &[u8],
        table: &[[u64; 8]; TABLE_ROWS],
        sig: &mut [u64; 8],
    ) {
        // Four 128-bit accumulators (2 × u64 each = 8 lanes total).
        // SAFETY: `sig` is &mut [u64; 8] = 64 bytes; loads are 16-byte
        // each from contiguous offsets 0/2/4/6.
        let mut a0: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr()) };
        let mut a1: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr().add(2)) };
        let mut a2: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr().add(4)) };
        let mut a3: uint64x2_t = unsafe { vld1q_u64(sig.as_ptr().add(6)) };

        for &b in bytes {
            let row_ptr = unsafe { table.as_ptr().add(b as usize).cast::<u64>() };
            // SAFETY: each row is 8 × u64 (64 bytes); loads cover the
            // entire row in 4 × 16-byte halves.
            let r0 = unsafe { vld1q_u64(row_ptr) };
            let r1 = unsafe { vld1q_u64(row_ptr.add(2)) };
            let r2 = unsafe { vld1q_u64(row_ptr.add(4)) };
            let r3 = unsafe { vld1q_u64(row_ptr.add(6)) };
            // SAFETY: NEON enabled by enclosing target_feature.
            a0 = unsafe { min_u64x2(a0, r0) };
            a1 = unsafe { min_u64x2(a1, r1) };
            a2 = unsafe { min_u64x2(a2, r2) };
            a3 = unsafe { min_u64x2(a3, r3) };
        }

        // SAFETY: 64 writable bytes at `sig`.
        unsafe { vst1q_u64(sig.as_mut_ptr(), a0) };
        unsafe { vst1q_u64(sig.as_mut_ptr().add(2), a1) };
        unsafe { vst1q_u64(sig.as_mut_ptr().add(4), a2) };
        unsafe { vst1q_u64(sig.as_mut_ptr().add(6), a3) };
    }

    /// Vectorized K-min `MinHash` update for general `K` using NEON.
    ///
    /// Iterates the `K` slots in groups of 2 — each group is one
    /// `vld1q_u64` (loading two contiguous u64 hashes from
    /// `table[b][group_start..group_start+2]`) followed by a
    /// synthesised lane-wise minimum reduction. Tail slots (when `K`
    /// is odd) fall back to scalar updates.
    ///
    /// `K` is a const generic so the compiler unrolls the per-group
    /// loop and propagates the per-row stride constant.
    ///
    /// # Safety
    ///
    /// The caller must ensure NEON is available (always true on
    /// AArch64).
    #[target_feature(enable = "neon")]
    pub unsafe fn update_minhash_kway<const K: usize>(
        bytes: &[u8],
        table: &[[u64; K]; TABLE_ROWS],
        sig: &mut [u64; K],
    ) {
        let groups = K / 2;
        let tail = K - groups * 2;

        for &b in bytes {
            let row_ptr = unsafe { table.as_ptr().add(b as usize).cast::<u64>() };
            for g in 0..groups {
                let base = g * 2;
                // SAFETY: row is K × u64; base + 2 <= K because g < K/2.
                let r = unsafe { vld1q_u64(row_ptr.add(base)) };
                // SAFETY: sig is K × u64 (8K bytes); load 16 bytes.
                let cur = unsafe { vld1q_u64(sig.as_ptr().add(base)) };
                // SAFETY: NEON enabled by enclosing target_feature.
                let merged = unsafe { min_u64x2(cur, r) };
                // SAFETY: 16 writable bytes at sig[base..base+2].
                unsafe { vst1q_u64(sig.as_mut_ptr().add(base), merged) };
            }
            if tail > 0 {
                let row = unsafe { &*table.as_ptr().add(b as usize) };
                let base = groups * 2;
                for k in base..K {
                    let h = row[k];
                    if h < sig[k] {
                        sig[k] = h;
                    }
                }
            }
        }
    }
}

/// SimHash 64-bit accumulator-update kernels.
///
/// Allocates a 16 KiB table on the heap, so this module is gated on
/// the `std` (or `alloc`) feature.
#[cfg(any(feature = "std", feature = "alloc"))]
///
/// SimHash builds a 64-lane signed accumulator over weighted features.
/// With a precomputed table `T : [u8 -> [i8; 64]]` where `T[b][i] = +1`
/// when bit `i` of `mix_word(b ^ seed_bit_i)` is set and `-1` otherwise,
/// the per-byte update is a fixed-shape i8 add into the accumulator.
///
/// Accumulator width is `i32` per lane. With ±1 contributions the
/// accumulator drifts by at most `bytes.len()` per lane, so i32 stays
/// safe for inputs up to 2 GiB — well past any practical sketch use.
///
/// ## State footprint
///
/// Table size: 256 * 64 = **16 KiB** (fits L1 on every modern x86).
/// Accumulator: 64 * 4 = 256 B (lives in registers).
pub mod simhash {
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;

    use super::TABLE_ROWS;

    /// Number of bits in the 64-bit SimHash signature.
    pub const BITS: usize = 64;

    /// Type of one SimHash contribution table.
    pub type Table = [[i8; BITS]; TABLE_ROWS];

    /// Builds a SimHash contribution table from one seed per bit.
    ///
    /// `seeds[i]` parameterizes the hash for bit position `i`. The
    /// returned table satisfies
    /// `T[b][i] = +1 if mix_word((b as u64) ^ seeds[i]) & 1 == 1 else -1`.
    /// Using only the low bit of each per-(byte, bit) hash keeps the
    /// contribution sign uncorrelated across bits.
    #[must_use]
    pub fn build_table_from_seeds(seeds: &[u64; BITS]) -> Box<Table> {
        let mut table: Box<Table> = Box::new([[0_i8; BITS]; TABLE_ROWS]);
        for (b, row) in table.iter_mut().enumerate() {
            for (i, slot) in row.iter_mut().enumerate() {
                let h = crate::hash::mix_word((b as u64) ^ seeds[i]);
                *slot = if h & 1 == 1 { 1 } else { -1 };
            }
        }
        table
    }

    /// Reference scalar update of the 64-lane i32 accumulator.
    ///
    /// `acc[i]` is increased by `T[b][i]` for every input byte `b`.
    /// Bit-identical to all SIMD paths below.
    pub fn update_accumulator_scalar(bytes: &[u8], table: &Table, acc: &mut [i32; BITS]) {
        for &b in bytes {
            let row = &table[b as usize];
            for i in 0..BITS {
                acc[i] += i32::from(row[i]);
            }
        }
    }

    /// Finalize an accumulator into a 64-bit SimHash signature.
    ///
    /// Each accumulator lane > 0 sets its corresponding bit.
    #[must_use]
    pub fn finalize(acc: &[i32; BITS]) -> u64 {
        let mut bits = 0_u64;
        for (i, &a) in acc.iter().enumerate() {
            if a > 0 {
                bits |= 1_u64 << i;
            }
        }
        bits
    }

    /// AVX2 gather-based SimHash accumulator update.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::{BITS, Table};

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepi8_epi32,
            _mm256_loadu_si256, _mm256_storeu_si256,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m128i, __m256i, _mm_loadl_epi64, _mm256_add_epi32, _mm256_cvtepi8_epi32,
            _mm256_loadu_si256, _mm256_storeu_si256,
        };

        /// Returns true when AVX2 is available at runtime.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("avx2")
        }

        /// Returns true when AVX2 is available at runtime.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// AVX2 implementation of the per-byte SimHash accumulator
        /// update.
        ///
        /// For each input byte we load the 64-entry contribution row,
        /// sign-extend each 8-byte slice from i8 to i32 (so 64 i8
        /// contributions become 8 × `__m256i` of 8 × i32 each), then
        /// add into 8 register-resident accumulators.
        ///
        /// # Safety
        ///
        /// Caller must ensure AVX2 is available at runtime.
        #[target_feature(enable = "avx2")]
        pub unsafe fn update_accumulator(bytes: &[u8], table: &Table, acc: &mut [i32; BITS]) {
            // Load all 8 accumulator lanes (8 × __m256i of 8 × i32 = 64 lanes).
            // SAFETY: `acc` is 256 bytes (64 × i32); AVX2 enabled.
            let mut a0 = unsafe { _mm256_loadu_si256(acc.as_ptr().cast::<__m256i>()) };
            let mut a1 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(8).cast::<__m256i>()) };
            let mut a2 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(16).cast::<__m256i>()) };
            let mut a3 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(24).cast::<__m256i>()) };
            let mut a4 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(32).cast::<__m256i>()) };
            let mut a5 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(40).cast::<__m256i>()) };
            let mut a6 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(48).cast::<__m256i>()) };
            let mut a7 = unsafe { _mm256_loadu_si256(acc.as_ptr().add(56).cast::<__m256i>()) };

            for &b in bytes {
                // SAFETY: row is 64 bytes inside `Table`; reads stay in-bounds.
                let row_base = unsafe { table.as_ptr().add(b as usize) }.cast::<u8>();

                // Sign-extend 8 i8 lanes per gather → __m256i of 8 × i32.
                // SAFETY: row_base..row_base+64 is inside the table.
                let s0 = unsafe { _mm256_cvtepi8_epi32(load64(row_base)) };
                let s1 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(8))) };
                let s2 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(16))) };
                let s3 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(24))) };
                let s4 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(32))) };
                let s5 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(40))) };
                let s6 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(48))) };
                let s7 = unsafe { _mm256_cvtepi8_epi32(load64(row_base.add(56))) };

                a0 = _mm256_add_epi32(a0, s0);
                a1 = _mm256_add_epi32(a1, s1);
                a2 = _mm256_add_epi32(a2, s2);
                a3 = _mm256_add_epi32(a3, s3);
                a4 = _mm256_add_epi32(a4, s4);
                a5 = _mm256_add_epi32(a5, s5);
                a6 = _mm256_add_epi32(a6, s6);
                a7 = _mm256_add_epi32(a7, s7);
            }

            // SAFETY: 256 writable bytes; AVX2 enabled.
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().cast::<__m256i>(), a0) };
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(8).cast::<__m256i>(), a1) };
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(16).cast::<__m256i>(), a2) };
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(24).cast::<__m256i>(), a3) };
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(32).cast::<__m256i>(), a4) };
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(40).cast::<__m256i>(), a5) };
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(48).cast::<__m256i>(), a6) };
            unsafe { _mm256_storeu_si256(acc.as_mut_ptr().add(56).cast::<__m256i>(), a7) };
        }

        /// Helper: 64-bit load into the low half of a `__m128i`, upper
        /// half zero-filled. `_mm256_cvtepi8_epi32` only sign-extends
        /// the low 8 lanes of its input — using a 64-bit load (instead
        /// of 128-bit) avoids reading past the end of the 64-byte
        /// table row at byte offset 56.
        ///
        /// # Safety
        ///
        /// `ptr..ptr+8` must be readable and inside an allocation.
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn load64(ptr: *const u8) -> __m128i {
            // SAFETY: caller guarantees 8 readable bytes.
            unsafe { _mm_loadl_epi64(ptr.cast::<__m128i>()) }
        }
    }

    /// Auto-dispatched SimHash accumulator update.
    pub fn update_accumulator_auto(bytes: &[u8], table: &Table, acc: &mut [i32; BITS]) {
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if avx2::is_available() {
                // SAFETY: availability checked above.
                unsafe { avx2::update_accumulator(bytes, table, acc) };
                return;
            }
        }
        update_accumulator_scalar(bytes, table, acc);
    }
}

/// Auto-dispatched 8-way K-min `MinHash` update.
///
/// Falls through to the scalar implementation when the requested SIMD
/// path is unavailable. This is the entry point intended for library
/// callers; the `avx2` / `avx512` / `neon` modules above remain pinned
/// for parity-testing and microbenchmarks.
pub fn update_minhash_8way_auto(bytes: &[u8], table: &[[u64; 8]; TABLE_ROWS], sig: &mut [u64; 8]) {
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if avx512::is_available() {
            // SAFETY: availability checked above.
            unsafe { avx512::update_minhash_8way(bytes, table, sig) };
            return;
        }
    }
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if avx2::is_available() {
            // SAFETY: availability checked above.
            unsafe { avx2::update_minhash_8way(bytes, table, sig) };
            return;
        }
    }
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        if neon::is_available() {
            // SAFETY: NEON is mandatory on AArch64.
            unsafe { neon::update_minhash_8way(bytes, table, sig) };
            return;
        }
    }
    update_minhash_scalar::<8>(bytes, table, sig);
}

/// Auto-dispatched K-way K-min `MinHash` update for general `K`.
///
/// Picks AVX-512 → AVX2 → NEON → scalar in priority order based on
/// runtime CPU detection, then dispatches the matching const-generic
/// kernel. Bit-exact with [`update_minhash_scalar`] for every `K` and
/// every input.
///
/// Use this entry point when `K` may vary at the call site; for
/// `K = 8`, prefer [`update_minhash_8way_auto`] which avoids the
/// const-generic indirection at the cost of a fixed signature width.
pub fn update_minhash_kway_auto<const K: usize>(
    bytes: &[u8],
    table: &[[u64; K]; TABLE_ROWS],
    sig: &mut [u64; K],
) {
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if avx512::is_available() {
            // SAFETY: availability checked above.
            unsafe { avx512::update_minhash_kway::<K>(bytes, table, sig) };
            return;
        }
    }
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if avx2::is_available() {
            // SAFETY: availability checked above.
            unsafe { avx2::update_minhash_kway::<K>(bytes, table, sig) };
            return;
        }
    }
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        if neon::is_available() {
            // SAFETY: NEON is mandatory on AArch64.
            unsafe { neon::update_minhash_kway::<K>(bytes, table, sig) };
            return;
        }
    }
    update_minhash_scalar::<K>(bytes, table, sig);
}

#[cfg(test)]
mod tests {
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;

    use super::*;

    #[test]
    fn scalar_update_matches_per_byte_reference_k8() {
        let seeds: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let table = build_table_from_seeds(&seeds);

        let bytes = b"the quick brown fox jumps over the lazy dog";
        let mut sig = [u64::MAX; 8];
        update_minhash_scalar::<8>(bytes, &table, &mut sig);

        // Hand-compute reference with the same per-byte hash family.
        let mut expected = [u64::MAX; 8];
        for &b in bytes {
            for k in 0..8 {
                let h = crate::hash::mix_word((b as u64) ^ seeds[k]);
                if h < expected[k] {
                    expected[k] = h;
                }
            }
        }
        assert_eq!(sig, expected);
    }

    #[test]
    fn build_table_from_seeds_is_deterministic_k16() {
        let seeds: [u64; 16] = core::array::from_fn(|i| (i as u64) * 0xDEAD_BEEF);
        let table_a = build_table_from_seeds(&seeds);
        let table_b = build_table_from_seeds(&seeds);
        assert_eq!(table_a, table_b);
        // First row spot-check.
        for k in 0..16 {
            assert_eq!(
                table_a[0][k],
                crate::hash::mix_word(seeds[k]),
                "row 0 column {k}"
            );
        }
    }

    /// Helper: heap-allocate a zeroed `Box<[[u64; K]; TABLE_ROWS]>`
    /// without ever materialising the array on the stack. We use
    /// `Box::<T>::new_uninit()` then `write_bytes(0)` over the heap
    /// memory before `assume_init`. Callers that just want the table
    /// filled by `build_table_from_seeds_into` could skip the zero
    /// step, but zeroing first lets the test prove the `_into`
    /// function actually overwrites the scratch.
    #[cfg(any(feature = "std", feature = "alloc"))]
    fn alloc_zeroed_table_box<const K: usize>() -> Box<[[u64; K]; TABLE_ROWS]> {
        use core::mem::MaybeUninit;
        let mut uninit: Box<MaybeUninit<[[u64; K]; TABLE_ROWS]>> = Box::new_uninit();
        // SAFETY: `uninit.as_mut_ptr()` is a valid, properly aligned
        // pointer to writable heap storage of `sizeof::<[[u64; K]; 256]>()`
        // bytes. After `write_bytes(0)` every u64 is the bit-pattern 0
        // (a valid u64), so `assume_init` is sound.
        unsafe {
            core::ptr::write_bytes(uninit.as_mut_ptr().cast::<u64>(), 0, K * TABLE_ROWS);
            uninit.assume_init()
        }
    }

    /// `build_table_from_seeds_into<K>` produces a bit-exact match with
    /// the by-value [`build_table_from_seeds`] across every documented
    /// `K` width (`8, 16, 32, 64, 128, 256`). This is the primary
    /// parity guarantee for audit-R5 finding #156: the kernel-safe
    /// variant must not drift in semantics from the legacy by-value form.
    ///
    /// At `K >= 16` the test only calls the `_into` form on two
    /// independently heap-allocated buffers built with the same seeds.
    /// Calling the by-value [`build_table_from_seeds`] directly here
    /// would put the entire `[[u64; K]; 256]` array on the test stack
    /// (32 KiB at K=16, up to 512 KiB at K=256) — exactly the hazard
    /// audit-R5 #156 flags. Two heap copies built from the same seeds
    /// must agree because the kernel is deterministic, and we
    /// cross-check the first row of each against the per-(byte, k)
    /// reference family directly.
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn build_table_from_seeds_into_matches_by_value() {
        // Stack-safe width (table footprint <= 16 KiB): compare the
        // by-value path directly against the `_into` path.
        {
            const K: usize = 8;
            let seeds: [u64; K] = core::array::from_fn(|i| 0x9E37_79B9_u64.wrapping_add(i as u64));
            let by_value = build_table_from_seeds::<K>(&seeds);
            let mut into = [[0_u64; K]; TABLE_ROWS];
            build_table_from_seeds_into::<K>(&seeds, &mut into);
            assert_eq!(by_value, into, "K={K}");
        }

        // K >= 16: use only heap-allocated buffers so the test does
        // not reproduce the audit hazard. Two `_into` calls with the
        // same seeds must produce bit-exact results because the
        // kernel is deterministic — that determinism is what
        // justifies the boxed-wrapper optimisation in the public API.
        macro_rules! check_k_heap {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0x9E37_79B9_u64.wrapping_add(i as u64));
                let mut a = alloc_zeroed_table_box::<$k>();
                let mut b = alloc_zeroed_table_box::<$k>();
                build_table_from_seeds_into::<$k>(&seeds, &mut a);
                build_table_from_seeds_into::<$k>(&seeds, &mut b);
                assert_eq!(*a, *b, "K={}", $k);
                // Spot-check the first row matches the per-(byte, k)
                // hash family directly — this proves both `_into`
                // calls agreed on the *correct* table, not just on
                // each other.
                for k in 0..$k {
                    assert_eq!(
                        a[0][k],
                        crate::hash::mix_word(seeds[k]),
                        "K={} row 0 column {k}",
                        $k
                    );
                }
            }};
        }
        check_k_heap!(16);
        check_k_heap!(32);
        check_k_heap!(64);
        check_k_heap!(128);
        check_k_heap!(256);
    }

    /// `build_table_from_seeds_into<K>` overwrites prior scratch
    /// contents — it does not OR/blend into existing data. Reusing a
    /// scratch buffer across calls is therefore safe without
    /// pre-clearing.
    #[cfg(any(feature = "std", feature = "alloc"))]
    #[test]
    fn build_table_from_seeds_into_overwrites_prior_scratch() {
        const K: usize = 16;
        let seeds_a: [u64; K] = core::array::from_fn(|i| 0xAAAA_AAAA_u64.wrapping_add(i as u64));
        let seeds_b: [u64; K] = core::array::from_fn(|i| 0xBBBB_BBBB_u64.wrapping_add(i as u64));
        let mut scratch = alloc_zeroed_table_box::<K>();

        // First fill with seeds_a, then overwrite with seeds_b.
        build_table_from_seeds_into::<K>(&seeds_a, &mut scratch);
        build_table_from_seeds_into::<K>(&seeds_b, &mut scratch);

        // K=16 table is 32 KiB — uncomfortable on the test stack;
        // build the expected reference into a heap buffer too rather
        // than calling the by-value form.
        let mut expected = alloc_zeroed_table_box::<K>();
        build_table_from_seeds_into::<K>(&seeds_b, &mut expected);
        assert_eq!(*scratch, *expected);
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_8way_matches_scalar() {
        if !avx2::is_available() {
            eprintln!("AVX2 + AVX-512VL not available; skipping avx2 gather parity test");
            return;
        }
        let seeds: [u64; 8] = core::array::from_fn(|i| 0xCAFE_BABE_u64 ^ (i as u64));
        let table = build_table_from_seeds(&seeds);

        for len in [0_usize, 1, 7, 16, 64, 1024, 4096, 65_535] {
            let bytes: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(31) ^ 0x5A) as u8)
                .collect();
            let mut s_scalar = [u64::MAX; 8];
            update_minhash_scalar::<8>(&bytes, &table, &mut s_scalar);
            let mut s_avx2 = [u64::MAX; 8];
            // SAFETY: availability checked above.
            unsafe { avx2::update_minhash_8way(&bytes, &table, &mut s_avx2) };
            assert_eq!(s_scalar, s_avx2, "AVX2 gather diverged at len={len}");
        }
    }

    #[cfg(all(
        any(feature = "std", feature = "alloc"),
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn simhash_avx2_matches_scalar_random() {
        if !simhash::avx2::is_available() {
            eprintln!("AVX2 unavailable; skipping simhash gather parity test");
            return;
        }
        let seeds: [u64; simhash::BITS] =
            core::array::from_fn(|i| 0xFEED_FACE_u64.wrapping_mul((i as u64) + 1));
        let table = simhash::build_table_from_seeds(&seeds);

        for len in [0_usize, 1, 7, 16, 64, 1024, 4096, 65_535] {
            let bytes: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(13) ^ 0xC3) as u8)
                .collect();
            let mut a_scalar = [0_i32; simhash::BITS];
            simhash::update_accumulator_scalar(&bytes, &table, &mut a_scalar);
            let mut a_avx2 = [0_i32; simhash::BITS];
            // SAFETY: availability checked above.
            unsafe { simhash::avx2::update_accumulator(&bytes, &table, &mut a_avx2) };
            assert_eq!(
                a_scalar, a_avx2,
                "SimHash AVX2 gather diverged at len={len}"
            );
        }
    }

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx512_8way_matches_scalar() {
        if !avx512::is_available() {
            eprintln!("AVX-512F not available; skipping avx512 gather parity test");
            return;
        }
        let seeds: [u64; 8] = core::array::from_fn(|i| 0x9E37_79B9_7F4A_7C15_u64 ^ (i as u64));
        let table = build_table_from_seeds(&seeds);

        for len in [0_usize, 1, 7, 16, 64, 1024, 4096, 65_535] {
            let bytes: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(11) ^ 0xA5) as u8)
                .collect();
            let mut s_scalar = [u64::MAX; 8];
            update_minhash_scalar::<8>(&bytes, &table, &mut s_scalar);
            let mut s_avx512 = [u64::MAX; 8];
            // SAFETY: availability checked above.
            unsafe { avx512::update_minhash_8way(&bytes, &table, &mut s_avx512) };
            assert_eq!(s_scalar, s_avx512, "AVX-512 gather diverged at len={len}");
        }
    }

    /// Generic K-way scalar reference table-driven path matches a
    /// per-byte hand-rolled K-min update across every documented `K`.
    #[test]
    fn scalar_kway_matches_per_byte_reference_k16_k32_k64() {
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0xFACE_FEED_u64.wrapping_add(i as u64));
                let table = build_table_from_seeds::<$k>(&seeds);
                let bytes: Vec<u8> = (0..1024_usize)
                    .map(|i| (i.wrapping_mul(7) ^ 0x42) as u8)
                    .collect();
                let mut s_scalar = [u64::MAX; $k];
                update_minhash_scalar::<$k>(&bytes, &table, &mut s_scalar);

                let mut expected = [u64::MAX; $k];
                for &b in &bytes {
                    for k in 0..$k {
                        let h = crate::hash::mix_word((b as u64) ^ seeds[k]);
                        if h < expected[k] {
                            expected[k] = h;
                        }
                    }
                }
                assert_eq!(s_scalar, expected, "K={}", $k);
            }};
        }
        check_k!(16);
        check_k!(32);
        check_k!(64);
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_kway_matches_scalar_for_k16_k32_k64() {
        if !avx2::is_available() {
            eprintln!("AVX2 unavailable; skipping avx2 K-way gather parity test");
            return;
        }
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0xCAFE_BABE_u64.wrapping_add(i as u64));
                let table = build_table_from_seeds::<$k>(&seeds);
                for len in [0_usize, 1, 7, 16, 64, 1024, 4096] {
                    let bytes: Vec<u8> = (0..len)
                        .map(|i| (i.wrapping_mul(31) ^ 0x5A) as u8)
                        .collect();
                    let mut s_scalar = [u64::MAX; $k];
                    update_minhash_scalar::<$k>(&bytes, &table, &mut s_scalar);
                    let mut s_avx2 = [u64::MAX; $k];
                    // SAFETY: availability checked above.
                    unsafe { avx2::update_minhash_kway::<$k>(&bytes, &table, &mut s_avx2) };
                    assert_eq!(s_scalar, s_avx2, "AVX2 kway K={} len={}", $k, len);
                }
            }};
        }
        check_k!(16);
        check_k!(32);
        check_k!(64);
    }

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx512_kway_matches_scalar_for_k16_k32_k64_k128() {
        if !avx512::is_available() {
            eprintln!("AVX-512F unavailable; skipping avx512 K-way gather parity test");
            return;
        }
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0x9E37_79B9_u64.wrapping_add(i as u64));
                let table = build_table_from_seeds::<$k>(&seeds);
                for len in [0_usize, 1, 7, 16, 64, 1024, 4096] {
                    let bytes: Vec<u8> = (0..len)
                        .map(|i| (i.wrapping_mul(11) ^ 0xA5) as u8)
                        .collect();
                    let mut s_scalar = [u64::MAX; $k];
                    update_minhash_scalar::<$k>(&bytes, &table, &mut s_scalar);
                    let mut s_avx512 = [u64::MAX; $k];
                    // SAFETY: availability checked above.
                    unsafe { avx512::update_minhash_kway::<$k>(&bytes, &table, &mut s_avx512) };
                    assert_eq!(s_scalar, s_avx512, "AVX-512 kway K={} len={}", $k, len);
                }
            }};
        }
        check_k!(16);
        check_k!(32);
        check_k!(64);
        check_k!(128);
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_8way_matches_scalar() {
        let seeds: [u64; 8] = core::array::from_fn(|i| 0xCAFE_BABE_u64 ^ (i as u64));
        let table = build_table_from_seeds(&seeds);

        for len in [0_usize, 1, 7, 16, 64, 1024, 4096, 65_535] {
            let bytes: Vec<u8> = (0..len)
                .map(|i| (i.wrapping_mul(31) ^ 0x5A) as u8)
                .collect();
            let mut s_scalar = [u64::MAX; 8];
            update_minhash_scalar::<8>(&bytes, &table, &mut s_scalar);
            let mut s_neon = [u64::MAX; 8];
            // SAFETY: NEON is mandatory on AArch64.
            unsafe { neon::update_minhash_8way(&bytes, &table, &mut s_neon) };
            assert_eq!(s_scalar, s_neon, "NEON gather diverged at len={len}");
        }
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_kway_matches_scalar_for_k16_k32_k64() {
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0xBABE_FACE_u64.wrapping_add(i as u64));
                let table = build_table_from_seeds::<$k>(&seeds);
                for len in [0_usize, 1, 7, 16, 64, 1024, 4096] {
                    let bytes: Vec<u8> = (0..len)
                        .map(|i| (i.wrapping_mul(13) ^ 0xC3) as u8)
                        .collect();
                    let mut s_scalar = [u64::MAX; $k];
                    update_minhash_scalar::<$k>(&bytes, &table, &mut s_scalar);
                    let mut s_neon = [u64::MAX; $k];
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { neon::update_minhash_kway::<$k>(&bytes, &table, &mut s_neon) };
                    assert_eq!(s_scalar, s_neon, "NEON kway K={} len={}", $k, len);
                }
            }};
        }
        check_k!(16);
        check_k!(32);
        check_k!(64);
    }

    /// Auto dispatcher matches scalar across every K we benchmark.
    #[test]
    fn auto_kway_matches_scalar() {
        macro_rules! check_k {
            ($k:literal) => {{
                let seeds: [u64; $k] =
                    core::array::from_fn(|i| 0xDEAD_BEEF_u64.wrapping_add(i as u64));
                let table = build_table_from_seeds::<$k>(&seeds);
                let bytes: Vec<u8> = (0..2048_usize)
                    .map(|i| (i.wrapping_mul(7) ^ 0xA5) as u8)
                    .collect();
                let mut s_scalar = [u64::MAX; $k];
                update_minhash_scalar::<$k>(&bytes, &table, &mut s_scalar);
                let mut s_auto = [u64::MAX; $k];
                update_minhash_kway_auto::<$k>(&bytes, &table, &mut s_auto);
                assert_eq!(s_scalar, s_auto, "auto K={}", $k);
            }};
        }
        check_k!(8);
        check_k!(16);
        check_k!(32);
        check_k!(64);
    }
}

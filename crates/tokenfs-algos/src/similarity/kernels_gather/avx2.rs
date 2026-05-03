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

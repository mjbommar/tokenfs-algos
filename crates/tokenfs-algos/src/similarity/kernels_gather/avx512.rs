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

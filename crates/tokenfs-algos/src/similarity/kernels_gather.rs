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

/// Build a permutation table compatible with the gather kernels.
///
/// `seeds[k]` parameterizes the hash for slot `k`; the result satisfies
/// `table[b][k] == crate::hash::mix_word(b as u64 ^ seeds[k])`. The table
/// occupies `K * 256 * 8` bytes — see the module-level state-footprint
/// table.
#[must_use]
pub fn build_table_from_seeds<const K: usize>(seeds: &[u64; K]) -> [[u64; K]; TABLE_ROWS] {
    let mut table = [[0_u64; K]; TABLE_ROWS];
    let mut byte = 0_usize;
    while byte < TABLE_ROWS {
        let mut k = 0;
        while k < K {
            table[byte][k] = crate::hash::mix_word((byte as u64) ^ seeds[k]);
            k += 1;
        }
        byte += 1;
    }
    table
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
        __m128i, __m256i, _mm256_blendv_epi8, _mm256_cmpgt_epi64, _mm256_i32gather_epi64,
        _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_setr_epi32, _mm256_storeu_si256,
        _mm256_xor_si256,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m128i, __m256i, _mm256_blendv_epi8, _mm256_cmpgt_epi64, _mm256_i32gather_epi64,
        _mm256_loadu_si256, _mm256_set1_epi64x, _mm256_setr_epi32, _mm256_storeu_si256,
        _mm256_xor_si256,
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
            core::arch::x86_64::_mm256_castsi256_si128(v)
        };
        let idx_hi: __m128i = {
            let v: __m256i = _mm256_setr_epi32(4, 5, 6, 7, 0, 0, 0, 0);
            core::arch::x86_64::_mm256_castsi256_si128(v)
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
/// callers; the `avx2` / `avx512` modules above remain pinned for
/// parity-testing and microbenchmarks.
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
    update_minhash_scalar::<8>(bytes, table, sig);
}

#[cfg(test)]
mod tests {
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
}

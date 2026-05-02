//! SIMD-accelerated bit-count over `&[u64]` and `&[u8]` slices.
//!
//! Foundation kernel for `bits::rank_select` and for `bitmap` cardinality.
//! See `docs/v0.2_planning/10_BITS.md` § 4 for the spec and per-backend
//! throughput targets.
//!
//! ## API surface
//!
//! * [`popcount_u64_slice`] — runtime-dispatched bit-count over `&[u64]`.
//! * [`popcount_u8_slice`] — runtime-dispatched bit-count over `&[u8]`.
//! * [`kernels::scalar`] — portable reference path.
//! * `kernels::avx2` — x86 Mula nibble-LUT (`feature = "avx2"`).
//! * `kernels::avx512` — x86 VPOPCNTQ (`feature = "avx512"`).
//! * `kernels::neon` — AArch64 VCNT + horizontal add (`feature = "neon"`).

/// Returns the total number of set bits across `words`.
///
/// Uses the best available kernel detected at runtime (AVX-512 VPOPCNTQ
/// > AVX2 Mula nibble-LUT > scalar on x86; NEON VCNT on AArch64).
///
/// Bit-exact with the scalar reference [`kernels::scalar::popcount_u64_slice`]
/// on every backend.
#[must_use]
pub fn popcount_u64_slice(words: &[u64]) -> u64 {
    kernels::auto::popcount_u64_slice(words)
}

/// Returns the total number of set bits across `bytes`.
///
/// Uses the best available kernel detected at runtime; bit-exact with
/// [`kernels::scalar::popcount_u8_slice`].
#[must_use]
pub fn popcount_u8_slice(bytes: &[u8]) -> u64 {
    kernels::auto::popcount_u8_slice(bytes)
}

/// Pinned popcount kernels.
pub mod kernels {
    /// Runtime-dispatched popcount kernels.
    pub mod auto {
        /// Runtime-dispatched popcount over `&[u64]`.
        #[must_use]
        pub fn popcount_u64_slice(words: &[u64]) -> u64 {
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512::popcount_u64_slice(words) };
                }
            }

            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::popcount_u64_slice(words) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { super::neon::popcount_u64_slice(words) };
                }
            }

            super::scalar::popcount_u64_slice(words)
        }

        /// Runtime-dispatched popcount over `&[u8]`.
        #[must_use]
        pub fn popcount_u8_slice(bytes: &[u8]) -> u64 {
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512::popcount_u8_slice(bytes) };
                }
            }

            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::popcount_u8_slice(bytes) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { super::neon::popcount_u8_slice(bytes) };
                }
            }

            super::scalar::popcount_u8_slice(bytes)
        }
    }

    /// Portable scalar popcount.
    pub mod scalar {
        /// Counts set bits across `words` using `u64::count_ones`.
        #[must_use]
        pub fn popcount_u64_slice(words: &[u64]) -> u64 {
            let mut sum = 0_u64;
            for &word in words {
                sum += u64::from(word.count_ones());
            }
            sum
        }

        /// Counts set bits across `bytes` using `u8::count_ones`.
        ///
        /// The hot loop folds eight bytes at a time into a u64 and calls
        /// `u64::count_ones`, which lowers to one POPCNT on x86_64 and a
        /// `cnt`+`addv` reduction on AArch64. This is ~5x faster than a
        /// per-byte `count_ones` loop on stable rustc and matches what
        /// the SIMD backends fall back to on tails.
        #[must_use]
        pub fn popcount_u8_slice(bytes: &[u8]) -> u64 {
            let mut sum = 0_u64;
            let chunks = bytes.chunks_exact(8);
            let remainder = chunks.remainder();
            for chunk in chunks {
                // SAFETY: chunks_exact(8) yields slices of length 8.
                let arr: [u8; 8] = chunk.try_into().expect("chunks_exact(8)");
                sum += u64::from(u64::from_le_bytes(arr).count_ones());
            }
            for &byte in remainder {
                sum += u64::from(byte.count_ones());
            }
            sum
        }
    }

    /// x86 AVX2 popcount via the Mula nibble-LUT method.
    ///
    /// Reference: Mula, Kurz, Lemire, "Faster population counts using AVX2
    /// instructions," The Computer Journal, 2018 (arXiv:1611.07612).
    ///
    /// AVX2 has no native 64-bit popcount instruction. The kernel uses a
    /// 16-entry lookup table holding the popcount of each 4-bit nibble,
    /// applies it via `_mm256_shuffle_epi8` (per-128-bit-lane PSHUFB) on
    /// the low and high nibbles of each byte, sums the two lookups into a
    /// per-byte popcount vector, and folds that into 64-bit lane sums via
    /// `_mm256_sad_epu8` (sum-of-absolute-differences against zero).
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::scalar;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m256i, _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_extract_epi64,
            _mm256_loadu_si256, _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setr_epi8,
            _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m256i, _mm256_add_epi8, _mm256_add_epi64, _mm256_and_si256, _mm256_extract_epi64,
            _mm256_loadu_si256, _mm256_sad_epu8, _mm256_set1_epi8, _mm256_setr_epi8,
            _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_srli_epi16,
        };

        /// 32 bytes (4 u64) per inner SIMD iteration.
        const VEC_BYTES: usize = 32;

        /// 8x unrolled = 256 bytes per outer iteration. Each per-byte
        /// popcount is in `0..=8`; eight unrolled per-byte sums fit in
        /// `0..=64`, well below the u8 saturation threshold of 255, so we
        /// can defer the SAD reduction to the end of the unrolled block.
        const UNROLL_VECTORS: usize = 8;

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

        /// AVX2 Mula nibble-LUT popcount over a `&[u64]` slice.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn popcount_u64_slice(words: &[u64]) -> u64 {
            // The byte-slice kernel is the same workload (popcount over a
            // contiguous run of bytes) and writing it once keeps the two
            // entry points bit-exact.
            // SAFETY: target_feature(enable = "avx2") on this fn satisfies
            // the unsafe precondition of `popcount_bytes_avx2`.
            let bytes_ptr = words.as_ptr().cast::<u8>();
            let bytes_len = core::mem::size_of_val(words);
            // SAFETY: `bytes_ptr` and `bytes_len` describe the same memory
            // region as `words`, which is borrowed for the duration of
            // this call.
            let bytes = unsafe { core::slice::from_raw_parts(bytes_ptr, bytes_len) };
            unsafe { popcount_bytes_avx2(bytes) }
        }

        /// AVX2 Mula nibble-LUT popcount over a `&[u8]` slice.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn popcount_u8_slice(bytes: &[u8]) -> u64 {
            // SAFETY: target_feature on this fn forwards to the inner
            // kernel.
            unsafe { popcount_bytes_avx2(bytes) }
        }

        /// Inner Mula nibble-LUT kernel.
        ///
        /// # Safety
        ///
        /// AVX2 must be available; caller asserts via `target_feature`.
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn popcount_bytes_avx2(bytes: &[u8]) -> u64 {
            // Nibble-popcount LUT, broadcast to both 128-bit lanes of the
            // AVX2 vector. Built via `_mm256_setr_epi8` so the source
            // ordering reads naturally as nibble values 0..=15.
            //
            // _mm256_shuffle_epi8 is **per-128-bit-lane** — the same 16
            // entries populate both halves and the shuffle indices are
            // interpreted modulo 16 within each lane.
            let lookup: __m256i = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                2, 3, 3, 4,
            );
            let low_mask = _mm256_set1_epi8(0x0F);

            let mut acc_u64 = _mm256_setzero_si256();

            let mut index = 0_usize;
            let unroll_bytes = VEC_BYTES * UNROLL_VECTORS;

            // Outer loop: process UNROLL_VECTORS vectors per iteration,
            // accumulating per-byte popcounts in an 8-bit vector and
            // folding to u64 lanes only once per outer iteration. Each
            // per-byte popcount is in 0..=8, so 8 iterations sum to
            // 0..=64 < 255 — safe in u8 lanes without overflow.
            while index + unroll_bytes <= bytes.len() {
                let mut acc_u8 = _mm256_setzero_si256();
                for k in 0..UNROLL_VECTORS {
                    // SAFETY: `index + k * VEC_BYTES + 32 <= index +
                    // UNROLL_VECTORS * VEC_BYTES <= bytes.len()`; AVX2
                    // is enabled by the enclosing target_feature so the
                    // helper `popcnt_per_byte` precondition is met.
                    let v = unsafe {
                        _mm256_loadu_si256(
                            bytes.as_ptr().add(index + k * VEC_BYTES).cast::<__m256i>(),
                        )
                    };
                    acc_u8 =
                        _mm256_add_epi8(acc_u8, unsafe { popcnt_per_byte(v, lookup, low_mask) });
                }
                // Horizontal sum each 8-byte qword of `acc_u8` into a u16
                // within a 64-bit lane via `_mm256_sad_epu8(acc, 0)`, then
                // accumulate into the persistent u64 register.
                let zero = _mm256_setzero_si256();
                acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(acc_u8, zero));
                index += unroll_bytes;
            }

            // Single-vector loop for the leftover 32-byte windows after
            // the unrolled block.
            while index + VEC_BYTES <= bytes.len() {
                // SAFETY: index + 32 <= bytes.len() bounds the load;
                // AVX2 is enabled by the enclosing target_feature.
                let v = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
                let per_byte = unsafe { popcnt_per_byte(v, lookup, low_mask) };
                let zero = _mm256_setzero_si256();
                acc_u64 = _mm256_add_epi64(acc_u64, _mm256_sad_epu8(per_byte, zero));
                index += VEC_BYTES;
            }

            // Reduce the four 64-bit lanes of `acc_u64` to a scalar.
            let s0 = _mm256_extract_epi64::<0>(acc_u64) as u64;
            let s1 = _mm256_extract_epi64::<1>(acc_u64) as u64;
            let s2 = _mm256_extract_epi64::<2>(acc_u64) as u64;
            let s3 = _mm256_extract_epi64::<3>(acc_u64) as u64;
            let mut total = s0 + s1 + s2 + s3;

            // Scalar tail for the remaining 0..VEC_BYTES bytes.
            total += scalar::popcount_u8_slice(&bytes[index..]);
            total
        }

        /// One Mula step on a 32-byte input vector: per-byte popcount.
        ///
        /// Returns a `__m256i` whose byte at lane `i` holds
        /// `bytes[i].count_ones()` for the input vector `v`.
        ///
        /// # Safety
        ///
        /// AVX2 must be available; caller asserts via `target_feature`.
        #[target_feature(enable = "avx2")]
        #[inline]
        unsafe fn popcnt_per_byte(v: __m256i, lookup: __m256i, low_mask: __m256i) -> __m256i {
            // Low nibbles: AND with 0x0F.
            let lo = _mm256_and_si256(v, low_mask);
            // High nibbles: shift right by 4 (per 16-bit element; both
            // halves get the same shift, so AND-ing with 0x0F afterwards
            // recovers the per-byte high nibble).
            let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(v), low_mask);
            let lo_pc = _mm256_shuffle_epi8(lookup, lo);
            let hi_pc = _mm256_shuffle_epi8(lookup, hi);
            _mm256_add_epi8(lo_pc, hi_pc)
        }
    }

    /// x86 AVX-512 popcount via VPOPCNTQ.
    ///
    /// `_mm512_popcnt_epi64` (one cycle per 64-bit lane on Ice Lake and
    /// later) is the native AVX-512 popcount. Requires the
    /// `AVX512VPOPCNTDQ` CPU feature; gated on the crate's `avx512`
    /// cargo feature (which itself implies `nightly` for the unstable
    /// AVX-512 intrinsic surface).
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx512 {
        use super::scalar;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_popcnt_epi64,
            _mm512_reduce_add_epi64, _mm512_setzero_si512,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m512i, _mm512_add_epi64, _mm512_loadu_si512, _mm512_popcnt_epi64,
            _mm512_reduce_add_epi64, _mm512_setzero_si512,
        };

        /// 64 bytes (8 u64) per AVX-512 vector iteration.
        const VEC_BYTES: usize = 64;

        /// 4x unrolled = 256 bytes per outer iteration. Four independent
        /// accumulators break the dependency chain through the
        /// `_mm512_add_epi64` reductions and let the OoO scheduler issue
        /// VPOPCNTQ + VPADDQ pairs in parallel.
        const UNROLL_VECTORS: usize = 4;

        /// Returns true when AVX-512F + AVX-512VPOPCNTDQ are available at
        /// runtime.
        ///
        /// `AVX512VPOPCNTDQ` shipped on Intel Ice Lake (2019) and AMD
        /// Zen 4 (2022). The base AVX-512F flag is implied by VPOPCNTDQ
        /// support but checked independently for clarity.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("avx512f")
                && std::is_x86_feature_detected!("avx512vpopcntdq")
        }

        /// Returns true when AVX-512F + AVX-512VPOPCNTDQ are available.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// AVX-512 VPOPCNTQ popcount over a `&[u64]` slice.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX-512F and
        /// AVX-512VPOPCNTDQ.
        #[target_feature(enable = "avx512f,avx512vpopcntdq")]
        #[must_use]
        pub unsafe fn popcount_u64_slice(words: &[u64]) -> u64 {
            // Reuse the byte-driven inner kernel; popcount is associative
            // over the bit decomposition so the result is identical.
            let bytes_ptr = words.as_ptr().cast::<u8>();
            let bytes_len = core::mem::size_of_val(words);
            // SAFETY: `bytes_ptr`/`bytes_len` describe the same memory as
            // `words`, borrowed for the duration of this call.
            let bytes = unsafe { core::slice::from_raw_parts(bytes_ptr, bytes_len) };
            unsafe { popcount_bytes_avx512(bytes) }
        }

        /// AVX-512 VPOPCNTQ popcount over a `&[u8]` slice.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX-512F and
        /// AVX-512VPOPCNTDQ.
        #[target_feature(enable = "avx512f,avx512vpopcntdq")]
        #[must_use]
        pub unsafe fn popcount_u8_slice(bytes: &[u8]) -> u64 {
            // SAFETY: target_feature on this fn forwards to the inner
            // kernel.
            unsafe { popcount_bytes_avx512(bytes) }
        }

        /// Inner VPOPCNTQ kernel.
        ///
        /// # Safety
        ///
        /// AVX-512F + AVX-512VPOPCNTDQ must be available.
        #[target_feature(enable = "avx512f,avx512vpopcntdq")]
        #[inline]
        unsafe fn popcount_bytes_avx512(bytes: &[u8]) -> u64 {
            let mut acc0 = _mm512_setzero_si512();
            let mut acc1 = _mm512_setzero_si512();
            let mut acc2 = _mm512_setzero_si512();
            let mut acc3 = _mm512_setzero_si512();

            let mut index = 0_usize;
            let unroll_bytes = VEC_BYTES * UNROLL_VECTORS;

            while index + unroll_bytes <= bytes.len() {
                // SAFETY: each load reads 64 bytes and `index + 4*64 <=
                // bytes.len()` is enforced by the loop condition.
                let v0 = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast::<__m512i>()) };
                let v1 = unsafe {
                    _mm512_loadu_si512(bytes.as_ptr().add(index + VEC_BYTES).cast::<__m512i>())
                };
                let v2 = unsafe {
                    _mm512_loadu_si512(bytes.as_ptr().add(index + 2 * VEC_BYTES).cast::<__m512i>())
                };
                let v3 = unsafe {
                    _mm512_loadu_si512(bytes.as_ptr().add(index + 3 * VEC_BYTES).cast::<__m512i>())
                };
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v0));
                acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(v1));
                acc2 = _mm512_add_epi64(acc2, _mm512_popcnt_epi64(v2));
                acc3 = _mm512_add_epi64(acc3, _mm512_popcnt_epi64(v3));
                index += unroll_bytes;
            }

            while index + VEC_BYTES <= bytes.len() {
                // SAFETY: index + 64 <= bytes.len() bounds the load.
                let v = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(index).cast::<__m512i>()) };
                acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(v));
                index += VEC_BYTES;
            }

            let sum01 = _mm512_add_epi64(acc0, acc1);
            let sum23 = _mm512_add_epi64(acc2, acc3);
            let sum = _mm512_add_epi64(sum01, sum23);
            let total_simd = _mm512_reduce_add_epi64(sum) as u64;

            // Scalar tail for the remaining 0..63 bytes. Aligned-by-8
            // tails go through the chunked u64 path inside
            // `scalar::popcount_u8_slice`.
            total_simd + scalar::popcount_u8_slice(&bytes[index..])
        }
    }

    /// AArch64 NEON popcount via VCNT + horizontal add.
    ///
    /// `vcntq_u8` produces a per-byte popcount in 0..=8 lanes; multiple
    /// vectors are accumulated in a u8 vector with `vaddq_u8` and folded
    /// into u16 lanes via `vpaddlq_u8` to avoid u8 saturation. The final
    /// horizontal sum uses `vaddvq_u16`.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    pub mod neon {
        use super::scalar;

        use core::arch::aarch64::{
            uint8x16_t, vaddq_u8, vaddvq_u16, vcntq_u8, vdupq_n_u8, vld1q_u8, vpaddlq_u8,
        };

        /// 16 bytes (2 u64) per NEON vector.
        const VEC_BYTES: usize = 16;

        /// 8x unrolled = 128 bytes per outer iteration. Eight `vcntq_u8`
        /// outputs each in 0..=8 sum to 0..=64 — safe under the u8
        /// saturation threshold of 255 and large enough to amortize the
        /// `vpaddlq_u8` pairwise widening cost across many bytes.
        const UNROLL_VECTORS: usize = 8;

        /// Returns true when NEON is available at runtime.
        ///
        /// NEON is mandatory on AArch64; this exists for API symmetry
        /// with the x86 `is_available` helpers.
        #[must_use]
        pub const fn is_available() -> bool {
            true
        }

        /// NEON VCNT popcount over a `&[u64]` slice.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports NEON.
        #[target_feature(enable = "neon")]
        #[must_use]
        pub unsafe fn popcount_u64_slice(words: &[u64]) -> u64 {
            let bytes_ptr = words.as_ptr().cast::<u8>();
            let bytes_len = core::mem::size_of_val(words);
            // SAFETY: `bytes_ptr`/`bytes_len` describe the same memory as
            // `words`, borrowed for the duration of this call.
            let bytes = unsafe { core::slice::from_raw_parts(bytes_ptr, bytes_len) };
            unsafe { popcount_bytes_neon(bytes) }
        }

        /// NEON VCNT popcount over a `&[u8]` slice.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports NEON.
        #[target_feature(enable = "neon")]
        #[must_use]
        pub unsafe fn popcount_u8_slice(bytes: &[u8]) -> u64 {
            // SAFETY: target_feature on this fn forwards to the inner
            // kernel.
            unsafe { popcount_bytes_neon(bytes) }
        }

        /// Inner NEON VCNT kernel.
        ///
        /// # Safety
        ///
        /// NEON must be available; caller asserts via `target_feature`.
        #[target_feature(enable = "neon")]
        #[inline]
        unsafe fn popcount_bytes_neon(bytes: &[u8]) -> u64 {
            let mut total = 0_u64;
            let mut index = 0_usize;
            let unroll_bytes = VEC_BYTES * UNROLL_VECTORS;

            while index + unroll_bytes <= bytes.len() {
                // SAFETY: each load reads 16 bytes; the loop condition
                // bounds `index + UNROLL_VECTORS * 16 <= bytes.len()`.
                let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
                for k in 0..UNROLL_VECTORS {
                    // SAFETY: as above.
                    let v = unsafe { vld1q_u8(bytes.as_ptr().add(index + k * VEC_BYTES)) };
                    acc_u8 = vaddq_u8(acc_u8, vcntq_u8(v));
                }
                // `vpaddlq_u8` widens 16x u8 → 8x u16 via pairwise
                // addition, avoiding u8 overflow (max sum per byte =
                // 8 * 8 = 64, sum of pairs = 128 < u16::MAX).
                let widened = vpaddlq_u8(acc_u8);
                total += u64::from(vaddvq_u16(widened));
                index += unroll_bytes;
            }

            // Single-vector loop for the residual after the unrolled
            // block. Up to UNROLL_VECTORS - 1 vectors remain, so a u8
            // accumulator stays well below saturation; no inner flush
            // is needed.
            let mut acc_u8: uint8x16_t = vdupq_n_u8(0);
            while index + VEC_BYTES <= bytes.len() {
                // SAFETY: index + 16 <= bytes.len() bounds the load.
                let v = unsafe { vld1q_u8(bytes.as_ptr().add(index)) };
                acc_u8 = vaddq_u8(acc_u8, vcntq_u8(v));
                index += VEC_BYTES;
            }
            total += u64::from(vaddvq_u16(vpaddlq_u8(acc_u8)));

            // Scalar tail for the residual bytes < VEC_BYTES.
            total + scalar::popcount_u8_slice(&bytes[index..])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{kernels, popcount_u8_slice, popcount_u64_slice};
    // `Vec` is not in the no-std prelude; alias it from `alloc` for the
    // alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    fn deterministic_u64_words(n: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_f491_4f6c_dd1d)
            })
            .collect()
    }

    fn naive_popcount_u64(words: &[u64]) -> u64 {
        words.iter().map(|w| u64::from(w.count_ones())).sum()
    }

    fn naive_popcount_u8(bytes: &[u8]) -> u64 {
        bytes.iter().map(|b| u64::from(b.count_ones())).sum()
    }

    #[test]
    fn empty_slices_yield_zero() {
        assert_eq!(popcount_u64_slice(&[]), 0);
        assert_eq!(popcount_u8_slice(&[]), 0);
    }

    #[test]
    fn single_all_zeros_word() {
        assert_eq!(popcount_u64_slice(&[0_u64]), 0);
        assert_eq!(popcount_u8_slice(&[0_u8; 8]), 0);
    }

    #[test]
    fn single_all_ones_word() {
        assert_eq!(popcount_u64_slice(&[u64::MAX]), 64);
        assert_eq!(popcount_u8_slice(&[0xff_u8; 8]), 64);
    }

    #[test]
    fn matches_scalar_for_small_inputs() {
        let cases: &[&[u64]] = &[
            &[],
            &[0],
            &[u64::MAX],
            &[0, u64::MAX, 0xa5a5_a5a5_a5a5_a5a5],
            &[0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80],
        ];
        for case in cases {
            assert_eq!(
                popcount_u64_slice(case),
                kernels::scalar::popcount_u64_slice(case),
                "default vs scalar diverged on len {}",
                case.len()
            );
        }
    }

    #[test]
    fn matches_scalar_at_sub_block_lengths() {
        // Cover lengths around every plausible SIMD block boundary:
        // 4 u64 (AVX2 32-byte vec), 8 u64 (AVX-512 64-byte vec),
        // 32 u64 (AVX2 8x-unrolled 256B), and beyond.
        let seed = 0xF22_C0FFEE_u64;
        for len in [
            0_usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257,
            511, 1023,
        ] {
            let words = deterministic_u64_words(len, seed.wrapping_add(len as u64));
            let expected = naive_popcount_u64(&words);
            assert_eq!(
                popcount_u64_slice(&words),
                expected,
                "popcount_u64_slice diverged at len {len}"
            );
            assert_eq!(
                kernels::scalar::popcount_u64_slice(&words),
                expected,
                "scalar popcount_u64_slice diverged at len {len}"
            );
        }
    }

    #[test]
    fn u8_path_matches_scalar_at_byte_lengths() {
        let seed = 0xC0DE_C0DE_C0DE_C0DE_u64;
        for len in [
            0_usize, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            1024, 1025,
        ] {
            let words = deterministic_u64_words(len.div_ceil(8) + 1, seed);
            let bytes_full: Vec<u8> = words
                .iter()
                .flat_map(|w| w.to_le_bytes().into_iter())
                .collect();
            let bytes = &bytes_full[..len];
            let expected = naive_popcount_u8(bytes);
            assert_eq!(
                popcount_u8_slice(bytes),
                expected,
                "popcount_u8_slice diverged at len {len}"
            );
            assert_eq!(
                kernels::scalar::popcount_u8_slice(bytes),
                expected,
                "scalar popcount_u8_slice diverged at len {len}"
            );
        }
    }

    #[test]
    fn u64_and_u8_paths_agree_on_same_bits() {
        let seed = 0x1234_5678_9abc_def0_u64;
        let words = deterministic_u64_words(64, seed);
        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter())
            .collect();
        assert_eq!(popcount_u64_slice(&words), popcount_u8_slice(&bytes));
    }

    #[test]
    fn long_random_slice_matches_scalar() {
        let words = deterministic_u64_words(8192, 0xA1A1_B2B2_C3C3_D4D4);
        let expected = naive_popcount_u64(&words);
        assert_eq!(popcount_u64_slice(&words), expected);
        assert_eq!(kernels::scalar::popcount_u64_slice(&words), expected);
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_kernel_matches_scalar_when_available() {
        if !kernels::avx2::is_available() {
            eprintln!("avx2 unavailable on this host; skipping inline AVX2 parity test");
            return;
        }
        let words = deterministic_u64_words(1024, 0x5151_5eed);
        let expected = kernels::scalar::popcount_u64_slice(&words);
        // SAFETY: availability checked above.
        let actual = unsafe { kernels::avx2::popcount_u64_slice(&words) };
        assert_eq!(actual, expected);

        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter())
            .collect();
        let expected_b = kernels::scalar::popcount_u8_slice(&bytes);
        // SAFETY: availability checked above.
        let actual_b = unsafe { kernels::avx2::popcount_u8_slice(&bytes) };
        assert_eq!(actual_b, expected_b);
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_kernel_matches_scalar_when_available() {
        let words = deterministic_u64_words(1024, 0x5151_5eed);
        let expected = kernels::scalar::popcount_u64_slice(&words);
        // SAFETY: NEON is mandatory on AArch64.
        let actual = unsafe { kernels::neon::popcount_u64_slice(&words) };
        assert_eq!(actual, expected);

        let bytes: Vec<u8> = words
            .iter()
            .flat_map(|w| w.to_le_bytes().into_iter())
            .collect();
        let expected_b = kernels::scalar::popcount_u8_slice(&bytes);
        // SAFETY: NEON is mandatory on AArch64.
        let actual_b = unsafe { kernels::neon::popcount_u8_slice(&bytes) };
        assert_eq!(actual_b, expected_b);
    }
}

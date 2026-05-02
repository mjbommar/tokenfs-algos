//! SIMD-accelerated "is `x` in this small u32 set?" primitive.
//!
//! Linear scan with broadcast-compare. The intended workload is a short
//! haystack (≤ 256 elements) — vocab tables, content-class membership
//! tables, Bloom pre-checks, and similar. For larger haystacks a hashset
//! beats the linear scan because the haystack stops fitting in L1.
//!
//! See `docs/v0.2_planning/12_HASH_BATCHED.md` § 3 for the spec and
//! per-backend throughput targets.
//!
//! ## API surface
//!
//! * [`contains_u32_simd`] — runtime-dispatched single-needle membership.
//! * [`contains_u32_batch_simd`] — batched form: `out[i] = haystack
//!   contains needles[i]`.
//! * [`kernels::scalar`] — portable reference path.
//! * `kernels::sse41` — x86 PCMPEQD + PTEST (always available on x86_64).
//! * `kernels::avx2` — x86 VPCMPEQD + VPMOVMSKB (`feature = "avx2"`).
//! * `kernels::avx512` — x86 VPCMPEQD + KMOV (`feature = "avx512"`).
//! * `kernels::neon` — AArch64 VCEQQ + horizontal max (`feature = "neon"`).
//!
//! All backends are bit-exact with the scalar reference. The function is
//! pure with respect to its inputs and never allocates.

/// Failure modes for the fallible batched set-membership API
/// ([`try_contains_u32_batch_simd`]).
///
/// Returned instead of panicking when the caller-supplied buffer
/// lengths are inconsistent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SetMembershipBatchError {
    /// `needles.len() != out.len()`.
    LengthMismatch {
        /// Caller-supplied `needles.len()`.
        needles_len: usize,
        /// Caller-supplied `out.len()`.
        out_len: usize,
    },
}

impl core::fmt::Display for SetMembershipBatchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch {
                needles_len,
                out_len,
            } => write!(
                f,
                "set_membership batch length mismatch: needles.len()={needles_len} but out.len()={out_len}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SetMembershipBatchError {}

/// Returns true if any element of `haystack` equals `needle`.
///
/// Optimized for short haystacks (≤ 256 elements). For longer haystacks,
/// prefer a hashset.
///
/// Uses the best available kernel detected at runtime (AVX-512 > AVX2 >
/// SSE4.1 on x86; NEON on AArch64; scalar elsewhere). Bit-exact with
/// [`kernels::scalar::contains_u32`] on every backend.
#[must_use]
pub fn contains_u32_simd(haystack: &[u32], needle: u32) -> bool {
    kernels::auto::contains_u32(haystack, needle)
}

/// Writes `out[i] = haystack.contains(&needles[i])` for each `i`.
///
/// Uses the best available kernel for each per-needle scan. The output
/// buffer must have the same length as `needles`.
///
/// # Panics
///
/// Panics if `needles.len() != out.len()`. Use
/// [`try_contains_u32_batch_simd`] for a fallible variant that returns
/// [`SetMembershipBatchError`] instead.
///
/// Only compiled when the `panicking-shape-apis` Cargo feature is
/// enabled (default). Kernel/FUSE consumers should disable that
/// feature and use [`try_contains_u32_batch_simd`] (audit-R5 #157).
#[cfg(feature = "panicking-shape-apis")]
pub fn contains_u32_batch_simd(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    assert_eq!(
        needles.len(),
        out.len(),
        "set_membership batch length mismatch: needles.len()={} but out.len()={}",
        needles.len(),
        out.len(),
    );
    kernels::auto::contains_u32_batch(haystack, needles, out);
}

/// Fallible variant of [`contains_u32_batch_simd`] that returns
/// [`SetMembershipBatchError::LengthMismatch`] when `needles.len() !=
/// out.len()`, instead of panicking.
pub fn try_contains_u32_batch_simd(
    haystack: &[u32],
    needles: &[u32],
    out: &mut [bool],
) -> Result<(), SetMembershipBatchError> {
    if needles.len() != out.len() {
        return Err(SetMembershipBatchError::LengthMismatch {
            needles_len: needles.len(),
            out_len: out.len(),
        });
    }
    kernels::auto::contains_u32_batch(haystack, needles, out);
    Ok(())
}

/// Pinned set-membership kernels.
pub mod kernels {
    /// Runtime-dispatched set-membership kernels.
    pub mod auto {
        /// Runtime-dispatched single-needle membership over `&[u32]`.
        #[must_use]
        pub fn contains_u32(haystack: &[u32], needle: u32) -> bool {
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx512::contains_u32(haystack, needle) };
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
                    return unsafe { super::avx2::contains_u32(haystack, needle) };
                }
            }

            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                if super::sse41::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::sse41::contains_u32(haystack, needle) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { super::neon::contains_u32(haystack, needle) };
                }
            }

            super::scalar::contains_u32(haystack, needle)
        }

        /// Runtime-dispatched batched membership.
        ///
        /// # Panics
        ///
        /// Panics if `needles.len() != out.len()`.
        pub fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
            assert_eq!(
                needles.len(),
                out.len(),
                "set_membership batch length mismatch: needles.len()={} but out.len()={}",
                needles.len(),
                out.len(),
            );
            // The hot dispatch decision is per-call, not per-needle: the
            // detected backend is constant across the batch on a given
            // host. Resolve once and run a tight per-needle loop in the
            // chosen kernel.
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: availability was checked immediately above.
                    unsafe { super::avx512::contains_u32_batch(haystack, needles, out) };
                    return;
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
                    unsafe { super::avx2::contains_u32_batch(haystack, needles, out) };
                    return;
                }
            }

            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                if super::sse41::is_available() {
                    // SAFETY: availability was checked immediately above.
                    unsafe { super::sse41::contains_u32_batch(haystack, needles, out) };
                    return;
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe { super::neon::contains_u32_batch(haystack, needles, out) };
                    return;
                }
            }

            super::scalar::contains_u32_batch(haystack, needles, out);
        }
    }

    /// Portable scalar set-membership.
    pub mod scalar {
        /// Linear scan delegating to `slice::contains`.
        ///
        /// Acts as the reference oracle for every SIMD backend in this
        /// module. Equivalent to `haystack.iter().any(|x| *x == needle)`
        /// without the manual-iter clippy lint.
        #[must_use]
        pub fn contains_u32(haystack: &[u32], needle: u32) -> bool {
            haystack.contains(&needle)
        }

        /// Per-needle scalar batch.
        ///
        /// # Panics
        ///
        /// Panics if `needles.len() != out.len()`.
        pub fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
            assert_eq!(needles.len(), out.len());
            for (needle, slot) in needles.iter().zip(out.iter_mut()) {
                *slot = contains_u32(haystack, *needle);
            }
        }
    }

    /// x86 SSE4.1 set-membership via PCMPEQD + PTEST.
    ///
    /// Processes 4-lane u32 chunks. Each chunk is broadcast-compared with
    /// `_mm_cmpeq_epi32` (SSE2) and reduced with `_mm_testz_si128` (SSE4.1)
    /// — `PTEST` is one fused micro-op that asserts the all-zeros predicate
    /// without going through a full `movemask` round-trip.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub mod sse41 {
        use super::scalar;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m128i, _mm_cmpeq_epi32, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi32,
            _mm_testz_si128,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m128i, _mm_cmpeq_epi32, _mm_loadu_si128, _mm_or_si128, _mm_set1_epi32,
            _mm_testz_si128,
        };

        /// 4 lanes (16 bytes) per SSE vector.
        const LANES: usize = 4;

        /// 4x unrolled = 16 lanes (64 B) per outer iteration. The OR of
        /// four independent compares amortises the testz cost across the
        /// unrolled block; hits inside still branch out on the next
        /// iteration's testz.
        const UNROLL_VECTORS: usize = 4;

        /// Returns true when SSE4.1 is available at runtime.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("sse4.1")
        }

        /// Returns true when SSE4.1 is available at runtime.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// SSE4.1 single-needle membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports SSE4.1.
        #[target_feature(enable = "sse4.1")]
        #[must_use]
        pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
            let broadcast = _mm_set1_epi32(needle as i32);
            let mut index = 0_usize;
            let unroll_lanes = LANES * UNROLL_VECTORS;

            while index + unroll_lanes <= haystack.len() {
                // SAFETY: each load reads 16 bytes (4 u32s); the loop
                // bound enforces `index + UNROLL_VECTORS * LANES <=
                // haystack.len()`. SSE4.1 is enabled by the enclosing
                // target_feature.
                let v0 = unsafe { _mm_loadu_si128(haystack.as_ptr().add(index).cast::<__m128i>()) };
                let v1 = unsafe {
                    _mm_loadu_si128(haystack.as_ptr().add(index + LANES).cast::<__m128i>())
                };
                let v2 = unsafe {
                    _mm_loadu_si128(haystack.as_ptr().add(index + 2 * LANES).cast::<__m128i>())
                };
                let v3 = unsafe {
                    _mm_loadu_si128(haystack.as_ptr().add(index + 3 * LANES).cast::<__m128i>())
                };
                let c0 = _mm_cmpeq_epi32(v0, broadcast);
                let c1 = _mm_cmpeq_epi32(v1, broadcast);
                let c2 = _mm_cmpeq_epi32(v2, broadcast);
                let c3 = _mm_cmpeq_epi32(v3, broadcast);
                let or01 = _mm_or_si128(c0, c1);
                let or23 = _mm_or_si128(c2, c3);
                let any = _mm_or_si128(or01, or23);
                if _mm_testz_si128(any, any) == 0 {
                    return true;
                }
                index += unroll_lanes;
            }

            // Single-vector loop for the leftover lanes after the unrolled
            // block.
            while index + LANES <= haystack.len() {
                // SAFETY: index + LANES <= haystack.len() bounds the load.
                let chunk =
                    unsafe { _mm_loadu_si128(haystack.as_ptr().add(index).cast::<__m128i>()) };
                let cmp = _mm_cmpeq_epi32(chunk, broadcast);
                if _mm_testz_si128(cmp, cmp) == 0 {
                    return true;
                }
                index += LANES;
            }

            scalar::contains_u32(&haystack[index..], needle)
        }

        /// SSE4.1 batched membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports SSE4.1.
        ///
        /// # Panics
        ///
        /// Panics if `needles.len() != out.len()`.
        #[target_feature(enable = "sse4.1")]
        pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
            assert_eq!(needles.len(), out.len());
            for (needle, slot) in needles.iter().zip(out.iter_mut()) {
                // SAFETY: target_feature(enable = "sse4.1") on this fn
                // forwards the SSE4.1 precondition to the inner kernel.
                *slot = unsafe { contains_u32(haystack, *needle) };
            }
        }
    }

    /// x86 AVX2 set-membership via VPCMPEQD + VPMOVMSKB.
    ///
    /// Processes 8-lane u32 chunks (one `__m256i` = 32 bytes). Each chunk
    /// is broadcast-compared with `_mm256_cmpeq_epi32` and the all-zeros
    /// predicate is checked with `_mm256_testz_si256` (AVX `VPTEST`-like).
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::scalar;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m256i, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_testz_si256,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m256i, _mm256_cmpeq_epi32, _mm256_loadu_si256, _mm256_set1_epi32, _mm256_testz_si256,
        };

        /// 8 lanes (32 bytes) per AVX2 vector.
        const LANES: usize = 8;

        /// 4x unrolled = 32 lanes (128 B) per outer iteration. The OR of
        /// four independent compares short-circuits the early-exit cost
        /// for haystacks that miss; for haystacks that hit, the early
        /// exit triggers within the 32-lane window so the unroll cost is
        /// at most one extra compare.
        const UNROLL_VECTORS: usize = 4;

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

        /// AVX2 single-needle membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
            let broadcast = _mm256_set1_epi32(needle as i32);
            let mut index = 0_usize;
            let unroll_lanes = LANES * UNROLL_VECTORS;

            // Outer loop: process UNROLL_VECTORS vectors per iteration; OR
            // their compare results together so the testz amortises across
            // the unrolled block. Hits inside the block still branch out
            // on the next iteration's testz.
            while index + unroll_lanes <= haystack.len() {
                // SAFETY: each load reads 32 bytes (8 u32s) and `index +
                // UNROLL_VECTORS * LANES <= haystack.len()` is enforced by
                // the loop condition. AVX2 is enabled by the enclosing
                // target_feature.
                let v0 =
                    unsafe { _mm256_loadu_si256(haystack.as_ptr().add(index).cast::<__m256i>()) };
                let v1 = unsafe {
                    _mm256_loadu_si256(haystack.as_ptr().add(index + LANES).cast::<__m256i>())
                };
                let v2 = unsafe {
                    _mm256_loadu_si256(haystack.as_ptr().add(index + 2 * LANES).cast::<__m256i>())
                };
                let v3 = unsafe {
                    _mm256_loadu_si256(haystack.as_ptr().add(index + 3 * LANES).cast::<__m256i>())
                };
                let c0 = _mm256_cmpeq_epi32(v0, broadcast);
                let c1 = _mm256_cmpeq_epi32(v1, broadcast);
                let c2 = _mm256_cmpeq_epi32(v2, broadcast);
                let c3 = _mm256_cmpeq_epi32(v3, broadcast);
                // _mm256_or_si256 isn't strictly needed: testz accepts two
                // operands and returns nonzero iff their AND is zero. We
                // can fold pairs without an extra OR by passing distinct
                // operands, but readability is better with explicit OR.
                use core::arch::x86_64::_mm256_or_si256;
                let or01 = _mm256_or_si256(c0, c1);
                let or23 = _mm256_or_si256(c2, c3);
                let any = _mm256_or_si256(or01, or23);
                if _mm256_testz_si256(any, any) == 0 {
                    return true;
                }
                index += unroll_lanes;
            }

            // Single-vector loop for the leftover lanes after the
            // unrolled block.
            while index + LANES <= haystack.len() {
                // SAFETY: index + LANES <= haystack.len() bounds the load.
                let v =
                    unsafe { _mm256_loadu_si256(haystack.as_ptr().add(index).cast::<__m256i>()) };
                let cmp = _mm256_cmpeq_epi32(v, broadcast);
                if _mm256_testz_si256(cmp, cmp) == 0 {
                    return true;
                }
                index += LANES;
            }

            scalar::contains_u32(&haystack[index..], needle)
        }

        /// AVX2 batched membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        ///
        /// # Panics
        ///
        /// Panics if `needles.len() != out.len()`.
        #[target_feature(enable = "avx2")]
        pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
            assert_eq!(needles.len(), out.len());
            for (needle, slot) in needles.iter().zip(out.iter_mut()) {
                // SAFETY: target_feature(enable = "avx2") on this fn
                // forwards the AVX2 precondition to the inner kernel.
                *slot = unsafe { contains_u32(haystack, *needle) };
            }
        }
    }

    /// x86 AVX-512 set-membership via VPCMPEQD-mask + KORTESTW.
    ///
    /// Processes 16-lane u32 chunks (one `__m512i` = 64 bytes).
    /// `_mm512_cmpeq_epi32_mask` returns a `__mmask16` directly; testing
    /// against zero is a single instruction and avoids the movemask round
    /// trip.
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx512 {
        use super::scalar;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m512i, _mm512_cmpeq_epi32_mask, _mm512_loadu_si512, _mm512_set1_epi32,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m512i, _mm512_cmpeq_epi32_mask, _mm512_loadu_si512, _mm512_set1_epi32,
        };

        /// 16 lanes (64 bytes) per AVX-512 vector.
        const LANES: usize = 16;

        /// Returns true when AVX-512F is available at runtime.
        ///
        /// `_mm512_cmpeq_epi32_mask` is part of AVX-512F.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("avx512f")
        }

        /// Returns true when AVX-512F is available at runtime.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// AVX-512 single-needle membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX-512F.
        #[target_feature(enable = "avx512f")]
        #[must_use]
        pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
            let broadcast = _mm512_set1_epi32(needle as i32);
            let mut index = 0_usize;

            while index + LANES <= haystack.len() {
                // SAFETY: index + LANES <= haystack.len() bounds the load
                // and the enclosing target_feature supplies AVX-512F.
                let v =
                    unsafe { _mm512_loadu_si512(haystack.as_ptr().add(index).cast::<__m512i>()) };
                let mask = _mm512_cmpeq_epi32_mask(v, broadcast);
                if mask != 0 {
                    return true;
                }
                index += LANES;
            }

            scalar::contains_u32(&haystack[index..], needle)
        }

        /// AVX-512 batched membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX-512F.
        ///
        /// # Panics
        ///
        /// Panics if `needles.len() != out.len()`.
        #[target_feature(enable = "avx512f")]
        pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
            assert_eq!(needles.len(), out.len());
            for (needle, slot) in needles.iter().zip(out.iter_mut()) {
                // SAFETY: target_feature(enable = "avx512f") on this fn
                // forwards the AVX-512F precondition to the inner kernel.
                *slot = unsafe { contains_u32(haystack, *needle) };
            }
        }
    }

    /// AArch64 NEON set-membership via VCEQQ + horizontal max.
    ///
    /// Processes 4-lane u32 chunks (one `uint32x4_t` = 16 bytes). NEON
    /// has no movemask, so the per-lane equality vector is reduced via
    /// `vmaxvq_u32` — non-zero iff any lane matched.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    pub mod neon {
        use super::scalar;

        use core::arch::aarch64::{vceqq_u32, vdupq_n_u32, vld1q_u32, vmaxvq_u32};

        /// 4 lanes (16 bytes) per NEON vector.
        const LANES: usize = 4;

        /// Returns true when NEON is available at runtime.
        ///
        /// NEON is mandatory on AArch64; this exists for API symmetry
        /// with the x86 `is_available` helpers.
        #[must_use]
        pub const fn is_available() -> bool {
            true
        }

        /// NEON single-needle membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports NEON.
        #[target_feature(enable = "neon")]
        #[must_use]
        pub unsafe fn contains_u32(haystack: &[u32], needle: u32) -> bool {
            let broadcast = vdupq_n_u32(needle);
            let mut index = 0_usize;

            while index + LANES <= haystack.len() {
                // SAFETY: index + LANES <= haystack.len() bounds the load
                // and the enclosing target_feature supplies NEON.
                let v = unsafe { vld1q_u32(haystack.as_ptr().add(index)) };
                let cmp = vceqq_u32(v, broadcast);
                if vmaxvq_u32(cmp) != 0 {
                    return true;
                }
                index += LANES;
            }

            scalar::contains_u32(&haystack[index..], needle)
        }

        /// NEON batched membership.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports NEON.
        ///
        /// # Panics
        ///
        /// Panics if `needles.len() != out.len()`.
        #[target_feature(enable = "neon")]
        pub unsafe fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
            assert_eq!(needles.len(), out.len());
            for (needle, slot) in needles.iter().zip(out.iter_mut()) {
                // SAFETY: target_feature(enable = "neon") on this fn
                // forwards the NEON precondition to the inner kernel.
                *slot = unsafe { contains_u32(haystack, *needle) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;

    #[cfg(feature = "panicking-shape-apis")]
    use super::contains_u32_batch_simd;
    use super::{SetMembershipBatchError, contains_u32_simd, kernels, try_contains_u32_batch_simd};

    fn deterministic_haystack(n: usize, seed: u64) -> Vec<u32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32
            })
            .collect()
    }

    #[test]
    fn empty_haystack_is_always_false() {
        assert!(!contains_u32_simd(&[], 0));
        assert!(!contains_u32_simd(&[], u32::MAX));
        assert!(!kernels::scalar::contains_u32(&[], 42));
    }

    #[test]
    fn single_element_haystack_matches_only_for_equal() {
        assert!(contains_u32_simd(&[7], 7));
        assert!(!contains_u32_simd(&[7], 8));
    }

    #[test]
    fn needle_at_first_middle_last_position() {
        let haystack: Vec<u32> = (10_u32..42).collect();
        assert!(contains_u32_simd(&haystack, 10));
        assert!(contains_u32_simd(&haystack, 25));
        assert!(contains_u32_simd(&haystack, 41));
        assert!(!contains_u32_simd(&haystack, 9));
        assert!(!contains_u32_simd(&haystack, 42));
    }

    #[test]
    fn matches_scalar_at_sub_block_lengths() {
        // Cover every plausible SIMD block boundary plus surrounding sizes:
        // 4 lanes (SSE/NEON), 8 lanes (AVX2), 16 lanes (AVX-512), 32 lanes
        // (AVX2 8x4 unroll), and beyond.
        let lengths = [
            0_usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256,
            257, 1023,
        ];
        let seed = 0xF22_C0FFEE_u64;
        for len in lengths {
            let haystack = deterministic_haystack(len, seed.wrapping_add(len as u64));
            // Probe a needle that is present (when len > 0) and one that is
            // absent.
            let present = if len > 0 { haystack[len / 2] } else { 0 };
            let absent = present.wrapping_add(0xDEAD_BEEF);
            // Re-derive the absent-ness in case `wrapping_add` collides; the
            // probability is 1/2^32 but tests should be deterministic.
            let absent = if haystack.contains(&absent) {
                absent.wrapping_add(1)
            } else {
                absent
            };

            assert_eq!(
                contains_u32_simd(&haystack, present),
                haystack.contains(&present),
                "len {len}: simd != contains for present needle"
            );
            assert_eq!(
                contains_u32_simd(&haystack, absent),
                haystack.contains(&absent),
                "len {len}: simd != contains for absent needle"
            );
            assert_eq!(
                kernels::scalar::contains_u32(&haystack, present),
                haystack.contains(&present),
                "len {len}: scalar != contains for present needle"
            );
        }
    }

    #[test]
    fn property_random_needles_match_slice_contains() {
        // Deterministic property test (no proptest dep needed at this level).
        let haystacks: &[(usize, u64)] = &[
            (0, 1),
            (1, 2),
            (4, 3),
            (8, 4),
            (16, 5),
            (33, 6),
            (256, 7),
            (1024, 8),
        ];
        for (len, seed) in haystacks {
            let haystack = deterministic_haystack(*len, *seed);
            let mut needle_state = seed.wrapping_mul(0x9E37_79B9);
            for _ in 0..32 {
                needle_state ^= needle_state >> 12;
                needle_state ^= needle_state << 25;
                needle_state ^= needle_state >> 27;
                let needle = needle_state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32;
                let expected = haystack.contains(&needle);
                let actual = contains_u32_simd(&haystack, needle);
                assert_eq!(
                    actual, expected,
                    "len {len} seed {seed} needle {needle} expected {expected} got {actual}"
                );
            }
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn batched_matches_scalar_batched() {
        let haystack: Vec<u32> = (0_u32..200).collect();
        let needles: Vec<u32> = (0_u32..512).map(|i| i.wrapping_mul(7)).collect();
        let mut out_simd = vec![false; needles.len()];
        let mut out_scalar = vec![false; needles.len()];
        contains_u32_batch_simd(&haystack, &needles, &mut out_simd);
        kernels::scalar::contains_u32_batch(&haystack, &needles, &mut out_scalar);
        assert_eq!(out_simd, out_scalar);
        // Spot-check against `slice::contains` for sanity.
        for (i, needle) in needles.iter().enumerate() {
            assert_eq!(out_simd[i], haystack.contains(needle));
        }
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn batched_empty_inputs_no_op() {
        let haystack: [u32; 4] = [1, 2, 3, 4];
        let needles: [u32; 0] = [];
        let mut out: [bool; 0] = [];
        contains_u32_batch_simd(&haystack, &needles, &mut out);
        contains_u32_batch_simd(&[], &needles, &mut out);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn batched_empty_haystack_yields_all_false() {
        let needles = [1_u32, 2, 3, u32::MAX, 0];
        let mut out = [true; 5];
        contains_u32_batch_simd(&[], &needles, &mut out);
        assert_eq!(out, [false; 5]);
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    #[should_panic(expected = "set_membership batch length mismatch")]
    fn batched_panics_on_length_mismatch() {
        let haystack = [1_u32, 2, 3];
        let needles = [1_u32, 2, 3, 4];
        let mut out = vec![false; 3];
        contains_u32_batch_simd(&haystack, &needles, &mut out);
    }

    #[test]
    fn try_batched_returns_err_on_length_mismatch() {
        let haystack = [1_u32, 2, 3];
        let needles = [1_u32, 2, 3, 4];
        let mut out = vec![false; 3];
        let err = try_contains_u32_batch_simd(&haystack, &needles, &mut out).unwrap_err();
        assert_eq!(
            err,
            SetMembershipBatchError::LengthMismatch {
                needles_len: 4,
                out_len: 3
            }
        );
    }

    #[cfg(feature = "panicking-shape-apis")]
    #[test]
    fn try_batched_returns_ok_and_matches_panic_version() {
        let haystack: Vec<u32> = (0_u32..200).collect();
        let needles: Vec<u32> = (0_u32..64).map(|i| i.wrapping_mul(7)).collect();
        let mut try_out = vec![false; needles.len()];
        try_contains_u32_batch_simd(&haystack, &needles, &mut try_out).unwrap();
        let mut panic_out = vec![false; needles.len()];
        contains_u32_batch_simd(&haystack, &needles, &mut panic_out);
        assert_eq!(try_out, panic_out);
        for (i, needle) in needles.iter().enumerate() {
            assert_eq!(try_out[i], haystack.contains(needle));
        }
    }

    #[test]
    fn try_batched_empty_inputs_no_op() {
        let haystack: [u32; 4] = [1, 2, 3, 4];
        let needles: [u32; 0] = [];
        let mut out: [bool; 0] = [];
        try_contains_u32_batch_simd(&haystack, &needles, &mut out).unwrap();
    }

    #[test]
    fn boundary_needle_values() {
        // u32::MAX, 0, sign-bit-set values: ensure the i32-cast inside
        // _mm*_set1_epi32 doesn't change the comparison semantics.
        let edge_values = [0_u32, 1, u32::MAX, 0x8000_0000, 0x7FFF_FFFF];
        for &needle in &edge_values {
            let haystack = vec![needle];
            assert!(contains_u32_simd(&haystack, needle));
            // Scrub the haystack to ensure absent works too.
            let absent_haystack = vec![needle.wrapping_add(1)];
            assert!(!contains_u32_simd(&absent_haystack, needle));
        }
    }

    // -----------------------------------------------------------------
    // Pinned per-backend parity: compare each kernel against scalar over
    // a deterministic test grid. Runtime-skip when the backend is not
    // available on this host.
    // -----------------------------------------------------------------

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn sse41_kernel_matches_scalar_when_available() {
        if !kernels::sse41::is_available() {
            eprintln!("sse4.1 unavailable on this host; skipping inline SSE4.1 parity test");
            return;
        }
        for len in [0_usize, 1, 3, 4, 7, 8, 16, 33, 64, 256, 1023] {
            let haystack = deterministic_haystack(len, 0x5151_5eed ^ (len as u64));
            for &needle in &[0_u32, u32::MAX, 0x8000_0000] {
                let expected = kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: availability checked above.
                let actual = unsafe { kernels::sse41::contains_u32(&haystack, needle) };
                assert_eq!(actual, expected, "len {len} needle {needle}");
            }
            if len > 0 {
                let needle = haystack[len / 2];
                let expected = kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: availability checked above.
                let actual = unsafe { kernels::sse41::contains_u32(&haystack, needle) };
                assert_eq!(actual, expected);
            }
        }
    }

    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_kernel_matches_scalar_when_available() {
        if !kernels::avx2::is_available() {
            eprintln!("avx2 unavailable on this host; skipping inline AVX2 parity test");
            return;
        }
        for len in [0_usize, 1, 3, 4, 7, 8, 16, 31, 32, 33, 64, 128, 256, 1023] {
            let haystack = deterministic_haystack(len, 0xA1A1_B2B2 ^ (len as u64));
            for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
                let expected = kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: availability checked above.
                let actual = unsafe { kernels::avx2::contains_u32(&haystack, needle) };
                assert_eq!(actual, expected, "len {len} needle {needle}");
            }
            if len > 0 {
                // Probe with a known-present needle at a few positions.
                for &pos in &[0_usize, len / 2, len - 1] {
                    let needle = haystack[pos];
                    let expected = kernels::scalar::contains_u32(&haystack, needle);
                    // SAFETY: availability checked above.
                    let actual = unsafe { kernels::avx2::contains_u32(&haystack, needle) };
                    assert_eq!(actual, expected, "len {len} pos {pos}");
                }
            }
        }
    }

    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx512_kernel_matches_scalar_when_available() {
        if !kernels::avx512::is_available() {
            eprintln!("avx512f unavailable on this host; skipping inline AVX-512 parity test");
            return;
        }
        for len in [0_usize, 1, 8, 15, 16, 17, 32, 64, 128, 256, 1023] {
            let haystack = deterministic_haystack(len, 0xC0DE_C0DE ^ (len as u64));
            for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
                let expected = kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: availability checked above.
                let actual = unsafe { kernels::avx512::contains_u32(&haystack, needle) };
                assert_eq!(actual, expected, "len {len} needle {needle}");
            }
            if len > 0 {
                for &pos in &[0_usize, len / 2, len - 1] {
                    let needle = haystack[pos];
                    let expected = kernels::scalar::contains_u32(&haystack, needle);
                    // SAFETY: availability checked above.
                    let actual = unsafe { kernels::avx512::contains_u32(&haystack, needle) };
                    assert_eq!(actual, expected, "len {len} pos {pos}");
                }
            }
        }
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_kernel_matches_scalar() {
        for len in [0_usize, 1, 3, 4, 7, 8, 16, 31, 32, 33, 64, 128, 256, 1023] {
            let haystack = deterministic_haystack(len, 0xBEEF_F00D ^ (len as u64));
            for &needle in &[0_u32, 1, u32::MAX, 0x8000_0000] {
                let expected = kernels::scalar::contains_u32(&haystack, needle);
                // SAFETY: NEON is mandatory on AArch64.
                let actual = unsafe { kernels::neon::contains_u32(&haystack, needle) };
                assert_eq!(actual, expected, "len {len} needle {needle}");
            }
            if len > 0 {
                for &pos in &[0_usize, len / 2, len - 1] {
                    let needle = haystack[pos];
                    let expected = kernels::scalar::contains_u32(&haystack, needle);
                    // SAFETY: NEON is mandatory on AArch64.
                    let actual = unsafe { kernels::neon::contains_u32(&haystack, needle) };
                    assert_eq!(actual, expected, "len {len} pos {pos}");
                }
            }
        }
    }
}

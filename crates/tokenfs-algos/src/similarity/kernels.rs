//! Pinned dense-distance kernels.
//!
//! Each backend submodule (`scalar`, `avx2`, `neon`) exposes the same set of
//! functions so the dispatcher in [`auto`] can pick at runtime without
//! reshaping caller code. AVX2 and NEON kernels land in follow-up tasks; until
//! then the dispatcher always reaches scalar.
//!
//! `u32` integer kernels accumulate into `u64` to avoid overflow on long
//! vectors of bin counts. `f32` kernels use Kahan-friendly straightforward
//! sums; documented numeric tolerance for SIMD reductions will be added when
//! the SIMD backends land.

/// Runtime-dispatched dense-distance kernels.
pub mod auto {
    use super::scalar;

    /// Inner product of two `u32` vectors using the best available kernel.
    #[must_use]
    pub fn dot_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::dot_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked immediately above; lengths match.
                return Some(unsafe { super::avx2::dot_u32(a, b) });
            }
        }
        scalar::dot_u32(a, b)
    }

    /// L1 distance of two `u32` vectors using the best available kernel.
    #[must_use]
    pub fn l1_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::l1_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::l1_u32(a, b) });
            }
        }
        scalar::l1_u32(a, b)
    }

    /// Squared L2 distance of two `u32` vectors.
    #[must_use]
    pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::l2_squared_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::l2_squared_u32(a, b) });
            }
        }
        scalar::l2_squared_u32(a, b)
    }

    /// L2 distance of two `u32` vectors as `f64`.
    #[must_use]
    pub fn l2_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        l2_squared_u32(a, b).map(|s| crate::math::sqrt_f64(s as f64))
    }

    /// Cosine similarity of two `u32` vectors as `f64`.
    #[must_use]
    pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::cosine_similarity_u32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::cosine_similarity_u32(a, b) });
            }
        }
        scalar::cosine_similarity_u32(a, b)
    }

    /// Inner product of two `f32` vectors.
    #[must_use]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::dot_f32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::dot_f32(a, b) });
            }
        }
        scalar::dot_f32(a, b)
    }

    /// Squared L2 distance of two `f32` vectors.
    #[must_use]
    pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::l2_squared_f32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::l2_squared_f32(a, b) });
            }
        }
        scalar::l2_squared_f32(a, b)
    }

    /// Cosine similarity of two `f32` vectors.
    #[must_use]
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        {
            if super::neon::is_available() {
                // SAFETY: NEON is mandatory on AArch64.
                return Some(unsafe { super::neon::cosine_similarity_f32(a, b) });
            }
        }
        #[cfg(all(
            feature = "std",
            feature = "avx2",
            any(target_arch = "x86", target_arch = "x86_64")
        ))]
        {
            if super::avx2::is_available() {
                // SAFETY: availability checked above; lengths match.
                return Some(unsafe { super::avx2::cosine_similarity_f32(a, b) });
            }
        }
        scalar::cosine_similarity_f32(a, b)
    }
}

/// Portable scalar dense-distance kernels.
///
/// These are the pinned reference implementations. SIMD backends must produce
/// identical results for integer kernels and stay within documented tolerance
/// for floating-point kernels.
pub mod scalar {
    /// Inner product of two `u32` vectors. Returns `None` on length mismatch.
    #[must_use]
    pub fn dot_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        let mut sum = 0_u64;
        for (&x, &y) in a.iter().zip(b) {
            sum = sum.wrapping_add(u64::from(x) * u64::from(y));
        }
        Some(sum)
    }

    /// Manhattan / L1 distance of two `u32` vectors.
    #[must_use]
    pub fn l1_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        let mut sum = 0_u64;
        for (&x, &y) in a.iter().zip(b) {
            sum = sum.wrapping_add(u64::from(x.abs_diff(y)));
        }
        Some(sum)
    }

    /// Squared L2 distance of two `u32` vectors.
    #[must_use]
    pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> Option<u64> {
        if a.len() != b.len() {
            return None;
        }
        let mut sum = 0_u64;
        for (&x, &y) in a.iter().zip(b) {
            let d = u64::from(x.abs_diff(y));
            sum = sum.wrapping_add(d * d);
        }
        Some(sum)
    }

    /// Cosine similarity of two `u32` vectors as `f64` in `[-1, 1]` (in
    /// practice `[0, 1]` for non-negative count vectors).
    ///
    /// Returns `None` on length mismatch. Returns `Some(0.0)` when either
    /// vector has zero norm — this is the convention adopted by the existing
    /// [`crate::divergence::cosine_distance_counts_u32`] (which then maps to
    /// distance = 1.0).
    #[must_use]
    pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
        if a.len() != b.len() {
            return None;
        }
        let mut dot = 0_u64;
        let mut norm_a = 0_u64;
        let mut norm_b = 0_u64;
        for (&x, &y) in a.iter().zip(b) {
            dot = dot.wrapping_add(u64::from(x) * u64::from(y));
            norm_a = norm_a.wrapping_add(u64::from(x) * u64::from(x));
            norm_b = norm_b.wrapping_add(u64::from(y) * u64::from(y));
        }
        if norm_a == 0 || norm_b == 0 {
            return Some(0.0);
        }
        let denom = crate::math::sqrt_f64((norm_a as f64) * (norm_b as f64));
        Some((dot as f64) / denom)
    }

    /// Inner product of two `f32` vectors.
    #[must_use]
    pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let mut sum = 0.0_f32;
        for (&x, &y) in a.iter().zip(b) {
            sum += x * y;
        }
        Some(sum)
    }

    /// Squared L2 distance of two `f32` vectors.
    #[must_use]
    pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let mut sum = 0.0_f32;
        for (&x, &y) in a.iter().zip(b) {
            let d = x - y;
            sum += d * d;
        }
        Some(sum)
    }

    /// Cosine similarity of two `f32` vectors.
    ///
    /// Returns `None` on length mismatch and `Some(0.0)` when either vector
    /// has zero norm.
    #[must_use]
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
        if a.len() != b.len() {
            return None;
        }
        let mut dot = 0.0_f32;
        let mut norm_a = 0.0_f32;
        let mut norm_b = 0.0_f32;
        for (&x, &y) in a.iter().zip(b) {
            dot += x * y;
            norm_a += x * x;
            norm_b += y * y;
        }
        if norm_a == 0.0 || norm_b == 0.0 {
            return Some(0.0);
        }
        Some(dot / crate::math::sqrt_f32(norm_a * norm_b))
    }
}

/// AVX2 dense-distance kernels.
///
/// Each kernel processes 8 lanes per iteration (256-bit register, 8×u32 or
/// 8×f32). Tails fall back to scalar.
///
/// **Numeric tolerance for SIMD reductions:** integer kernels (`dot_u32`,
/// `l1_u32`, `l2_squared_u32`) are bit-exact with [`scalar`]. Floating-point
/// kernels (`dot_f32`, `l2_squared_f32`, `cosine_similarity_f32`) use a
/// different summation order than scalar (8-way tree reduction vs.
/// left-to-right) so results can differ by a few ULP on long vectors. The
/// SIMD parity tests assert relative tolerance of `1e-5` for f32.
#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub mod avx2 {
    use super::scalar;

    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        __m256, __m256i, _mm_add_epi64, _mm_extract_epi64, _mm256_add_epi64,
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_loadu_si256, _mm256_max_epu32, _mm256_min_epu32, _mm256_mul_epu32,
        _mm256_setzero_ps, _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_storeu_ps,
        _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_ps,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        __m256, __m256i, _mm_add_epi64, _mm_extract_epi64, _mm256_add_epi64,
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_loadu_si256, _mm256_max_epu32, _mm256_min_epu32, _mm256_mul_epu32,
        _mm256_setzero_ps, _mm256_setzero_si256, _mm256_shuffle_epi32, _mm256_storeu_ps,
        _mm256_storeu_si256, _mm256_sub_epi32, _mm256_sub_ps,
    };

    const LANES_U32: usize = 8;
    const LANES_F32: usize = 8;

    /// Returns true when AVX2 is available at runtime.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn is_available() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    /// Returns true when AVX2 is available at runtime.
    #[cfg(not(feature = "std"))]
    #[must_use]
    pub const fn is_available() -> bool {
        false
    }

    /// Reduces a `__m256i` of four `u64` lanes to a scalar `u64`.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn hadd_u64x4(v: __m256i) -> u64 {
        // SAFETY: AVX2 enabled.
        let lo = _mm256_castsi256_si128(v);
        let hi = _mm256_extracti128_si256::<1>(v);
        let sum = _mm_add_epi64(lo, hi);
        // sum = [a, b]; we want a + b.
        let a = _mm_extract_epi64::<0>(sum) as u64;
        let b = _mm_extract_epi64::<1>(sum) as u64;
        a.wrapping_add(b)
    }

    /// Reduces a `__m256` of eight `f32` lanes to a scalar `f32`.
    #[target_feature(enable = "avx2,fma")]
    #[inline]
    unsafe fn hadd_f32x8(v: __m256) -> f32 {
        let mut tmp = [0.0_f32; 8];
        // SAFETY: tmp is 32-byte writable; alignment-tolerant store.
        unsafe { _mm256_storeu_ps(tmp.as_mut_ptr(), v) };
        // Pairwise tree reduction (deterministic, lower error than left-to-right).
        let p0 = tmp[0] + tmp[1];
        let p1 = tmp[2] + tmp[3];
        let p2 = tmp[4] + tmp[5];
        let p3 = tmp[6] + tmp[7];
        (p0 + p1) + (p2 + p3)
    }

    /// Multiplies low-32-of-each-u64 lane and accumulates into u64 lanes.
    /// `_mm256_mul_epu32` reads the *even* 32-bit positions (0, 2, 4, 6).
    /// To cover the odd positions (1, 3, 5, 7) we shuffle them down first.
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mul_accum_u32_pair(
        a: __m256i,
        b: __m256i,
        acc_lo: __m256i,
        acc_hi: __m256i,
    ) -> (__m256i, __m256i) {
        // SAFETY: AVX2 enabled.
        let prod_even = _mm256_mul_epu32(a, b);
        // Shuffle 0xB1 = swap each adjacent 32-bit pair so odd lanes become even.
        let a_odd = _mm256_shuffle_epi32::<0xB1>(a);
        let b_odd = _mm256_shuffle_epi32::<0xB1>(b);
        let prod_odd = _mm256_mul_epu32(a_odd, b_odd);
        (
            _mm256_add_epi64(acc_lo, prod_even),
            _mm256_add_epi64(acc_hi, prod_odd),
        )
    }

    /// Inner product of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn dot_u32(a: &[u32], b: &[u32]) -> u64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        let mut i = 0;
        while i + LANES_U32 <= a.len() {
            // SAFETY: bounds checked by loop condition.
            let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
            let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
            (acc_lo, acc_hi) = unsafe { mul_accum_u32_pair(va, vb, acc_lo, acc_hi) };
            i += LANES_U32;
        }
        // SAFETY: AVX2 enabled.
        let total = unsafe { hadd_u64x4(_mm256_add_epi64(acc_lo, acc_hi)) };
        let tail = scalar::dot_u32(&a[i..], &b[i..]).unwrap_or(0);
        total.wrapping_add(tail)
    }

    /// L1 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn l1_u32(a: &[u32], b: &[u32]) -> u64 {
        debug_assert_eq!(a.len(), b.len());
        // Use max - min for unsigned absolute-difference (no overflow risk).
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        let one_vec = unsafe {
            // Build a vector of 1u32 by storing then reloading; cheaper than _mm256_set1.
            let mut buf = [1_u32; LANES_U32];
            _mm256_loadu_si256(buf.as_mut_ptr().cast::<__m256i>())
        };
        let mut i = 0;
        while i + LANES_U32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
            let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
            let mx = _mm256_max_epu32(va, vb);
            let mn = _mm256_min_epu32(va, vb);
            let d = _mm256_sub_epi32(mx, mn);
            // Multiply each diff by 1 to widen to u64 lanes via mul_epu32.
            (acc_lo, acc_hi) = unsafe { mul_accum_u32_pair(d, one_vec, acc_lo, acc_hi) };
            i += LANES_U32;
        }
        // SAFETY: AVX2 enabled.
        let total = unsafe { hadd_u64x4(_mm256_add_epi64(acc_lo, acc_hi)) };
        let tail = scalar::l1_u32(&a[i..], &b[i..]).unwrap_or(0);
        total.wrapping_add(tail)
    }

    /// Squared L2 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc_lo = _mm256_setzero_si256();
        let mut acc_hi = _mm256_setzero_si256();
        let mut i = 0;
        while i + LANES_U32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { _mm256_loadu_si256(a.as_ptr().add(i).cast::<__m256i>()) };
            let vb = unsafe { _mm256_loadu_si256(b.as_ptr().add(i).cast::<__m256i>()) };
            let mx = _mm256_max_epu32(va, vb);
            let mn = _mm256_min_epu32(va, vb);
            let d = _mm256_sub_epi32(mx, mn);
            // Square: d * d via mul_epu32 (safe: d fits in u32, d*d fits in u64).
            (acc_lo, acc_hi) = unsafe { mul_accum_u32_pair(d, d, acc_lo, acc_hi) };
            i += LANES_U32;
        }
        // SAFETY: AVX2 enabled.
        let total = unsafe { hadd_u64x4(_mm256_add_epi64(acc_lo, acc_hi)) };
        let tail = scalar::l2_squared_u32(&a[i..], &b[i..]).unwrap_or(0);
        total.wrapping_add(tail)
    }

    /// Cosine similarity of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2 is available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2")]
    #[must_use]
    pub unsafe fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        // Compute dot, norm_a, norm_b in three SIMD passes through the data.
        // Three passes vs. one fused pass: the integer multiply-accumulate
        // pattern is L1-bandwidth-bound for these vector widths, so the
        // simpler three-pass code is competitive and easier to reason about.
        // SAFETY: AVX2 enabled and lengths match.
        let dot = unsafe { dot_u32(a, b) };
        let norm_a = unsafe { dot_u32(a, a) };
        let norm_b = unsafe { dot_u32(b, b) };
        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }
        let denom = ((norm_a as f64) * (norm_b as f64)).sqrt();
        (dot as f64) / denom
    }

    /// Inner product of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2,fma")]
    #[must_use]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + LANES_F32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
            let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += LANES_F32;
        }
        // SAFETY: AVX2 enabled.
        let total = unsafe { hadd_f32x8(acc) };
        let tail = scalar::dot_f32(&a[i..], &b[i..]).unwrap_or(0.0);
        total + tail
    }

    /// Squared L2 distance of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2,fma")]
    #[must_use]
    pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = _mm256_setzero_ps();
        let mut i = 0;
        while i + LANES_F32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { _mm256_loadu_ps(a.as_ptr().add(i)) };
            let vb = unsafe { _mm256_loadu_ps(b.as_ptr().add(i)) };
            let d = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(d, d, acc);
            i += LANES_F32;
        }
        // SAFETY: AVX2 enabled.
        let total = unsafe { hadd_f32x8(acc) };
        let tail = scalar::l2_squared_f32(&a[i..], &b[i..]).unwrap_or(0.0);
        total + tail
    }

    /// Cosine similarity of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure AVX2+FMA are available and `a.len() == b.len()`.
    #[target_feature(enable = "avx2,fma")]
    #[must_use]
    pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: AVX2+FMA enabled, lengths match.
        let dot = unsafe { dot_f32(a, b) };
        let norm_a = unsafe { dot_f32(a, a) };
        let norm_b = unsafe { dot_f32(b, b) };
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b).sqrt()
    }

    // Suppress dead-code warnings for store/load helpers that may become
    // useful as the kernel set grows (e.g. once we add fixed-size
    // specializations in #20).
    #[allow(dead_code)]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn store_u64x4(dst: *mut u64, v: __m256i) {
        // SAFETY: caller responsible for valid 32-byte writable destination.
        unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), v) };
    }
}

/// AArch64 NEON dense-distance kernels.
///
/// Same shape as the AVX2 path: 4-lane integer / 4-lane f32 inner loop, with
/// scalar tail. NEON's natural register width is 128 bits; the kernels could
/// be widened to two-vector unrolling later if needed.
///
/// **Numeric tolerance for SIMD reductions:** integer kernels are bit-exact
/// with [`scalar`]. Floating-point kernels use a different summation order
/// than scalar (4-way then horizontal-add vs. left-to-right), so results can
/// differ by a few ULP on long vectors. The SIMD parity tests assert relative
/// tolerance of `1e-4` for f32.
#[cfg(all(feature = "neon", target_arch = "aarch64"))]
pub mod neon {
    use super::scalar;

    use core::arch::aarch64::{
        vaddvq_f32, vaddvq_u64, vdupq_n_f32, vdupq_n_u32, vfmaq_f32, vget_high_u32, vget_low_u32,
        vld1q_f32, vld1q_u32, vmaxq_u32, vminq_u32, vmlal_u32, vsubq_f32, vsubq_u32,
    };

    const LANES_U32: usize = 4;
    const LANES_F32: usize = 4;

    /// Returns true when NEON is available at runtime.
    ///
    /// NEON is mandatory on AArch64; this is unconditionally true.
    #[must_use]
    pub const fn is_available() -> bool {
        true
    }

    /// Inner product of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available (always true on AArch64) and
    /// `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn dot_u32(a: &[u32], b: &[u32]) -> u64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc64 = core::arch::aarch64::vdupq_n_u64(0);
        let mut acc64_hi = core::arch::aarch64::vdupq_n_u64(0);
        let mut i = 0;
        while i + LANES_U32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { vld1q_u32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_u32(b.as_ptr().add(i)) };
            // vmlal_u32 multiplies two uint32x2_t into uint64x2_t.
            // Process low-2 and high-2 lanes separately.
            acc64 = vmlal_u32(acc64, vget_low_u32(va), vget_low_u32(vb));
            acc64_hi = vmlal_u32(acc64_hi, vget_high_u32(va), vget_high_u32(vb));
            i += LANES_U32;
        }
        // Combine the two u64×2 accumulators and horizontally add.
        let total_vec = core::arch::aarch64::vaddq_u64(acc64, acc64_hi);
        let total = vaddvq_u64(total_vec);
        let tail = scalar::dot_u32(&a[i..], &b[i..]).unwrap_or(0);
        total.wrapping_add(tail)
    }

    /// L1 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn l1_u32(a: &[u32], b: &[u32]) -> u64 {
        debug_assert_eq!(a.len(), b.len());
        let one = vdupq_n_u32(1);
        let mut acc64 = core::arch::aarch64::vdupq_n_u64(0);
        let mut acc64_hi = core::arch::aarch64::vdupq_n_u64(0);
        let mut i = 0;
        while i + LANES_U32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { vld1q_u32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_u32(b.as_ptr().add(i)) };
            // unsigned absolute difference via max - min.
            let mx = vmaxq_u32(va, vb);
            let mn = vminq_u32(va, vb);
            let d = vsubq_u32(mx, mn);
            // Multiply by 1 to widen to u64 lanes.
            acc64 = vmlal_u32(acc64, vget_low_u32(d), vget_low_u32(one));
            acc64_hi = vmlal_u32(acc64_hi, vget_high_u32(d), vget_high_u32(one));
            i += LANES_U32;
        }
        let total_vec = core::arch::aarch64::vaddq_u64(acc64, acc64_hi);
        let total = vaddvq_u64(total_vec);
        let tail = scalar::l1_u32(&a[i..], &b[i..]).unwrap_or(0);
        total.wrapping_add(tail)
    }

    /// Squared L2 distance of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn l2_squared_u32(a: &[u32], b: &[u32]) -> u64 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc64 = core::arch::aarch64::vdupq_n_u64(0);
        let mut acc64_hi = core::arch::aarch64::vdupq_n_u64(0);
        let mut i = 0;
        while i + LANES_U32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { vld1q_u32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_u32(b.as_ptr().add(i)) };
            let mx = vmaxq_u32(va, vb);
            let mn = vminq_u32(va, vb);
            let d = vsubq_u32(mx, mn);
            acc64 = vmlal_u32(acc64, vget_low_u32(d), vget_low_u32(d));
            acc64_hi = vmlal_u32(acc64_hi, vget_high_u32(d), vget_high_u32(d));
            i += LANES_U32;
        }
        let total_vec = core::arch::aarch64::vaddq_u64(acc64, acc64_hi);
        let total = vaddvq_u64(total_vec);
        let tail = scalar::l2_squared_u32(&a[i..], &b[i..]).unwrap_or(0);
        total.wrapping_add(tail)
    }

    /// Cosine similarity of two `u32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: NEON enabled, lengths match.
        let dot = unsafe { dot_u32(a, b) };
        let norm_a = unsafe { dot_u32(a, a) };
        let norm_b = unsafe { dot_u32(b, b) };
        if norm_a == 0 || norm_b == 0 {
            return 0.0;
        }
        let denom = ((norm_a as f64) * (norm_b as f64)).sqrt();
        (dot as f64) / denom
    }

    /// Inner product of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + LANES_F32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { vld1q_f32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_f32(b.as_ptr().add(i)) };
            acc = vfmaq_f32(acc, va, vb);
            i += LANES_F32;
        }
        let total = vaddvq_f32(acc);
        let tail = scalar::dot_f32(&a[i..], &b[i..]).unwrap_or(0.0);
        total + tail
    }

    /// Squared L2 distance of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn l2_squared_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0;
        while i + LANES_F32 <= a.len() {
            // SAFETY: bounds checked.
            let va = unsafe { vld1q_f32(a.as_ptr().add(i)) };
            let vb = unsafe { vld1q_f32(b.as_ptr().add(i)) };
            let d = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, d, d);
            i += LANES_F32;
        }
        let total = vaddvq_f32(acc);
        let tail = scalar::l2_squared_f32(&a[i..], &b[i..]).unwrap_or(0.0);
        total + tail
    }

    /// Cosine similarity of two `f32` vectors.
    ///
    /// # Safety
    ///
    /// Caller must ensure NEON is available and `a.len() == b.len()`.
    #[target_feature(enable = "neon")]
    #[must_use]
    pub unsafe fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        // SAFETY: NEON enabled, lengths match.
        let dot = unsafe { dot_f32(a, b) };
        let norm_a = unsafe { dot_f32(a, a) };
        let norm_b = unsafe { dot_f32(b, b) };
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b).sqrt()
    }
}

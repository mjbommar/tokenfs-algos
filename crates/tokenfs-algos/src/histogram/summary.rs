//! Streaming moment summaries over byte values.
//!
//! Computes mean, variance, skewness, and kurtosis of byte VALUES (each
//! byte interpreted as an integer in `[0, 255]`) in a single pass using
//! Pébay's numerically-stable update formulas for arbitrary-order central
//! moments. Reference: Philippe Pébay, *Formulas for Robust, One-Pass
//! Parallel Computation of Covariances and Arbitrary-Order Statistical
//! Moments*, Sandia Report SAND2008-6212, 2008. The order-2 update
//! reduces to Welford 1962 / Knuth TAOCP vol. 2.
//!
//! ## State footprint
//!
//! `ByteValueMoments` is a fixed-size POD: one `u64` plus four `f64`s, so
//! 40 bytes plus padding (typically 40 with no padding). The accumulator
//! state used during the single pass is also fixed-size (`n`, `mean`,
//! `m2`, `m3`, `m4` — five scalars), `O(1)` regardless of input length.
//!
//! ## Accuracy
//!
//! - **Mean and variance:** the Welford update keeps relative error at the
//!   `f64` round-off level (≈ 1e-16) for any sequence of bounded byte
//!   values, even at `N = 10^9`; bounded inputs with bounded mean cannot
//!   trigger the catastrophic cancellation that plagues the textbook
//!   `E[x^2] - E[x]^2` formulation.
//! - **Skewness and kurtosis:** Pébay's third- and fourth-moment updates
//!   are also Welford-form (delta / delta_n based) and stay numerically
//!   stable. With byte inputs in `[0, 255]` and sample sizes up to ~10^9
//!   the standardized statistics are accurate to roughly 1e-12 relative
//!   error.
//! - The reported variance is the **unbiased** sample variance (`m2 /
//!   (n - 1)`); skewness is the **standardized** third central moment
//!   (`g_1 = m_3 / m_2^{3/2} * sqrt(n)`), and kurtosis is the **excess**
//!   sample kurtosis (`g_2 = n * m_4 / m_2^2 - 3`). These match the
//!   conventional definitions used by NumPy `scipy.stats.skew` /
//!   `scipy.stats.kurtosis` (with `bias=False`-equivalent normalization on
//!   the variance and `fisher=True` excess form on the kurtosis).
//!
//! Degenerate cases:
//! - `n == 0`: every field is `0.0` (mean by convention; the higher
//!   moments are mathematically undefined).
//! - `n == 1`: variance, skewness, and kurtosis are returned as `0.0`
//!   (variance is `NaN` under the unbiased convention; we return `0.0`
//!   instead so callers can scan large file sets without filtering).
//! - constant input (variance zero with `n >= 2`): skewness and
//!   kurtosis are returned as `0.0` rather than `NaN`. `f64::NaN` would
//!   propagate through downstream feature vectors and break standard
//!   distance metrics; `0.0` matches the "perfectly symmetric / zero
//!   excess" interpretation that callers actually want for constant
//!   blocks.

use crate::math;

/// Streaming moment summary over the byte values of an input.
///
/// All fields are computed in a single pass via the Welford-Pébay update
/// formulas. See the [module-level docs](self) for accuracy and degenerate
/// case behavior.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ByteValueMoments {
    /// Number of bytes summarized.
    pub n: u64,
    /// Sample mean of the byte values.
    pub mean: f64,
    /// Unbiased sample variance (`m_2 / (n - 1)`).
    pub variance: f64,
    /// Standardized sample skewness (`g_1`).
    pub skewness: f64,
    /// Excess sample kurtosis (`g_2 = n * m_4 / m_2^2 - 3`).
    pub kurtosis: f64,
}

impl ByteValueMoments {
    /// All-zero summary — equivalent to summarizing an empty input.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }

    /// Memory footprint of the summary in bytes.
    #[must_use]
    pub const fn memory_bytes() -> usize {
        core::mem::size_of::<Self>()
    }
}

impl Default for ByteValueMoments {
    fn default() -> Self {
        Self::empty()
    }
}

/// Single-pass mean / variance / skewness / kurtosis of `bytes`
/// interpreted as integers in `[0, 255]`.
///
/// Implementation is the canonical Welford-Pébay update for the four
/// running central moments. See [`ByteValueMoments`] and the
/// [module-level docs](self) for accuracy guarantees and behavior on
/// degenerate inputs (empty, constant, single byte).
#[must_use]
pub fn byte_value_moments(bytes: &[u8]) -> ByteValueMoments {
    if bytes.is_empty() {
        return ByteValueMoments::empty();
    }

    // Welford-Pébay accumulators for n, mean, m2, m3, m4.
    let mut n = 0_u64;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    let mut m3 = 0.0_f64;
    let mut m4 = 0.0_f64;

    for &byte in bytes {
        // n_old, n_new, and the standard delta / delta_n decomposition
        // from Pébay 2008 eq. 2.1 (single-observation update).
        let n_old = n;
        n += 1;
        let n_new_f = n as f64;

        let x = f64::from(byte);
        let delta = x - mean;
        let delta_n = delta / n_new_f;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n_old as f64);

        // Order matters: m4 must be updated before m3, which must be
        // updated before m2, because each formula references the
        // pre-update value of the lower-order accumulator.
        m4 += term1 * delta_n2 * (n_new_f * n_new_f - 3.0 * n_new_f + 3.0) + 6.0 * delta_n2 * m2
            - 4.0 * delta_n * m3;
        m3 += term1 * delta_n * (n_new_f - 2.0) - 3.0 * delta_n * m2;
        m2 += term1;
        mean += delta_n;
    }

    finalize(n, mean, m2, m3, m4)
}

fn finalize(n: u64, mean: f64, m2: f64, m3: f64, m4: f64) -> ByteValueMoments {
    if n == 0 {
        return ByteValueMoments::empty();
    }
    if n == 1 {
        // With one observation variance is undefined; report zero so
        // callers can emit feature vectors without NaN-filtering.
        return ByteValueMoments {
            n,
            mean,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        };
    }

    let n_f = n as f64;
    let variance = m2 / (n_f - 1.0);

    // Constant input: m2 == 0 → skew/kurt mathematically undefined.
    // Report 0.0 (see module docs for rationale).
    let (skewness, kurtosis) = if m2 <= 0.0 {
        (0.0, 0.0)
    } else {
        let m2_pow_3_2 = m2 * math::sqrt_f64(m2);
        let g1 = math::sqrt_f64(n_f) * m3 / m2_pow_3_2;
        let g2 = n_f * m4 / (m2 * m2) - 3.0;
        (g1, g2)
    };

    ByteValueMoments {
        n,
        mean,
        variance,
        skewness,
        kurtosis,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]

    use super::*;
    // `Vec` and `vec!` are not in the no-std prelude; alias them from
    // `alloc` for the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    fn xorshift_byte_stream(seed: u64, n: usize) -> Vec<u8> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push((state & 0xff) as u8);
        }
        out
    }

    #[test]
    fn empty_input_is_zero() {
        let m = byte_value_moments(&[]);
        assert_eq!(m.n, 0);
        assert_eq!(m.mean, 0.0);
        assert_eq!(m.variance, 0.0);
        assert_eq!(m.skewness, 0.0);
        assert_eq!(m.kurtosis, 0.0);
    }

    #[test]
    fn constant_input_zero_variance_zero_higher_moments() {
        // 1024 copies of 0x42; variance must be exactly 0; skew/kurt 0
        // (per documented degenerate-case convention).
        let bytes = vec![0x42_u8; 1024];
        let m = byte_value_moments(&bytes);
        assert_eq!(m.n, 1024);
        assert_eq!(m.mean, 66.0);
        assert_eq!(m.variance, 0.0);
        assert_eq!(m.skewness, 0.0);
        assert_eq!(m.kurtosis, 0.0);
    }

    #[test]
    fn pure_zeros_are_zero() {
        let bytes = vec![0_u8; 4096];
        let m = byte_value_moments(&bytes);
        assert_eq!(m.mean, 0.0);
        assert_eq!(m.variance, 0.0);
    }

    #[test]
    fn single_byte_returns_value_and_zero_higher_moments() {
        let m = byte_value_moments(&[200]);
        assert_eq!(m.n, 1);
        assert_eq!(m.mean, 200.0);
        assert_eq!(m.variance, 0.0);
        assert_eq!(m.skewness, 0.0);
        assert_eq!(m.kurtosis, 0.0);
    }

    #[test]
    fn uniform_random_matches_theoretical_moments() {
        // Discrete uniform on {0..=255}: mean = 127.5, variance =
        // (256^2 - 1) / 12 = 5461.25, skewness = 0, excess kurtosis
        // = -1.2. Tolerance ±5% per the spec.
        let bytes = xorshift_byte_stream(0xC8C2_5E0F_2C5C_3F6D, 200_000);
        let m = byte_value_moments(&bytes);
        assert!(
            (m.mean - 127.5).abs() / 127.5 < 0.05,
            "mean off: {}",
            m.mean
        );
        assert!(
            (m.variance - 5461.25).abs() / 5461.25 < 0.05,
            "var off: {}",
            m.variance
        );
        // Skewness scales like the absolute moment, so use absolute slack.
        assert!(m.skewness.abs() < 0.1, "skew off: {}", m.skewness);
        assert!(
            (m.kurtosis - (-1.2)).abs() < 0.06,
            "ex-kurt off: {}",
            m.kurtosis
        );
    }

    #[test]
    fn ascii_text_has_text_like_moments() {
        // Mostly-ASCII Lorem ipsum: mean lands in 70-100 (printable
        // ASCII band), positive skew (rare punctuation pulls left of the
        // mean less than spaces+letters pull right), positive kurtosis
        // (peaked near letter band, fat tails into low-byte whitespace).
        const FILLER: &[u8] = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
            Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
            Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris \
            nisi ut aliquip ex ea commodo consequat. ";
        let mut bytes = Vec::with_capacity(8192);
        while bytes.len() < 8192 {
            let take = (8192 - bytes.len()).min(FILLER.len());
            bytes.extend_from_slice(&FILLER[..take]);
        }
        let m = byte_value_moments(&bytes);
        assert!(
            (70.0..=100.0).contains(&m.mean),
            "ASCII mean out of band: {}",
            m.mean
        );
        assert!(m.variance > 0.0);
        // Whitespace characters (space=0x20, newline=0x0a) sit well
        // below the alphabetic band (0x61-0x7a), pulling the
        // distribution's left tail and producing positive excess
        // kurtosis on top of mild left-skew. We assert kurt > 0 only;
        // skew sign depends on filler ratio so we check |skew| > 0.
        assert!(
            m.kurtosis > 0.0,
            "ASCII text kurt should be > 0: {}",
            m.kurtosis
        );
        assert!(m.skewness.abs() > 0.0);
    }

    #[test]
    fn stable_on_one_million_bytes() {
        let bytes = xorshift_byte_stream(0xDEAD_BEEF_CAFE_F00D, 1_000_000);
        let m = byte_value_moments(&bytes);
        // Same theoretical bounds as uniform_random_matches_theoretical_moments,
        // tighter tolerance because N is 5x larger.
        assert!((m.mean - 127.5).abs() / 127.5 < 0.02);
        assert!((m.variance - 5461.25).abs() / 5461.25 < 0.02);
        assert!(m.skewness.abs() < 0.05);
        assert!((m.kurtosis - (-1.2)).abs() < 0.03);
    }

    #[test]
    fn variance_matches_two_pass_on_small_input() {
        // Cross-check the streaming variance against the textbook
        // two-pass formula, which is numerically-safe for small inputs.
        let bytes: Vec<u8> = (0..=255).collect();
        let m = byte_value_moments(&bytes);

        let n = bytes.len() as f64;
        let mean = bytes.iter().map(|&b| f64::from(b)).sum::<f64>() / n;
        let two_pass_var = bytes
            .iter()
            .map(|&b| {
                let d = f64::from(b) - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);

        assert!((m.mean - mean).abs() < 1e-12);
        assert!(
            (m.variance - two_pass_var).abs() / two_pass_var < 1e-12,
            "stream var {} vs two-pass {}",
            m.variance,
            two_pass_var
        );
    }

    #[test]
    fn memory_footprint_is_documented() {
        // 1 u64 + 4 f64 = 40 bytes; the struct may add padding but
        // remains well under our 64-byte budget.
        assert!(ByteValueMoments::memory_bytes() <= 64);
    }
}

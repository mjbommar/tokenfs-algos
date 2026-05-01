//! Online streaming P² (P-squared) quantile estimator.
//!
//! Implements Jain & Chlamtac, "The P² Algorithm for Dynamic Calculation
//! of Quantiles and Histograms Without Storing Observations", CACM 1985.
//!
//! P² is a fixed-state, single-pass estimator that approximates an
//! arbitrary quantile of a stream of `f64` observations. It maintains
//! exactly 5 markers (positions 0, p/2, p, (1+p)/2, 1.0 of the empirical
//! distribution) and updates them on every observation via a parabolic or
//! linear interpolation. Memory is `O(1)` — five `f64`s and five `i64`s
//! plus the target quantile and the count, ~120 bytes total. No heap
//! allocation. No re-scan of past observations.
//!
//! ## Bias and accuracy
//!
//! The estimator is biased on small sample counts (N < 5: degenerate
//! special case; N < 100: error bands wide). Documented practical
//! accuracy from the paper and follow-up surveys: ~1-3% relative error
//! at p=0.5 once N ≥ 1000 on smoothly-distributed inputs; tail quantiles
//! (p ≥ 0.95 or p ≤ 0.05) need N ≥ 10000 for similar accuracy.
//!
//! Use [`Estimator::estimate`] for the running estimate; the value is
//! defined for any `count ≥ 1`. For `count < 5` the estimate is a sorted
//! sample (effectively exact); for `count ≥ 5` it follows the P² update
//! rules.

use crate::math;

/// Online P² quantile estimator for one fixed quantile `p ∈ (0, 1)`.
#[derive(Clone, Debug)]
pub struct Estimator {
    /// Target quantile in `(0, 1)`.
    p: f64,
    /// Observation count seen so far.
    count: u64,
    /// Marker heights (estimated values at positions 0, p/2, p, (1+p)/2, 1).
    /// The five entries are `q[0] = min`, `q[1] = ~p/2 quantile`,
    /// `q[2] = ~p quantile (the answer)`, `q[3] = ~(1+p)/2 quantile`,
    /// `q[4] = max`.
    q: [f64; 5],
    /// Marker positions (1-indexed, integer rank in the sample).
    n: [i64; 5],
    /// Desired marker positions (real-valued, advance on each observation).
    np: [f64; 5],
    /// Increment per observation for each `np[i]`.
    dn: [f64; 5],
}

impl Estimator {
    /// Builds an estimator for quantile `p ∈ (0, 1)`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not in `(0, 1)` or is NaN.
    #[must_use]
    pub fn new(p: f64) -> Self {
        assert!(
            p > 0.0 && p < 1.0,
            "P² target quantile must be in (0, 1), got {p}"
        );
        Self {
            p,
            count: 0,
            q: [0.0; 5],
            n: [0; 5],
            np: [0.0; 5],
            // Per Jain-Chlamtac: marker desired-position increments.
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
        }
    }

    /// Target quantile.
    #[must_use]
    pub const fn p(&self) -> f64 {
        self.p
    }

    /// Number of observations seen.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Memory footprint in bytes.
    #[must_use]
    pub const fn memory_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Records one observation.
    pub fn update(&mut self, value: f64) {
        if value.is_nan() {
            return; // ignore NaN per the original P² spec
        }
        self.count += 1;

        // Initialization: collect the first 5 observations sorted.
        if self.count <= 5 {
            self.q[(self.count - 1) as usize] = value;
            if self.count == 5 {
                // Sort the initial 5 markers ascending and seed n / np.
                // Manual insertion sort keeps this no_std-clean (slice
                // sort_by is alloc/std-only).
                insertion_sort_5(&mut self.q);
                for (i, ni) in self.n.iter_mut().enumerate() {
                    *ni = (i as i64) + 1;
                }
                self.np[0] = 1.0;
                self.np[1] = 1.0 + 2.0 * self.p;
                self.np[2] = 1.0 + 4.0 * self.p;
                self.np[3] = 3.0 + 2.0 * self.p;
                self.np[4] = 5.0;
            }
            return;
        }

        // Find the cell `k` such that q[k] ≤ value < q[k+1], adjusting the
        // extreme markers if `value` falls outside the current range.
        let k = if value < self.q[0] {
            self.q[0] = value;
            0
        } else if value < self.q[1] {
            0
        } else if value < self.q[2] {
            1
        } else if value < self.q[3] {
            2
        } else if value <= self.q[4] {
            3
        } else {
            self.q[4] = value;
            3
        };

        // Increment markers' actual positions for cells ≥ k+1.
        for i in (k + 1)..5 {
            self.n[i] += 1;
        }
        // Increment all markers' desired positions.
        for i in 0..5 {
            self.np[i] += self.dn[i];
        }

        // Adjust heights for the three middle markers.
        for i in 1..4 {
            let d = self.np[i] - self.n[i] as f64;
            let n_minus_1 = self.n[i - 1] as f64;
            let n_i = self.n[i] as f64;
            let n_plus_1 = self.n[i + 1] as f64;

            if (d >= 1.0 && self.n[i + 1] - self.n[i] > 1)
                || (d <= -1.0 && self.n[i - 1] - self.n[i] < -1)
            {
                let dsign = d.signum();
                // P² parabolic prediction.
                let parabolic = self.q[i]
                    + (dsign / (n_plus_1 - n_minus_1))
                        * ((n_i - n_minus_1 + dsign) * (self.q[i + 1] - self.q[i])
                            / (n_plus_1 - n_i)
                            + (n_plus_1 - n_i - dsign) * (self.q[i] - self.q[i - 1])
                                / (n_i - n_minus_1));
                // If parabolic prediction violates monotonicity, fall back
                // to linear prediction.
                let new_q = if self.q[i - 1] < parabolic && parabolic < self.q[i + 1] {
                    parabolic
                } else if dsign > 0.0 {
                    self.q[i] + (self.q[i + 1] - self.q[i]) / (n_plus_1 - n_i)
                } else {
                    self.q[i] - (self.q[i - 1] - self.q[i]) / (n_minus_1 - n_i)
                };
                self.q[i] = new_q;
                self.n[i] += dsign as i64;
            }
        }
    }

    /// Records `value` `count` times.
    pub fn update_n(&mut self, value: f64, count: u64) {
        for _ in 0..count {
            self.update(value);
        }
    }

    /// Returns the current quantile estimate.
    ///
    /// For `count < 5`, returns a sorted-sample estimate (effectively
    /// exact for the empirical distribution seen so far). For `count ≥ 5`
    /// returns the P² marker-2 height.
    #[must_use]
    pub fn estimate(&self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        if self.count < 5 {
            // Use a sorted copy of the partially-filled `q` for the
            // exact estimate.
            let n = self.count as usize;
            let mut buf = [0.0_f64; 5];
            buf[..n].copy_from_slice(&self.q[..n]);
            // Manual insertion sort over at most 5 entries: avoids the
            // alloc/std-only slice `sort_by`.
            insertion_sort_partial(&mut buf, n);
            // Linear interpolation at the target quantile within the
            // sample's empirical CDF: position = p * (n - 1).
            let pos = self.p * ((n - 1) as f64);
            let lo = math::round_f32(pos as f32 - 0.5) as usize;
            let lo = lo.min(n - 1);
            let hi = (lo + 1).min(n - 1);
            let frac = pos - lo as f64;
            buf[lo] + frac * (buf[hi] - buf[lo])
        } else {
            self.q[2]
        }
    }

    /// Resets the estimator to its initial state. Keeps the target `p`.
    pub fn clear(&mut self) {
        self.count = 0;
        self.q = [0.0; 5];
        self.n = [0; 5];
        self.np = [0.0; 5];
    }
}

/// Ascending insertion sort over a 5-element `f64` array. NaN-safe via
/// `partial_cmp` fallback to `Equal`.
#[inline]
fn insertion_sort_5(a: &mut [f64; 5]) {
    insertion_sort_partial(a.as_mut_slice(), 5);
}

/// Ascending insertion sort over the first `n` entries of `a`.
#[inline]
fn insertion_sort_partial(a: &mut [f64], n: usize) {
    let mut i = 1;
    while i < n {
        let key = a[i];
        let mut j = i;
        while j > 0
            && a[j - 1]
                .partial_cmp(&key)
                .unwrap_or(core::cmp::Ordering::Equal)
                == core::cmp::Ordering::Greater
        {
            a[j] = a[j - 1];
            j -= 1;
        }
        a[j] = key;
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn empty_estimator_returns_nan() {
        let est = Estimator::new(0.5);
        assert!(est.estimate().is_nan());
    }

    #[test]
    fn small_sample_estimate_matches_sorted_position() {
        // For count ≤ 5 we use exact sample interpolation.
        let mut est = Estimator::new(0.5);
        for v in [3.0, 1.0, 4.0, 1.0, 5.0] {
            est.update(v);
        }
        // sorted: [1, 1, 3, 4, 5]; median = 3.0
        let m = est.estimate();
        assert!((m - 3.0).abs() < 1e-9, "got {m}");
    }

    #[test]
    fn uniform_distribution_median_within_3pct() {
        let mut est = Estimator::new(0.5);
        // Deterministic xorshift PRNG over [0, 1).
        let mut state: u64 = 0xC8C2_5E0F_2C5C_3F6D;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = (state as f64) / (u64::MAX as f64);
            est.update(v);
        }
        let m = est.estimate();
        assert!((m - 0.5).abs() < 0.03, "median est={m}, expected ~0.5");
    }

    #[test]
    fn uniform_distribution_p95_within_5pct() {
        let mut est = Estimator::new(0.95);
        let mut state: u64 = 0xDEAD_BEEF_CAFE_F00D;
        for _ in 0..50_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let v = (state as f64) / (u64::MAX as f64);
            est.update(v);
        }
        let m = est.estimate();
        assert!((m - 0.95).abs() < 0.05, "p95 est={m}, expected ~0.95");
    }

    #[test]
    fn constant_input_estimate_equals_constant() {
        let mut est = Estimator::new(0.5);
        for _ in 0..1000 {
            est.update(42.0);
        }
        let m = est.estimate();
        assert!((m - 42.0).abs() < 1e-9, "got {m}");
    }

    #[test]
    fn nan_observations_are_ignored() {
        let mut est = Estimator::new(0.5);
        for _ in 0..10 {
            est.update(f64::NAN);
        }
        assert_eq!(est.count(), 0);
        assert!(est.estimate().is_nan());
    }

    #[test]
    fn clear_resets_state() {
        let mut est = Estimator::new(0.5);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            est.update(v);
        }
        est.clear();
        assert_eq!(est.count(), 0);
        assert!(est.estimate().is_nan());
    }

    #[test]
    fn update_n_matches_repeated_update() {
        let mut a = Estimator::new(0.5);
        let mut b = Estimator::new(0.5);
        a.update_n(7.0, 100);
        for _ in 0..100 {
            b.update(7.0);
        }
        assert!((a.estimate() - b.estimate()).abs() < 1e-9);
    }

    #[test]
    fn memory_footprint_is_bounded() {
        // Sanity bound: 5 f64 + 5 i64 + 5 f64 + 5 f64 + p + count + padding
        // = 5*8 + 5*8 + 5*8 + 5*8 + 8 + 8 = 176 bytes max
        assert!(Estimator::memory_bytes() <= 256);
    }

    #[test]
    fn invalid_p_panics() {
        let result = std::panic::catch_unwind(|| Estimator::new(0.0));
        assert!(result.is_err());
        let result = std::panic::catch_unwind(|| Estimator::new(1.0));
        assert!(result.is_err());
        let result = std::panic::catch_unwind(|| Estimator::new(f64::NAN));
        assert!(result.is_err());
    }
}

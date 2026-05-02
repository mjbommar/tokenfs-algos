//! Joint entropy estimators over adjacent byte pairs.

use crate::{
    histogram::{BytePairHistogram, BytePairScratch},
    math,
};

/// Strategy recommendation for adjacent-pair entropy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PairEntropyStrategy {
    /// Input is too short to contain adjacent pairs.
    None,
    /// Use a caller-owned scratch table with lazy reset.
    ReusedScratchExact,
    /// Use a fresh dense 65,536-bin table.
    DenseExact,
    /// Use a hash-bin sketch instead of exact dense H2.
    HashSketch,
}

/// Planner input for adjacent-pair entropy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PairEntropyShape {
    /// Input length in bytes.
    pub bytes: usize,
    /// Whether the caller can provide reusable scratch state.
    pub caller_scratch_available: bool,
    /// Whether exact H2 is required for calibration/reproducibility.
    pub exact_required: bool,
}

impl PairEntropyShape {
    /// Creates a pair-entropy planning shape.
    #[must_use]
    pub const fn new(bytes: usize) -> Self {
        Self {
            bytes,
            caller_scratch_available: false,
            exact_required: false,
        }
    }

    /// Marks reusable scratch availability.
    #[must_use]
    pub const fn with_scratch(mut self, available: bool) -> Self {
        self.caller_scratch_available = available;
        self
    }

    /// Marks whether exact H2 is required.
    #[must_use]
    pub const fn require_exact(mut self, exact: bool) -> Self {
        self.exact_required = exact;
        self
    }
}

/// Computes exact adjacent-pair joint entropy `H(X_i, X_{i+1})`.
///
/// This uses a dense 65,536-bin byte-pair histogram and does not allocate.
#[must_use]
pub fn h2_pairs(bytes: &[u8]) -> f32 {
    let histogram = BytePairHistogram::from_bytes(bytes);
    h2_from_pair_histogram(&histogram)
}

/// Computes exact adjacent-pair joint entropy using reusable scratch state.
#[must_use]
pub fn h2_pairs_with_scratch(bytes: &[u8], scratch: &mut BytePairScratch) -> f32 {
    scratch.reset_and_add_bytes(bytes);
    h2_from_pair_scratch(scratch)
}

/// Computes joint entropy from a dense byte-pair histogram.
#[must_use]
pub fn h2_from_pair_histogram(histogram: &BytePairHistogram) -> f32 {
    entropy_counts_u32(histogram.counts(), histogram.observations())
}

/// Computes joint entropy from reusable pair scratch state.
#[must_use]
pub fn h2_from_pair_scratch(scratch: &BytePairScratch) -> f32 {
    entropy_nonzero_counts(
        scratch.iter_nonzero().map(|(_, _, count)| count),
        scratch.observations(),
    )
}

/// Plans an adjacent-pair entropy strategy for hot callers.
///
/// Exact calibration paths should set `exact_required`; production planners can
/// use the hash-sketch recommendation for small or one-shot calls where a fresh
/// dense table would dominate useful work.
#[must_use]
pub const fn plan_pair_entropy(shape: PairEntropyShape) -> PairEntropyStrategy {
    if shape.bytes < 2 {
        PairEntropyStrategy::None
    } else if shape.caller_scratch_available {
        PairEntropyStrategy::ReusedScratchExact
    } else if shape.exact_required {
        PairEntropyStrategy::DenseExact
    } else if shape.bytes < 16 * 1024 {
        PairEntropyStrategy::HashSketch
    } else {
        PairEntropyStrategy::DenseExact
    }
}

pub(crate) fn entropy_counts_u32(counts: &[u32], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }

    let total = total as f64;
    let mut entropy = 0.0_f64;
    for &count in counts {
        if count != 0 {
            let p = f64::from(count) / total;
            entropy -= p * math::log2_f64(p);
        }
    }
    entropy as f32
}

pub(crate) fn entropy_nonzero_counts<I>(counts: I, total: u64) -> f32
where
    I: IntoIterator<Item = u32>,
{
    if total == 0 {
        return 0.0;
    }

    let total = total as f64;
    let mut entropy = 0.0_f64;
    for count in counts {
        if count != 0 {
            let p = f64::from(count) / total;
            entropy -= p * math::log2_f64(p);
        }
    }
    entropy as f32
}

#[cfg(test)]
mod tests {
    use super::{
        PairEntropyShape, PairEntropyStrategy, h2_pairs, h2_pairs_with_scratch, plan_pair_entropy,
    };
    use crate::histogram::BytePairScratch;
    // `Box` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;

    #[test]
    fn repeated_pair_joint_entropy_is_zero() {
        assert_eq!(h2_pairs(b"aaaaaaaa"), 0.0);
    }

    #[test]
    fn alternating_pairs_have_one_bit_joint_entropy() {
        let entropy = h2_pairs(b"abababa");
        assert!((entropy - 1.0).abs() < 1e-6, "h2_pairs={entropy}");
    }

    #[test]
    fn scratch_h2_matches_dense_h2() {
        let mut scratch = Box::new(BytePairScratch::new());
        let bytes = b"abacabadabacaba";
        assert_eq!(h2_pairs(bytes), h2_pairs_with_scratch(bytes, &mut scratch));
    }

    #[test]
    fn pair_entropy_policy_prefers_scratch_when_available() {
        assert_eq!(
            plan_pair_entropy(PairEntropyShape::new(1)),
            PairEntropyStrategy::None
        );
        assert_eq!(
            plan_pair_entropy(PairEntropyShape::new(4096)),
            PairEntropyStrategy::HashSketch
        );
        assert_eq!(
            plan_pair_entropy(PairEntropyShape::new(4096).with_scratch(true)),
            PairEntropyStrategy::ReusedScratchExact
        );
        assert_eq!(
            plan_pair_entropy(PairEntropyShape::new(4096).require_exact(true)),
            PairEntropyStrategy::DenseExact
        );
    }
}

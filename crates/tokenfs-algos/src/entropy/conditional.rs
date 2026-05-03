//! Conditional entropy estimators over adjacent byte pairs.

use crate::{
    entropy::joint,
    histogram::{BytePairHistogram, BytePairScratch},
};

/// Computes exact conditional entropy `H(X_{i+1} | X_i)`.
///
/// This uses a dense byte-pair histogram and a 256-bin predecessor histogram.
/// It does not allocate.
///
/// **Stack footprint**: builds [`BytePairHistogram`] (~256 KiB) on the
/// call frame. Available only with `feature = "userspace"` (audit-R9 #5
/// kernel-stack hazard). Kernel-adjacent callers should use
/// [`h_next_given_prev_with_scratch`] (lazy-clear scratch path).
#[cfg(feature = "userspace")]
#[must_use]
pub fn h_next_given_prev(bytes: &[u8]) -> f32 {
    if bytes.len() < 2 {
        return 0.0;
    }

    let mut pairs = BytePairHistogram::new();
    let mut predecessors = [0_u32; 256];
    for pair in bytes.windows(2) {
        pairs.add_pair(pair[0], pair[1]);
        predecessors[pair[0] as usize] += 1;
    }

    h_next_given_prev_from_counts(&pairs, &predecessors)
}

/// Computes exact conditional entropy using reusable pair scratch state.
#[must_use]
pub fn h_next_given_prev_with_scratch(bytes: &[u8], scratch: &mut BytePairScratch) -> f32 {
    scratch.reset_and_add_bytes(bytes);
    h_next_given_prev_from_scratch(scratch)
}

/// Computes `H(X_{i+1} | X_i)` from pair and predecessor counts.
#[must_use]
pub fn h_next_given_prev_from_counts(
    pairs: &BytePairHistogram,
    predecessor_counts: &[u32; 256],
) -> f32 {
    if pairs.observations() == 0 {
        return 0.0;
    }

    let joint = joint::h2_from_pair_histogram(pairs);
    let predecessor = joint::entropy_counts_u32(predecessor_counts, pairs.observations());
    (joint - predecessor).max(0.0)
}

/// Computes `H(X_{i+1} | X_i)` from reusable pair scratch state.
#[must_use]
pub fn h_next_given_prev_from_scratch(scratch: &BytePairScratch) -> f32 {
    if scratch.observations() == 0 {
        return 0.0;
    }

    let joint = joint::h2_from_pair_scratch(scratch);
    let predecessor = joint::entropy_nonzero_counts(
        scratch.iter_predecessors().map(|(_, count)| count),
        scratch.observations(),
    );
    (joint - predecessor).max(0.0)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "userspace")]
    use super::h_next_given_prev;
    use super::h_next_given_prev_with_scratch;
    use crate::histogram::BytePairScratch;
    // `Box` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;

    #[cfg(feature = "userspace")]
    #[test]
    fn deterministic_next_byte_has_zero_conditional_entropy() {
        assert_eq!(h_next_given_prev(b"abababab"), 0.0);
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn mixed_next_byte_has_positive_conditional_entropy() {
        assert!(h_next_given_prev(b"abacabad") > 0.0);
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn scratch_conditional_entropy_matches_dense_path() {
        let bytes = b"abacabadabacaba";
        let mut scratch = Box::new(BytePairScratch::new());
        assert_eq!(
            h_next_given_prev(bytes),
            h_next_given_prev_with_scratch(bytes, &mut scratch)
        );
    }
}

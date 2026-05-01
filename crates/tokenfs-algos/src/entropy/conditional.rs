//! Conditional entropy estimators over adjacent byte pairs.

use crate::{entropy::joint, histogram::BytePairHistogram};

/// Computes exact conditional entropy `H(X_{i+1} | X_i)`.
///
/// This uses a dense byte-pair histogram and a 256-bin predecessor histogram.
/// It does not allocate.
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

#[cfg(test)]
mod tests {
    use super::h_next_given_prev;

    #[test]
    fn deterministic_next_byte_has_zero_conditional_entropy() {
        assert_eq!(h_next_given_prev(b"abababab"), 0.0);
    }

    #[test]
    fn mixed_next_byte_has_positive_conditional_entropy() {
        assert!(h_next_given_prev(b"abacabad") > 0.0);
    }
}

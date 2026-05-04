//! Neighbor-selection heuristics for HNSW INSERT (Algorithms 3 + 4
//! from the Malkov & Yashunin paper).
//!
//! Phase 4.1 ships [`select_neighbors_simple`] (Algorithm 3): "the M
//! nearest candidates win." Phase 4.5 (later) lands
//! `select_neighbors_heuristic` (Algorithm 4) — the diversity-
//! preserving variant that keeps long-range edges connecting
//! clusters.
//!
//! Both signatures take a sorted-ascending candidate list (smallest
//! distance first) and return up to `M` selected candidates. The
//! input is consumed; output ordering matches input ordering.

#![cfg(feature = "std")]

use super::candidates::Candidate;

/// Algorithm 3 — SELECT-NEIGHBORS-SIMPLE.
///
/// Returns the `m` candidates with smallest distance to the query.
/// `candidates` MUST be sorted ascending by `(distance, NodeId)`
/// (deterministic tie-break per `DETERMINISM.md`).
///
/// Complexity: O(min(m, candidates.len())). Truncates the input.
pub fn select_neighbors_simple(candidates: Vec<Candidate>, m: usize) -> Vec<Candidate> {
    let mut out = candidates;
    if out.len() > m {
        out.truncate(m);
    }
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn select_simple_returns_first_m() {
        let cands = vec![
            Candidate::new(10, 0),
            Candidate::new(20, 1),
            Candidate::new(30, 2),
            Candidate::new(40, 3),
            Candidate::new(50, 4),
        ];
        let r = select_neighbors_simple(cands, 3);
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].distance, 10);
        assert_eq!(r[1].distance, 20);
        assert_eq!(r[2].distance, 30);
    }

    #[test]
    fn select_simple_returns_all_when_fewer_than_m() {
        let cands = vec![Candidate::new(10, 0), Candidate::new(20, 1)];
        let r = select_neighbors_simple(cands, 5);
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn select_simple_returns_empty_for_zero_input() {
        let r = select_neighbors_simple(Vec::<Candidate>::new(), 5);
        assert!(r.is_empty());
    }

    #[test]
    fn select_simple_zero_m_returns_empty() {
        let cands = vec![Candidate::new(10, 0), Candidate::new(20, 1)];
        let r = select_neighbors_simple(cands, 0);
        assert!(r.is_empty());
    }

    #[test]
    fn select_simple_preserves_input_order() {
        // Input sorted by (distance, NodeId) — output should match
        // exactly the first M elements (no re-sorting).
        let cands = vec![
            Candidate::new(10, 5),
            Candidate::new(10, 7),
            Candidate::new(20, 3),
        ];
        let r = select_neighbors_simple(cands, 2);
        assert_eq!(r[0], Candidate::new(10, 5));
        assert_eq!(r[1], Candidate::new(10, 7));
    }
}

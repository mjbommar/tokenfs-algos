//! Portable scalar array × array kernels.
//!
//! Sorted-merge implementations of intersect / union / difference / XOR
//! over two `&[u16]` slices. These are the reference oracles for the
//! SIMD backends (Schlegel pcmpistrm and friends) and are also the
//! production path for run / scalar fallbacks.
//!
//! Galloping-search intersect (also known as binary-search-doubling) is
//! provided for adversarially-imbalanced inputs where one side is much
//! smaller than the other. The Schlegel paper's heuristic is to gallop
//! when `min(|a|, |b|) * gallop_threshold < max(|a|, |b|)`; we expose the
//! galloping primitive separately so the dispatch wrapper can apply the
//! threshold once per call.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Intersect two sorted `u16` slices via linear merge.
///
/// Pushes matching elements onto `out` in ascending order. Linear-time
/// in `|a| + |b|`. Used by both the scalar fallback and as the parity
/// oracle for the SIMD kernels.
pub fn intersect(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    out.clear();
    intersect_into_remainder(a, b, out);
}

/// Append-mode merge — scalar tail used by the SIMD kernels after the
/// vector body has consumed the head of both inputs. Preserves whatever
/// is already in `out` and pushes additional matches.
pub fn intersect_into_remainder(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    while ia < a.len() && ib < b.len() {
        let va = a[ia];
        let vb = b[ib];
        match va.cmp(&vb) {
            core::cmp::Ordering::Less => ia += 1,
            core::cmp::Ordering::Greater => ib += 1,
            core::cmp::Ordering::Equal => {
                out.push(va);
                ia += 1;
                ib += 1;
            }
        }
    }
}

/// Returns the cardinality of `a INTERSECT b` without building the result.
#[must_use]
pub fn intersect_cardinality(a: &[u16], b: &[u16]) -> u32 {
    let mut card: u32 = 0;
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    while ia < a.len() && ib < b.len() {
        let va = a[ia];
        let vb = b[ib];
        match va.cmp(&vb) {
            core::cmp::Ordering::Less => ia += 1,
            core::cmp::Ordering::Greater => ib += 1,
            core::cmp::Ordering::Equal => {
                card += 1;
                ia += 1;
                ib += 1;
            }
        }
    }
    card
}

/// Galloping-search intersect for adversarial selectivity.
///
/// Iterates the smaller side and uses exponential-then-binary search to
/// find each element in the larger side. Linear-time in
/// `|small| * log(|large|)`, which beats the linear merge when
/// `|large| / |small| > log(|large|)`.
pub fn intersect_galloping(small: &[u16], large: &[u16], out: &mut Vec<u16>) {
    out.clear();
    let mut large_start = 0_usize;
    for &needle in small {
        // Galloping bound search: exponentially expand the search window
        // starting from `large_start`, then binary-search the final
        // window.
        let mut step = 1_usize;
        let mut lo = large_start;
        let mut hi = lo + 1;
        while hi < large.len() && large[hi] < needle {
            lo = hi;
            hi += step;
            step *= 2;
        }
        let hi = hi.min(large.len());
        // `large[lo..hi]` is the bounded search range.
        match large[lo..hi].binary_search(&needle) {
            Ok(idx) => {
                out.push(needle);
                // Subsequent needle ≥ current needle; the next search
                // can resume from just past the match.
                large_start = lo + idx + 1;
            }
            Err(idx) => {
                // Resume from the insertion point — that's the first
                // position in `large` strictly greater than `needle`.
                large_start = lo + idx;
            }
        }
    }
}

/// Union of two sorted `u16` slices via linear merge.
///
/// Pushes the merged stream onto `out` in ascending order. Duplicate
/// values present in both inputs appear once.
pub fn union(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    out.clear();
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    while ia < a.len() && ib < b.len() {
        let va = a[ia];
        let vb = b[ib];
        match va.cmp(&vb) {
            core::cmp::Ordering::Less => {
                out.push(va);
                ia += 1;
            }
            core::cmp::Ordering::Greater => {
                out.push(vb);
                ib += 1;
            }
            core::cmp::Ordering::Equal => {
                out.push(va);
                ia += 1;
                ib += 1;
            }
        }
    }
    while ia < a.len() {
        out.push(a[ia]);
        ia += 1;
    }
    while ib < b.len() {
        out.push(b[ib]);
        ib += 1;
    }
}

/// Set difference (`a \ b`) via linear merge.
///
/// Emits elements of `a` not present in `b`.
pub fn difference(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    out.clear();
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    while ia < a.len() && ib < b.len() {
        let va = a[ia];
        let vb = b[ib];
        match va.cmp(&vb) {
            core::cmp::Ordering::Less => {
                out.push(va);
                ia += 1;
            }
            core::cmp::Ordering::Greater => ib += 1,
            core::cmp::Ordering::Equal => {
                ia += 1;
                ib += 1;
            }
        }
    }
    while ia < a.len() {
        out.push(a[ia]);
        ia += 1;
    }
}

/// Symmetric difference (`a XOR b`) via linear merge.
///
/// Emits elements present in exactly one of `a` or `b`.
pub fn symmetric_difference(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    out.clear();
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    while ia < a.len() && ib < b.len() {
        let va = a[ia];
        let vb = b[ib];
        match va.cmp(&vb) {
            core::cmp::Ordering::Less => {
                out.push(va);
                ia += 1;
            }
            core::cmp::Ordering::Greater => {
                out.push(vb);
                ib += 1;
            }
            core::cmp::Ordering::Equal => {
                ia += 1;
                ib += 1;
            }
        }
    }
    while ia < a.len() {
        out.push(a[ia]);
        ia += 1;
    }
    while ib < b.len() {
        out.push(b[ib]);
        ib += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // `Vec` and `vec!` are not in the no-std prelude; alias them from
    // `alloc` for the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    #[test]
    fn intersect_basic() {
        let a = [1_u16, 3, 5, 7, 9];
        let b = [2_u16, 3, 5, 8, 10];
        let mut out = Vec::new();
        intersect(&a, &b, &mut out);
        assert_eq!(out, vec![3, 5]);
        assert_eq!(intersect_cardinality(&a, &b), 2);
    }

    #[test]
    fn union_basic() {
        let a = [1_u16, 3, 5];
        let b = [2_u16, 3, 6];
        let mut out = Vec::new();
        union(&a, &b, &mut out);
        assert_eq!(out, vec![1, 2, 3, 5, 6]);
    }

    #[test]
    fn difference_basic() {
        let a = [1_u16, 2, 3, 4, 5];
        let b = [2_u16, 4];
        let mut out = Vec::new();
        difference(&a, &b, &mut out);
        assert_eq!(out, vec![1, 3, 5]);
    }

    #[test]
    fn symmetric_difference_basic() {
        let a = [1_u16, 2, 3];
        let b = [2_u16, 3, 4];
        let mut out = Vec::new();
        symmetric_difference(&a, &b, &mut out);
        assert_eq!(out, vec![1, 4]);
    }

    #[test]
    fn empty_inputs() {
        let empty = [0_u16; 0];
        let nonempty = [1_u16, 2, 3];
        let mut out = Vec::new();
        intersect(&empty, &nonempty, &mut out);
        assert!(out.is_empty());
        union(&empty, &nonempty, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
        difference(&empty, &nonempty, &mut out);
        assert!(out.is_empty());
        difference(&nonempty, &empty, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
        symmetric_difference(&empty, &nonempty, &mut out);
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn galloping_matches_linear_merge() {
        // 5 small needles vs 1000 hay values; both forms must agree.
        let small: Vec<u16> = (0..5).map(|i| i * 200).collect();
        let large: Vec<u16> = (0..1000).map(|i| i as u16).collect();
        let mut linear = Vec::new();
        intersect(&small, &large, &mut linear);
        let mut gallop = Vec::new();
        intersect_galloping(&small, &large, &mut gallop);
        assert_eq!(linear, gallop);
    }

    #[test]
    fn galloping_with_no_matches() {
        let small = [u16::MAX, u16::MAX - 1];
        let large: Vec<u16> = (0..1000).collect();
        let mut out = Vec::new();
        intersect_galloping(&small, &large, &mut out);
        assert!(out.is_empty());
    }
}

//! Recursive binary chunking.
//!
//! `recursive_split_fold(bytes, op)` splits `bytes` at a midpoint, recursively
//! processes each half, then folds the two children's results into one. The
//! accumulator type and the combine semantics are caller-controlled via a
//! [`FoldOp`] trait — typical instantiations:
//!
//! - **Histogram merge**: leaf computes a `[u64; 256]`; combine adds two
//!   arrays element-wise. Folding over the binary tree is a balanced
//!   parallel reduction in `O(log N)` depth.
//! - **Min-max-min boundary detection**: leaf returns `(min, max, len)`;
//!   combine yields `(min(a.min, b.min), max(a.max, b.max), a.len + b.len)`.
//! - **Approximate-fingerprint merge**: leaf computes a small fingerprint;
//!   combine concatenates / xor-folds the two children's fingerprints.
//!
//! Default split policy is mid-point (binary tree, depth `ceil(log2(N /
//! min_leaf_bytes))`). A caller can substitute a content-defined boundary
//! by implementing the `split_at` method on a custom [`SplitPolicy`].
//!
//! ## Allocation
//!
//! The recursive walk uses the call stack — depth is bounded by
//! `ceil(log2(N / min_leaf_bytes))`. For typical TokenFS sizes (≤ tens
//! of MiB, min_leaf ≥ 1 KiB), depth stays under 20 frames; safe for
//! kernel-adjacent and FUSE callers. There is no heap allocation.
//!
//! ## Parallel mode
//!
//! Under `feature = "parallel"`, [`recursive_split_fold_par`] uses
//! `rayon::join` at each split. Same algorithm, same fold semantics; the
//! tree-shape work decomposition gives near-perfect speedup on tree-balanced
//! inputs and graceful degradation on skewed splits.

use core::ops::Range;

/// User-supplied fold operation.
///
/// `Acc` is the accumulator type — anything `Send + Sync` if you plan to
/// use [`recursive_split_fold_par`]; otherwise unconstrained.
pub trait FoldOp {
    /// The accumulator type produced by [`leaf`](Self::leaf) and combined
    /// by [`combine`](Self::combine).
    type Acc;

    /// Computes the accumulator for a leaf segment.
    ///
    /// Called when a region is at or below the minimum leaf size, or when
    /// the region cannot be split further.
    fn leaf(&self, bytes: &[u8]) -> Self::Acc;

    /// Combines two child accumulators into a parent accumulator.
    ///
    /// Must be associative for the fold to be deterministic across
    /// different split policies. Commutativity is not required because
    /// the recursion always combines `(left, right)` in that order.
    fn combine(&self, left: Self::Acc, right: Self::Acc) -> Self::Acc;

    /// Identity element for the fold. Returned for an empty input.
    ///
    /// Default implementation panics — override when the fold needs an
    /// identity (e.g. the histogram-merge fold returns `[0; 256]`).
    fn identity(&self) -> Self::Acc {
        panic!("FoldOp::identity not implemented; cannot fold empty input")
    }
}

/// Strategy for choosing where to split a region.
pub trait SplitPolicy {
    /// Returns the offset in `bytes` at which to split, or `None` to
    /// stop recursing and treat `bytes` as a leaf.
    ///
    /// # Contract
    ///
    /// The returned offset must satisfy `0 < offset < bytes.len()` for
    /// the recursion to make progress. Implementations that violate this
    /// contract — including returning `Some(0)`, `Some(bytes.len())`, or
    /// `Some(n)` with `n > bytes.len()` — will be **silently treated as
    /// a leaf** (i.e., the same as returning `None`). This guarantees the
    /// recursive driver cannot infinite-loop or panic on a hostile or
    /// buggy `SplitPolicy` implementation supplied by a caller.
    ///
    /// This silent-leaf policy is intentional: the driver returns
    /// `F::Acc` (not `Result`), so there is no error channel to surface
    /// the violation through. Kernel/FUSE callers that need to enforce
    /// the contract strictly should validate inside their own
    /// `SplitPolicy::split_at` and only return values the driver will
    /// honor (or return `None` to stop early).
    fn split_at(&self, bytes: &[u8]) -> Option<usize>;
}

/// Mid-point split: yields a perfectly balanced binary tree.
#[derive(Copy, Clone, Debug)]
pub struct MidpointSplit {
    /// Minimum leaf size in bytes. Regions at or below this size become
    /// leaves and are not split further.
    pub min_leaf_bytes: usize,
}

impl MidpointSplit {
    /// Builds a midpoint splitter with the given minimum leaf size.
    #[must_use]
    pub const fn new(min_leaf_bytes: usize) -> Self {
        Self { min_leaf_bytes }
    }
}

impl Default for MidpointSplit {
    fn default() -> Self {
        Self {
            min_leaf_bytes: 4 * 1024,
        }
    }
}

impl SplitPolicy for MidpointSplit {
    fn split_at(&self, bytes: &[u8]) -> Option<usize> {
        if bytes.len() <= self.min_leaf_bytes.max(1) {
            return None;
        }
        Some(bytes.len() / 2)
    }
}

/// Performs the recursive split-fold over `bytes` using `op` and `policy`.
///
/// Returns `op.identity()` when `bytes` is empty (so callers must
/// implement [`FoldOp::identity`] for the empty case if they want one).
#[must_use]
pub fn recursive_split_fold<F, S>(bytes: &[u8], op: &F, policy: &S) -> F::Acc
where
    F: FoldOp,
    S: SplitPolicy,
{
    if bytes.is_empty() {
        return op.identity();
    }
    walk(bytes, op, policy, 0..bytes.len())
}

fn walk<F, S>(bytes: &[u8], op: &F, policy: &S, range: Range<usize>) -> F::Acc
where
    F: FoldOp,
    S: SplitPolicy,
{
    let slice = &bytes[range.clone()];
    match policy.split_at(slice) {
        None => op.leaf(slice),
        Some(offset) => {
            // Validate the caller-supplied offset before slicing/recursing.
            // A `SplitPolicy` that returns `Some(0)` or `Some(slice.len())`
            // would make the recursion never progress (one child is empty
            // and the other equals the parent), and `Some(n)` with
            // `n > slice.len()` would panic on the slice. Treat any
            // contract violation as a terminal leaf — see
            // `SplitPolicy::split_at` rustdoc for the silent-leaf policy.
            if offset == 0 || offset >= slice.len() {
                return op.leaf(slice);
            }
            let mid = range.start + offset;
            let left = walk(bytes, op, policy, range.start..mid);
            let right = walk(bytes, op, policy, mid..range.end);
            op.combine(left, right)
        }
    }
}

/// Parallel recursive split-fold using `rayon::join` at each split.
///
/// Requires `Acc: Send` and `FoldOp: Sync`. Produces the same result as
/// [`recursive_split_fold`] when the fold is associative.
#[cfg(feature = "parallel")]
#[must_use]
pub fn recursive_split_fold_par<F, S>(bytes: &[u8], op: &F, policy: &S) -> F::Acc
where
    F: FoldOp + Sync,
    F::Acc: Send,
    S: SplitPolicy + Sync,
{
    if bytes.is_empty() {
        return op.identity();
    }
    walk_par(bytes, op, policy, 0..bytes.len())
}

#[cfg(feature = "parallel")]
fn walk_par<F, S>(bytes: &[u8], op: &F, policy: &S, range: Range<usize>) -> F::Acc
where
    F: FoldOp + Sync,
    F::Acc: Send,
    S: SplitPolicy + Sync,
{
    let slice = &bytes[range.clone()];
    match policy.split_at(slice) {
        None => op.leaf(slice),
        Some(offset) => {
            // Mirror the sequential `walk` validation: silent-leaf on any
            // SplitPolicy contract violation (offset == 0 or >= slice.len()).
            // Required to keep `rayon::join` from splitting into an empty
            // child + a recursive same-size child (infinite loop) or
            // panicking on slice bounds.
            if offset == 0 || offset >= slice.len() {
                return op.leaf(slice);
            }
            let mid = range.start + offset;
            let (left, right) = rayon::join(
                || walk_par(bytes, op, policy, range.start..mid),
                || walk_par(bytes, op, policy, mid..range.end),
            );
            op.combine(left, right)
        }
    }
}

// ============================================================================
// Built-in fold ops for common cases
// ============================================================================

/// 256-bin histogram merge fold.
///
/// `leaf` produces a per-segment byte histogram; `combine` adds two
/// histograms element-wise. The result is the sum of all leaf histograms
/// — equal to the byte histogram of the full input, modulo associativity
/// of u64 addition (which is exact). Useful for producing a histogram of
/// a large buffer with parallel work decomposition under
/// [`recursive_split_fold_par`].
pub struct HistogramFold;

impl FoldOp for HistogramFold {
    type Acc = [u64; 256];

    fn leaf(&self, bytes: &[u8]) -> Self::Acc {
        let mut counts = [0_u64; 256];
        for &byte in bytes {
            counts[byte as usize] += 1;
        }
        counts
    }

    fn combine(&self, mut left: Self::Acc, right: Self::Acc) -> Self::Acc {
        for (l, r) in left.iter_mut().zip(right.iter()) {
            *l = l.saturating_add(*r);
        }
        left
    }

    fn identity(&self) -> Self::Acc {
        [0; 256]
    }
}

/// `(min, max, len)` reduction fold. Useful for bound-finding over a buffer.
pub struct MinMaxLenFold;

impl FoldOp for MinMaxLenFold {
    type Acc = (u8, u8, usize);

    fn leaf(&self, bytes: &[u8]) -> Self::Acc {
        if bytes.is_empty() {
            return (u8::MAX, 0, 0);
        }
        let (min, max) = bytes
            .iter()
            .fold((u8::MAX, 0_u8), |(lo, hi), &b| (lo.min(b), hi.max(b)));
        (min, max, bytes.len())
    }

    fn combine(&self, l: Self::Acc, r: Self::Acc) -> Self::Acc {
        (l.0.min(r.0), l.1.max(r.1), l.2 + r.2)
    }

    fn identity(&self) -> Self::Acc {
        (u8::MAX, 0, 0)
    }
}

/// Leaf-counting fold — useful for testing the recursion structure.
pub struct LeafCountFold;

impl FoldOp for LeafCountFold {
    type Acc = usize;

    fn leaf(&self, _bytes: &[u8]) -> Self::Acc {
        1
    }

    fn combine(&self, left: Self::Acc, right: Self::Acc) -> Self::Acc {
        left + right
    }

    fn identity(&self) -> Self::Acc {
        0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn empty_input_returns_identity() {
        let result = recursive_split_fold(&[], &HistogramFold, &MidpointSplit::new(64));
        assert_eq!(result, [0_u64; 256]);

        let count = recursive_split_fold(&[], &LeafCountFold, &MidpointSplit::new(64));
        assert_eq!(count, 0);
    }

    #[test]
    fn histogram_fold_matches_direct_count() {
        let bytes: Vec<u8> = (0..4096_u32).map(|i| (i.wrapping_mul(17)) as u8).collect();
        let folded = recursive_split_fold(&bytes, &HistogramFold, &MidpointSplit::new(128));

        let mut direct = [0_u64; 256];
        for &b in &bytes {
            direct[b as usize] += 1;
        }
        assert_eq!(folded, direct);
    }

    #[test]
    fn histogram_fold_is_independent_of_split_size() {
        let bytes: Vec<u8> = (0..8192_u32).map(|i| (i ^ (i >> 3)) as u8).collect();
        let small = recursive_split_fold(&bytes, &HistogramFold, &MidpointSplit::new(64));
        let medium = recursive_split_fold(&bytes, &HistogramFold, &MidpointSplit::new(512));
        let large = recursive_split_fold(&bytes, &HistogramFold, &MidpointSplit::new(4096));
        assert_eq!(small, medium);
        assert_eq!(small, large);
    }

    #[test]
    fn leaf_count_for_balanced_split() {
        // 4096 bytes, min_leaf = 256 → 4096/256 = 16 leaves at the bottom.
        // Midpoint splitter halves at each step until len ≤ 256.
        let bytes = vec![0_u8; 4096];
        let count = recursive_split_fold(&bytes, &LeafCountFold, &MidpointSplit::new(256));
        // 4096 / 2 = 2048, 1024, 512, 256 → 4 levels of split, 16 leaves.
        assert_eq!(count, 16);
    }

    #[test]
    fn leaf_count_for_below_min_leaf() {
        // 100 bytes, min_leaf = 256 → no split, 1 leaf.
        let bytes = vec![0_u8; 100];
        let count = recursive_split_fold(&bytes, &LeafCountFold, &MidpointSplit::new(256));
        assert_eq!(count, 1);
    }

    #[test]
    fn min_max_len_fold_matches_direct_scan() {
        let bytes: Vec<u8> = (0..1000_u32).map(|i| (i % 200) as u8).collect();
        let (min, max, len) = recursive_split_fold(&bytes, &MinMaxLenFold, &MidpointSplit::new(64));
        assert_eq!(len, 1000);
        assert_eq!(min, 0);
        assert_eq!(max, 199);
    }

    #[test]
    fn min_leaf_zero_or_one_does_not_infinite_loop() {
        // The MidpointSplit guards `min_leaf_bytes.max(1)`, so even 0 is safe.
        let bytes = vec![0_u8; 64];
        let count = recursive_split_fold(&bytes, &LeafCountFold, &MidpointSplit::new(0));
        // With min_leaf=1, every byte becomes a leaf → 64 leaves.
        assert_eq!(count, 64);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_fold_matches_sequential_fold() {
        let bytes: Vec<u8> = (0..16384_u32).map(|i| (i.wrapping_mul(31)) as u8).collect();
        let policy = MidpointSplit::new(256);
        let seq = recursive_split_fold(&bytes, &HistogramFold, &policy);
        let par = recursive_split_fold_par(&bytes, &HistogramFold, &policy);
        assert_eq!(seq, par);
    }

    /// `SplitPolicy` shim that always returns the same caller-supplied
    /// offset — used to exercise the silent-leaf guard in the recursive
    /// driver against `Some(0)`, `Some(len)`, and `Some(>len)` contract
    /// violations. Stateless so it satisfies `Sync` for `walk_par`.
    ///
    /// For an invalid offset the guard converts the very first recursion
    /// attempt into a leaf, so `split_at` is invoked at most once per call
    /// regardless of the configured offset.
    struct FixedOffsetSplit {
        offset: usize,
    }

    impl FixedOffsetSplit {
        fn new(offset: usize) -> Self {
            Self { offset }
        }
    }

    impl SplitPolicy for FixedOffsetSplit {
        fn split_at(&self, _bytes: &[u8]) -> Option<usize> {
            Some(self.offset)
        }
    }

    #[test]
    fn split_policy_some_zero_is_treated_as_leaf() {
        // SplitPolicy returns `Some(0)`, which would cause infinite recursion
        // (left child empty, right child equals parent) absent the guard.
        // Expect: a single leaf — no recursion, no panic, no infinite loop.
        let bytes = vec![0_u8; 1024];
        let policy = FixedOffsetSplit::new(0);
        let count = recursive_split_fold(&bytes, &LeafCountFold, &policy);
        assert_eq!(count, 1, "Some(0) must be treated as a leaf");
    }

    #[test]
    fn split_policy_some_len_is_treated_as_leaf() {
        // SplitPolicy returns `Some(bytes.len())` — left child equals parent,
        // right child empty. Same-size recursion would loop forever.
        let bytes = vec![0_u8; 1024];
        let policy = FixedOffsetSplit::new(bytes.len());
        let count = recursive_split_fold(&bytes, &LeafCountFold, &policy);
        assert_eq!(count, 1, "Some(len) must be treated as a leaf");
    }

    #[test]
    fn split_policy_some_greater_than_len_does_not_panic() {
        // SplitPolicy returns `Some(bytes.len() + 1)` — would panic on the
        // out-of-bounds slice in the parent recursion absent the guard.
        let bytes = vec![0_u8; 1024];
        let policy = FixedOffsetSplit::new(bytes.len() + 1);
        let count = recursive_split_fold(&bytes, &LeafCountFold, &policy);
        assert_eq!(
            count, 1,
            "Some(len + 1) must be treated as a leaf, not panic"
        );
    }

    #[test]
    fn split_policy_some_far_greater_than_len_does_not_panic() {
        // Stress: pathological hostile offset (`usize::MAX`).
        let bytes = vec![0_u8; 256];
        let policy = FixedOffsetSplit::new(usize::MAX);
        let count = recursive_split_fold(&bytes, &LeafCountFold, &policy);
        assert_eq!(count, 1, "usize::MAX offset must be treated as a leaf");
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_split_policy_some_zero_is_treated_as_leaf() {
        // Mirror of the sequential guard test for `walk_par`.
        let bytes = vec![0_u8; 1024];
        let policy = FixedOffsetSplit::new(0);
        let count = recursive_split_fold_par(&bytes, &LeafCountFold, &policy);
        assert_eq!(count, 1);
    }
}

//! Joint entropy estimators over adjacent byte pairs.

use crate::{
    histogram::{
        BytePairHistogram, BytePairScratch,
        pair::{BytePairCountsScratch, BytePairHistogramView},
    },
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
/// This uses a dense 65,536-bin byte-pair histogram and does not heap-
/// allocate.
///
/// Available only with `feature = "userspace"`. Kernel/FUSE callers
/// should use [`h2_pairs_with_scratch`] (heap-free; caller-provided
/// scratch) or [`h2_pairs_with_dense_scratch`] (heap-free; caller-
/// provided dense counter table) instead.
///
/// # Stack
///
/// Materialises a [`BytePairHistogram`] (~256 KiB) on the call frame —
/// well above typical kernel/FUSE stack budgets (8-16 KiB; see
/// `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`). Kernel-adjacent
/// callers must instead use [`h2_pairs_with_scratch`] (lazy-clear scratch
/// path) or [`h2_pairs_with_dense_scratch`] (caller-provided dense
/// counter table) — both keep the dense counters off the stack
/// (audit-R8 #6a). Audit-R8 #6b removes this by-value form from the
/// kernel-default surface entirely.
#[cfg(feature = "userspace")]
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

/// Heap-free / kernel-safe sibling of [`h2_pairs`]: computes exact
/// adjacent-pair joint entropy `H(X_i, X_{i+1})` over `bytes` using a
/// caller-provided dense counter table.
///
/// Mirrors the §156 caller-provided-scratch convention used by
/// [`crate::similarity::kernels_gather::build_table_from_seeds_into`]
/// (audit-R5). Allocate `scratch` once via a path that does not stack-
/// materialise the inner array (`Box::<BytePairCountsScratch>::new_uninit`,
/// thread-local pool, postgres memory context, kernel `kmalloc`'d slab)
/// and pass a `&mut` borrow. The caller decides where the 256 KiB dense
/// counter table lives, so the kernel-stack hazard documented on
/// [`h2_pairs`] does not apply (audit-R8 #6a).
///
/// `scratch` is fully overwritten on entry — its prior contents are
/// discarded — so callers may reuse a single buffer across many calls
/// without pre-clearing it.
///
/// Bit-exact with [`h2_pairs`] for every input.
#[must_use]
pub fn h2_pairs_with_dense_scratch(bytes: &[u8], scratch: &mut BytePairCountsScratch) -> f32 {
    let view = BytePairHistogram::with_scratch(bytes, scratch);
    h2_from_pair_view(&view)
}

/// Computes joint entropy from a dense byte-pair histogram.
#[must_use]
pub fn h2_from_pair_histogram(histogram: &BytePairHistogram) -> f32 {
    entropy_counts_u32(histogram.counts(), histogram.observations())
}

/// Computes joint entropy from a borrowed dense byte-pair histogram view.
///
/// Companion of [`h2_from_pair_histogram`] for the heap-free /
/// caller-provided-scratch path: takes a [`BytePairHistogramView`]
/// returned by [`BytePairHistogram::with_scratch`] so callers in
/// kernel-adjacent contexts can compute joint entropy without the dense
/// counter table living on the call frame (audit-R8 #6a).
#[must_use]
pub fn h2_from_pair_view(view: &BytePairHistogramView<'_>) -> f32 {
    entropy_counts_u32(view.counts(), view.observations())
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
        PairEntropyShape, PairEntropyStrategy, h2_pairs_with_dense_scratch, plan_pair_entropy,
    };
    #[cfg(feature = "userspace")]
    use super::{h2_pairs, h2_pairs_with_scratch};
    #[cfg(feature = "userspace")]
    use crate::histogram::BytePairScratch;
    use crate::histogram::pair::BytePairCountsScratch;
    // `Box` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::boxed::Box;

    /// Heap-allocate a zeroed `BytePairCountsScratch` without ever
    /// materialising the 256 KiB inner array on the test stack — the
    /// kernel-safe constructor under test would defeat itself if its
    /// own test triggered the hazard.
    fn alloc_zeroed_dense_scratch() -> Box<BytePairCountsScratch> {
        use core::mem::MaybeUninit;
        let mut uninit: Box<MaybeUninit<BytePairCountsScratch>> = Box::new_uninit();
        // SAFETY: heap storage of `sizeof::<BytePairCountsScratch>()` is
        // valid; bit-pattern 0 is a valid u32 in every counter slot.
        unsafe {
            core::ptr::write_bytes(uninit.as_mut_ptr().cast::<u32>(), 0, 256 * 256);
            uninit.assume_init()
        }
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn repeated_pair_joint_entropy_is_zero() {
        assert_eq!(h2_pairs(b"aaaaaaaa"), 0.0);
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn alternating_pairs_have_one_bit_joint_entropy() {
        let entropy = h2_pairs(b"abababa");
        assert!((entropy - 1.0).abs() < 1e-6, "h2_pairs={entropy}");
    }

    #[cfg(feature = "userspace")]
    #[test]
    fn scratch_h2_matches_dense_h2() {
        let mut scratch = Box::new(BytePairScratch::new());
        let bytes = b"abacabadabacaba";
        assert_eq!(h2_pairs(bytes), h2_pairs_with_scratch(bytes, &mut scratch));
    }

    /// `h2_pairs_with_dense_scratch` must be bit-exact with the legacy
    /// by-value `h2_pairs` form across representative inputs. This is
    /// the parity guarantee for audit-R8 #6a: the kernel-safe
    /// dense-scratch path must not drift in semantics.
    ///
    /// Gated on `userspace` because `h2_pairs` itself is now gated
    /// (audit-R8 #6b). The kernel-safe path covers the same inputs in
    /// `dense_scratch_h2_handles_trivial_input` and
    /// `dense_scratch_h2_reuses_buffer_across_calls`.
    #[cfg(feature = "userspace")]
    #[test]
    fn dense_scratch_h2_matches_h2_pairs() {
        let mut scratch = alloc_zeroed_dense_scratch();
        for &payload in &[
            b"" as &[u8],
            b"a",
            b"aaaaaaaa",
            b"abababa",
            b"abacabadabacaba",
            b"the quick brown fox jumps over the lazy dog 0123456789!@#$%^&*()",
        ] {
            assert_eq!(
                h2_pairs(payload),
                h2_pairs_with_dense_scratch(payload, &mut scratch),
                "payload={payload:?}",
            );
        }
    }

    /// Trivial input (empty / single byte) returns 0.0 with no UB on a
    /// freshly cleared dense scratch.
    #[test]
    fn dense_scratch_h2_handles_trivial_input() {
        let mut scratch = alloc_zeroed_dense_scratch();
        assert_eq!(h2_pairs_with_dense_scratch(b"", &mut scratch), 0.0);
        assert_eq!(h2_pairs_with_dense_scratch(b"x", &mut scratch), 0.0);
    }

    /// Reusing a single dense scratch across multiple calls produces
    /// independent results — the per-call clear inside
    /// `h2_pairs_with_dense_scratch` discards prior contents.
    #[test]
    fn dense_scratch_h2_reuses_buffer_across_calls() {
        let mut scratch = alloc_zeroed_dense_scratch();
        let _ = h2_pairs_with_dense_scratch(b"aaaaaaaa", &mut scratch);
        let entropy = h2_pairs_with_dense_scratch(b"abababa", &mut scratch);
        assert!((entropy - 1.0).abs() < 1e-6, "after reuse: {entropy}");
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

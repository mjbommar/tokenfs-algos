//! HNSW search-only graph traversal — Algorithm 5 + Algorithm 2.
//!
//! Implements `try_search` (kernel-safe, audit-R10 surface) over an
//! [`HnswView`]. Composes [`VisitedSet`] (`O(1)`-clear bitset) and
//! [`MaxHeap`] (sorted-vec heap with deterministic tie-break) plus the
//! scalar distance kernels in [`kernels::scalar`].
//!
//! Algorithm references (per `docs/hnsw/research/HNSW_ALGORITHM_NOTES.md`):
//!
//! - **Algorithm 5: K-NN-SEARCH** — descent through upper layers with
//!   `ef = 1`, then a single base-layer call with `ef = ef_search`.
//! - **Algorithm 2: SEARCH-LAYER** — beam-search inside one layer.
//!   Standard early-termination when the closest unexplored candidate
//!   is farther than the current furthest result.
//!
//! Edge cases handled:
//!
//! - Empty index → returns empty `Vec`.
//! - Single-node index → returns the entry-point as the only result.
//! - `k > node_count` → returns all nodes; no padding.
//! - `ef_search < k` → bumped to `k` internally (matches hnswlib).
//! - Tied distances → tie-break by `(distance, NodeId)` ascending
//!   (the [`MaxHeap`] / [`Candidate`] ordering, per
//!   `DETERMINISM.md` §4).
//! - Disconnected component → silently returns suboptimal `W`. The
//!   hierarchy mitigates but does not eliminate this; documented as
//!   a known HNSW caveat (paper §4-§6, `HNSW_ALGORITHM_NOTES.md`
//!   §1 Algorithm 2).
//!
//! # Audit posture
//!
//! - [`try_search`] is the kernel-safe entry; never panics on caller
//!   input. Reachable in `--no-default-features --features alloc`.
//! - Phase 1 walker uses scalar f32/i8/u8/binary metrics only.
//!   Kernel-mode FPU bracketing for f32 paths lands in Phase 5.
//! - SIMD distance dispatch lands in Phase 2.

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use super::candidates::{Candidate, MaxHeap};
use super::header::{HnswHeaderError, MetricKind, ScalarKind};
use super::kernels::scalar;
use super::kernels::{HnswKernelError, encode_f32};
use super::view::{HnswView, HnswViewError};
use super::visit::VisitedSet;
use super::{Distance, NodeKey};

/// Search configuration. Constructed by the caller and passed to
/// [`try_search`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SearchConfig {
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Dynamic candidate-list size at the base layer. Internally
    /// bumped to `max(k, ef_search)`.
    pub ef_search: usize,
}

impl SearchConfig {
    /// Convenience constructor.
    pub const fn new(k: usize, ef_search: usize) -> Self {
        SearchConfig { k, ef_search }
    }

    /// Default per `HNSW_ALGORITHM_NOTES.md` §2: `k = 16`, `ef = 64`.
    /// Suitable for binary Hamming on 32-byte F22 fingerprints.
    pub const DEFAULT: SearchConfig = SearchConfig {
        k: 16,
        ef_search: 64,
    };
}

/// Errors produced by [`try_search`]. Distinct from
/// [`HnswViewError`] so callers can react to "valid view, bad
/// query" without conflating with view-construction failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HnswSearchError {
    /// `config.k == 0`.
    InvalidK,
    /// Query vector length does not match the view's
    /// `bytes_per_vector`.
    QueryLengthMismatch {
        /// Length the caller supplied.
        got: usize,
        /// Length expected, from `view.bytes_per_vector()`.
        expected: usize,
    },
    /// View's metric / scalar combination has no v0.7.0 distance
    /// kernel (e.g. `f64` scalar, `Pearson` metric, `Sorensen` metric).
    UnsupportedMetricScalar {
        /// Metric kind from the view's header.
        metric: MetricKind,
        /// Scalar kind from the view's header.
        scalar: ScalarKind,
    },
    /// View accessor returned an error during traversal (out-of-range
    /// node id, out-of-range neighbor slot id, etc.). The walker
    /// considers this fatal because `HnswView::try_new` should have
    /// rejected such inputs at construction time.
    ViewCorruption(HnswViewError),
    /// Distance kernel returned an error (typically a length mismatch
    /// or zero-vector cosine).
    KernelError(HnswKernelError),
}

impl From<HnswViewError> for HnswSearchError {
    fn from(value: HnswViewError) -> Self {
        HnswSearchError::ViewCorruption(value)
    }
}

impl From<HnswKernelError> for HnswSearchError {
    fn from(value: HnswKernelError) -> Self {
        HnswSearchError::KernelError(value)
    }
}

impl From<HnswHeaderError> for HnswSearchError {
    fn from(value: HnswHeaderError) -> Self {
        HnswSearchError::ViewCorruption(HnswViewError::Header(value))
    }
}

impl core::fmt::Display for HnswSearchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidK => f.write_str("config.k must be > 0"),
            Self::QueryLengthMismatch { got, expected } => write!(
                f,
                "query length mismatch: got {got} bytes, expected {expected}"
            ),
            Self::UnsupportedMetricScalar { metric, scalar } => {
                write!(f, "no v0.7.0 distance kernel for ({metric:?}, {scalar:?})")
            }
            Self::ViewCorruption(inner) => write!(f, "view corruption: {inner}"),
            Self::KernelError(inner) => write!(f, "distance kernel error: {inner}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HnswSearchError {}

/// Search-only HNSW K-NN entry point.
///
/// Returns up to `config.k` (`NodeKey`, `Distance`) pairs sorted by
/// ascending distance. Length may be less than `config.k` if the index
/// has fewer than `k` nodes.
///
/// Never panics on caller input. Reachable from
/// `--no-default-features --features alloc` builds.
pub fn try_search(
    view: &HnswView<'_>,
    query: &[u8],
    config: &SearchConfig,
) -> Result<Vec<(NodeKey, Distance)>, HnswSearchError> {
    if config.k == 0 {
        return Err(HnswSearchError::InvalidK);
    }
    let expected_bytes = view.bytes_per_vector();
    if query.len() != expected_bytes {
        return Err(HnswSearchError::QueryLengthMismatch {
            got: query.len(),
            expected: expected_bytes,
        });
    }

    let node_count = view.node_count();
    if node_count == 0 {
        return Ok(Vec::new());
    }

    // Confirm the (metric, scalar) combo has a v1 kernel; fail fast
    // with a clear error otherwise.
    let metric = view.header().metric_kind();
    let scalar = view.header().scalar_kind();
    if !is_supported_combo(metric, scalar) {
        return Err(HnswSearchError::UnsupportedMetricScalar { metric, scalar });
    }

    // Descent phase (Algorithm 5 lines 4-6): walk down from `top_level`
    // to layer 1 with `ef = 1`. Upper layers are sparse shortcut
    // layers; a single best-candidate suffices.
    let entry = view
        .entry_point()
        .expect("node_count > 0 implies entry_point is Some");
    let entry_node = view.try_node(entry)?;
    let entry_dist = compute_distance(query, entry_node.vector_bytes(), metric, scalar)?;
    let mut current_best = Candidate::new(entry_dist, entry);

    let top_level = view.max_level();

    // Visited set is reused across all layer searches in this query.
    let mut visited = VisitedSet::try_with_capacity(node_count);

    let ctx = SearchCtx {
        view,
        query,
        metric,
        scalar,
    };

    // Layer descent: top_level..=1 (skip layer 0 — that's the final
    // call below).
    for layer in (1..=top_level).rev() {
        // SEARCH-LAYER with ef = 1; entry is the running best.
        let layer_w = search_layer(&ctx, current_best, layer, 1, &mut visited)?;
        if let Some(best) = layer_w.peek_best().copied() {
            current_best = best;
        }
        // Reset visited between layers so a node visited in layer N is
        // re-considered in layer N-1 (upper-layer adjacency is sparse;
        // the same node may be a useful neighbor on a lower layer).
        visited.clear();
    }

    // Base-layer search (Algorithm 5 line 7): ef = max(k, ef_search).
    let ef = config.ef_search.max(config.k);
    let base = search_layer(&ctx, current_best, 0, ef, &mut visited)?;

    // Algorithm 5 line 8: return top-K from the base-layer working set.
    // base is already sorted ascending by (distance, NodeId); take the
    // first `k`.
    let mut out = Vec::with_capacity(config.k.min(base.len()));
    for cand in base.iter_best_first().take(config.k) {
        // Map slot → NodeKey by reading the on-disk key field.
        let node = view.try_node(cand.node)?;
        out.push((node.key(), cand.distance));
    }
    Ok(out)
}

/// Per-query immutable context bundled to keep `search_layer`'s
/// signature under clippy's `too-many-arguments` cap.
struct SearchCtx<'a, 'b> {
    view: &'a HnswView<'b>,
    query: &'a [u8],
    metric: MetricKind,
    scalar: ScalarKind,
}

/// Algorithm 2 — SEARCH-LAYER. Returns the working set `W` (a
/// max-heap capped at `ef`).
///
/// `entry` is the single starting candidate. (HNSW paper allows
/// multi-element entry sets; in our walker only Algorithm 5's
/// descent invokes this and that always passes a single entry.)
fn search_layer(
    ctx: &SearchCtx<'_, '_>,
    entry: Candidate,
    layer: u8,
    ef: usize,
    visited: &mut VisitedSet,
) -> Result<MaxHeap, HnswSearchError> {
    // C: candidate priority queue (min-heap by distance — we use a
    // separate MaxHeap inverted via `peek_best`/`pop_best` semantics).
    // We need extract-nearest, so a sorted-vec is perfect.
    let mut candidates = MaxHeap::try_with_capacity(ef.max(1));
    // W: results (max-heap capped at ef).
    let mut nearest = MaxHeap::try_with_capacity(ef.max(1));

    visited.mark(entry.node);
    candidates.try_push(entry);
    nearest.try_push(entry);

    while !candidates.is_empty() {
        // Line 5: extract nearest from C.
        let c = *candidates
            .peek_best()
            .expect("candidates non-empty (loop condition)");
        candidates_remove_best(&mut candidates);

        // Line 6-8: early-terminate when c is worse than W's worst.
        let worst_w = nearest
            .peek_worst()
            .copied()
            .expect("nearest non-empty (entry was pushed)");
        if c.distance > worst_w.distance {
            break;
        }

        // Line 9-17: explore c's neighborhood at this layer.
        let c_node = ctx.view.try_node(c.node)?;
        let neighbors = c_node.try_neighbors(layer)?;

        for e_id in neighbors.iter() {
            // Line 10-11: skip if visited.
            if !visited.mark(e_id) {
                continue;
            }
            // Line 12-15: compute distance, conditionally add to W and C.
            let e_node = ctx.view.try_node(e_id)?;
            let e_dist =
                compute_distance(ctx.query, e_node.vector_bytes(), ctx.metric, ctx.scalar)?;
            let e_cand = Candidate::new(e_dist, e_id);

            // Line 13: if d(e,q) < d(f,q) OR |W| < ef, accept.
            let accept_into_w = !nearest.is_full()
                || nearest
                    .peek_worst()
                    .is_some_and(|w| e_cand.distance < w.distance);
            if accept_into_w {
                // Line 14-15: add to candidates (for further expansion)
                // and to W.
                candidates.try_push(e_cand);
                nearest.try_push(e_cand);
                // Lines 16-17 (cap |W| <= ef) handled by MaxHeap's
                // bounded try_push: it auto-evicts when full.
            }
        }
    }

    Ok(nearest)
}

/// Remove the heap's best (smallest) candidate. Mirrors the
/// "extract nearest from C" step in Algorithm 2 line 5.
///
/// `MaxHeap` already exposes `pop_worst` (back of the sorted vec); we
/// need front-of-vec removal. Implemented here rather than as a
/// general MaxHeap API because the front-removal is only needed by
/// the walker's beam search.
fn candidates_remove_best(candidates: &mut MaxHeap) {
    if candidates.is_empty() {
        return;
    }
    // Drain into a sorted vec, drop the first, rebuild. O(cap) per
    // call but cap is small (= ef) and beam-search runs at most O(ef
    // * log N) iterations. For Phase 1 correctness this is fine; if
    // profiling later shows it dominates we'll add a proper
    // pop_best to MaxHeap.
    let mut v = core::mem::replace(
        candidates,
        MaxHeap::try_with_capacity(candidates.capacity()),
    )
    .into_sorted_vec();
    if !v.is_empty() {
        v.remove(0);
    }
    for c in v {
        candidates.try_push(c);
    }
}

/// Distance dispatch on (metric, scalar) — Phase 1 uses scalar
/// kernels only. SIMD dispatch (`kernels::auto::*`) lands Phase 2.
fn compute_distance(
    query: &[u8],
    target: &[u8],
    metric: MetricKind,
    scalar: ScalarKind,
) -> Result<Distance, HnswSearchError> {
    match (metric, scalar) {
        // L2² — three scalar widths.
        (MetricKind::L2Squared, ScalarKind::F32) => {
            scalar::try_l2_squared_f32(bytes_as_f32(query), bytes_as_f32(target))
                .map_err(Into::into)
        }
        (MetricKind::L2Squared, ScalarKind::I8) => {
            scalar::try_l2_squared_i8(bytes_as_i8(query), bytes_as_i8(target)).map_err(Into::into)
        }
        (MetricKind::L2Squared, ScalarKind::U8) => {
            scalar::try_l2_squared_u8(query, target).map_err(Into::into)
        }

        // Cosine — three scalar widths.
        (MetricKind::Cosine, ScalarKind::F32) => {
            scalar::try_cosine_f32(bytes_as_f32(query), bytes_as_f32(target)).map_err(Into::into)
        }
        (MetricKind::Cosine, ScalarKind::I8) => {
            scalar::try_cosine_i8(bytes_as_i8(query), bytes_as_i8(target)).map_err(Into::into)
        }
        (MetricKind::Cosine, ScalarKind::U8) => {
            scalar::try_cosine_u8(query, target).map_err(Into::into)
        }

        // Dot (inner product, returned negated).
        (MetricKind::InnerProduct, ScalarKind::F32) => {
            scalar::try_dot_f32(bytes_as_f32(query), bytes_as_f32(target)).map_err(Into::into)
        }
        (MetricKind::InnerProduct, ScalarKind::I8) => {
            scalar::try_dot_i8(bytes_as_i8(query), bytes_as_i8(target)).map_err(Into::into)
        }
        (MetricKind::InnerProduct, ScalarKind::U8) => {
            scalar::try_dot_u8(query, target).map_err(Into::into)
        }

        // Binary metrics — only (B1x8, Hamming|Jaccard) supported.
        (MetricKind::Hamming, ScalarKind::B1x8) => {
            scalar::try_hamming_binary(query, target).map_err(Into::into)
        }
        (MetricKind::Jaccard, ScalarKind::B1x8) | (MetricKind::Tanimoto, ScalarKind::B1x8) => {
            // Tanimoto on binary collapses to Jaccard per
            // USEARCH_DEEP_DIVE §5 / SIMD_PRIOR_ART.
            scalar::try_jaccard_binary(query, target).map_err(Into::into)
        }

        // Anything else fell through is_supported_combo and is a bug.
        _ => Err(HnswSearchError::UnsupportedMetricScalar { metric, scalar }),
    }
}

/// Whether Phase 1 ships a kernel for this combo. Kept in sync with
/// [`compute_distance`].
fn is_supported_combo(metric: MetricKind, scalar: ScalarKind) -> bool {
    matches!(
        (metric, scalar),
        (
            MetricKind::L2Squared,
            ScalarKind::F32 | ScalarKind::I8 | ScalarKind::U8
        ) | (
            MetricKind::Cosine,
            ScalarKind::F32 | ScalarKind::I8 | ScalarKind::U8
        ) | (
            MetricKind::InnerProduct,
            ScalarKind::F32 | ScalarKind::I8 | ScalarKind::U8
        ) | (
            MetricKind::Hamming | MetricKind::Jaccard | MetricKind::Tanimoto,
            ScalarKind::B1x8
        )
    )
}

/// Reinterpret a byte slice as `&[f32]`. Length must be a multiple
/// of 4; otherwise the trailing bytes are ignored (the kernel's
/// length check will catch the mismatch).
#[inline]
fn bytes_as_f32(bytes: &[u8]) -> &[f32] {
    // SAFETY: f32 has the same alignment as i32 / u32 (4 bytes), and
    // the byte slice may not be aligned. We use the unaligned-load
    // path via from_le_bytes inside the kernel; here we still need to
    // expose &[f32] because the scalar kernels take that shape.
    //
    // To avoid alignment UB we copy into a temporary via from_raw_parts
    // only when alignment permits; otherwise we fall back to a slice
    // that the kernel will read via misaligned-load equivalents.
    //
    // For Phase 1 we accept that the toy fixture's vectors are
    // aligned (they sit at file offset 0x08, an 8-byte boundary
    // post-shape-header; for u8 it doesn't matter, for f32 it would
    // need to be 4-aligned). The HnswView's vector_bytes() returns a
    // subslice of the mmap'd file which may not be 4-aligned in
    // general. Phase 2's SIMD kernels handle this with explicit
    // unaligned loads; for the scalar reference we use the
    // reinterpret_cast since `from_le_bytes` is what really runs
    // inside the loop.
    debug_assert!(
        bytes.len().is_multiple_of(4),
        "f32 byte slice length {} not a multiple of 4",
        bytes.len()
    );
    // NOTE: bytemuck would be the right tool here but is not yet a
    // dep. For Phase 1 use a manual cast guarded by alignment check.
    let ptr = bytes.as_ptr() as *const f32;
    let len = bytes.len() / 4;
    if (ptr as usize).is_multiple_of(core::mem::align_of::<f32>()) {
        // Safe: bytes is borrowed for &self, alignment satisfied,
        // length is len * 4 bytes which fits.
        unsafe { core::slice::from_raw_parts(ptr, len) }
    } else {
        // Fallback: would need a per-element from_le_bytes. For
        // Phase 1 this branch is unreachable on the toy fixture
        // (vectors are 4-aligned). Phase 2 SIMD kernels handle
        // unaligned loads natively. Returning an empty slice would
        // make the kernel return a length-mismatch error, which the
        // walker correctly surfaces.
        &[]
    }
}

/// Reinterpret a byte slice as `&[i8]`. Always safe — i8 has
/// alignment 1 and the same in-memory layout as u8.
#[inline]
fn bytes_as_i8(bytes: &[u8]) -> &[i8] {
    let ptr = bytes.as_ptr() as *const i8;
    let len = bytes.len();
    // SAFETY: i8 and u8 are layout-identical; alignment 1.
    unsafe { core::slice::from_raw_parts(ptr, len) }
}

// Force the encode_f32 import to be used (for documentation symmetry
// with kernels::scalar; the actual encoding happens inside scalar
// kernels themselves).
#[allow(dead_code)]
const _: fn(f32) -> Distance = encode_f32;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::super::tests::build_toy_fixture;
    use super::super::view::HnswView;
    use super::*;

    /// L2² toy index: query vector matching the bytes of node 1's
    /// vector should rank node 1 first.
    #[test]
    fn try_search_returns_node_0_for_node_0_query() {
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();

        // Query = node 0's vector (toy_vector(0))
        let query = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
        let cfg = SearchConfig::new(2, 4);
        let result = try_search(&view, &query, &cfg).unwrap();
        assert_eq!(result.len(), 2);
        // Node 0's key (TOY_KEY_BASE | 0) should be first with distance 0.
        assert_eq!(result[0].0, 0x1000_0000_0000_0000);
        assert_eq!(result[0].1, 0); // L2² of identical vectors = 0
    }

    #[test]
    fn try_search_returns_node_2_for_node_2_query() {
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();

        let query = [0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37];
        let cfg = SearchConfig::new(1, 4);
        let result = try_search(&view, &query, &cfg).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 0x1000_0000_0000_0002);
        assert_eq!(result[0].1, 0);
    }

    #[test]
    fn try_search_returns_top_k_in_distance_order() {
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();

        // Query close to node 0 — node 0 should be first.
        let query = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
        let cfg = SearchConfig::new(4, 8);
        let result = try_search(&view, &query, &cfg).unwrap();
        assert_eq!(result.len(), 4);

        // Distances should be monotonically non-decreasing.
        for w in result.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "results not sorted by distance: {:?} > {:?}",
                w[0],
                w[1]
            );
        }

        // Node 0 should be first (exact match).
        assert_eq!(result[0].0, 0x1000_0000_0000_0000);
    }

    #[test]
    fn try_search_invalid_k_returns_error() {
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();
        let query = [0u8; 8];
        let cfg = SearchConfig::new(0, 4);
        let err = try_search(&view, &query, &cfg).unwrap_err();
        assert_eq!(err, HnswSearchError::InvalidK);
    }

    #[test]
    fn try_search_query_length_mismatch_returns_error() {
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();
        let query = [0u8; 16]; // toy expects 8 bytes
        let cfg = SearchConfig::new(2, 4);
        let err = try_search(&view, &query, &cfg).unwrap_err();
        assert_eq!(
            err,
            HnswSearchError::QueryLengthMismatch {
                got: 16,
                expected: 8,
            }
        );
    }

    #[test]
    fn try_search_k_larger_than_node_count_returns_all_nodes() {
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();
        let query = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
        let cfg = SearchConfig::new(100, 100);
        let result = try_search(&view, &query, &cfg).unwrap();
        assert_eq!(result.len(), 4); // toy has 4 nodes; no padding
    }

    #[test]
    fn try_search_ef_below_k_bumped_internally() {
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();
        let query = [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17];
        // ef_search = 1 but k = 4 — internally bumped to 4.
        let cfg = SearchConfig::new(4, 1);
        let result = try_search(&view, &query, &cfg).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].0, 0x1000_0000_0000_0000);
    }

    #[test]
    fn try_search_results_match_brute_force_on_toy() {
        // Brute-force oracle: compute L2² from query to every node and
        // sort. The walker should return the same top-k.
        let fixture = build_toy_fixture();
        let view = HnswView::try_new(&fixture).unwrap();

        let query = [0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F];

        // Brute force.
        let mut brute: Vec<(NodeKey, Distance)> = (0..4u32)
            .map(|slot| {
                let n = view.try_node(slot).unwrap();
                let d = scalar::try_l2_squared_u8(&query, n.vector_bytes()).unwrap();
                (n.key(), d)
            })
            .collect();
        brute.sort_by_key(|(_, d)| *d);

        let cfg = SearchConfig::new(4, 8);
        let walker_result = try_search(&view, &query, &cfg).unwrap();

        assert_eq!(walker_result, brute);
    }
}

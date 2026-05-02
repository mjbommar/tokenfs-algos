//! Hilbert curve ordering for 2D and N-D point data.
//!
//! See `docs/v0.2_planning/14_PERMUTATION.md` § 4 for the spec. The
//! algorithm:
//!
//! 1. Find the per-axis min/max of the input points.
//! 2. Quantize each axis to a `u32` grid of `0..=GRID_MAX` so the
//!    Hilbert key computation operates on integer coordinates.
//! 3. Compute the Hilbert key per point.
//! 4. Argsort point indices by ascending Hilbert key, breaking ties on
//!    the lower input index for determinism.
//!
//! ## Vendor decision
//!
//! Per spec § 4, the bit-twiddling Hilbert key computation is delegated
//! to two mature crates:
//!
//! * [`fast_hilbert`](https://crates.io/crates/fast_hilbert) for the 2D
//!   path (LUT-based, MIT, no transitive deps, `no_std` compatible).
//! * [`hilbert`](https://crates.io/crates/hilbert) for the general N-D
//!   path (Skilling's algorithm, MIT). This crate has heavier transitive
//!   deps (`num`, `rand`, `criterion`, `spectral`) — it is opt-in via
//!   the `permutation_hilbert` Cargo feature for that reason.
//!
//! The wrappers translate to/from our shared [`Permutation`] type and
//! handle the float → integer quantization that neither crate
//! performs.
//!
//! ## Determinism
//!
//! * Tie-breaks on equal Hilbert keys go to the lower input index.
//! * Empty inputs return `Permutation::identity(0)`.
//! * All-coincident inputs (zero-range on every axis) collapse to the
//!   identity permutation in original index order.
//! * The same input always produces the same permutation, independent
//!   of host architecture.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::Permutation;

/// Number of bits used per axis when quantizing to the integer grid.
///
/// 16 bits per axis is the canonical choice for both the 2D and N-D
/// paths:
///
/// * For the 2D wrapper (`fast_hilbert::xy2h::<u32>`), the resulting
///   Hilbert key fits in `u64` for any pair of `u16`-valued coordinates.
/// * For the N-D wrapper (`hilbert::Point::hilbert_transform`), 16 bits
///   per dimension keeps the working `BigUint` index compact while
///   preserving a 65 536 × 65 536 grid resolution per axis — well above
///   what F22 / PCA-projected fingerprints require for similarity-scan
///   locality.
const BITS_PER_AXIS: u32 = 16;

/// Maximum coordinate value on the quantized grid (inclusive).
const GRID_MAX: u32 = (1 << BITS_PER_AXIS) - 1;

/// Computes a Hilbert curve ordering for 2D points.
///
/// Quantizes each `(f32, f32)` to a 16-bits-per-axis integer grid using
/// the per-axis min/max of the input, then sorts by Hilbert key (ties
/// broken by lower input index).
///
/// Returns a [`Permutation`] of length `points.len()` where
/// `perm[old_id] = new_id` — i.e. element `i` of the input is moved to
/// position `perm.as_slice()[i]` in the Hilbert-ordered output.
///
/// # Edge cases
///
/// * `points.is_empty()` returns `Permutation::identity(0)`.
/// * `points.len() == 1` returns `Permutation::identity(1)`.
/// * If every coordinate value coincides (zero range on both axes),
///   the result is the identity permutation in original index order
///   (all points map to grid origin and tie-break on input index).
/// * `NaN` and infinities are clamped to `0` after the min/max sweep:
///   the routine treats them as the lower bound to keep quantization
///   deterministic. Callers that want strict NaN handling should
///   filter upstream.
///
/// # Panics
///
/// Does not panic. Quantization is saturating; the sort is total.
#[must_use]
pub fn hilbert_2d(points: &[(f32, f32)]) -> Permutation {
    let n = points.len();
    if n <= 1 {
        return Permutation::identity(n);
    }

    let (x_min, x_max) = axis_range(points.iter().map(|&(x, _)| x));
    let (y_min, y_max) = axis_range(points.iter().map(|&(_, y)| y));

    // Compute Hilbert key per point and tag with original index for
    // deterministic argsort.
    let mut keyed: Vec<(u64, u32)> = (0..n)
        .map(|i| {
            let (x, y) = points[i];
            let qx = quantize(x, x_min, x_max);
            let qy = quantize(y, y_min, y_max);
            // `fast_hilbert::xy2h::<u32>(_, _, order)` returns u64.
            // `order` is the curve order (number of bits per axis); we
            // standardize on 16 bits per axis.
            let key = fast_hilbert::xy2h::<u32>(qx, qy, BITS_PER_AXIS as u8);
            // SAFETY-LIKE: i < n <= u32::MAX as usize is enforced by
            // the n > 1 path above and the `usize -> u32` cast below
            // is bounded by `Permutation` length limits.
            assert!(
                i <= u32::MAX as usize,
                "hilbert_2d: input length exceeds u32 vertex space"
            );
            (key, i as u32)
        })
        .collect();

    // Sort by (key, original_index) for determinism on ties. The
    // tuple's natural Ord is exactly what we want: ascending key
    // first, ascending original input index second.
    keyed.sort_unstable();

    // `sorted[rank] == original_index` -> `perm[original_index] = rank`.
    let mut perm = vec![0_u32; n];
    for (rank, &(_, original_idx)) in keyed.iter().enumerate() {
        // `rank < n <= u32::MAX as usize` so the cast is safe.
        perm[original_idx as usize] = rank as u32;
    }
    Permutation::from_vec_unchecked(perm)
}

/// Computes a Hilbert curve ordering for N-dimensional points.
///
/// Each `points[i]` must have length `dim`. Quantizes every axis to a
/// 16-bits-per-axis integer grid using its own per-axis min/max, then
/// computes the Hilbert key per point via Skilling's algorithm and
/// argsorts (ties broken by lower input index).
///
/// Returns a [`Permutation`] of length `points.len()`.
///
/// # Edge cases
///
/// * `points.is_empty()` returns `Permutation::identity(0)`.
/// * `points.len() == 1` returns `Permutation::identity(1)`.
/// * If every coordinate value on a given axis coincides, that axis
///   collapses to grid origin without affecting the ordering on the
///   remaining axes.
///
/// # Panics
///
/// * Panics if any `points[i].len() != dim`.
/// * Panics if `dim == 0` and `points.len() > 1` — the Hilbert curve
///   is undefined for a zero-dimensional space with more than one
///   point. (Single-point and empty inputs short-circuit to identity
///   above.)
#[must_use]
pub fn hilbert_nd(points: &[Vec<f32>], dim: usize) -> Permutation {
    let n = points.len();
    if n <= 1 {
        return Permutation::identity(n);
    }
    assert!(
        dim > 0,
        "hilbert_nd: dim must be >= 1 when points.len() > 1"
    );
    for (idx, p) in points.iter().enumerate() {
        assert_eq!(
            p.len(),
            dim,
            "hilbert_nd: points[{idx}].len() ({}) != dim ({dim})",
            p.len()
        );
    }

    // Per-axis min/max sweep.
    let mut mins = Vec::with_capacity(dim);
    let mut maxs = Vec::with_capacity(dim);
    for axis in 0..dim {
        let (lo, hi) = axis_range(points.iter().map(|p| p[axis]));
        mins.push(lo);
        maxs.push(hi);
    }

    // Build `hilbert::Point` instances with quantized u32 coordinates,
    // each carrying its original input index as the point id.
    let mut quantized_coords: Vec<u32> = Vec::with_capacity(dim);
    let mut hpoints: Vec<hilbert::Point> = Vec::with_capacity(n);
    for (i, p) in points.iter().enumerate() {
        quantized_coords.clear();
        for axis in 0..dim {
            quantized_coords.push(quantize(p[axis], mins[axis], maxs[axis]));
        }
        assert!(
            i <= u32::MAX as usize,
            "hilbert_nd: input length exceeds u32 vertex space"
        );
        hpoints.push(hilbert::Point::new(i, &quantized_coords));
    }

    // Compute Hilbert indices alongside original ids. The `hilbert`
    // crate's `hilbert_transform` returns `BigUint`; we sort
    // `(BigUint, id)` pairs for determinism on ties.
    let mut keyed: Vec<(num::BigUint, u32)> = hpoints
        .iter()
        .map(|p| {
            let key = p.hilbert_transform(BITS_PER_AXIS as usize);
            // `p.get_id()` returned the original input index we set
            // above; cast back to `u32`.
            (key, p.get_id() as u32)
        })
        .collect();

    keyed.sort_unstable_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));

    let mut perm = vec![0_u32; n];
    for (rank, (_, original_idx)) in keyed.iter().enumerate() {
        perm[*original_idx as usize] = rank as u32;
    }
    Permutation::from_vec_unchecked(perm)
}

/// Returns the `(min, max)` of a finite-valued axis iterator.
///
/// `NaN` / infinities are skipped via `is_finite()`. If every value is
/// non-finite (or the iterator is empty), returns `(0.0, 0.0)`, which
/// causes `quantize` to map every point to grid origin.
fn axis_range<I: IntoIterator<Item = f32>>(iter: I) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for v in iter {
        if v.is_finite() {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
    }
    if lo.is_infinite() || hi.is_infinite() {
        // No finite values seen: degenerate axis. Treat as collapsed
        // to a single point so quantization yields 0 across the board.
        return (0.0, 0.0);
    }
    (lo, hi)
}

/// Quantizes `value` into the integer grid `0..=GRID_MAX` using the
/// per-axis `(min, max)` range.
///
/// Saturates non-finite inputs to `min`, and clamps the integer output
/// to `0..=GRID_MAX` defensively against floating-point rounding at the
/// upper boundary.
fn quantize(value: f32, min: f32, max: f32) -> u32 {
    if !value.is_finite() {
        return 0;
    }
    let span = max - min;
    if span <= 0.0 {
        // Degenerate axis (all values coincident or single-valued):
        // map every input to grid origin.
        return 0;
    }
    let normalized = ((value - min) / span).clamp(0.0, 1.0);
    // Multiply by GRID_MAX (not GRID_MAX + 1) so the maximum mapped
    // value is exactly GRID_MAX (no off-by-one out-of-grid case).
    let scaled = normalized * (GRID_MAX as f32);
    // Round-to-nearest then clamp; `as u32` truncates toward zero on
    // f32, so we add 0.5 first. The clamp handles any residual
    // floating-point drift.
    let rounded = (scaled + 0.5) as u32;
    rounded.min(GRID_MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts that `perm` is a valid permutation of `0..n`.
    fn assert_valid_permutation(perm: &Permutation, n: usize) {
        assert_eq!(perm.len(), n, "permutation length mismatch");
        let mut seen = vec![false; n];
        for &id in perm.as_slice() {
            let id = id as usize;
            assert!(id < n, "permutation contains id {id} >= n {n}");
            assert!(!seen[id], "permutation contains duplicate id {id}");
            seen[id] = true;
        }
        assert!(seen.iter().all(|b| *b), "permutation missing an id");
    }

    #[test]
    fn hilbert_2d_empty_returns_identity() {
        let perm = hilbert_2d(&[]);
        assert!(perm.is_empty());
    }

    #[test]
    fn hilbert_2d_single_point_returns_identity() {
        let perm = hilbert_2d(&[(0.5, 0.5)]);
        assert_eq!(perm.as_slice(), &[0_u32]);
    }

    #[test]
    fn hilbert_2d_two_points_produces_valid_permutation() {
        let perm = hilbert_2d(&[(0.0, 0.0), (1.0, 1.0)]);
        assert_valid_permutation(&perm, 2);
    }

    #[test]
    fn hilbert_2d_all_coincident_returns_identity_order() {
        // Every point at (0.5, 0.5): zero range on both axes. The
        // tie-break-by-input-index rule means rank == original index.
        let pts = vec![(0.5_f32, 0.5_f32); 8];
        let perm = hilbert_2d(&pts);
        assert_valid_permutation(&perm, pts.len());
        for (i, &rank) in perm.as_slice().iter().enumerate() {
            assert_eq!(
                rank as usize, i,
                "all-coincident input should preserve input order"
            );
        }
    }

    #[test]
    fn hilbert_2d_locality_property_small_grid() {
        // 4x4 grid of points spanning [0..3] x [0..3]. Build a
        // bounding box covering the upper-left 2x2 quadrant; verify
        // that those 4 points get Hilbert keys in a contiguous range
        // among the 16 ranks.
        let mut pts: Vec<(f32, f32)> = Vec::new();
        let mut bbox_indices: Vec<u32> = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                let idx = pts.len() as u32;
                pts.push((x as f32, y as f32));
                if x < 2 && y < 2 {
                    bbox_indices.push(idx);
                }
            }
        }
        let perm = hilbert_2d(&pts);
        assert_valid_permutation(&perm, pts.len());

        // Look up the rank assigned to each bbox point.
        let p = perm.as_slice();
        let mut bbox_ranks: Vec<u32> = bbox_indices.iter().map(|&i| p[i as usize]).collect();
        bbox_ranks.sort_unstable();
        // For a 4x4 grid covered by a Hilbert curve, the 2x2 corner
        // quadrant lands in a contiguous run of 4 consecutive ranks
        // by construction of the curve. The "contiguous run" check is
        // the canonical locality assertion.
        let span = bbox_ranks[3] - bbox_ranks[0];
        assert_eq!(
            span, 3,
            "expected 4 bbox points in a contiguous rank run; got ranks {bbox_ranks:?}"
        );
    }

    #[test]
    fn hilbert_2d_round_trip_via_apply_and_inverse() {
        // Random-ish 2D points; verify apply + inverse round-trips
        // arbitrary payload data.
        let pts: Vec<(f32, f32)> = (0..16_u32)
            .map(|i| {
                let x = ((i * 7919) % 32) as f32 / 31.0;
                let y = ((i * 6781) % 32) as f32 / 31.0;
                (x, y)
            })
            .collect();
        let perm = hilbert_2d(&pts);
        let n = pts.len();
        assert_valid_permutation(&perm, n);

        let payload: Vec<u32> = (100..(100 + n as u32)).collect();
        let permuted = perm.apply(&payload);
        let inv = perm.inverse();
        let recovered = inv.apply(&permuted);
        assert_eq!(recovered, payload);
    }

    #[test]
    fn hilbert_2d_determinism_same_input_same_output() {
        let pts: Vec<(f32, f32)> = (0..64_u32)
            .map(|i| {
                let x = ((i * 1103) % 1024) as f32 / 1023.0;
                let y = ((i * 9133) % 1024) as f32 / 1023.0;
                (x, y)
            })
            .collect();
        let p1 = hilbert_2d(&pts);
        let p2 = hilbert_2d(&pts);
        let p3 = hilbert_2d(&pts);
        assert_eq!(p1, p2);
        assert_eq!(p2, p3);
    }

    #[test]
    fn hilbert_2d_very_wide_range_quantizes_safely() {
        // Span includes the f32 limits we care about for build-time
        // pipelines: from -1e6 to +1e6, plus an outlier near zero.
        let pts: Vec<(f32, f32)> = vec![
            (-1.0e6, -1.0e6),
            (0.0, 0.0),
            (1.0e6, 1.0e6),
            (-1.0e6, 1.0e6),
            (1.0e6, -1.0e6),
        ];
        let perm = hilbert_2d(&pts);
        assert_valid_permutation(&perm, pts.len());
    }

    #[test]
    fn hilbert_2d_non_finite_values_do_not_panic() {
        // NaN / infinity inputs are handled (mapped to grid origin)
        // without panicking; the resulting ordering is still a valid
        // permutation, although locality of the bad points is lost.
        let pts = vec![
            (0.0_f32, 0.0_f32),
            (f32::NAN, 1.0),
            (1.0, f32::INFINITY),
            (-1.0, f32::NEG_INFINITY),
            (2.0, 2.0),
        ];
        let perm = hilbert_2d(&pts);
        assert_valid_permutation(&perm, pts.len());
    }

    #[test]
    fn hilbert_nd_empty_returns_identity() {
        let pts: Vec<Vec<f32>> = Vec::new();
        let perm = hilbert_nd(&pts, 3);
        assert!(perm.is_empty());
    }

    #[test]
    fn hilbert_nd_single_point_returns_identity() {
        let pts = vec![vec![0.1_f32, 0.2, 0.3]];
        let perm = hilbert_nd(&pts, 3);
        assert_eq!(perm.as_slice(), &[0_u32]);
    }

    #[test]
    fn hilbert_nd_3d_produces_valid_permutation() {
        // 3D grid: 3x3x3 = 27 points.
        let mut pts: Vec<Vec<f32>> = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    pts.push(vec![x as f32, y as f32, z as f32]);
                }
            }
        }
        let perm = hilbert_nd(&pts, 3);
        assert_valid_permutation(&perm, pts.len());
    }

    #[test]
    fn hilbert_nd_4d_round_trips_payload() {
        let pts: Vec<Vec<f32>> = (0..16_u32)
            .map(|i| {
                vec![
                    ((i * 17) % 64) as f32 / 63.0,
                    ((i * 23) % 64) as f32 / 63.0,
                    ((i * 41) % 64) as f32 / 63.0,
                    ((i * 53) % 64) as f32 / 63.0,
                ]
            })
            .collect();
        let perm = hilbert_nd(&pts, 4);
        let n = pts.len();
        assert_valid_permutation(&perm, n);

        let payload: Vec<i64> = (0..n as i64).map(|i| i * 1000).collect();
        let permuted = perm.apply(&payload);
        let inv = perm.inverse();
        let recovered = inv.apply(&permuted);
        assert_eq!(recovered, payload);
    }

    #[test]
    fn hilbert_nd_determinism_same_input_same_output() {
        let pts: Vec<Vec<f32>> = (0..32_u32)
            .map(|i| {
                vec![
                    ((i * 7) % 50) as f32 / 49.0,
                    ((i * 11) % 50) as f32 / 49.0,
                    ((i * 13) % 50) as f32 / 49.0,
                ]
            })
            .collect();
        let p1 = hilbert_nd(&pts, 3);
        let p2 = hilbert_nd(&pts, 3);
        assert_eq!(p1, p2);
    }

    #[test]
    #[should_panic(expected = "dim must be >= 1")]
    fn hilbert_nd_zero_dim_with_multiple_points_panics() {
        let pts = vec![vec![], vec![]];
        let _ = hilbert_nd(&pts, 0);
    }

    #[test]
    #[should_panic(expected = "!= dim")]
    fn hilbert_nd_dim_mismatch_panics() {
        let pts = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        let _ = hilbert_nd(&pts, 3);
    }

    #[test]
    fn hilbert_nd_all_coincident_returns_identity_order() {
        let pts: Vec<Vec<f32>> = (0..6).map(|_| vec![0.5_f32, 0.5, 0.5]).collect();
        let perm = hilbert_nd(&pts, 3);
        assert_valid_permutation(&perm, pts.len());
        for (i, &rank) in perm.as_slice().iter().enumerate() {
            assert_eq!(rank as usize, i);
        }
    }
}

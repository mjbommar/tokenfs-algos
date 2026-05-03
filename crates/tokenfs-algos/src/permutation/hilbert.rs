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
//! Per spec § 4 (updated post audit-R5 finding #158):
//!
//! * The 2D path delegates to [`fast_hilbert`](https://crates.io/crates/fast_hilbert)
//!   (LUT-based, MIT, no transitive deps, `no_std` compatible).
//! * The N-D path is implemented in this module from Skilling's 2004
//!   algorithm directly. The previous `hilbert` 0.1 dependency was
//!   replaced because it transitively pulled `criterion 0.3` →
//!   `atty 0.2` (RUSTSEC-2021-0145) and `spectral 0.6` → `num 0.1` →
//!   `rustc-serialize 0.3` (RUSTSEC-2022-0004) into the dep tree
//!   whenever the `permutation_hilbert` feature was enabled.
//!
//! The wrappers translate to/from our shared [`Permutation`] type and
//! handle the float → integer quantization that neither path performs
//! on its own.
//!
//! ## Determinism
//!
//! * Tie-breaks on equal Hilbert keys go to the lower input index.
//! * Empty inputs return `Permutation::try_identity(0).expect("identity construction within u32::MAX")`.
//! * All-coincident inputs (zero-range on every axis) collapse to the
//!   identity permutation in original index order.
//! * The same input always produces the same permutation, independent
//!   of host architecture.

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec;
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
/// * For the N-D wrapper ([`skilling_hilbert_key`]), 16 bits per
///   dimension keeps the working integer compact while preserving a
///   65 536 × 65 536 grid resolution per axis — well above what F22 /
///   PCA-projected fingerprints require for similarity-scan locality.
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
/// * `points.is_empty()` returns `Permutation::try_identity(0).expect("identity construction within u32::MAX")`.
/// * `points.len() == 1` returns `Permutation::try_identity(1).expect("identity construction within u32::MAX")`.
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
        return Permutation::try_identity(n).expect("identity construction within u32::MAX");
    }
    // Saturate at u32::MAX entries — vertex IDs are u32, and input
    // sequences longer than that cannot be a valid Permutation. We
    // truncate-and-identity rather than panic (audit-R10 #1 / #216 +
    // matches the docstring's "does not panic" guarantee).
    if n > u32::MAX as usize {
        return Permutation::try_identity(u32::MAX as usize)
            .expect("identity construction within u32::MAX");
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
            // SAFETY (cast): i < n <= u32::MAX guaranteed by the
            // truncate-and-identity guard above.
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
    // SAFETY: `keyed` was built by mapping each input index `0..n` to
    // exactly one entry, then sorted; the loop assigns each `original_idx`
    // (covering `0..n` exactly once) the unique `rank` in `0..n`. Hence
    // `perm` is a bijection on `0..n`, and `n <= u32::MAX as usize` is
    // enforced by the asserts above.
    unsafe { Permutation::from_vec_unchecked(perm) }
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
/// * `points.is_empty()` returns `Permutation::try_identity(0).expect("identity construction within u32::MAX")`.
/// * `points.len() == 1` returns `Permutation::try_identity(1).expect("identity construction within u32::MAX")`.
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
/// * Panics if any `points[i].len() != dim`.
///
/// Available only with `feature = "userspace"` (audit-R10 #1 / #216).
/// Kernel/FUSE callers needing Hilbert ordering should validate
/// `dim > 0` and per-row `len == dim` upfront before calling.
#[cfg(feature = "userspace")]
#[must_use]
pub fn hilbert_nd(points: &[Vec<f32>], dim: usize) -> Permutation {
    let n = points.len();
    if n <= 1 {
        return Permutation::try_identity(n).expect("identity construction within u32::MAX");
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

    // Quantize and compute Skilling Hilbert keys. The keys are
    // big-endian byte vectors so we can lex-compare them directly.
    let mut quantized_coords: Vec<u32> = Vec::with_capacity(dim);
    let mut keyed: Vec<(Vec<u8>, u32)> = Vec::with_capacity(n);
    for (i, p) in points.iter().enumerate() {
        quantized_coords.clear();
        for axis in 0..dim {
            quantized_coords.push(quantize(p[axis], mins[axis], maxs[axis]));
        }
        assert!(
            i <= u32::MAX as usize,
            "hilbert_nd: input length exceeds u32 vertex space"
        );
        let key = skilling_hilbert_key(&quantized_coords, BITS_PER_AXIS as usize);
        keyed.push((key, i as u32));
    }

    // Sort by (key, original_index) using natural lex ordering on the
    // big-endian byte vector. Ties on equal keys break on lower input
    // index for determinism.
    keyed.sort_unstable_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));

    let mut perm = vec![0_u32; n];
    for (rank, (_, original_idx)) in keyed.iter().enumerate() {
        perm[*original_idx as usize] = rank as u32;
    }
    // SAFETY: `keyed` was built by mapping each input index `0..n` to
    // exactly one entry, then sorted; the loop assigns each `original_idx`
    // (covering `0..n` exactly once) the unique `rank` in `0..n`. Hence
    // `perm` is a bijection on `0..n`, and `n <= u32::MAX as usize` is
    // enforced by the assert in the build loop above.
    unsafe { Permutation::from_vec_unchecked(perm) }
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

/// Computes the Hilbert key for an N-dimensional integer point using
/// Skilling's 2004 algorithm, returning the key as a big-endian
/// byte vector suitable for direct lexicographic comparison.
///
/// `coords` is the N-dimensional point with each coordinate already
/// quantized to a non-negative integer in `0..(1 << bits)`. `bits` is
/// the number of bits of precision per coordinate (≤ 32). The returned
/// byte vector contains exactly `ceil(bits * dim / 8)` bytes; the
/// high-order pad bits (when `bits * dim` is not a multiple of 8) are
/// zero, so two equally-sized keys can be compared with the standard
/// lexicographic `Vec<u8>` ordering.
///
/// # Algorithm
///
/// Three stages, faithful to Skilling, J., "Programming the Hilbert
/// curve", AIP Conf. Proc. 707 (2004), and matched bit-exactly against
/// the published `hilbert` 0.1.2 crate's `hilbert_index_transposed` +
/// `interleave_be` pipeline (the upstream ancestor of this routine):
///
/// 1. **Inverse undo / Skilling rotation**: walk `q = 1<<(bits-1)`
///    down to `q == 1`. At each level, for every coordinate `i`, if
///    bit `q` of `x[i]` is set then invert the lower `q-1` bits of
///    `x[0]`; else swap the lower `q-1` bits of `x[0]` and `x[i]`.
/// 2. **Gray encode**: prefix-XOR the coordinates so `x[i] ^= x[i-1]`
///    for `i = 1..n`, then compute a per-position correction `t` from
///    the high bits of `x[n-1]` and XOR it back into every coordinate.
/// 3. **Big-endian bit interleave**: walk the bits high → low. For
///    each bit position, read one bit from each coordinate in turn
///    (coord 0 first), packing into the output byte vector starting
///    at the high bit of byte 0 (after `pad_bits` of leading zeros).
///
/// # Edge cases
///
/// * `coords.len() < 2`: returns a key built from a 1D Skilling step
///   that effectively just truncates to the requested bit width. The
///   outer wrappers short-circuit single-point inputs before reaching
///   this point, so this branch is just for defensive use.
/// * `bits == 0`: returns an empty byte vector.
///
/// # Panics
///
/// * Panics if `bits > 32` (each coordinate is `u32`).
fn skilling_hilbert_key(coords: &[u32], bits: usize) -> Vec<u8> {
    assert!(bits <= 32, "skilling_hilbert_key: bits ({bits}) > 32");
    if bits == 0 {
        return Vec::new();
    }
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }

    let mut x = coords.to_vec();

    // Stage 1: Skilling "inverse undo" / coordinate-rotation step.
    //
    // Translates the cube-corner traversal order at each octave (bit
    // level) by either flipping the low bits of x[0] (if x[i] has the
    // current high bit set) or swapping the low bits between x[0] and
    // x[i] otherwise. The body matches `hilbert::transform::
    // hilbert_index_transposed` line-for-line, with the X[0] hoist
    // collapsed back into array form for clarity (the hoist is a
    // performance micro-opt; the bit-twiddle is identical).
    if bits >= 1 {
        let mut q: u32 = 1u32 << (bits - 1);
        while q > 1 {
            let p = q - 1;
            // First iteration: i == 0. Only the inversion branch is
            // reachable since the swap with self is a no-op.
            if x[0] & q != 0 {
                x[0] ^= p;
            }
            for i in 1..n {
                let xi = x[i];
                if xi & q != 0 {
                    x[0] ^= p;
                } else {
                    let t = (x[0] ^ xi) & p;
                    x[0] ^= t;
                    x[i] = xi ^ t;
                }
            }
            q >>= 1;
        }
    }

    // Stage 2a: Gray encode by prefix XOR (x[i] ^= x[i-1]).
    for i in 1..n {
        x[i] ^= x[i - 1];
    }

    // Stage 2b: Compute per-position correction `t` from the high bits
    // of x[n-1] and apply it across every coordinate.
    let mut t: u32 = 0;
    if bits >= 1 {
        let mut q: u32 = 1u32 << (bits - 1);
        while q > 1 {
            if x[n - 1] & q != 0 {
                t ^= q - 1;
            }
            q >>= 1;
        }
    }
    for xi in &mut x {
        *xi ^= t;
    }

    // Stage 3: Big-endian bit interleave into the output byte vector.
    interleave_be(&x, bits)
}

/// Interleaves the low `bits` of each coordinate into a big-endian
/// byte vector.
///
/// This is the byte-layout half of [`skilling_hilbert_key`]: the
/// transposed Skilling output `x[0..n]` is folded into a single
/// `bits * n`-bit number stored MSB-first.
///
/// Layout: the output is a `Vec<u8>` of length `ceil(bits*n / 8)`.
/// The high-order bit of `x[0]` becomes the highest non-pad bit of
/// `byte_vector[0]`. The next-highest output bit takes the high bit
/// of `x[1]`, then `x[2]`, etc., wrapping across all coordinates per
/// "octave". Any leftover high bits of `byte_vector[0]` (when
/// `bits*n` is not a multiple of 8) are zero — so equally-sized
/// keys can be `Vec<u8>`-compared lexicographically.
///
/// Matches `hilbert::transform::fast_hilbert::interleave_be` from the
/// upstream `hilbert` 0.1.2 crate bit-exactly; the parity test
/// `skilling_hilbert_key_matches_upstream_fixture` enforces this on a
/// 22-case fixture so the dropped dep can never silently regress.
fn interleave_be(coords: &[u32], bits: usize) -> Vec<u8> {
    let dim = coords.len();
    if dim == 0 || bits == 0 {
        return Vec::new();
    }
    let num_bits = dim * bits;
    let bytes_needed = num_bits.div_ceil(8);
    let mut byte_vector = vec![0_u8; bytes_needed];
    let pad_bits = bytes_needed * 8 - num_bits;

    for i_bit in 0..num_bits {
        // Round-robin across coords: bit 0 from coord 0, bit 1 from
        // coord 1, ..., bit dim from coord 0 again, etc.
        let i_from_uint_vector = i_bit % dim;
        // Each "octave" steps from the high bit of the input down to
        // the low bit.
        let i_from_uint_bit = bits - (i_bit / dim) - 1;
        // Skip past the leading pad bits when writing into the
        // output byte vector.
        let i_to_byte_vector = (i_bit + pad_bits) >> 3;
        let i_to_byte_bit = 0x7 - ((i_bit + pad_bits) & 0x7);

        let bit: u8 =
            (((coords[i_from_uint_vector] >> i_from_uint_bit) & 1_u32) << i_to_byte_bit) as u8;
        byte_vector[i_to_byte_vector] |= bit;
    }
    byte_vector
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
        let permuted = perm.try_apply(&payload).expect("apply: shape match");
        let inv = perm.inverse();
        let recovered = inv.try_apply(&permuted).expect("apply: shape match");
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
        let permuted = perm.try_apply(&payload).expect("apply: shape match");
        let inv = perm.inverse();
        let recovered = inv.try_apply(&permuted).expect("apply: shape match");
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

    /// Bit-exact parity fixture vs. the published `hilbert` 0.1.2
    /// crate's `Point::hilbert_transform` output, captured one-shot
    /// by running the upstream crate against the same coordinates and
    /// recording the resulting `BigUint::to_bytes_be()` representation
    /// here. See R5 finding #158 for context: the dep was dropped to
    /// avoid the upstream's vulnerable transitive deps, but the
    /// algorithm output is preserved bit-for-bit.
    ///
    /// Each row is `(bits, coords, expected_be_bytes)`. The fixture
    /// covers `dim ∈ {3, 4, 5}` at multiple bit depths, edge values
    /// (origin, axis maxima, midpoints), and a handful of arbitrary
    /// interior points.
    #[test]
    fn skilling_hilbert_key_matches_upstream_fixture() {
        // (bits_per_coord, coords, expected big-endian key bytes)
        let cases: &[(usize, &[u32], &[u8])] = &[
            // dim=3 cases at 16 bits/coord (the production setting).
            (16, &[0, 0, 0], &[0x00]),
            (
                16,
                &[65535, 65535, 65535],
                &[0xb6, 0xdb, 0x6d, 0xb6, 0xdb, 0x6d],
            ),
            (
                16,
                &[32768, 32768, 32768],
                &[0xa0, 0x00, 0x00, 0x00, 0x00, 0x00],
            ),
            (16, &[1, 2, 3], &[0x24]),
            (16, &[100, 200, 300], &[0x07, 0x68, 0x28, 0x12]),
            (
                16,
                &[60000, 100, 60000],
                &[0xdb, 0x4c, 0xb2, 0x5f, 0xa4, 0x76],
            ),
            // dim=4 cases at 16 bits/coord.
            (16, &[0, 0, 0, 0], &[0x00]),
            (16, &[10, 20, 30, 40], &[0x70, 0xbe, 0xcc]),
            (
                16,
                &[65535, 0, 65535, 0],
                &[0xc2, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22, 0x22],
            ),
            (
                16,
                &[12345, 23456, 34567, 45678],
                &[0x2f, 0x43, 0x13, 0x32, 0x15, 0x3f, 0xe2, 0x20],
            ),
            // dim=5 cases at 16 bits/coord.
            (16, &[0, 0, 0, 0, 0], &[0x00]),
            (16, &[1, 2, 3, 4, 5], &[0x23, 0x93]),
            (
                16,
                &[100, 200, 300, 400, 500],
                &[0x11, 0xbc, 0x63, 0x40, 0x2c, 0x42],
            ),
            (
                16,
                &[32768, 32768, 32768, 32768, 32768],
                &[0xa8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
            ),
            // Smaller-bit-depth cases for cross-check at sub-16 bits.
            (5, &[0, 0, 0], &[0x00]),
            (5, &[1, 2, 3], &[0x16]),
            (5, &[31, 31, 31], &[0x5b, 0x6d]),
            (5, &[10, 20, 30], &[0x2c, 0xb0]),
            (4, &[1, 2, 3, 4], &[0x0f, 0x64]),
            (4, &[15, 15, 15, 15], &[0xaa, 0xaa]),
            (3, &[0, 1, 2, 3, 4], &[0x07, 0x80]),
            (3, &[7, 7, 7, 7, 7], &[0x56, 0xb5]),
        ];

        for (i, (bits, coords, expected)) in cases.iter().enumerate() {
            let actual = skilling_hilbert_key(coords, *bits);
            // The upstream `BigUint::to_bytes_be()` output trims
            // leading zero bytes, but our `skilling_hilbert_key`
            // returns a fixed-width buffer of `ceil(bits*dim / 8)`
            // bytes (so equally-sized keys lex-compare correctly).
            // Match by stripping leading zeros from our output before
            // comparing; the comparison is still bit-exact.
            let mut trimmed: &[u8] = actual.as_slice();
            while trimmed.len() > 1 && trimmed[0] == 0 {
                trimmed = &trimmed[1..];
            }
            assert_eq!(
                trimmed,
                *expected,
                "case #{i}: dim={}, bits={}, coords={:?} -> got {:02x?}, expected {:02x?}",
                coords.len(),
                bits,
                coords,
                actual,
                expected
            );
        }
    }

    /// Verifies that two equally-sized Hilbert keys produced by
    /// `skilling_hilbert_key` lex-compare in the same order as the
    /// notional integer Hilbert distances they represent.
    ///
    /// This guards against a regression where the byte-vector key
    /// lexicographic ordering diverges from the Hilbert-distance
    /// ordering — which could happen if [`interleave_be`] silently
    /// produced different-length outputs for inputs with the same
    /// `bits * dim` budget.
    #[test]
    fn skilling_hilbert_key_byte_order_matches_distance_order() {
        // 3-bit, 2-D Hilbert curve has 64 cells; the expected
        // sequence of (x, y) coordinates was generated from the
        // published `hilbert` crate at index = 0..63.
        let bits = 3_usize;
        let dim = 2_usize;
        let n = 1_usize << (bits * dim);

        // For 2D specifically, fast_hilbert::xy2h gives us the
        // reference Hilbert distance per (x, y). Use it to rank a
        // shuffled set of points and verify that
        // skilling_hilbert_key produces a byte ordering consistent
        // with that ranking.
        let mut pts: Vec<(u32, u32)> = Vec::new();
        for y in 0..(1u32 << bits) {
            for x in 0..(1u32 << bits) {
                pts.push((x, y));
            }
        }
        // Compute (skilling_key, fast_hilbert_distance, idx).
        let triples: Vec<(Vec<u8>, u64, usize)> = pts
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| {
                let k = skilling_hilbert_key(&[x, y], bits);
                let d = fast_hilbert::xy2h::<u32>(x, y, bits as u8);
                (k, d, i)
            })
            .collect();

        // All keys at the same (bits, dim) must be the same length.
        let key_len = triples[0].0.len();
        for (k, _, _) in &triples {
            assert_eq!(k.len(), key_len, "Skilling keys must be same-width");
        }

        // Sort by Skilling key; the resulting ranks must match the
        // ranks one would get from sorting by fast_hilbert distance.
        let mut by_skilling = triples.clone();
        by_skilling.sort_by(|a, b| (&a.0, a.2).cmp(&(&b.0, b.2)));
        let mut by_distance = triples;
        by_distance.sort_by_key(|t| (t.1, t.2));

        let skilling_ranks: Vec<usize> = by_skilling.iter().map(|t| t.2).collect();
        let distance_ranks: Vec<usize> = by_distance.iter().map(|t| t.2).collect();
        assert_eq!(
            skilling_ranks, distance_ranks,
            "Skilling key byte order diverges from Hilbert-distance order on a {n}-cell grid"
        );
    }
}

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Computes the per-neighbour modularity-gain score for one
/// fixed community `u`.
///
/// Returns a vector of `i128` scores, one per neighbour,
/// satisfying:
///
/// ```text
/// score[i] = i128(m_doubled) * i128(neighbor_weights[i])
///          - i128(self_degree) * i128(neighbor_degrees[i])
/// ```
///
/// `m_doubled` is `u128` because `2 * m` can exceed `u64::MAX`
/// when `m` itself approaches `u64::MAX / 2`. The intermediate
/// products and the final difference fit in `i128` for any
/// `u64`-sized inputs because `u64 * u64` fits in `u128`, and
/// the difference of two non-negative `u128` values fits in
/// `i128` when each operand is at most `i128::MAX`.
///
/// # Panics
///
/// Panics if `neighbor_weights.len() != neighbor_degrees.len()`.
///
/// Also panics if `m_doubled` overflows `i128` when cast (only
/// possible if `m_doubled > i128::MAX`, which would require
/// `total_edge_weight > i128::MAX / 2 ≈ 2^126`, unreachable for
/// any realistic graph).
///
/// Available only with `feature = "userspace"` (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
#[must_use]
pub fn modularity_gains_neighbor_batch(
    neighbor_weights: &[u64],
    neighbor_degrees: &[u64],
    self_degree: u64,
    m_doubled: u128,
) -> Vec<i128> {
    assert_eq!(
        neighbor_weights.len(),
        neighbor_degrees.len(),
        "scalar::modularity_gains_neighbor_batch: neighbor_weights.len() ({}) != neighbor_degrees.len() ({})",
        neighbor_weights.len(),
        neighbor_degrees.len()
    );
    modularity_gains_neighbor_batch_inner(
        neighbor_weights,
        neighbor_degrees,
        self_degree,
        m_doubled,
    )
}

/// Unchecked variant of [`modularity_gains_neighbor_batch`].
///
/// # Safety
///
/// Caller must ensure
/// `neighbor_weights.len() == neighbor_degrees.len()` and
/// `m_doubled <= i128::MAX`. (The second invariant always holds for
/// realistic graphs; the assert in the userspace-gated entry exists
/// as a defensive check.)
#[must_use]
pub fn modularity_gains_neighbor_batch_unchecked(
    neighbor_weights: &[u64],
    neighbor_degrees: &[u64],
    self_degree: u64,
    m_doubled: u128,
) -> Vec<i128> {
    modularity_gains_neighbor_batch_inner(
        neighbor_weights,
        neighbor_degrees,
        self_degree,
        m_doubled,
    )
}

#[inline]
fn modularity_gains_neighbor_batch_inner(
    neighbor_weights: &[u64],
    neighbor_degrees: &[u64],
    self_degree: u64,
    m_doubled: u128,
) -> Vec<i128> {
    let two_m =
        i128::try_from(m_doubled).expect("m_doubled exceeds i128::MAX: total edge weight > 2^126");
    let deg_u = i128::from(self_degree);

    let mut out = Vec::with_capacity(neighbor_weights.len());
    for (&w, &deg_v) in neighbor_weights.iter().zip(neighbor_degrees) {
        let w_i = i128::from(w);
        let deg_v_i = i128::from(deg_v);
        out.push(two_m * w_i - deg_u * deg_v_i);
    }
    out
}

/// Returns true when the i64 fast path is eligible for the
/// given inputs.
///
/// The predicate is `m_doubled < 2^31 && self_degree < 2^31 &&
/// max(neighbor_weights) < 2^31 && max(neighbor_degrees) <
/// 2^31`. Under that bound both products `m_doubled * w` and
/// `self_degree * deg` fit in `i63 ⊂ i64` (since each operand
/// is at most `2^31 - 1`, the product is at most
/// `(2^31 - 1)^2 < 2^62`), and the SIMD backends evaluate the
/// score
///
/// ```text
/// score = 2 * m * w - deg(u) * deg(v)
/// ```
///
/// in `i64` lanes without overflow. The result still widens
/// cleanly to `i128` for the API return type.
///
/// The bound is intentionally conservative: AVX2's
/// `_mm256_mul_epu32` produces an unsigned 64-bit product
/// that we reinterpret as `i64` for the lane-wise subtraction.
/// Reinterpreting a `u64 > i64::MAX` as `i64` would silently
/// flip its sign, so we cap inputs at `2^31` to keep the
/// product comfortably inside `i64::MAX`.
///
/// Public so external callers (benches, `dispatch::planner`)
/// can pre-classify their inputs without re-deriving the bound.
#[must_use]
pub fn fast_path_eligible(
    neighbor_weights: &[u64],
    neighbor_degrees: &[u64],
    self_degree: u64,
    m_doubled: u128,
) -> bool {
    const BOUND: u64 = 1_u64 << 31;
    if m_doubled >= u128::from(BOUND) {
        return false;
    }
    if self_degree >= BOUND {
        return false;
    }
    if neighbor_weights.iter().any(|&w| w >= BOUND) {
        return false;
    }
    if neighbor_degrees.iter().any(|&d| d >= BOUND) {
        return false;
    }
    true
}

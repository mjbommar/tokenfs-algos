//! Portable scalar dense-distance kernels.
//!
//! These are the pinned reference implementations. SIMD backends must
//! produce identical results for integer kernels and stay within the
//! documented Higham §3 / Wilkinson L1-norm tolerance for floating-point
//! kernels (see `docs/v0.2_planning/13_VECTOR.md` § 5).
//!
//! The reduction order for floating-point kernels is **strictly
//! left-to-right** — pinning the order is part of the contract so callers
//! who care about reproducibility can compare against a deterministic
//! oracle. SIMD backends use a different (pairwise-tree) reduction order
//! that is documented per-backend.
//!
//! `u32` integer kernels accumulate into `u64` to absorb pairwise
//! products of typical embedding magnitudes; see [`dot_u32`] for the
//! overflow regime and the [`try_dot_u32`] / [`try_l1_u32`] /
//! [`try_l2_squared_u32`] checked counterparts.

/// Inner product of two `u32` vectors. Returns `None` on length mismatch.
///
/// # Overflow regime
///
/// Each pairwise term is at most `u32::MAX * u32::MAX ≈ 2^64`,
/// which can saturate `u64` accumulation in a single pair when
/// both inputs are `u32::MAX`. The accumulator uses
/// `wrapping_add`, so adversarial inputs near `u32::MAX` can wrap
/// silently. **For security-relevant decisions on caller-controlled
/// data, use [`try_dot_u32`] instead — it returns
/// `Some(None)` on overflow.** The wrapping variant is kept as the
/// fast path for non-adversarial callers; SIMD parity tests pin the
/// same wrap behaviour.
///
/// Practical safe regime: any vector of length `n` whose elements
/// average `m` accumulates at most `n * m^2`. For typical
/// embedding-style values (≤ 2^16 per element, ≤ 2^16 elements)
/// the sum stays well below `u64::MAX`.
#[must_use]
pub fn dot_u32(a: &[u32], b: &[u32]) -> Option<u64> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        sum = sum.wrapping_add(u64::from(x) * u64::from(y));
    }
    Some(sum)
}

/// Overflow-checked counterpart of [`dot_u32`].
///
/// Returns `None` on length mismatch; returns `Some(None)` if the
/// accumulator would overflow `u64` at any point during the
/// reduction; otherwise returns `Some(Some(sum))`.
#[must_use]
pub fn try_dot_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        let term = u64::from(x).checked_mul(u64::from(y))?;
        sum = match sum.checked_add(term) {
            Some(v) => v,
            None => return Some(None),
        };
    }
    Some(Some(sum))
}

/// Manhattan / L1 distance of two `u32` vectors.
///
/// # Overflow regime
///
/// Each pairwise term is at most `u32::MAX = 2^32 - 1`. The `u64`
/// accumulator handles at most `2^32` such terms before potential
/// overflow; vectors longer than that with adversarial inputs can
/// wrap silently. Use [`try_l1_u32`] when correctness of an
/// adversarial-input result matters.
#[must_use]
pub fn l1_u32(a: &[u32], b: &[u32]) -> Option<u64> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        sum = sum.wrapping_add(u64::from(x.abs_diff(y)));
    }
    Some(sum)
}

/// Overflow-checked counterpart of [`l1_u32`].
///
/// Returns `None` on length mismatch; `Some(None)` on accumulator
/// overflow; `Some(Some(sum))` otherwise.
#[must_use]
pub fn try_l1_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        sum = match sum.checked_add(u64::from(x.abs_diff(y))) {
            Some(v) => v,
            None => return Some(None),
        };
    }
    Some(Some(sum))
}

/// Squared L2 distance of two `u32` vectors.
///
/// # Overflow regime
///
/// Each pairwise term is at most `(u32::MAX)^2 ≈ 2^64`. With
/// adversarial inputs a single pair already saturates the `u64`
/// accumulator and any further term wraps. Use [`try_l2_squared_u32`]
/// when correctness on adversarial inputs matters; non-adversarial
/// callers (typical embedding distances) stay well below the cap.
#[must_use]
pub fn l2_squared_u32(a: &[u32], b: &[u32]) -> Option<u64> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = u64::from(x.abs_diff(y));
        sum = sum.wrapping_add(d * d);
    }
    Some(sum)
}

/// Overflow-checked counterpart of [`l2_squared_u32`].
///
/// Returns `None` on length mismatch; `Some(None)` on overflow;
/// `Some(Some(sum))` otherwise.
#[must_use]
pub fn try_l2_squared_u32(a: &[u32], b: &[u32]) -> Option<Option<u64>> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = u64::from(x.abs_diff(y));
        let term = d.checked_mul(d)?;
        sum = match sum.checked_add(term) {
            Some(v) => v,
            None => return Some(None),
        };
    }
    Some(Some(sum))
}

/// Cosine similarity of two `u32` vectors as `f64` in `[-1, 1]` (in
/// practice `[0, 1]` for non-negative count vectors).
///
/// Returns `None` on length mismatch. Returns `Some(0.0)` when either
/// vector has zero norm — this is the convention adopted by the existing
/// [`crate::divergence::cosine_distance_counts_u32`] (which then maps to
/// distance = 1.0).
#[must_use]
pub fn cosine_similarity_u32(a: &[u32], b: &[u32]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let mut dot = 0_u64;
    let mut norm_a = 0_u64;
    let mut norm_b = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        dot = dot.wrapping_add(u64::from(x) * u64::from(y));
        norm_a = norm_a.wrapping_add(u64::from(x) * u64::from(x));
        norm_b = norm_b.wrapping_add(u64::from(y) * u64::from(y));
    }
    if norm_a == 0 || norm_b == 0 {
        return Some(0.0);
    }
    let denom = crate::math::sqrt_f64((norm_a as f64) * (norm_b as f64));
    Some((dot as f64) / denom)
}

/// Inner product of two `f32` vectors.
///
/// # Numerics
///
/// Reduction order is strictly left-to-right (`sum += a[i] * b[i]`).
/// SIMD backends use a different (pairwise-tree) reduction order; the
/// difference is bounded by the Higham §3 model: roughly `n * eps *
/// sum(|a_i * b_i|)`. See `docs/v0.2_planning/13_VECTOR.md` § 5.
#[must_use]
pub fn dot_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0.0_f32;
    for (&x, &y) in a.iter().zip(b) {
        sum += x * y;
    }
    Some(sum)
}

/// Squared L2 distance of two `f32` vectors.
///
/// # Numerics
///
/// Reduction order is strictly left-to-right. All squared terms are
/// non-negative so cancellation is impossible; SIMD parity is much
/// tighter than for [`dot_f32`].
#[must_use]
pub fn l2_squared_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0.0_f32;
    for (&x, &y) in a.iter().zip(b) {
        let d = x - y;
        sum += d * d;
    }
    Some(sum)
}

/// Cosine similarity of two `f32` vectors.
///
/// Returns `None` on length mismatch and `Some(0.0)` when either vector
/// has zero norm.
#[must_use]
pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (&x, &y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return Some(0.0);
    }
    Some(dot / crate::math::sqrt_f32(norm_a * norm_b))
}

/// Hamming distance of two packed `u64` bitvector slices.
///
/// Returns `None` on length mismatch. Inner loop is XOR + per-word
/// `count_ones`. SIMD backends reuse [`crate::bits::popcount`].
#[must_use]
pub fn hamming_u64(a: &[u64], b: &[u64]) -> Option<u64> {
    if a.len() != b.len() {
        return None;
    }
    let mut sum = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        sum += u64::from((x ^ y).count_ones());
    }
    Some(sum)
}

/// Jaccard similarity of two packed `u64` bitvector slices.
///
/// Returns `None` on length mismatch. Returns `Some(1.0)` when both
/// vectors are all-zeros (the union is empty by convention; aligns with
/// the `divergence` module's "treat zero norm as zero distance" rule).
#[must_use]
pub fn jaccard_u64(a: &[u64], b: &[u64]) -> Option<f64> {
    if a.len() != b.len() {
        return None;
    }
    let mut intersect = 0_u64;
    let mut union = 0_u64;
    for (&x, &y) in a.iter().zip(b) {
        intersect += u64::from((x & y).count_ones());
        union += u64::from((x | y).count_ones());
    }
    if union == 0 {
        // Both inputs are empty; conventionally the Jaccard of two
        // empty sets is 1 (they are identical).
        return Some(1.0);
    }
    Some((intersect as f64) / (union as f64))
}

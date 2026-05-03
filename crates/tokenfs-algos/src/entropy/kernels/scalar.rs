use super::{ByteHistogram, entropy};

/// Computes H1 Shannon entropy.
#[must_use]
pub fn h1(histogram: &ByteHistogram) -> f32 {
    entropy::shannon::h1(histogram)
}

/// Computes byte min-entropy.
#[must_use]
pub fn min_h1(histogram: &ByteHistogram) -> f32 {
    entropy::min::h1(histogram)
}

/// Computes byte collision entropy.
#[must_use]
pub fn collision_h1(histogram: &ByteHistogram) -> f32 {
    entropy::renyi::collision_h1(histogram)
}

/// Computes dense exact adjacent-pair joint entropy.
///
/// Available only with `feature = "userspace"`. Kernel/FUSE callers
/// should use [`entropy::joint::h2_pairs_with_scratch`] or
/// [`entropy::joint::h2_pairs_with_dense_scratch`] directly — both
/// keep the dense 256x256 counter table off the call frame
/// (audit-R8 #6b).
#[cfg(feature = "userspace")]
#[must_use]
pub fn joint_h2_pairs(bytes: &[u8]) -> f32 {
    entropy::joint::h2_pairs(bytes)
}

/// Computes dense exact conditional entropy `H(next | previous)`.
///
/// Available only with `feature = "userspace"`. Kernel/FUSE callers
/// should use [`entropy::conditional::h_next_given_prev_with_scratch`]
/// directly — it keeps the dense byte-pair counter table off the call
/// frame (audit-R9 #5).
#[cfg(feature = "userspace")]
#[must_use]
pub fn conditional_h_next_given_prev(bytes: &[u8]) -> f32 {
    entropy::conditional::h_next_given_prev(bytes)
}

use super::RankSelectDict;

/// Per-position rank fan-out.
///
/// # Panics
///
/// Panics if `out.len() < positions.len()` or any position
/// exceeds `dict.len_bits()`. Available only with
/// `feature = "userspace"`; kernel-safe callers must use
/// [`rank1_batch_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
pub fn rank1_batch(dict: &RankSelectDict<'_>, positions: &[usize], out: &mut [usize]) {
    assert!(
        out.len() >= positions.len(),
        "rank1_batch: out.len() = {} < positions.len() = {}",
        out.len(),
        positions.len()
    );
    for (slot, &p) in out.iter_mut().zip(positions.iter()) {
        *slot = dict
            .try_rank1(p)
            .expect("rank1_batch: position validated by upstream dispatcher");
    }
}

/// Unchecked variant of [`rank1_batch`].
///
/// # Safety
///
/// Caller must ensure `out.len() >= positions.len()` and every
/// position is in `0..=dict.len_bits()`. Used by the userspace-gated
/// `rank1_batch` after assertion + by the kernel-safe
/// `try_rank1_batch` after upfront validation.
pub fn rank1_batch_unchecked(dict: &RankSelectDict<'_>, positions: &[usize], out: &mut [usize]) {
    for (slot, &p) in out.iter_mut().zip(positions.iter()) {
        *slot = dict
            .try_rank1(p)
            .expect("rank1_batch_unchecked: position validated upstream");
    }
}

/// Per-position select fan-out.
///
/// # Panics
///
/// Panics if `out.len() < ks.len()`. Available only with
/// `feature = "userspace"`; kernel-safe callers must use
/// [`select1_batch_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
pub fn select1_batch(dict: &RankSelectDict<'_>, ks: &[usize], out: &mut [Option<usize>]) {
    assert!(
        out.len() >= ks.len(),
        "select1_batch: out.len() = {} < ks.len() = {}",
        out.len(),
        ks.len()
    );
    for (slot, &k) in out.iter_mut().zip(ks.iter()) {
        *slot = dict.select1(k);
    }
}

/// Unchecked variant of [`select1_batch`].
///
/// # Safety
///
/// Caller must ensure `out.len() >= ks.len()`.
pub fn select1_batch_unchecked(dict: &RankSelectDict<'_>, ks: &[usize], out: &mut [Option<usize>]) {
    for (slot, &k) in out.iter_mut().zip(ks.iter()) {
        *slot = dict.select1(k);
    }
}

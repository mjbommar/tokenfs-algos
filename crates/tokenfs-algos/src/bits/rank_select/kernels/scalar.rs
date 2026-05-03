use super::RankSelectDict;

/// Per-position rank fan-out.
///
/// # Panics
///
/// Panics if `out.len() < positions.len()` or any position
/// exceeds `dict.len_bits()`.
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

/// Per-position select fan-out.
///
/// # Panics
///
/// Panics if `out.len() < ks.len()`.
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

/// Writes the K Kirsch-Mitzenmacher positions into `out`.
///
/// `out[i] = (h1.wrapping_add((i as u64).wrapping_mul(h2))) %
/// bits` for `i in 0..k`. Acts as the parity oracle for every
/// SIMD backend in this module.
///
/// # Panics
///
/// Panics if `out.len() < k` or `bits == 0`. Available only with
/// `feature = "userspace"` — kernel/FUSE callers must pre-validate
/// and use [`positions_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
pub fn positions(h1: u64, h2: u64, k: usize, bits: usize, out: &mut [u64]) {
    assert!(bits > 0, "BloomFilter bits must be > 0");
    assert!(out.len() >= k, "out buffer too small: {} < {k}", out.len());
    // SAFETY: precondition checked above.
    unsafe { positions_unchecked(h1, h2, k, bits, out) }
}

/// Unchecked variant of [`positions`] — no bounds asserts.
///
/// # Safety
///
/// Caller must ensure `out.len() >= k` and `bits > 0`. The kernel-safe
/// dispatcher (`super::auto::positions`) calls this after upstream
/// validation by [`super::super::BloomFilter::try_insert_simd`] /
/// [`try_contains_simd`] / etc. Used by audit-R10 #1 / #216 to keep
/// the dispatcher reachable in non-userspace builds without inheriting
/// the panicking assertions.
pub unsafe fn positions_unchecked(h1: u64, h2: u64, k: usize, bits: usize, out: &mut [u64]) {
    let bits_u64 = bits as u64;
    for (i, slot) in out.iter_mut().take(k).enumerate() {
        let raw = h1.wrapping_add((i as u64).wrapping_mul(h2));
        *slot = raw % bits_u64;
    }
}

/// Linear scan delegating to `slice::contains`.
///
/// Acts as the reference oracle for every SIMD backend in this
/// module. Equivalent to `haystack.iter().any(|x| *x == needle)`
/// without the manual-iter clippy lint.
#[must_use]
pub fn contains_u32(haystack: &[u32], needle: u32) -> bool {
    haystack.contains(&needle)
}

/// Per-needle scalar batch.
///
/// # Panics
///
/// Panics if `needles.len() != out.len()`. Available only with
/// `feature = "userspace"`; kernel-safe callers must use
/// [`contains_u32_batch_unchecked`] (audit-R10 #1 / #216).
#[cfg(feature = "userspace")]
pub fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    assert_eq!(needles.len(), out.len());
    contains_u32_batch_inner(haystack, needles, out);
}

/// Unchecked variant of [`contains_u32_batch`].
///
/// # Safety
///
/// Caller must ensure `needles.len() == out.len()`.
pub unsafe fn contains_u32_batch_unchecked(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    contains_u32_batch_inner(haystack, needles, out);
}

#[inline]
fn contains_u32_batch_inner(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    for (needle, slot) in needles.iter().zip(out.iter_mut()) {
        *slot = contains_u32(haystack, *needle);
    }
}

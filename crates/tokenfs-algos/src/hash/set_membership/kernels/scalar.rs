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
/// Panics if `needles.len() != out.len()`.
pub fn contains_u32_batch(haystack: &[u32], needles: &[u32], out: &mut [bool]) {
    assert_eq!(needles.len(), out.len());
    for (needle, slot) in needles.iter().zip(out.iter_mut()) {
        *slot = contains_u32(haystack, *needle);
    }
}

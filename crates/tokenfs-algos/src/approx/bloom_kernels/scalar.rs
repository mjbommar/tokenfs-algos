/// Writes the K Kirsch-Mitzenmacher positions into `out`.
///
/// `out[i] = (h1.wrapping_add((i as u64).wrapping_mul(h2))) %
/// bits` for `i in 0..k`. Acts as the parity oracle for every
/// SIMD backend in this module.
///
/// # Panics
///
/// Panics if `out.len() < k` or `bits == 0`.
pub fn positions(h1: u64, h2: u64, k: usize, bits: usize, out: &mut [u64]) {
    assert!(bits > 0, "BloomFilter bits must be > 0");
    assert!(out.len() >= k, "out buffer too small: {} < {k}", out.len());
    let bits_u64 = bits as u64;
    for (i, slot) in out.iter_mut().take(k).enumerate() {
        let raw = h1.wrapping_add((i as u64).wrapping_mul(h2));
        *slot = raw % bits_u64;
    }
}

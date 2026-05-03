/// Software CRC32C over one 32-bit word, suitable as a portable hash.
#[must_use]
#[inline]
pub fn crc32c_u32(seed: u32, value: u32) -> u32 {
    super::super::crc32c_u32_scalar(seed, value)
}

/// Software CRC32C over one byte. Pure-Rust polynomial step that
/// matches `_mm_crc32_u8` / `__crc32cb` bit-for-bit.
#[must_use]
#[inline]
pub fn crc32c_u8(seed: u32, value: u8) -> u32 {
    super::super::crc32c_byte(seed, value)
}

/// Software CRC32C over a contiguous byte slice. Bit-exact with the
/// SSE4.2 / NEON `crc32c_bytes` siblings (Castagnoli polynomial
/// 0x1EDC6F41) so the streaming [`super::super::Crc32cHasher`] is
/// portable across backends.
#[must_use]
pub fn crc32c_bytes(seed: u32, bytes: &[u8]) -> u32 {
    super::super::crc32c_bytes_scalar(seed, bytes)
}

/// Counts 4-grams into a CRC32C-hashed fixed bin array.
pub fn crc32_hash4_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    super::super::crc32_hash4_bins_with(bytes, bins, crc32c_u32);
}

/// Counts 2-grams into a CRC32C-hashed fixed bin array.
pub fn crc32_hash2_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    super::super::crc32_hash_ngram_bins_with::<2, BINS>(bytes, bins, crc32c_u32);
}

/// Counts `N`-grams, for `1 <= N <= 4`, into CRC32C-hashed bins.
pub fn crc32_hash_ngram_bins<const N: usize, const BINS: usize>(
    bytes: &[u8],
    bins: &mut [u32; BINS],
) {
    super::super::crc32_hash_ngram_bins_with::<N, BINS>(bytes, bins, crc32c_u32);
}

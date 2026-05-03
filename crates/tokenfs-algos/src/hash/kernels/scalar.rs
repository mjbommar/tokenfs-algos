use crate::hash::mix_word;

/// FNV-1a 64-bit offset basis.
pub const FNV1A64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime.
pub const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Computes FNV-1a 64-bit hash.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A64_OFFSET;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

/// Computes a lightweight 64-bit byte-stream mix.
///
/// This is a deterministic SplitMix-style streaming mixer. It is useful
/// as a second hash family for sketches and tests where CRC32C's
/// polynomial shape is not desirable.
#[must_use]
pub fn mix64(bytes: &[u8], seed: u64) -> u64 {
    let mut state = seed ^ ((bytes.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    let mut chunks = bytes.chunks_exact(8);
    for chunk in &mut chunks {
        let value = u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        state = mix_word(state ^ value);
    }

    let mut tail = 0_u64;
    for (shift, &byte) in chunks.remainder().iter().enumerate() {
        tail |= u64::from(byte) << (shift * 8);
    }
    mix_word(state ^ tail)
}

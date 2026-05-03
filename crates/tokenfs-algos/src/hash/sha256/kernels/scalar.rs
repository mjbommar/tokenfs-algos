use super::super::{BLOCK_BYTES, DIGEST_BYTES, H0, K};

/// Computes SHA-256 of `bytes` with the portable scalar kernel.
#[must_use]
pub fn sha256(bytes: &[u8]) -> [u8; DIGEST_BYTES] {
    let mut state = H0;

    let full_blocks = bytes.len() / BLOCK_BYTES;
    let tail_start = full_blocks * BLOCK_BYTES;

    for block_index in 0..full_blocks {
        let off = block_index * BLOCK_BYTES;
        let block: &[u8; BLOCK_BYTES] = (&bytes[off..off + BLOCK_BYTES])
            .try_into()
            .expect("BLOCK_BYTES slice");
        compress(&mut state, block);
    }

    // Padding: append 0x80, zero-fill, then big-endian 64-bit bit length.
    let tail = &bytes[tail_start..];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut last = [0_u8; BLOCK_BYTES * 2];
    last[..tail.len()].copy_from_slice(tail);
    last[tail.len()] = 0x80;

    let total = if tail.len() + 1 + 8 <= BLOCK_BYTES {
        BLOCK_BYTES
    } else {
        BLOCK_BYTES * 2
    };
    let length_off = total - 8;
    last[length_off..total].copy_from_slice(&bit_len.to_be_bytes());

    let last_block: &[u8; BLOCK_BYTES] = (&last[..BLOCK_BYTES])
        .try_into()
        .expect("BLOCK_BYTES slice");
    compress(&mut state, last_block);
    if total == BLOCK_BYTES * 2 {
        let next_block: &[u8; BLOCK_BYTES] = (&last[BLOCK_BYTES..total])
            .try_into()
            .expect("BLOCK_BYTES slice");
        compress(&mut state, next_block);
    }

    let mut digest = [0_u8; DIGEST_BYTES];
    for (i, word) in state.iter().enumerate() {
        digest[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// Compress one 64-byte block with the FIPS 180-4 reference path.
///
/// This is exposed (crate-private) so backend modules and tests
/// can reuse the bit-exact reference for the padding block step.
pub fn compress(state: &mut [u32; 8], block: &[u8; BLOCK_BYTES]) {
    let mut w = [0_u32; 64];
    for (i, slot) in w.iter_mut().enumerate().take(16) {
        let off = i * 4;
        *slot = u32::from_be_bytes([block[off], block[off + 1], block[off + 2], block[off + 3]]);
    }
    for i in 16..64 {
        let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
        let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16]
            .wrapping_add(s0)
            .wrapping_add(w[i - 7])
            .wrapping_add(s1);
    }

    let mut a = state[0];
    let mut b = state[1];
    let mut c = state[2];
    let mut d = state[3];
    let mut e = state[4];
    let mut f = state[5];
    let mut g = state[6];
    let mut h = state[7];

    for i in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ (!e & g);
        let temp1 = h
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(K[i])
            .wrapping_add(w[i]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);

        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    state[0] = state[0].wrapping_add(a);
    state[1] = state[1].wrapping_add(b);
    state[2] = state[2].wrapping_add(c);
    state[3] = state[3].wrapping_add(d);
    state[4] = state[4].wrapping_add(e);
    state[5] = state[5].wrapping_add(f);
    state[6] = state[6].wrapping_add(g);
    state[7] = state[7].wrapping_add(h);
}

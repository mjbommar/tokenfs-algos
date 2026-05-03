use super::super::encoded_len_bytes;

/// Packs `values` at width `w` into `out`.
///
/// # Panics
///
/// Panics if `out.len() < ceil(values.len()*w/8)` or `w` is
/// outside `1..=32`.
pub fn encode_u32_slice(w: u32, values: &[u32], out: &mut [u8]) {
    assert!((1..=32).contains(&w), "width must be 1..=32");
    let needed = encoded_len_bytes(values.len(), w);
    assert!(
        out.len() >= needed,
        "encode output buffer too small: {} < {}",
        out.len(),
        needed
    );
    // SAFETY: asserts above establish the precondition.
    unsafe { encode_u32_slice_unchecked(w, values, out) };
}

/// Scalar encoder body without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure `1 <= w <= 32` and
/// `out.len() >= ceil(values.len() * w / 8)`. The body is
/// otherwise identical to [`encode_u32_slice`] but skips the
/// leading `assert!` guards, which keeps it panic-free for
/// callers (e.g. the `try_*` APIs) that have already
/// pre-validated the buffers.
pub unsafe fn encode_u32_slice_unchecked(w: u32, values: &[u32], out: &mut [u8]) {
    let needed = encoded_len_bytes(values.len(), w);
    // Zero exactly the needed prefix; the OR-into-existing-bytes
    // strategy below requires a clean slate.
    for byte in &mut out[..needed] {
        *byte = 0;
    }

    // Byte-aligned fast paths. Hard-code these widths because
    // they degenerate to memcpy / byte-cast and are the
    // canonical token / fingerprint widths.
    match w {
        8 => {
            for (i, &v) in values.iter().enumerate() {
                out[i] = v as u8;
            }
            return;
        }
        16 => {
            for (i, &v) in values.iter().enumerate() {
                let bytes = (v as u16).to_le_bytes();
                out[2 * i] = bytes[0];
                out[2 * i + 1] = bytes[1];
            }
            return;
        }
        32 => {
            for (i, &v) in values.iter().enumerate() {
                let bytes = v.to_le_bytes();
                let off = 4 * i;
                out[off] = bytes[0];
                out[off + 1] = bytes[1];
                out[off + 2] = bytes[2];
                out[off + 3] = bytes[3];
            }
            return;
        }
        _ => {}
    }

    let mask: u64 = if w == 32 {
        u32::MAX as u64
    } else {
        (1_u64 << w) - 1
    };
    for (i, &v) in values.iter().enumerate() {
        let bit_pos = i * (w as usize);
        let byte = bit_pos / 8;
        let shift = (bit_pos % 8) as u32;
        // Up to 5 bytes are touched: 32 bits + 7 in-byte
        // offset = 39 bits, fits in 5 bytes.
        let masked = (v as u64) & mask;
        let shifted = masked << shift;
        let span_bits = shift + w;
        let span_bytes = (span_bits as usize).div_ceil(8);
        let bytes = shifted.to_le_bytes();
        for k in 0..span_bytes {
            out[byte + k] |= bytes[k];
        }
    }
}

/// Unpacks `n` values at width `w` from `input` into `out`.
///
/// # Panics
///
/// Panics if `input.len() < ceil(n*w/8)`, `out.len() < n`, or
/// `w` is outside `1..=32`.
pub fn decode_u32_slice(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    assert!((1..=32).contains(&w), "width must be 1..=32");
    let needed = encoded_len_bytes(n, w);
    assert!(
        input.len() >= needed,
        "decode input buffer too small: {} < {}",
        input.len(),
        needed
    );
    assert!(
        out.len() >= n,
        "decode output buffer too small: {} < {}",
        out.len(),
        n
    );
    // SAFETY: asserts above establish the precondition.
    unsafe { decode_u32_slice_unchecked(w, input, n, out) };
}

/// Scalar decoder body without bounds-checking asserts.
///
/// # Safety
///
/// Caller must ensure `1 <= w <= 32`,
/// `input.len() >= ceil(n * w / 8)`, and `out.len() >= n`.
/// The body is otherwise identical to [`decode_u32_slice`] but
/// skips the leading `assert!` guards, which keeps it panic-free
/// for callers (e.g. the `try_*` APIs) that have already
/// pre-validated the buffers.
pub unsafe fn decode_u32_slice_unchecked(w: u32, input: &[u8], n: usize, out: &mut [u32]) {
    let needed = encoded_len_bytes(n, w);
    // Byte-aligned fast paths mirror the encode-side specialization.
    match w {
        8 => {
            for (i, slot) in out.iter_mut().take(n).enumerate() {
                *slot = input[i] as u32;
            }
            return;
        }
        16 => {
            for (i, slot) in out.iter_mut().take(n).enumerate() {
                let bytes = [input[2 * i], input[2 * i + 1]];
                *slot = u16::from_le_bytes(bytes) as u32;
            }
            return;
        }
        32 => {
            for (i, slot) in out.iter_mut().take(n).enumerate() {
                let off = 4 * i;
                let bytes = [input[off], input[off + 1], input[off + 2], input[off + 3]];
                *slot = u32::from_le_bytes(bytes);
            }
            return;
        }
        _ => {}
    }

    let mask: u64 = if w == 32 {
        u32::MAX as u64
    } else {
        (1_u64 << w) - 1
    };
    // Fast inner loop: load 8 bytes whenever there are at least
    // 8 bytes ahead in the buffer. This covers all but the last
    // few elements without bounds-checking each byte.
    let bulk = if needed >= 8 {
        let max_byte = input.len() - 8;
        let mut count = 0_usize;
        while count < n {
            let bit_pos = count * (w as usize);
            if bit_pos / 8 > max_byte {
                break;
            }
            count += 1;
        }
        count
    } else {
        0
    };

    for (i, slot) in out.iter_mut().take(bulk).enumerate() {
        let bit_pos = i * (w as usize);
        let byte = bit_pos / 8;
        let shift = (bit_pos % 8) as u32;
        let raw = u64::from_le_bytes([
            input[byte],
            input[byte + 1],
            input[byte + 2],
            input[byte + 3],
            input[byte + 4],
            input[byte + 5],
            input[byte + 6],
            input[byte + 7],
        ]);
        *slot = ((raw >> shift) & mask) as u32;
    }

    // Tail: each value loads only the bytes that actually
    // exist. Constructed via a 64-bit accumulator so the same
    // shift/mask logic works.
    for i in bulk..n {
        let bit_pos = i * (w as usize);
        let byte = bit_pos / 8;
        let shift = (bit_pos % 8) as u32;
        let span_bits = shift + w;
        let span_bytes = (span_bits as usize).div_ceil(8);
        let mut acc = 0_u64;
        for k in 0..span_bytes {
            acc |= (input[byte + k] as u64) << (8 * k);
        }
        out[i] = ((acc >> shift) & mask) as u32;
    }
}

//! Fuzz target: streamvbyte encode -> decode must round-trip every input
//! exactly. The 256-entry shuffle table on the SIMD decode paths is the
//! historic SIMD-bug surface; fuzz wide so every control-byte pattern
//! flows through the dispatched decoder.
//!
//! Input layout:
//! - First 4 bytes (LE u32): `n` value count, capped at 8192.
//! - Remaining bytes: `4 * n` bytes parsed as `n` little-endian u32 values.
//!   Short inputs are zero-padded.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::bits::{
    streamvbyte_control_len, streamvbyte_data_max_len, streamvbyte_decode_u32,
    streamvbyte_encode_u32,
};

/// Cap value count to keep fuzz pool memory bounded. 8192 u32s fits in
/// well under 64 KiB of input/output and exercises full + partial groups.
const MAX_VALUES: usize = 8192;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n_raw = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let n = n_raw % (MAX_VALUES + 1);
    let payload = &data[4..];

    // Build a Vec<u32> of length `n`, taking 4 bytes from `payload` per
    // value when available and zero-padding the tail. This way every
    // (n, value-width) shape is reachable from a plain libFuzzer corpus.
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        let v = if off + 4 <= payload.len() {
            u32::from_le_bytes([
                payload[off],
                payload[off + 1],
                payload[off + 2],
                payload[off + 3],
            ])
        } else {
            // Synthesize a deterministic-but-varied tail so empty payload
            // still exercises non-zero shuffle codes.
            (i as u32).wrapping_mul(0x9e37_79b9)
        };
        values.push(v);
    }

    let ctrl_len = streamvbyte_control_len(n);
    let data_max = streamvbyte_data_max_len(n);
    let mut ctrl = vec![0_u8; ctrl_len];
    let mut data_buf = vec![0_u8; data_max];
    let written = streamvbyte_encode_u32(&values, &mut ctrl, &mut data_buf);

    let mut decoded = vec![0_u32; n];
    let consumed = streamvbyte_decode_u32(&ctrl, &data_buf[..written], n, &mut decoded);

    assert_eq!(
        consumed, written,
        "streamvbyte: decoder consumed {consumed} bytes but encoder wrote {written} (n={n})"
    );
    assert_eq!(
        decoded, values,
        "streamvbyte round-trip diverged at n={n}"
    );
});

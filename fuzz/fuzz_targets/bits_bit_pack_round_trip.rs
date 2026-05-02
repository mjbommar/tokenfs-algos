//! Fuzz target: DynamicBitPacker encode -> decode must round-trip every
//! width W in 1..=32 and every value count.
//!
//! Input layout:
//! - Byte 0: width selector → `1 + (byte0 % 32)`, giving W ∈ [1, 32].
//! - Bytes 1..5 (LE u32): value count, capped at 8192.
//! - Remaining bytes: source values, 4 bytes per value (LE u32). The
//!   packer silently masks high bits beyond W, so an unmasked u32 source
//!   is fine — we mask in the harness so the comparison is W-bit only.
//! - Short inputs are zero-padded.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::bits::DynamicBitPacker;

/// Cap value count to keep fuzz pool memory bounded.
const MAX_VALUES: usize = 8192;

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    let w = 1_u32 + (data[0] as u32 % 32); // 1..=32
    let n_raw = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
    let n = n_raw % (MAX_VALUES + 1);
    let payload = &data[5..];

    let mask: u32 = if w == 32 { u32::MAX } else { (1_u32 << w) - 1 };

    // Build values clamped to W bits so the round-trip comparison is well-defined.
    let mut values = Vec::with_capacity(n);
    for i in 0..n {
        let off = i * 4;
        let raw = if off + 4 <= payload.len() {
            u32::from_le_bytes([
                payload[off],
                payload[off + 1],
                payload[off + 2],
                payload[off + 3],
            ])
        } else {
            (i as u32).wrapping_mul(0x9e37_79b9)
        };
        values.push(raw & mask);
    }

    let packer = DynamicBitPacker::new(w);
    let encoded_len = packer.encoded_len(n);
    let mut encoded = vec![0_u8; encoded_len];
    packer.encode_u32_slice(&values, &mut encoded);

    let mut decoded = vec![0_u32; n];
    packer.decode_u32_slice(&encoded, n, &mut decoded);

    assert_eq!(
        decoded, values,
        "bit_pack round-trip diverged at w={w} n={n}"
    );
});

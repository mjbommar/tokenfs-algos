//! Fuzz target: `sha256_batch_st` over N message slices must agree
//! exactly with serial-iteration over `sha256` for each message.
//!
//! Input layout:
//! - First 2 bytes (LE u16): batch size, capped at 64.
//! - Then for each message: 2-byte LE length (capped at 1024) followed
//!   by that many bytes. When the input runs out we synthesise a short
//!   deterministic message so empty inputs still exercise the batch loop.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::hash::sha256::{DIGEST_BYTES, sha256};
use tokenfs_algos::hash::sha256_batch_st;

/// Cap the batch and per-message sizes to keep fuzz pool work bounded.
const MAX_BATCH: usize = 64;
const MAX_MSG_LEN: usize = 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let batch_raw = u16::from_le_bytes([data[0], data[1]]) as usize;
    let batch = batch_raw % (MAX_BATCH + 1);
    let mut cursor = 2_usize;

    // Build owned message buffers so we can hand &[u8] slices into the API.
    let mut owned: Vec<Vec<u8>> = Vec::with_capacity(batch);
    for i in 0..batch {
        let mut len = 0_usize;
        if cursor + 2 <= data.len() {
            let raw = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
            len = raw % (MAX_MSG_LEN + 1);
            cursor += 2;
        }
        let take = len.min(data.len().saturating_sub(cursor));
        let mut buf = data[cursor..cursor + take].to_vec();
        cursor += take;
        // Pad short messages with a deterministic synthesis so every
        // batch slot has the requested length even when the corpus is
        // exhausted.
        if buf.len() < len {
            for k in buf.len()..len {
                buf.push(((i.wrapping_mul(31) ^ k) & 0xff) as u8);
            }
        }
        owned.push(buf);
    }

    let messages: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
    let mut out_batch = vec![[0_u8; DIGEST_BYTES]; batch];
    sha256_batch_st(&messages, &mut out_batch);

    let mut out_serial = vec![[0_u8; DIGEST_BYTES]; batch];
    for (i, msg) in messages.iter().enumerate() {
        out_serial[i] = sha256(msg);
    }

    assert_eq!(
        out_batch, out_serial,
        "sha256_batch_st diverged from serial sha256 (batch={batch})"
    );
});

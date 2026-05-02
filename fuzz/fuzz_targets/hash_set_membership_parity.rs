//! Fuzz target: SIMD set-membership scan must agree with `slice::contains`
//! for both the single-needle entry point and the batched form.
//!
//! Input layout:
//! - Bytes 0..2 (LE u16): haystack length, capped at 4096 u32s.
//! - Bytes 2..4 (LE u16): needles length, capped at 256 u32s.
//! - Then `4 * haystack_len` bytes for the haystack (LE u32 per element).
//! - Then `4 * needles_len` bytes for the needles. Short inputs are
//!   zero-padded.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::hash::{contains_u32_batch_simd, contains_u32_simd};

const MAX_HAYSTACK: usize = 4096;
const MAX_NEEDLES: usize = 256;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let h_raw = u16::from_le_bytes([data[0], data[1]]) as usize;
    let n_raw = u16::from_le_bytes([data[2], data[3]]) as usize;
    let h_len = h_raw % (MAX_HAYSTACK + 1);
    let n_len = n_raw % (MAX_NEEDLES + 1);

    let mut cursor = 4_usize;
    let haystack = take_u32_vec(data, &mut cursor, h_len);
    let needles = take_u32_vec(data, &mut cursor, n_len);

    // Per-needle parity.
    for &needle in &needles {
        let actual = contains_u32_simd(&haystack, needle);
        let expected = haystack.contains(&needle);
        assert_eq!(
            actual, expected,
            "contains_u32_simd diverged: needle={needle} haystack_len={}",
            haystack.len()
        );
    }

    // Batched form must produce identical results.
    let mut out = vec![false; needles.len()];
    contains_u32_batch_simd(&haystack, &needles, &mut out);
    for (i, &needle) in needles.iter().enumerate() {
        let expected = haystack.contains(&needle);
        assert_eq!(
            out[i], expected,
            "contains_u32_batch_simd[{i}] diverged: needle={needle} haystack_len={}",
            haystack.len()
        );
    }

    // Spot-check the empty haystack property: never matches.
    if !needles.is_empty() {
        let empty: [u32; 0] = [];
        assert!(
            !contains_u32_simd(&empty, needles[0]),
            "empty haystack matched a needle"
        );
    }
});

fn take_u32_vec(data: &[u8], cursor: &mut usize, n: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let off = *cursor + i * 4;
        let v = if off + 4 <= data.len() {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            // Synthesize a deterministic small tail so the haystack
            // density stays varied even when the corpus runs short.
            (i as u32).wrapping_mul(0x9e37_79b9)
        };
        out.push(v);
    }
    *cursor += n * 4;
    out
}

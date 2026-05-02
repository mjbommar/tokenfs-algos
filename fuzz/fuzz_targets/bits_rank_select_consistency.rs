//! Fuzz target: RankSelectDict invariants on arbitrary bitvectors.
//!
//! Properties asserted:
//! - `rank1(0) == 0`
//! - `rank1(n_bits) == count_ones`
//! - `rank1(i + 1) - rank1(i) ∈ {0, 1}` for every i in 0..n_bits
//! - if bit i is set, then `select1(rank1(i)) == i`
//! - `select1(rank1(n_bits) - 1)` is in 0..n_bits when count_ones > 0
//! - `select1(k) >= k` (a selected index can't be smaller than its rank)
//! - `select0` mirror: `rank0(select0(k)) == k`
//!
//! Input layout:
//! - First 4 bytes (LE u32): n_bits, capped at 65 536 (one bitmap container worth).
//! - Remaining bytes: raw bit storage, treated as little-endian u8 stream
//!   into the underlying u64 word array. Short inputs zero-pad.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::bits::RankSelectDict;

/// Cap n_bits to keep fuzz pool memory bounded and queries fast. Still
/// large enough to span ~16 superblocks (4096 bits each).
const MAX_BITS: usize = 65_536;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let n_raw = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let n_bits = n_raw % (MAX_BITS + 1);
    let payload = &data[4..];

    // Build a u64 word slice large enough for n_bits, padding with zeros.
    let n_words = n_bits.div_ceil(64);
    let mut words = vec![0_u64; n_words];
    for (i, word) in words.iter_mut().enumerate() {
        let mut buf = [0_u8; 8];
        let off = i * 8;
        for k in 0..8 {
            if off + k < payload.len() {
                buf[k] = payload[off + k];
            }
        }
        *word = u64::from_le_bytes(buf);
    }

    let dict = RankSelectDict::build(&words, n_bits);

    // Property: rank1(0) == 0, rank1(n_bits) == count_ones.
    assert_eq!(dict.rank1(0), 0);
    assert_eq!(dict.rank1(n_bits), dict.count_ones());
    assert_eq!(dict.rank0(0), 0);
    assert_eq!(dict.rank0(n_bits), n_bits - dict.count_ones());

    // Property: rank1 monotone, increment in {0, 1}. Sample a bounded
    // number of positions to keep per-input cost finite even at MAX_BITS.
    // Stride keeps coverage uniform across the bitvector.
    let stride = (n_bits / 1024).max(1);
    let mut prev_rank = 0_usize;
    let mut prev_i = 0_usize;
    let mut i = 0_usize;
    while i < n_bits {
        let r = dict.rank1(i);
        // Monotonic: rank1 cannot decrease.
        assert!(
            r >= prev_rank,
            "rank1 not monotonic at i={i}: {r} < {prev_rank}"
        );
        // Bounded by index span — strictly fewer 1-bits than total bits scanned.
        assert!(r <= i, "rank1({i}) = {r} exceeds i");
        // Per-step difference is at most the number of bits skipped.
        assert!(
            r - prev_rank <= i - prev_i,
            "rank1 grew by {} over {} bits (rank({prev_i})={prev_rank} -> rank({i})={r})",
            r - prev_rank,
            i - prev_i
        );
        prev_rank = r;
        prev_i = i;
        i += stride;
    }

    // Property: select1(rank1(i)) == i if bit i is set. Sample bit indices.
    let total_ones = dict.count_ones();
    if total_ones > 0 {
        // Walk a bounded number of set bits via select1 and round-trip
        // each one through rank1.
        let select_stride = (total_ones / 256).max(1);
        let mut k = 0_usize;
        while k < total_ones {
            let pos = dict
                .select1(k)
                .expect("select1(k) must succeed for k < count_ones");
            assert!(pos < n_bits, "select1({k}) = {pos} >= n_bits {n_bits}");
            // The rank of (pos+1) counts the bit at pos itself, so equals k+1.
            assert_eq!(
                dict.rank1(pos + 1),
                k + 1,
                "rank1(select1({k}) + 1) != k + 1"
            );
            // The rank of pos itself counts strictly before, so equals k.
            assert_eq!(dict.rank1(pos), k, "rank1(select1({k})) != k");
            k += select_stride;
        }
        // Out-of-range k must return None.
        assert!(dict.select1(total_ones).is_none());
    } else {
        assert!(dict.select1(0).is_none());
    }

    // Property mirror: select0 round-trip via rank0.
    let total_zeros = n_bits - total_ones;
    if total_zeros > 0 {
        let zselect_stride = (total_zeros / 256).max(1);
        let mut k = 0_usize;
        while k < total_zeros {
            let pos = dict
                .select0(k)
                .expect("select0(k) must succeed for k < total_zeros");
            assert!(pos < n_bits, "select0({k}) = {pos} >= n_bits {n_bits}");
            assert_eq!(dict.rank0(pos + 1), k + 1, "rank0(select0+1) != k+1");
            assert_eq!(dict.rank0(pos), k, "rank0(select0) != k");
            k += zselect_stride;
        }
        assert!(dict.select0(total_zeros).is_none());
    }
});

//! Portable scalar bitmap × bitmap kernels.
//!
//! These are the reference oracles that every SIMD backend must match
//! bit-exactly. The kernels operate on `&[u64; 1024]` slices in place
//! into a caller-provided output, returning the cardinality of the
//! result for the `*_card` variants. The `*_nocard` variants skip the
//! popcount; the `*_justcard` variants skip the store and produce only
//! the cardinality.

use crate::bitmap::containers::BITMAP_WORDS;

/// AND each pair of words from `a` and `b` into `out` and return the
/// popcount of the result.
pub fn and_into(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        let word = a[i] & b[i];
        out[i] = word;
        card += word.count_ones();
    }
    card
}

/// AND each pair of words from `a` and `b` into `out` (no cardinality).
pub fn and_into_nocard(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) {
    for i in 0..BITMAP_WORDS {
        out[i] = a[i] & b[i];
    }
}

/// Return only the cardinality of `a AND b` without materialising the result.
pub fn and_cardinality(a: &[u64; BITMAP_WORDS], b: &[u64; BITMAP_WORDS]) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        card += (a[i] & b[i]).count_ones();
    }
    card
}

/// OR variant — see [`and_into`].
pub fn or_into(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        let word = a[i] | b[i];
        out[i] = word;
        card += word.count_ones();
    }
    card
}

/// OR variant — see [`and_into_nocard`].
pub fn or_into_nocard(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) {
    for i in 0..BITMAP_WORDS {
        out[i] = a[i] | b[i];
    }
}

/// OR variant — see [`and_cardinality`].
pub fn or_cardinality(a: &[u64; BITMAP_WORDS], b: &[u64; BITMAP_WORDS]) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        card += (a[i] | b[i]).count_ones();
    }
    card
}

/// XOR variant — see [`and_into`].
pub fn xor_into(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        let word = a[i] ^ b[i];
        out[i] = word;
        card += word.count_ones();
    }
    card
}

/// XOR variant — see [`and_into_nocard`].
pub fn xor_into_nocard(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) {
    for i in 0..BITMAP_WORDS {
        out[i] = a[i] ^ b[i];
    }
}

/// XOR variant — see [`and_cardinality`].
pub fn xor_cardinality(a: &[u64; BITMAP_WORDS], b: &[u64; BITMAP_WORDS]) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        card += (a[i] ^ b[i]).count_ones();
    }
    card
}

/// AND-NOT (`a AND NOT b`) variant — see [`and_into`].
pub fn andnot_into(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        let word = a[i] & !b[i];
        out[i] = word;
        card += word.count_ones();
    }
    card
}

/// AND-NOT variant — see [`and_into_nocard`].
pub fn andnot_into_nocard(
    a: &[u64; BITMAP_WORDS],
    b: &[u64; BITMAP_WORDS],
    out: &mut [u64; BITMAP_WORDS],
) {
    for i in 0..BITMAP_WORDS {
        out[i] = a[i] & !b[i];
    }
}

/// AND-NOT variant — see [`and_cardinality`].
pub fn andnot_cardinality(a: &[u64; BITMAP_WORDS], b: &[u64; BITMAP_WORDS]) -> u32 {
    let mut card: u32 = 0;
    for i in 0..BITMAP_WORDS {
        card += (a[i] & !b[i]).count_ones();
    }
    card
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distinct_bitmaps() -> ([u64; BITMAP_WORDS], [u64; BITMAP_WORDS]) {
        let mut a = [0_u64; BITMAP_WORDS];
        let mut b = [0_u64; BITMAP_WORDS];
        // Sprinkle differing bit patterns so AND/OR/XOR/ANDNOT all produce
        // non-trivial results.
        let mut state = 0xC0FFEE_u64;
        for w in 0..BITMAP_WORDS {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            a[w] = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            b[w] = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        }
        (a, b)
    }

    #[test]
    fn and_card_matches_into() {
        let (a, b) = distinct_bitmaps();
        let mut out = [0_u64; BITMAP_WORDS];
        let card = and_into(&a, &b, &mut out);
        assert_eq!(card, and_cardinality(&a, &b));
        let mut out2 = [0_u64; BITMAP_WORDS];
        and_into_nocard(&a, &b, &mut out2);
        assert_eq!(out, out2);
    }

    #[test]
    fn or_card_matches_into() {
        let (a, b) = distinct_bitmaps();
        let mut out = [0_u64; BITMAP_WORDS];
        let card = or_into(&a, &b, &mut out);
        assert_eq!(card, or_cardinality(&a, &b));
        let mut out2 = [0_u64; BITMAP_WORDS];
        or_into_nocard(&a, &b, &mut out2);
        assert_eq!(out, out2);
    }

    #[test]
    fn xor_card_matches_into() {
        let (a, b) = distinct_bitmaps();
        let mut out = [0_u64; BITMAP_WORDS];
        let card = xor_into(&a, &b, &mut out);
        assert_eq!(card, xor_cardinality(&a, &b));
        let mut out2 = [0_u64; BITMAP_WORDS];
        xor_into_nocard(&a, &b, &mut out2);
        assert_eq!(out, out2);
    }

    #[test]
    fn andnot_card_matches_into() {
        let (a, b) = distinct_bitmaps();
        let mut out = [0_u64; BITMAP_WORDS];
        let card = andnot_into(&a, &b, &mut out);
        assert_eq!(card, andnot_cardinality(&a, &b));
        let mut out2 = [0_u64; BITMAP_WORDS];
        andnot_into_nocard(&a, &b, &mut out2);
        assert_eq!(out, out2);
    }

    #[test]
    fn empty_bitmaps_yield_zero_cardinality() {
        let a = [0_u64; BITMAP_WORDS];
        let b = [0_u64; BITMAP_WORDS];
        let mut out = [0_u64; BITMAP_WORDS];
        assert_eq!(and_into(&a, &b, &mut out), 0);
        assert_eq!(or_into(&a, &b, &mut out), 0);
        assert_eq!(xor_into(&a, &b, &mut out), 0);
        assert_eq!(andnot_into(&a, &b, &mut out), 0);
    }

    #[test]
    fn full_bitmaps_yield_max_cardinality() {
        let a = [u64::MAX; BITMAP_WORDS];
        let b = [u64::MAX; BITMAP_WORDS];
        let mut out = [0_u64; BITMAP_WORDS];
        assert_eq!(and_into(&a, &b, &mut out), 65_536);
        assert_eq!(or_into(&a, &b, &mut out), 65_536);
        assert_eq!(xor_into(&a, &b, &mut out), 0);
        assert_eq!(andnot_into(&a, &b, &mut out), 0);
    }

    #[test]
    fn andnot_is_set_difference() {
        // a = {0, 1, 2}; b = {1, 2, 3}; a \ b = {0}.
        let mut a = [0_u64; BITMAP_WORDS];
        let mut b = [0_u64; BITMAP_WORDS];
        for v in [0_u16, 1, 2] {
            a[v as usize >> 6] |= 1 << (v & 63);
        }
        for v in [1_u16, 2, 3] {
            b[v as usize >> 6] |= 1 << (v & 63);
        }
        let mut out = [0_u64; BITMAP_WORDS];
        let card = andnot_into(&a, &b, &mut out);
        assert_eq!(card, 1);
        assert_eq!(out[0] & 1, 1);
    }
}

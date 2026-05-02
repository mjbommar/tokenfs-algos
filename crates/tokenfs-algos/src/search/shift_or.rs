//! Shift-Or branchless bitwise pattern matcher (Baeza-Yates-Gonnet 1992).
//!
//! Shift-Or is the dual of Bitap with the bit polarity inverted: bit `i`
//! of the state register is **0** when the prefix `needle[..=i]` is a
//! suffix of the bytes consumed so far, and **1** otherwise. The mask
//! table stores `mask[c]` = "bit `i` is 0 iff `needle[i] == c`". The
//! recurrence is then:
//!
//! ```text
//! state := (state << 1) | mask[byte]
//! match := (state & (1 << (m - 1))) == 0
//! ```
//!
//! This formulation is fully branchless inside the loop (one shift, one
//! load, one OR, one AND, one compare) and easy for the compiler to
//! unroll. We support needles up to 64 bytes via a `u64` register.
//!
//! # Complexity & state footprint
//!
//! - Init: `O(m + 256)` to fill the mask table to all-ones then clear
//!   bits per needle byte.
//! - Search: `O(n)` per call, `n = haystack.len()`.
//! - Space: `[u64; 256]` mask table = 2048 bytes plus the needle
//!   reference and a single `u64` running state.
//!
//! # Subtle correctness corner
//!
//! The initial state is `!0_u64` (all ones), meaning "no prefix is alive
//! yet". Initialising to `0` (which is the dual of Bitap's empty state)
//! is the most common bug: it would imply every prefix is alive at
//! position 0, and the very first byte would falsely report a length-1
//! match. Note also the polarity of `mask[c]`: bit `i` is `0` iff
//! `needle[i] == c`, the *opposite* of Bitap.

/// Shift-Or matcher for needles of length `1..=64`.
#[derive(Clone, Copy, Debug)]
pub struct ShiftOr<'a> {
    needle: &'a [u8],
    char_mask: [u64; 256],
}

impl<'a> ShiftOr<'a> {
    /// Maximum supported needle length.
    pub const MAX_NEEDLE_LEN: usize = 64;

    /// Builds a [`ShiftOr`] for `needle`. Returns `None` when the needle
    /// is empty or longer than [`Self::MAX_NEEDLE_LEN`].
    #[must_use]
    pub fn new(needle: &'a [u8]) -> Option<Self> {
        if needle.is_empty() || needle.len() > Self::MAX_NEEDLE_LEN {
            return None;
        }
        // Start every entry "all ones" (every prefix dead by default),
        // then clear bit i wherever needle[i] == c.
        let mut char_mask = [!0_u64; 256];
        for (i, &b) in needle.iter().enumerate() {
            char_mask[b as usize] &= !(1_u64 << i);
        }
        Some(Self { needle, char_mask })
    }

    /// Returns the start position of the first match in `haystack`, or
    /// `None`.
    #[must_use]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        let m = self.needle.len();
        if m == 0 || haystack.len() < m {
            return None;
        }
        let hit_bit: u64 = 1_u64 << (m - 1);
        let mut state: u64 = !0_u64;
        for (i, &b) in haystack.iter().enumerate() {
            state = (state << 1) | self.char_mask[b as usize];
            if state & hit_bit == 0 {
                return Some(i + 1 - m);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate alloc;

    fn naive_find(needle: &[u8], haystack: &[u8]) -> Option<usize> {
        if needle.is_empty() {
            return Some(0);
        }
        if haystack.len() < needle.len() {
            return None;
        }
        for i in 0..=haystack.len() - needle.len() {
            if &haystack[i..i + needle.len()] == needle {
                return Some(i);
            }
        }
        None
    }

    #[test]
    fn rejects_empty_or_too_long() {
        assert!(ShiftOr::new(b"").is_none());
        let too_long = [0_u8; 65];
        assert!(ShiftOr::new(&too_long).is_none());
        assert!(ShiftOr::new(b"a").is_some());
        assert!(ShiftOr::new(&[0_u8; 64]).is_some());
    }

    #[test]
    fn empty_haystack() {
        let m = ShiftOr::new(b"abc").expect("matcher");
        assert_eq!(m.find(b""), None);
    }

    #[test]
    fn single_byte_needle() {
        let m = ShiftOr::new(b"x").expect("matcher");
        assert_eq!(m.find(b"haystack with x in it"), Some(14));
        assert_eq!(m.find(b"no hit"), None);
    }

    #[test]
    fn match_at_start_middle_end() {
        let m = ShiftOr::new(b"abc").expect("matcher");
        assert_eq!(m.find(b"abcdefg"), Some(0));
        assert_eq!(m.find(b"xxabcyy"), Some(2));
        assert_eq!(m.find(b"yyyabc"), Some(3));
    }

    #[test]
    fn no_match() {
        let m = ShiftOr::new(b"needle").expect("matcher");
        assert_eq!(m.find(b"haystack"), None);
    }

    #[test]
    fn cross_check_against_naive() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"abc", b"xxabcxx"),
            (b"x", b"xxxxx"),
            (b"abcabc", b"abcabcabcabc"),
            (b"hello", b"world"),
            (b"abab", b"ababab"),
            (b"longerthanhaystack", b"short"),
        ];
        for &(needle, haystack) in cases {
            if let Some(m) = ShiftOr::new(needle) {
                assert_eq!(m.find(haystack), naive_find(needle, haystack));
            }
        }
    }

    #[test]
    fn long_needle_at_64_bytes() {
        let needle: alloc::vec::Vec<u8> = (0..64).map(|i| (i ^ 0x5A) as u8).collect();
        let m = ShiftOr::new(&needle).expect("matcher");
        let mut hay = alloc::vec::Vec::new();
        hay.extend_from_slice(&[0xFF_u8; 32]);
        hay.extend_from_slice(&needle);
        hay.extend_from_slice(&[0xFF_u8; 32]);
        assert_eq!(m.find(&hay), Some(32));
    }
}

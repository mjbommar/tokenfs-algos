//! Bitap (Wu-Manber) shift-bitmap matchers for short needles.
//!
//! The Bitap (a.k.a. Wu-Manber) algorithm represents a needle by a
//! per-byte bitmask: `mask[c]` has bit `i` set iff `needle[i] == c`. A
//! single state register tracks which prefixes of the needle are still
//! "live" given the bytes seen so far. On each haystack byte we shift the
//! state left by 1 and AND with `mask[byte]`. A match ends at the current
//! position when bit `m - 1` of the state is set.
//!
//! Two variants are provided:
//! - [`Bitap16`] — 16-bit register, supports needles of length `1..=16`.
//! - [`Bitap64`] — 64-bit register, supports needles of length `1..=64`.
//!
//! # Complexity & state footprint
//!
//! - Time: `O(n)` where `n = haystack.len()`. Each step is two memory
//!   loads, one shift, and one AND.
//! - Space: `Bitap16` holds a `[u16; 256]` mask = 512 bytes. `Bitap64`
//!   holds a `[u64; 256]` mask = 2048 bytes. Both store an additional
//!   reference to the needle.
//!
//! # Encoding choice (live-prefix encoding)
//!
//! We encode "we are `i + 1` bytes into a partial match" as bit `i` of the
//! state register. This is the "live-prefix" / unset-bit encoding used by
//! Wu-Manber 1992 directly, in contrast to Baeza-Yates-Gonnet's "dead bit"
//! variant. With `MASK_HIT = 1 << (m - 1)` this lets us check for a hit
//! with a single AND.
//!
//! # Subtle correctness corner
//!
//! The shift-and-OR-with-1 step (`(state << 1) | 1`) is critical: every
//! iteration we are also "starting a new match attempt at the current
//! position", which corresponds to OR-ing in bit 0. Without that OR, the
//! state collapses to zero after the first mismatched byte and never
//! re-enters a partial match. This is the most common bug in hand-written
//! Bitap.

/// Bitap matcher for needles of length `1..=16`.
#[derive(Clone, Copy, Debug)]
pub struct Bitap16<'a> {
    needle: &'a [u8],
    mask: [u16; 256],
}

impl<'a> Bitap16<'a> {
    /// Maximum supported needle length.
    pub const MAX_NEEDLE_LEN: usize = 16;

    /// Builds a [`Bitap16`] for `needle`. Returns `None` when the needle
    /// is empty or longer than [`Self::MAX_NEEDLE_LEN`].
    #[must_use]
    pub fn new(needle: &'a [u8]) -> Option<Self> {
        if needle.is_empty() || needle.len() > Self::MAX_NEEDLE_LEN {
            return None;
        }
        let mut mask = [0_u16; 256];
        for (i, &b) in needle.iter().enumerate() {
            mask[b as usize] |= 1_u16 << i;
        }
        Some(Self { needle, mask })
    }

    /// Returns the start position of the first match in `haystack`, or
    /// `None`.
    #[must_use]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        let m = self.needle.len();
        if m == 0 || haystack.len() < m {
            return None;
        }
        let hit_bit: u16 = 1_u16 << (m - 1);
        let mut state: u16 = 0;
        for (i, &b) in haystack.iter().enumerate() {
            // Live-prefix shift: every position can start a new attempt
            // (bit 0), and surviving prefixes shift up. The AND with
            // mask[b] only keeps prefixes consistent with the byte just
            // read.
            state = ((state << 1) | 1) & self.mask[b as usize];
            if state & hit_bit != 0 {
                // Match ends at index `i` inclusive => starts at i + 1 - m.
                return Some(i + 1 - m);
            }
        }
        None
    }

    /// Iterator over all match start positions, returned in ascending order.
    pub fn find_iter(&self, haystack: &'a [u8]) -> Bitap16Iter<'a, '_> {
        Bitap16Iter {
            matcher: self,
            haystack,
            pos: 0,
            state: 0,
        }
    }
}

/// Iterator returned by [`Bitap16::find_iter`].
#[derive(Debug)]
pub struct Bitap16Iter<'h, 'm> {
    matcher: &'m Bitap16<'m>,
    haystack: &'h [u8],
    pos: usize,
    state: u16,
}

impl<'h, 'm> Iterator for Bitap16Iter<'h, 'm> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let m = self.matcher.needle.len();
        if m == 0 {
            return None;
        }
        let hit_bit: u16 = 1_u16 << (m - 1);
        while self.pos < self.haystack.len() {
            let b = self.haystack[self.pos];
            self.state = ((self.state << 1) | 1) & self.matcher.mask[b as usize];
            let cur = self.pos;
            self.pos += 1;
            if self.state & hit_bit != 0 {
                // Match starts at cur + 1 - m.
                return Some(cur + 1 - m);
            }
        }
        None
    }
}

/// Bitap matcher for needles of length `1..=64`.
#[derive(Clone, Copy, Debug)]
pub struct Bitap64<'a> {
    needle: &'a [u8],
    mask: [u64; 256],
}

impl<'a> Bitap64<'a> {
    /// Maximum supported needle length.
    pub const MAX_NEEDLE_LEN: usize = 64;

    /// Builds a [`Bitap64`] for `needle`. Returns `None` when the needle
    /// is empty or longer than [`Self::MAX_NEEDLE_LEN`].
    #[must_use]
    pub fn new(needle: &'a [u8]) -> Option<Self> {
        if needle.is_empty() || needle.len() > Self::MAX_NEEDLE_LEN {
            return None;
        }
        let mut mask = [0_u64; 256];
        for (i, &b) in needle.iter().enumerate() {
            mask[b as usize] |= 1_u64 << i;
        }
        Some(Self { needle, mask })
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
        let mut state: u64 = 0;
        for (i, &b) in haystack.iter().enumerate() {
            state = ((state << 1) | 1) & self.mask[b as usize];
            if state & hit_bit != 0 {
                return Some(i + 1 - m);
            }
        }
        None
    }

    /// Iterator over all match start positions.
    pub fn find_iter(&self, haystack: &'a [u8]) -> Bitap64Iter<'a, '_> {
        Bitap64Iter {
            matcher: self,
            haystack,
            pos: 0,
            state: 0,
        }
    }
}

/// Iterator returned by [`Bitap64::find_iter`].
#[derive(Debug)]
pub struct Bitap64Iter<'h, 'm> {
    matcher: &'m Bitap64<'m>,
    haystack: &'h [u8],
    pos: usize,
    state: u64,
}

impl<'h, 'm> Iterator for Bitap64Iter<'h, 'm> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let m = self.matcher.needle.len();
        if m == 0 {
            return None;
        }
        let hit_bit: u64 = 1_u64 << (m - 1);
        while self.pos < self.haystack.len() {
            let b = self.haystack[self.pos];
            self.state = ((self.state << 1) | 1) & self.matcher.mask[b as usize];
            let cur = self.pos;
            self.pos += 1;
            if self.state & hit_bit != 0 {
                return Some(cur + 1 - m);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn naive_find_all(needle: &[u8], haystack: &[u8]) -> alloc::vec::Vec<usize> {
        let mut out = alloc::vec::Vec::new();
        if needle.is_empty() || haystack.len() < needle.len() {
            return out;
        }
        for i in 0..=haystack.len() - needle.len() {
            if &haystack[i..i + needle.len()] == needle {
                out.push(i);
            }
        }
        out
    }

    extern crate alloc;

    #[test]
    fn rejects_empty_or_too_long() {
        assert!(Bitap16::new(b"").is_none());
        let too_long = [0_u8; 17];
        assert!(Bitap16::new(&too_long).is_none());
        assert!(Bitap16::new(b"a").is_some());
        assert!(Bitap16::new(&[1_u8; 16]).is_some());

        assert!(Bitap64::new(b"").is_none());
        let too_long = [0_u8; 65];
        assert!(Bitap64::new(&too_long).is_none());
        assert!(Bitap64::new(b"a").is_some());
        assert!(Bitap64::new(&[1_u8; 64]).is_some());
    }

    #[test]
    fn empty_haystack() {
        let m = Bitap16::new(b"abc").expect("matcher");
        assert_eq!(m.find(b""), None);
        let m64 = Bitap64::new(b"abc").expect("matcher");
        assert_eq!(m64.find(b""), None);
    }

    #[test]
    fn single_byte_needle() {
        let m = Bitap16::new(b"x").expect("matcher");
        assert_eq!(m.find(b"haystack with x in it"), Some(14));
        assert_eq!(m.find(b"no hit"), None);
    }

    #[test]
    fn match_at_start_middle_end() {
        let needle = b"abc";
        let m = Bitap16::new(needle).expect("matcher");
        assert_eq!(m.find(b"abcdefg"), Some(0));
        assert_eq!(m.find(b"xxabcyy"), Some(2));
        assert_eq!(m.find(b"yyyabc"), Some(3));
    }

    #[test]
    fn no_match() {
        let m = Bitap16::new(b"needle").expect("matcher");
        assert_eq!(m.find(b"haystack"), None);
    }

    #[test]
    fn multiple_matches() {
        let m = Bitap16::new(b"ab").expect("matcher");
        let hits: alloc::vec::Vec<usize> = m.find_iter(b"abxabyab").collect();
        assert_eq!(hits, alloc::vec![0, 3, 6]);
    }

    #[test]
    fn overlapping_matches() {
        let m = Bitap16::new(b"aa").expect("matcher");
        let hits: alloc::vec::Vec<usize> = m.find_iter(b"aaaa").collect();
        assert_eq!(hits, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn bitap64_long_needle() {
        let needle = [0xAA_u8; 64];
        let m = Bitap64::new(&needle).expect("matcher");
        let mut hay = alloc::vec::Vec::new();
        hay.extend_from_slice(&[0_u8; 32]);
        hay.extend_from_slice(&needle);
        hay.extend_from_slice(&[0_u8; 32]);
        assert_eq!(m.find(&hay), Some(32));
    }

    #[test]
    fn cross_check_bitap16_naive() {
        // A few representative shapes; full proptest cross-check with all
        // five algorithms lives in `super::tests`.
        let cases: &[(&[u8], &[u8])] = &[
            (b"abc", b"xxabcxx"),
            (b"x", b"xxxxx"),
            (b"abcde", b"abcabcdeabcde"),
            (b"hello", b"world"),
            (b"a", b""),
        ];
        for &(needle, haystack) in cases {
            let m = Bitap16::new(needle).expect("matcher");
            assert_eq!(m.find(haystack), naive_find(needle, haystack));
            let hits: alloc::vec::Vec<usize> = m.find_iter(haystack).collect();
            assert_eq!(hits, naive_find_all(needle, haystack));
        }
    }
}

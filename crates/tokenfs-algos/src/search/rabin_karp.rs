//! Rabin-Karp rolling-hash substring search.
//!
//! Uses a polynomial hash modulo `2^64` (i.e. natural `u64` wrapping
//! arithmetic). The base constant is the golden-ratio multiplier
//! `0x9E37_79B9_7F4A_7C15` — a well-known choice that gives good
//! avalanche on byte sequences and is used in practice as a Fibonacci
//! hash multiplier.
//!
//! On every hash collision we verify by direct slice compare, so the
//! return is always exact. The only remaining cost of a hash collision
//! is the verify; with a 64-bit hash collisions are extremely rare for
//! reasonable haystacks.
//!
//! # Complexity & state footprint
//!
//! - Init: `O(m)` to fold the needle hash and `O(m)` to compute the
//!   leading-byte cancellation power `BASE^(m-1)`.
//! - Search: `O(n)` expected; `O(n m)` worst case if every window
//!   collides (impossible in practice with the 64-bit modulus).
//! - Space: three `u64`s and the needle reference. No per-byte tables.
//!
//! # Subtle correctness corner
//!
//! The leading-byte cancellation power `BASE^(m-1)` is required to
//! "unhash" the byte that just left the rolling window. Off-by-one here
//! (using `BASE^m`) silently produces wrong rolling hashes that still
//! match the needle's hash for some patterns — the verify step catches
//! the false positive, but you forfeit the rolling speedup. Make sure
//! `base_pow == BASE^(m-1)`.

/// Multiplier constant: the golden-ratio Fibonacci hash multiplier.
const BASE: u64 = 0x9E37_79B9_7F4A_7C15;

/// Rabin-Karp substring matcher.
#[derive(Clone, Copy, Debug)]
pub struct RabinKarp<'a> {
    needle: &'a [u8],
    needle_hash: u64,
    /// `BASE^(m-1)` for `m = needle.len() >= 1`. Unused when `m <= 1`
    /// (the rolling step degenerates).
    base_pow: u64,
}

impl<'a> RabinKarp<'a> {
    /// Builds a [`RabinKarp`] for `needle`.
    #[must_use]
    pub fn new(needle: &'a [u8]) -> Self {
        let needle_hash = hash_block(needle);
        let m = needle.len();
        let mut base_pow: u64 = 1;
        // base_pow = BASE^(m-1)
        for _ in 1..m {
            base_pow = base_pow.wrapping_mul(BASE);
        }
        Self {
            needle,
            needle_hash,
            base_pow,
        }
    }

    /// Returns the start position of the first match in `haystack`, or
    /// `None`.
    #[must_use]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        let m = self.needle.len();
        if m == 0 {
            return Some(0);
        }
        if haystack.len() < m {
            return None;
        }

        // Hash the first window.
        let mut h = hash_block(&haystack[..m]);
        if h == self.needle_hash && haystack[..m] == *self.needle {
            return Some(0);
        }

        // Roll: each step removes haystack[i] (the leaving byte) and
        // adds haystack[i + m]. The hash function is
        //   H(s_0..s_{m-1}) = s_0 * B^(m-1) + s_1 * B^(m-2) + ... + s_{m-1}.
        // To roll forward by one we subtract s_0 * B^(m-1), multiply by
        // B, and add the new byte.
        let limit = haystack.len() - m;
        for i in 0..limit {
            let leaving = u64::from(haystack[i]);
            let entering = u64::from(haystack[i + m]);
            h = h.wrapping_sub(leaving.wrapping_mul(self.base_pow));
            h = h.wrapping_mul(BASE);
            h = h.wrapping_add(entering);
            let start = i + 1;
            if h == self.needle_hash && haystack[start..start + m] == *self.needle {
                return Some(start);
            }
        }
        None
    }
}

/// Polynomial hash of a byte slice with the [`BASE`] multiplier.
fn hash_block(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0;
    for &b in bytes {
        h = h.wrapping_mul(BASE).wrapping_add(u64::from(b));
    }
    h
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
    fn empty_haystack() {
        let m = RabinKarp::new(b"abc");
        assert_eq!(m.find(b""), None);
    }

    #[test]
    fn empty_needle_returns_zero() {
        let m = RabinKarp::new(b"");
        assert_eq!(m.find(b"anything"), Some(0));
        assert_eq!(m.find(b""), Some(0));
    }

    #[test]
    fn single_byte_needle() {
        let m = RabinKarp::new(b"x");
        assert_eq!(m.find(b"haystack with x in it"), Some(14));
        assert_eq!(m.find(b"no hit"), None);
    }

    #[test]
    fn match_at_start_middle_end() {
        let m = RabinKarp::new(b"abc");
        assert_eq!(m.find(b"abcdefg"), Some(0));
        assert_eq!(m.find(b"xxabcyy"), Some(2));
        assert_eq!(m.find(b"yyyabc"), Some(3));
    }

    #[test]
    fn no_match() {
        let m = RabinKarp::new(b"needle");
        assert_eq!(m.find(b"haystack"), None);
    }

    #[test]
    fn periodic_input() {
        let m = RabinKarp::new(b"abab");
        assert_eq!(m.find(b"abababab"), Some(0));
        assert_eq!(m.find(b"xxxabababxx"), Some(3));
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
            let m = RabinKarp::new(needle);
            assert_eq!(m.find(haystack), naive_find(needle, haystack));
        }
    }

    #[test]
    fn big_haystack_finds_late_needle() {
        let needle = b"FINDME";
        let mut hay = alloc::vec::Vec::new();
        for _ in 0..1000 {
            hay.extend_from_slice(b"ABCDEFG");
        }
        hay.extend_from_slice(needle);
        let m = RabinKarp::new(needle);
        assert_eq!(m.find(&hay), Some(7000));
    }
}

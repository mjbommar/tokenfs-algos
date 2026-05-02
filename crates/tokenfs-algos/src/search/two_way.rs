//! Two-Way (Crochemore-Perrin 1991) substring search.
//!
//! General-purpose `O(n + m)` substring search with `O(1)` extra space
//! (no preprocessing tables beyond the critical position and period of
//! the needle). Worst case is bounded by the constant.
//!
//! The algorithm works in two phases:
//!
//! 1. **Critical factorization.** The needle is split at a position
//!    `crit_pos` so that the period of the needle equals the local period
//!    at the split. We compute `crit_pos` and `period` from the needle
//!    using two maximal-suffix passes (one with `<=`, one with `>=`).
//! 2. **Scan.** Starting at each candidate window in the haystack we
//!    verify the right factor (positions `crit_pos..m`) left to right;
//!    on success we verify the left factor (positions `0..crit_pos`)
//!    right to left. Memory of the previous match suffix lets us shift
//!    by `period` after a successful right-factor compare and skip
//!    characters known to match.
//!
//! The implementation here is the textbook "long-period" variant
//! (period-as-shift), which handles all needles correctly including
//! periodic ones at a small constant overhead.
//!
//! # Complexity & state footprint
//!
//! - Time: `O(n + m)` where `n = haystack.len()`, `m = needle.len()`.
//! - Space: a `usize` for `crit_pos` and a `usize` for `period`. No
//!   per-byte tables.
//!
//! # Subtle correctness corner
//!
//! Critical factorization uses the lexicographically maximal suffix under
//! both the natural and the inverted byte order, then picks the *later*
//! split. Getting the comparison direction wrong gives a `crit_pos`
//! whose `period` is larger than the true period, and the scanner then
//! over-shifts and misses matches. The pair of `maximal_suffix` calls
//! and the `>` (not `>=`) at the end is the most common bug here.

/// Two-Way (Crochemore-Perrin) substring matcher.
#[derive(Clone, Copy, Debug)]
pub struct TwoWay<'a> {
    needle: &'a [u8],
    crit_pos: usize,
    period: usize,
}

impl<'a> TwoWay<'a> {
    /// Builds a [`TwoWay`] for `needle`.
    #[must_use]
    pub fn new(needle: &'a [u8]) -> Self {
        let (crit_pos, period) = critical_factorization(needle);
        Self {
            needle,
            crit_pos,
            period,
        }
    }

    /// Returns the position of the first match of the needle in
    /// `haystack`, or `None`.
    #[must_use]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        if self.needle.is_empty() {
            return Some(0);
        }
        if haystack.len() < self.needle.len() {
            return None;
        }
        // Decide between the periodic and non-periodic scanners. A needle
        // is "small-period" iff the prefix `needle[..crit_pos]` is a
        // suffix of the prefix `needle[crit_pos..crit_pos + crit_pos]`.
        // We collapse both into the periodic scanner with `memory`,
        // which is correct for all needles at a small constant overhead.
        scan(self.needle, self.crit_pos, self.period, haystack, false)
    }

    /// Iterator over all match start positions (non-overlapping is *not*
    /// implied — overlapping matches are reported).
    pub fn find_iter(&self, haystack: &'a [u8]) -> TwoWayIter<'a, '_> {
        TwoWayIter {
            matcher: self,
            haystack,
            pos: 0,
        }
    }
}

/// Iterator returned by [`TwoWay::find_iter`].
#[derive(Debug)]
pub struct TwoWayIter<'h, 'm> {
    matcher: &'m TwoWay<'m>,
    haystack: &'h [u8],
    pos: usize,
}

impl<'h, 'm> Iterator for TwoWayIter<'h, 'm> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.matcher.needle.is_empty() {
            // Avoid an infinite stream; report position 0 once.
            if self.pos == 0 {
                self.pos = 1;
                return Some(0);
            }
            return None;
        }
        if self.pos >= self.haystack.len() {
            return None;
        }
        let sub = &self.haystack[self.pos..];
        let hit = scan(
            self.matcher.needle,
            self.matcher.crit_pos,
            self.matcher.period,
            sub,
            false,
        )?;
        let abs = self.pos + hit;
        // Advance by 1 to allow overlapping matches.
        self.pos = abs + 1;
        Some(abs)
    }
}

// ---------- Internals ----------

/// Returns `(crit_pos, period)` for `needle`. For an empty needle returns
/// `(0, 1)` (period is irrelevant; callers must short-circuit).
fn critical_factorization(needle: &[u8]) -> (usize, usize) {
    if needle.is_empty() {
        return (0, 1);
    }
    // Compute (split index, period) for both maximal suffixes (one with
    // standard ordering, one with inverted) and pick the one with the
    // larger split index.
    let (i_le, p_le) = maximal_suffix(needle, false);
    let (i_ge, p_ge) = maximal_suffix(needle, true);
    if i_le > i_ge {
        (i_le, p_le)
    } else {
        (i_ge, p_ge)
    }
}

/// Computes the maximal suffix of `needle` and returns
/// `(split_index, period)`.
///
/// The returned `split_index` is the position immediately *after* the
/// last byte of the prefix `u`, so `needle = u v` where `v` is the
/// maximal suffix. Period `p` is the (eventually-)periodic period of
/// `v`. When `invert` is true the comparison is reversed (i.e. we
/// compute the *minimal* suffix under standard ordering).
///
/// Algorithm: walking comparison á la Crochemore-Perrin. `ms` is the
/// last-known starting position of the best suffix so far; `j` is the
/// current candidate; `k` is the offset within the current run of
/// comparisons; `p` is the running period.
fn maximal_suffix(needle: &[u8], invert: bool) -> (usize, usize) {
    let m = needle.len();
    let mut ms: isize = -1;
    let mut j: isize = 0;
    let mut k: isize = 1;
    let mut p: isize = 1;
    while (j + k) < (m as isize) {
        let a = needle[(j + k) as usize];
        let b = needle[(ms + k) as usize];
        // Three-way compare under chosen ordering.
        let a_smaller = if invert { a > b } else { a < b };
        let a_larger = if invert { a < b } else { a > b };
        if a_smaller {
            // Suffix at j is no longer maximal: skip.
            j += k;
            k = 1;
            p = j - ms;
        } else if a == b {
            if k == p {
                j += p;
                k = 1;
            } else {
                k += 1;
            }
        } else {
            // a_larger
            debug_assert!(a_larger);
            ms = j;
            j = ms + 1;
            k = 1;
            p = 1;
        }
    }
    ((ms + 1) as usize, p as usize)
}

/// Two-way scanner. Returns the first match offset relative to the start
/// of `haystack`.
///
/// The `_long_period` flag is reserved for future tuning; this scanner
/// uses memory and works for all needles.
fn scan(
    needle: &[u8],
    crit_pos: usize,
    period: usize,
    haystack: &[u8],
    _long_period: bool,
) -> Option<usize> {
    let m = needle.len();
    let n = haystack.len();
    if m == 0 {
        return Some(0);
    }
    if n < m {
        return None;
    }

    // Decide whether we are in the small-period regime: needle[..crit_pos]
    // must be a suffix of needle[crit_pos..crit_pos + crit_pos] and the
    // suffix factor must repeat the period inside the needle.
    let small_period =
        crit_pos < m - crit_pos && needle[..crit_pos] == needle[period..period + crit_pos];

    if small_period {
        // Periodic scanner with memory.
        let mut pos: usize = 0;
        let mut memory: usize = 0;
        while pos + m <= n {
            // Right factor: needle[max(crit_pos, memory)..m].
            let mut i = core::cmp::max(crit_pos, memory);
            while i < m && needle[i] == haystack[pos + i] {
                i += 1;
            }
            if i < m {
                // Mismatch on the right factor: shift past it.
                pos += i - crit_pos + 1;
                memory = 0;
                continue;
            }
            // Right factor matched; verify left factor.
            let mut j = (crit_pos as isize) - 1;
            while j >= memory as isize && needle[j as usize] == haystack[pos + j as usize] {
                j -= 1;
            }
            if j < memory as isize {
                return Some(pos);
            }
            // Left mismatch: shift by `period`, remember `m - period`.
            pos += period;
            memory = m - period;
        }
        None
    } else {
        // Non-periodic scanner: shift by max(crit_pos + 1, mismatch + 1).
        let shift = core::cmp::max(crit_pos + 1, m - crit_pos);
        let mut pos: usize = 0;
        while pos + m <= n {
            let mut i = crit_pos;
            while i < m && needle[i] == haystack[pos + i] {
                i += 1;
            }
            if i < m {
                pos += i - crit_pos + 1;
                continue;
            }
            // Right matched; verify left right-to-left.
            let mut j = (crit_pos as isize) - 1;
            while j >= 0 && needle[j as usize] == haystack[pos + j as usize] {
                j -= 1;
            }
            if j < 0 {
                return Some(pos);
            }
            pos += shift;
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

    #[test]
    fn empty_haystack() {
        let m = TwoWay::new(b"abc");
        assert_eq!(m.find(b""), None);
    }

    #[test]
    fn empty_needle_returns_zero() {
        let m = TwoWay::new(b"");
        assert_eq!(m.find(b"anything"), Some(0));
        assert_eq!(m.find(b""), Some(0));
    }

    #[test]
    fn match_at_start_middle_end() {
        let m = TwoWay::new(b"abc");
        assert_eq!(m.find(b"abcdefg"), Some(0));
        assert_eq!(m.find(b"xxabcyy"), Some(2));
        assert_eq!(m.find(b"yyyabc"), Some(3));
    }

    #[test]
    fn no_match() {
        let m = TwoWay::new(b"needle");
        assert_eq!(m.find(b"haystack"), None);
    }

    #[test]
    fn multiple_matches() {
        let m = TwoWay::new(b"ab");
        let hits: alloc::vec::Vec<usize> = m.find_iter(b"abxabyab").collect();
        assert_eq!(hits, alloc::vec![0, 3, 6]);
    }

    #[test]
    fn overlapping_matches() {
        let m = TwoWay::new(b"aa");
        let hits: alloc::vec::Vec<usize> = m.find_iter(b"aaaa").collect();
        assert_eq!(hits, alloc::vec![0, 1, 2]);
    }

    #[test]
    fn periodic_needle() {
        // Needles like "abab" exercise the small-period branch.
        let m = TwoWay::new(b"abab");
        assert_eq!(m.find(b"xxxabababxx"), Some(3));
        let m2 = TwoWay::new(b"abcabcabc");
        assert_eq!(m2.find(b"yyabcabcabczz"), Some(2));
    }

    #[test]
    fn long_repeating_needle() {
        let needle: alloc::vec::Vec<u8> = (0..96).map(|i| (i % 7) as u8).collect();
        let mut hay = alloc::vec::Vec::new();
        hay.extend_from_slice(&[0_u8; 256]);
        hay.extend_from_slice(&needle);
        hay.extend_from_slice(&[0_u8; 256]);
        let m = TwoWay::new(&needle);
        assert_eq!(m.find(&hay), Some(256));
    }

    #[test]
    fn cross_check_against_naive() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"abc", b"xxabcxx"),
            (b"x", b"xxxxx"),
            (b"abcabc", b"abcabcabcabc"),
            (b"hello", b"world"),
            (b"abab", b"ababab"),
            (b"a", b""),
            (b"longerthanhaystack", b"short"),
            (b"abcabd", b"abcabcabd"),
        ];
        for &(needle, haystack) in cases {
            let m = TwoWay::new(needle);
            assert_eq!(
                m.find(haystack),
                naive_find(needle, haystack),
                "first hit: needle={needle:?} haystack={haystack:?}",
            );
            let hits: alloc::vec::Vec<usize> = m.find_iter(haystack).collect();
            assert_eq!(
                hits,
                naive_find_all(needle, haystack),
                "all hits: needle={needle:?} haystack={haystack:?}",
            );
        }
    }
}

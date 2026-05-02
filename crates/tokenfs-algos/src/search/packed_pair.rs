//! Packed pair (memchr-style) hybrid 2-byte pre-filter substring search.
//!
//! For needles of length ≥ 2 we pick two byte offsets `(0, offset)` from
//! the needle (rare bytes preferred) and SIMD-scan the haystack for
//! windows where `haystack[i] == byte0 && haystack[i + offset] == byte1`.
//! Each pre-filter hit is then verified by a slice compare.
//!
//! On x86_64 with AVX2 the pre-filter loads 32 bytes at a time, computes
//! the two compares with `_mm256_cmpeq_epi8`, ANDs the masks together,
//! and movemasks the result; the popcount of the mask tells us how many
//! verifies to do per 32-byte window. A scalar fallback handles
//! non-AVX2 builds, the AVX2 tail, and the case where `offset` makes
//! the second compare exceed the loaded vector.
//!
//! Byte selection: we score every byte position in the needle by an
//! approximate "rare in English text" prior so that the pre-filter is
//! more selective than picking the first two bytes. The current ranks
//! map ASCII letters/punctuation to a small integer; bytes outside this
//! range default to a low rank (rare). The two highest-rank bytes
//! (different positions) are picked, with the first byte always at
//! offset 0 (to preserve a clean shift in the AVX2 loop) and the second
//! byte at the rarest non-zero offset.
//!
//! # Complexity & state footprint
//!
//! - Init: `O(m)` to scan the needle and pick the rare-byte offset.
//! - Search: `O(n)` expected; verification cost dominates only when the
//!   pre-filter is poorly selective (rare bytes appear elsewhere too).
//! - Space: four bytes (`byte0`, `byte1`, two `usize`s) plus the needle
//!   reference. No per-byte tables.

#[cfg(all(feature = "std", target_arch = "x86"))]
use core::arch::x86;
#[cfg(all(feature = "std", target_arch = "x86_64"))]
use core::arch::x86_64 as x86;

/// Packed-pair substring matcher.
#[derive(Clone, Copy, Debug)]
pub struct PackedPair<'a> {
    needle: &'a [u8],
    byte0: u8,
    byte1: u8,
    offset: usize,
}

impl<'a> PackedPair<'a> {
    /// Builds a [`PackedPair`] for `needle`. Returns `None` when the
    /// needle is shorter than 2 bytes.
    #[must_use]
    pub fn new(needle: &'a [u8]) -> Option<Self> {
        if needle.len() < 2 {
            return None;
        }
        // Always anchor byte0 at offset 0; pick byte1 at the rarest
        // remaining position (heuristic = highest-rank byte).
        let byte0 = needle[0];
        let mut best_offset = 1;
        let mut best_rank = byte_rarity_rank(needle[1]);
        for (i, &b) in needle.iter().enumerate().skip(2) {
            let r = byte_rarity_rank(b);
            if r > best_rank {
                best_rank = r;
                best_offset = i;
            }
        }
        Some(Self {
            needle,
            byte0,
            byte1: needle[best_offset],
            offset: best_offset,
        })
    }

    /// Returns the position of the first match in `haystack`, or `None`.
    #[must_use]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        let m = self.needle.len();
        if haystack.len() < m {
            return None;
        }

        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 availability checked immediately above.
                return unsafe { self.find_avx2(haystack) };
            }
        }
        self.find_scalar(haystack)
    }

    /// Scalar pre-filter + verify path. Always available.
    fn find_scalar(&self, haystack: &[u8]) -> Option<usize> {
        let m = self.needle.len();
        let last = haystack.len().checked_sub(m)?;
        let off = self.offset;
        for i in 0..=last {
            // Pre-filter: cheap two-byte compare.
            if haystack[i] == self.byte0 && haystack[i + off] == self.byte1 {
                // Verify the full needle.
                if &haystack[i..i + m] == self.needle {
                    return Some(i);
                }
            }
        }
        None
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    /// AVX2 pre-filter + verify path.
    ///
    /// # Safety
    ///
    /// Caller must ensure the current CPU supports AVX2.
    #[target_feature(enable = "avx2")]
    unsafe fn find_avx2(&self, haystack: &[u8]) -> Option<usize> {
        let m = self.needle.len();
        let off = self.offset;
        let last = haystack.len().checked_sub(m)?;
        // We need to load a 32-byte window starting at `i` for byte0,
        // and at `i + off` for byte1; the latter must stay in-bounds.
        // Vector loop bound: i_max + 31 + off < haystack.len()
        // <=> i_max <= haystack.len() - 32 - off
        // We additionally need i_max <= last (so verify stays in
        // bounds). Take the tighter of the two as the vector loop bound.
        let vec_max = match haystack.len().checked_sub(32 + off) {
            Some(v) => v.min(last),
            None => return self.find_scalar(haystack),
        };

        // byte0/byte1 splatted to a 256-bit register. The intrinsic is
        // pure (no memory access), safe inside a target_feature("avx2")
        // function.
        let v0 = x86::_mm256_set1_epi8(self.byte0 as i8);
        let v1 = x86::_mm256_set1_epi8(self.byte1 as i8);

        let mut i: usize = 0;
        while i <= vec_max {
            // SAFETY: i <= vec_max => i + 31 + off < haystack.len(), so
            // both 32-byte loads are entirely within `haystack`.
            let chunk0 =
                unsafe { x86::_mm256_loadu_si256(haystack.as_ptr().add(i).cast::<x86::__m256i>()) };
            let chunk1 = unsafe {
                x86::_mm256_loadu_si256(haystack.as_ptr().add(i + off).cast::<x86::__m256i>())
            };
            let eq0 = x86::_mm256_cmpeq_epi8(chunk0, v0);
            let eq1 = x86::_mm256_cmpeq_epi8(chunk1, v1);
            let both = x86::_mm256_and_si256(eq0, eq1);
            let mut mask = x86::_mm256_movemask_epi8(both) as u32;
            while mask != 0 {
                let bit = mask.trailing_zeros() as usize;
                let cand = i + bit;
                if cand <= last && haystack[cand..cand + m] == *self.needle {
                    return Some(cand);
                }
                mask &= mask - 1;
            }
            i += 32;
        }

        // Scalar tail covers everything from `i` onwards (positions that
        // didn't fit a full 32-byte vector window).
        for j in i..=last {
            if haystack[j] == self.byte0
                && haystack[j + off] == self.byte1
                && &haystack[j..j + m] == self.needle
            {
                return Some(j);
            }
        }
        None
    }
}

/// A rough "rarity in typical text" rank — higher = rarer. Common ASCII
/// letters and whitespace get low ranks; everything else defaults to the
/// rarest tier. This is a heuristic; the algorithm is correct for any
/// choice, but a smarter pre-filter offset reduces verify work.
const fn byte_rarity_rank(b: u8) -> u8 {
    match b {
        b' ' => 0,
        b'e' | b'E' => 1,
        b't' | b'T' => 1,
        b'a' | b'A' => 1,
        b'o' | b'O' => 1,
        b'i' | b'I' => 1,
        b'n' | b'N' => 1,
        b's' | b'S' => 1,
        b'h' | b'H' => 1,
        b'r' | b'R' => 1,
        b'l' | b'L' | b'd' | b'D' | b'u' | b'U' | b'c' | b'C' | b'm' | b'M' => 2,
        b',' | b'.' | b'\n' | b'\r' | b'\t' => 2,
        // Other ASCII letters / digits.
        0x30..=0x39 | 0x41..=0x5A | 0x61..=0x7A => 3,
        // Punctuation and high-bit bytes.
        _ => 4,
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
    fn rejects_short_needle() {
        assert!(PackedPair::new(b"").is_none());
        assert!(PackedPair::new(b"a").is_none());
        assert!(PackedPair::new(b"ab").is_some());
    }

    #[test]
    fn empty_haystack() {
        let m = PackedPair::new(b"abc").expect("matcher");
        assert_eq!(m.find(b""), None);
    }

    #[test]
    fn match_at_start_middle_end() {
        let m = PackedPair::new(b"abc").expect("matcher");
        assert_eq!(m.find(b"abcdefg"), Some(0));
        assert_eq!(m.find(b"xxabcyy"), Some(2));
        assert_eq!(m.find(b"yyyabc"), Some(3));
    }

    #[test]
    fn no_match() {
        let m = PackedPair::new(b"needle").expect("matcher");
        assert_eq!(m.find(b"haystack"), None);
    }

    #[test]
    fn cross_check_against_naive() {
        let cases: &[(&[u8], &[u8])] = &[
            (b"abc", b"xxabcxx"),
            (b"abcabc", b"abcabcabcabc"),
            (b"hello", b"world"),
            (b"abab", b"ababab"),
            (b"longerthanhaystack", b"short"),
            (b"xy", b"xxyyxyxy"),
        ];
        for &(needle, haystack) in cases {
            let m = PackedPair::new(needle).expect("matcher");
            assert_eq!(
                m.find(haystack),
                naive_find(needle, haystack),
                "needle={needle:?} haystack={haystack:?}",
            );
        }
    }

    #[test]
    fn pre_filter_hits_without_full_match() {
        // byte0 = 'a', byte1 = 'c' — needle has many positions where
        // ('a', 'c') appears but full needle does not.
        let needle = b"abcd";
        let m = PackedPair::new(needle).expect("matcher");
        // Many 'a..c' pairs but only one full "abcd" match (at index 14).
        let hay = b"axcyabczzabbczzabcd";
        let expected = naive_find(needle, hay);
        assert_eq!(m.find(hay), expected);
    }

    #[test]
    fn long_haystack_avx2_path() {
        let needle = b"NEEDLE!";
        let mut hay = alloc::vec::Vec::new();
        hay.extend_from_slice(&[b'A'; 1024]);
        hay.extend_from_slice(needle);
        hay.extend_from_slice(&[b'Z'; 1024]);
        let m = PackedPair::new(needle).expect("matcher");
        assert_eq!(m.find(&hay), Some(1024));
    }
}

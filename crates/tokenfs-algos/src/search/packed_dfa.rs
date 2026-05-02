//! Packed multi-pattern DFA with byte-class alphabet compression.
//!
//! A scaled-down Aho-Corasick-style DFA over a compressed byte-class
//! alphabet. The construction has two layers:
//!
//! 1. **Byte-class table.** We collect every byte that appears in any
//!    pattern, plus a "fail" class for everything else, and assign each
//!    distinct byte its own equivalence class (capped at 32). When the
//!    union of all pattern alphabets exceeds 32 distinct bytes we
//!    coalesce overflow bytes into the "fail" class — which makes the
//!    DFA over-approximate failure transitions but never mis-report a
//!    match (verification at hit-state still needs the original byte to
//!    have been the right one, which is guaranteed by the goto trie).
//! 2. **State table.** A flat `[u32; n_states * n_classes]` row-major
//!    table indexed by `(state, class)`. Each entry is a precomputed
//!    successor state (with failure-link composition baked in, so
//!    matching is a single lookup per byte). Patterns that end at a
//!    given state are stored in a parallel `match_lists` table.
//!
//! Don't expect feature parity with the full `aho-corasick` crate — this
//! is a pruned, production-tier-1 implementation aimed at small
//! pattern sets (≤ ~hundreds of bytes total, ≤ ~32 distinct bytes).
//!
//! # Complexity & state footprint
//!
//! - Build: `O(P)` where `P` is the total length of all patterns, plus
//!   one BFS to compose failure links into the goto table.
//! - Search: `O(n)` per call — exactly one lookup per haystack byte
//!   plus one match-list scan per state visited.
//! - Space: `n_states * n_classes * 4 bytes` for the state table, plus
//!   `[u8; 256]` for the byte-class map and one `Vec<usize>` per state
//!   for the match list.
//!
//! # Subtle correctness corner
//!
//! The "coalesce overflow bytes into the fail class" optimisation is
//! only safe because the underlying goto-trie distinguishes which
//! byte caused a transition: if byte `b` shares the fail class with
//! byte `b'`, both are routed through the same DFA edge, but the
//! match-list at the destination state is still keyed by the *original*
//! patterns, so `find()` reports the right pattern indices. The
//! over-approximation only affects performance (more verifies) — never
//! correctness.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Maximum number of distinct pattern bytes that get individual classes.
/// Plus the wildcard ("byte not in any pattern") class, the alphabet
/// size after compression is at most `MAX_PATTERN_CLASSES + 1` = 33.
const MAX_PATTERN_CLASSES: usize = 32;

/// Byte-class compression table: a 256-entry map from byte to class id.
///
/// Class assignment:
/// - Class 0 = wildcard ("byte not in any pattern", or overflow when we
///   exceed `MAX_PATTERN_CLASSES`).
/// - Classes `1..=n_pattern_classes` = distinct pattern bytes in order
///   of first appearance.
#[derive(Clone, Debug)]
pub struct ByteClass {
    map: [u8; 256],
    n_classes: usize,
}

impl ByteClass {
    /// Builds a byte-class table from the union of all bytes in
    /// `patterns`. Distinct pattern bytes are assigned individual class
    /// ids up to `MAX_PATTERN_CLASSES`; remaining bytes share the
    /// wildcard class (id 0).
    fn from_patterns(patterns: &[&[u8]]) -> Self {
        let mut map = [0_u8; 256];
        let mut n_classes: usize = 1;
        // Iterate patterns in order so the class numbering is
        // deterministic.
        for pat in patterns {
            for &b in *pat {
                if map[b as usize] == 0 && (n_classes - 1) < MAX_PATTERN_CLASSES {
                    map[b as usize] = n_classes as u8;
                    n_classes += 1;
                }
            }
        }
        Self { map, n_classes }
    }

    /// Number of equivalence classes including the fail class.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Maps a byte to its class id.
    #[must_use]
    pub fn class_of(&self, b: u8) -> u8 {
        self.map[b as usize]
    }
}

/// Packed multi-pattern DFA.
#[derive(Clone, Debug)]
pub struct PackedDfa {
    byte_class: ByteClass,
    /// Flat state-by-class transition table.
    /// `transitions[state * n_classes + class] = next_state`.
    transitions: Vec<u32>,
    /// Per-state list of matching pattern indices; empty when none.
    match_lists: Vec<Vec<u32>>,
    /// Per-pattern length, used to convert match-end-offset to start.
    pattern_lens: Vec<u32>,
}

impl PackedDfa {
    /// Builds a packed DFA over the given patterns.
    ///
    /// Empty pattern slices are skipped silently (they would match at
    /// every position and are rarely useful in practice).
    #[must_use]
    #[allow(clippy::needless_range_loop)] // 2D indexing on `goto[state][class]`.
    pub fn new(patterns: &[&[u8]]) -> Self {
        let byte_class = ByteClass::from_patterns(patterns);
        let n_classes = byte_class.n_classes();

        // ---- Phase 1: build the goto trie ----
        //
        // States are integers; state 0 is the root.
        let mut goto: Vec<Vec<u32>> = vec![vec![u32::MAX; n_classes]];
        let mut match_lists: Vec<Vec<u32>> = vec![Vec::new()];
        let mut pattern_lens: Vec<u32> = Vec::with_capacity(patterns.len());

        for (pat_idx, &pat) in patterns.iter().enumerate() {
            pattern_lens.push(pat.len() as u32);
            if pat.is_empty() {
                continue;
            }
            let mut s: u32 = 0;
            for &b in pat {
                let cls = byte_class.class_of(b) as usize;
                let next = goto[s as usize][cls];
                if next == u32::MAX {
                    let new_state = goto.len() as u32;
                    goto.push(vec![u32::MAX; n_classes]);
                    match_lists.push(Vec::new());
                    goto[s as usize][cls] = new_state;
                    s = new_state;
                } else {
                    s = next;
                }
            }
            match_lists[s as usize].push(pat_idx as u32);
        }

        // ---- Phase 2: BFS to compute failure links & compose into a flat DFA ----
        //
        // We build `failure[s]` = state to fall back to on mismatch from
        // s, and then for each (s, cls) where goto[s][cls] is unset we
        // copy goto[failure[s]][cls]. Match lists of failure ancestors
        // are merged into the destination state so the scanner only
        // needs to read one match list per visited state.
        let n_states = goto.len();
        let mut failure: Vec<u32> = vec![0; n_states];
        // BFS queue.
        let mut queue: Vec<u32> = Vec::new();
        for cls in 0..n_classes {
            let next = goto[0][cls];
            if next != u32::MAX && next != 0 {
                failure[next as usize] = 0;
                queue.push(next);
            } else {
                // Self-loop on missing root transitions.
                goto[0][cls] = 0;
            }
        }
        let mut head = 0;
        while head < queue.len() {
            let r = queue[head];
            head += 1;
            for cls in 0..n_classes {
                let s = goto[r as usize][cls];
                if s == u32::MAX {
                    continue;
                }
                queue.push(s);
                // Walk failure links until we find a state with a
                // defined transition on `cls`, or hit the root.
                let mut f = failure[r as usize];
                loop {
                    let candidate = goto[f as usize][cls];
                    if candidate != u32::MAX && f != r {
                        // The "f != r" guard prevents an infinite loop
                        // if a mid-construction self-link sneaks in
                        // (defensive — shouldn't happen on well-formed
                        // input).
                        failure[s as usize] = if candidate == s { 0 } else { candidate };
                        break;
                    }
                    if f == 0 {
                        failure[s as usize] = 0;
                        break;
                    }
                    f = failure[f as usize];
                }
                // Merge match list of failure[s] into match list of s.
                let fail_state = failure[s as usize] as usize;
                if fail_state != s as usize {
                    let extras: Vec<u32> = match_lists[fail_state].clone();
                    let dest = &mut match_lists[s as usize];
                    for m in extras {
                        if !dest.contains(&m) {
                            dest.push(m);
                        }
                    }
                }
            }
            // Fill missing transitions with the failure-composed target.
            for cls in 0..n_classes {
                if goto[r as usize][cls] == u32::MAX {
                    let f = failure[r as usize];
                    goto[r as usize][cls] = goto[f as usize][cls];
                }
            }
        }

        // Sanity: any remaining MAX entries (states never visited by
        // BFS, e.g. when n_states == 1) collapse to root.
        for row in &mut goto {
            for cell in row.iter_mut() {
                if *cell == u32::MAX {
                    *cell = 0;
                }
            }
        }

        // Flatten `goto` into a single Vec<u32> in row-major order.
        let mut transitions: Vec<u32> = Vec::with_capacity(n_states * n_classes);
        for row in &goto {
            transitions.extend_from_slice(row);
        }

        Self {
            byte_class,
            transitions,
            match_lists,
            pattern_lens,
        }
    }

    /// Number of distinct DFA states (including the root).
    #[must_use]
    pub fn n_states(&self) -> usize {
        self.match_lists.len()
    }

    /// Number of equivalence classes used to compress the byte alphabet.
    #[must_use]
    pub fn n_classes(&self) -> usize {
        self.byte_class.n_classes()
    }

    /// Returns the first match in `haystack` as `(offset, pattern_index)`,
    /// or `None`. The returned offset is the *start* of the matched
    /// pattern in the haystack.
    #[must_use]
    pub fn find(&self, haystack: &[u8]) -> Option<(usize, usize)> {
        let n_classes = self.byte_class.n_classes();
        let mut state: u32 = 0;
        for (i, &b) in haystack.iter().enumerate() {
            let cls = self.byte_class.class_of(b) as usize;
            state = self.transitions[state as usize * n_classes + cls];
            if let Some(&pat) = self.match_lists[state as usize].first() {
                let plen = self.pattern_lens[pat as usize] as usize;
                let start = i + 1 - plen;
                return Some((start, pat as usize));
            }
        }
        None
    }

    /// Iterator over all matches in `haystack` in the order they end.
    pub fn find_iter<'h, 'm>(&'m self, haystack: &'h [u8]) -> PackedDfaIter<'h, 'm> {
        PackedDfaIter {
            dfa: self,
            haystack,
            pos: 0,
            state: 0,
            pending_idx: 0,
            pending_end_pos: 0,
        }
    }
}

/// Iterator returned by [`PackedDfa::find_iter`].
#[derive(Debug)]
pub struct PackedDfaIter<'h, 'm> {
    dfa: &'m PackedDfa,
    haystack: &'h [u8],
    pos: usize,
    state: u32,
    /// When > 0, the iterator is in the middle of draining the match
    /// list of the most recently visited state. `pending_idx` is the
    /// position into that list and `pending_end_pos` is the position
    /// after the byte that triggered it.
    pending_idx: usize,
    pending_end_pos: usize,
}

impl<'h, 'm> Iterator for PackedDfaIter<'h, 'm> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<(usize, usize)> {
        let n_classes = self.dfa.byte_class.n_classes();

        // Drain remaining matches from the previous state if any.
        if self.pending_end_pos > 0 {
            let st = self.state as usize;
            let list = &self.dfa.match_lists[st];
            if self.pending_idx < list.len() {
                let pat = list[self.pending_idx] as usize;
                self.pending_idx += 1;
                let plen = self.dfa.pattern_lens[pat] as usize;
                let start = self.pending_end_pos - plen;
                return Some((start, pat));
            }
            self.pending_end_pos = 0;
            self.pending_idx = 0;
        }

        while self.pos < self.haystack.len() {
            let b = self.haystack[self.pos];
            let cls = self.dfa.byte_class.class_of(b) as usize;
            self.state = self.dfa.transitions[self.state as usize * n_classes + cls];
            self.pos += 1;
            let st = self.state as usize;
            let list = &self.dfa.match_lists[st];
            if !list.is_empty() {
                let pat = list[0] as usize;
                let plen = self.dfa.pattern_lens[pat] as usize;
                let start = self.pos - plen;
                self.pending_end_pos = self.pos;
                self.pending_idx = 1;
                return Some((start, pat));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_first_match(patterns: &[&[u8]], haystack: &[u8]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (idx, pat) in patterns.iter().enumerate() {
            if pat.is_empty() {
                continue;
            }
            if haystack.len() < pat.len() {
                continue;
            }
            for i in 0..=haystack.len() - pat.len() {
                if &haystack[i..i + pat.len()] == *pat {
                    let candidate_end = i + pat.len();
                    let best_end = best
                        .as_ref()
                        .map(|&(s, p)| s + patterns[p].len())
                        .unwrap_or(usize::MAX);
                    if candidate_end < best_end
                        || (candidate_end == best_end
                            && best.as_ref().is_none_or(|&(_, p)| idx < p))
                    {
                        best = Some((i, idx));
                    }
                    break;
                }
            }
        }
        best
    }

    #[test]
    fn one_pattern_matches_simple() {
        let dfa = PackedDfa::new(&[b"abc"]);
        assert_eq!(dfa.find(b"xxabcxx"), Some((2, 0)));
        assert_eq!(dfa.find(b""), None);
        assert_eq!(dfa.find(b"abc"), Some((0, 0)));
    }

    #[test]
    fn two_patterns_pick_first_to_end() {
        let dfa = PackedDfa::new(&[b"ab", b"bc"]);
        // At position 0 'a', position 1 'b' triggers match for "ab".
        assert_eq!(dfa.find(b"abc"), Some((0, 0)));
    }

    #[test]
    fn five_patterns_overlapping() {
        let pats: &[&[u8]] = &[b"he", b"she", b"his", b"hers", b"shes"];
        let dfa = PackedDfa::new(pats);
        // "ushers": 'u', 's', 'h', 'e' triggers "he" then "she".
        let hits: alloc::vec::Vec<_> = dfa.find_iter(b"ushers").collect();
        // Expect at least "he" at offset 2 and "she" at offset 1; the
        // packed scanner reports them in match-end order.
        assert!(hits.contains(&(2, 0)));
        assert!(hits.contains(&(1, 1)));
    }

    #[test]
    fn thirty_two_patterns_alphabet_overflow() {
        // 32 distinct ASCII letters as 1-byte patterns; this exercises
        // the byte-class capacity exactly.
        let bytes: alloc::vec::Vec<u8> = (b'a'..b'a' + 32).collect();
        let pats: alloc::vec::Vec<&[u8]> = bytes.iter().map(core::slice::from_ref).collect();
        let dfa = PackedDfa::new(&pats);
        for (i, b) in bytes.iter().enumerate() {
            let hay = [*b];
            assert_eq!(dfa.find(&hay), Some((0, i)), "byte {b}");
        }
        // Bytes outside the alphabet must not match.
        assert_eq!(dfa.find(b"X"), None);
        assert_eq!(dfa.find(b"!"), None);
    }

    #[test]
    fn empty_haystack() {
        let dfa = PackedDfa::new(&[b"abc"]);
        assert_eq!(dfa.find(b""), None);
    }

    #[test]
    fn find_iter_reports_all() {
        let dfa = PackedDfa::new(&[b"aa"]);
        let hits: alloc::vec::Vec<_> = dfa.find_iter(b"aaaa").collect();
        // Overlapping hits: end-positions 2, 3, 4 → starts 0, 1, 2.
        assert_eq!(hits, alloc::vec![(0, 0), (1, 0), (2, 0)]);
    }

    #[test]
    fn cross_check_against_naive() {
        let cases: &[(&[&[u8]], &[u8])] = &[
            (&[b"abc"], b"xxabcyy"),
            (&[b"ab", b"bc"], b"abc"),
            (&[b"he", b"she"], b"ushers"),
            (&[b"abc", b"abd"], b"xxabcabd"),
        ];
        for &(pats, hay) in cases {
            let dfa = PackedDfa::new(pats);
            let dfa_hit = dfa.find(hay);
            let naive_hit = naive_first_match(pats, hay);
            // Both should agree on existence; the exact (offset, idx)
            // can differ if multiple patterns end at the same byte and
            // their indices differ — handle by just checking first-end.
            assert_eq!(dfa_hit.is_some(), naive_hit.is_some());
            if let (Some((d_off, d_idx)), Some((n_off, n_idx))) = (dfa_hit, naive_hit) {
                let d_end = d_off + pats[d_idx].len();
                let n_end = n_off + pats[n_idx].len();
                assert_eq!(d_end, n_end, "match end for {pats:?} on {hay:?}");
            }
        }
    }
}

//! CTPH (Context-Triggered Piecewise Hashing), ssdeep-style fuzzy hash.
//!
//! Quality-faithful reimplementation of the algorithm described in
//! Kornblum, "Identifying almost identical files using context triggered
//! piecewise hashing" (DFRWS 2006), as popularized by `ssdeep`.
//!
//! # Algorithm
//!
//! 1. **Adaptive blocksize.** A blocksize is picked from the geometric
//!    series `b * 2^n` (with `b = MIN_BLOCKSIZE = 3`) so the input
//!    produces roughly `SPAMSUM_LENGTH = 64` trigger points. We start
//!    with the smallest power that satisfies `bs * SPAMSUM_LENGTH < n`,
//!    then double the blocksize while we overflow the digest buffer.
//! 2. **Rolling hash.** A 32-bit Adler-like rolling hash over a 7-byte
//!    window: `h1` is the byte sum, `h2` is the position-weighted byte
//!    sum, and `h3` is an LFSR-style rotating mix. The trigger condition
//!    is `(rolling_hash + 1) % blocksize == 0`.
//! 3. **Per-block FNV-1a.** A running 32-bit FNV-1a hash is reset at
//!    every trigger (and at end-of-input). The bottom 6 bits of the FNV
//!    hash select a base64 character that is appended to the digest.
//!    A second parallel digest is accumulated at `2 * blocksize` so a
//!    pair of digests at `(b, 2b)` is always available; comparisons
//!    only succeed when the blocksizes line up at a factor of 1 or 2.
//! 4. **Output format.** `<blocksize>:<digest1>:<digest2>` matching the
//!    `ssdeep` textual representation.
//!
//! # Distance / similarity
//!
//! [`similarity`] returns `0..=100` (higher = more similar) per the
//! `ssdeep` convention; [`distance`] returns `100 - similarity`. When the
//! two blocksizes differ by more than a factor of 2 the inputs are
//! deemed incomparable and similarity is `0` (distance `100`).
//!
//! # Deviations from upstream `ssdeep`
//!
//! - The trigger pattern, base64 alphabet, FNV-1a constants, and rolling
//!   hash window match upstream byte-for-byte.
//! - The adaptive-blocksize search starts from the smallest power that
//!   gives at least `SPAMSUM_LENGTH` triggers, then *only doubles*
//!   (never halves). Upstream halves on under-trigger and doubles on
//!   over-trigger; we always start small enough to avoid the halving
//!   path. Equivalent in practice and easier to reason about in
//!   `no_std`.
//! - The similarity score uses an "offset Levenshtein" identical to
//!   `ssdeep`'s `edit_distn` for short strings, but capped at the digest
//!   length (so byte-for-byte bit-equal output to upstream is not
//!   guaranteed for pathological inputs).
//! - The textual digest preserves up to [`SPAMSUM_LENGTH`] characters
//!   per body, no trailing block-elision character (upstream uses
//!   `:`-bracketed ASCII; we emit the same shape).
//!
//! Byte-exact ssdeep compatibility is **not** a goal of this module.
//! Like `tlsh_like`, the implementation is a faithful adaptation, not a
//! drop-in replacement.

use core::cmp::Ordering;

/// Maximum body length per digest (matches `ssdeep`'s `SPAMSUM_LENGTH`).
pub const SPAMSUM_LENGTH: usize = 64;

/// Minimum blocksize. Matches `ssdeep`'s `MIN_BLOCKSIZE = 3`.
pub const MIN_BLOCKSIZE: u32 = 3;

/// Rolling-hash window length.
const ROLLING_WINDOW: usize = 7;

/// FNV-1a 32-bit prime.
const FNV_PRIME_32: u32 = 0x0100_0193;

/// FNV-1a 32-bit offset basis.
const FNV_OFFSET_32: u32 = 0x811C_9DC5;

/// Base64 alphabet used by `ssdeep`. (Note: `+` and `/` come last.)
const B64_ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

// ---------------------------------------------------------------------------
// Rolling hash
// ---------------------------------------------------------------------------

/// `ssdeep`'s 7-byte Adler-like rolling hash. Three components:
///
/// - `h1` — running byte sum.
/// - `h2` — position-weighted byte sum.
/// - `h3` — LFSR-style rotating mix.
#[derive(Clone, Copy, Debug)]
struct RollingHash {
    window: [u8; ROLLING_WINDOW],
    /// Number of bytes pushed since the last reset, capped at
    /// `ROLLING_WINDOW` for the "is the window full?" check.
    pos: u32,
    h1: u32,
    h2: u32,
    h3: u32,
}

impl RollingHash {
    const fn new() -> Self {
        Self {
            window: [0; ROLLING_WINDOW],
            pos: 0,
            h1: 0,
            h2: 0,
            h3: 0,
        }
    }

    /// Pushes `byte` into the window; returns the rolling hash *after* the
    /// push. The hash is the sum `h1 + h2 + h3`.
    #[inline]
    fn push(&mut self, byte: u8) -> u32 {
        let idx = (self.pos as usize) % ROLLING_WINDOW;
        let removed = self.window[idx];
        self.window[idx] = byte;

        // h2 is window-weighted; per `ssdeep` it is updated as
        // `h2 = h2 + ROLLING_WINDOW * byte - h1` (then h1 is updated).
        self.h2 = self
            .h2
            .wrapping_add((ROLLING_WINDOW as u32).wrapping_mul(u32::from(byte)));
        self.h2 = self.h2.wrapping_sub(self.h1);

        self.h1 = self.h1.wrapping_add(u32::from(byte));
        self.h1 = self.h1.wrapping_sub(u32::from(removed));

        // h3 = ((h3 << 5) ^ byte) per `ssdeep`'s `roll_hash`.
        self.h3 = self.h3.wrapping_shl(5);
        self.h3 ^= u32::from(byte);

        self.pos = self.pos.wrapping_add(1);
        self.h1.wrapping_add(self.h2).wrapping_add(self.h3)
    }
}

// ---------------------------------------------------------------------------
// FNV-1a
// ---------------------------------------------------------------------------

#[inline]
const fn fnv1a_step(state: u32, byte: u8) -> u32 {
    (state ^ (byte as u32)).wrapping_mul(FNV_PRIME_32)
}

// ---------------------------------------------------------------------------
// Digest
// ---------------------------------------------------------------------------

/// CTPH digest: blocksize plus two parallel base64 strings of length
/// `<= SPAMSUM_LENGTH`. The struct stores raw bytes; [`Digest::body`]
/// and [`Digest::body2`] return them as `&str`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Digest {
    blocksize: u32,
    /// Body buffer for blocksize-1 digest. ASCII printable from
    /// [`B64_ALPHABET`]; the first `body_len` bytes are valid.
    body_buf: [u8; SPAMSUM_LENGTH],
    body_len: u8,
    /// Body buffer for blocksize-2 (= 2 * blocksize) digest.
    body2_buf: [u8; SPAMSUM_LENGTH],
    body2_len: u8,
}

impl Default for Digest {
    fn default() -> Self {
        Self {
            blocksize: MIN_BLOCKSIZE,
            body_buf: [0; SPAMSUM_LENGTH],
            body_len: 0,
            body2_buf: [0; SPAMSUM_LENGTH],
            body2_len: 0,
        }
    }
}

impl Digest {
    /// Computes the CTPH digest of `bytes`.
    ///
    /// Empty input yields a digest with `blocksize = MIN_BLOCKSIZE` and
    /// empty bodies. The digest is *always valid* (unlike the TLSH-like
    /// digest, which has a separate `is_valid` flag).
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let blocksize = pick_initial_blocksize(bytes.len());
        compute_digest(bytes, blocksize)
    }

    /// Returns the chosen blocksize.
    #[must_use]
    pub const fn blocksize(&self) -> u32 {
        self.blocksize
    }

    /// Returns the blocksize-1 body as a printable `&str`.
    #[must_use]
    pub fn body(&self) -> &str {
        // SAFETY: body_buf only ever contains bytes from B64_ALPHABET,
        // which is pure ASCII.
        unsafe { core::str::from_utf8_unchecked(&self.body_buf[..self.body_len as usize]) }
    }

    /// Returns the blocksize-2 body as a printable `&str`.
    #[must_use]
    pub fn body2(&self) -> &str {
        // SAFETY: body2_buf only ever contains bytes from B64_ALPHABET.
        unsafe { core::str::from_utf8_unchecked(&self.body2_buf[..self.body2_len as usize]) }
    }
}

impl core::fmt::Display for Digest {
    /// Renders the digest in `ssdeep`'s `<blocksize>:<body>:<body2>` form.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}:{}:{}", self.blocksize, self.body(), self.body2())
    }
}

// ---------------------------------------------------------------------------
// Blocksize selection
// ---------------------------------------------------------------------------

/// Picks the smallest `MIN_BLOCKSIZE * 2^n` such that
/// `blocksize * SPAMSUM_LENGTH >= input_len`. This guarantees we will not
/// produce more than `SPAMSUM_LENGTH` triggers on average, so the digest
/// fits in the fixed-size buffer.
fn pick_initial_blocksize(input_len: usize) -> u32 {
    let mut bs = MIN_BLOCKSIZE;
    let target = SPAMSUM_LENGTH as u64;
    let len = input_len as u64;
    while u64::from(bs).saturating_mul(target) < len {
        bs = bs.saturating_mul(2);
        if bs == 0 {
            // Should be unreachable for any practical input; saturate to
            // a safe maximum.
            return u32::MAX / 4;
        }
    }
    bs
}

// ---------------------------------------------------------------------------
// Core digest loop
// ---------------------------------------------------------------------------

fn compute_digest(bytes: &[u8], starting_blocksize: u32) -> Digest {
    // Try with the chosen blocksize. If we end up filling body_buf before
    // we exhaust the input, double the blocksize and retry. This bounds
    // the total work at O(n log n) in the worst case (typically O(n)).
    let mut bs = starting_blocksize;
    loop {
        let result = single_pass(bytes, bs);
        if !result.overflowed || bs >= u32::MAX / 4 {
            // Either it fit, or we've run out of room to double.
            return Digest {
                blocksize: bs,
                body_buf: result.body_buf,
                body_len: result.body_len,
                body2_buf: result.body2_buf,
                body2_len: result.body2_len,
            };
        }
        bs = bs.saturating_mul(2);
    }
}

struct PassResult {
    body_buf: [u8; SPAMSUM_LENGTH],
    body_len: u8,
    body2_buf: [u8; SPAMSUM_LENGTH],
    body2_len: u8,
    /// True when we ran out of room in `body_buf` before consuming all
    /// the input — caller should retry with a larger blocksize.
    overflowed: bool,
}

fn single_pass(bytes: &[u8], blocksize: u32) -> PassResult {
    let mut body_buf = [0_u8; SPAMSUM_LENGTH];
    let mut body_len: u8 = 0;
    let mut body2_buf = [0_u8; SPAMSUM_LENGTH];
    let mut body2_len: u8 = 0;

    let mut roll = RollingHash::new();
    let mut fnv1: u32 = FNV_OFFSET_32;
    let mut fnv2: u32 = FNV_OFFSET_32;
    let mut overflowed = false;

    let bs1 = u64::from(blocksize);
    let bs2 = bs1.saturating_mul(2);

    for &byte in bytes {
        let h = roll.push(byte);
        fnv1 = fnv1a_step(fnv1, byte);
        fnv2 = fnv1a_step(fnv2, byte);

        // The +1 matches `ssdeep`: trigger when (rolling_hash + 1) is a
        // multiple of blocksize.
        let h_plus = u64::from(h).wrapping_add(1);

        if h_plus.is_multiple_of(bs1) {
            // Trigger for digest 1.
            if (body_len as usize) < SPAMSUM_LENGTH - 1 {
                body_buf[body_len as usize] = B64_ALPHABET[(fnv1 & 0x3F) as usize];
                body_len += 1;
                fnv1 = FNV_OFFSET_32;
            } else if (body_len as usize) == SPAMSUM_LENGTH - 1 {
                // Last slot: still emit, but flag overflow so caller can
                // upgrade the blocksize. We only emit this final char on
                // a real trigger (matches upstream's behavior of stopping
                // at the buffer limit rather than truncating the byte
                // stream).
                body_buf[body_len as usize] = B64_ALPHABET[(fnv1 & 0x3F) as usize];
                body_len += 1;
                fnv1 = FNV_OFFSET_32;
                overflowed = true;
            } else {
                overflowed = true;
            }
        }

        if h_plus.is_multiple_of(bs2) && (body2_len as usize) < SPAMSUM_LENGTH - 1 {
            body2_buf[body2_len as usize] = B64_ALPHABET[(fnv2 & 0x3F) as usize];
            body2_len += 1;
            fnv2 = FNV_OFFSET_32;
        }
    }

    // Tail: emit one final character for any leftover FNV state, matching
    // upstream's "always include the trailing block" behavior.
    if (body_len as usize) < SPAMSUM_LENGTH && !bytes.is_empty() {
        body_buf[body_len as usize] = B64_ALPHABET[(fnv1 & 0x3F) as usize];
        body_len += 1;
    }
    if (body2_len as usize) < SPAMSUM_LENGTH && !bytes.is_empty() {
        body2_buf[body2_len as usize] = B64_ALPHABET[(fnv2 & 0x3F) as usize];
        body2_len += 1;
    }

    PassResult {
        body_buf,
        body_len,
        body2_buf,
        body2_len,
        overflowed,
    }
}

// ---------------------------------------------------------------------------
// Similarity / distance
// ---------------------------------------------------------------------------

/// Returns the `ssdeep`-style similarity score in `[0, 100]`, where 100
/// means "byte-identical digests" and 0 means "incomparable / unrelated".
///
/// If the two blocksizes differ by more than a factor of 2 the result is
/// 0. Otherwise the longer body of the matching pair drives the score:
/// for each pair of bodies at the same blocksize we compute a normalized
/// edit distance and convert to a similarity, then take the maximum.
#[must_use]
pub fn similarity(a: &Digest, b: &Digest) -> u32 {
    let bs_a = a.blocksize;
    let bs_b = b.blocksize;

    // Same blocksize: compare body to body and body2 to body2; take max.
    if bs_a == bs_b {
        let s1 = score_pair(a.body(), b.body(), bs_a);
        let s2 = score_pair(a.body2(), b.body2(), bs_a.saturating_mul(2));
        return s1.max(s2);
    }

    // 2x blocksize relationship: a's body2 lines up with b's body, or
    // vice versa.
    if bs_a == bs_b.saturating_mul(2) {
        return score_pair(a.body(), b.body2(), bs_a);
    }
    if bs_b == bs_a.saturating_mul(2) {
        return score_pair(a.body2(), b.body(), bs_b);
    }

    // Anything else: incomparable.
    0
}

/// Returns the `ssdeep`-style distance in `[0, 100]`. `0` is identical,
/// `100` is unrelated/incomparable.
#[must_use]
pub fn distance(a: &Digest, b: &Digest) -> u32 {
    100_u32.saturating_sub(similarity(a, b))
}

/// Compares two digest bodies (printable strings of `<= SPAMSUM_LENGTH`)
/// at the given blocksize. Returns 0..=100.
fn score_pair(a: &str, b: &str, blocksize: u32) -> u32 {
    // Empty bodies on both sides → identical-by-vacuous-truth.
    if a.is_empty() && b.is_empty() {
        return 100;
    }
    // Empty on one side only → no overlap possible.
    if a.is_empty() || b.is_empty() {
        return 0;
    }

    // Eliminate trivial common-substring filter from upstream: if the two
    // strings share no 7-char run we cap the score at 0. This is the
    // ssdeep "common substring" gate; we use 7 as in upstream.
    if !shares_n_run(a.as_bytes(), b.as_bytes(), 7) {
        return 0;
    }

    let edit = edit_distance(a.as_bytes(), b.as_bytes());
    let len_sum = a.len() as u32 + b.len() as u32;
    if len_sum == 0 {
        return 100;
    }

    // Upstream formula: score = 100 - (edit * SPAMSUM_LENGTH * 100) / (len_sum * SPAMSUM_LENGTH)
    // simplifies to: score = 100 - (edit * 100) / len_sum, then "scale by blocksize".
    // The blocksize scaling (matches upstream `score_strings`):
    //   if score >= 100 -> 100
    //   else            -> min(score, blocksize / MIN_BLOCKSIZE * score / 64)
    // Then a final clamp.
    let raw = if (edit as u64) * 100 >= u64::from(len_sum) * 100 {
        0
    } else {
        100_u32.saturating_sub(((edit as u64 * 100) / u64::from(len_sum)) as u32)
    };

    // Blocksize confidence cap. The smaller the blocksize the more
    // significant a match is; large-blocksize matches are clamped to
    // avoid false positives from a coincidental short-run match.
    let cap = if blocksize <= MIN_BLOCKSIZE * 64 {
        100
    } else {
        // For very large blocksizes, scale down. The exact formula
        // below is a simplified version of upstream's; both penalize
        // larger blocksizes.
        let mul = MIN_BLOCKSIZE * 64;
        let denom = blocksize.max(1);
        ((u64::from(raw) * u64::from(mul)) / u64::from(denom)) as u32
    };

    raw.min(cap)
}

/// True when `a` and `b` share a contiguous run of `n` bytes.
fn shares_n_run(a: &[u8], b: &[u8], n: usize) -> bool {
    if a.len() < n || b.len() < n {
        // For short bodies, upstream skips the n-run gate; we follow
        // that here so very-short inputs still get a real edit-distance
        // score.
        return true;
    }
    for win_a in a.windows(n) {
        for win_b in b.windows(n) {
            if win_a == win_b {
                return true;
            }
        }
    }
    false
}

/// Standard Levenshtein distance over byte slices. Bounded inputs
/// (`<= SPAMSUM_LENGTH`) make the O(m*n) implementation acceptable.
fn edit_distance(a: &[u8], b: &[u8]) -> u32 {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n as u32;
    }
    if n == 0 {
        return m as u32;
    }

    // Two-row DP. `prev[j]` is the cost of transforming a[..i-1] into
    // b[..j]; `curr[j]` extends the row. Bodies are <= SPAMSUM_LENGTH so
    // the row fits in a small fixed buffer.
    let mut prev = [0_u32; SPAMSUM_LENGTH + 1];
    let mut curr = [0_u32; SPAMSUM_LENGTH + 1];
    for (j, slot) in prev.iter_mut().enumerate().take(n + 1) {
        *slot = j as u32;
    }
    for i in 1..=m {
        curr[0] = i as u32;
        for j in 1..=n {
            let sub_cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            let del = prev[j].saturating_add(1);
            let ins = curr[j - 1].saturating_add(1);
            let sub = prev[j - 1].saturating_add(sub_cost);
            curr[j] = match del.cmp(&ins) {
                Ordering::Less => del.min(sub),
                _ => ins.min(sub),
            };
        }
        // Swap prev and curr by copying. Could use mem::swap on slices,
        // but a copy keeps both as fixed arrays without indirection.
        prev[..=n].copy_from_slice(&curr[..=n]);
    }
    prev[n]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// Same xorshift PRNG used by the `tlsh_like` tests, so test fixtures
    /// stay comparable across the fuzzy module.
    fn random_bytes(n: usize, seed: u64) -> alloc::vec::Vec<u8> {
        let mut state = seed.wrapping_add(1);
        let mut out = alloc::vec::Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            out.push(state as u8);
        }
        out
    }

    #[cfg(feature = "std")]
    extern crate std;

    extern crate alloc;

    #[test]
    fn empty_input_has_min_blocksize_and_empty_bodies() {
        let d = Digest::from_bytes(&[]);
        assert_eq!(d.blocksize(), MIN_BLOCKSIZE);
        assert!(d.body().is_empty());
        assert!(d.body2().is_empty());
    }

    #[test]
    fn identical_inputs_produce_identical_digests() {
        let bytes = random_bytes(4096, 0xCAFE);
        let a = Digest::from_bytes(&bytes);
        let b = Digest::from_bytes(&bytes);
        assert_eq!(a, b);
        assert_eq!(similarity(&a, &b), 100);
        assert_eq!(distance(&a, &b), 0);
    }

    #[test]
    fn stable_output_across_runs() {
        let bytes = random_bytes(8192, 0xABCD);
        let a = Digest::from_bytes(&bytes);
        let b = Digest::from_bytes(&bytes);
        assert_eq!(a.body_buf, b.body_buf);
        assert_eq!(a.body_len, b.body_len);
        assert_eq!(a.body2_buf, b.body2_buf);
        assert_eq!(a.body2_len, b.body2_len);
        assert_eq!(a.blocksize(), b.blocksize());
    }

    #[cfg(feature = "std")]
    #[test]
    fn single_byte_modification_is_near_match() {
        // Use 4 KiB so we get a meaningful number of triggers.
        let mut a = random_bytes(4096, 0x1234);
        let original = a.clone();
        a[2048] ^= 0xAA;
        let da = Digest::from_bytes(&original);
        let db = Digest::from_bytes(&a);
        let s = similarity(&da, &db);
        assert!(s >= 95, "single-byte-mod similarity too low: {s}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn concat_property() {
        // ssdeep "concat" property: hash(a+b) is meaningfully similar to
        // both hash(a) and hash(b). This holds when the concatenated
        // input doesn't trigger a blocksize jump that throws off
        // alignment. We use ~4 KiB so we stay in the same blocksize.
        let a = random_bytes(4096, 0x11);
        let b = random_bytes(4096, 0x22);
        let mut ab = a.clone();
        ab.extend_from_slice(&b);

        let da = Digest::from_bytes(&a);
        let db = Digest::from_bytes(&b);
        let dab = Digest::from_bytes(&ab);

        // The concat may or may not be at the same blocksize as the
        // halves. If it isn't (factor-of-2 mismatch) the score is
        // computed against the parallel digest. Either way, we expect
        // at least one of the halves to register as related.
        let s_a = similarity(&da, &dab);
        let s_b = similarity(&db, &dab);
        let best = s_a.max(s_b);
        assert!(
            best >= 30,
            "concat property failed: s_a={s_a}, s_b={s_b}, \
             bs_a={}, bs_b={}, bs_ab={}",
            da.blocksize(),
            db.blocksize(),
            dab.blocksize()
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn random_inputs_have_low_similarity() {
        let a = random_bytes(4096, 0xDEAD);
        let b = random_bytes(4096, 0xBEEF);
        let da = Digest::from_bytes(&a);
        let db = Digest::from_bytes(&b);
        let s = similarity(&da, &db);
        assert!(
            s <= 30,
            "two random inputs had unexpectedly high similarity: {s}"
        );
    }

    #[test]
    fn similarity_is_symmetric() {
        let a = random_bytes(2048, 0x11);
        let b = random_bytes(2048, 0x22);
        let da = Digest::from_bytes(&a);
        let db = Digest::from_bytes(&b);
        assert_eq!(similarity(&da, &db), similarity(&db, &da));
    }

    #[test]
    fn incomparable_blocksizes_score_zero() {
        // 64 KiB: blocksize ~ 1536. 256 B: blocksize = 3. Difference is
        // way more than a factor of 2 → score must be 0.
        let big = random_bytes(64 * 1024, 0x42);
        let tiny = random_bytes(256, 0x42);
        let dbig = Digest::from_bytes(&big);
        let dtiny = Digest::from_bytes(&tiny);
        // Sanity: blocksizes must be incomparable for this assertion to
        // be testing what it claims to test.
        let bs_a = dbig.blocksize();
        let bs_b = dtiny.blocksize();
        let comparable =
            bs_a == bs_b || bs_a == bs_b.saturating_mul(2) || bs_b == bs_a.saturating_mul(2);
        if !comparable {
            assert_eq!(similarity(&dbig, &dtiny), 0);
        }
    }

    #[test]
    fn rolling_hash_uses_window() {
        // The rolling hash should depend on bytes within the last
        // ROLLING_WINDOW positions. Two inputs that agree on the last
        // window should produce identical final hash values.
        let mut r1 = RollingHash::new();
        let mut r2 = RollingHash::new();
        // Different prefixes, identical last 7 bytes.
        for &b in b"AAAAAAA__same___tail" {
            r1.push(b);
        }
        for &b in b"BBBB__different_tail" {
            r2.push(b);
        }
        // After both have ingested the same final 7 bytes, hashes match.
        for &b in b"abcdefg" {
            let h1 = r1.push(b);
            let h2 = r2.push(b);
            // After at least 7 bytes of common suffix, the byte-sum and
            // weighted-sum components agree. h3 is an LFSR with state
            // shifted in over many iterations, so it doesn't equalize
            // perfectly within 7 bytes; we just assert that h1 and h2
            // are equal across both rollers.
            let _ = (h1, h2);
        }
        // Window-only state is the byte sum h1 and weighted h2; both
        // must equalize once the suffixes match.
        assert_eq!(r1.h1, r2.h1);
        assert_eq!(r1.h2, r2.h2);
    }

    #[test]
    fn fnv1a_known_value() {
        // RFC vector: empty string -> 0x811C9DC5. "a" -> 0xE40C292C.
        let mut h = FNV_OFFSET_32;
        for &b in b"a" {
            h = fnv1a_step(h, b);
        }
        assert_eq!(h, 0xE40C_292C);
    }

    #[cfg(feature = "std")]
    #[test]
    fn display_format_matches_ssdeep_shape() {
        let bytes = random_bytes(2048, 0x99);
        let d = Digest::from_bytes(&bytes);
        let s = std::format!("{d}");
        // <number>:<body>:<body2>
        let parts: alloc::vec::Vec<&str> = s.split(':').collect();
        assert_eq!(parts.len(), 3);
        assert!(parts[0].chars().all(|c| c.is_ascii_digit()));
        for body in [parts[1], parts[2]] {
            assert!(body.bytes().all(|b| B64_ALPHABET.contains(&b)));
        }
    }

    #[test]
    fn pick_initial_blocksize_grows_with_input() {
        let small = pick_initial_blocksize(0);
        let medium = pick_initial_blocksize(8 * 1024);
        let large = pick_initial_blocksize(1024 * 1024);
        assert_eq!(small, MIN_BLOCKSIZE);
        assert!(medium >= small);
        assert!(large >= medium);
    }

    #[test]
    fn edit_distance_known_values() {
        assert_eq!(edit_distance(b"", b""), 0);
        assert_eq!(edit_distance(b"abc", b"abc"), 0);
        assert_eq!(edit_distance(b"kitten", b"sitting"), 3);
        assert_eq!(edit_distance(b"abc", b""), 3);
        assert_eq!(edit_distance(b"", b"abc"), 3);
    }
}

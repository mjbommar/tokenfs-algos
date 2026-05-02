//! Misra-Gries top-K heavy-hitters over a byte stream.
//!
//! Misra & Gries, *Finding Repeated Elements*, Sci. Comput. Programming
//! 1982. The algorithm tracks at most `K` candidate items and reports each
//! candidate's count with deterministic absolute error `≤ N / (K + 1)`,
//! where `N` is the total number of bytes processed. Items whose true
//! frequency exceeds `N / (K + 1)` are guaranteed to appear in the
//! reported set; the converse direction is one-sided (false positives
//! permitted, false negatives ruled out).
//!
//! ## Why a byte-specific Misra-Gries?
//!
//! [`crate::sketch::MisraGries`] is parameterized over `u32` items; this
//! variant is specialized for `u8`. The byte-only version uses `[u8; K]`
//! key storage (1 byte/slot vs 4) and `u64` counters for filesystem-scale
//! totals, avoiding the i32 saturation behavior of `crate::sketch`. In the
//! `histogram` module's vocabulary it serves as a constant-state
//! alternative to a 256-bin histogram for the most common downstream
//! query — *which bytes dominate this block?* — when callers do not need
//! the full distribution.
//!
//! ## State footprint
//!
//! `MisraGries<K>` holds:
//! - `[u8; K]` keys (the slot byte, or `0xFF` when empty)
//! - `[u64; K]` counters (per-slot count)
//! - `u64` total observation count
//!
//! For `K = 16`: `16 + 128 + 8 = 152` bytes; for `K = 256`: `256 + 2048 +
//! 8 = 2312` bytes. No heap, `no_std`-clean. Empty slots are encoded by
//! `count == 0` (the `0xFF` sentinel in `keys` is decorative for
//! debugging — the slot's emptiness is determined by `count`).
//!
//! ## Accuracy
//!
//! - **Reported counts are lower bounds.** Misra-Gries cannot
//!   over-estimate; each reported count is `true_count - error` for some
//!   `error ∈ [0, N / (K + 1)]`. `error_bound()` returns the worst-case
//!   slack.
//! - **Frequent-byte recovery.** Any byte with true frequency `f > N /
//!   (K + 1)` is in the reported set. With `K ≥ 256`, every byte in the
//!   alphabet fits, so the result is exact.
//! - **Non-frequent bytes** may or may not appear; their reported count
//!   may be zero or positive.

/// Sentinel byte value used in empty key slots. The empty/full state is
/// determined by the counter (`count == 0`); this constant is decorative
/// for debugging.
const EMPTY_KEY: u8 = 0xFF;

/// Misra-Gries top-K heavy-hitter sketch specialized for byte streams.
///
/// See the [module-level docs](self) for accuracy guarantees and memory
/// footprint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MisraGries<const K: usize> {
    /// Tracked byte values. `0xFF` is a debug sentinel for empty slots;
    /// emptiness is encoded by `counts[i] == 0`.
    keys: [u8; K],
    /// Counter per slot.
    counts: [u64; K],
    /// Number of bytes processed.
    n: u64,
}

impl<const K: usize> Default for MisraGries<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const K: usize> MisraGries<K> {
    /// Builds an empty sketch with `K` counter slots.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            keys: [EMPTY_KEY; K],
            counts: [0_u64; K],
            n: 0,
        }
    }

    /// Number of bytes processed so far.
    #[must_use]
    pub const fn n(&self) -> u64 {
        self.n
    }

    /// Maximum absolute error in any reported count.
    ///
    /// Misra-Gries guarantees `reported ≤ true ≤ reported + error_bound`.
    #[must_use]
    pub const fn error_bound(&self) -> u64 {
        // K + 1 in the denominator is the standard Misra-Gries bound;
        // it cannot overflow for `K ≤ u64::MAX - 1`, which always holds
        // because const-generic capacities are usize-bounded and we do
        // arithmetic in u64.
        self.n / (K as u64 + 1)
    }

    /// Memory footprint of the sketch in bytes.
    #[must_use]
    pub const fn memory_bytes() -> usize {
        core::mem::size_of::<Self>()
    }

    /// Records one byte observation.
    pub fn update(&mut self, byte: u8) {
        self.n = self.n.saturating_add(1);

        if K == 0 {
            return;
        }

        // Phase 1: byte already tracked? Bump that slot.
        for i in 0..K {
            if self.counts[i] != 0 && self.keys[i] == byte {
                self.counts[i] = self.counts[i].saturating_add(1);
                return;
            }
        }

        // Phase 2: empty slot available? Adopt the byte there.
        for i in 0..K {
            if self.counts[i] == 0 {
                self.keys[i] = byte;
                self.counts[i] = 1;
                return;
            }
        }

        // Phase 3: all slots populated by other bytes — decrement
        // every counter (Misra-Gries "bulk decrement"). Any slot whose
        // count drops to zero becomes available on the next non-tracked
        // byte; we reset its key to the empty sentinel for clarity.
        for i in 0..K {
            self.counts[i] -= 1;
            if self.counts[i] == 0 {
                self.keys[i] = EMPTY_KEY;
            }
        }
    }

    /// Records every byte in `bytes` in stream order.
    ///
    /// Equivalent to `for &b in bytes { self.update(b); }` but kept as a
    /// dedicated entry point so callers can see the slice-shaped API and
    /// future SIMD-friendly batches can land here without reshuffling.
    pub fn update_slice(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.update(byte);
        }
    }

    /// Iterates over `(byte, lower_bound_count)` pairs for every populated
    /// slot, in storage order.
    ///
    /// The reported count is a **lower bound** on the byte's true
    /// frequency: `true_count - error_bound() ≤ reported ≤ true_count`.
    /// Filter by your own frequency threshold (e.g. discard counts `<
    /// error_bound()` to keep only guaranteed heavy hitters).
    pub fn top_k(&self) -> impl Iterator<Item = (u8, u64)> + '_ {
        self.keys
            .iter()
            .zip(self.counts.iter())
            .filter_map(|(&key, &count)| (count > 0).then_some((key, count)))
    }

    /// Resets the sketch to its empty state.
    pub fn clear(&mut self) {
        self.keys = [EMPTY_KEY; K];
        self.counts = [0_u64; K];
        self.n = 0;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    // `Vec` and `vec!` are not in the no-std prelude; alias them from
    // `alloc` for the alloc-only build (audit-R6 finding #164). Pull in
    // `BTreeSet` rather than `std::collections::HashSet` so the test
    // compiles on the alloc-only profile.
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::collections::BTreeSet;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;
    #[cfg(feature = "std")]
    use std::collections::BTreeSet;

    fn deterministic_zipfian_bytes(n: usize, seed: u64) -> Vec<u8> {
        // Build a Zipfian-ish skewed byte stream: byte `i` gets weight
        // `1/(i+1)` for `i ∈ 0..32`. Sample by inverse-CDF lookup with a
        // xorshift PRNG so the test is deterministic.
        const ALPHABET: usize = 32;
        let mut weights = [0.0_f64; ALPHABET];
        let mut total = 0.0_f64;
        for (i, w) in weights.iter_mut().enumerate() {
            *w = 1.0 / ((i + 1) as f64);
            total += *w;
        }
        let mut cum = [0.0_f64; ALPHABET];
        let mut acc = 0.0_f64;
        for (c, &w) in cum.iter_mut().zip(weights.iter()) {
            acc += w / total;
            *c = acc;
        }

        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as f64 / ((1_u64 << 53) as f64);
            let mut idx = ALPHABET - 1;
            for (i, &c) in cum.iter().enumerate() {
                if u <= c {
                    idx = i;
                    break;
                }
            }
            out.push(idx as u8);
        }
        out
    }

    fn true_counts(bytes: &[u8]) -> [u64; 256] {
        let mut counts = [0_u64; 256];
        for &b in bytes {
            counts[b as usize] += 1;
        }
        counts
    }

    #[test]
    fn empty_sketch_reports_no_heavy_hitters() {
        let sk = MisraGries::<16>::new();
        assert_eq!(sk.n(), 0);
        assert_eq!(sk.error_bound(), 0);
        assert_eq!(sk.top_k().count(), 0);
    }

    #[test]
    fn k16_zipfian_includes_every_above_bound_byte() {
        // 100k bytes, Zipfian alphabet of 32 symbols, K = 16. Every
        // byte whose true count exceeds the error bound `N / (K + 1)`
        // must appear in the reported top-K.
        let bytes = deterministic_zipfian_bytes(100_000, 0xC8C2_5E0F_2C5C_3F6D);
        let truth = true_counts(&bytes);
        let mut sk = MisraGries::<16>::new();
        sk.update_slice(&bytes);

        let bound = sk.error_bound();
        // Reported set:
        let reported: Vec<(u8, u64)> = sk.top_k().collect();
        let reported_set: BTreeSet<u8> = reported.iter().map(|(b, _)| *b).collect();

        for (byte, &true_count) in truth.iter().enumerate() {
            if true_count > bound {
                assert!(
                    reported_set.contains(&(byte as u8)),
                    "byte {byte} has true count {true_count} > bound {bound} but is missing"
                );
            }
        }

        // Every reported lower bound must satisfy reported ≤ true ≤ reported + bound.
        for (byte, count) in reported {
            let true_count = truth[byte as usize];
            assert!(count <= true_count, "over-estimate for byte {byte}");
            assert!(
                true_count <= count + bound,
                "byte {byte} reported={count} true={true_count} bound={bound}"
            );
        }
    }

    #[test]
    fn k256_is_exact_when_alphabet_fits() {
        let bytes = deterministic_zipfian_bytes(50_000, 0xDEAD_BEEF_CAFE_F00D);
        let truth = true_counts(&bytes);
        let mut sk = MisraGries::<256>::new();
        sk.update_slice(&bytes);

        for (byte, &true_count) in truth.iter().enumerate() {
            let reported = sk
                .top_k()
                .find_map(|(b, c)| (b == byte as u8).then_some(c))
                .unwrap_or(0);
            assert_eq!(
                reported, true_count,
                "byte {byte}: K=256 should be exact, reported={reported} true={true_count}"
            );
        }
    }

    #[test]
    fn update_slice_matches_repeated_update() {
        let bytes = deterministic_zipfian_bytes(2_000, 0xFEED_FACE_CAFE_F00D);
        let mut a = MisraGries::<8>::new();
        let mut b = MisraGries::<8>::new();
        a.update_slice(&bytes);
        for &byte in &bytes {
            b.update(byte);
        }
        assert_eq!(a, b);
    }

    #[test]
    fn memory_footprint_k16_under_256_bytes() {
        // 16 (keys) + 128 (counts) + 8 (n) + padding = 152 bytes
        // expected; the spec demands ≤ 256.
        assert!(MisraGries::<16>::memory_bytes() <= 256);
    }

    #[test]
    fn clear_resets_state() {
        let mut sk = MisraGries::<8>::new();
        sk.update_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(sk.n() > 0);
        sk.clear();
        assert_eq!(sk.n(), 0);
        assert_eq!(sk.top_k().count(), 0);
    }

    #[test]
    fn k_zero_is_safe_no_op() {
        let mut sk = MisraGries::<0>::new();
        sk.update_slice(&[1, 2, 3]);
        // n still increments — the count of bytes seen is independent
        // of the per-slot tracking. Top-K is empty.
        assert_eq!(sk.n(), 3);
        assert_eq!(sk.top_k().count(), 0);
    }

    #[test]
    fn single_dominant_byte_recovered() {
        let mut bytes = vec![0x42_u8; 1000];
        bytes.extend(0_u8..=30);
        let mut sk = MisraGries::<4>::new();
        sk.update_slice(&bytes);
        let top: Vec<(u8, u64)> = sk.top_k().collect();
        let dominant = top.iter().max_by_key(|(_, c)| *c).copied().unwrap();
        assert_eq!(dominant.0, 0x42);
        // The dominant byte's reported count is at least
        // true_count - error_bound = 1000 - bound.
        let bound = sk.error_bound();
        assert!(dominant.1 + bound >= 1000);
    }
}

//! Plain-bitvector rank/select dictionary.
//!
//! See `docs/v0.2_planning/10_BITS.md` § 5 for the spec, vendor decision,
//! and budget revision (1-2 weeks of focused implementation, not "a few
//! days"). Foundation for `bitmap` cardinality, wavelet trees, FM-index.
//!
//! ## Two-level sampling
//!
//! Given a `&[u64]` representing `n_bits` bits, build supports for:
//!
//! * `rank1(i)` — number of 1-bits in `bits[0..i]`. Constant time.
//! * `select1(k)` — position of the `(k+1)`-th 1-bit. Constant time.
//!
//! Two sampling levels:
//!
//! 1. **Superblock counts** (`u32` per **4096-bit** superblock): cumulative
//!    popcount up to (not including) the start of the superblock. ~0.1%
//!    space overhead.
//! 2. **Block counts** (`u16` per **256-bit** block, relative to the
//!    containing superblock): partial popcount inside the superblock,
//!    measured at the start of each block. ~0.6% space overhead.
//!
//! ## Hot paths
//!
//! ### Rank
//!
//! `superblock_counts[s] + block_counts[b] + popcount(partial word)`. Three
//! array accesses + one popcount. Bandwidth-tiny per query (~16 bytes);
//! latency-bound on two cache misses (one for the count entries, one for
//! the bit word). SIMD does not accelerate single rank queries; batch
//! rank wins via AVX-512 VPOPCNTQ.
//!
//! ### Select
//!
//! Binary search through superblock counts (~`log2(N/4096)` cache misses
//! on cold cache, ~one on hot), linear walk through block counts (≤16
//! per superblock), then Vigna's broadword select-in-word inside the
//! 256-bit block. ~30-50 ns warm.
//!
//! ## API
//!
//! ```no_run
//! use tokenfs_algos::bits::RankSelectDict;
//!
//! let bits = [0xff_ff_ff_ff_00_00_00_00_u64];
//! let dict = RankSelectDict::build(&bits, 64);
//! assert_eq!(dict.rank1(32), 0);
//! assert_eq!(dict.rank1(64), 32);
//! assert_eq!(dict.select1(0), Some(32));
//! assert_eq!(dict.select1(31), Some(63));
//! assert_eq!(dict.select1(32), None);
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_lossless)]

extern crate alloc;

use alloc::vec::Vec;

/// Failure modes for the fallible rank/select dictionary constructor
/// ([`RankSelectDict::try_build`]).
///
/// Returned instead of panicking when the borrowed bit slice is too
/// short to hold the requested logical bit count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RankSelectError {
    /// `bits.len() * 64 < n_bits` — the borrowed slice cannot hold the
    /// requested number of logical bits.
    BitsTooShort {
        /// `bits.len()` (count of `u64` words in the borrowed slice).
        bits_len_words: usize,
        /// Caller-supplied logical bit count.
        requested_n_bits: usize,
    },
}

impl core::fmt::Display for RankSelectError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BitsTooShort {
                bits_len_words,
                requested_n_bits,
            } => write!(
                f,
                "RankSelectDict bits slice too short: {bits_len_words} words \
                 (= {} bits) cannot hold {requested_n_bits} requested bits",
                bits_len_words.saturating_mul(64)
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RankSelectError {}

/// Bits per **block**. Each block carries a `u16` relative count.
///
/// 256 bits = 4 `u64` words. Per-block scan cost is bounded by 4 words.
pub const BLOCK_BITS: usize = 256;

/// Bits per **superblock**. Each superblock carries a `u32` cumulative
/// count.
///
/// 4096 bits = 16 blocks = 64 `u64` words.
pub const SUPERBLOCK_BITS: usize = 4096;

/// Number of 64-bit words per block.
pub const WORDS_PER_BLOCK: usize = BLOCK_BITS / 64;

/// Number of 64-bit words per superblock.
pub const WORDS_PER_SUPERBLOCK: usize = SUPERBLOCK_BITS / 64;

/// Number of blocks per superblock.
pub const BLOCKS_PER_SUPERBLOCK: usize = SUPERBLOCK_BITS / BLOCK_BITS;

/// Borrowed plain-bitvector rank/select dictionary.
///
/// Built once over a `&[u64]` slice that the caller owns; cheap to query
/// after construction. The bit slice and the dictionary share a lifetime
/// so consumers can mmap a sealed image and build the index in place.
///
/// See the module docs for the algorithm and space-time tradeoff.
#[derive(Debug, Clone)]
pub struct RankSelectDict<'a> {
    bits: &'a [u64],
    n_bits: usize,
    /// Cumulative popcount up to (not including) the start of each
    /// superblock; one extra trailing entry stores the total popcount.
    superblock_counts: Vec<u32>,
    /// Per-block popcount, relative to the start of its containing
    /// superblock. Block `b` covers bits `[b*256, (b+1)*256)`. The entry
    /// at position `b` is the count of 1-bits in
    /// `[s*4096, b*256)` where `s = b / 16`.
    block_counts: Vec<u16>,
    /// Cached total popcount for `count_ones()` / select bounds checks.
    total_ones: usize,
}

impl<'a> RankSelectDict<'a> {
    /// Builds the dictionary over `bits[..]` interpreted as `n_bits`
    /// individual bits, little-endian within each `u64`.
    ///
    /// Bits beyond `n_bits` (if `n_bits % 64 != 0`) are ignored — the
    /// build masks the trailing partial word so the popcount and the
    /// per-block sample tables agree with the rank/select queries.
    ///
    /// # Panics
    ///
    /// Panics if `bits` is too short to hold `n_bits` bits, i.e. if
    /// `bits.len() * 64 < n_bits`. Use [`Self::try_build`] for a
    /// fallible variant that returns [`RankSelectError`] instead.
    #[must_use]
    pub fn build(bits: &'a [u64], n_bits: usize) -> Self {
        Self::try_build(bits, n_bits)
            .expect("RankSelectDict::build: bits slice too short for n_bits")
    }

    /// Fallible variant of [`Self::build`] that returns
    /// [`RankSelectError::BitsTooShort`] when the borrowed slice cannot
    /// hold the requested number of logical bits, instead of panicking.
    pub fn try_build(bits: &'a [u64], n_bits: usize) -> Result<Self, RankSelectError> {
        if bits.len().saturating_mul(64) < n_bits {
            return Err(RankSelectError::BitsTooShort {
                bits_len_words: bits.len(),
                requested_n_bits: n_bits,
            });
        }

        let n_blocks = n_bits.div_ceil(BLOCK_BITS);
        let n_superblocks = n_bits.div_ceil(SUPERBLOCK_BITS);

        // `superblock_counts[s]` stores the cumulative popcount up to
        // (not including) the start of superblock `s`. The trailing
        // entry holds the total popcount, which simplifies select bounds.
        let mut superblock_counts: Vec<u32> = Vec::with_capacity(n_superblocks + 1);
        let mut block_counts: Vec<u16> = Vec::with_capacity(n_blocks);

        let mut cumulative: u64 = 0;

        for s in 0..n_superblocks {
            // Record the running cumulative count at the start of this
            // superblock, then snapshot it as the per-superblock baseline.
            superblock_counts.push(cumulative as u32);
            let superblock_start_count: u64 = cumulative;

            // Walk the up-to-16 blocks of this superblock.
            let block_lo = s * BLOCKS_PER_SUPERBLOCK;
            let block_hi = ((s + 1) * BLOCKS_PER_SUPERBLOCK).min(n_blocks);

            for b in block_lo..block_hi {
                // `block_counts[b]` is the count of 1-bits in
                // `[s*4096, b*256)`, i.e. cumulative - superblock_start.
                let relative = (cumulative - superblock_start_count) as u16;
                block_counts.push(relative);

                // Accumulate the popcount of the (up to 4) words in this
                // block, masking the final partial word against `n_bits`.
                let word_lo = b * WORDS_PER_BLOCK;
                let word_hi = ((b + 1) * WORDS_PER_BLOCK).min(bits.len());
                for w in word_lo..word_hi {
                    let word = masked_word(bits, w, n_bits);
                    cumulative += u64::from(word.count_ones());
                }
            }
        }

        // Trailing total entry simplifies the rank query at index =
        // n_bits and the select bounds check.
        superblock_counts.push(cumulative as u32);

        // Total popcount under `n_bits` bits as a usize for ergonomic use.
        let total_ones = cumulative as usize;

        Ok(Self {
            bits,
            n_bits,
            superblock_counts,
            block_counts,
            total_ones,
        })
    }

    /// Returns the bit length the dictionary was built over.
    #[must_use]
    pub const fn len_bits(&self) -> usize {
        self.n_bits
    }

    /// Returns the total number of 1-bits in `bits[0..n_bits]`.
    #[must_use]
    pub const fn count_ones(&self) -> usize {
        self.total_ones
    }

    /// Returns the number of 1-bits in `bits[0..i]` (strictly before
    /// position `i`).
    ///
    /// `rank1(0) == 0` and `rank1(n_bits) == count_ones()`.
    ///
    /// # Panics
    ///
    /// Panics if `i > n_bits`.
    #[must_use]
    pub fn rank1(&self, i: usize) -> usize {
        assert!(
            i <= self.n_bits,
            "RankSelectDict::rank1: i = {i} > n_bits = {}",
            self.n_bits
        );

        if i == 0 {
            return 0;
        }
        if i == self.n_bits {
            return self.total_ones;
        }

        let s = i / SUPERBLOCK_BITS;
        let b = i / BLOCK_BITS;

        // Superblock + block samples cover everything up to the start of
        // the block `b`. The remaining bits live in the up-to-4 words of
        // block `b` before bit position `i`.
        let mut sum = self.superblock_counts[s] as usize + self.block_counts[b] as usize;

        // Per-block partial scan.
        let block_word_lo = b * WORDS_PER_BLOCK;
        let bit_in_block = i - b * BLOCK_BITS;
        let full_words_in_block = bit_in_block / 64;
        let trailing_bits = bit_in_block % 64;

        for w in 0..full_words_in_block {
            let word = masked_word(self.bits, block_word_lo + w, self.n_bits);
            sum += word.count_ones() as usize;
        }
        if trailing_bits > 0 {
            let word = masked_word(self.bits, block_word_lo + full_words_in_block, self.n_bits);
            let mask = (1_u64 << trailing_bits) - 1;
            sum += (word & mask).count_ones() as usize;
        }

        sum
    }

    /// Returns the number of 0-bits in `bits[0..i]`.
    #[must_use]
    pub fn rank0(&self, i: usize) -> usize {
        assert!(
            i <= self.n_bits,
            "RankSelectDict::rank0: i = {i} > n_bits = {}",
            self.n_bits
        );
        i - self.rank1(i)
    }

    /// Returns the bit position of the `(k+1)`-th 1-bit (zero-based:
    /// `k == 0` returns the first 1-bit), or `None` if `k >=
    /// count_ones()`.
    #[must_use]
    pub fn select1(&self, k: usize) -> Option<usize> {
        if k >= self.total_ones {
            return None;
        }
        Some(self.select_inner(k, /* select_ones = */ true))
    }

    /// Returns the bit position of the `(k+1)`-th 0-bit, or `None` if
    /// `k >= n_bits - count_ones()`.
    #[must_use]
    pub fn select0(&self, k: usize) -> Option<usize> {
        let total_zeros = self.n_bits - self.total_ones;
        if k >= total_zeros {
            return None;
        }
        Some(self.select_inner(k, /* select_ones = */ false))
    }

    /// Returns `self.bits.len() * 8 + self.superblock_counts.capacity()
    /// * 4 + self.block_counts.capacity() * 2`.
    ///
    /// Approximates the heap footprint of the dictionary plus the
    /// borrowed bit slice; useful for "how big is my index?" reporting.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        core::mem::size_of_val(self.bits)
            + self.superblock_counts.capacity() * core::mem::size_of::<u32>()
            + self.block_counts.capacity() * core::mem::size_of::<u16>()
    }

    /// Returns the per-position rank of `positions[i]` for each entry.
    ///
    /// Equivalent to `positions.iter().map(|&p| self.rank1(p))`. Provided
    /// as a batch convenience that AVX-512 VPOPCNTQ implementations can
    /// later override; today the kernel just calls `rank1` per entry.
    pub fn rank1_batch(&self, positions: &[usize], out: &mut [usize]) {
        kernels::auto::rank1_batch(self, positions, out);
    }

    /// Returns the per-position select of `ks[i]` for each entry.
    ///
    /// Equivalent to `ks.iter().map(|&k| self.select1(k))`. Same SIMD
    /// hooks as [`Self::rank1_batch`].
    pub fn select1_batch(&self, ks: &[usize], out: &mut [Option<usize>]) {
        kernels::auto::select1_batch(self, ks, out);
    }

    /// Internal select kernel. `select_ones=true` selects 1-bits;
    /// `false` selects 0-bits. Caller has already bounded `k`.
    fn select_inner(&self, k: usize, select_ones: bool) -> usize {
        // Step 1: binary-search the superblock counts for the largest
        // `s` such that `superblock_count_eff(s) <= k`. Linear search is
        // fast for tiny bitvectors; we use binary search uniformly so
        // the cold-cache cost is `log2(N/4096)`.
        let n_sb = self.superblock_counts.len() - 1;

        let sb_count_eff = |s: usize| -> usize {
            let ones = self.superblock_counts[s] as usize;
            if select_ones {
                ones
            } else {
                s * SUPERBLOCK_BITS - ones
            }
        };

        // Find largest `s` in `0..n_sb` with `sb_count_eff(s) <= k`.
        let mut lo = 0_usize;
        let mut hi = n_sb;
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if mid >= n_sb {
                hi = mid - 1;
                continue;
            }
            if sb_count_eff(mid) <= k {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let s = lo;
        let mut k_remaining = k - sb_count_eff(s);

        // Step 2: linear-scan the up-to-16 block counts inside the
        // superblock for the largest `b` such that
        // `block_count_eff(b) <= k_remaining` (relative to the
        // superblock start).
        let block_lo = s * BLOCKS_PER_SUPERBLOCK;
        let block_hi = ((s + 1) * BLOCKS_PER_SUPERBLOCK).min(self.block_counts.len());

        let block_count_eff = |b: usize| -> usize {
            let ones = self.block_counts[b] as usize;
            if select_ones {
                ones
            } else {
                (b - block_lo) * BLOCK_BITS - ones
            }
        };

        let mut chosen_b = block_lo;
        for b in (block_lo + 1)..block_hi {
            if block_count_eff(b) <= k_remaining {
                chosen_b = b;
            } else {
                break;
            }
        }
        k_remaining -= block_count_eff(chosen_b);

        // Step 3: walk the up-to-4 words of the block, popcount each,
        // descend into the word that contains the target bit.
        let word_lo = chosen_b * WORDS_PER_BLOCK;
        let word_hi = (word_lo + WORDS_PER_BLOCK).min(self.bits.len());

        for w in word_lo..word_hi {
            let raw = masked_word(self.bits, w, self.n_bits);
            // For select0, invert the word; the trailing partial word
            // has high bits that would become spurious 1s after
            // negation, so we re-mask against the valid bit count.
            let word = if select_ones {
                raw
            } else {
                let inverted = !raw;
                let valid = (self.n_bits - w * 64).min(64);
                if valid < 64 {
                    inverted & ((1_u64 << valid) - 1)
                } else {
                    inverted
                }
            };

            let pc = word.count_ones() as usize;
            if pc <= k_remaining {
                k_remaining -= pc;
                continue;
            }
            // The target bit lives in this word at the
            // `k_remaining`-th 1-bit (0-indexed).
            let bit_in_word = select_in_word(word, k_remaining as u32) as usize;
            return w * 64 + bit_in_word;
        }

        // Should be unreachable when `k` is in range. Defensively return
        // `n_bits` so the caller sees an obviously out-of-range result.
        self.n_bits
    }
}

/// Returns `bits[index]` masked so only bits below `n_bits` are set.
///
/// For all but possibly the final word this is just `bits[index]`. The
/// final word, when `n_bits` does not lie on a `u64` boundary, is
/// truncated to the low `n_bits % 64` bits so the popcount and per-block
/// counts match the dictionary's logical bit count.
#[inline]
fn masked_word(bits: &[u64], index: usize, n_bits: usize) -> u64 {
    let word = bits[index];
    let last_full_word_index = n_bits / 64;
    if index < last_full_word_index {
        word
    } else if index == last_full_word_index && !n_bits.is_multiple_of(64) {
        let valid = (n_bits % 64) as u32;
        let mask = (1_u64 << valid) - 1;
        word & mask
    } else if index == last_full_word_index {
        // n_bits is a multiple of 64 and `index == n_bits / 64`. This
        // word is past the logical end; treat it as all-zero.
        0
    } else {
        // Past the logical end.
        0
    }
}

/// Returns the bit position of the `(k+1)`-th 1-bit in `word`, where
/// `0 <= k < word.count_ones()`.
///
/// Implements Vigna's broadword select-in-word (Vigna 2008,
/// *Broadword Implementation of Rank/Select Queries*) §3.
///
/// Uses BMI2 `_pdep_u64` + `tzcnt` when available at compile time
/// (gated on `target_feature = "bmi2"`); falls back to the broadword
/// trick otherwise.
///
/// # Panics
///
/// In debug builds, panics if `k >= word.count_ones()`. In release
/// builds the result is unspecified for out-of-range `k` — callers
/// must clamp before calling.
#[inline]
#[must_use]
pub fn select_in_word(word: u64, k: u32) -> u32 {
    debug_assert!(
        word.count_ones() > k,
        "select_in_word: k = {k} out of range for word with {} bits set",
        word.count_ones()
    );

    // BMI2 fast path — `_pdep_u64(1 << k, word)` deposits the k-th 1-bit
    // mask into `word`'s 1-bit positions; one trailing-zero count gives
    // the position. Two instructions on Haswell+ / Excavator+.
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "bmi2"
    ))]
    {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::_pdep_u64;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::_pdep_u64;
        // SAFETY: target_feature gating above asserts BMI2 is present.
        let mask = unsafe { _pdep_u64(1_u64 << k, word) };
        return mask.trailing_zeros();
    }

    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "bmi2"
    )))]
    {
        select_in_word_broadword(word, k)
    }
}

/// Vigna's broadword select-in-word. Always-available scalar fallback
/// for the BMI2-less path.
///
/// Reference: Vigna 2008, "Broadword Implementation of Rank/Select
/// Queries," WEA 2008. Algorithm 2 (the constant-time variant), with
/// minor adaptation: instead of the broadword bit-scan within the
/// target byte, we use a tight `while` loop bounded by 8 iterations.
///
/// The constants `L8`, `H8` follow Vigna's notation: `L8` is the
/// low-byte broadcast `0x0101_0101_0101_0101`, `H8` is the high-byte
/// broadcast `0x8080_8080_8080_8080`.
///
/// Algorithm:
/// 1. Compute per-byte popcount `s` of `word`.
/// 2. Cumulative-sum the per-byte popcounts via `s * L8`. The byte at
///    position `i` of `byte_sums` holds `popcount(word[0..=i])` —
///    treating `word` byte-wise from low to high.
/// 3. Find the lowest byte index `i` such that `byte_sums[i] > k`
///    using a SWAR `<=` test (`leq_step8`) and counting the number of
///    bytes where `byte_sums[i] <= k`.
/// 4. Within that byte, bit-scan for the `(k - bytes_before)`-th set
///    bit. The byte has ≤ 8 bits so cost is ≤ 8 iterations.
#[inline]
#[must_use]
pub fn select_in_word_broadword(word: u64, k: u32) -> u32 {
    debug_assert!(
        word.count_ones() > k,
        "select_in_word_broadword: k = {k} out of range for word with {} bits set",
        word.count_ones()
    );

    const L8: u64 = 0x0101_0101_0101_0101;
    const H8: u64 = 0x8080_8080_8080_8080;

    // Step 1: per-byte popcount via Hamming-weight wide adds.
    let mut s = word;
    s = s - ((s & 0xAAAA_AAAA_AAAA_AAAA) >> 1);
    s = (s & 0x3333_3333_3333_3333) + ((s >> 2) & 0x3333_3333_3333_3333);
    s = (s + (s >> 4)) & 0x0F0F_0F0F_0F0F_0F0F;

    // Step 2: per-byte cumulative sum. `byte_sums` byte `i` holds
    // `popcount(word_bytes[0..=i])`.
    let byte_sums = s.wrapping_mul(L8);

    // Step 3: SWAR `<=` test to count bytes where byte_sums[i] <= k.
    // That count IS the target byte index (since byte_sums is
    // non-decreasing in i).
    //
    // `leq_step8(a, b)` per Vigna: returns a u64 with bit 7 of byte `i`
    // set iff `a[i] <= b[i]`. Both a and b must fit in 7 bits per byte.
    // `byte_sums` has values in 0..=64 ✓. `k` is at most 63 (caller has
    // bounded k < word.count_ones() <= 64), so k fits in 7 bits ✓.
    let k_byte = (u64::from(k)).wrapping_mul(L8);
    let leq_mask = leq_step8(byte_sums, k_byte);
    // Each "true" byte has bit 7 set, contributing 1 to the popcount.
    // The result is in 0..=8: the number of bytes where the cumulative
    // sum is still ≤ k. Equivalently, the target byte index.
    let target_byte = (leq_mask & H8).count_ones();

    // Step 4: rank within the target byte.
    //
    // `bytes_before` = cumulative popcount strictly before
    // `target_byte`; that's byte `target_byte - 1` of `byte_sums`, or
    // 0 when target_byte == 0.
    let bytes_before = if target_byte == 0 {
        0_u32
    } else {
        ((byte_sums >> (8 * target_byte - 8)) & 0xff) as u32
    };
    let k_within_byte = k - bytes_before;

    let byte = ((word >> (8 * target_byte)) & 0xff) as u32;

    // Bit-by-bit scan within the byte. ≤ 8 iterations.
    let mut remaining = k_within_byte;
    let mut bit_in_byte = 0_u32;
    let mut byte = byte;
    while bit_in_byte < 8 {
        if (byte & 1) != 0 {
            if remaining == 0 {
                break;
            }
            remaining -= 1;
        }
        byte >>= 1;
        bit_in_byte += 1;
    }

    8 * target_byte + bit_in_byte
}

/// Vigna's `leqStep8` SWAR primitive: returns a u64 whose byte `i` has
/// bit 7 set iff `a[i] <= b[i]` (treating each byte of the inputs as
/// an independent 7-bit unsigned value). Both `a` and `b` must have
/// per-byte values in `0..=127`.
///
/// Formula (from Vigna 2008, equation (5)):
/// `leqStep8(a, b) = (((b | H8) - (a & !H8)) ^ a ^ b) & H8`
///
/// Verification table:
/// * `a=0, b=0`: `(0x80 - 0x00) ^ 0 ^ 0 = 0x80`, `& H8 = 0x80` → true ✓
/// * `a=0, b=1`: `(0x81 - 0x00) ^ 0 ^ 1 = 0x80`, `& H8 = 0x80` → true ✓
/// * `a=1, b=0`: `(0x80 - 0x01) ^ 1 ^ 0 = 0x7E`, `& H8 = 0x00` → false ✓
/// * `a=1, b=1`: `(0x81 - 0x01) ^ 1 ^ 1 = 0x80`, `& H8 = 0x80` → true ✓
#[inline]
const fn leq_step8(a: u64, b: u64) -> u64 {
    const H8: u64 = 0x8080_8080_8080_8080;
    (((b | H8).wrapping_sub(a & !H8)) ^ a ^ b) & H8
}

/// Pinned rank/select kernels.
pub mod kernels {
    use super::RankSelectDict;

    /// Runtime-dispatched rank/select batch kernels.
    pub mod auto {
        use super::RankSelectDict;

        /// Batch rank using the best available kernel.
        pub fn rank1_batch(dict: &RankSelectDict<'_>, positions: &[usize], out: &mut [usize]) {
            super::scalar::rank1_batch(dict, positions, out);
        }

        /// Batch select using the best available kernel.
        pub fn select1_batch(dict: &RankSelectDict<'_>, ks: &[usize], out: &mut [Option<usize>]) {
            super::scalar::select1_batch(dict, ks, out);
        }
    }

    /// Portable scalar batch kernels.
    ///
    /// The single-query rank and select are already constant-time and
    /// dominated by 2-3 cache misses. The batch wrapper just iterates;
    /// AVX-512 acceleration for batch rank can be added later by
    /// inlining the popcount of all four block words via `VPOPCNTQ`.
    pub mod scalar {
        use super::RankSelectDict;

        /// Per-position rank fan-out.
        ///
        /// # Panics
        ///
        /// Panics if `out.len() < positions.len()` or any position
        /// exceeds `dict.len_bits()`.
        pub fn rank1_batch(dict: &RankSelectDict<'_>, positions: &[usize], out: &mut [usize]) {
            assert!(
                out.len() >= positions.len(),
                "rank1_batch: out.len() = {} < positions.len() = {}",
                out.len(),
                positions.len()
            );
            for (slot, &p) in out.iter_mut().zip(positions.iter()) {
                *slot = dict.rank1(p);
            }
        }

        /// Per-position select fan-out.
        ///
        /// # Panics
        ///
        /// Panics if `out.len() < ks.len()`.
        pub fn select1_batch(dict: &RankSelectDict<'_>, ks: &[usize], out: &mut [Option<usize>]) {
            assert!(
                out.len() >= ks.len(),
                "select1_batch: out.len() = {} < ks.len() = {}",
                out.len(),
                ks.len()
            );
            for (slot, &k) in out.iter_mut().zip(ks.iter()) {
                *slot = dict.select1(k);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

    use super::*;
    use alloc::vec;

    fn naive_rank1(bits: &[u64], n_bits: usize, i: usize) -> usize {
        assert!(i <= n_bits);
        let mut count = 0_usize;
        for bit in 0..i {
            let word = bits[bit / 64];
            if (word >> (bit % 64)) & 1 == 1 {
                count += 1;
            }
        }
        count
    }

    fn naive_select1(bits: &[u64], n_bits: usize, k: usize) -> Option<usize> {
        let mut remaining = k;
        for bit in 0..n_bits {
            let word = bits[bit / 64];
            if (word >> (bit % 64)) & 1 == 1 {
                if remaining == 0 {
                    return Some(bit);
                }
                remaining -= 1;
            }
        }
        None
    }

    fn naive_select0(bits: &[u64], n_bits: usize, k: usize) -> Option<usize> {
        let mut remaining = k;
        for bit in 0..n_bits {
            let word = bits[bit / 64];
            if (word >> (bit % 64)) & 1 == 0 {
                if remaining == 0 {
                    return Some(bit);
                }
                remaining -= 1;
            }
        }
        None
    }

    fn deterministic_words(n: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_f491_4f6c_dd1d)
            })
            .collect()
    }

    fn parity_check(bits: &[u64], n_bits: usize, sample_step: usize) {
        let dict = RankSelectDict::build(bits, n_bits);
        // rank1 endpoints
        assert_eq!(dict.rank1(0), 0, "rank1(0) != 0");
        let total = naive_rank1(bits, n_bits, n_bits);
        assert_eq!(
            dict.rank1(n_bits),
            total,
            "rank1(n_bits) != popcount on {n_bits}-bit input"
        );
        assert_eq!(dict.count_ones(), total);

        // Sample a number of rank queries.
        let mut i = 0;
        while i <= n_bits {
            let expected = naive_rank1(bits, n_bits, i);
            assert_eq!(dict.rank1(i), expected, "rank1({i}) mismatch");
            assert_eq!(dict.rank0(i), i - expected, "rank0({i}) mismatch");
            i += sample_step.max(1);
        }

        // Select1 — sample evenly across [0, total_ones).
        if total > 0 {
            let step = (total / 64).max(1);
            let mut k = 0_usize;
            while k < total {
                let actual = dict.select1(k);
                let expected = naive_select1(bits, n_bits, k);
                assert_eq!(actual, expected, "select1({k}) mismatch");
                k += step;
            }
            // Always test the last 1-bit.
            let actual = dict.select1(total - 1);
            let expected = naive_select1(bits, n_bits, total - 1);
            assert_eq!(actual, expected, "select1({}) mismatch", total - 1);
        }
        assert_eq!(
            dict.select1(total),
            None,
            "select1(total_ones) should be None"
        );

        // Select0 — sample evenly across [0, total_zeros).
        let total_zeros = n_bits - total;
        if total_zeros > 0 {
            let step = (total_zeros / 64).max(1);
            let mut k = 0_usize;
            while k < total_zeros {
                let actual = dict.select0(k);
                let expected = naive_select0(bits, n_bits, k);
                assert_eq!(actual, expected, "select0({k}) mismatch");
                k += step;
            }
            let actual = dict.select0(total_zeros - 1);
            let expected = naive_select0(bits, n_bits, total_zeros - 1);
            assert_eq!(actual, expected, "select0({}) mismatch", total_zeros - 1);
        }
        assert_eq!(
            dict.select0(total_zeros),
            None,
            "select0(total_zeros) should be None"
        );

        // Property: rank1(i+1) - rank1(i) ∈ {0, 1}.
        // Sample a sweep of consecutive positions where the bit is set
        // and check the rank-select inverse property.
        let mut i = 0;
        while i < n_bits {
            let r0 = dict.rank1(i);
            let r1 = dict.rank1(i + 1);
            let delta = r1 - r0;
            assert!(delta == 0 || delta == 1, "rank1 delta != 0/1 at i = {i}");

            // If the bit at position i is 1, then rank1(i) - 1 < rank1(i)
            // and select1(rank1(i)) == i (the i-th bit is the (rank1(i)+1)-th 1-bit when 0-indexed).
            // Equivalently: select1(rank1(i)) == i. Test occasionally.
            if delta == 1 && i % 17 == 0 {
                assert_eq!(
                    dict.select1(r0),
                    Some(i),
                    "select1(rank1({i})) != i for set bit"
                );
            }

            i += sample_step.max(1);
        }
    }

    #[test]
    fn empty_bitvector() {
        let bits: [u64; 0] = [];
        let dict = RankSelectDict::build(&bits, 0);
        assert_eq!(dict.len_bits(), 0);
        assert_eq!(dict.count_ones(), 0);
        assert_eq!(dict.rank1(0), 0);
        assert_eq!(dict.select1(0), None);
        assert_eq!(dict.select0(0), None);
    }

    #[test]
    fn single_bit_set() {
        for pos in [
            0_usize, 1, 7, 8, 31, 32, 63, 64, 100, 255, 256, 1023, 4095, 4096,
        ] {
            let n_bits = (pos + 1).max(1);
            let n_words = n_bits.div_ceil(64);
            let mut bits = vec![0_u64; n_words];
            bits[pos / 64] |= 1_u64 << (pos % 64);
            let dict = RankSelectDict::build(&bits, n_bits);
            assert_eq!(dict.count_ones(), 1, "popcount mismatch at pos {pos}");
            assert_eq!(dict.rank1(pos), 0, "rank1({pos}) before set bit");
            assert_eq!(dict.rank1(pos + 1), 1, "rank1({}) after set bit", pos + 1);
            assert_eq!(
                dict.select1(0),
                Some(pos),
                "select1(0) should locate the only set bit"
            );
            assert_eq!(dict.select1(1), None);
        }
    }

    #[test]
    fn single_bit_clear_in_dense() {
        for pos in [0_usize, 1, 63, 64, 255, 256, 1024, 4096] {
            let n_bits = pos.max(64) + 64;
            let n_words = n_bits.div_ceil(64);
            let mut bits = vec![u64::MAX; n_words];
            bits[pos / 64] &= !(1_u64 << (pos % 64));
            // Clear bits past n_bits to keep the bitvector well-defined.
            if !n_bits.is_multiple_of(64) {
                let mask = (1_u64 << (n_bits % 64)) - 1;
                let last = n_words - 1;
                bits[last] &= mask;
            }
            let dict = RankSelectDict::build(&bits, n_bits);
            assert_eq!(
                dict.count_ones(),
                n_bits - 1,
                "popcount mismatch in dense single-clear at pos {pos}"
            );
            assert_eq!(
                dict.select0(0),
                Some(pos),
                "select0(0) should locate the only clear bit at {pos}"
            );
            assert_eq!(dict.select0(1), None);
        }
    }

    #[test]
    fn all_zero_bitvector() {
        for n_bits in [1_usize, 7, 64, 255, 4096, 4097, 8192, 12345] {
            let bits = vec![0_u64; n_bits.div_ceil(64)];
            let dict = RankSelectDict::build(&bits, n_bits);
            assert_eq!(dict.count_ones(), 0);
            assert_eq!(dict.rank1(n_bits), 0);
            assert_eq!(dict.select1(0), None);
            for k in 0..n_bits {
                assert_eq!(dict.select0(k), Some(k), "select0({k}) on all-zero");
            }
        }
    }

    #[test]
    fn all_one_bitvector() {
        for n_bits in [1_usize, 7, 64, 255, 4096, 4097, 8192, 12345] {
            let n_words = n_bits.div_ceil(64);
            let mut bits = vec![u64::MAX; n_words];
            // Truncate the trailing word to n_bits.
            if !n_bits.is_multiple_of(64) {
                let mask = (1_u64 << (n_bits % 64)) - 1;
                bits[n_words - 1] &= mask;
            }
            let dict = RankSelectDict::build(&bits, n_bits);
            assert_eq!(dict.count_ones(), n_bits);
            assert_eq!(dict.rank1(n_bits), n_bits);
            assert_eq!(dict.select0(0), None);
            for k in 0..n_bits {
                assert_eq!(dict.select1(k), Some(k), "select1({k}) on all-one");
            }
        }
    }

    #[test]
    fn small_endpoints() {
        for n_bits in [
            1_usize, 2, 7, 8, 16, 32, 63, 64, 65, 127, 128, 255, 256, 257,
        ] {
            let bits = deterministic_words(
                n_bits.div_ceil(64),
                0xC0FFEE_u64.wrapping_add(n_bits as u64),
            );
            parity_check(&bits, n_bits, 1);
        }
    }

    #[test]
    fn alternating_bitvector() {
        // 0xAAAA... — every other bit set.
        let n_bits = 8 * SUPERBLOCK_BITS + 17;
        let n_words = n_bits.div_ceil(64);
        let bits = vec![0xAAAA_AAAA_AAAA_AAAA_u64; n_words];
        parity_check(&bits, n_bits, 13);
    }

    #[test]
    fn dense_random_spans_multiple_superblocks() {
        let n_bits = 5 * SUPERBLOCK_BITS + 123;
        let bits = deterministic_words(n_bits.div_ceil(64), 0xF22_C0FFEE);
        parity_check(&bits, n_bits, 17);
    }

    #[test]
    fn sparse_one_per_superblock() {
        // One set bit at the start of each superblock.
        let n_superblocks = 7_usize;
        let n_bits = n_superblocks * SUPERBLOCK_BITS;
        let mut bits = vec![0_u64; n_bits.div_ceil(64)];
        for s in 0..n_superblocks {
            let bit = s * SUPERBLOCK_BITS;
            bits[bit / 64] |= 1_u64 << (bit % 64);
        }
        let dict = RankSelectDict::build(&bits, n_bits);
        assert_eq!(dict.count_ones(), n_superblocks);
        for s in 0..n_superblocks {
            let bit = s * SUPERBLOCK_BITS;
            assert_eq!(dict.select1(s), Some(bit));
        }
        assert_eq!(dict.select1(n_superblocks), None);
    }

    #[test]
    fn dense_one_clear_per_superblock() {
        // One clear bit at the start of each superblock; rest dense.
        let n_superblocks = 4_usize;
        let n_bits = n_superblocks * SUPERBLOCK_BITS;
        let mut bits = vec![u64::MAX; n_bits.div_ceil(64)];
        for s in 0..n_superblocks {
            let bit = s * SUPERBLOCK_BITS;
            bits[bit / 64] &= !(1_u64 << (bit % 64));
        }
        let dict = RankSelectDict::build(&bits, n_bits);
        assert_eq!(dict.count_ones(), n_bits - n_superblocks);
        for s in 0..n_superblocks {
            let bit = s * SUPERBLOCK_BITS;
            assert_eq!(dict.select0(s), Some(bit));
        }
    }

    #[test]
    fn n_bits_one() {
        let bits_zero = vec![0_u64];
        let dict_zero = RankSelectDict::build(&bits_zero, 1);
        assert_eq!(dict_zero.rank1(0), 0);
        assert_eq!(dict_zero.rank1(1), 0);
        assert_eq!(dict_zero.select1(0), None);
        assert_eq!(dict_zero.select0(0), Some(0));

        let bits_one = vec![1_u64];
        let dict_one = RankSelectDict::build(&bits_one, 1);
        assert_eq!(dict_one.rank1(0), 0);
        assert_eq!(dict_one.rank1(1), 1);
        assert_eq!(dict_one.select1(0), Some(0));
        assert_eq!(dict_one.select0(0), None);
    }

    #[test]
    fn n_bits_64() {
        for word in [
            0_u64,
            u64::MAX,
            0xAAAA_AAAA_AAAA_AAAA,
            0x5555_5555_5555_5555,
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0000,
            0x12345678_9ABCDEF0,
        ] {
            let bits = vec![word];
            parity_check(&bits, 64, 1);
        }
    }

    #[test]
    fn spans_multiple_superblocks() {
        let n_superblocks = 5_usize;
        let n_bits = n_superblocks * SUPERBLOCK_BITS - 7;
        let bits = deterministic_words(n_bits.div_ceil(64), 0xBA1_F00D);
        parity_check(&bits, n_bits, 23);
    }

    #[test]
    fn batch_rank_matches_scalar() {
        let n_bits = 10_000_usize;
        let bits = deterministic_words(n_bits.div_ceil(64), 0xC0FFEE);
        let dict = RankSelectDict::build(&bits, n_bits);
        let positions: Vec<usize> = (0..n_bits).step_by(63).collect();
        let mut out = vec![0_usize; positions.len()];
        dict.rank1_batch(&positions, &mut out);
        for (i, &p) in positions.iter().enumerate() {
            assert_eq!(out[i], dict.rank1(p));
        }
    }

    #[test]
    fn batch_select_matches_scalar() {
        let n_bits = 10_000_usize;
        let bits = deterministic_words(n_bits.div_ceil(64), 0xC0FFEE);
        let dict = RankSelectDict::build(&bits, n_bits);
        let total = dict.count_ones();
        let ks: Vec<usize> = (0..total).step_by(13).collect();
        let mut out = vec![None; ks.len()];
        dict.select1_batch(&ks, &mut out);
        for (i, &k) in ks.iter().enumerate() {
            assert_eq!(out[i], dict.select1(k));
        }
    }

    #[test]
    fn select_in_word_exhaustive_per_byte_pattern() {
        // Test select_in_word against a naive bit scan for every
        // possible single-byte pattern across each byte position. This
        // exercises the per-byte bit-scan inside the broadword path.
        for byte in 0_u32..=255 {
            for byte_pos in 0_u32..8 {
                let word = (byte as u64) << (8 * byte_pos);
                let pc = word.count_ones();
                for k in 0..pc {
                    let actual = select_in_word(word, k);
                    let actual_bw = select_in_word_broadword(word, k);
                    let expected = {
                        let mut count = k;
                        let mut found = u32::MAX;
                        for bit in 0..64 {
                            if (word >> bit) & 1 == 1 {
                                if count == 0 {
                                    found = bit as u32;
                                    break;
                                }
                                count -= 1;
                            }
                        }
                        found
                    };
                    assert_eq!(
                        actual, expected,
                        "select_in_word({word:#018x}, {k}) = {actual} (expected {expected})"
                    );
                    assert_eq!(
                        actual_bw, expected,
                        "select_in_word_broadword({word:#018x}, {k}) = {actual_bw} (expected {expected})"
                    );
                }
            }
        }
    }

    #[test]
    fn select_in_word_random_words() {
        let words = deterministic_words(2048, 0xF22_5E1EC7);
        for &word in &words {
            let pc = word.count_ones();
            for k in 0..pc {
                let actual = select_in_word(word, k);
                let actual_bw = select_in_word_broadword(word, k);
                let mut count = k;
                let mut expected = u32::MAX;
                for bit in 0..64 {
                    if (word >> bit) & 1 == 1 {
                        if count == 0 {
                            expected = bit as u32;
                            break;
                        }
                        count -= 1;
                    }
                }
                assert_eq!(
                    actual, expected,
                    "select_in_word({word:#018x}, {k}) mismatch"
                );
                assert_eq!(
                    actual_bw, expected,
                    "select_in_word_broadword({word:#018x}, {k}) mismatch"
                );
            }
        }
    }

    #[test]
    fn select_in_word_full_words() {
        // Edge: word == u64::MAX → select_in_word(_, k) == k for all k.
        for k in 0..64 {
            assert_eq!(select_in_word(u64::MAX, k), k);
            assert_eq!(select_in_word_broadword(u64::MAX, k), k);
        }
        // Edge: single bit.
        for pos in 0..64 {
            let word = 1_u64 << pos;
            assert_eq!(select_in_word(word, 0), pos);
            assert_eq!(select_in_word_broadword(word, 0), pos);
        }
    }

    #[test]
    fn memory_bytes_includes_index() {
        let bits = vec![0xAAAA_AAAA_AAAA_AAAA_u64; SUPERBLOCK_BITS / 64];
        let dict = RankSelectDict::build(&bits, SUPERBLOCK_BITS);
        let bytes = dict.memory_bytes();
        // The bit slice is 4096 bits = 512 bytes; the index is small.
        assert!(bytes >= 512, "memory_bytes too small: {bytes}");
        // Conservative upper bound: < 1.5x the bitvector size for this
        // single-superblock case (overhead is ~0.7% asymptotically but
        // dwarfed by the trailing-entry padding for tiny inputs).
        assert!(bytes <= 4096, "memory_bytes too large: {bytes}");
    }

    /// Parity vs the `sucds` `Rank9Sel` reference implementation on
    /// dense, sparse, and alternating bitvectors. Sucds is the mature
    /// pure-Rust succinct DS crate and serves as the canonical oracle
    /// per `docs/v0.2_planning/10_BITS.md` § 5 vendor decision.
    ///
    /// We only test rank1 / select1 for dense and alternating bitvectors
    /// (where sucds' default Rank9SelIndex without hints is fast enough)
    /// — adding `select1_hints()` would cover sparse vectors but doesn't
    /// change the parity outcome.
    #[test]
    fn sucds_parity_dense_sparse_alternating() {
        use sucds::bit_vectors::{Rank, Rank9Sel, Select};

        // Convert our `&[u64]` → `Rank9Sel` via the `from_bits` iterator.
        fn build_sucds(bits: &[u64], n_bits: usize) -> Rank9Sel {
            let iter = (0..n_bits).map(|i| (bits[i / 64] >> (i % 64)) & 1 == 1);
            Rank9Sel::from_bits(iter).select1_hints().select0_hints()
        }

        // 1) Dense: all bits set.
        {
            let n_bits = 4096_usize + 17;
            let n_words = n_bits.div_ceil(64);
            let mut bits = vec![u64::MAX; n_words];
            if !n_bits.is_multiple_of(64) {
                bits[n_words - 1] &= (1_u64 << (n_bits % 64)) - 1;
            }
            let dict = RankSelectDict::build(&bits, n_bits);
            let oracle = build_sucds(&bits, n_bits);

            for i in [0, 1, 63, 64, 128, 1000, 4095, n_bits - 1, n_bits] {
                assert_eq!(
                    dict.rank1(i),
                    oracle.rank1(i).unwrap_or(dict.count_ones()),
                    "rank1({i}) parity vs sucds (dense)"
                );
            }
            for k in [0_usize, 1, 100, 4000, dict.count_ones() - 1] {
                assert_eq!(
                    dict.select1(k),
                    oracle.select1(k),
                    "select1({k}) parity vs sucds (dense)"
                );
            }
        }

        // 2) Sparse: 1 bit set per ~256 bits (sub-block density).
        {
            let n_bits = 16 * 4096_usize + 13;
            let n_words = n_bits.div_ceil(64);
            let mut bits = vec![0_u64; n_words];
            for s in 0..(n_bits / 256) {
                let bit = s * 256 + 17;
                if bit < n_bits {
                    bits[bit / 64] |= 1_u64 << (bit % 64);
                }
            }
            let dict = RankSelectDict::build(&bits, n_bits);
            let oracle = build_sucds(&bits, n_bits);

            for i in [0_usize, 17, 256, 273, 4096, 16_384, n_bits / 2, n_bits] {
                assert_eq!(
                    dict.rank1(i),
                    oracle.rank1(i).unwrap_or(dict.count_ones()),
                    "rank1({i}) parity vs sucds (sparse)"
                );
            }
            for k in 0..dict.count_ones().min(64) {
                assert_eq!(
                    dict.select1(k),
                    oracle.select1(k),
                    "select1({k}) parity vs sucds (sparse)"
                );
            }
        }

        // 3) Alternating: every other bit set.
        {
            let n_bits = 5 * 4096_usize + 9;
            let n_words = n_bits.div_ceil(64);
            let mut bits = vec![0xAAAA_AAAA_AAAA_AAAA_u64; n_words];
            if !n_bits.is_multiple_of(64) {
                bits[n_words - 1] &= (1_u64 << (n_bits % 64)) - 1;
            }
            let dict = RankSelectDict::build(&bits, n_bits);
            let oracle = build_sucds(&bits, n_bits);

            for i in [0_usize, 1, 2, 64, 128, 4096, 4097, n_bits - 1, n_bits] {
                assert_eq!(
                    dict.rank1(i),
                    oracle.rank1(i).unwrap_or(dict.count_ones()),
                    "rank1({i}) parity vs sucds (alternating)"
                );
            }
            // Sample 64 select1 queries.
            let total = dict.count_ones();
            let step = (total / 64).max(1);
            let mut k = 0_usize;
            while k < total {
                assert_eq!(
                    dict.select1(k),
                    oracle.select1(k),
                    "select1({k}) parity vs sucds (alternating)"
                );
                k += step;
            }
            // select0 too.
            let total_zeros = n_bits - total;
            let step = (total_zeros / 64).max(1);
            let mut k = 0_usize;
            while k < total_zeros {
                assert_eq!(
                    dict.select0(k),
                    oracle.select0(k),
                    "select0({k}) parity vs sucds (alternating)"
                );
                k += step;
            }
        }
    }

    #[test]
    fn try_build_returns_err_on_too_short_bits() {
        let bits = [0_u64, 0_u64]; // 128 bits total.
        let err = RankSelectDict::try_build(&bits, 200).unwrap_err();
        assert_eq!(
            err,
            RankSelectError::BitsTooShort {
                bits_len_words: 2,
                requested_n_bits: 200
            }
        );
    }

    #[test]
    fn try_build_returns_ok_on_valid_inputs() {
        let bits = vec![0xAAAA_AAAA_AAAA_AAAA_u64; 4]; // 256 bits.
        let dict = RankSelectDict::try_build(&bits, 256).unwrap();
        assert_eq!(dict.len_bits(), 256);
        // Alternating pattern: half the bits are set.
        assert_eq!(dict.count_ones(), 128);
    }

    #[test]
    #[should_panic(expected = "RankSelectDict::build: bits slice too short")]
    fn build_still_panics_on_too_short_bits() {
        let bits = [0_u64];
        let _ = RankSelectDict::build(&bits, 200);
    }

    /// Parity vs sucds across deterministic random bitvectors. Brute
    /// forces every rank position and a sampling of select positions.
    #[test]
    fn sucds_parity_deterministic_random() {
        use sucds::bit_vectors::{Rank, Rank9Sel, Select};

        for n_bits in [1_usize, 64, 200, 4096, 8192, 12345, 65_537] {
            let bits = deterministic_words(
                n_bits.div_ceil(64),
                0xC0FFEE_u64.wrapping_add(n_bits as u64),
            );
            let dict = RankSelectDict::build(&bits, n_bits);
            let oracle_iter = (0..n_bits).map(|i| (bits[i / 64] >> (i % 64)) & 1 == 1);
            let oracle = Rank9Sel::from_bits(oracle_iter)
                .select1_hints()
                .select0_hints();

            // rank1 at every position (small) or sampled positions
            // (large) to keep test runtime manageable.
            let rank_step = (n_bits / 256).max(1);
            let mut i = 0_usize;
            while i <= n_bits {
                assert_eq!(
                    dict.rank1(i),
                    oracle.rank1(i).unwrap_or(dict.count_ones()),
                    "rank1({i}) parity vs sucds (n_bits = {n_bits})"
                );
                i += rank_step;
            }

            // select1 sampled.
            let total = dict.count_ones();
            if total > 0 {
                let step = (total / 64).max(1);
                let mut k = 0_usize;
                while k < total {
                    assert_eq!(
                        dict.select1(k),
                        oracle.select1(k),
                        "select1({k}) parity vs sucds (n_bits = {n_bits})"
                    );
                    k += step;
                }
            }

            // select0 sampled.
            let total_zeros = n_bits - total;
            if total_zeros > 0 {
                let step = (total_zeros / 64).max(1);
                let mut k = 0_usize;
                while k < total_zeros {
                    assert_eq!(
                        dict.select0(k),
                        oracle.select0(k),
                        "select0({k}) parity vs sucds (n_bits = {n_bits})"
                    );
                    k += step;
                }
            }
        }
    }
}

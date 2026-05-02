//! Sliding, strided, and packed byte-window primitives.
//!
//! The hot APIs are allocation-free and operate directly over `&[u8]`.

/// Iterator over overlapping fixed-width byte windows.
#[derive(Clone, Debug)]
pub struct NGramWindows<'a, const N: usize> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a, const N: usize> NGramWindows<'a, N> {
    /// Creates an iterator over all overlapping `N`-byte windows.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    /// Returns the number of windows that have not yet been yielded.
    #[must_use]
    pub fn remaining(&self) -> usize {
        if N == 0 {
            0
        } else {
            self.bytes
                .len()
                .saturating_sub(N)
                .saturating_add(1)
                .saturating_sub(self.position)
        }
    }

    /// Returns true when no more windows remain.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }
}

impl<const N: usize> Iterator for NGramWindows<'_, N> {
    type Item = [u8; N];

    fn next(&mut self) -> Option<Self::Item> {
        if N == 0 || self.position + N > self.bytes.len() {
            return None;
        }

        let mut out = [0_u8; N];
        out.copy_from_slice(&self.bytes[self.position..self.position + N]);
        self.position += 1;
        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<const N: usize> ExactSizeIterator for NGramWindows<'_, N> {}

/// Iterator over fixed-width windows separated by a caller-provided stride.
#[derive(Clone, Debug)]
pub struct StrideWindows<'a> {
    bytes: &'a [u8],
    width: usize,
    stride: usize,
    position: usize,
}

impl<'a> StrideWindows<'a> {
    /// Creates a strided window iterator.
    ///
    /// `width == 0` or `stride == 0` produces an empty iterator.
    #[must_use]
    pub const fn new(bytes: &'a [u8], width: usize, stride: usize) -> Self {
        Self {
            bytes,
            width,
            stride,
            position: 0,
        }
    }
}

impl<'a> Iterator for StrideWindows<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.width == 0 || self.stride == 0 {
            return None;
        }
        // `self.position` may have saturated to usize::MAX after a
        // prior `saturating_add`; both `position + width` (the bounds
        // check) and `start + width` (the slice index below) need
        // checked arithmetic so adversarial widths can't panic the
        // iterator. End-of-stream simply yields None.
        let end = self.position.checked_add(self.width)?;
        if end > self.bytes.len() {
            return None;
        }
        let start = self.position;
        self.position = self.position.saturating_add(self.stride);
        Some(&self.bytes[start..end])
    }
}

/// Returns overlapping fixed-width byte windows.
#[must_use]
pub const fn ngrams<const N: usize>(bytes: &[u8]) -> NGramWindows<'_, N> {
    NGramWindows::new(bytes)
}

/// Returns strided windows over `bytes`.
#[must_use]
pub const fn strided(bytes: &[u8], width: usize, stride: usize) -> StrideWindows<'_> {
    StrideWindows::new(bytes, width, stride)
}

/// Packs an `N`-gram into the low bytes of a little-endian `u64`.
///
/// Returns `None` when `N == 0`, `N > 8`, or `bytes.len() < N`.
#[must_use]
pub fn pack_ngram_le<const N: usize>(bytes: &[u8]) -> Option<u64> {
    if N == 0 || N > 8 || bytes.len() < N {
        return None;
    }

    let mut out = 0_u64;
    for (shift, &byte) in bytes[..N].iter().enumerate() {
        out |= u64::from(byte) << (shift * 8);
    }
    Some(out)
}

/// Unpacks a little-endian `u64` into an `N`-byte n-gram.
///
/// Values with `N > 8` return `None`.
#[must_use]
pub fn unpack_ngram_le<const N: usize>(value: u64) -> Option<[u8; N]> {
    if N > 8 {
        return None;
    }

    let mut out = [0_u8; N];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = ((value >> (index * 8)) & 0xff) as u8;
    }
    Some(out)
}

/// Rolling Gear-style hash over a byte stream.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GearHash64 {
    value: u64,
}

impl GearHash64 {
    /// Creates a zero-initialized rolling hash state.
    #[must_use]
    pub const fn new() -> Self {
        Self { value: 0 }
    }

    /// Creates a rolling hash state from an existing value.
    #[must_use]
    pub(crate) const fn from_value(value: u64) -> Self {
        Self { value }
    }

    /// Returns the current hash value.
    #[must_use]
    pub const fn value(self) -> u64 {
        self.value
    }

    /// Resets the rolling hash to zero.
    pub const fn clear(&mut self) {
        self.value = 0;
    }

    /// Updates the rolling hash with one byte and returns the new value.
    pub fn update(&mut self, byte: u8) -> u64 {
        self.value = gear_update(self.value, byte);
        self.value
    }

    /// Updates the rolling hash with all bytes and returns the final value.
    pub fn update_all(&mut self, bytes: &[u8]) -> u64 {
        for &byte in bytes {
            self.update(byte);
        }
        self.value
    }
}

/// Updates a Gear-style rolling hash with one byte.
#[must_use]
pub fn gear_update(hash: u64, byte: u8) -> u64 {
    hash.rotate_left(1).wrapping_add(GEAR_TABLE[byte as usize])
}

/// Deterministic 256-entry Gear table used by streaming/chunking primitives.
pub const GEAR_TABLE: [u64; 256] = build_gear_table();

const fn build_gear_table() -> [u64; 256] {
    let mut table = [0_u64; 256];
    let mut index = 0_usize;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    while index < 256 {
        state = splitmix64(state.wrapping_add(index as u64));
        table[index] = state;
        index += 1;
    }
    table
}

const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{GearHash64, StrideWindows, ngrams, pack_ngram_le, strided, unpack_ngram_le};
    // `Vec` and `vec!` are not in the no-std prelude; alias them from
    // `alloc` for the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    #[test]
    fn ngrams_yield_overlapping_arrays() {
        let grams = ngrams::<3>(b"abcde").collect::<Vec<_>>();
        assert_eq!(grams, vec![*b"abc", *b"bcd", *b"cde"]);
        assert_eq!(ngrams::<3>(b"ab").count(), 0);
        assert_eq!(ngrams::<0>(b"abc").count(), 0);
    }

    #[test]
    fn strided_windows_skip_by_stride() {
        let windows = strided(b"abcdefgh", 3, 2).collect::<Vec<_>>();
        assert_eq!(windows, vec![b"abc".as_slice(), b"cde", b"efg"]);
        assert_eq!(strided(b"abc", 0, 1).count(), 0);
        assert_eq!(strided(b"abc", 1, 0).count(), 0);
    }

    #[test]
    fn strided_windows_overflow_safe_at_extreme_position() {
        // Regression: pre-fix the iterator's `position + width`
        // bounds check could overflow once `position` saturated to
        // usize::MAX (e.g. after a stride that pushed past the end
        // of an unusually-large slice). We can't actually allocate
        // a usize::MAX byte slice — instead drive the position to
        // the saturation point directly via repeated `.next()` and
        // verify the next call returns None instead of panicking.
        let bytes = [0u8; 8];
        let mut it = StrideWindows::new(&bytes, 4, usize::MAX / 2);
        // First iteration succeeds (position=0, width=4 <= len=8).
        assert!(it.next().is_some());
        // After saturating_add(usize::MAX / 2) the position is
        // basically usize::MAX. Next call must observe position +
        // width would overflow and return None gracefully.
        assert!(it.next().is_none());
        // Repeated calls past end remain None (idempotent).
        assert!(it.next().is_none());
    }

    #[test]
    fn ngrams_pack_and_unpack_little_endian() {
        let packed = pack_ngram_le::<4>(b"abcd").expect("4 bytes");
        assert_eq!(packed, 0x6463_6261);
        assert_eq!(unpack_ngram_le::<4>(packed), Some(*b"abcd"));
        assert_eq!(pack_ngram_le::<9>(b"abcdefghi"), None);
    }

    #[test]
    fn gear_hash_is_incremental() {
        let mut left = GearHash64::new();
        left.update_all(b"abcd");

        let mut right = GearHash64::new();
        right.update_all(b"ab");
        right.update_all(b"cd");

        assert_eq!(left, right);
        assert_ne!(left.value(), 0);
    }
}

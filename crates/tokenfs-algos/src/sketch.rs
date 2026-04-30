//! Small fixed-memory sketches used by fingerprint and calibration kernels.

/// Fixed-capacity Misra-Gries heavy-hitter sketch.
///
/// `K` is the number of counters, not the theoretical `k - 1` parameter.
/// It is intentionally array-backed so it remains usable in kernel-adjacent
/// call paths without allocation.
#[derive(Clone, Debug)]
pub struct MisraGries<const K: usize> {
    counters: [(u32, u32); K],
}

impl<const K: usize> Default for MisraGries<K> {
    fn default() -> Self {
        Self {
            counters: [(0, 0); K],
        }
    }
}

impl<const K: usize> MisraGries<K> {
    /// Creates an empty sketch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Observes one item.
    pub fn update(&mut self, item: u32) {
        if K == 0 {
            return;
        }

        for (candidate, count) in &mut self.counters {
            if *count != 0 && *candidate == item {
                *count += 1;
                return;
            }
        }

        for (candidate, count) in &mut self.counters {
            if *count == 0 {
                *candidate = item;
                *count = 1;
                return;
            }
        }

        for (_, count) in &mut self.counters {
            *count -= 1;
        }
    }

    /// Returns the live candidate counters in stable storage order.
    #[must_use]
    pub fn candidates(&self) -> [(u32, u32); K] {
        self.counters
    }
}

/// Software CRC32C over one 32-bit word, suitable as a portable hash.
#[must_use]
#[inline]
pub fn crc32c_u32(seed: u32, value: u32) -> u32 {
    let mut crc = seed;
    for shift in [0, 8, 16, 24] {
        crc = crc32c_byte(crc, ((value >> shift) & 0xff) as u8);
    }
    crc
}

#[inline]
fn crc32c_byte(seed: u32, byte: u8) -> u32 {
    let mut crc = seed ^ u32::from(byte);
    for _ in 0..8 {
        crc = (crc >> 1) ^ (0x82f6_3b78 & ((crc & 1).wrapping_neg()));
    }
    crc
}

/// Counts 4-grams into a CRC32C-hashed fixed bin array.
pub fn crc32_hash4_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    debug_assert!(BINS.is_power_of_two());
    if bytes.len() < 4 || BINS == 0 {
        return;
    }

    for window in bytes.windows(4) {
        let quad = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
        let bin = (crc32c_u32(0, quad) as usize) & (BINS - 1);
        bins[bin] += 1;
    }
}

/// Returns `count * log2(count)`.
///
/// This is the scalar fallback for the F23a `c * log2(c)` lookup-table idea.
/// The public function gives callers a stable primitive while architecture-
/// specific LUT storage can be added behind it.
#[must_use]
pub fn c_log2_c(count: u32) -> f64 {
    if count == 0 {
        0.0
    } else {
        let count = f64::from(count);
        count * count.log2()
    }
}

/// Computes entropy from integer counts using the `c * log2(c)` formulation.
#[must_use]
pub fn entropy_from_counts_u32(counts: &[u32], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }

    let sum = counts.iter().copied().map(c_log2_c).sum::<f64>();
    let total = total as f64;
    let entropy = total.log2() - sum / total;
    entropy.max(0.0) as f32
}

#[cfg(test)]
mod tests {
    use super::{MisraGries, crc32_hash4_bins, entropy_from_counts_u32};

    #[test]
    fn misra_gries_tracks_heavy_candidate() {
        let mut sketch = MisraGries::<4>::new();
        for _ in 0..50 {
            sketch.update(7);
        }
        for value in 0..25 {
            sketch.update(value);
        }
        assert!(
            sketch
                .candidates()
                .iter()
                .any(|(item, count)| *item == 7 && *count != 0)
        );
    }

    #[test]
    fn crc32_hash4_bins_counts_windows() {
        let mut bins = [0_u32; 4096];
        crc32_hash4_bins(b"abcdef", &mut bins);
        assert_eq!(bins.iter().sum::<u32>(), 3);
    }

    #[test]
    fn entropy_from_counts_matches_two_symbols() {
        let entropy = entropy_from_counts_u32(&[2, 2], 4);
        assert!((entropy - 1.0).abs() < 1e-6);
    }
}

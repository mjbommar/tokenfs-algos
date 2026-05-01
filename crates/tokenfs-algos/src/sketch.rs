//! Small fixed-memory sketches used by fingerprint and calibration kernels.

/// Pinned sketch kernels.
pub mod kernels {
    /// Portable scalar sketch kernels.
    pub mod scalar {
        /// Software CRC32C over one 32-bit word, suitable as a portable hash.
        #[must_use]
        #[inline]
        pub fn crc32c_u32(seed: u32, value: u32) -> u32 {
            super::super::crc32c_u32_scalar(seed, value)
        }

        /// Counts 4-grams into a CRC32C-hashed fixed bin array.
        pub fn crc32_hash4_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
            super::super::crc32_hash4_bins_with(bytes, bins, crc32c_u32);
        }

        /// Counts 2-grams into a CRC32C-hashed fixed bin array.
        pub fn crc32_hash2_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
            super::super::crc32_hash_ngram_bins_with::<2, BINS>(bytes, bins, crc32c_u32);
        }

        /// Counts `N`-grams, for `1 <= N <= 4`, into CRC32C-hashed bins.
        pub fn crc32_hash_ngram_bins<const N: usize, const BINS: usize>(
            bytes: &[u8],
            bins: &mut [u32; BINS],
        ) {
            super::super::crc32_hash_ngram_bins_with::<N, BINS>(bytes, bins, crc32c_u32);
        }
    }

    /// x86 SSE4.2 CRC32C sketch kernels.
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod sse42 {
        /// Returns true when the current CPU supports SSE4.2 CRC32C.
        #[must_use]
        #[inline]
        pub fn is_available() -> bool {
            std::arch::is_x86_feature_detected!("sse4.2")
        }

        /// Hardware CRC32C over one 32-bit word.
        ///
        /// # Safety
        ///
        /// The caller must ensure that SSE4.2 is available on the current CPU.
        #[must_use]
        #[target_feature(enable = "sse4.2")]
        pub unsafe fn crc32c_u32(seed: u32, value: u32) -> u32 {
            #[cfg(target_arch = "x86")]
            {
                core::arch::x86::_mm_crc32_u32(seed, value)
            }
            #[cfg(target_arch = "x86_64")]
            {
                core::arch::x86_64::_mm_crc32_u32(seed, value)
            }
        }

        /// Counts 4-grams into a CRC32C-hashed fixed bin array.
        ///
        /// # Safety
        ///
        /// The caller must ensure that SSE4.2 is available on the current CPU.
        #[target_feature(enable = "sse4.2")]
        pub unsafe fn crc32_hash4_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
            super::super::crc32_hash4_bins_with(bytes, bins, |seed, value| {
                // SAFETY: this function's target_feature contract guarantees SSE4.2.
                unsafe { crc32c_u32(seed, value) }
            });
        }

        /// Counts 2-grams into a CRC32C-hashed fixed bin array.
        ///
        /// # Safety
        ///
        /// The caller must ensure that SSE4.2 is available on the current CPU.
        #[target_feature(enable = "sse4.2")]
        pub unsafe fn crc32_hash2_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
            super::super::crc32_hash_ngram_bins_with::<2, BINS>(bytes, bins, |seed, value| {
                // SAFETY: this function's target_feature contract guarantees SSE4.2.
                unsafe { crc32c_u32(seed, value) }
            });
        }

        /// Counts `N`-grams, for `1 <= N <= 4`, into CRC32C-hashed bins.
        ///
        /// # Safety
        ///
        /// The caller must ensure that SSE4.2 is available on the current CPU.
        #[target_feature(enable = "sse4.2")]
        pub unsafe fn crc32_hash_ngram_bins<const N: usize, const BINS: usize>(
            bytes: &[u8],
            bins: &mut [u32; BINS],
        ) {
            super::super::crc32_hash_ngram_bins_with::<N, BINS>(bytes, bins, |seed, value| {
                // SAFETY: this function's target_feature contract guarantees SSE4.2.
                unsafe { crc32c_u32(seed, value) }
            });
        }
    }
}

/// Fixed-capacity Misra-Gries heavy-hitter sketch.
///
/// `K` is the number of counters, not the theoretical `k - 1` parameter.
/// It is intentionally array-backed so it remains usable in kernel-adjacent
/// call paths without allocation.
#[derive(Clone, Debug)]
pub struct MisraGries<const K: usize> {
    counters: [(u32, u32); K],
    observations: u64,
}

impl<const K: usize> Default for MisraGries<K> {
    fn default() -> Self {
        Self {
            counters: [(0, 0); K],
            observations: 0,
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
        self.observations += 1;
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

    /// Observes one item multiple times.
    pub fn update_repeated(&mut self, item: u32, count: u32) {
        for _ in 0..count {
            self.update(item);
        }
    }

    /// Observes all items from an iterator.
    pub fn update_iter<I>(&mut self, items: I)
    where
        I: IntoIterator<Item = u32>,
    {
        for item in items {
            self.update(item);
        }
    }

    /// Clears the sketch.
    pub fn clear(&mut self) {
        self.counters = [(0, 0); K];
        self.observations = 0;
    }

    /// Number of observed items.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns the live candidate counters in stable storage order.
    #[must_use]
    pub fn candidates(&self) -> [(u32, u32); K] {
        self.counters
    }

    /// Returns the current approximate count for `item`.
    #[must_use]
    pub fn estimate(&self, item: u32) -> u32 {
        self.counters
            .iter()
            .find_map(|(candidate, count)| (*count != 0 && *candidate == item).then_some(*count))
            .unwrap_or(0)
    }
}

/// Fixed-size Count-Min Sketch.
///
/// `ROWS` and `COLS` are compile-time constants so the sketch is array-backed
/// and does not allocate in hot paths.
#[derive(Clone, Debug)]
pub struct CountMinSketch<const ROWS: usize, const COLS: usize> {
    counters: [[u32; COLS]; ROWS],
    observations: u64,
}

impl<const ROWS: usize, const COLS: usize> Default for CountMinSketch<ROWS, COLS> {
    fn default() -> Self {
        Self {
            counters: [[0; COLS]; ROWS],
            observations: 0,
        }
    }
}

impl<const ROWS: usize, const COLS: usize> CountMinSketch<ROWS, COLS> {
    /// Creates an empty sketch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Clears all counters.
    pub fn clear(&mut self) {
        self.counters = [[0; COLS]; ROWS];
        self.observations = 0;
    }

    /// Observes one item.
    pub fn update(&mut self, item: u32) {
        self.update_by(item, 1);
    }

    /// Observes `count` occurrences of one item.
    pub fn update_by(&mut self, item: u32, count: u32) {
        if ROWS == 0 || COLS == 0 || count == 0 {
            return;
        }
        self.observations = self.observations.saturating_add(u64::from(count));
        for row in 0..ROWS {
            let seed = row_seed(row);
            let index = hash_index(seed, item, COLS);
            self.counters[row][index] = self.counters[row][index].saturating_add(count);
        }
    }

    /// Estimates the count for `item`.
    #[must_use]
    pub fn estimate(&self, item: u32) -> u32 {
        if ROWS == 0 || COLS == 0 {
            return 0;
        }
        let mut best = u32::MAX;
        for row in 0..ROWS {
            let seed = row_seed(row);
            let index = hash_index(seed, item, COLS);
            best = best.min(self.counters[row][index]);
        }
        best
    }

    /// Number of observed items.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }

    /// Returns raw counters for pinned tests and diagnostics.
    #[must_use]
    pub const fn counters(&self) -> &[[u32; COLS]; ROWS] {
        &self.counters
    }
}

/// Fixed-size CRC32C hash-bin n-gram sketch.
///
/// The sketch is dense and array-backed. It is meant for fast comparisons
/// against calibrated references without allocating maps in the hot path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashBinSketch<const BINS: usize> {
    bins: [u32; BINS],
    ngram: u8,
    observations: u64,
}

impl<const BINS: usize> Default for HashBinSketch<BINS> {
    fn default() -> Self {
        Self {
            bins: [0; BINS],
            ngram: 0,
            observations: 0,
        }
    }
}

impl<const BINS: usize> HashBinSketch<BINS> {
    /// Creates an empty sketch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds a sketch from `N`-grams, for `1 <= N <= 4`.
    #[must_use]
    pub fn from_ngrams<const N: usize>(bytes: &[u8]) -> Self {
        let mut sketch = Self::new();
        sketch.update_ngrams::<N>(bytes);
        sketch
    }

    /// Clears the sketch.
    pub fn clear(&mut self) {
        self.bins = [0; BINS];
        self.ngram = 0;
        self.observations = 0;
    }

    /// Adds `N`-gram observations from `bytes`, for `1 <= N <= 4`.
    pub fn update_ngrams<const N: usize>(&mut self, bytes: &[u8]) {
        if !(1..=4).contains(&N) {
            return;
        }
        let observations = ngram_windows::<N>(bytes.len());
        crc32_hash_ngram_bins::<N, BINS>(bytes, &mut self.bins);
        self.ngram = N as u8;
        self.observations = self.observations.saturating_add(observations);
    }

    /// Returns the dense bins.
    #[must_use]
    pub const fn bins(&self) -> &[u32; BINS] {
        &self.bins
    }

    /// Returns the configured n-gram length.
    #[must_use]
    pub const fn ngram(&self) -> u8 {
        self.ngram
    }

    /// Number of observed n-gram windows.
    #[must_use]
    pub const fn observations(&self) -> u64 {
        self.observations
    }
}

fn row_seed(row: usize) -> u32 {
    0x9e37_79b9_u32.wrapping_mul((row as u32).wrapping_add(1))
}

fn hash_index(seed: u32, item: u32, cols: usize) -> usize {
    let hash = crc32c_u32(seed, item) as usize;
    if cols.is_power_of_two() {
        hash & (cols - 1)
    } else {
        hash % cols
    }
}

/// CRC32C over one 32-bit word, suitable as a fast non-cryptographic hash.
#[must_use]
#[inline]
pub fn crc32c_u32(seed: u32, value: u32) -> u32 {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::sse42::is_available() {
            // SAFETY: availability was checked immediately above.
            return unsafe { kernels::sse42::crc32c_u32(seed, value) };
        }
    }

    crc32c_u32_scalar(seed, value)
}

#[inline]
fn crc32c_u32_scalar(seed: u32, value: u32) -> u32 {
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
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::sse42::is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe { kernels::sse42::crc32_hash4_bins(bytes, bins) };
            return;
        }
    }

    kernels::scalar::crc32_hash4_bins(bytes, bins);
}

/// Counts 2-grams into a CRC32C-hashed fixed bin array.
pub fn crc32_hash2_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    crc32_hash_ngram_bins::<2, BINS>(bytes, bins);
}

/// Counts `N`-grams, for `1 <= N <= 4`, into CRC32C-hashed bins.
pub fn crc32_hash_ngram_bins<const N: usize, const BINS: usize>(
    bytes: &[u8],
    bins: &mut [u32; BINS],
) {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::sse42::is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe { kernels::sse42::crc32_hash_ngram_bins::<N, BINS>(bytes, bins) };
            return;
        }
    }

    kernels::scalar::crc32_hash_ngram_bins::<N, BINS>(bytes, bins);
}

fn crc32_hash4_bins_with<const BINS: usize>(
    bytes: &[u8],
    bins: &mut [u32; BINS],
    crc32: fn(u32, u32) -> u32,
) {
    crc32_hash_ngram_bins_with::<4, BINS>(bytes, bins, crc32);
}

fn crc32_hash_ngram_bins_with<const N: usize, const BINS: usize>(
    bytes: &[u8],
    bins: &mut [u32; BINS],
    crc32: fn(u32, u32) -> u32,
) {
    if BINS == 0 || !(1..=4).contains(&N) || bytes.len() < N {
        return;
    }

    for window in bytes.windows(N) {
        let word = pack_ngram_le(window);
        let hash = crc32(0, word) as usize;
        let bin = if BINS.is_power_of_two() {
            hash & (BINS - 1)
        } else {
            hash % BINS
        };
        bins[bin] += 1;
    }
}

fn pack_ngram_le(window: &[u8]) -> u32 {
    let mut value = 0_u32;
    for (offset, &byte) in window.iter().take(4).enumerate() {
        value |= u32::from(byte) << (offset * 8);
    }
    value
}

fn ngram_windows<const N: usize>(len: usize) -> u64 {
    if !(1..=4).contains(&N) || len < N {
        0
    } else {
        (len - N + 1) as u64
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

/// Fixed-size lookup table for `count * log2(count)` reductions.
#[derive(Clone, Debug)]
pub struct CLog2Lut<const N: usize> {
    values: [f64; N],
}

impl<const N: usize> Default for CLog2Lut<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> CLog2Lut<N> {
    /// Builds a lookup table for counts `0..N`.
    #[must_use]
    pub fn new() -> Self {
        let mut values = [0.0_f64; N];
        let mut count = 1_usize;
        while count < N {
            let value = count as f64;
            values[count] = value * value.log2();
            count += 1;
        }
        Self { values }
    }

    /// Returns `count * log2(count)`, using the table when possible.
    #[must_use]
    pub fn get(&self, count: u32) -> f64 {
        self.values
            .get(count as usize)
            .copied()
            .unwrap_or_else(|| c_log2_c(count))
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

/// Computes entropy from integer counts using a caller-provided lookup table.
#[must_use]
pub fn entropy_from_counts_u32_lut<const N: usize>(
    counts: &[u32],
    total: u64,
    lut: &CLog2Lut<N>,
) -> f32 {
    if total == 0 {
        return 0.0;
    }

    let sum = counts
        .iter()
        .copied()
        .map(|count| lut.get(count))
        .sum::<f64>();
    let total = total as f64;
    let entropy = total.log2() - sum / total;
    entropy.max(0.0) as f32
}

/// Fraction of observations covered by the top `k` counts.
#[must_use]
pub fn top_k_coverage_u32(counts: &[u32], k: usize, total: u64) -> f32 {
    if total == 0 || k == 0 {
        return 0.0;
    }

    let mut top = [0_u32; 32];
    let k = k.min(top.len());
    for &count in counts {
        if count <= top[k - 1] {
            continue;
        }
        top[k - 1] = count;
        let mut index = k - 1;
        while index > 0 && top[index] > top[index - 1] {
            top.swap(index, index - 1);
            index -= 1;
        }
    }

    let covered = top[..k].iter().map(|&count| u64::from(count)).sum::<u64>();
    covered as f32 / total as f32
}

/// Dominance ratio of the largest counter.
#[must_use]
pub fn concentration_ratio_u32(counts: &[u32], total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    counts.iter().copied().max().unwrap_or(0) as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::{
        CLog2Lut, CountMinSketch, HashBinSketch, MisraGries, concentration_ratio_u32,
        crc32_hash_ngram_bins, crc32_hash2_bins, crc32_hash4_bins, entropy_from_counts_u32,
        entropy_from_counts_u32_lut, top_k_coverage_u32,
    };

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
        assert_eq!(sketch.observations(), 75);
    }

    #[test]
    fn crc32_hash4_bins_counts_windows() {
        let mut bins = [0_u32; 4096];
        crc32_hash4_bins(b"abcdef", &mut bins);
        assert_eq!(bins.iter().sum::<u32>(), 3);
    }

    #[test]
    fn crc32_hash2_bins_counts_windows() {
        let mut bins = [0_u32; 256];
        crc32_hash2_bins(b"abcdef", &mut bins);
        assert_eq!(bins.iter().sum::<u32>(), 5);
    }

    #[test]
    fn generic_ngram_bins_match_pinned_hash4() {
        let mut generic = [0_u32; 1024];
        let mut hash4 = [0_u32; 1024];
        crc32_hash_ngram_bins::<4, 1024>(b"abcdefghijklmnopqrstuvwxyz", &mut generic);
        crc32_hash4_bins(b"abcdefghijklmnopqrstuvwxyz", &mut hash4);
        assert_eq!(generic, hash4);
    }

    #[test]
    fn hash_bin_sketch_records_dense_bins_and_observations() {
        let sketch = HashBinSketch::<256>::from_ngrams::<2>(b"abcdef");

        assert_eq!(sketch.ngram(), 2);
        assert_eq!(sketch.observations(), 5);
        assert_eq!(sketch.bins().iter().sum::<u32>(), 5);
    }

    #[test]
    fn entropy_from_counts_matches_two_symbols() {
        let entropy = entropy_from_counts_u32(&[2, 2], 4);
        assert!((entropy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn count_min_never_underestimates_seen_items() {
        let mut sketch = CountMinSketch::<4, 128>::new();
        for _ in 0..10 {
            sketch.update(42);
        }
        for item in 0..20 {
            sketch.update(item);
        }

        assert!(sketch.estimate(42) >= 10);
        assert_eq!(sketch.observations(), 30);
    }

    #[test]
    fn lut_entropy_matches_direct_entropy() {
        let counts = [4, 4, 8, 0, 0, 0];
        let lut = CLog2Lut::<257>::new();
        let direct = entropy_from_counts_u32(&counts, 16);
        let via_lut = entropy_from_counts_u32_lut(&counts, 16, &lut);
        assert!((direct - via_lut).abs() < 1e-6);
    }

    #[test]
    fn top_k_and_concentration_report_coverage() {
        let counts = [10, 5, 1, 0];
        assert!((top_k_coverage_u32(&counts, 2, 16) - 15.0 / 16.0).abs() < 1e-6);
        assert!((concentration_ratio_u32(&counts, 16) - 10.0 / 16.0).abs() < 1e-6);
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn sse42_crc32_matches_scalar_when_available() {
        if !super::kernels::sse42::is_available() {
            return;
        }

        let scalar = super::kernels::scalar::crc32c_u32(0x1234, 0xfeed_beef);
        // SAFETY: availability was checked immediately above.
        let sse = unsafe { super::kernels::sse42::crc32c_u32(0x1234, 0xfeed_beef) };
        assert_eq!(scalar, sse);

        let mut scalar_bins = [0_u32; 256];
        let mut sse_bins = [0_u32; 256];
        super::kernels::scalar::crc32_hash4_bins(b"abcdefghijklmnopqrstuvwxyz", &mut scalar_bins);
        // SAFETY: availability was checked immediately above.
        unsafe {
            super::kernels::sse42::crc32_hash4_bins(b"abcdefghijklmnopqrstuvwxyz", &mut sse_bins);
        }
        assert_eq!(scalar_bins, sse_bins);

        let mut scalar_ngram = [0_u32; 256];
        let mut sse_ngram = [0_u32; 256];
        super::kernels::scalar::crc32_hash_ngram_bins::<2, 256>(
            b"abcdefghijklmnopqrstuvwxyz",
            &mut scalar_ngram,
        );
        // SAFETY: availability was checked immediately above.
        unsafe {
            super::kernels::sse42::crc32_hash_ngram_bins::<2, 256>(
                b"abcdefghijklmnopqrstuvwxyz",
                &mut sse_ngram,
            );
        }
        assert_eq!(scalar_ngram, sse_ngram);
    }
}

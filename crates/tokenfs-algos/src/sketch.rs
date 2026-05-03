//! Small fixed-memory sketches used by fingerprint and calibration kernels.

use crate::math;

/// Pinned sketch kernels.
pub mod kernels {
    /// Portable scalar sketch kernels.
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    /// x86 SSE4.2 CRC32C sketch kernels with pipelined hash4-bins
    /// implementations.
    ///
    /// `crc32_hash4_bins_pipelined` is the productized fast path for the
    /// F22 fingerprint extent's hash4 stage. Per the audit (docs/PRIMITIVE
    /// _KERNEL_BUFFET.md and the #35 sub-agent survey of CRC32C
    /// implementations), the F22 hash4_bins bottleneck is NOT the CRC
    /// dependency chain (each window hashes from `seed=0` so successive
    /// CRCs are independent) — it's the *single-stream* loop shape:
    /// `_mm_crc32_u32` has 3-cycle latency and 1-cycle throughput on
    /// Skylake-class hardware, and the loop body waits for the hash
    /// before scattering to one shared `bins` array. Two fixes here:
    /// (1) issue 4 independent CRCs per iteration so the port stays
    /// saturated, and (2) keep 4 separate per-stream bin tables to
    /// eliminate the scatter aliasing — merge them at the end.
    ///
    /// ## VPCLMULQDQ exploration (issue #53) — negative result
    ///
    /// We considered two ways VPCLMULQDQ could accelerate this path:
    ///
    /// **(a) Replace `_mm_crc32_u32` with PCLMULQDQ-based CRC32C inside
    /// the 4-stream pipeline.** Rejected. `_mm_crc32_u32` is the
    /// dedicated CRC32C instruction (3-cycle latency, 1-cycle
    /// throughput on Skylake / Zen 4) and produces a 32-bit hash from
    /// a 32-bit input directly. PCLMULQDQ produces a 128-bit
    /// polynomial product that needs Barrett reduction back to 32 bits
    /// to match CRC32C — that's 3-4 ops per window vs the 1 op
    /// `_mm_crc32_u32` already provides. The wider
    /// `_mm512_clmulepi64_epi128` does 4 lanes of 64×64 → 128-bit poly
    /// multiply in one op, but the per-lane reduction back to 32 bits
    /// (Barrett or fold) adds cost that exceeds the 4-stream
    /// `_mm_crc32_u32` baseline. Net: no speedup.
    ///
    /// **(b) Folly-style "fold by 4" CRC32C over 256-byte blocks.**
    /// Rejected for THIS code path. The hash4_bins use case is
    /// `for each 4-byte sliding window: bins[crc32c(0, window) %
    /// BINS] += 1`. Each window's CRC is computed from `seed = 0`
    /// independently, so there is no long CRC chain to fold across.
    /// Folly's fold-by-4 technique applies when computing the CRC of
    /// a contiguous N-byte buffer (file checksum, block integrity
    /// hash); it does not apply to per-window hashing. If a future
    /// caller needs single-shot CRC32C of a long buffer (kilobytes to
    /// megabytes), VPCLMULQDQ + fold-by-4/16 is the correct approach
    /// per Intel's "Fast CRC Computation Using PCLMULQDQ Instruction"
    /// whitepaper — the implementation should live in a separate
    /// `crc32c_long_buffer` primitive, not in the hash4_bins path.
    ///
    /// **Where VPCLMULQDQ might still help (not implemented):** if a
    /// `tokenfs-algos` caller needs CRC32C as a per-extent file
    /// checksum (one CRC per million bytes, not one per 4 bytes),
    /// implementing Intel's fold-by-4 via `_mm512_clmulepi64_epi128`
    /// would deliver ~2-3x over the scalar `_mm_crc32_u64`-chained
    /// loop on long inputs. That kernel would belong as a peer of
    /// `crc32c_u32`, with a tag like `crc32c_long`. We declined to
    /// add it speculatively without a concrete caller.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod sse42;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "std",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod sse42;

    /// AArch64 NEON CRC32C sketch kernels with pipelined hash4-bins
    /// implementations.
    ///
    /// `crc32_hash4_bins_pipelined` is the productized fast path on
    /// AArch64 hosts. Mirrors the SSE4.2 sibling byte-for-byte: maintain
    /// 4 in-flight CRCs (`__crc32cw` has 3-cycle latency / 1-cycle
    /// throughput on Cortex-A76 and Apple Firestorm-class cores, same
    /// pipeline shape as Skylake's `_mm_crc32_u32`) and 4 separate
    /// per-stream bin tables to break aliasing on the scatter, then merge
    /// the 4 tables at the end. Output is bit-exact with the scalar
    /// reference and the SSE4.2 path for any `(bytes, bins)` pair (CRC32C
    /// is the same Castagnoli polynomial 0x1EDC6F41 across all backends).
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "neon",
        target_arch = "aarch64"
    ))]
    pub mod neon;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "neon",
        target_arch = "aarch64"
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod neon;
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

/// Pure-software CRC32C over a byte slice. The 4-byte / 1-byte split mirrors
/// the SSE4.2 / NEON `crc32c_bytes` so all backends produce bit-exact output
/// for the same `(seed, bytes)` pair.
#[inline]
fn crc32c_bytes_scalar(seed: u32, bytes: &[u8]) -> u32 {
    let mut crc = seed;
    let mut input = bytes;
    while input.len() >= 4 {
        let value = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        crc = crc32c_u32_scalar(crc, value);
        input = &input[4..];
    }
    for &b in input {
        crc = crc32c_byte(crc, b);
    }
    crc
}

/// CRC32C over a contiguous byte slice using the fastest available backend.
///
/// Output is bit-exact across scalar, SSE4.2, and NEON backends (Castagnoli
/// polynomial 0x1EDC6F41). Use [`Crc32cHasher`] for incremental / streaming
/// inputs.
#[must_use]
pub fn crc32c_bytes(seed: u32, bytes: &[u8]) -> u32 {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::sse42::is_available() {
            // SAFETY: availability checked above.
            return unsafe { kernels::sse42::crc32c_bytes(seed, bytes) };
        }
    }
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        if kernels::neon::is_available() {
            // SAFETY: availability checked above.
            return unsafe { kernels::neon::crc32c_bytes(seed, bytes) };
        }
    }
    crc32c_bytes_scalar(seed, bytes)
}

/// Streaming CRC32C hasher.
///
/// Bytes can arrive in any-sized chunks via [`Crc32cHasher::update`]; the
/// running CRC32C state is folded forward by the per-backend `crc32c_bytes`
/// kernel selected at construction. Output is bit-exact with the one-shot
/// [`crc32c_bytes`] entry point regardless of how the input is split, and
/// bit-exact across the scalar / SSE4.2 / NEON backends (all use the
/// Castagnoli polynomial 0x1EDC6F41).
///
/// This is the right primitive for FUSE-style write paths where bytes
/// arrive in 4 KiB-ish chunks rather than as a single slice.
///
/// ## Convention
///
/// `Crc32cHasher` follows the **raw** Castagnoli CRC convention used
/// elsewhere in this module: no implicit initial XOR with `0xFFFFFFFF`, no
/// final complement. To reproduce the standard / iSCSI CRC32C value
/// (e.g. RFC 3720 Appendix B.4: `crc32c("123456789") == 0xE3069283`),
/// seed with `!0` and complement the output:
///
/// ```
/// use tokenfs_algos::sketch::Crc32cHasher;
/// let mut h = Crc32cHasher::with_seed(!0_u32);
/// h.update(b"123456789");
/// assert_eq!(h.finalize() ^ !0_u32, 0xE306_9283);
/// ```
#[derive(Clone, Debug)]
pub struct Crc32cHasher {
    state: u32,
    backend: Crc32cBackend,
}

/// CRC32C backend selected for a streaming [`Crc32cHasher`] instance.
///
/// Decided once at construction via the same runtime feature detection used by
/// the one-shot dispatcher; subsequent updates pay no per-call detection cost.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Crc32cBackend {
    /// Portable scalar fallback. Always available.
    Scalar,
    /// x86 SSE4.2 (`_mm_crc32_u32` / `_mm_crc32_u64`). Only constructible
    /// when the `std` feature is enabled (runtime detection requires
    /// `std::is_x86_feature_detected!`).
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    Sse42,
    /// AArch64 FEAT_CRC32 (`__crc32cw` / `__crc32cd`). Only constructible
    /// when the `neon` cargo feature is enabled.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    Neon,
}

impl Default for Crc32cHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32cHasher {
    /// Construct a fresh streaming CRC32C hasher with `state = 0`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_seed(0)
    }

    /// Construct a fresh streaming CRC32C hasher with the given seed.
    ///
    /// Useful for chaining hashers where the previous output is used as the
    /// next seed (e.g., to compute the CRC32C of a logically-concatenated
    /// sequence of disjoint slices).
    #[must_use]
    pub fn with_seed(seed: u32) -> Self {
        Self {
            state: seed,
            backend: detect_crc32c_backend(),
        }
    }

    /// Returns the backend selected for this hasher instance.
    #[must_use]
    pub const fn backend(&self) -> Crc32cBackend {
        self.backend
    }

    /// Reset the running CRC32C state to zero. Backend selection preserved.
    pub fn reset(&mut self) {
        self.state = 0;
    }

    /// Feed `bytes` into the running CRC32C.
    pub fn update(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        self.state = match self.backend {
            Crc32cBackend::Scalar => crc32c_bytes_scalar(self.state, bytes),
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            Crc32cBackend::Sse42 => {
                // SAFETY: variant only stored when `is_available` returned true.
                unsafe { kernels::sse42::crc32c_bytes(self.state, bytes) }
            }
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            Crc32cBackend::Neon => {
                // SAFETY: variant only stored when `is_available` returned true.
                unsafe { kernels::neon::crc32c_bytes(self.state, bytes) }
            }
        };
    }

    /// Consume the hasher and return the running CRC32C value.
    #[must_use]
    pub fn finalize(self) -> u32 {
        self.state
    }

    /// Peek at the running CRC32C without consuming the hasher.
    ///
    /// Equivalent to `finalize()` but leaves the hasher available for further
    /// updates; useful for emitting checkpoints in a streaming write path.
    #[must_use]
    pub const fn current(&self) -> u32 {
        self.state
    }
}

#[inline]
fn detect_crc32c_backend() -> Crc32cBackend {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::sse42::is_available() {
            return Crc32cBackend::Sse42;
        }
    }
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        if kernels::neon::is_available() {
            return Crc32cBackend::Neon;
        }
    }
    Crc32cBackend::Scalar
}

/// Counts 4-grams into a CRC32C-hashed fixed bin array.
///
/// On SSE4.2 hosts, dispatches to the pipelined kernel that issues 4
/// independent CRCs per iteration into 4 per-stream bin tables —
/// substantially faster than the single-stream loop on the F22 fingerprint
/// extent's hash4 stage.
pub fn crc32_hash4_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if kernels::sse42::is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe { kernels::sse42::crc32_hash4_bins_pipelined(bytes, bins) };
            return;
        }
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    {
        if kernels::neon::is_available() {
            // SAFETY: availability was checked immediately above.
            unsafe { kernels::neon::crc32_hash4_bins(bytes, bins) };
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
        count * math::log2_f64(count)
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
            values[count] = value * math::log2_f64(value);
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
    let entropy = math::log2_f64(total) - sum / total;
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
    let entropy = math::log2_f64(total) - sum / total;
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
        CLog2Lut, CountMinSketch, Crc32cHasher, HashBinSketch, MisraGries, concentration_ratio_u32,
        crc32_hash_ngram_bins, crc32_hash2_bins, crc32_hash4_bins, crc32c_bytes,
        entropy_from_counts_u32, entropy_from_counts_u32_lut, top_k_coverage_u32,
    };
    // `Crc32cBackend::Sse42` is only constructible under `feature = "std"`;
    // pull the enum into scope only on that build so we don't emit an
    // unused-import warning when the cross-backend parity test is gated
    // out (audit-R6 finding #164).
    #[cfg(feature = "std")]
    use super::Crc32cBackend;
    // `Vec` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

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

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn sse42_pipelined_with_scratch_matches_in_place_variant() {
        // The new _with_scratch variant must produce bit-exact identical
        // results to the existing stack-allocating wrapper. This pins
        // the contract so kernel callers (heap-backed scratch) can swap
        // freely with userland callers (stack-backed scratch).
        if !super::kernels::sse42::is_available() {
            return;
        }
        let bytes: Vec<u8> = (0_u8..=255).cycle().take(2048).collect();
        let mut bins_inplace = [0_u32; 1024];
        let mut bins_scratch = [0_u32; 1024];
        let mut scratch = [[0_u32; 1024]; 4];
        // SAFETY: SSE4.2 availability checked above.
        unsafe {
            super::kernels::sse42::crc32_hash4_bins_pipelined(&bytes, &mut bins_inplace);
            super::kernels::sse42::crc32_hash4_bins_pipelined_with_scratch(
                &bytes,
                &mut bins_scratch,
                &mut scratch,
            );
        }
        assert_eq!(bins_inplace, bins_scratch);

        // Reuse the scratch — the impl must zero it on entry, so a
        // second call with dirty scratch (non-zero from the previous
        // run) must still match the in-place variant.
        let bytes2: Vec<u8> = (0_u8..=255).cycle().take(4096).collect();
        let mut bins_inplace_2 = [0_u32; 1024];
        let mut bins_scratch_2 = [0_u32; 1024];
        // scratch is intentionally not re-zeroed; the impl handles it.
        unsafe {
            super::kernels::sse42::crc32_hash4_bins_pipelined(&bytes2, &mut bins_inplace_2);
            super::kernels::sse42::crc32_hash4_bins_pipelined_with_scratch(
                &bytes2,
                &mut bins_scratch_2,
                &mut scratch,
            );
        }
        assert_eq!(bins_inplace_2, bins_scratch_2);
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_pipelined_with_scratch_matches_in_place_variant() {
        if !super::kernels::neon::is_available() {
            return;
        }
        let bytes: Vec<u8> = (0_u8..=255).cycle().take(2048).collect();
        let mut bins_inplace = [0_u32; 1024];
        let mut bins_scratch = [0_u32; 1024];
        let mut scratch = [[0_u32; 1024]; 4];
        // SAFETY: FEAT_CRC32 checked above.
        unsafe {
            super::kernels::neon::crc32_hash4_bins_pipelined(&bytes, &mut bins_inplace);
            super::kernels::neon::crc32_hash4_bins_pipelined_with_scratch(
                &bytes,
                &mut bins_scratch,
                &mut scratch,
            );
        }
        assert_eq!(bins_inplace, bins_scratch);
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_crc32_matches_scalar_when_available() {
        if !super::kernels::neon::is_available() {
            return;
        }

        let scalar = super::kernels::scalar::crc32c_u32(0x1234, 0xfeed_beef);
        // SAFETY: availability was checked immediately above.
        let neon = unsafe { super::kernels::neon::crc32c_u32(0x1234, 0xfeed_beef) };
        assert_eq!(scalar, neon);

        // Deterministic input — alphabet exercises the inner 4-stream
        // loop plus the tail path (26 windows, 6 full groups + 2 tail).
        let payload: &[u8] = b"abcdefghijklmnopqrstuvwxyz";

        let mut scalar_bins = [0_u32; 256];
        let mut neon_bins = [0_u32; 256];
        super::kernels::scalar::crc32_hash4_bins(payload, &mut scalar_bins);
        // SAFETY: availability was checked immediately above.
        unsafe {
            super::kernels::neon::crc32_hash4_bins(payload, &mut neon_bins);
        }
        assert_eq!(
            scalar_bins, neon_bins,
            "neon hash4 bins diverged from scalar reference"
        );

        // Larger BINS exercises the F22 fingerprint hash4 size used in
        // production (4096) — same dispatch path as the public entry.
        let mut scalar_4k = [0_u32; 4096];
        let mut neon_4k = [0_u32; 4096];
        super::kernels::scalar::crc32_hash4_bins(payload, &mut scalar_4k);
        // SAFETY: availability was checked immediately above.
        unsafe {
            super::kernels::neon::crc32_hash4_bins(payload, &mut neon_4k);
        }
        assert_eq!(scalar_4k, neon_4k);
    }

    // ----- streaming Crc32cHasher tests --------------------------------------

    fn random_bytes(n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        while out.len() < n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out.extend_from_slice(&state.to_le_bytes());
        }
        out.truncate(n);
        out
    }

    /// Raw CRC32C of the canonical "123456789" input, computed with the
    /// "raw" convention used throughout this module (no init complement, no
    /// final complement). The standard / iSCSI CRC32C check value is
    /// `0xE3069283`; that value can be reproduced by seeding with `!0` and
    /// complementing the output (see [`nist_iscsi_crc32c_check_value`]).
    ///
    /// The raw value here was cross-checked against the same scalar polynomial
    /// implemented in Python; SSE4.2 and NEON parity is verified separately
    /// via [`crc32c_cross_backend_parity`].
    #[test]
    fn raw_crc32c_check_value() {
        assert_eq!(crc32c_bytes(0, b"123456789"), 0x58E3_FA20);
        let mut h = Crc32cHasher::new();
        h.update(b"123456789");
        assert_eq!(h.finalize(), 0x58E3_FA20);
    }

    /// Standard / iSCSI CRC32C check value from RFC 3720 Appendix B.4 /
    /// CRC-32C/ISCSI: `crc32c("123456789") == 0xE3069283` when seeded with
    /// `0xFFFFFFFF` and final-complemented. This proves the streaming hasher
    /// can reproduce the canonical convention by wrapping it appropriately.
    #[test]
    fn nist_iscsi_crc32c_check_value() {
        let mut h = Crc32cHasher::with_seed(!0_u32);
        h.update(b"123456789");
        assert_eq!(h.finalize() ^ !0_u32, 0xE306_9283);
    }

    #[test]
    fn empty_streaming_crc32c_is_seed() {
        let h = Crc32cHasher::new();
        assert_eq!(h.finalize(), 0);
        let h = Crc32cHasher::with_seed(0xdead_beef);
        assert_eq!(h.finalize(), 0xdead_beef);
    }

    #[test]
    fn crc32c_streaming_matches_one_shot_for_all_chunk_sizes() {
        let payload = random_bytes(64 * 1024);
        let expected = crc32c_bytes(0, &payload);
        for &chunk in &[1_usize, 7, 17, 64, 65, 1024, 4096] {
            let mut h = Crc32cHasher::new();
            for block in payload.chunks(chunk) {
                h.update(block);
            }
            assert_eq!(h.finalize(), expected, "mismatch for chunk={chunk}");
        }
    }

    #[test]
    fn crc32c_streaming_matches_one_shot_with_seed() {
        let payload = random_bytes(8 * 1024);
        let seed = 0xCAFE_F00D_u32;
        let expected = crc32c_bytes(seed, &payload);
        let mut h = Crc32cHasher::with_seed(seed);
        for block in payload.chunks(123) {
            h.update(block);
        }
        assert_eq!(h.finalize(), expected);
    }

    #[test]
    fn crc32c_current_does_not_consume() {
        let mut h = Crc32cHasher::new();
        h.update(b"abc");
        let snap = h.current();
        h.update(b"");
        assert_eq!(h.current(), snap);
        h.update(b"def");
        let final_value = h.finalize();
        assert_eq!(final_value, crc32c_bytes(0, b"abcdef"));
    }

    #[test]
    fn crc32c_reset_returns_to_zero() {
        let mut h = Crc32cHasher::new();
        h.update(b"poison");
        h.reset();
        assert_eq!(h.current(), 0);
        h.update(b"123456789");
        // Raw CRC32C value (no init/final XOR); see [`raw_crc32c_check_value`]
        // for context on why this differs from the iSCSI 0xE3069283.
        assert_eq!(h.finalize(), 0x58E3_FA20);
    }

    /// Force the streaming hasher onto every available backend and confirm
    /// they all produce the same CRC32C for the same chunked input.
    ///
    /// Gated on `feature = "std"` because the SSE4.2 backend variant
    /// `Crc32cBackend::Sse42` and the runtime detection
    /// `kernels::sse42::is_available()` require std (audit-R6 #164).
    #[cfg(feature = "std")]
    #[test]
    fn crc32c_cross_backend_parity() {
        let payload = random_bytes(8_192);
        let expected = super::crc32c_bytes_scalar(0, &payload);

        let mut backends: Vec<Crc32cBackend> = vec![Crc32cBackend::Scalar];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if super::kernels::sse42::is_available() {
            backends.push(Crc32cBackend::Sse42);
        }
        #[cfg(all(feature = "neon", target_arch = "aarch64"))]
        if super::kernels::neon::is_available() {
            backends.push(Crc32cBackend::Neon);
        }

        for backend in backends {
            for &chunk in &[1_usize, 17, 64, 65, 4096] {
                let mut h = Crc32cHasher::new();
                h.backend = backend;
                for block in payload.chunks(chunk) {
                    h.update(block);
                }
                assert_eq!(h.finalize(), expected, "backend={backend:?} chunk={chunk}");
            }
        }
    }

    #[test]
    fn crc32c_default_matches_new() {
        assert_eq!(
            Crc32cHasher::default().finalize(),
            Crc32cHasher::new().finalize()
        );
    }

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn sse42_crc32c_bytes_matches_scalar() {
        if !super::kernels::sse42::is_available() {
            return;
        }
        let payload = random_bytes(1024);
        let scalar = super::crc32c_bytes_scalar(0, &payload);
        // SAFETY: availability checked above.
        let sse = unsafe { super::kernels::sse42::crc32c_bytes(0, &payload) };
        assert_eq!(scalar, sse);
        // Various lengths exercise the head/body/tail code paths
        // (8-byte loop on x86_64, 4-byte step, 1-byte tail).
        for len in [0_usize, 1, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 100, 1023] {
            let s = super::crc32c_bytes_scalar(0xdead, &payload[..len]);
            // SAFETY: availability checked above.
            let v = unsafe { super::kernels::sse42::crc32c_bytes(0xdead, &payload[..len]) };
            assert_eq!(s, v, "len={len}");
        }
    }

    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_crc32c_bytes_matches_scalar() {
        if !super::kernels::neon::is_available() {
            return;
        }
        let payload = random_bytes(1024);
        let scalar = super::crc32c_bytes_scalar(0, &payload);
        // SAFETY: availability checked above.
        let neon = unsafe { super::kernels::neon::crc32c_bytes(0, &payload) };
        assert_eq!(scalar, neon);
        for len in [0_usize, 1, 3, 4, 5, 7, 8, 9, 16, 31, 32, 33, 100, 1023] {
            let s = super::crc32c_bytes_scalar(0xdead, &payload[..len]);
            // SAFETY: availability checked above.
            let v = unsafe { super::kernels::neon::crc32c_bytes(0xdead, &payload[..len]) };
            assert_eq!(s, v, "len={len}");
        }
    }
}

use core::arch::aarch64::{__crc32cb, __crc32cd, __crc32cw};

/// Returns true when the NEON CRC32C kernel is available.
///
/// NEON is mandatory in the AArch64 base ABI; the only runtime
/// question is whether the CPU exposes the FEAT_CRC32 extension.
/// That extension is universally present on ARMv8.1-A and later,
/// and on every aarch64-linux core in production today; the check
/// exists for API symmetry with [`super::sse42::is_available`] and
/// for forward compatibility with hypothetical bare-ARMv8.0
/// targets, mirroring `fingerprint::neon::public::is_available`.
#[cfg(feature = "std")]
#[must_use]
#[inline]
pub fn is_available() -> bool {
    std::arch::is_aarch64_feature_detected!("crc")
}

/// Returns true when the NEON CRC32C kernel is available.
#[cfg(not(feature = "std"))]
#[must_use]
#[inline]
pub const fn is_available() -> bool {
    // Without `std`, `is_aarch64_feature_detected!` is unavailable.
    // Conservatively report unavailable so callers fall back to the
    // scalar reference.
    false
}

/// Hardware CRC32C over one 32-bit word.
///
/// # Safety
///
/// The caller must ensure that the FEAT_CRC32 (`crc`) extension is
/// available on the current CPU.
#[must_use]
#[target_feature(enable = "crc")]
pub unsafe fn crc32c_u32(seed: u32, value: u32) -> u32 {
    __crc32cw(seed, value)
}

/// Hardware CRC32C over one byte (`__crc32cb`).
///
/// Used by the streaming [`super::super::Crc32cHasher`] tail and by
/// any caller that needs to feed unaligned head/tail bytes into a
/// wider CRC32C pipeline.
///
/// # Safety
///
/// The caller must ensure that the FEAT_CRC32 (`crc`) extension is
/// available on the current CPU.
#[must_use]
#[target_feature(enable = "crc")]
pub unsafe fn crc32c_u8(seed: u32, value: u8) -> u32 {
    __crc32cb(seed, value)
}

/// Hardware CRC32C over a 64-bit word (`__crc32cd`). One CRC32C step
/// over an 8-byte chunk; preferred for the body of [`crc32c_bytes`].
///
/// # Safety
///
/// The caller must ensure that the FEAT_CRC32 (`crc`) extension is
/// available on the current CPU.
#[must_use]
#[target_feature(enable = "crc")]
pub unsafe fn crc32c_u64(seed: u32, value: u64) -> u32 {
    __crc32cd(seed, value)
}

/// CRC32C over a contiguous byte slice, suitable for streaming.
///
/// Processes 8-byte groups via `__crc32cd`, falling back to
/// `__crc32cw` for a 4-byte step and `__crc32cb` for the last 0-3
/// bytes. The result is bit-exact with the scalar table-style
/// reference [`super::super::crc32c_bytes`].
///
/// # Safety
///
/// The caller must ensure that the FEAT_CRC32 (`crc`) extension is
/// available on the current CPU.
#[must_use]
#[target_feature(enable = "crc")]
pub unsafe fn crc32c_bytes(seed: u32, bytes: &[u8]) -> u32 {
    let mut crc = seed;
    let mut input = bytes;
    while input.len() >= 8 {
        let value = u64::from_le_bytes([
            input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
        ]);
        // SAFETY: caller guarantees `crc` extension.
        crc = unsafe { crc32c_u64(crc, value) };
        input = &input[8..];
    }
    if input.len() >= 4 {
        let value = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        // SAFETY: caller guarantees `crc` extension.
        crc = unsafe { crc32c_u32(crc, value) };
        input = &input[4..];
    }
    for &b in input {
        // SAFETY: caller guarantees `crc` extension.
        crc = unsafe { crc32c_u8(crc, b) };
    }
    crc
}

/// Counts 4-grams into a CRC32C-hashed fixed bin array.
///
/// # Safety
///
/// The caller must ensure that the FEAT_CRC32 (`crc`) extension is
/// available on the current CPU.
#[target_feature(enable = "crc")]
pub unsafe fn crc32_hash4_bins<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    // SAFETY: caller guarantees the `crc` feature is enabled, which
    // satisfies the `target_feature` contract on the pipelined entry
    // point that this delegates to.
    unsafe { crc32_hash4_bins_pipelined::<BINS>(bytes, bins) };
}

/// Pipelined hash4-bins: 4 windows in flight per iteration, 4
/// per-stream bin tables merged at the end.
///
/// Output is bit-exact with [`super::scalar::crc32_hash4_bins`] and
/// the SSE4.2 sibling [`super::sse42::crc32_hash4_bins_pipelined`]
/// for any `(bytes, bins)` pair.
///
/// # Safety
///
/// The caller must ensure that the FEAT_CRC32 (`crc`) extension is
/// available on the current CPU. `BINS` must be a power of two —
/// non-power-of-two `BINS` values would force a `% BINS` division
/// per window, which the scheduler can't pipeline. The function
/// falls back to the single-stream path for that case.
#[target_feature(enable = "crc")]
pub unsafe fn crc32_hash4_bins_pipelined<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    // Stack-allocated 4×BINS scratch (64 KiB at BINS=4096 —
    // see the SSE4.2 sibling for the same constraint and the
    // `_with_scratch` companion).
    let mut scratch = [[0_u32; BINS]; 4];
    // SAFETY: caller guarantees `crc`; bins is mutable; scratch
    // is exclusive and freshly zeroed.
    unsafe { hash4_bins_pipelined_impl::<BINS>(bytes, bins, &mut scratch) }
}

/// Heap/stack-agnostic variant: caller provides the 4×`BINS`
/// scratch. Mirrors `super::sse42::crc32_hash4_bins_pipelined_with_scratch`.
///
/// # Safety
///
/// Same FEAT_CRC32 (`crc`) precondition as
/// [`crc32_hash4_bins_pipelined`]. `scratch` must be exclusive
/// to the call.
#[target_feature(enable = "crc")]
pub unsafe fn crc32_hash4_bins_pipelined_with_scratch<const BINS: usize>(
    bytes: &[u8],
    bins: &mut [u32; BINS],
    scratch: &mut [[u32; BINS]; 4],
) {
    // SAFETY: caller guarantees `crc`; scratch is exclusive
    // and the impl zeroes it before use.
    unsafe { hash4_bins_pipelined_impl::<BINS>(bytes, bins, scratch) }
}

#[target_feature(enable = "crc")]
unsafe fn hash4_bins_pipelined_impl<const BINS: usize>(
    bytes: &[u8],
    bins: &mut [u32; BINS],
    scratch: &mut [[u32; BINS]; 4],
) {
    if BINS == 0 || bytes.len() < 4 {
        return;
    }
    // Non-power-of-two BINS would force `% BINS` per window; the
    // scheduler can't pipeline a div, so fall back to the
    // single-stream path which uses the same modulo expression.
    if !BINS.is_power_of_two() {
        super::super::crc32_hash4_bins_with(bytes, bins, |seed, value| {
            // SAFETY: this function's target_feature contract
            // guarantees the `crc` extension.
            unsafe { crc32c_u32(seed, value) }
        });
        return;
    }
    let mask = BINS - 1;

    // Four per-stream bin tables avoid the scatter aliasing
    // through one shared `bins` array. Merged at the end.
    // Caller-provided scratch is zeroed here for reuse.
    for table in scratch.iter_mut() {
        for cell in table.iter_mut() {
            *cell = 0;
        }
    }
    let [bin0, bin1, bin2, bin3] = scratch;

    // Total number of 4-byte sliding windows.
    let n_windows = bytes.len() - 3;
    let mut i = 0;

    // Inner loop: 4 independent CRCs in flight, 4 independent bin
    // increments. Each `__crc32cw(0, word)` has 3-cycle latency on
    // Cortex-A76 / Apple Firestorm but 1-cycle throughput; with 4
    // in flight the CRC unit stays saturated. Same shape as the
    // SSE4.2 sibling (Skylake `_mm_crc32_u32`).
    while i + 4 <= n_windows {
        // Pack 4 windows. We deliberately use unaligned u32 reads
        // because the windows overlap (stride 1).
        let w0 = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        let w1 = u32::from_le_bytes([bytes[i + 1], bytes[i + 2], bytes[i + 3], bytes[i + 4]]);
        let w2 = u32::from_le_bytes([bytes[i + 2], bytes[i + 3], bytes[i + 4], bytes[i + 5]]);
        let w3 = u32::from_le_bytes([bytes[i + 3], bytes[i + 4], bytes[i + 5], bytes[i + 6]]);
        // SAFETY: `crc` extension enabled by the surrounding
        // target_feature.
        let h0 = unsafe { crc32c_u32(0, w0) } as usize;
        let h1 = unsafe { crc32c_u32(0, w1) } as usize;
        let h2 = unsafe { crc32c_u32(0, w2) } as usize;
        let h3 = unsafe { crc32c_u32(0, w3) } as usize;
        bin0[h0 & mask] += 1;
        bin1[h1 & mask] += 1;
        bin2[h2 & mask] += 1;
        bin3[h3 & mask] += 1;
        i += 4;
    }

    // Tail: remaining windows go into bin0 (already in flight).
    while i < n_windows {
        let word = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        // SAFETY: `crc` extension enabled.
        let h = unsafe { crc32c_u32(0, word) } as usize;
        bin0[h & mask] += 1;
        i += 1;
    }

    // Merge the 4 per-stream tables into the caller's bins.
    for k in 0..BINS {
        bins[k] = bins[k]
            .wrapping_add(bin0[k])
            .wrapping_add(bin1[k])
            .wrapping_add(bin2[k])
            .wrapping_add(bin3[k]);
    }
}

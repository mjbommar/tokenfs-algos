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

/// Hardware CRC32C over one byte (`_mm_crc32_u8`).
///
/// Used by the streaming [`super::super::Crc32cHasher`] tail and by any
/// caller that needs to feed unaligned head/tail bytes into a wider
/// CRC32C pipeline.
///
/// # Safety
///
/// The caller must ensure that SSE4.2 is available on the current CPU.
#[must_use]
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32c_u8(seed: u32, value: u8) -> u32 {
    #[cfg(target_arch = "x86")]
    {
        core::arch::x86::_mm_crc32_u8(seed, value)
    }
    #[cfg(target_arch = "x86_64")]
    {
        core::arch::x86_64::_mm_crc32_u8(seed, value)
    }
}

/// Hardware CRC32C over a 64-bit word. Two CRC32C polynomial steps in
/// one instruction; preferred for the body of [`crc32c_bytes`].
///
/// # Safety
///
/// The caller must ensure that SSE4.2 is available on the current CPU.
/// Only available on `x86_64`.
#[cfg(target_arch = "x86_64")]
#[must_use]
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32c_u64(seed: u64, value: u64) -> u64 {
    core::arch::x86_64::_mm_crc32_u64(seed, value)
}

/// CRC32C over a contiguous byte slice, suitable for streaming.
///
/// Processes 8-byte groups via `_mm_crc32_u64` on `x86_64`, falling
/// back to `_mm_crc32_u32` and `_mm_crc32_u8` for the head/tail. The
/// result is bit-exact with the scalar table-style reference
/// [`super::super::crc32c_bytes`].
///
/// # Safety
///
/// The caller must ensure SSE4.2 is available on the current CPU.
#[must_use]
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32c_bytes(seed: u32, bytes: &[u8]) -> u32 {
    let mut crc = seed;
    let mut input = bytes;

    #[cfg(target_arch = "x86_64")]
    {
        let mut crc64 = u64::from(crc);
        while input.len() >= 8 {
            let value = u64::from_le_bytes([
                input[0], input[1], input[2], input[3], input[4], input[5], input[6], input[7],
            ]);
            // SAFETY: caller guarantees SSE4.2.
            crc64 = unsafe { crc32c_u64(crc64, value) };
            input = &input[8..];
        }
        crc = crc64 as u32;
    }

    while input.len() >= 4 {
        let value = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        // SAFETY: caller guarantees SSE4.2.
        crc = unsafe { crc32c_u32(crc, value) };
        input = &input[4..];
    }
    for &b in input {
        // SAFETY: caller guarantees SSE4.2.
        crc = unsafe { crc32c_u8(crc, b) };
    }
    crc
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

/// Pipelined hash4-bins: 4 windows in flight per iteration, 4
/// per-stream bin tables merged at the end.
///
/// Output is bit-exact with [`crc32_hash4_bins`] for any
/// `(bytes, bins)` pair.
///
/// # Safety
///
/// The caller must ensure that SSE4.2 is available on the current
/// CPU. `BINS` must be a power of two — non-power-of-two `BINS`
/// values would force a `% BINS` division per window, which the
/// scheduler can't pipeline. The function falls back to the
/// single-stream path for that case.
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32_hash4_bins_pipelined<const BINS: usize>(bytes: &[u8], bins: &mut [u32; BINS]) {
    // Stack-allocated 4×BINS scratch. At BINS=4096 (F22 hash4
    // size) that is 4*4096*4 = 64 KiB of stack — fine for
    // user-space, NOT safe for kernel/embedded stacks. Use
    // `crc32_hash4_bins_pipelined_with_scratch` to bring your
    // own scratch (heap, mmap, thread-local, etc.) in those
    // contexts.
    let mut scratch = [[0_u32; BINS]; 4];
    // SAFETY: caller guarantees SSE4.2; bins is mutable; scratch
    // is exclusive to this call and freshly zeroed.
    unsafe { hash4_bins_pipelined_impl::<BINS>(bytes, bins, &mut scratch) }
}

/// Heap/stack-agnostic variant: caller provides the 4×`BINS`
/// scratch. Use this from kernel-adjacent callers with a tight
/// stack (the inline-stack variant burns 64 KiB at BINS=4096).
///
/// The scratch is zeroed at the top of the function so the
/// caller may reuse a single buffer across many calls without
/// pre-clearing it.
///
/// # Safety
///
/// Same SSE4.2 precondition as
/// [`crc32_hash4_bins_pipelined`]. `scratch` must be exclusive
/// to the call (the function writes into all 4 sub-arrays).
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32_hash4_bins_pipelined_with_scratch<const BINS: usize>(
    bytes: &[u8],
    bins: &mut [u32; BINS],
    scratch: &mut [[u32; BINS]; 4],
) {
    // SAFETY: caller guarantees SSE4.2; scratch is exclusive
    // and the impl zeroes it before use.
    unsafe { hash4_bins_pipelined_impl::<BINS>(bytes, bins, scratch) }
}

#[target_feature(enable = "sse4.2")]
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
        // SAFETY: caller guarantees SSE4.2.
        unsafe { crc32_hash4_bins::<BINS>(bytes, bins) };
        return;
    }
    let mask = BINS - 1;

    // Four per-stream bin tables avoid the scatter aliasing
    // through one shared `bins` array. Merged at the end.
    // The caller-provided scratch is zeroed here so it may be
    // reused across calls without pre-clearing.
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
    // increments. Each `crc32_u32(0, word)` has 3-cycle latency on
    // Skylake but 1-cycle throughput; with 4 in flight the port
    // stays saturated.
    while i + 4 <= n_windows {
        // Pack 4 windows. We deliberately use unaligned u32 reads
        // because the windows overlap (stride 1).
        let w0 = u32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
        let w1 = u32::from_le_bytes([bytes[i + 1], bytes[i + 2], bytes[i + 3], bytes[i + 4]]);
        let w2 = u32::from_le_bytes([bytes[i + 2], bytes[i + 3], bytes[i + 4], bytes[i + 5]]);
        let w3 = u32::from_le_bytes([bytes[i + 3], bytes[i + 4], bytes[i + 5], bytes[i + 6]]);
        // SAFETY: SSE4.2 enabled by the surrounding target_feature.
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
        // SAFETY: SSE4.2 enabled.
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

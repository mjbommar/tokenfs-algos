#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::primitives::histogram_scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_set1_epi8,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const RLE_LANES: usize = 32;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const LANES: usize = 32;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MAX_PALETTE: usize = 16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SAMPLE_BYTES: usize = 4096;

/// Counts bytes with four private `u32` tables under an AVX2-dispatched entry.
///
/// **Planner placeholder, not a real AVX2 kernel.** x86 has no native byte
/// histogram instruction; this body performs the same scalar four-stripe
/// counting as [`crate::primitives::histogram_scalar::add_block_striped_u32`]
/// with `LANES = 4` but is reachable from the planner via an AVX2-gated
/// dispatch slot so a future genuine AVX2 implementation (gather-free /
/// pshufb 16-bin pass / radix) can replace it without re-plumbing the
/// dispatch tables. Bit-exact parity with the scalar reference is enforced
/// by `tests/avx2_parity.rs`.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2. (The body is
/// currently scalar, but the `target_feature` attribute requires the contract
/// regardless and a future vectorized body will rely on it.)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_block_stripe4_u32(bytes: &[u8], counts: &mut [u64; 256]) {
    for chunk in bytes.chunks(u32::MAX as usize) {
        let mut h0 = [0_u32; 256];
        let mut h1 = [0_u32; 256];
        let mut h2 = [0_u32; 256];
        let mut h3 = [0_u32; 256];

        let groups = chunk.len() / 4;
        for group in 0..groups {
            let base = group * 4;
            h0[chunk[base] as usize] += 1;
            h1[chunk[base + 1] as usize] += 1;
            h2[chunk[base + 2] as usize] += 1;
            h3[chunk[base + 3] as usize] += 1;
        }

        for &byte in &chunk[groups * 4..] {
            h0[byte as usize] += 1;
        }

        for index in 0..256 {
            counts[index] += u64::from(h0[index] + h1[index] + h2[index] + h3[index]);
        }
    }
}

/// Counts bytes with an AVX2 palette fast path and scalar fallback.
///
/// This is exact. If the spread sample has more than 16 distinct bytes, the
/// function falls back to the scalar local-table kernel. Otherwise each 32-byte
/// vector is compared against the sampled palette. Fully covered vectors are
/// counted with movemask/popcnt; bytes outside the sampled palette are counted
/// scalar one by one.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_block_palette_u32(bytes: &[u8], counts: &mut [u64; 256]) {
    let Some((palette, palette_len)) = sampled_palette(bytes) else {
        histogram_scalar::add_block_local_u32(bytes, counts);
        return;
    };

    if palette_len == 0 {
        return;
    }

    let mut palette_counts = [0_u64; MAX_PALETTE];
    let mut index = 0_usize;

    while index + LANES <= bytes.len() {
        let chunk = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
        let mut matched_mask = 0_u32;

        for slot in 0..palette_len {
            let needle = _mm256_set1_epi8(palette[slot] as i8);
            let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle)) as u32;
            palette_counts[slot] += u64::from(mask.count_ones());
            matched_mask |= mask;
        }

        if matched_mask != u32::MAX {
            let mut unmatched = !matched_mask;
            while unmatched != 0 {
                let lane = unmatched.trailing_zeros() as usize;
                let byte = unsafe { *bytes.as_ptr().add(index + lane) };
                counts[byte as usize] += 1;
                unmatched &= unmatched - 1;
            }
        }

        index += LANES;
    }

    for &byte in &bytes[index..] {
        counts[byte as usize] += 1;
    }

    for slot in 0..palette_len {
        counts[palette[slot] as usize] += palette_counts[slot];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn sampled_palette(bytes: &[u8]) -> Option<([u8; MAX_PALETTE], usize)> {
    let mut palette = [0_u8; MAX_PALETTE];
    let mut palette_len = 0_usize;

    if bytes.len() <= SAMPLE_BYTES {
        add_palette_sample(
            &bytes[..bytes.len().min(SAMPLE_BYTES)],
            &mut palette,
            &mut palette_len,
        )?;
        return Some((palette, palette_len));
    }

    let segment_len = SAMPLE_BYTES / 4;
    for start in [
        0,
        bytes.len() / 3,
        (bytes.len() * 2) / 3,
        bytes.len().saturating_sub(segment_len),
    ] {
        let end = (start + segment_len).min(bytes.len());
        add_palette_sample(&bytes[start..end], &mut palette, &mut palette_len)?;
    }

    Some((palette, palette_len))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn add_palette_sample(
    sample: &[u8],
    palette: &mut [u8; MAX_PALETTE],
    palette_len: &mut usize,
) -> Option<()> {
    for &byte in sample {
        if palette[..*palette_len].contains(&byte) {
            continue;
        }
        if *palette_len == MAX_PALETTE {
            return None;
        }
        palette[*palette_len] = byte;
        *palette_len += 1;
    }

    Some(())
}

/// AVX2 byte-histogram with a constant-chunk RLE fast path bolted onto a
/// scalar four-stripe core.
///
/// For each 32-byte chunk we compare the chunk against
/// `_mm256_set1_epi8(chunk[0])`. If every lane matches, the chunk is a
/// 32-byte run of one value and we increment that bin by 32 in a single
/// counter update. Otherwise the chunk falls through to the four-stripe
/// scalar path. On real-world inputs (text, code, executables) constant
/// chunks are common (zero-fill, space-fill, padding) and the fast path
/// converts 32 increments into 1. On uniform random inputs no chunk is
/// constant and the kernel degenerates to the four-stripe path; the
/// constant-chunk probe is one AVX2 broadcast + cmpeq + movemask + branch
/// per 32 bytes, which costs at most a few percent of the stripe core.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn add_block_rle_stripe4_u32(bytes: &[u8], counts: &mut [u64; 256]) {
    if bytes.is_empty() {
        return;
    }

    let mut h0 = [0_u32; 256];
    let mut h1 = [0_u32; 256];
    let mut h2 = [0_u32; 256];
    let mut h3 = [0_u32; 256];

    let mut index = 0_usize;
    let mut runs = [0_u64; 256];

    while index + RLE_LANES <= bytes.len() {
        // SAFETY: index + RLE_LANES <= bytes.len() by the loop guard.
        let chunk = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
        // SAFETY: index < bytes.len() because RLE_LANES > 0.
        let head = unsafe { *bytes.as_ptr().add(index) };
        let needle = _mm256_set1_epi8(head as i8);
        let mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, needle)) as u32;

        if mask == u32::MAX {
            // Constant chunk: bin += 32 in one go.
            runs[head as usize] += RLE_LANES as u64;
        } else {
            // Heterogeneous chunk: scalar four-stripe over the chunk's
            // 32 bytes. 32 is divisible by 4 so there is no inner tail.
            let chunk_bytes = &bytes[index..index + RLE_LANES];
            for group in 0..(RLE_LANES / 4) {
                let base = group * 4;
                h0[chunk_bytes[base] as usize] += 1;
                h1[chunk_bytes[base + 1] as usize] += 1;
                h2[chunk_bytes[base + 2] as usize] += 1;
                h3[chunk_bytes[base + 3] as usize] += 1;
            }
        }
        index += RLE_LANES;
    }

    // Scalar tail (< 32 bytes). Use stripe-0 for simplicity; spreading
    // across the four stripes here would not move the needle.
    for &byte in &bytes[index..] {
        h0[byte as usize] += 1;
    }

    // Reduce stripes + RLE accumulator into the public table.
    for slot in 0..256 {
        counts[slot] += u64::from(h0[slot] + h1[slot] + h2[slot] + h3[slot]) + runs[slot];
    }
}

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
const LANES: usize = 32;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MAX_PALETTE: usize = 16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const SAMPLE_BYTES: usize = 4096;

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

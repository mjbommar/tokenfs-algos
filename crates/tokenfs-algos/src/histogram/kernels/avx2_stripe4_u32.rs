use super::{ByteHistogram, histogram_scalar};
use crate::primitives::histogram_avx2;

/// Returns true when AVX2 is available at runtime.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2")
}

/// Returns true when AVX2 is available at runtime.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Builds a byte histogram with this pinned kernel.
///
/// If AVX2 is unavailable at runtime, this falls back to `stripe4-u32`.
#[must_use]
pub fn block(bytes: &[u8]) -> ByteHistogram {
    let mut histogram = ByteHistogram::new();
    add_block(bytes, &mut histogram);
    histogram
}

/// Adds bytes into an existing histogram with this pinned kernel.
pub fn add_block(bytes: &[u8], histogram: &mut ByteHistogram) {
    if is_available() {
        // SAFETY: availability was checked immediately above.
        unsafe {
            histogram_avx2::add_block_stripe4_u32(bytes, histogram.counts_mut_for_primitives());
        }
    } else {
        histogram_scalar::add_block_striped_u32::<4>(bytes, histogram.counts_mut_for_primitives());
    }
    histogram.add_to_total_for_primitives(bytes.len() as u64);
}

/// Adds bytes with AVX2 without checking runtime availability.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2.
pub unsafe fn add_block_unchecked(bytes: &[u8], histogram: &mut ByteHistogram) {
    unsafe {
        histogram_avx2::add_block_stripe4_u32(bytes, histogram.counts_mut_for_primitives());
    }
    histogram.add_to_total_for_primitives(bytes.len() as u64);
}

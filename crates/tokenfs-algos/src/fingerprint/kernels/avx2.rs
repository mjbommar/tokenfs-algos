use super::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, extent_auto};

/// Returns true when the fused AVX2/SSE4.2 block kernel is available.
#[cfg(feature = "std")]
#[must_use]
pub fn is_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("sse4.2")
}

/// Returns true when the fused AVX2/SSE4.2 block kernel is available.
#[cfg(not(feature = "std"))]
#[must_use]
pub const fn is_available() -> bool {
    false
}

/// Computes a compact F22/content fingerprint for one 256-byte block.
///
/// If the current CPU does not support AVX2 and SSE4.2, this falls
/// back to the runtime-dispatched default path.
#[must_use]
pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    if is_available() {
        // SAFETY: availability was checked immediately above.
        unsafe { super::super::block_avx2_unchecked(bytes) }
    } else {
        super::block_auto(bytes)
    }
}

/// Computes a compact F22/content fingerprint without checking CPU
/// features.
///
/// # Safety
///
/// The caller must ensure the current CPU supports AVX2 and SSE4.2.
#[must_use]
pub unsafe fn block_unchecked(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    unsafe { super::super::block_avx2_unchecked(bytes) }
}

/// Computes an aggregate F22/content fingerprint for any byte slice.
///
/// The extent path uses the runtime-dispatched default extent
/// accumulator; the AVX2-specific implementation is block-scoped.
#[must_use]
pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
    extent_auto(bytes)
}

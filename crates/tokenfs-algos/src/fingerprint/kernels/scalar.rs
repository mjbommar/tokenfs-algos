use super::{BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, block_scalar, extent_scalar};

/// Computes a compact F22/content fingerprint for one 256-byte block.
#[must_use]
pub fn block(bytes: &[u8; BLOCK_SIZE]) -> BlockFingerprint {
    block_scalar(bytes)
}

/// Computes an exact aggregate F22/content fingerprint for any byte
/// slice.
#[must_use]
pub fn extent(bytes: &[u8]) -> ExtentFingerprint {
    extent_scalar(bytes)
}

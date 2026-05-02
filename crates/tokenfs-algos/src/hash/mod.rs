//! Small non-cryptographic hash primitives, plus a SHA-256 module with
//! hardware-accelerated backends.
//!
//! The mixers in [`kernels::scalar`] (FNV-1a 64, SplitMix-style mix64) are
//! for sketching, bucketing, and deterministic benchmarking. They are *not*
//! integrity or security hashes.
//!
//! For cryptographic SHA-256 — with x86 SHA-NI and AArch64 FEAT_SHA2
//! backends behind a runtime feature gate — see the [`sha256`] submodule.

pub mod sha256;

/// Pinned hash kernels.
pub mod kernels {
    /// Runtime-dispatched hash kernels.
    ///
    /// These currently use scalar implementations. Backend-specific hashes can
    /// be added here without changing public call sites.
    pub mod auto {
        /// Computes FNV-1a 64-bit hash.
        #[must_use]
        pub fn fnv1a64(bytes: &[u8]) -> u64 {
            super::scalar::fnv1a64(bytes)
        }

        /// Computes a lightweight 64-bit byte-stream mix.
        #[must_use]
        pub fn mix64(bytes: &[u8], seed: u64) -> u64 {
            super::scalar::mix64(bytes, seed)
        }
    }

    /// Portable scalar hash kernels.
    pub mod scalar {
        use crate::hash::mix_word;

        /// FNV-1a 64-bit offset basis.
        pub const FNV1A64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        /// FNV-1a 64-bit prime.
        pub const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

        /// Computes FNV-1a 64-bit hash.
        #[must_use]
        pub fn fnv1a64(bytes: &[u8]) -> u64 {
            let mut hash = FNV1A64_OFFSET;
            for &byte in bytes {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(FNV1A64_PRIME);
            }
            hash
        }

        /// Computes a lightweight 64-bit byte-stream mix.
        ///
        /// This is a deterministic SplitMix-style streaming mixer. It is useful
        /// as a second hash family for sketches and tests where CRC32C's
        /// polynomial shape is not desirable.
        #[must_use]
        pub fn mix64(bytes: &[u8], seed: u64) -> u64 {
            let mut state = seed ^ ((bytes.len() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            let mut chunks = bytes.chunks_exact(8);
            for chunk in &mut chunks {
                let value = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                state = mix_word(state ^ value);
            }

            let mut tail = 0_u64;
            for (shift, &byte) in chunks.remainder().iter().enumerate() {
                tail |= u64::from(byte) << (shift * 8);
            }
            mix_word(state ^ tail)
        }
    }
}

/// Computes FNV-1a 64-bit hash with the public default path.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    kernels::auto::fnv1a64(bytes)
}

/// Computes a lightweight 64-bit byte-stream mix with the public default path.
#[must_use]
pub fn mix64(bytes: &[u8], seed: u64) -> u64 {
    kernels::auto::mix64(bytes, seed)
}

/// SplitMix64 finalizer used by the scalar mixer.
#[must_use]
pub const fn mix_word(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::{fnv1a64, kernels, mix64};

    #[test]
    fn fnv1a64_matches_known_values() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"hello"), 0xa430_d846_80aa_bd0b);
    }

    #[test]
    fn public_hashes_match_pinned_scalar() {
        let bytes = b"tokenfs-algos";
        assert_eq!(fnv1a64(bytes), kernels::scalar::fnv1a64(bytes));
        assert_eq!(mix64(bytes, 123), kernels::scalar::mix64(bytes, 123));
    }

    #[test]
    fn mix64_uses_seed() {
        assert_ne!(mix64(b"abc", 1), mix64(b"abc", 2));
    }
}

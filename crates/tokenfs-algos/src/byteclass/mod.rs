//! Byte classification primitives.
//!
//! The public [`classify`] path uses runtime dispatch when a tested optimized
//! backend is available. Pinned kernels live under [`kernels`] for
//! reproducible benchmarks and forensic comparisons.

/// Counts coarse byte classes in one pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ByteClassCounts {
    /// Printable ASCII bytes, excluding bytes counted as whitespace.
    pub printable_ascii: u64,
    /// ASCII whitespace bytes.
    pub whitespace: u64,
    /// ASCII control bytes excluding whitespace.
    pub control: u64,
    /// Bytes with the high bit set.
    pub high_bit: u64,
    /// Other bytes.
    pub other: u64,
}

impl ByteClassCounts {
    /// Counts all bytes in the class summary.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.printable_ascii + self.whitespace + self.control + self.high_bit + self.other
    }
}

/// Byte-class kernels.
pub mod kernels {
    use super::ByteClassCounts;

    /// Runtime-dispatched byte-class classifier.
    pub mod auto {
        use super::ByteClassCounts;

        /// Counts coarse byte classes using the best available kernel.
        #[must_use]
        pub fn classify(bytes: &[u8]) -> ByteClassCounts {
            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::classify(bytes) };
                }
            }

            super::scalar::classify(bytes)
        }
    }

    /// Portable scalar byte-class classifier.
    pub mod scalar {
        use super::ByteClassCounts;

        /// Counts coarse byte classes in one scalar pass.
        #[must_use]
        pub fn classify(bytes: &[u8]) -> ByteClassCounts {
            let mut counts = ByteClassCounts::default();
            add(bytes, &mut counts);
            counts
        }

        pub(super) fn add(bytes: &[u8], counts: &mut ByteClassCounts) {
            for &byte in bytes {
                match byte {
                    b'\t' | b'\n' | b'\r' | b' ' => counts.whitespace += 1,
                    0x20..=0x7e => counts.printable_ascii += 1,
                    0x00..=0x1f | 0x7f => counts.control += 1,
                    0x80..=0xff => counts.high_bit += 1,
                }
            }
        }
    }

    /// AVX2 byte-class classifier.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::{ByteClassCounts, scalar};

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m256i, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_loadu_si256,
            _mm256_movemask_epi8, _mm256_set1_epi8,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m256i, _mm256_cmpeq_epi8, _mm256_cmpgt_epi8, _mm256_loadu_si256,
            _mm256_movemask_epi8, _mm256_set1_epi8,
        };

        const LANES: usize = 32;

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

        /// Counts coarse byte classes with AVX2.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn classify(bytes: &[u8]) -> ByteClassCounts {
            let mut counts = ByteClassCounts::default();
            let mut index = 0;

            let space = _mm256_set1_epi8(b' ' as i8);
            let tab = _mm256_set1_epi8(b'\t' as i8);
            let newline = _mm256_set1_epi8(b'\n' as i8);
            let carriage_return = _mm256_set1_epi8(b'\r' as i8);
            let delete = _mm256_set1_epi8(0x7f_i8);

            while index + LANES <= bytes.len() {
                let chunk =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };

                let high_bit_mask = _mm256_movemask_epi8(chunk) as u32;
                let low_control_mask =
                    (_mm256_movemask_epi8(_mm256_cmpgt_epi8(space, chunk)) as u32) & !high_bit_mask;
                let space_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, space)) as u32;
                let whitespace_mask = space_mask
                    | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, tab)) as u32)
                    | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, newline)) as u32)
                    | (_mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, carriage_return)) as u32);
                let delete_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, delete)) as u32;
                let control_mask = (low_control_mask | delete_mask) & !whitespace_mask;
                let printable_mask = !(high_bit_mask | low_control_mask | delete_mask | space_mask);

                counts.high_bit += u64::from(high_bit_mask.count_ones());
                counts.whitespace += u64::from(whitespace_mask.count_ones());
                counts.control += u64::from(control_mask.count_ones());
                counts.printable_ascii += u64::from(printable_mask.count_ones());

                index += LANES;
            }

            scalar::add(&bytes[index..], &mut counts);
            counts
        }
    }
}

/// Counts coarse byte classes using the public runtime-dispatched path.
#[must_use]
pub fn classify(bytes: &[u8]) -> ByteClassCounts {
    kernels::auto::classify(bytes)
}

/// Returns true when the slice is strongly ASCII/text dominated.
#[must_use]
pub fn is_ascii_dominant(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    let counts = classify(bytes);
    (counts.printable_ascii + counts.whitespace) * 10 >= bytes.len() as u64 * 9
}

#[cfg(test)]
mod tests {
    use super::{classify, is_ascii_dominant, kernels};

    #[test]
    fn classifies_ascii_text() {
        let counts = classify(b"abc 123\n");
        assert_eq!(counts.whitespace, 2);
        assert_eq!(counts.printable_ascii, 6);
        assert!(is_ascii_dominant(b"abc 123\n"));
    }

    #[test]
    fn public_default_matches_scalar_on_edge_cases() {
        for bytes in byteclass_cases() {
            assert_eq!(classify(&bytes), kernels::scalar::classify(&bytes));
        }
    }

    #[cfg(all(
        feature = "std",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !kernels::avx2::is_available() {
            return;
        }

        let base = byteclass_cases()
            .into_iter()
            .flatten()
            .cycle()
            .take(16 * 1024 + 63)
            .collect::<Vec<_>>();
        for start in 0..64 {
            for len in [0_usize, 1, 2, 7, 31, 32, 33, 255, 256, 4096, 8191] {
                let end = (start + len).min(base.len());
                let bytes = &base[start..end];
                // SAFETY: availability was checked above.
                let actual = unsafe { kernels::avx2::classify(bytes) };
                assert_eq!(
                    actual,
                    kernels::scalar::classify(bytes),
                    "AVX2 mismatch at start {start}, len {}",
                    bytes.len()
                );
            }
        }
    }

    fn byteclass_cases() -> Vec<Vec<u8>> {
        vec![
            Vec::new(),
            vec![0],
            vec![0; 4096],
            b"abc 123\n\t\r".to_vec(),
            (0_u8..=255).collect(),
            (0_u8..=255).cycle().take(4097).collect(),
            (0_usize..8192)
                .map(|i| (i.wrapping_mul(37) ^ (i >> 3).wrapping_mul(11)) as u8)
                .collect(),
            b"\x00\x01\x02\t\n\r hello world \x7f\x80\xff"
                .repeat(257)
                .to_vec(),
        ]
    }
}

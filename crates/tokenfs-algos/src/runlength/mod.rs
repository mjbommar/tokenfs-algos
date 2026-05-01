//! Run-length detection and statistics.
//!
//! [`summarize`] returns a full [`RunLengthSummary`] (transitions, longest
//! run, runs ≥ 4, bytes covered by long runs). It is currently scalar.
//!
//! [`transitions`] is a focused fast-path that counts only adjacent-byte
//! mismatches; it has a runtime-dispatched AVX2 backend with bit-exact scalar
//! parity. Pin the scalar reference via [`kernels::scalar::transitions`] when
//! reproducibility matters.

/// Summary of equal-byte runs in a byte slice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunLengthSummary {
    /// Total bytes scanned.
    pub total_bytes: usize,
    /// Number of transitions where `bytes[i] != bytes[i - 1]`.
    pub transitions: u64,
    /// Longest equal-byte run.
    pub longest_run: usize,
    /// Number of equal-byte runs with length at least 4.
    pub runs_ge4: u64,
    /// Bytes that belong to equal-byte runs with length at least 4.
    pub bytes_in_runs_ge4: u64,
}

impl RunLengthSummary {
    /// Fraction of bytes that belong to runs with length at least 4.
    #[must_use]
    pub fn run_fraction(self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            self.bytes_in_runs_ge4 as f32 / self.total_bytes as f32
        }
    }

    /// Fraction of adjacent byte pairs that are transitions.
    #[must_use]
    pub fn transition_rate(self) -> f32 {
        if self.total_bytes <= 1 {
            0.0
        } else {
            self.transitions as f32 / (self.total_bytes - 1) as f32
        }
    }
}

/// Run-length kernels.
pub mod kernels {
    /// Runtime-dispatched run-length kernels.
    pub mod auto {
        use crate::runlength::RunLengthSummary;

        /// Counts transitions using the best available kernel.
        #[must_use]
        pub fn transitions(bytes: &[u8]) -> u64 {
            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::avx2::transitions(bytes) };
                }
            }

            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { super::neon::transitions(bytes) };
                }
            }

            super::scalar::transitions(bytes)
        }

        /// Builds a full [`RunLengthSummary`] using the best available kernel.
        ///
        /// The full summary is currently scalar; the auto path matches
        /// [`crate::runlength::kernels::scalar::summarize`].
        #[must_use]
        pub fn summarize(bytes: &[u8]) -> RunLengthSummary {
            super::scalar::summarize(bytes)
        }
    }

    /// Portable scalar run-length kernels.
    pub mod scalar {
        use crate::runlength::RunLengthSummary;

        /// Counts transitions where `bytes[i] != bytes[i - 1]`.
        #[must_use]
        pub fn transitions(bytes: &[u8]) -> u64 {
            if bytes.len() < 2 {
                return 0;
            }
            let mut count = 0_u64;
            for index in 1..bytes.len() {
                if bytes[index] != bytes[index - 1] {
                    count += 1;
                }
            }
            count
        }

        /// Builds a full [`RunLengthSummary`] in one scalar pass.
        #[must_use]
        pub fn summarize(bytes: &[u8]) -> RunLengthSummary {
            if bytes.is_empty() {
                return RunLengthSummary::default();
            }

            let mut summary = RunLengthSummary {
                total_bytes: bytes.len(),
                longest_run: 1,
                ..RunLengthSummary::default()
            };
            let mut run_len = 1_usize;

            for index in 1..bytes.len() {
                if bytes[index] == bytes[index - 1] {
                    run_len += 1;
                } else {
                    flush_run(run_len, &mut summary);
                    summary.transitions += 1;
                    run_len = 1;
                }
            }
            flush_run(run_len, &mut summary);

            summary
        }

        fn flush_run(run_len: usize, summary: &mut RunLengthSummary) {
            summary.longest_run = summary.longest_run.max(run_len);
            if run_len >= 4 {
                summary.runs_ge4 += 1;
                summary.bytes_in_runs_ge4 =
                    summary.bytes_in_runs_ge4.saturating_add(run_len as u64);
            }
        }
    }

    /// AVX2 run-length kernels.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    pub mod avx2 {
        use super::scalar;

        #[cfg(target_arch = "x86")]
        use core::arch::x86::{
            __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        };
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{
            __m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
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

        /// Counts transitions where `bytes[i] != bytes[i - 1]`, AVX2 path.
        ///
        /// Equivalent to [`scalar::transitions`]: each iteration loads the
        /// current 32-byte window and the same window shifted one byte
        /// earlier, then `_mm256_cmpeq_epi8` + popcnt of the inverted
        /// movemask gives the transition count for indices `i..i+32`. Tail
        /// bytes are handled scalar.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        #[target_feature(enable = "avx2")]
        #[must_use]
        pub unsafe fn transitions(bytes: &[u8]) -> u64 {
            if bytes.len() < 2 {
                return 0;
            }

            let mut count = 0_u64;
            let mut index = 1_usize;

            // Vector loop: process 32 transition checks per iteration.
            // We compare bytes[index .. index+32] against bytes[index-1 ..
            // index+31] using two unaligned loads. Modern x86 cores handle
            // overlapping unaligned loads at full L1 bandwidth, so the
            // double-load is essentially free relative to a single shifted
            // permutation.
            while index + LANES <= bytes.len() {
                // SAFETY: index >= 1 and index + LANES <= bytes.len(), so
                // both loads stay inside `bytes` (the earlier load reads
                // indices index-1 ..= index-1+31 = index+30, the later
                // reads index ..= index+31).
                let curr =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
                let prev =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index - 1).cast::<__m256i>()) };
                let eq = _mm256_cmpeq_epi8(curr, prev);
                let eq_mask = _mm256_movemask_epi8(eq) as u32;
                count += u64::from(LANES as u32 - eq_mask.count_ones());
                index += LANES;
            }

            // Scalar tail covers any positions index..bytes.len() that
            // didn't fit a full 32-byte window. The slice starts at
            // bytes[index - 1] so the very first scalar comparison
            // recovers the (bytes[index - 1], bytes[index]) pair that
            // would otherwise be split across the boundary.
            count + scalar::transitions(&bytes[index - 1..])
        }
    }

    /// AArch64 NEON run-length kernels.
    ///
    /// NEON has no movemask, so the lane-equality mask is reduced to a
    /// running count via mask-AND-with-`1` plus `vaddlvq_u8`. The 16-byte
    /// vector width means we process 16 transition checks per inner
    /// iteration vs. AVX2's 32, but the L1 bandwidth profile is similar.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    pub mod neon {
        use super::scalar;

        use core::arch::aarch64::{vaddlvq_u8, vandq_u8, vceqq_u8, vdupq_n_u8, vld1q_u8};

        const LANES: usize = 16;

        /// Returns true when NEON is available at runtime.
        ///
        /// NEON is mandatory on AArch64; the function exists for API
        /// symmetry with [`super::avx2::is_available`].
        #[must_use]
        pub const fn is_available() -> bool {
            true
        }

        /// Counts transitions where `bytes[i] != bytes[i - 1]`, NEON path.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports NEON. On AArch64
        /// this is part of the base ABI; the precondition is always met for
        /// `target_arch = "aarch64"` builds.
        #[target_feature(enable = "neon")]
        #[must_use]
        pub unsafe fn transitions(bytes: &[u8]) -> u64 {
            if bytes.len() < 2 {
                return 0;
            }

            let mut count = 0_u64;
            let mut index = 1_usize;
            let one = vdupq_n_u8(1);

            while index + LANES <= bytes.len() {
                // SAFETY: index >= 1 and index + LANES <= bytes.len(), so
                // both loads stay inside `bytes`.
                let curr = unsafe { vld1q_u8(bytes.as_ptr().add(index)) };
                let prev = unsafe { vld1q_u8(bytes.as_ptr().add(index - 1)) };
                let eq = vceqq_u8(curr, prev);
                // eq is 0xff for equal lanes, 0x00 otherwise; AND with 1 to
                // get a per-lane 0/1 indicator, then horizontal-add as u16
                // for an equality count in 0..=16.
                let eq_count = u64::from(vaddlvq_u8(vandq_u8(eq, one)));
                count += LANES as u64 - eq_count;
                index += LANES;
            }

            count + scalar::transitions(&bytes[index - 1..])
        }
    }
}

/// Counts transitions where `bytes[i] != bytes[i - 1]` using the public
/// runtime-dispatched path.
#[must_use]
pub fn transitions(bytes: &[u8]) -> u64 {
    kernels::auto::transitions(bytes)
}

/// Builds a full [`RunLengthSummary`] using the public runtime-dispatched path.
#[must_use]
pub fn summarize(bytes: &[u8]) -> RunLengthSummary {
    kernels::auto::summarize(bytes)
}

/// Returns the longest equal-byte run.
#[must_use]
pub fn longest_run(bytes: &[u8]) -> usize {
    summarize(bytes).longest_run
}

#[cfg(test)]
mod tests {
    use super::{kernels, summarize, transitions};

    #[test]
    fn summarizes_runs_and_transitions() {
        let summary = summarize(b"aaaabcc");
        assert_eq!(summary.total_bytes, 7);
        assert_eq!(summary.longest_run, 4);
        assert_eq!(summary.runs_ge4, 1);
        assert_eq!(summary.bytes_in_runs_ge4, 4);
        assert_eq!(summary.transitions, 2);
    }

    #[test]
    fn transitions_match_summary_for_small_inputs() {
        for case in [
            &b""[..],
            &b"a"[..],
            &b"ab"[..],
            &b"aaaa"[..],
            &b"aaaabcc"[..],
            &b"abababab"[..],
        ] {
            assert_eq!(transitions(case), kernels::scalar::transitions(case));
            assert_eq!(transitions(case), summarize(case).transitions);
        }
    }
}

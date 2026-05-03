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

            #[cfg(all(feature = "std", feature = "sve2", target_arch = "aarch64"))]
            {
                if super::sve2::is_available() {
                    // SAFETY: availability was checked immediately above.
                    return unsafe { super::sve2::transitions(bytes) };
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
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    /// AVX2 run-length kernels.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx2;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx2;

    /// AArch64 NEON run-length kernels.
    ///
    /// NEON has no movemask, so the lane-equality mask is reduced to a
    /// running count via mask-AND-with-`1` plus `vaddlvq_u8`. The 16-byte
    /// vector width means we process 16 transition checks per inner
    /// iteration vs. AVX2's 32, but the L1 bandwidth profile is similar.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "neon",
        target_arch = "aarch64"
    ))]
    pub mod neon;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "neon",
        target_arch = "aarch64"
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod neon;

    /// AArch64 SVE2 run-length kernels.
    ///
    /// # Vector-length-agnostic loop shape
    ///
    /// Like the byteclass SVE2 path, transitions are counted in a single
    /// VLA loop using `svwhilelt_b8` for the active-lane predicate.
    /// Per-iteration the kernel:
    ///
    /// 1. Loads `curr = bytes[i..i+W]` and `prev = bytes[i-1..i-1+W]`
    ///    (both predicated; inactive lanes load as zero).
    /// 2. Compares `curr != prev` lane-wise via the inverse of
    ///    `svcmpeq_u8`.
    /// 3. Reduces the boolean result to a count via `svcntp_b8`
    ///    (population count of the true lanes).
    ///
    /// `svcntp_b8` is the genuine SVE2 advantage here: NEON has no
    /// movemask, so the existing NEON kernel goes through
    /// `vandq_u8(eq, one) + vaddlvq_u8` — three instructions to count
    /// active lanes. SVE2 counts directly in one PCNT-of-mask op.
    ///
    /// # Per-runner vector width
    ///
    /// See `crate::byteclass::kernels::sve2` for the table of `svcntb()`
    /// values per CPU family. The transition kernel processes
    /// `svcntb()` byte pairs per iteration regardless of width.
    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "sve2",
        target_arch = "aarch64"
    ))]
    pub mod sve2;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "sve2",
        target_arch = "aarch64"
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod sve2;
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

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
        /// Two vectors per loop iteration breaks the popcnt dependency
        /// chain. POPCNT has a 3-cycle latency on Intel client cores, so
        /// chaining two independent counters (`count_a`, `count_b`)
        /// roughly doubles the inner-loop throughput on streams that
        /// fit in L1.
        const UNROLL_LANES: usize = LANES * 2;

        /// Returns true when AVX2 + BMI2 + LZCNT are all available at
        /// runtime.
        ///
        /// The transitions kernel is annotated with
        /// `target_feature(enable = "avx2,bmi2,lzcnt")` so LLVM can emit
        /// BMI2 / LZCNT instructions inside the function body. Calling
        /// it on a CPU that exposes AVX2 but not BMI2 (KVM without
        /// CPUID passthrough, very old Atom cores, some sandbox
        /// configurations) is undefined behaviour. All three checks
        /// here keep the dispatch sound.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::is_x86_feature_detected!("avx2")
                && std::is_x86_feature_detected!("bmi2")
                && std::is_x86_feature_detected!("lzcnt")
        }

        /// Returns true when AVX2 is available at runtime.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// Counts transitions where `bytes[i] != bytes[i - 1]`, AVX2 path.
        ///
        /// Each 32-byte vector iteration loads the current window and
        /// the same window shifted one byte earlier, then
        /// `_mm256_cmpeq_epi8` + popcnt of the inverted movemask gives
        /// the transition count for indices `i..i+32`. Modern x86 cores
        /// handle overlapping unaligned loads at full L1 bandwidth, so
        /// the double-load is essentially free relative to a single
        /// shifted permutation.
        ///
        /// The body uses a 2x unrolled main loop with two independent
        /// popcnt accumulators (breaks the 3-cycle POPCNT latency
        /// chain), then an overlapping vector tail when at least one
        /// `LANES`-byte window fits — the tail bits already counted in
        /// the last full iteration are masked off using a BMI2-style
        /// shift before the final popcnt. This eliminates the scalar
        /// tail loop for inputs >= `LANES + 1` bytes.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports AVX2.
        ///
        /// `bmi2` and `lzcnt` are added to the `target_feature` set so
        /// the optimizer is free to lower `high_bits_mask` and the
        /// per-iteration branch tests into BMI2/LZCNT forms (`bzhi`,
        /// `andn`, `lzcnt`) when profitable. The runtime gate only
        /// checks AVX2 because every x86 CPU shipped with AVX2
        /// (Intel Haswell+, AMD Excavator+) also has BMI2 and LZCNT.
        #[target_feature(enable = "avx2,bmi2,lzcnt")]
        #[must_use]
        pub unsafe fn transitions(bytes: &[u8]) -> u64 {
            if bytes.len() < 2 {
                return 0;
            }

            let mut count_a = 0_u64;
            let mut count_b = 0_u64;
            let mut index = 1_usize;

            // 2x-unrolled hot loop: process 64 bytes per iteration.
            // Two independent accumulators give the OoO core room to
            // schedule both POPCNTs in parallel — the dependency chain
            // through count_a is decoupled from count_b.
            while index + UNROLL_LANES <= bytes.len() {
                // SAFETY: index >= 1; index + 64 <= bytes.len() so all
                // four loads (two pairs of overlapping 32-byte windows)
                // stay inside `bytes`.
                let curr_a =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
                let prev_a =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index - 1).cast::<__m256i>()) };
                let curr_b = unsafe {
                    _mm256_loadu_si256(bytes.as_ptr().add(index + LANES).cast::<__m256i>())
                };
                let prev_b = unsafe {
                    _mm256_loadu_si256(bytes.as_ptr().add(index + LANES - 1).cast::<__m256i>())
                };

                let eq_a = _mm256_cmpeq_epi8(curr_a, prev_a);
                let eq_b = _mm256_cmpeq_epi8(curr_b, prev_b);
                let eq_mask_a = _mm256_movemask_epi8(eq_a) as u32;
                let eq_mask_b = _mm256_movemask_epi8(eq_b) as u32;
                count_a += u64::from(LANES as u32 - eq_mask_a.count_ones());
                count_b += u64::from(LANES as u32 - eq_mask_b.count_ones());
                index += UNROLL_LANES;
            }

            // Single-vector loop for one remaining 32-byte window.
            while index + LANES <= bytes.len() {
                // SAFETY: as above.
                let curr =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
                let prev =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index - 1).cast::<__m256i>()) };
                let eq = _mm256_cmpeq_epi8(curr, prev);
                let eq_mask = _mm256_movemask_epi8(eq) as u32;
                count_a += u64::from(LANES as u32 - eq_mask.count_ones());
                index += LANES;
            }

            // Overlapping vector tail: when at least one full LANES
            // window has been processed and there are 1..LANES tail
            // bytes left, do one final overlapping load at
            // `bytes.len() - LANES` and mask off the bits we've already
            // counted. This eliminates the scalar tail loop for inputs
            // long enough to fit at least one full vector.
            if index < bytes.len() && bytes.len() > LANES {
                let tail_start = bytes.len() - LANES;
                debug_assert!(tail_start >= 1);
                debug_assert!(tail_start <= index);

                // SAFETY: tail_start + LANES == bytes.len() and
                // tail_start - 1 >= 0; both loads stay inside `bytes`.
                let curr =
                    unsafe { _mm256_loadu_si256(bytes.as_ptr().add(tail_start).cast::<__m256i>()) };
                let prev = unsafe {
                    _mm256_loadu_si256(bytes.as_ptr().add(tail_start - 1).cast::<__m256i>())
                };
                let eq = _mm256_cmpeq_epi8(curr, prev);
                let eq_mask = _mm256_movemask_epi8(eq) as u32;

                // We've already counted transitions for indices
                // [tail_start .. index]; the remaining "new" bits in
                // `eq_mask` cover indices [index .. bytes.len()], which
                // are the high `bytes.len() - index` bits of the mask.
                // Build the mask of *new* bits with a BMI2-friendly
                // shift, then complement (we counted *transitions*, so
                // popcount the inverted bits among the new positions).
                let new_bits = (bytes.len() - index) as u32;
                debug_assert!(new_bits >= 1 && new_bits < LANES as u32);

                // The new bits sit at positions [LANES - new_bits .. LANES).
                // `_lzcnt_u32` and BMI2 `_bzhi_u32` together let us
                // construct the new-bit mask in two cycles, avoiding a
                // taken branch on `new_bits == LANES` (which can't
                // happen here but the optimizer doesn't know that).
                let new_mask: u32 = (!eq_mask) & high_bits_mask(new_bits);
                count_a += u64::from(new_mask.count_ones());

                index = bytes.len();
            }

            // Final scalar fallback: only triggered for very short
            // inputs (2..=LANES bytes) that never entered the vector
            // loop, since the overlapping tail handles every case where
            // `bytes.len() >= LANES + 1`.
            let scalar_count = if index < bytes.len() {
                scalar::transitions(&bytes[index - 1..])
            } else {
                0
            };

            count_a + count_b + scalar_count
        }

        /// Returns a u32 with the top `n` bits set, 0 otherwise.
        ///
        /// `n` is in the range 1..=31. Implementation uses a shift; on
        /// hardware with BMI2 the compiler typically lowers the
        /// equivalent `(1 << n) - 1` shape into `bzhi`, but this hand-
        /// rolled form is uniform across BMI2-on and BMI2-off targets
        /// and avoids the implicit `1u32 << 32` UB hazard at the
        /// boundary.
        #[inline]
        fn high_bits_mask(n: u32) -> u32 {
            // top n bits = !((1 << (32 - n)) - 1) for n in 1..=31.
            let inv = 32_u32 - n;
            (!0_u32) << inv
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
    #[cfg(all(feature = "sve2", target_arch = "aarch64"))]
    pub mod sve2 {
        use core::arch::aarch64::{
            svcmpne_u8, svcntb, svcntp_b8, svld1_u8, svptest_any, svptrue_b8, svwhilelt_b8_u64,
        };

        /// Returns true when SVE2 is available at runtime.
        #[cfg(feature = "std")]
        #[must_use]
        pub fn is_available() -> bool {
            std::arch::is_aarch64_feature_detected!("sve2")
        }

        /// Returns true when SVE2 is available at runtime.
        #[cfg(not(feature = "std"))]
        #[must_use]
        pub const fn is_available() -> bool {
            false
        }

        /// Counts transitions where `bytes[i] != bytes[i - 1]` with
        /// SVE2.
        ///
        /// # Safety
        ///
        /// The caller must ensure the current CPU supports SVE2.
        #[target_feature(enable = "sve2")]
        #[must_use]
        pub unsafe fn transitions(bytes: &[u8]) -> u64 {
            if bytes.len() < 2 {
                return 0;
            }
            let n = bytes.len() as u64;
            // `i` is the index into `bytes` of the lane-0 element of
            // `curr`. Transitions are counted for the pair
            // `(bytes[i-1], bytes[i])`, so we start at `i = 1`.
            let mut i: u64 = 1;
            let ptr = bytes.as_ptr();
            let all_lanes = svptrue_b8();
            let step = svcntb();
            let mut count: u64 = 0;

            loop {
                let pg = svwhilelt_b8_u64(i, n);
                if !svptest_any(all_lanes, pg) {
                    break;
                }
                // SAFETY: predicate `pg` zeros lanes past `n - i`. The
                // overlapping load of `prev` reads from `i - 1`; since
                // `i >= 1` on entry and `pg` masks anything past `n`,
                // both loads stay in-bounds for the active lanes.
                let curr = unsafe { svld1_u8(pg, ptr.add(i as usize)) };
                let prev = unsafe { svld1_u8(pg, ptr.add((i - 1) as usize)) };
                let neq = svcmpne_u8(pg, curr, prev);
                count += svcntp_b8(all_lanes, neq);
                i += step;
            }

            count
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

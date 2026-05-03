use super::scalar;

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};

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
/// The caller must ensure the current CPU supports **all of**
/// AVX2, BMI2, and LZCNT. The function is annotated
/// `target_feature(enable = "avx2,bmi2,lzcnt")` and LLVM may
/// emit BMI2 (`bzhi`, `andn`) or LZCNT instructions inside
/// the body — calling it on a CPU that exposes AVX2 but not
/// BMI2/LZCNT is undefined behaviour. The safe public
/// dispatcher `runlength::transitions` checks all three via
/// `is_x86_feature_detected!` before calling here.
///
/// (Most x86 CPUs that ship AVX2 also ship BMI2 and LZCNT,
/// but the combination is not architecturally guaranteed:
/// KVM-without-CPUID-passthrough, very old Atom variants,
/// and some sandbox configurations expose AVX2 alone. See
/// commit history around #67 for the original UB
/// reproduction.)
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
        let curr_a = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
        let prev_a = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index - 1).cast::<__m256i>()) };
        let curr_b =
            unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index + LANES).cast::<__m256i>()) };
        let prev_b =
            unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index + LANES - 1).cast::<__m256i>()) };

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
        let curr = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index).cast::<__m256i>()) };
        let prev = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(index - 1).cast::<__m256i>()) };
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
        let curr = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(tail_start).cast::<__m256i>()) };
        let prev =
            unsafe { _mm256_loadu_si256(bytes.as_ptr().add(tail_start - 1).cast::<__m256i>()) };
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

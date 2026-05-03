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

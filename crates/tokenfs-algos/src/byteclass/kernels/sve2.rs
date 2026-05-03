use crate::byteclass::ByteClassCounts;

use core::arch::aarch64::{
    svand_b_z, svcmpeq_n_u8, svcmpge_n_u8, svcmplt_n_u8, svcntb, svcntp_b8, svld1_u8, svnot_b_z,
    svorr_b_z, svptest_any, svptrue_b8, svwhilelt_b8_u64,
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

/// Counts coarse byte classes with SVE2.
///
/// One vector-length-agnostic loop processes the whole input,
/// using `svwhilelt_b8` for the active-lane predicate and a
/// 4-way `svcntp_b8` (PCNT-of-mask) reduction per class.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SVE2. SVE2 is
/// a strict superset of SVE, so this implies SVE availability as
/// well.
#[target_feature(enable = "sve2")]
#[must_use]
pub unsafe fn classify(bytes: &[u8]) -> ByteClassCounts {
    let mut counts = ByteClassCounts::default();
    let n = bytes.len() as u64;
    let mut i: u64 = 0;
    let ptr = bytes.as_ptr();
    let all_lanes = svptrue_b8();
    let step = svcntb();

    // The classification predicates are derived once per input
    // chunk via SVE's predicated `svcmp*_n_u8` family. The "_n_"
    // suffix means the second operand is a scalar (broadcast),
    // matching NEON's `vceqq_u8(_, vdupq_n_u8(c))` shape.
    loop {
        // Active-lane predicate: lane `j` active iff `i + j < n`.
        let pg = svwhilelt_b8_u64(i, n);
        // Exit when the predicate is empty (i.e. we've consumed
        // the whole input). Equivalent to `i >= n` rounded up to
        // the next vector boundary.
        if !svptest_any(all_lanes, pg) {
            break;
        }

        // SAFETY: the predicate `pg` zeros lanes past `n - i`,
        // so the vector load reads at most `step` bytes from
        // `ptr.add(i)`; SVE guarantees inactive lanes don't
        // fault the load past valid memory.
        let v = unsafe { svld1_u8(pg, ptr.add(i as usize)) };

        // High bit (>= 0x80).
        let high_bit_mask = svcmpge_n_u8(pg, v, 0x80);

        // Low ASCII control (< 0x20). High-bit and low-control
        // are disjoint by construction (mutually exclusive value
        // ranges), so we use the active-lane predicate `pg` for
        // both compares without further masking.
        let low_control_mask = svcmplt_n_u8(pg, v, 0x20);

        // Whitespace = space | tab | newline | carriage-return.
        let space_mask = svcmpeq_n_u8(pg, v, b' ');
        let tab_mask = svcmpeq_n_u8(pg, v, b'\t');
        let nl_mask = svcmpeq_n_u8(pg, v, b'\n');
        let cr_mask = svcmpeq_n_u8(pg, v, b'\r');
        let whitespace_mask = svorr_b_z(
            pg,
            svorr_b_z(pg, space_mask, tab_mask),
            svorr_b_z(pg, nl_mask, cr_mask),
        );

        // DEL (0x7f).
        let delete_mask = svcmpeq_n_u8(pg, v, 0x7f);

        // Control = (low_control | delete) & !whitespace.
        let control_raw = svorr_b_z(pg, low_control_mask, delete_mask);
        let not_ws = svnot_b_z(pg, whitespace_mask);
        let control_mask = svand_b_z(pg, control_raw, not_ws);

        // Printable = pg & !(high_bit | low_control | delete |
        // whitespace). All four disjuncts are within `pg` so the
        // outer AND with `pg` is implicit in the zeroing
        // `svnot_b_z` (inactive lanes stay zero).
        let high_or_low = svorr_b_z(pg, high_bit_mask, low_control_mask);
        let del_or_ws = svorr_b_z(pg, delete_mask, whitespace_mask);
        let nonprintable = svorr_b_z(pg, high_or_low, del_or_ws);
        let printable_mask = svand_b_z(pg, pg, svnot_b_z(pg, nonprintable));

        counts.high_bit += svcntp_b8(all_lanes, high_bit_mask);
        counts.whitespace += svcntp_b8(all_lanes, whitespace_mask);
        counts.control += svcntp_b8(all_lanes, control_mask);
        counts.printable_ascii += svcntp_b8(all_lanes, printable_mask);

        i += step;
    }

    counts
}

/// Validates UTF-8 with the SVE2 fast-path.
///
/// # Strategy
///
/// SVE2 doesn't bring a Keiser-Lemire shuffle DFA win over NEON
/// for the general UTF-8 problem (the table-driven shuffles are
/// the same instruction count), but the vector-length-agnostic
/// ASCII-fast-path is a clear win: one predicated comparison
/// over the whole input checks "any high-bit byte?" in roughly
/// `n / svcntb()` cycles. When the input is pure ASCII (the
/// common case for code, JSON, logs, etc.) we return immediately
/// without invoking the heavier scalar/NEON DFA.
///
/// On non-ASCII input we fall through to the existing NEON DFA
/// in [`crate::byteclass::utf8_neon`], which is bit-exact with
/// the scalar reference.
///
/// # Safety
///
/// The caller must ensure the current CPU supports SVE2. NEON is
/// part of the AArch64 base ABI so it is implicitly available
/// on every host where SVE2 is reported.
#[target_feature(enable = "sve2,neon")]
#[must_use]
pub unsafe fn validate_utf8(bytes: &[u8]) -> crate::byteclass::Utf8Validation {
    let n = bytes.len() as u64;
    let mut i: u64 = 0;
    let ptr = bytes.as_ptr();
    let all_lanes = svptrue_b8();
    let step = svcntb();

    // ASCII pre-scan: walk the input checking for any byte with
    // the high bit set. If we reach the end with none, the slice
    // is valid ASCII (and therefore valid UTF-8 by construction).
    loop {
        let pg = svwhilelt_b8_u64(i, n);
        if !svptest_any(all_lanes, pg) {
            return crate::byteclass::Utf8Validation {
                valid: true,
                valid_up_to: bytes.len(),
                error_len: 0,
            };
        }
        // SAFETY: `pg` zeros lanes past `n - i`; SVE guarantees
        // inactive lanes do not fault past valid memory.
        let v = unsafe { svld1_u8(pg, ptr.add(i as usize)) };
        let high_bit_mask = svcmpge_n_u8(pg, v, 0x80);
        if svptest_any(all_lanes, high_bit_mask) {
            break;
        }
        i += step;
    }

    // Non-ASCII detected; defer to the NEON DFA path. NEON is
    // mandatory on AArch64 so this is always safe to call.
    // SAFETY: target_feature("sve2") on this fn implies SVE2 is
    // available, which on every shipping AArch64 CPU also
    // implies NEON (the AArch64 base ABI).
    unsafe { crate::byteclass::utf8_neon::validate_utf8(bytes) }
}

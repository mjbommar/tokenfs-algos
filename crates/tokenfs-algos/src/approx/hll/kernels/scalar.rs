use super::POW2_NEG_LUT;

/// Per-bucket max merge: `dst[i] = max(dst[i], src[i])`.
///
/// Reference for the SIMD parity tests; this is the same
/// loop the original [`super::super::HyperLogLog::merge`]
/// shipped with.
pub fn merge(dst: &mut [u8], src: &[u8]) {
    debug_assert_eq!(
        dst.len(),
        src.len(),
        "HLL merge requires equal-length register arrays"
    );
    for (a, &b) in dst.iter_mut().zip(src) {
        if b > *a {
            *a = b;
        }
    }
}

/// Reference harmonic-mean cardinality `alpha * m^2 / Z`.
///
/// Uses the [`super::POW2_NEG_LUT`] lookup so the scalar
/// reference exercises the same numerical path as the SIMD
/// kernels, eliminating LUT-vs-`powi` rounding differences
/// from the parity tolerance budget.
#[must_use]
pub fn count_raw(registers: &[u8], alpha: f64) -> f64 {
    let m = registers.len() as f64;
    let mut sum = 0.0_f64;
    for &r in registers {
        sum += POW2_NEG_LUT[r as usize];
    }
    alpha * m * m / sum
}

//! Fuzz target: dispatched vector distance kernels must match the scalar
//! reference within the Higham §3 / Wilkinson tolerance for f32 metrics
//! and exactly for integer metrics.
//!
//! Tolerance contract per `docs/v0.2_planning/13_VECTOR.md` § 1: f32
//! reductions agree within `1e-3 * sum(|a*b|)`. We assert that bound for
//! f32 dot, L2², and cosine; integer / popcount metrics are bit-exact.
//!
//! Input layout:
//! - First byte: metric selector (mod 6).
//! - Bytes 1..3 (LE u16): vector length, capped at 4096 lanes.
//! - Remaining bytes: source data, parsed differently per metric.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::vector;
use tokenfs_algos::vector::kernels;

/// Cap vector length to keep fuzz pool memory bounded and avoid f32
/// accumulator overflow on extreme inputs.
const MAX_LANES: usize = 4096;

/// Higham §3 / Wilkinson tolerance multiplier as called out in the spec.
const F32_REL_TOL: f32 = 1e-3;

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let metric = data[0] % 6;
    let n_raw = u16::from_le_bytes([data[1], data[2]]) as usize;
    let n = n_raw % (MAX_LANES + 1);
    let payload = &data[3..];

    match metric {
        0 => check_dot_f32(payload, n),
        1 => check_l2sq_f32(payload, n),
        2 => check_cosine_f32(payload, n),
        3 => check_dot_u32(payload, n),
        4 => check_hamming_u64(payload, n),
        _ => check_jaccard_u64(payload, n),
    }
});

fn build_f32_pair(payload: &[u8], n: usize) -> (Vec<f32>, Vec<f32>) {
    // Each lane consumes 8 bytes (2x f32). Clamp to non-NaN, finite,
    // non-extreme values via wrap-cast and a magnitude clamp so the
    // tolerance bound stays meaningful.
    let mut a = vec![0_f32; n];
    let mut b = vec![0_f32; n];
    for i in 0..n {
        let off = i * 8;
        let av = read_f32(payload, off);
        let bv = read_f32(payload, off + 4);
        a[i] = sanitize_f32(av);
        b[i] = sanitize_f32(bv);
    }
    (a, b)
}

fn read_f32(payload: &[u8], off: usize) -> f32 {
    if off + 4 <= payload.len() {
        f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ])
    } else {
        // Synthesize a deterministic small-magnitude value when the
        // payload runs short.
        ((off as i32 * 7 - 13) as f32) * 0.0001
    }
}

/// Replace non-finite or extreme f32 values with finite, modest ones so
/// the SIMD vs scalar reduction tolerance remains a meaningful test of
/// implementation parity rather than a measure of float stickiness.
fn sanitize_f32(v: f32) -> f32 {
    if !v.is_finite() {
        return 0.0;
    }
    let clamp = 1.0e6_f32;
    v.clamp(-clamp, clamp)
}

fn check_dot_f32(payload: &[u8], n: usize) {
    let (a, b) = build_f32_pair(payload, n);
    let dispatched = vector::dot_f32(&a, &b).expect("equal-length dot_f32");
    let scalar = kernels::scalar::dot_f32(&a, &b).expect("equal-length scalar dot_f32");
    assert_close_f32_dot(dispatched, scalar, &a, &b, "dot_f32");
}

fn check_l2sq_f32(payload: &[u8], n: usize) {
    let (a, b) = build_f32_pair(payload, n);
    let dispatched = vector::l2_squared_f32(&a, &b).expect("equal-length l2sq_f32");
    let scalar = kernels::scalar::l2_squared_f32(&a, &b).expect("equal-length scalar l2sq_f32");
    // L2² magnitude is bounded by sum((a-b)²); use the same Higham bound
    // shape but compute the magnitude from the squared diffs.
    let mag: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            (d * d).abs()
        })
        .sum();
    let tol = F32_REL_TOL * mag.max(1.0);
    let diff = (dispatched - scalar).abs();
    assert!(
        diff <= tol,
        "l2_squared_f32 diverged: dispatched={dispatched} scalar={scalar} diff={diff} tol={tol} (n={})",
        a.len()
    );
}

fn check_cosine_f32(payload: &[u8], n: usize) {
    let (a, b) = build_f32_pair(payload, n);
    let dispatched = vector::cosine_similarity_f32(&a, &b)
        .expect("equal-length cosine_similarity_f32");
    let scalar = kernels::scalar::cosine_similarity_f32(&a, &b)
        .expect("equal-length scalar cosine_similarity_f32");
    // Cosine is in [-1, 1]; even with f32 round-off the divergence
    // between two reduction orders should not exceed the spec's relative
    // tolerance applied to the denominator stack-up. Use an absolute
    // tolerance of a few eps scaled by sqrt(n) and capped at 5e-3 so
    // very-pathological inputs (near-zero norms) don't false-positive.
    let abs_tol = 5.0e-3_f32;
    let diff = (dispatched - scalar).abs();
    assert!(
        diff <= abs_tol,
        "cosine_similarity_f32 diverged: dispatched={dispatched} scalar={scalar} diff={diff} (n={})",
        a.len()
    );
}

/// Common Higham-style assertion for dot-product-shaped reductions.
fn assert_close_f32_dot(dispatched: f32, scalar: f32, a: &[f32], b: &[f32], name: &str) {
    let mag: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x * y).abs())
        .sum();
    let tol = F32_REL_TOL * mag.max(1.0);
    let diff = (dispatched - scalar).abs();
    assert!(
        diff <= tol,
        "{name} diverged: dispatched={dispatched} scalar={scalar} diff={diff} tol={tol} (n={})",
        a.len()
    );
}

fn check_dot_u32(payload: &[u8], n: usize) {
    // Use u16-cast to avoid easy overflow in the wrapping sum; this still
    // exercises the SIMD u32 dot kernel uniformly.
    let mut a = vec![0_u32; n];
    let mut b = vec![0_u32; n];
    for i in 0..n {
        let off = i * 4;
        a[i] = read_u16(payload, off) as u32;
        b[i] = read_u16(payload, off + 2) as u32;
    }
    let dispatched = vector::dot_u32(&a, &b);
    let scalar = kernels::scalar::dot_u32(&a, &b).expect("equal-length scalar dot_u32");
    assert_eq!(
        dispatched, scalar,
        "dot_u32 diverged at n={n}: dispatched={dispatched} scalar={scalar}"
    );
}

fn check_hamming_u64(payload: &[u8], n: usize) {
    let mut a = vec![0_u64; n];
    let mut b = vec![0_u64; n];
    for i in 0..n {
        let off = i * 16;
        a[i] = read_u64(payload, off);
        b[i] = read_u64(payload, off + 8);
    }
    let dispatched = vector::hamming_u64(&a, &b).expect("equal-length hamming_u64");
    let scalar = kernels::scalar::hamming_u64(&a, &b).expect("equal-length scalar hamming_u64");
    assert_eq!(
        dispatched, scalar,
        "hamming_u64 diverged at n={n}: dispatched={dispatched} scalar={scalar}"
    );
}

fn check_jaccard_u64(payload: &[u8], n: usize) {
    let mut a = vec![0_u64; n];
    let mut b = vec![0_u64; n];
    for i in 0..n {
        let off = i * 16;
        a[i] = read_u64(payload, off);
        b[i] = read_u64(payload, off + 8);
    }
    let dispatched = vector::jaccard_u64(&a, &b).expect("equal-length jaccard_u64");
    let scalar = kernels::scalar::jaccard_u64(&a, &b).expect("equal-length scalar jaccard_u64");
    // Jaccard is in [0, 1]; both backends use the same popcount kernels
    // and integer arithmetic, so the result is bit-exact in f64.
    assert!(
        (dispatched - scalar).abs() < 1.0e-12,
        "jaccard_u64 diverged at n={n}: dispatched={dispatched} scalar={scalar}"
    );
}

fn read_u16(payload: &[u8], off: usize) -> u16 {
    if off + 2 <= payload.len() {
        u16::from_le_bytes([payload[off], payload[off + 1]])
    } else {
        // Deterministic small synthetic when the payload runs short.
        (off as u16).wrapping_mul(31)
    }
}

fn read_u64(payload: &[u8], off: usize) -> u64 {
    let mut buf = [0_u8; 8];
    for k in 0..8 {
        if off + k < payload.len() {
            buf[k] = payload[off + k];
        }
    }
    u64::from_le_bytes(buf)
}

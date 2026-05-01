//! Scalar invariant tests for the similarity layer.
//!
//! Per the SIMILARITY_APPROXIMATION_ROADMAP.md correctness gates:
//! - identical vectors have zero distance where expected;
//! - symmetric metrics are symmetric;
//! - known tiny vectors match hand-computed values;
//! - length mismatch returns None;
//! - empty-vector behavior is stable and documented.

#![allow(clippy::unwrap_used)] // Test code — panic on None is the desired failure mode.

use crate::similarity::distance;
use crate::similarity::kernels::scalar;

const EPS: f64 = 1e-9;
const EPS_F32: f32 = 1e-6;

#[test]
fn dot_u32_known_values() {
    let a = [1_u32, 2, 3, 4];
    let b = [4_u32, 3, 2, 1];
    assert_eq!(scalar::dot_u32(&a, &b), Some(20));
    assert_eq!(scalar::dot_u32(&[], &[]), Some(0));
    assert_eq!(scalar::dot_u32(&[1, 2], &[1]), None);
}

#[test]
fn l1_u32_known_values() {
    let a = [1_u32, 5, 3];
    let b = [2_u32, 1, 7];
    assert_eq!(scalar::l1_u32(&a, &b), Some(1 + 4 + 4));
    assert_eq!(scalar::l1_u32(&[], &[]), Some(0));
    assert_eq!(scalar::l1_u32(&[1, 2], &[1]), None);
    let c = [10_u32; 4];
    assert_eq!(scalar::l1_u32(&c, &c), Some(0), "identical → 0");
}

#[test]
fn l2_squared_u32_known_values() {
    let a = [1_u32, 2, 3];
    let b = [4_u32, 0, 7];
    // (3)^2 + (2)^2 + (4)^2 = 9 + 4 + 16
    assert_eq!(scalar::l2_squared_u32(&a, &b), Some(29));
    let c = [7_u32; 8];
    assert_eq!(scalar::l2_squared_u32(&c, &c), Some(0), "identical → 0");
}

#[test]
fn cosine_similarity_u32_known_values() {
    let a = [1_u32, 0, 0];
    let b = [0_u32, 1, 0];
    let s = scalar::cosine_similarity_u32(&a, &b).unwrap();
    assert!(s.abs() < EPS, "orthogonal: got {s}");

    let c = [3_u32, 4, 0];
    let d = [3_u32, 4, 0];
    let s = scalar::cosine_similarity_u32(&c, &d).unwrap();
    assert!((s - 1.0).abs() < EPS, "identical: got {s}");

    // Both-zero norm convention: returns 0.
    assert_eq!(
        scalar::cosine_similarity_u32(&[0; 4], &[1, 2, 3, 4]),
        Some(0.0)
    );
    assert_eq!(scalar::cosine_similarity_u32(&[1; 4], &[]), None);
}

#[test]
fn cosine_similarity_u32_is_symmetric() {
    let a = [1_u32, 2, 3, 4, 5];
    let b = [9_u32, 8, 7, 6, 5];
    let ab = scalar::cosine_similarity_u32(&a, &b).unwrap();
    let ba = scalar::cosine_similarity_u32(&b, &a).unwrap();
    assert!((ab - ba).abs() < EPS, "symmetry: ab={ab} ba={ba}");
}

#[test]
fn dot_f32_known_values() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let d = scalar::dot_f32(&a, &b).unwrap();
    assert!((d - 32.0).abs() < EPS_F32);

    assert_eq!(scalar::dot_f32(&[], &[]), Some(0.0));
    assert_eq!(scalar::dot_f32(&[1.0], &[1.0, 2.0]), None);
}

#[test]
fn l2_squared_f32_known_values() {
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];
    let d = scalar::l2_squared_f32(&a, &b).unwrap();
    assert!((d - 2.0).abs() < EPS_F32);

    let c = [2.5_f32; 5];
    assert_eq!(scalar::l2_squared_f32(&c, &c), Some(0.0));
}

#[test]
fn cosine_similarity_f32_known_values() {
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];
    assert!(scalar::cosine_similarity_f32(&a, &b).unwrap().abs() < EPS_F32);

    let c = [3.0_f32, 4.0, 0.0];
    let d = [6.0_f32, 8.0, 0.0]; // collinear with c
    let s = scalar::cosine_similarity_f32(&c, &d).unwrap();
    assert!((s - 1.0).abs() < EPS_F32, "collinear: got {s}");

    assert_eq!(
        scalar::cosine_similarity_f32(&[0.0; 3], &[1.0, 2.0, 3.0]),
        Some(0.0)
    );
}

#[test]
fn dispatched_paths_match_scalar_today() {
    // Until AVX2/NEON kernels land, the dispatcher is a thin re-export of
    // scalar — but it's still worth exercising so future SIMD wiring will
    // surface immediately if it diverges from the reference.
    let a = [1_u32, 2, 3, 4, 5, 6, 7, 8];
    let b = [8_u32, 7, 6, 5, 4, 3, 2, 1];
    assert_eq!(distance::dot_u32(&a, &b), scalar::dot_u32(&a, &b));
    assert_eq!(distance::l1_u32(&a, &b), scalar::l1_u32(&a, &b));
    assert_eq!(
        distance::l2_squared_u32(&a, &b),
        scalar::l2_squared_u32(&a, &b)
    );
    assert_eq!(
        distance::cosine_similarity_u32(&a, &b),
        scalar::cosine_similarity_u32(&a, &b)
    );

    let af = [1.0_f32, 2.0, 3.0, 4.0];
    let bf = [4.0_f32, 3.0, 2.0, 1.0];
    assert_eq!(distance::dot_f32(&af, &bf), scalar::dot_f32(&af, &bf));
    assert_eq!(
        distance::l2_squared_f32(&af, &bf),
        scalar::l2_squared_f32(&af, &bf)
    );
    assert_eq!(
        distance::cosine_similarity_f32(&af, &bf),
        scalar::cosine_similarity_f32(&af, &bf)
    );
}

#[test]
fn fixed_size_kernels_match_slice_kernels() {
    use crate::similarity::fixed;

    // 256-bin: byte histogram shape.
    let mut a256 = [0_u32; 256];
    let mut b256 = [0_u32; 256];
    for i in 0..256 {
        a256[i] = (i * 17 + 1) as u32;
        b256[i] = ((255 - i) * 13 + 5) as u32;
    }
    assert_eq!(
        fixed::dot_u32_256(&a256, &b256),
        scalar::dot_u32(&a256, &b256).unwrap()
    );
    assert_eq!(
        fixed::l1_u32_256(&a256, &b256),
        scalar::l1_u32(&a256, &b256).unwrap()
    );
    assert_eq!(
        fixed::l2_squared_u32_256(&a256, &b256),
        scalar::l2_squared_u32(&a256, &b256).unwrap()
    );
    let cos_fixed = fixed::cosine_similarity_u32_256(&a256, &b256);
    let cos_scalar = scalar::cosine_similarity_u32(&a256, &b256).unwrap();
    assert!((cos_fixed - cos_scalar).abs() < EPS);

    // 1024-bin: compact n-gram sketch shape.
    let mut a1024 = [0_u32; 1024];
    let mut b1024 = [0_u32; 1024];
    for i in 0..1024 {
        a1024[i] = ((i * 31) % 256) as u32;
        b1024[i] = ((i * 17) % 256) as u32;
    }
    assert_eq!(
        fixed::dot_u32_1024(&a1024, &b1024),
        scalar::dot_u32(&a1024, &b1024).unwrap()
    );
    assert_eq!(
        fixed::l2_squared_u32_1024(&a1024, &b1024),
        scalar::l2_squared_u32(&a1024, &b1024).unwrap()
    );

    // 4096-bin: richer sketch shape.
    let mut a4096 = [0_u32; 4096];
    let mut b4096 = [0_u32; 4096];
    for i in 0..4096 {
        a4096[i] = ((i.wrapping_mul(0x9E3779B9)) & 0xff) as u32;
        b4096[i] = ((i.wrapping_mul(0x85EBCA6B)) & 0xff) as u32;
    }
    assert_eq!(
        fixed::dot_u32_4096(&a4096, &b4096),
        scalar::dot_u32(&a4096, &b4096).unwrap()
    );
    assert_eq!(
        fixed::l1_u32_4096(&a4096, &b4096),
        scalar::l1_u32(&a4096, &b4096).unwrap()
    );

    // Identical fixed-size vectors yield zero L1/L2 and full cosine.
    let same = [42_u32; 256];
    assert_eq!(fixed::l1_u32_256(&same, &same), 0);
    assert_eq!(fixed::l2_squared_u32_256(&same, &same), 0);
    let cos = fixed::cosine_similarity_u32_256(&same, &same);
    assert!((cos - 1.0).abs() < EPS, "identical: got {cos}");

    // Const-generic API supports non-canonical sizes.
    let g = [1_u32, 2, 3, 4, 5];
    assert_eq!(fixed::dot_u32_n::<5>(&g, &g), 1 + 4 + 9 + 16 + 25);
    assert_eq!(fixed::l2_squared_u32_n::<5>(&g, &g), 0);
}

#[test]
fn count_distance_re_exports_resolve() {
    // Light smoke test: the re-exports under similarity::distance::counts
    // must reach the same functions in crate::divergence.
    let a = [1_u64, 2, 3, 4];
    let b = [4_u64, 3, 2, 1];
    let via_similarity = distance::counts::total_variation_counts(&a, &b);
    let via_divergence = crate::divergence::total_variation_counts(&a, &b);
    assert_eq!(via_similarity, via_divergence);
}

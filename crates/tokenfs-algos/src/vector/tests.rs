//! Vector module unit tests.
//!
//! Cross-references the SIMD parity tests in `tests/avx2_parity.rs` /
//! `tests/neon_parity.rs` (cross-process, on real hardware) — this
//! file covers the scalar oracle and the public dispatcher.

#![allow(clippy::unwrap_used)] // Test code — panic on None is the desired failure mode.

use super::distance;
use super::kernels::scalar;

const EPS: f64 = 1e-9;
const EPS_F32: f32 = 1e-6;

// ---------- scalar oracle: known-value tests ----------

#[test]
fn scalar_dot_u32_known_values() {
    let a = [1_u32, 2, 3, 4];
    let b = [4_u32, 3, 2, 1];
    assert_eq!(scalar::dot_u32(&a, &b), Some(20));
    assert_eq!(scalar::dot_u32(&[], &[]), Some(0));
    assert_eq!(scalar::dot_u32(&[1, 2], &[1]), None);
}

#[test]
fn scalar_l2_squared_u32_known_values() {
    let a = [1_u32, 2, 3];
    let b = [4_u32, 0, 7];
    // (3)^2 + (2)^2 + (4)^2 = 9 + 4 + 16
    assert_eq!(scalar::l2_squared_u32(&a, &b), Some(29));
    let c = [7_u32; 8];
    assert_eq!(scalar::l2_squared_u32(&c, &c), Some(0), "identical → 0");
}

#[test]
fn scalar_cosine_similarity_u32_known_values() {
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
fn scalar_dot_f32_known_values() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let d = scalar::dot_f32(&a, &b).unwrap();
    assert!((d - 32.0).abs() < EPS_F32);
    assert_eq!(scalar::dot_f32(&[], &[]), Some(0.0));
    assert_eq!(scalar::dot_f32(&[1.0], &[1.0, 2.0]), None);
}

#[test]
fn scalar_l2_squared_f32_known_values() {
    let a = [1.0_f32, 0.0, 0.0];
    let b = [0.0_f32, 1.0, 0.0];
    let d = scalar::l2_squared_f32(&a, &b).unwrap();
    assert!((d - 2.0).abs() < EPS_F32);
    let c = [2.5_f32; 5];
    assert_eq!(scalar::l2_squared_f32(&c, &c), Some(0.0));
}

#[test]
fn scalar_cosine_similarity_f32_known_values() {
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

// ---------- hamming / jaccard scalar tests ----------

#[test]
fn scalar_hamming_u64_known_values() {
    // identical → 0.
    assert_eq!(scalar::hamming_u64(&[0xff_u64; 4], &[0xff_u64; 4]), Some(0));
    // disjoint → 64 * len.
    assert_eq!(scalar::hamming_u64(&[u64::MAX; 4], &[0_u64; 4]), Some(256));
    // mixed.
    let a = [0b1011_u64, 0b0110_u64];
    let b = [0b0110_u64, 0b1010_u64];
    // 1011 xor 0110 = 1101 → 3 bits. 0110 xor 1010 = 1100 → 2 bits.
    assert_eq!(scalar::hamming_u64(&a, &b), Some(5));
    // length mismatch.
    assert_eq!(scalar::hamming_u64(&[0_u64], &[0_u64; 2]), None);
    // empty.
    assert_eq!(scalar::hamming_u64(&[], &[]), Some(0));
}

#[test]
fn scalar_jaccard_u64_known_values() {
    // identical non-empty → 1.0.
    let a = [0xff_u64; 4];
    let s = scalar::jaccard_u64(&a, &a).unwrap();
    assert!((s - 1.0).abs() < EPS, "identical: got {s}");
    // disjoint → 0.0.
    let b = [0_u64; 4];
    let c = [u64::MAX; 4];
    let s = scalar::jaccard_u64(&b, &c).unwrap();
    assert!(s.abs() < EPS, "disjoint: got {s}");
    // both empty → 1.0 by convention.
    let s = scalar::jaccard_u64(&b, &b).unwrap();
    assert!((s - 1.0).abs() < EPS, "both empty: got {s}");
    // mixed: a=1011, b=0110. AND = 0010 (1 bit). OR = 1111 (4 bits). 1/4.
    let a = [0b1011_u64];
    let b = [0b0110_u64];
    let s = scalar::jaccard_u64(&a, &b).unwrap();
    assert!((s - 0.25).abs() < EPS, "mixed: got {s}");
    // length mismatch.
    assert_eq!(scalar::jaccard_u64(&[0_u64], &[0_u64; 2]), None);
}

// ---------- dispatcher parity vs scalar (small inputs) ----------

#[test]
fn dispatcher_paths_match_scalar_today() {
    let a = [1_u32, 2, 3, 4, 5, 6, 7, 8];
    let b = [8_u32, 7, 6, 5, 4, 3, 2, 1];
    assert_eq!(distance::dot_u32(&a, &b), scalar::dot_u32(&a, &b).unwrap());
    assert_eq!(
        distance::l2_squared_u32(&a, &b),
        scalar::l2_squared_u32(&a, &b).unwrap()
    );
    assert_eq!(
        distance::cosine_similarity_u32(&a, &b),
        scalar::cosine_similarity_u32(&a, &b)
    );
    assert_eq!(distance::try_dot_u32(&a, &b), scalar::try_dot_u32(&a, &b));

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

    let au = [0xff_u64, 0x33];
    let bu = [0x0f_u64, 0xf0];
    assert_eq!(
        distance::hamming_u64(&au, &bu),
        scalar::hamming_u64(&au, &bu)
    );
    assert_eq!(
        distance::jaccard_u64(&au, &bu),
        scalar::jaccard_u64(&au, &bu)
    );
}

// ---------- length-mismatch behavior of public APIs ----------

#[test]
fn dispatcher_length_mismatch_returns_none_or_zero() {
    let a = [1_u32, 2, 3];
    let b = [1_u32, 2];
    assert_eq!(distance::dot_u32(&a, &b), 0);
    assert_eq!(distance::l2_squared_u32(&a, &b), 0);
    assert_eq!(distance::cosine_similarity_u32(&a, &b), None);
    assert_eq!(distance::try_dot_u32(&a, &b), None);

    let af = [1.0_f32, 2.0, 3.0];
    let bf = [1.0_f32];
    assert_eq!(distance::dot_f32(&af, &bf), None);
    assert_eq!(distance::l2_squared_f32(&af, &bf), None);
    assert_eq!(distance::cosine_similarity_f32(&af, &bf), None);

    let au = [0_u64, 1];
    let bu = [0_u64];
    assert_eq!(distance::hamming_u64(&au, &bu), None);
    assert_eq!(distance::jaccard_u64(&au, &bu), None);
}

// ---------- length-sweep parity (auto vs scalar) ----------

#[test]
fn dispatcher_matches_scalar_at_sub_block_lengths() {
    // Cover lengths around every plausible SIMD block boundary:
    // 4 (NEON 4xf32), 8 (AVX2 8xf32), 16 (AVX-512 16xf32), and tails.
    for len in [
        0_usize, 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
        257, 1023, 1024,
    ] {
        let a: Vec<u32> = (0..len)
            .map(|i| (i.wrapping_mul(17) ^ (i >> 2).wrapping_mul(31)) as u32)
            .collect();
        let b: Vec<u32> = (0..len).map(|i| ((i * 31) % 256) as u32).collect();
        assert_eq!(
            distance::dot_u32(&a, &b),
            scalar::dot_u32(&a, &b).unwrap(),
            "dot_u32 dispatcher diverged at len {len}"
        );
        assert_eq!(
            distance::l2_squared_u32(&a, &b),
            scalar::l2_squared_u32(&a, &b).unwrap(),
            "l2_squared_u32 dispatcher diverged at len {len}"
        );
    }
}

#[test]
fn dispatcher_hamming_jaccard_matches_scalar_at_sub_block_lengths() {
    for len in [
        0_usize, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256,
        257, 1023, 1024, 2049,
    ] {
        let mut state = 0xF22_C0FFEE_u64 ^ (len as u64);
        let a: Vec<u64> = (0..len)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                state.wrapping_mul(0x2545_f491_4f6c_dd1d)
            })
            .collect();
        let mut state2 = 0x5151_5eed_u64 ^ (len as u64);
        let b: Vec<u64> = (0..len)
            .map(|_| {
                state2 ^= state2 >> 12;
                state2 ^= state2 << 25;
                state2 ^= state2 >> 27;
                state2.wrapping_mul(0x2545_f491_4f6c_dd1d)
            })
            .collect();
        assert_eq!(
            distance::hamming_u64(&a, &b),
            scalar::hamming_u64(&a, &b),
            "hamming_u64 dispatcher diverged at len {len}"
        );
        let dispatched = distance::jaccard_u64(&a, &b);
        let oracle = scalar::jaccard_u64(&a, &b);
        match (dispatched, oracle) {
            (Some(d), Some(o)) => assert!(
                (d - o).abs() < 1e-12,
                "jaccard_u64 dispatcher diverged at len {len}: got {d}, want {o}"
            ),
            (None, None) => {}
            other => panic!("jaccard_u64 None/Some mismatch at len {len}: {other:?}"),
        }
    }
}

// ---------- batched many-vs-one parity ----------

#[test]
fn batched_dot_f32_one_to_many_matches_serial() {
    let stride = 16_usize;
    let n_rows = 7_usize;
    let query: Vec<f32> = (0..stride).map(|i| (i as f32) / 16.0).collect();
    let mut state = 0x1234_5678_u32;
    let db: Vec<f32> = (0..n_rows * stride)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    let mut out_batched = vec![0.0_f32; n_rows];
    super::batch::dot_f32_one_to_many(&query, &db, stride, &mut out_batched);

    for (i, expected_slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let serial = distance::dot_f32(&query, row).unwrap();
        assert!(
            (expected_slot - serial).abs() < EPS_F32 * 100.0,
            "batched dot_f32 row {i} diverged: got {expected_slot}, serial {serial}"
        );
    }
}

#[test]
fn batched_l2_squared_f32_one_to_many_matches_serial() {
    let stride = 32_usize;
    let n_rows = 5_usize;
    let query: Vec<f32> = (0..stride).map(|i| (i as f32).sin()).collect();
    let db: Vec<f32> = (0..n_rows * stride)
        .map(|i| ((i % 13) as f32) * 0.1)
        .collect();

    let mut out_batched = vec![0.0_f32; n_rows];
    super::batch::l2_squared_f32_one_to_many(&query, &db, stride, &mut out_batched);

    for (i, expected_slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let serial = distance::l2_squared_f32(&query, row).unwrap();
        assert!(
            (expected_slot - serial).abs() < EPS_F32 * 100.0,
            "batched l2_squared_f32 row {i} diverged: got {expected_slot}, serial {serial}"
        );
    }
}

#[test]
fn batched_cosine_similarity_f32_one_to_many_matches_serial() {
    let stride = 64_usize;
    let n_rows = 4_usize;
    let query: Vec<f32> = (0..stride).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let db: Vec<f32> = (0..n_rows * stride).map(|i| (i as f32) * 0.001).collect();

    let mut out_batched = vec![0.0_f32; n_rows];
    super::batch::cosine_similarity_f32_one_to_many(&query, &db, stride, &mut out_batched);

    for (i, expected_slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let serial = distance::cosine_similarity_f32(&query, row).unwrap();
        // Loose tolerance: the batched form uses a precomputed query
        // norm, which may diverge from the serial form by ULP-level
        // rounding.
        assert!(
            (expected_slot - serial).abs() < 1e-4,
            "batched cosine row {i} diverged: got {expected_slot}, serial {serial}"
        );
    }
}

#[test]
fn batched_cosine_similarity_f32_zero_query_returns_zero() {
    let stride = 8_usize;
    let n_rows = 3_usize;
    let query = vec![0.0_f32; stride];
    let db: Vec<f32> = (0..n_rows * stride).map(|i| i as f32).collect();
    let mut out = vec![1.0_f32; n_rows];
    super::batch::cosine_similarity_f32_one_to_many(&query, &db, stride, &mut out);
    for slot in out {
        assert_eq!(slot, 0.0);
    }
}

#[test]
fn batched_hamming_u64_one_to_many_matches_serial() {
    let stride = 8_usize; // 512-bit signature.
    let n_rows = 9_usize;
    let mut state = 0xF22_C0FFEE_u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let query: Vec<u64> = (0..stride).map(|_| next()).collect();
    let db: Vec<u64> = (0..n_rows * stride).map(|_| next()).collect();

    let mut out_batched = vec![0_u32; n_rows];
    super::batch::hamming_u64_one_to_many(&query, &db, stride, &mut out_batched);

    for (i, slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let serial = distance::hamming_u64(&query, row).unwrap();
        assert_eq!(*slot as u64, serial, "batched hamming row {i} diverged");
    }
}

#[test]
fn batched_jaccard_u64_one_to_many_matches_serial() {
    let stride = 4_usize; // 256-bit signature (typical MinHash).
    let n_rows = 11_usize;
    let mut state = 0x5151_5eed_u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let query: Vec<u64> = (0..stride).map(|_| next()).collect();
    let db: Vec<u64> = (0..n_rows * stride).map(|_| next()).collect();

    let mut out_batched = vec![0_f64; n_rows];
    super::batch::jaccard_u64_one_to_many(&query, &db, stride, &mut out_batched);

    for (i, slot) in out_batched.iter().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        let serial = distance::jaccard_u64(&query, row).unwrap();
        assert!(
            (slot - serial).abs() < 1e-12,
            "batched jaccard row {i} diverged: got {slot}, serial {serial}"
        );
    }
}

#[test]
fn batched_apis_handle_empty_db() {
    let stride = 16_usize;
    let query = vec![0.0_f32; stride];
    let db: Vec<f32> = Vec::new();
    let mut out: Vec<f32> = Vec::new();
    super::batch::dot_f32_one_to_many(&query, &db, stride, &mut out);
    super::batch::l2_squared_f32_one_to_many(&query, &db, stride, &mut out);
    super::batch::cosine_similarity_f32_one_to_many(&query, &db, stride, &mut out);

    let q64 = vec![0_u64; stride];
    let db64: Vec<u64> = Vec::new();
    let mut out_u32: Vec<u32> = Vec::new();
    let mut out_f64: Vec<f64> = Vec::new();
    super::batch::hamming_u64_one_to_many(&q64, &db64, stride, &mut out_u32);
    super::batch::jaccard_u64_one_to_many(&q64, &db64, stride, &mut out_f64);
}

#[test]
#[should_panic(expected = "stride must be > 0")]
fn batched_panics_on_zero_stride() {
    let query = vec![0.0_f32; 4];
    let db = vec![0.0_f32; 4];
    let mut out = vec![0.0_f32; 1];
    super::batch::dot_f32_one_to_many(&query, &db, 0, &mut out);
}

#[test]
#[should_panic(expected = "not a multiple of stride")]
fn batched_panics_on_non_multiple_db_length() {
    let query = vec![0.0_f32; 4];
    let db = vec![0.0_f32; 5]; // not a multiple of 4
    let mut out = vec![0.0_f32; 1];
    super::batch::dot_f32_one_to_many(&query, &db, 4, &mut out);
}

#[test]
#[should_panic(expected = "but db has")]
fn batched_panics_on_out_length_mismatch() {
    let query = vec![0.0_f32; 4];
    let db = vec![0.0_f32; 12]; // 3 rows
    let mut out = vec![0.0_f32; 2]; // wrong
    super::batch::dot_f32_one_to_many(&query, &db, 4, &mut out);
}

// ---------- AVX-512 backend tests (gated; skip if unavailable) ----------

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn avx512_f32_kernels_match_scalar_when_available() {
    use super::kernels::avx512;
    if !avx512::is_available() {
        eprintln!("avx512 unavailable on this host; skipping inline AVX-512 parity test");
        return;
    }
    let n = 1024_usize;
    let mut state = 0x9E37_79B9_u32;
    let a: Vec<f32> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let b: Vec<f32> = a.iter().map(|&x| x * 0.5 + 0.25).collect();

    let dot_s = scalar::dot_f32(&a, &b).unwrap();
    // SAFETY: availability checked above.
    let dot_v = unsafe { avx512::dot_f32(&a, &b) };
    let l1: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| (x * y).abs()).sum();
    let scale = l1.max(dot_s.abs()).max(dot_v.abs()).max(1.0);
    assert!(
        (dot_s - dot_v).abs() / scale < 1e-3,
        "avx512 dot_f32 diverged: scalar={dot_s} avx512={dot_v}"
    );

    let l2_s = scalar::l2_squared_f32(&a, &b).unwrap();
    // SAFETY: availability checked above.
    let l2_v = unsafe { avx512::l2_squared_f32(&a, &b) };
    let scale = l2_s.abs().max(l2_v.abs()).max(1.0);
    assert!(
        (l2_s - l2_v).abs() / scale < 5e-4,
        "avx512 l2_squared_f32 diverged: scalar={l2_s} avx512={l2_v}"
    );

    let cos_s = scalar::cosine_similarity_f32(&a, &b).unwrap();
    // SAFETY: availability checked above.
    let cos_v = unsafe { avx512::cosine_similarity_f32(&a, &b) };
    assert!(
        (cos_s - cos_v).abs() < 1e-4,
        "avx512 cosine_similarity_f32 diverged: scalar={cos_s} avx512={cos_v}"
    );
}

#[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
#[test]
fn avx512_hamming_jaccard_match_scalar_when_available() {
    use super::kernels::avx512;
    if !avx512::is_popcnt_available() {
        eprintln!("avx512vpopcntdq unavailable on this host; skipping AVX-512 popcount parity");
        return;
    }
    let n = 256_usize;
    let mut state = 0xA1A1_B2B2_C3C3_D4D4_u64;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let a: Vec<u64> = (0..n).map(|_| next()).collect();
    let b: Vec<u64> = (0..n).map(|_| next()).collect();

    let h_s = scalar::hamming_u64(&a, &b).unwrap();
    // SAFETY: availability checked above.
    let h_v = unsafe { avx512::hamming_u64(&a, &b) };
    assert_eq!(h_s, h_v);

    let j_s = scalar::jaccard_u64(&a, &b).unwrap();
    // SAFETY: availability checked above.
    let j_v = unsafe { avx512::jaccard_u64(&a, &b) };
    assert!(
        (j_s - j_v).abs() < 1e-12,
        "avx512 jaccard diverged: scalar={j_s} avx512={j_v}"
    );
}

// ---------- Specific edge-case parity (length 1, length not a multiple of SIMD width) ----------

#[test]
fn dispatcher_handles_length_1() {
    let a = [3.0_f32];
    let b = [4.0_f32];
    assert_eq!(distance::dot_f32(&a, &b), Some(12.0));
    assert_eq!(distance::l2_squared_f32(&a, &b), Some(1.0));
    let c = [42_u32];
    let d = [42_u32];
    assert_eq!(distance::dot_u32(&c, &d), 42 * 42);
    assert_eq!(distance::l2_squared_u32(&c, &d), 0);
    let e = [0xff_u64];
    let f = [0x0f_u64];
    assert_eq!(distance::hamming_u64(&e, &f), Some(4));
}

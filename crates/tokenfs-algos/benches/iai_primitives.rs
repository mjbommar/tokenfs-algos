//! `bench-iai-primitives`: deterministic hardware-counter benches.
//!
//! Run via `cargo bench --bench iai_primitives` (requires `valgrind`
//! installed). Reports instruction count, branch misses, L1/L2/LL
//! cache references per benchmark — deterministic across runs on the
//! same binary.
//!
//! Used by `.github/workflows/iai-bench.yml` to catch instruction-count
//! regressions at ~1% sensitivity, complementing the noisier wall-clock
//! criterion benches that drive `bench-regression.yml` (audit-R10 T3.4).
//!
//! ## Scope
//!
//! Focused subset of hot primitives:
//!   * `bits::popcount_u64_slice` — canonical SIMD-unrolled tight loop.
//!   * `bits::try_streamvbyte_encode_u32` / `try_streamvbyte_decode_u32`
//!     — codec hot paths.
//!   * `hash::sha256::try_sha256` — hash hot path.
//!   * `vector::dot_f32` / `vector::l2_squared_f32` — SIMD-heavy.
//!
//! Each input size is fixed so the instruction count is deterministic;
//! we don't sweep across cache tiers (criterion already does that).

#![allow(missing_docs)]

use std::hint::black_box;

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use tokenfs_algos::{bits, hash, vector};

// ----- input fixtures (deterministic, large enough to be measurable) -----

const POPCOUNT_WORDS: usize = 4 * 1024; // 32 KiB — comfortably L1-resident
const STREAMVBYTE_VALUES: usize = 1024;
const SHA256_BYTES: usize = 4 * 1024; // hot kernel size
const VECTOR_LEN: usize = 1024;

fn deterministic_words(n: usize, seed: u64) -> Vec<u64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        })
        .collect()
}

fn deterministic_u32s(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32
        })
        .collect()
}

fn deterministic_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 56) as u8
        })
        .collect()
}

fn deterministic_f32s(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            // Scale to [-1, 1] so the dot product / L2 stays finite for
            // any plausible vector length.
            let bits = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32) as u32;
            (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

// ----- popcount -----

#[library_benchmark]
fn iai_popcount_u64_slice() -> u64 {
    let words = deterministic_words(POPCOUNT_WORDS, 0xF22_C0FFEE);
    bits::popcount_u64_slice(black_box(&words))
}

#[library_benchmark]
fn iai_popcount_u64_slice_scalar() -> u64 {
    let words = deterministic_words(POPCOUNT_WORDS, 0xF22_C0FFEE);
    bits::kernels::scalar::popcount_u64_slice(black_box(&words))
}

// ----- streamvbyte -----

#[library_benchmark]
fn iai_streamvbyte_encode() -> usize {
    let values = deterministic_u32s(STREAMVBYTE_VALUES, 0xDEAD_BEEF);
    let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(STREAMVBYTE_VALUES)];
    let mut data = vec![0_u8; bits::streamvbyte_data_max_len(STREAMVBYTE_VALUES)];
    bits::try_streamvbyte_encode_u32(black_box(&values), &mut ctrl, &mut data)
        .expect("encode within sized buffers")
}

#[library_benchmark]
fn iai_streamvbyte_decode() -> usize {
    // Pre-encode so the bench measures decode only.
    let values = deterministic_u32s(STREAMVBYTE_VALUES, 0xDEAD_BEEF);
    let mut ctrl = vec![0_u8; bits::streamvbyte_control_len(STREAMVBYTE_VALUES)];
    let mut data = vec![0_u8; bits::streamvbyte_data_max_len(STREAMVBYTE_VALUES)];
    let written = bits::try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data)
        .expect("encode within sized buffers");

    let mut decoded = vec![0_u32; STREAMVBYTE_VALUES];
    bits::try_streamvbyte_decode_u32(
        black_box(&ctrl),
        black_box(&data[..written]),
        STREAMVBYTE_VALUES,
        &mut decoded,
    )
    .expect("decode within sized buffer")
}

// ----- sha256 -----

#[library_benchmark]
fn iai_sha256() -> [u8; 32] {
    let bytes = deterministic_bytes(SHA256_BYTES, 0xCAFE_BABE);
    hash::sha256::try_sha256(black_box(&bytes)).expect("sha256 input within bit-length cap")
}

// ----- vector -----

#[library_benchmark]
fn iai_vector_dot_f32() -> Option<f32> {
    let a = deterministic_f32s(VECTOR_LEN, 0xA1B2_C3D4);
    let b = deterministic_f32s(VECTOR_LEN, 0x5566_7788);
    vector::dot_f32(black_box(&a), black_box(&b))
}

#[library_benchmark]
fn iai_vector_l2_squared_f32() -> Option<f32> {
    let a = deterministic_f32s(VECTOR_LEN, 0xA1B2_C3D4);
    let b = deterministic_f32s(VECTOR_LEN, 0x5566_7788);
    vector::l2_squared_f32(black_box(&a), black_box(&b))
}

// ----- HNSW distance kernels (Phase 2.4 iai-bench rows) -----
//
// Per-(metric, scalar_kind, backend) instruction-count rows for the
// HNSW i8/u8 kernels. Fixed dim=128 (typical embedding chunk) so
// counts are deterministic and comparable across runs. Each kernel
// has a `_scalar` companion so the iai regression gate compares
// apples-to-apples within a backend. Phase 2.5 will replace the
// SIMD bodies (currently scalar fallbacks) with true vectorized
// impls; the gate will then catch any IR regression on the SIMD path.

const HNSW_DIM: usize = 128;

#[library_benchmark]
fn iai_hnsw_dot_i8_auto() -> u32 {
    let a: Vec<i8> = (0..HNSW_DIM).map(|i| ((i as i32) - 64) as i8).collect();
    let b: Vec<i8> = (0..HNSW_DIM)
        .map(|i| ((i as i32 * 7) % 251 - 125) as i8)
        .collect();
    tokenfs_algos::similarity::hnsw::kernels::auto::distance_dot_i8(black_box(&a), black_box(&b))
        .expect("equal lengths")
}

#[library_benchmark]
fn iai_hnsw_dot_i8_scalar() -> u32 {
    let a: Vec<i8> = (0..HNSW_DIM).map(|i| ((i as i32) - 64) as i8).collect();
    let b: Vec<i8> = (0..HNSW_DIM)
        .map(|i| ((i as i32 * 7) % 251 - 125) as i8)
        .collect();
    tokenfs_algos::similarity::hnsw::kernels::scalar::try_dot_i8(black_box(&a), black_box(&b))
        .expect("equal lengths")
}

#[library_benchmark]
fn iai_hnsw_l2_squared_i8_auto() -> u32 {
    let a: Vec<i8> = (0..HNSW_DIM).map(|i| ((i as i32) - 64) as i8).collect();
    let b: Vec<i8> = (0..HNSW_DIM)
        .map(|i| ((i as i32 * 7) % 251 - 125) as i8)
        .collect();
    tokenfs_algos::similarity::hnsw::kernels::auto::distance_l2_squared_i8(
        black_box(&a),
        black_box(&b),
    )
    .expect("equal lengths")
}

#[library_benchmark]
fn iai_hnsw_l2_squared_u8_auto() -> u32 {
    let a: Vec<u8> = (0..HNSW_DIM).map(|i| (i & 0xFF) as u8).collect();
    let b: Vec<u8> = (0..HNSW_DIM).map(|i| ((i * 7) & 0xFF) as u8).collect();
    tokenfs_algos::similarity::hnsw::kernels::auto::distance_l2_squared_u8(
        black_box(&a),
        black_box(&b),
    )
    .expect("equal lengths")
}

// ----- harness -----

library_benchmark_group!(
    name = bits_group;
    benchmarks = iai_popcount_u64_slice, iai_popcount_u64_slice_scalar,
                 iai_streamvbyte_encode, iai_streamvbyte_decode
);

library_benchmark_group!(
    name = hash_group;
    benchmarks = iai_sha256
);

library_benchmark_group!(
    name = vector_group;
    benchmarks = iai_vector_dot_f32, iai_vector_l2_squared_f32
);

library_benchmark_group!(
    name = hnsw_group;
    benchmarks = iai_hnsw_dot_i8_auto, iai_hnsw_dot_i8_scalar,
                 iai_hnsw_l2_squared_i8_auto, iai_hnsw_l2_squared_u8_auto
);

main!(
    library_benchmark_groups = bits_group,
    hash_group,
    vector_group,
    hnsw_group
);

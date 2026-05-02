#![no_std]
#![allow(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

use tokenfs_algos::{bits, hash, permutation, vector};

/// Smoke: each kernel-claimed-safe primitive compiles and links under
/// `no_std + alloc` with the documented kernel-safe feature set.
pub fn smoke() {
    let words: [u64; 4] = [0xDEAD_BEEF_0000_0000, 0, 0xFFFF_FFFF, 0xCAFE];
    let _popcount = bits::popcount_u64_slice(&words);

    let bytes = [0_u8; 64];
    let mut digest = [[0_u8; 32]; 1];
    let messages: [&[u8]; 1] = [&bytes];
    hash::sha256_batch_st(&messages, &mut digest);

    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let _dot = vector::dot_f32(&a, &b);

    let haystack: [u32; 4] = [10, 20, 30, 40];
    let _found = hash::contains_u32_simd(&haystack, 20);

    // Permutation: apply is kernel-safe (construction is build-time-only,
    // so we only verify the apply path via a hand-built permutation).
    let perm = permutation::Permutation::identity(4);
    let src: [u32; 4] = [100, 200, 300, 400];
    let mut dst: Vec<u32> = alloc::vec![0; 4];
    perm.apply_into(&src, &mut dst);
}

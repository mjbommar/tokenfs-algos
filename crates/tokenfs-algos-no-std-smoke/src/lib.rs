#![no_std]
#![allow(missing_docs)]

extern crate alloc;

use alloc::vec::Vec;

use tokenfs_algos::{bits, hash, permutation, vector};

/// Smoke: each kernel-claimed-safe primitive compiles and links under
/// `no_std + alloc` with the documented kernel-safe feature set
/// (audit-R10 #10 / T2.5).
///
/// Anything reachable from this function MUST stay available without
/// `userspace`, `panicking-shape-apis`, `arch-pinned-kernels`, `std`,
/// or `parallel`. Adding a new primitive that the audit narrative
/// describes as "kernel-safe-by-default" should also add it here.
pub fn smoke() {
    smoke_bits();
    smoke_hash();
    smoke_vector();
    smoke_permutation();
    smoke_search_membership();
}

fn smoke_bits() {
    let words: [u64; 4] = [0xDEAD_BEEF_0000_0000, 0, 0xFFFF_FFFF, 0xCAFE];
    let _popcount = bits::popcount_u64_slice(&words);

    // streamvbyte fallible roundtrip — kernel-safe surface (audit-R6 #162).
    let values: [u32; 4] = [1, 256, 65_536, u32::MAX];
    let mut ctrl = [0_u8; 1];
    let mut data = [0_u8; bits::streamvbyte_data_max_len(4)];
    let written =
        bits::try_streamvbyte_encode_u32(&values, &mut ctrl, &mut data).expect("smoke encode");
    let mut decoded = [0_u32; 4];
    let _read =
        bits::try_streamvbyte_decode_u32(&ctrl, &data[..written], values.len(), &mut decoded)
            .expect("smoke decode");
}

fn smoke_hash() {
    // Audit-R5 #157: kernel/FUSE consumers compile without
    // `panicking-shape-apis`, so only the fallible `try_*` SHA-256
    // batched entry is reachable. This smoke also pins that the
    // fallible variant satisfies the kernel-safe surface.
    let bytes = [0_u8; 64];
    let mut digest = [[0_u8; 32]; 1];
    let messages: [&[u8]; 1] = [&bytes];
    hash::try_sha256_batch_st(&messages, &mut digest)
        .expect("smoke: messages.len() matches digest.len()");

    // Audit-R10 #4: try_sha256 + per-backend dispatcher must be
    // reachable from `no_std + alloc` without `userspace`.
    let _digest = hash::sha256::try_sha256(&bytes).expect("smoke try_sha256");
}

fn smoke_vector() {
    let a = [1.0_f32, 2.0, 3.0];
    let b = [4.0_f32, 5.0, 6.0];
    let _dot = vector::dot_f32(&a, &b);
    let _l2 = vector::l2_squared_f32(&a, &b);
}

fn smoke_permutation() {
    // Permutation: kernel-safe apply via `try_apply_into`. The
    // panicking `Permutation::identity` is already in the
    // panic-surface allowlist and tracked in #216 for cleanup;
    // exercise `Permutation::try_from_vec` here so the smoke pins
    // the actual fallible constructor surface.
    let perm = permutation::Permutation::try_from_vec(alloc::vec![0_u32, 1, 2, 3])
        .expect("smoke try_from_vec");
    let src: [u32; 4] = [100, 200, 300, 400];
    let mut dst: Vec<u32> = alloc::vec![0; 4];
    perm.try_apply_into(&src, &mut dst)
        .expect("smoke try_apply_into");
}

fn smoke_search_membership() {
    // Audit-R10 #1: SIMD set-membership try_* — kernel-safe surface.
    let haystack: [u32; 4] = [10, 20, 30, 40];
    let _found = hash::contains_u32_simd(&haystack, 20);
    let needles: [u32; 2] = [20, 40];
    let mut out = [false; 2];
    hash::try_contains_u32_batch_simd(&haystack, &needles, &mut out).expect("smoke contains_batch");

    // PackedDfa try_new (audit-R10 #2): the kernel-safe constructor.
    let patterns: [&[u8]; 2] = [b"foo", b"bar"];
    let dfa =
        tokenfs_algos::search::packed_dfa::PackedDfa::try_new(&patterns).expect("smoke try_new");
    let _hits = dfa.find(b"the foo and bar").is_some();
}

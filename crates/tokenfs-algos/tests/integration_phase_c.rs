//! Phase C composition integration tests.
//!
//! The `examples/inverted_index.rs`, `examples/build_pipeline.rs`, and
//! `examples/similarity_scan.rs` programs each demonstrate a multi-primitive
//! composition end-to-end, but `cargo test` does not execute them. This file
//! lifts the load-bearing assertions from those examples into actual
//! integration tests so that `cargo test --workspace` catches composition
//! regressions across the bitmap, bits, hash, permutation, and vector
//! primitives.
//!
//! Each test uses fixed PRNG seeds and small fixture sizes (well under one
//! second per test) so the suite stays cheap on CI.
//!
//! Gated on `panicking-shape-apis` because the example compositions use the
//! ergonomic panicking entry points (`streamvbyte_encode_u32`, `sha256_batch_st`,
//! `l2_squared_f32_one_to_many`, `DynamicBitPacker::new`) which audit-R5 #157
//! moved behind that on-by-default feature.

#![allow(missing_docs)]
#![cfg(feature = "panicking-shape-apis")]

use tokenfs_algos::bitmap::{ArrayContainer, Container};
use tokenfs_algos::bits::{
    DynamicBitPacker, streamvbyte_control_len, streamvbyte_data_max_len, streamvbyte_decode_u32,
    streamvbyte_encode_u32,
};
use tokenfs_algos::hash::sha256::sha256;
use tokenfs_algos::hash::sha256_batch_st;
use tokenfs_algos::permutation::{CsrGraph, rcm};
use tokenfs_algos::vector::{kernels::scalar as vec_scalar, l2_squared_f32_one_to_many};

// =============================================================================
// Shared deterministic PRNG.
//
// Same xorshift64* shape used by the source examples; reproduced inline so
// each test stays dependency-free and the seeds in this file map directly
// to the ones in the example sources.
// =============================================================================

/// Tiny xorshift64* PRNG. State must be non-zero.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 {
            0x9E37_79B9_7F4A_7C15_u64
        } else {
            seed
        };
        Self { state }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

// =============================================================================
// Test 1 — inverted_index composition (bitmap::Container + bits::streamvbyte).
//
// Source: crates/tokenfs-algos/examples/inverted_index.rs
//
// Builds per-bigram posting lists from a synthetic corpus, encodes each list
// both as Stream-VByte (delta-coded) and as a Roaring `ArrayContainer`, and
// verifies that:
//   * Stream-VByte encode/decode round-trips bit-exactly.
//   * Cardinalities agree between the two representations.
//   * Pairwise intersection cardinality computed via the Roaring container
//     matches the answer derived from decoded Stream-VByte lists.
// =============================================================================

const T1_NUM_DOCS: u32 = 50;
const T1_TOKENS_PER_DOC: usize = 50;
const T1_VOCAB_SIZE: u32 = 64;
const T1_RNG_SEED: u64 = 0x0005_EEDC_0FFE_EF38;

#[test]
fn inverted_index_composition_roundtrips_and_agrees_on_intersection() {
    // ----- 1. Synthesize a tiny deterministic corpus. -----
    let mut rng = Xorshift64::new(T1_RNG_SEED);
    let mut corpus: Vec<Vec<u32>> = Vec::with_capacity(T1_NUM_DOCS as usize);
    for _ in 0..T1_NUM_DOCS {
        let mut doc = Vec::with_capacity(T1_TOKENS_PER_DOC);
        for _ in 0..T1_TOKENS_PER_DOC {
            // Cheap Zipf-ish skew so a handful of bigrams dominate; same
            // shape as draw_token in examples/inverted_index.rs.
            let u_bits = rng.next_u64() >> 40;
            let u = (u_bits as f64) / ((1_u64 << 24) as f64);
            let scaled = (u * u) * (T1_VOCAB_SIZE as f64);
            let id = (scaled as u32).min(T1_VOCAB_SIZE - 1);
            doc.push(id);
        }
        corpus.push(doc);
    }

    // ----- 2. Build per-bigram inverted index over 2-grams. -----
    use std::collections::BTreeSet;
    let table_len = (T1_VOCAB_SIZE * T1_VOCAB_SIZE) as usize;
    let mut sets: Vec<BTreeSet<u16>> = (0..table_len).map(|_| BTreeSet::new()).collect();
    for (doc_id, doc) in corpus.iter().enumerate() {
        let doc_id_u16 = u16::try_from(doc_id).expect("T1_NUM_DOCS fits in u16");
        for window in doc.windows(2) {
            assert!(window[0] < T1_VOCAB_SIZE && window[1] < T1_VOCAB_SIZE);
            let key = (window[0] * T1_VOCAB_SIZE + window[1]) as usize;
            sets[key].insert(doc_id_u16);
        }
    }
    let index: Vec<Option<Vec<u16>>> = sets
        .into_iter()
        .map(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.into_iter().collect())
            }
        })
        .collect();

    // ----- 3. Encode + decode every posting list with Stream-VByte and
    //         build a Roaring ArrayContainer; check round-trip + cardinality. -----
    let mut containers: Vec<Option<Container>> = Vec::with_capacity(index.len());
    let mut nonempty_keys: Vec<(u16, usize)> = Vec::new();

    for (key, slot) in index.iter().enumerate() {
        match slot {
            None => containers.push(None),
            Some(postings) => {
                // Delta-code, then encode with Stream-VByte.
                let mut deltas = Vec::with_capacity(postings.len());
                let mut prev: u32 = 0;
                for (i, &v) in postings.iter().enumerate() {
                    let v = u32::from(v);
                    deltas.push(if i == 0 { v } else { v - prev });
                    prev = v;
                }
                let n = deltas.len();
                let mut control = vec![0_u8; streamvbyte_control_len(n)];
                let mut data = vec![0_u8; streamvbyte_data_max_len(n)];
                let written = streamvbyte_encode_u32(&deltas, &mut control, &mut data);
                data.truncate(written);

                // Decode; re-accumulate to undo the delta coding.
                let mut decoded_deltas = vec![0_u32; n];
                streamvbyte_decode_u32(&control, &data, n, &mut decoded_deltas);
                let mut decoded: Vec<u16> = Vec::with_capacity(n);
                let mut acc: u32 = 0;
                for (i, &d) in decoded_deltas.iter().enumerate() {
                    acc = if i == 0 { d } else { acc + d };
                    decoded.push(u16::try_from(acc).expect("posting fits in u16"));
                }

                assert_eq!(
                    &decoded, postings,
                    "stream-vbyte round-trip diverged for bigram key {key}"
                );

                // Build the Roaring `ArrayContainer` from the same sorted ids.
                let arr = ArrayContainer::from_sorted(postings.clone());
                let card_arr = arr.cardinality();
                assert_eq!(
                    card_arr as usize,
                    decoded.len(),
                    "roaring cardinality disagrees with decoded vbyte for key {key}"
                );

                containers.push(Some(Container::Array(arr)));
                nonempty_keys.push((
                    u16::try_from(key).expect("vocab^2 fits in u16"),
                    postings.len(),
                ));
            }
        }
    }

    // ----- 4. Pick two specific bigrams (the two with the largest postings
    //         lists, deterministic with this seed) and intersect them two
    //         ways; the cardinalities must agree. -----
    nonempty_keys.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    assert!(
        nonempty_keys.len() >= 2,
        "fixture must produce at least two non-empty bigrams"
    );

    let key_a = nonempty_keys[0].0;
    let key_b = nonempty_keys[1].0;
    let raw_a = index[key_a as usize]
        .as_ref()
        .expect("hot bigram has postings");
    let raw_b = index[key_b as usize]
        .as_ref()
        .expect("hot bigram has postings");

    // Scalar intersection over the decoded sorted u16 lists.
    let mut scalar_intersection: Vec<u16> = Vec::new();
    let (mut i, mut j) = (0_usize, 0_usize);
    while i < raw_a.len() && j < raw_b.len() {
        match raw_a[i].cmp(&raw_b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                scalar_intersection.push(raw_a[i]);
                i += 1;
                j += 1;
            }
        }
    }

    let cont_a = containers[key_a as usize]
        .as_ref()
        .expect("hot bigram container present");
    let cont_b = containers[key_b as usize]
        .as_ref()
        .expect("hot bigram container present");
    let roaring_card = cont_a.intersect_cardinality(cont_b);

    assert_eq!(
        roaring_card as usize,
        scalar_intersection.len(),
        "Roaring intersect_cardinality disagrees with scalar intersection ({} vs {}) for keys 0x{:04x} & 0x{:04x}",
        roaring_card,
        scalar_intersection.len(),
        key_a,
        key_b
    );
}

// =============================================================================
// Test 2 — build_pipeline composition (sha256_batch_st + rcm + Permutation).
//
// Source: crates/tokenfs-algos/examples/build_pipeline.rs
//
// Hashes a batch of synthetic extents both via the batched API and the
// per-message API, asserts agreement, builds a tiny similarity CSR, runs RCM,
// and round-trips a metadata array through the resulting permutation and its
// inverse.
// =============================================================================

const T2_NUM_EXTENTS: usize = 100;
const T2_PAYLOAD_BYTES: usize = 256;
const T2_RNG_SEED: u64 = 0xF22C_2BAB_EDEA_DBEE;

#[test]
fn build_pipeline_composition_hash_batches_match_and_rcm_round_trips() {
    // ----- 1. Synthesize 100 deterministic 256-byte payloads. -----
    let mut rng = Xorshift64::new(T2_RNG_SEED);
    let total = T2_NUM_EXTENTS * T2_PAYLOAD_BYTES;
    let mut bytes = vec![0_u8; total];
    for byte in &mut bytes {
        *byte = (rng.next_u64() >> 56) as u8;
    }
    let mut extents: Vec<&[u8]> = Vec::with_capacity(T2_NUM_EXTENTS);
    for i in 0..T2_NUM_EXTENTS {
        let start = i * T2_PAYLOAD_BYTES;
        extents.push(&bytes[start..start + T2_PAYLOAD_BYTES]);
    }

    // ----- 2. Hash via batched API and per-message reference. -----
    let mut batched = vec![[0_u8; 32]; T2_NUM_EXTENTS];
    sha256_batch_st(&extents, &mut batched);
    for (i, extent) in extents.iter().enumerate() {
        let direct = sha256(extent);
        assert_eq!(
            batched[i], direct,
            "sha256_batch_st diverged from sha256 at extent {i}"
        );
    }

    // ----- 3. Build a tiny similarity CSR: extent i connected to extent
    //         i + 1 (a path graph). Symmetrise, so each interior vertex has
    //         degree 2 and the two endpoints have degree 1. -----
    let n = T2_NUM_EXTENTS;
    let mut offsets: Vec<u32> = Vec::with_capacity(n + 1);
    let mut neighbors: Vec<u32> = Vec::with_capacity(2 * (n - 1));
    let mut running: u32 = 0;
    offsets.push(0);
    for v in 0..n {
        if v > 0 {
            neighbors.push((v - 1) as u32);
            running += 1;
        }
        if v + 1 < n {
            neighbors.push((v + 1) as u32);
            running += 1;
        }
        offsets.push(running);
    }
    let graph = CsrGraph {
        n: n as u32,
        offsets: &offsets,
        neighbors: &neighbors,
    };

    // ----- 4. Apply RCM and verify the result is a valid permutation. -----
    let perm = rcm(graph);
    assert_eq!(perm.len(), n, "rcm produced wrong-length permutation");
    let mut seen = vec![false; n];
    for &new_id in perm.as_slice() {
        let id = new_id as usize;
        assert!(
            id < n,
            "rcm permutation entry out of range: got {id}, n = {n}"
        );
        assert!(
            !seen[id],
            "rcm permutation entry {id} appears more than once"
        );
        seen[id] = true;
    }
    assert!(
        seen.iter().all(|b| *b),
        "rcm permutation does not cover 0..n"
    );

    // ----- 5. Apply permutation to a metadata array via apply_into; round-
    //         trip via the inverse. -----
    let metadata_in: Vec<u32> = (0..n as u32).map(|i| i.wrapping_mul(0x9E37_79B9)).collect();
    let mut permuted = vec![0_u32; n];
    perm.apply_into(&metadata_in, &mut permuted);
    let inverse = perm.inverse();
    let mut recovered = vec![0_u32; n];
    inverse.apply_into(&permuted, &mut recovered);
    assert_eq!(
        recovered, metadata_in,
        "permutation inverse round-trip lost data"
    );
}

// =============================================================================
// Test 3 — similarity_scan composition (vector::*_one_to_many + bit_pack).
//
// Source: crates/tokenfs-algos/examples/similarity_scan.rs
//
// Bit-packs a synthetic fingerprint database at width 11, verifies the
// round-trip, then computes one-to-many L2 distances both via the batched
// dispatcher and via a serial scalar reference. The two outputs must agree
// within the documented Higham tolerance (1e-3 of the L1 norm of products).
// =============================================================================

const T3_DATABASE_SIZE: usize = 100;
const T3_FINGERPRINT_LANES: usize = 8;
const T3_PACK_WIDTH: u32 = 11;
const T3_RNG_SEED: u64 = 0xF22_FACE_DEED_BEEF;

#[test]
fn similarity_scan_composition_bitpack_roundtrip_and_distance_parity() {
    // ----- 1. Synthesize 100 fingerprints (8 lanes of u32 each), masked to
    //         the pack width so encode does not silently truncate. -----
    let mask = (1_u32 << T3_PACK_WIDTH) - 1;
    let total = T3_DATABASE_SIZE * T3_FINGERPRINT_LANES;
    let mut state = T3_RNG_SEED;
    let mut raw_db = Vec::with_capacity(total);
    for _ in 0..total {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let mixed = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32;
        raw_db.push(mixed & mask);
    }

    // ----- 2. Bit-pack at width 11; verify round-trip is bit-exact. -----
    let packer = DynamicBitPacker::new(T3_PACK_WIDTH);
    let packed_len = packer.encoded_len(raw_db.len());
    let mut packed = vec![0_u8; packed_len];
    packer.encode_u32_slice(&raw_db, &mut packed);

    let mut decoded = vec![0_u32; raw_db.len()];
    packer.decode_u32_slice(&packed, raw_db.len(), &mut decoded);
    assert_eq!(
        decoded, raw_db,
        "bit-pack round-trip diverged at width {T3_PACK_WIDTH}"
    );

    // ----- 3. Cast to f32 and compute L2-squared two ways. -----
    let db_f32: Vec<f32> = decoded.iter().map(|&v| v as f32).collect();
    let query_index = 0_usize;
    let query: Vec<f32> = db_f32
        [query_index * T3_FINGERPRINT_LANES..(query_index + 1) * T3_FINGERPRINT_LANES]
        .to_vec();

    // Batched dispatcher.
    let mut batched_out = vec![0_f32; T3_DATABASE_SIZE];
    l2_squared_f32_one_to_many(&query, &db_f32, T3_FINGERPRINT_LANES, &mut batched_out);

    // Serial scalar reference path: bypasses the dispatcher.
    let mut scalar_out = vec![0_f32; T3_DATABASE_SIZE];
    for (i, slot) in scalar_out.iter_mut().enumerate() {
        let row = &db_f32[i * T3_FINGERPRINT_LANES..(i + 1) * T3_FINGERPRINT_LANES];
        *slot = vec_scalar::l2_squared_f32(&query, row).unwrap_or(0.0);
    }

    // Self-distance is exactly zero on both paths regardless of reduction
    // tree order, so it is a sharp diagnostic for shape bugs.
    assert_eq!(
        batched_out[query_index], 0.0,
        "batched l2 self-distance is nonzero"
    );
    assert_eq!(
        scalar_out[query_index], 0.0,
        "scalar l2 self-distance is nonzero"
    );

    // ----- 4. Verify Higham-bounded agreement: the divergence is at most
    //         1e-3 * max_pair_l1_norm_of_products. The l1 norm of products
    //         (sum |q_i * d_i|) is the published Wilkinson bound for f32
    //         pairwise reductions; see vector/mod.rs. -----
    let mut max_l1_products: f64 = 0.0;
    for i in 0..T3_DATABASE_SIZE {
        let row = &db_f32[i * T3_FINGERPRINT_LANES..(i + 1) * T3_FINGERPRINT_LANES];
        let mut sum = 0.0_f64;
        for (&q, &d) in query.iter().zip(row) {
            sum += (q as f64).abs() * (d as f64).abs();
        }
        if sum > max_l1_products {
            max_l1_products = sum;
        }
    }
    let tolerance = 1e-3_f64 * max_l1_products.max(1.0);

    let mut max_diff = 0.0_f64;
    for (&s, &b) in scalar_out.iter().zip(&batched_out) {
        let diff = ((s as f64) - (b as f64)).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(
        max_diff <= tolerance,
        "scalar vs batched l2 diverged beyond Higham bound: max_diff = {max_diff:.3e}, tolerance = {tolerance:.3e} (max_l1_products = {max_l1_products:.3e})"
    );
}

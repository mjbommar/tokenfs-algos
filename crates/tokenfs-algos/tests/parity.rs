#![allow(missing_docs)]

use tokenfs_algos::{byteclass, chunk, entropy, hash, histogram::ByteHistogram};

fn reference_counts(bytes: &[u8]) -> [u64; 256] {
    let mut counts = [0_u64; 256];
    for &byte in bytes {
        counts[byte as usize] += 1;
    }
    counts
}

#[test]
fn byte_histogram_matches_scalar_reference() {
    let bytes = (0..8192)
        .map(|i| {
            let value = i * 17 + (i >> 3) * 31;
            value as u8
        })
        .collect::<Vec<_>>();

    let histogram = ByteHistogram::from_block(&bytes);

    assert_eq!(histogram.total(), bytes.len() as u64);
    assert_eq!(histogram.counts(), &reference_counts(&bytes));
}

#[test]
fn new_pinned_scalar_paths_match_public_defaults() {
    let bytes = (0..8192)
        .map(|i| {
            let value = i * 41 + (i >> 2) * 13;
            value as u8
        })
        .collect::<Vec<_>>();

    assert_eq!(
        byteclass::validate_utf8(&bytes),
        byteclass::kernels::scalar::validate_utf8(&bytes)
    );

    let histogram = ByteHistogram::from_block(&bytes);
    assert_eq!(
        entropy::kernels::auto::h1(&histogram),
        entropy::kernels::scalar::h1(&histogram)
    );
    assert_eq!(
        entropy::kernels::auto::min_h1(&histogram),
        entropy::kernels::scalar::min_h1(&histogram)
    );
    assert_eq!(
        entropy::kernels::auto::collision_h1(&histogram),
        entropy::kernels::scalar::collision_h1(&histogram)
    );
    assert_eq!(
        entropy::kernels::auto::joint_h2_pairs(&bytes),
        entropy::kernels::scalar::joint_h2_pairs(&bytes)
    );
    assert_eq!(
        entropy::kernels::auto::conditional_h_next_given_prev(&bytes),
        entropy::kernels::scalar::conditional_h_next_given_prev(&bytes)
    );

    assert_eq!(
        hash::fnv1a64(&bytes),
        hash::kernels::scalar::fnv1a64(&bytes)
    );
    assert_eq!(
        hash::mix64(&bytes, 7),
        hash::kernels::scalar::mix64(&bytes, 7)
    );
}

#[test]
fn pinned_chunk_kernels_match_public_boundaries() {
    let bytes = (0..131_072)
        .map(|i| {
            let value = i * 17 + (i >> 5) * 101;
            value as u8
        })
        .collect::<Vec<_>>();

    let gear = chunk::ChunkConfig::with_sizes(1024, 4096, 16 * 1024);
    assert_eq!(
        chunk::find_boundary(&bytes, gear),
        chunk::kernels::gear::find_boundary(&bytes, gear)
    );

    let fastcdc = chunk::FastCdcConfig::with_sizes(1024, 4096, 16 * 1024);
    assert_eq!(
        chunk::fastcdc_find_boundary(&bytes, fastcdc),
        chunk::kernels::fastcdc::find_boundary(&bytes, fastcdc)
    );
}

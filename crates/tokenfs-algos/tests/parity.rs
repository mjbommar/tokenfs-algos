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
    // `entropy::kernels::{auto,scalar}::joint_h2_pairs` are gated on
    // `userspace` because they build a 256 KiB dense pair histogram
    // on the stack (audit-R8 #6b). The heap-free with_scratch path
    // is exercised in the lib unit tests.
    #[cfg(feature = "userspace")]
    {
        assert_eq!(
            entropy::kernels::auto::joint_h2_pairs(&bytes),
            entropy::kernels::scalar::joint_h2_pairs(&bytes)
        );
    }
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
fn pipelined_hash4_bins_matches_scalar_reference() {
    use tokenfs_algos::sketch;

    // SSE4.2 is the only path that dispatches into the pipelined kernel;
    // on hosts without it the public crc32_hash4_bins falls through to
    // scalar and the test is trivially true.
    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if !sketch::kernels::sse42::is_available() {
            return;
        }
    }

    // Cover lengths around the 4-window inner-loop boundary plus a few
    // longer cases that exercise both the inner loop and the tail.
    for len in [
        4_usize, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 31, 32, 33, 64, 65, 1023, 1024, 1025, 4095,
        4096, 8192,
    ] {
        let bytes: Vec<u8> = (0..len)
            .map(|i| (i.wrapping_mul(17) ^ (i >> 2)) as u8)
            .collect();

        // Reference: scalar (no SIMD, no pipelining).
        const BINS: usize = 1024;
        let mut expected = [0_u32; BINS];
        sketch::kernels::scalar::crc32_hash4_bins::<BINS>(&bytes, &mut expected);

        // Public path (dispatches to pipelined SSE4.2 on this host).
        let mut actual = [0_u32; BINS];
        sketch::crc32_hash4_bins::<BINS>(&bytes, &mut actual);

        assert_eq!(
            actual, expected,
            "pipelined hash4_bins diverged from scalar at len {len}"
        );
    }
}

#[test]
fn pipelined_hash4_bins_handles_non_power_of_two_bins() {
    // The pipelined kernel falls back to the single-stream path for
    // non-power-of-two BINS. Both paths must agree.
    use tokenfs_algos::sketch;

    #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if !sketch::kernels::sse42::is_available() {
            return;
        }
    }

    const BINS: usize = 1000; // not a power of two
    let bytes: Vec<u8> = (0_u32..2048).map(|i| (i.wrapping_mul(31)) as u8).collect();

    let mut expected = [0_u32; BINS];
    sketch::kernels::scalar::crc32_hash4_bins::<BINS>(&bytes, &mut expected);

    let mut actual = [0_u32; BINS];
    sketch::crc32_hash4_bins::<BINS>(&bytes, &mut actual);

    assert_eq!(actual, expected);
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

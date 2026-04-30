#![allow(missing_docs)]

use proptest::prelude::*;
use tokenfs_algos::histogram::ByteHistogram;

fn reference_counts(bytes: &[u8]) -> [u64; 256] {
    let mut counts = [0_u64; 256];
    for &byte in bytes {
        counts[byte as usize] += 1;
    }
    counts
}

#[test]
fn edge_lengths_match_reference_counts() {
    let lengths = [
        0_usize, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
    ];

    for len in lengths {
        let bytes = (0..len)
            .map(|i| (i.wrapping_mul(17) ^ (i >> 1).wrapping_mul(31)) as u8)
            .collect::<Vec<_>>();
        let histogram = ByteHistogram::from_block(&bytes);

        assert_eq!(histogram.total(), len as u64, "length {len}");
        assert_eq!(
            histogram.counts(),
            &reference_counts(&bytes),
            "length {len}"
        );
    }
}

#[test]
fn unaligned_subslices_match_reference_counts() {
    let bytes = (0_usize..2048)
        .map(|i| (i.wrapping_mul(19) ^ (i >> 2).wrapping_mul(23)) as u8)
        .collect::<Vec<_>>();

    for start in 0..64 {
        for len in [0_usize, 1, 15, 16, 17, 31, 32, 33, 127, 255, 511] {
            let end = (start + len).min(bytes.len());
            let slice = &bytes[start..end];
            let histogram = ByteHistogram::from_block(slice);

            assert_eq!(histogram.total(), slice.len() as u64);
            assert_eq!(histogram.counts(), &reference_counts(slice));
        }
    }
}

#[test]
fn clear_resets_histogram() {
    let mut histogram = ByteHistogram::from_block(b"not empty");

    histogram.clear();

    assert!(histogram.is_empty());
    assert_eq!(histogram.total(), 0);
    assert_eq!(histogram.counts(), &[0; 256]);
}

proptest! {
    #[test]
    fn counts_sum_to_total(bytes in proptest::collection::vec(any::<u8>(), 0..16384)) {
        let histogram = ByteHistogram::from_block(&bytes);
        let sum = histogram.counts().iter().sum::<u64>();

        prop_assert_eq!(histogram.total(), bytes.len() as u64);
        prop_assert_eq!(sum, histogram.total());
    }

    #[test]
    fn merging_matches_concatenated_histogram(
        left in proptest::collection::vec(any::<u8>(), 0..8192),
        right in proptest::collection::vec(any::<u8>(), 0..8192),
    ) {
        let mut concatenated = left.clone();
        concatenated.extend_from_slice(&right);

        let merged = ByteHistogram::from_block(&left) + &ByteHistogram::from_block(&right);
        let expected = ByteHistogram::from_block(&concatenated);

        prop_assert_eq!(merged, expected);
    }
}

#![allow(missing_docs)]

use tokenfs_algos::histogram::ByteHistogram;

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

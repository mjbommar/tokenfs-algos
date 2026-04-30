#![allow(missing_docs)]

use tokenfs_algos::{entropy::shannon, histogram::ByteHistogram};

#[test]
fn empty_histogram_has_zero_entropy() {
    let histogram = ByteHistogram::new();
    assert_eq!(shannon::h1(&histogram), 0.0);
}

#[test]
fn constant_input_has_zero_entropy() {
    let histogram = ByteHistogram::from_block(&[42; 4096]);
    assert_eq!(shannon::h1(&histogram), 0.0);
}

#[test]
fn uniform_byte_input_has_eight_bits_per_byte() {
    let bytes = (0_u8..=255).cycle().take(256 * 16).collect::<Vec<_>>();
    let histogram = ByteHistogram::from_block(&bytes);

    assert!((shannon::h1(&histogram) - 8.0).abs() < 0.000_001);
}

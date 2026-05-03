#![allow(missing_docs)]

use tokenfs_algos::{
    entropy::{min, renyi, shannon},
    hash,
    histogram::ByteHistogram,
};
// Used only by the userspace-gated `adjacent_pair_entropy_known_values`
// test (audit-R8 #6b).
#[cfg(feature = "userspace")]
use tokenfs_algos::entropy::{conditional, joint};

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

#[test]
fn min_and_collision_entropy_match_uniform_distribution() {
    let bytes = (0_u8..=255).collect::<Vec<_>>();
    let histogram = ByteHistogram::from_block(&bytes);

    assert!((min::h1(&histogram) - 8.0).abs() < 0.000_001);
    assert!((renyi::collision_h1(&histogram) - 8.0).abs() < 0.000_001);
}

// `joint::h2_pairs` builds a 256 KiB dense pair histogram on the stack
// and is gated on the `userspace` feature (audit-R8 #6b). This test
// exercises the by-value entry; kernel-default builds skip it. The
// heap-free `h2_pairs_with_scratch` / `h2_pairs_with_dense_scratch`
// siblings are tested in the lib's `entropy::joint::tests` module.
#[cfg(feature = "userspace")]
#[test]
fn adjacent_pair_entropy_known_values() {
    assert_eq!(joint::h2_pairs(b"aaaaaaaa"), 0.0);
    assert_eq!(conditional::h_next_given_prev(b"abababab"), 0.0);
    assert!((joint::h2_pairs(b"abababa") - 1.0).abs() < 0.000_001);
}

#[test]
fn fnv1a64_known_values() {
    assert_eq!(hash::fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(hash::fnv1a64(b"hello"), 0xa430_d846_80aa_bd0b);
}

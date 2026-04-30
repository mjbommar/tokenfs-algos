#![allow(missing_docs)]

use proptest::prelude::*;
use tokenfs_algos::{entropy::shannon, histogram::ByteHistogram};

proptest! {
    #[test]
    fn h1_stays_within_byte_entropy_bounds(bytes in proptest::collection::vec(any::<u8>(), 0..8192)) {
        let histogram = ByteHistogram::from_block(&bytes);
        let h1 = shannon::h1(&histogram);

        prop_assert!(h1 >= 0.0);
        prop_assert!(h1 <= 8.0 + f32::EPSILON);
    }

    #[test]
    fn h1_is_deterministic(bytes in proptest::collection::vec(any::<u8>(), 0..8192)) {
        let histogram = ByteHistogram::from_block(&bytes);

        prop_assert_eq!(shannon::h1(&histogram), shannon::h1(&histogram));
    }
}

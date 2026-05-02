//! Container cardinality dispatch.
//!
//! Single-container cardinality routes to the per-kind helper:
//!
//! * [`super::BitmapContainer`]'s `cardinality` uses
//!   [`crate::bits::popcount`] (which itself dispatches to AVX-512
//!   VPOPCNTQ / AVX2 Mula / NEON VCNT).
//! * [`super::ArrayContainer`]'s `cardinality` is `data.len()`.
//! * [`super::RunContainer`]'s `cardinality` sums `(length_minus_one +
//!   1)` over the run-list.

use crate::bitmap::containers::Container;

/// Returns the number of elements in `container`.
#[must_use]
pub fn cardinality(container: &Container) -> u32 {
    match container {
        Container::Bitmap(bm) => bm.cardinality(),
        Container::Array(arr) => arr.cardinality(),
        Container::Run(rc) => rc.cardinality(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitmap::containers::{ArrayContainer, BitmapContainer, RunContainer};

    #[test]
    fn cardinality_bitmap() {
        let mut bm = BitmapContainer::empty();
        for v in [0_u16, 1, 100, 65535] {
            bm.insert(v);
        }
        assert_eq!(cardinality(&Container::Bitmap(bm)), 4);
    }

    #[test]
    fn cardinality_array() {
        let arr = ArrayContainer::from_sorted(vec![1, 2, 3]);
        assert_eq!(cardinality(&Container::Array(arr)), 3);
    }

    #[test]
    fn cardinality_run() {
        let rc = RunContainer::from_runs(vec![(0, 9), (100, 0)]);
        assert_eq!(cardinality(&Container::Run(rc)), 11);
    }
}

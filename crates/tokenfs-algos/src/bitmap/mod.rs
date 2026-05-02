//! Roaring-style SIMD container kernels.
//!
//! Sprint 20-29 of the v0.2 plan (`docs/v0.2_planning/03_EXECUTION_PLAN.md`)
//! lands this module. Spec lives in
//! `docs/v0.2_planning/11_BITMAP.md`.
//!
//! ## Goal
//!
//! Provide SIMD kernels for the **inner loops** of Roaring set algebra
//! at primitive granularity. We do NOT ship a high-level `Bitmap` type
//! — the existing `roaring-rs` crate already provides container-of-
//! containers, build, ser/de, and iteration. Our role is to expose the
//! container-level kernels (Schlegel intersect, AVX-512 AND + VPOPCNTQ,
//! VPCOMPRESSD output materialisation) as composable primitives that
//! `roaring-rs`, `tokenfs-paper`, or other consumers can plug in.
//!
//! ## Container types
//!
//! Roaring partitions a `u32` set into 16-bit "high keys" and 16-bit
//! "low keys"; this module ships the three low-key container shapes:
//!
//! * [`BitmapContainer`] — dense 65 536-bit bitmap = 1024 × `u64` = 8 KiB.
//! * [`ArrayContainer`] — sorted-`u16` array, ≤ 4096 entries.
//! * [`RunContainer`] — sorted `(start, length_minus_one)` pairs.
//!
//! The [`Container`] enum dispatches the five Boolean ops (intersect,
//! union, difference, symmetric difference, cardinality) over every
//! pair of container kinds per the table in spec § 2.
//!
//! ## Public API
//!
//! ```
//! use tokenfs_algos::bitmap::{ArrayContainer, BitmapContainer, Container};
//!
//! // Build two array containers and intersect.
//! let a = ArrayContainer::from_sorted(vec![1_u16, 3, 5, 7, 9]);
//! let b = ArrayContainer::from_sorted(vec![2_u16, 3, 5, 8, 10]);
//! let result = Container::Array(a).intersect(&Container::Array(b));
//! match result {
//!     Container::Array(arr) => assert_eq!(arr.data(), &[3_u16, 5][..]),
//!     _ => unreachable!(),
//! }
//!
//! // Bitmap × bitmap with a `_justcard` query.
//! let mut bm_a = BitmapContainer::empty();
//! let mut bm_b = BitmapContainer::empty();
//! for v in [0_u16, 5, 10, 100] {
//!     bm_a.insert(v);
//! }
//! for v in [5_u16, 10, 200] {
//!     bm_b.insert(v);
//! }
//! let card = Container::Bitmap(bm_a)
//!     .intersect_cardinality(&Container::Bitmap(bm_b));
//! assert_eq!(card, 2); // {5, 10}
//! ```
//!
//! ## Three variants per bitmap × bitmap op
//!
//! Per CRoaring's `_card` / `_nocard` / `_justcard` discipline, we ship
//! three forms of every bitmap × bitmap op:
//!
//! ```
//! use tokenfs_algos::bitmap::BitmapContainer;
//!
//! let mut a = BitmapContainer::empty();
//! let mut b = BitmapContainer::empty();
//! for v in [0_u16, 5, 10, 100] {
//!     a.insert(v);
//! }
//! for v in [5_u16, 100, 200] {
//!     b.insert(v);
//! }
//! let mut out = BitmapContainer::empty();
//!
//! // Materialises out and returns cardinality.
//! let card = a.and_into(&b, &mut out);
//!
//! // Materialises out without computing cardinality.
//! a.and_into_nocard(&b, &mut out);
//!
//! // Returns cardinality without materialising.
//! let card_only = a.and_cardinality(&b);
//!
//! assert_eq!(card, card_only);
//! ```

#![allow(clippy::module_name_repetitions)]

pub mod cardinality;
pub mod containers;
pub mod difference;
pub mod intersect;
pub mod kernels;
pub mod union;
pub mod xor;

pub use containers::{ArrayContainer, BitmapContainer, BitmapIter, Container, RunContainer};

impl Container {
    /// Returns `a ∩ b`, automatically choosing the cheapest container
    /// representation for the result.
    #[must_use]
    pub fn intersect(&self, other: &Container) -> Container {
        intersect::intersect(self, other)
    }

    /// Returns `a ∪ b`.
    #[must_use]
    pub fn union(&self, other: &Container) -> Container {
        union::union(self, other)
    }

    /// Returns `a \ b`.
    #[must_use]
    pub fn difference(&self, other: &Container) -> Container {
        difference::difference(self, other)
    }

    /// Returns `a △ b` (symmetric difference / XOR).
    #[must_use]
    pub fn symmetric_difference(&self, other: &Container) -> Container {
        xor::symmetric_difference(self, other)
    }

    /// Returns the number of set values.
    #[must_use]
    pub fn cardinality(&self) -> u32 {
        cardinality::cardinality(self)
    }

    /// Returns the cardinality of `a ∩ b` without materialising the result.
    ///
    /// This is the `_justcard` variant — typically 2-3x faster than
    /// `intersect(...).cardinality()` because the result store is
    /// skipped entirely.
    #[must_use]
    pub fn intersect_cardinality(&self, other: &Container) -> u32 {
        intersect::intersect_cardinality(self, other)
    }
}

impl BitmapContainer {
    /// `_card` variant of bitmap AND: materialises `self & other` into
    /// `out` and returns the result cardinality.
    ///
    /// The dispatch is identical to [`Container::intersect`] for two
    /// bitmap containers but exposes the bitmap-only entry point so
    /// callers that already have raw `BitmapContainer` handles can
    /// avoid the [`Container`] wrapping.
    pub fn and_into(&self, other: &Self, out: &mut Self) -> u32 {
        kernels_dispatch::and_into(self, other, out)
    }

    /// `_nocard` variant of bitmap AND.
    pub fn and_into_nocard(&self, other: &Self, out: &mut Self) {
        kernels_dispatch::and_into_nocard(self, other, out);
    }

    /// `_justcard` variant of bitmap AND.
    #[must_use]
    pub fn and_cardinality(&self, other: &Self) -> u32 {
        kernels_dispatch::and_cardinality(self, other)
    }

    /// `_card` variant of bitmap OR.
    pub fn or_into(&self, other: &Self, out: &mut Self) -> u32 {
        kernels_dispatch::or_into(self, other, out)
    }

    /// `_nocard` variant of bitmap OR.
    pub fn or_into_nocard(&self, other: &Self, out: &mut Self) {
        kernels_dispatch::or_into_nocard(self, other, out);
    }

    /// `_justcard` variant of bitmap OR.
    #[must_use]
    pub fn or_cardinality(&self, other: &Self) -> u32 {
        kernels_dispatch::or_cardinality(self, other)
    }

    /// `_card` variant of bitmap XOR.
    pub fn xor_into(&self, other: &Self, out: &mut Self) -> u32 {
        kernels_dispatch::xor_into(self, other, out)
    }

    /// `_nocard` variant of bitmap XOR.
    pub fn xor_into_nocard(&self, other: &Self, out: &mut Self) {
        kernels_dispatch::xor_into_nocard(self, other, out);
    }

    /// `_justcard` variant of bitmap XOR.
    #[must_use]
    pub fn xor_cardinality(&self, other: &Self) -> u32 {
        kernels_dispatch::xor_cardinality(self, other)
    }

    /// `_card` variant of bitmap AND-NOT (`self AND NOT other`).
    pub fn andnot_into(&self, other: &Self, out: &mut Self) -> u32 {
        kernels_dispatch::andnot_into(self, other, out)
    }

    /// `_nocard` variant of bitmap AND-NOT.
    pub fn andnot_into_nocard(&self, other: &Self, out: &mut Self) {
        kernels_dispatch::andnot_into_nocard(self, other, out);
    }

    /// `_justcard` variant of bitmap AND-NOT.
    #[must_use]
    pub fn andnot_cardinality(&self, other: &Self) -> u32 {
        kernels_dispatch::andnot_cardinality(self, other)
    }
}

/// Internal dispatch wrappers for the bitmap × bitmap kernels.
///
/// These are pulled out into a private module so the public
/// `BitmapContainer` impl block reads as a thin facade and so the
/// per-op dispatch logic lives in one place.
mod kernels_dispatch {
    use super::{BitmapContainer, kernels};

    macro_rules! dispatch_bitmap_into {
        ($name:ident, $avx512_op:path, $avx2_op:path, $neon_op:path, $scalar_op:path) => {
            pub(super) fn $name(
                a: &BitmapContainer,
                b: &BitmapContainer,
                out: &mut BitmapContainer,
            ) -> u32 {
                #[cfg(all(
                    feature = "std",
                    feature = "avx512",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if kernels::bitmap_x_bitmap_avx512::is_available() {
                        // SAFETY: availability checked.
                        return unsafe { $avx512_op(&a.words, &b.words, &mut out.words) };
                    }
                }
                #[cfg(all(
                    feature = "std",
                    feature = "avx2",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if kernels::bitmap_x_bitmap_avx2::is_available() {
                        // SAFETY: availability checked.
                        return unsafe { $avx2_op(&a.words, &b.words, &mut out.words) };
                    }
                }
                #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
                {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { $neon_op(&a.words, &b.words, &mut out.words) };
                }
                #[allow(unreachable_code)]
                $scalar_op(&a.words, &b.words, &mut out.words)
            }
        };
    }

    macro_rules! dispatch_bitmap_into_nocard {
        ($name:ident, $avx512_op:path, $avx2_op:path, $neon_op:path, $scalar_op:path) => {
            pub(super) fn $name(
                a: &BitmapContainer,
                b: &BitmapContainer,
                out: &mut BitmapContainer,
            ) {
                #[cfg(all(
                    feature = "std",
                    feature = "avx512",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if kernels::bitmap_x_bitmap_avx512::is_available() {
                        // SAFETY: availability checked.
                        unsafe {
                            $avx512_op(&a.words, &b.words, &mut out.words);
                        }
                        return;
                    }
                }
                #[cfg(all(
                    feature = "std",
                    feature = "avx2",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if kernels::bitmap_x_bitmap_avx2::is_available() {
                        // SAFETY: availability checked.
                        unsafe {
                            $avx2_op(&a.words, &b.words, &mut out.words);
                        }
                        return;
                    }
                }
                #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
                {
                    // SAFETY: NEON is mandatory on AArch64.
                    unsafe {
                        $neon_op(&a.words, &b.words, &mut out.words);
                    }
                    return;
                }
                #[allow(unreachable_code)]
                $scalar_op(&a.words, &b.words, &mut out.words)
            }
        };
    }

    macro_rules! dispatch_bitmap_card {
        ($name:ident, $avx512_op:path, $avx2_op:path, $neon_op:path, $scalar_op:path) => {
            pub(super) fn $name(a: &BitmapContainer, b: &BitmapContainer) -> u32 {
                #[cfg(all(
                    feature = "std",
                    feature = "avx512",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if kernels::bitmap_x_bitmap_avx512::is_available() {
                        // SAFETY: availability checked.
                        return unsafe { $avx512_op(&a.words, &b.words) };
                    }
                }
                #[cfg(all(
                    feature = "std",
                    feature = "avx2",
                    any(target_arch = "x86", target_arch = "x86_64")
                ))]
                {
                    if kernels::bitmap_x_bitmap_avx2::is_available() {
                        // SAFETY: availability checked.
                        return unsafe { $avx2_op(&a.words, &b.words) };
                    }
                }
                #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
                {
                    // SAFETY: NEON is mandatory on AArch64.
                    return unsafe { $neon_op(&a.words, &b.words) };
                }
                #[allow(unreachable_code)]
                $scalar_op(&a.words, &b.words)
            }
        };
    }

    // bitmap×bitmap AND
    dispatch_bitmap_into!(
        and_into,
        kernels::bitmap_x_bitmap_avx512::and_into,
        kernels::bitmap_x_bitmap_avx2::and_into,
        kernels::bitmap_x_bitmap_neon::and_into,
        kernels::bitmap_x_bitmap_scalar::and_into
    );
    dispatch_bitmap_into_nocard!(
        and_into_nocard,
        kernels::bitmap_x_bitmap_avx512::and_into_nocard,
        kernels::bitmap_x_bitmap_avx2::and_into_nocard,
        kernels::bitmap_x_bitmap_neon::and_into_nocard,
        kernels::bitmap_x_bitmap_scalar::and_into_nocard
    );
    dispatch_bitmap_card!(
        and_cardinality,
        kernels::bitmap_x_bitmap_avx512::and_cardinality,
        kernels::bitmap_x_bitmap_avx2::and_cardinality,
        kernels::bitmap_x_bitmap_neon::and_cardinality,
        kernels::bitmap_x_bitmap_scalar::and_cardinality
    );

    // bitmap×bitmap OR
    dispatch_bitmap_into!(
        or_into,
        kernels::bitmap_x_bitmap_avx512::or_into,
        kernels::bitmap_x_bitmap_avx2::or_into,
        kernels::bitmap_x_bitmap_neon::or_into,
        kernels::bitmap_x_bitmap_scalar::or_into
    );
    dispatch_bitmap_into_nocard!(
        or_into_nocard,
        kernels::bitmap_x_bitmap_avx512::or_into_nocard,
        kernels::bitmap_x_bitmap_avx2::or_into_nocard,
        kernels::bitmap_x_bitmap_neon::or_into_nocard,
        kernels::bitmap_x_bitmap_scalar::or_into_nocard
    );
    dispatch_bitmap_card!(
        or_cardinality,
        kernels::bitmap_x_bitmap_avx512::or_cardinality,
        kernels::bitmap_x_bitmap_avx2::or_cardinality,
        kernels::bitmap_x_bitmap_neon::or_cardinality,
        kernels::bitmap_x_bitmap_scalar::or_cardinality
    );

    // bitmap×bitmap XOR
    dispatch_bitmap_into!(
        xor_into,
        kernels::bitmap_x_bitmap_avx512::xor_into,
        kernels::bitmap_x_bitmap_avx2::xor_into,
        kernels::bitmap_x_bitmap_neon::xor_into,
        kernels::bitmap_x_bitmap_scalar::xor_into
    );
    dispatch_bitmap_into_nocard!(
        xor_into_nocard,
        kernels::bitmap_x_bitmap_avx512::xor_into_nocard,
        kernels::bitmap_x_bitmap_avx2::xor_into_nocard,
        kernels::bitmap_x_bitmap_neon::xor_into_nocard,
        kernels::bitmap_x_bitmap_scalar::xor_into_nocard
    );
    dispatch_bitmap_card!(
        xor_cardinality,
        kernels::bitmap_x_bitmap_avx512::xor_cardinality,
        kernels::bitmap_x_bitmap_avx2::xor_cardinality,
        kernels::bitmap_x_bitmap_neon::xor_cardinality,
        kernels::bitmap_x_bitmap_scalar::xor_cardinality
    );

    // bitmap×bitmap ANDNOT
    dispatch_bitmap_into!(
        andnot_into,
        kernels::bitmap_x_bitmap_avx512::andnot_into,
        kernels::bitmap_x_bitmap_avx2::andnot_into,
        kernels::bitmap_x_bitmap_neon::andnot_into,
        kernels::bitmap_x_bitmap_scalar::andnot_into
    );
    dispatch_bitmap_into_nocard!(
        andnot_into_nocard,
        kernels::bitmap_x_bitmap_avx512::andnot_into_nocard,
        kernels::bitmap_x_bitmap_avx2::andnot_into_nocard,
        kernels::bitmap_x_bitmap_neon::andnot_into_nocard,
        kernels::bitmap_x_bitmap_scalar::andnot_into_nocard
    );
    dispatch_bitmap_card!(
        andnot_cardinality,
        kernels::bitmap_x_bitmap_avx512::andnot_cardinality,
        kernels::bitmap_x_bitmap_avx2::andnot_cardinality,
        kernels::bitmap_x_bitmap_neon::andnot_cardinality,
        kernels::bitmap_x_bitmap_scalar::andnot_cardinality
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_bitmap(values: &[u16]) -> BitmapContainer {
        let mut bm = BitmapContainer::empty();
        for &v in values {
            bm.insert(v);
        }
        bm
    }

    #[test]
    fn three_variants_agree_for_and() {
        let a = fill_bitmap(&(0..1000).step_by(7).map(|i| i as u16).collect::<Vec<_>>());
        let b = fill_bitmap(&(0..1000).step_by(11).map(|i| i as u16).collect::<Vec<_>>());

        let mut out_card = BitmapContainer::empty();
        let card = a.and_into(&b, &mut out_card);

        let mut out_nocard = BitmapContainer::empty();
        a.and_into_nocard(&b, &mut out_nocard);

        let card_only = a.and_cardinality(&b);

        assert_eq!(card, card_only);
        assert_eq!(out_card.cardinality(), card);
        assert_eq!(out_card, out_nocard);
    }

    #[test]
    fn three_variants_agree_for_or() {
        let a = fill_bitmap(&[0, 5, 10]);
        let b = fill_bitmap(&[5, 10, 100]);

        let mut out_card = BitmapContainer::empty();
        let card = a.or_into(&b, &mut out_card);

        let mut out_nocard = BitmapContainer::empty();
        a.or_into_nocard(&b, &mut out_nocard);

        let card_only = a.or_cardinality(&b);

        assert_eq!(card, card_only);
        assert_eq!(card, 4);
        assert_eq!(out_card, out_nocard);
    }

    #[test]
    fn three_variants_agree_for_xor() {
        let a = fill_bitmap(&[0, 5, 10]);
        let b = fill_bitmap(&[5, 10, 100]);

        let mut out_card = BitmapContainer::empty();
        let card = a.xor_into(&b, &mut out_card);

        let mut out_nocard = BitmapContainer::empty();
        a.xor_into_nocard(&b, &mut out_nocard);

        let card_only = a.xor_cardinality(&b);

        assert_eq!(card, card_only);
        assert_eq!(card, 2); // {0, 100}
        assert_eq!(out_card, out_nocard);
    }

    #[test]
    fn three_variants_agree_for_andnot() {
        let a = fill_bitmap(&[0, 5, 10, 100]);
        let b = fill_bitmap(&[5, 100, 200]);

        let mut out_card = BitmapContainer::empty();
        let card = a.andnot_into(&b, &mut out_card);

        let mut out_nocard = BitmapContainer::empty();
        a.andnot_into_nocard(&b, &mut out_nocard);

        let card_only = a.andnot_cardinality(&b);

        assert_eq!(card, card_only);
        assert_eq!(card, 2); // {0, 10}
        assert_eq!(out_card, out_nocard);
    }
}

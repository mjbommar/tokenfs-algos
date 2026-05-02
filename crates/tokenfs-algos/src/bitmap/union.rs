//! Container × container union dispatch.
//!
//! Mirrors [`crate::bitmap::intersect`]. Where intersect can shrink a
//! container kind (bitmap → array via promotion), union typically grows
//! one (array → bitmap when the result exceeds the array threshold).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bitmap::containers::{
    ARRAY_MAX_CARDINALITY, ArrayContainer, BITMAP_WORDS, BitmapContainer, Container, RunContainer,
};
use crate::bitmap::kernels;

/// Top-level dispatch.
#[must_use]
pub fn union(a: &Container, b: &Container) -> Container {
    match (a, b) {
        (Container::Bitmap(a), Container::Bitmap(b)) => union_bitmap_bitmap(a, b),
        (Container::Array(a), Container::Array(b)) => union_array_array(a, b),
        (Container::Array(a), Container::Bitmap(b))
        | (Container::Bitmap(b), Container::Array(a)) => union_array_bitmap(a, b),
        (Container::Run(a), Container::Run(b)) => union_run_run(a, b),
        (Container::Run(a), Container::Array(b)) | (Container::Array(b), Container::Run(a)) => {
            union_run_array(a, b)
        }
        (Container::Run(a), Container::Bitmap(b)) | (Container::Bitmap(b), Container::Run(a)) => {
            union_run_bitmap(a, b)
        }
    }
}

/// bitmap × bitmap union with `_card`.
fn union_bitmap_bitmap(a: &BitmapContainer, b: &BitmapContainer) -> Container {
    let mut out = BitmapContainer::empty();
    let _card = bitmap_x_bitmap_or_into(a, b, &mut out);
    Container::Bitmap(out)
}

fn bitmap_x_bitmap_or_into(
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
            return unsafe {
                kernels::bitmap_x_bitmap_avx512::or_into(&a.words, &b.words, &mut out.words)
            };
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
            return unsafe {
                kernels::bitmap_x_bitmap_avx2::or_into(&a.words, &b.words, &mut out.words)
            };
        }
    }
    #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe {
            kernels::bitmap_x_bitmap_neon::or_into(&a.words, &b.words, &mut out.words)
        };
    }
    #[allow(unreachable_code)]
    kernels::bitmap_x_bitmap_scalar::or_into(&a.words, &b.words, &mut out.words)
}

/// array × array union via merge-sort scalar kernel, then promote to
/// bitmap if it exceeds the array threshold.
fn union_array_array(a: &ArrayContainer, b: &ArrayContainer) -> Container {
    let mut out = Vec::with_capacity(a.data.len() + b.data.len());
    kernels::array_x_array_scalar::union(&a.data, &b.data, &mut out);
    if out.len() > ARRAY_MAX_CARDINALITY {
        let mut bm = BitmapContainer::empty();
        for v in &out {
            bm.insert(*v);
        }
        Container::Bitmap(bm)
    } else {
        Container::Array(ArrayContainer::from_sorted(out))
    }
}

/// array ∪ bitmap = bitmap with the array's bits OR'd in.
fn union_array_bitmap(array: &ArrayContainer, bitmap: &BitmapContainer) -> Container {
    let mut out = BitmapContainer::empty();
    kernels::array_x_bitmap_scalar::union_array_bitmap(&array.data, &bitmap.words, &mut out.words);
    Container::Bitmap(out)
}

/// run × run union — scalar interval-merge with coalescing.
fn union_run_run(a: &RunContainer, b: &RunContainer) -> Container {
    // Walk both run lists and emit a coalesced merge.
    let mut all: Vec<(u32, u32)> = Vec::with_capacity(a.runs.len() + b.runs.len());
    for &(start, len_m1) in &a.runs {
        all.push((u32::from(start), u32::from(start) + u32::from(len_m1)));
    }
    for &(start, len_m1) in &b.runs {
        all.push((u32::from(start), u32::from(start) + u32::from(len_m1)));
    }
    all.sort_by_key(|&(s, _)| s);

    let mut merged: Vec<(u16, u16)> = Vec::new();
    let mut i = 0_usize;
    while i < all.len() {
        let (s, mut e) = all[i];
        i += 1;
        while i < all.len() && all[i].0 <= e + 1 {
            // Merge with next: consume any run that overlaps or touches.
            e = e.max(all[i].1);
            i += 1;
        }
        merged.push((s as u16, (e - s) as u16));
    }

    let rc = RunContainer::from_runs(merged);
    Container::Run(rc)
}

/// run × array union — convert array to run-extension-then-merge form.
fn union_run_array(run: &RunContainer, array: &ArrayContainer) -> Container {
    // Extend `array` into single-element runs and union with `run`.
    let array_runs: Vec<(u16, u16)> = array.data.iter().map(|&v| (v, 0)).collect();
    let array_rc = RunContainer { runs: array_runs };
    union_run_run(run, &array_rc)
}

/// run × bitmap union — set bits for every run.
fn union_run_bitmap(run: &RunContainer, bitmap: &BitmapContainer) -> Container {
    let mut out = bitmap.clone();
    for &(start, len_m1) in &run.runs {
        let start = u32::from(start);
        let end = start + u32::from(len_m1);
        for v in start..=end {
            out.insert(v as u16);
        }
    }
    Container::Bitmap(out)
}

// `BITMAP_WORDS` is referenced by the SIMD branches; suppress the
// unused-import warning when only the scalar fallback path compiles.
#[allow(dead_code)]
const _BITMAP_WORDS_REF: usize = BITMAP_WORDS;

#[cfg(test)]
mod tests {
    use super::*;
    // `vec!` is not in the no-std prelude; alias it from `alloc` for
    // the alloc-only build (audit-R6 finding #164).
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;

    fn fill_bitmap(values: &[u16]) -> BitmapContainer {
        let mut bm = BitmapContainer::empty();
        for &v in values {
            bm.insert(v);
        }
        bm
    }

    #[test]
    fn bitmap_x_bitmap_union_correct() {
        let a = fill_bitmap(&[0, 5, 10]);
        let b = fill_bitmap(&[5, 10, 100]);
        let result = union(&Container::Bitmap(a), &Container::Bitmap(b));
        match result {
            Container::Bitmap(bm) => {
                assert_eq!(bm.cardinality(), 4);
                assert!(bm.contains(0));
                assert!(bm.contains(5));
                assert!(bm.contains(10));
                assert!(bm.contains(100));
            }
            other => panic!("expected bitmap, got {other:?}"),
        }
    }

    #[test]
    fn array_x_array_union_correct() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5]);
        let b = ArrayContainer::from_sorted(vec![2, 3, 6]);
        let result = union(&Container::Array(a), &Container::Array(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![1, 2, 3, 5, 6]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn array_x_bitmap_union_promotes_to_bitmap() {
        let array = ArrayContainer::from_sorted(vec![1, 5, 1000]);
        let bitmap = fill_bitmap(&[1, 100, 200]);
        let result = union(&Container::Array(array), &Container::Bitmap(bitmap));
        match result {
            Container::Bitmap(bm) => {
                assert_eq!(bm.cardinality(), 5);
                assert!(bm.contains(1));
                assert!(bm.contains(5));
                assert!(bm.contains(100));
                assert!(bm.contains(200));
                assert!(bm.contains(1000));
            }
            other => panic!("expected bitmap, got {other:?}"),
        }
    }

    #[test]
    fn run_x_run_union_coalesces_overlap() {
        let a = RunContainer::from_runs(vec![(0, 9), (50, 9)]);
        // a = 0..=9 ∪ 50..=59.
        let b = RunContainer::from_runs(vec![(5, 9)]);
        // b = 5..=14. Should coalesce: 0..=14 ∪ 50..=59.
        let result = union(&Container::Run(a), &Container::Run(b));
        match result {
            Container::Run(rc) => {
                assert_eq!(rc.runs, vec![(0, 14), (50, 9)]);
                assert_eq!(rc.cardinality(), 15 + 10);
            }
            other => panic!("expected run, got {other:?}"),
        }
    }

    #[test]
    fn array_x_array_union_promotes_to_bitmap_when_overflowing() {
        // Build two arrays whose union is > 4096.
        let a = ArrayContainer::from_sorted((0..3000).map(|i| i as u16).collect());
        let b = ArrayContainer::from_sorted((3000..6000).map(|i| i as u16).collect());
        let result = union(&Container::Array(a), &Container::Array(b));
        assert!(matches!(result, Container::Bitmap(_)));
    }
}

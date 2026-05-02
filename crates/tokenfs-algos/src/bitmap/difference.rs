//! Container × container set difference dispatch (`a \ b`).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bitmap::containers::{
    ARRAY_MAX_CARDINALITY, ArrayContainer, BITMAP_WORDS, BitmapContainer, Container, RunContainer,
};
use crate::bitmap::kernels;

/// Top-level dispatch for `a \ b`.
#[must_use]
pub fn difference(a: &Container, b: &Container) -> Container {
    match (a, b) {
        (Container::Bitmap(a), Container::Bitmap(b)) => difference_bitmap_bitmap(a, b),
        (Container::Array(a), Container::Array(b)) => difference_array_array(a, b),
        (Container::Array(a), Container::Bitmap(b)) => difference_array_bitmap(a, b),
        (Container::Bitmap(a), Container::Array(b)) => difference_bitmap_array(a, b),
        (Container::Run(a), Container::Run(b)) => difference_run_run(a, b),
        (Container::Run(a), Container::Array(b)) => difference_run_array(a, b),
        (Container::Array(a), Container::Run(b)) => difference_array_run(a, b),
        (Container::Run(a), Container::Bitmap(b)) => difference_run_bitmap(a, b),
        (Container::Bitmap(a), Container::Run(b)) => difference_bitmap_run(a, b),
    }
}

/// bitmap × bitmap difference (`a AND NOT b`).
fn difference_bitmap_bitmap(a: &BitmapContainer, b: &BitmapContainer) -> Container {
    let mut out = BitmapContainer::empty();
    let card = bitmap_x_bitmap_andnot_into(a, b, &mut out);
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        Container::Array(ArrayContainer {
            data: out.to_array(),
        })
    } else {
        Container::Bitmap(out)
    }
}

fn bitmap_x_bitmap_andnot_into(
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
                kernels::bitmap_x_bitmap_avx512::andnot_into(&a.words, &b.words, &mut out.words)
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
                kernels::bitmap_x_bitmap_avx2::andnot_into(&a.words, &b.words, &mut out.words)
            };
        }
    }
    #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe {
            kernels::bitmap_x_bitmap_neon::andnot_into(&a.words, &b.words, &mut out.words)
        };
    }
    #[allow(unreachable_code)]
    kernels::bitmap_x_bitmap_scalar::andnot_into(&a.words, &b.words, &mut out.words)
}

fn difference_array_array(a: &ArrayContainer, b: &ArrayContainer) -> Container {
    let mut out = Vec::new();
    kernels::array_x_array_scalar::difference(&a.data, &b.data, &mut out);
    Container::Array(ArrayContainer::from_sorted(out))
}

fn difference_array_bitmap(array: &ArrayContainer, bitmap: &BitmapContainer) -> Container {
    let mut out = Vec::new();
    kernels::array_x_bitmap_scalar::difference_array_bitmap(&array.data, &bitmap.words, &mut out);
    Container::Array(ArrayContainer::from_sorted(out))
}

fn difference_bitmap_array(bitmap: &BitmapContainer, array: &ArrayContainer) -> Container {
    let mut out = BitmapContainer::empty();
    kernels::array_x_bitmap_scalar::difference_bitmap_array(
        &bitmap.words,
        &array.data,
        &mut out.words,
    );
    let card = out.cardinality();
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        Container::Array(ArrayContainer {
            data: out.to_array(),
        })
    } else {
        Container::Bitmap(out)
    }
}

fn difference_run_run(a: &RunContainer, b: &RunContainer) -> Container {
    // Convert to a bitmap, AND-NOT, then optimise back. Run-based
    // difference is irregular enough that the bitmap detour is the
    // cleanest correct path; the cost is one bitmap allocation.
    let mut bm_a = BitmapContainer::empty();
    for &(start, len_m1) in &a.runs {
        let s = u32::from(start);
        let e = s + u32::from(len_m1);
        for v in s..=e {
            bm_a.insert(v as u16);
        }
    }
    let mut bm_b = BitmapContainer::empty();
    for &(start, len_m1) in &b.runs {
        let s = u32::from(start);
        let e = s + u32::from(len_m1);
        for v in s..=e {
            bm_b.insert(v as u16);
        }
    }
    difference_bitmap_bitmap(&bm_a, &bm_b)
}

fn difference_run_array(run: &RunContainer, array: &ArrayContainer) -> Container {
    let mut bm = BitmapContainer::empty();
    for &(start, len_m1) in &run.runs {
        let s = u32::from(start);
        let e = s + u32::from(len_m1);
        for v in s..=e {
            bm.insert(v as u16);
        }
    }
    for &v in &array.data {
        bm.remove(v);
    }
    let card = bm.cardinality();
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        Container::Array(ArrayContainer {
            data: bm.to_array(),
        })
    } else {
        Container::Bitmap(bm)
    }
}

fn difference_array_run(array: &ArrayContainer, run: &RunContainer) -> Container {
    let mut out = Vec::new();
    for &v in &array.data {
        if !run.contains(v) {
            out.push(v);
        }
    }
    Container::Array(ArrayContainer::from_sorted(out))
}

fn difference_run_bitmap(run: &RunContainer, bitmap: &BitmapContainer) -> Container {
    let mut bm = BitmapContainer::empty();
    for &(start, len_m1) in &run.runs {
        let s = u32::from(start);
        let e = s + u32::from(len_m1);
        for v in s..=e {
            bm.insert(v as u16);
        }
    }
    difference_bitmap_bitmap(&bm, bitmap)
}

fn difference_bitmap_run(bitmap: &BitmapContainer, run: &RunContainer) -> Container {
    let mut bm = bitmap.clone();
    for &(start, len_m1) in &run.runs {
        let s = u32::from(start);
        let e = s + u32::from(len_m1);
        for v in s..=e {
            bm.remove(v as u16);
        }
    }
    let card = bm.cardinality();
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        Container::Array(ArrayContainer {
            data: bm.to_array(),
        })
    } else {
        Container::Bitmap(bm)
    }
}

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
    fn bitmap_x_bitmap_difference_correct() {
        let a = fill_bitmap(&[0, 5, 10, 100]);
        let b = fill_bitmap(&[5, 100]);
        let result = difference(&Container::Bitmap(a), &Container::Bitmap(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![0, 10]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn array_x_array_difference_correct() {
        let a = ArrayContainer::from_sorted(vec![1, 2, 3, 4, 5]);
        let b = ArrayContainer::from_sorted(vec![2, 4]);
        let result = difference(&Container::Array(a), &Container::Array(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![1, 3, 5]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn array_x_bitmap_difference_correct() {
        let array = ArrayContainer::from_sorted(vec![1, 5, 10, 100]);
        let bitmap = fill_bitmap(&[5, 100, 200]);
        let result = difference(&Container::Array(array), &Container::Bitmap(bitmap));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![1, 10]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn bitmap_x_array_difference_correct() {
        let bitmap = fill_bitmap(&[1, 5, 10, 100, 200]);
        let array = ArrayContainer::from_sorted(vec![5, 100]);
        let result = difference(&Container::Bitmap(bitmap), &Container::Array(array));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![1, 10, 200]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn run_x_run_difference_correct() {
        let a = RunContainer::from_runs(vec![(0, 9)]);
        // a = 0..=9
        let b = RunContainer::from_runs(vec![(3, 2)]);
        // b = 3..=5
        let result = difference(&Container::Run(a), &Container::Run(b));
        let card = match &result {
            Container::Bitmap(bm) => bm.cardinality(),
            Container::Array(arr) => arr.cardinality(),
            Container::Run(r) => r.cardinality(),
        };
        // 0..=9 \ 3..=5 = {0, 1, 2, 6, 7, 8, 9} = 7 elements.
        assert_eq!(card, 7);
    }
}

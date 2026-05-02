//! Container × container symmetric difference dispatch (`a XOR b`).

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bitmap::containers::{
    ARRAY_MAX_CARDINALITY, ArrayContainer, BITMAP_WORDS, BitmapContainer, Container, RunContainer,
};
use crate::bitmap::kernels;

/// Top-level dispatch for `a XOR b`.
#[must_use]
pub fn symmetric_difference(a: &Container, b: &Container) -> Container {
    match (a, b) {
        (Container::Bitmap(a), Container::Bitmap(b)) => xor_bitmap_bitmap(a, b),
        (Container::Array(a), Container::Array(b)) => xor_array_array(a, b),
        (Container::Array(a), Container::Bitmap(b))
        | (Container::Bitmap(b), Container::Array(a)) => xor_array_bitmap(a, b),
        (Container::Run(a), Container::Run(b)) => xor_run_run(a, b),
        (Container::Run(a), Container::Array(b)) | (Container::Array(b), Container::Run(a)) => {
            xor_run_array(a, b)
        }
        (Container::Run(a), Container::Bitmap(b)) | (Container::Bitmap(b), Container::Run(a)) => {
            xor_run_bitmap(a, b)
        }
    }
}

/// bitmap × bitmap XOR.
fn xor_bitmap_bitmap(a: &BitmapContainer, b: &BitmapContainer) -> Container {
    let mut out = BitmapContainer::empty();
    let card = bitmap_x_bitmap_xor_into(a, b, &mut out);
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        Container::Array(ArrayContainer {
            data: out.to_array(),
        })
    } else {
        Container::Bitmap(out)
    }
}

fn bitmap_x_bitmap_xor_into(
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
                kernels::bitmap_x_bitmap_avx512::xor_into(&a.words, &b.words, &mut out.words)
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
                kernels::bitmap_x_bitmap_avx2::xor_into(&a.words, &b.words, &mut out.words)
            };
        }
    }
    #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe {
            kernels::bitmap_x_bitmap_neon::xor_into(&a.words, &b.words, &mut out.words)
        };
    }
    #[allow(unreachable_code)]
    kernels::bitmap_x_bitmap_scalar::xor_into(&a.words, &b.words, &mut out.words)
}

fn xor_array_array(a: &ArrayContainer, b: &ArrayContainer) -> Container {
    let mut out = Vec::with_capacity(a.data.len() + b.data.len());
    kernels::array_x_array_scalar::symmetric_difference(&a.data, &b.data, &mut out);
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

fn xor_array_bitmap(array: &ArrayContainer, bitmap: &BitmapContainer) -> Container {
    let mut out = BitmapContainer::empty();
    kernels::array_x_bitmap_scalar::xor_array_bitmap(&array.data, &bitmap.words, &mut out.words);
    let card = out.cardinality();
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        Container::Array(ArrayContainer {
            data: out.to_array(),
        })
    } else {
        Container::Bitmap(out)
    }
}

fn xor_run_run(a: &RunContainer, b: &RunContainer) -> Container {
    // Convert to bitmaps; XOR; optimise back.
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
    xor_bitmap_bitmap(&bm_a, &bm_b)
}

fn xor_run_array(run: &RunContainer, array: &ArrayContainer) -> Container {
    let mut bm = BitmapContainer::empty();
    for &(start, len_m1) in &run.runs {
        let s = u32::from(start);
        let e = s + u32::from(len_m1);
        for v in s..=e {
            bm.insert(v as u16);
        }
    }
    for &v in &array.data {
        let v = v as usize;
        bm.words[v >> 6] ^= 1_u64 << (v & 63);
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

fn xor_run_bitmap(run: &RunContainer, bitmap: &BitmapContainer) -> Container {
    let mut bm = bitmap.clone();
    for &(start, len_m1) in &run.runs {
        let s = u32::from(start);
        let e = s + u32::from(len_m1);
        for v in s..=e {
            let vi = v as usize;
            bm.words[vi >> 6] ^= 1_u64 << (vi & 63);
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

    fn fill_bitmap(values: &[u16]) -> BitmapContainer {
        let mut bm = BitmapContainer::empty();
        for &v in values {
            bm.insert(v);
        }
        bm
    }

    #[test]
    fn bitmap_x_bitmap_xor_correct() {
        let a = fill_bitmap(&[0, 5, 10]);
        let b = fill_bitmap(&[5, 10, 100]);
        let result = symmetric_difference(&Container::Bitmap(a), &Container::Bitmap(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![0, 100]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn array_x_array_xor_correct() {
        let a = ArrayContainer::from_sorted(vec![1, 2, 3]);
        let b = ArrayContainer::from_sorted(vec![2, 3, 4]);
        let result = symmetric_difference(&Container::Array(a), &Container::Array(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![1, 4]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn xor_with_self_is_empty() {
        let a = fill_bitmap(&[1, 2, 3, 100, 200]);
        let result = symmetric_difference(&Container::Bitmap(a.clone()), &Container::Bitmap(a));
        let card = match &result {
            Container::Bitmap(bm) => bm.cardinality(),
            Container::Array(arr) => arr.cardinality(),
            Container::Run(r) => r.cardinality(),
        };
        assert_eq!(card, 0);
    }
}

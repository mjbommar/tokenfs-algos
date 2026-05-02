//! Container × container intersect dispatch.
//!
//! Picks the best available kernel per (left container kind, right
//! container kind) pair following the table in
//! `docs/v0.2_planning/11_BITMAP.md` § 2:
//!
//! | pair | strategy |
//! |---|---|
//! | bitmap × bitmap | AVX-512 / AVX2 / NEON 8 KiB AND with VPOPCNTQ cardinality |
//! | array × array | Schlegel SSE4.2 pcmpistrm + galloping-search fallback |
//! | array × bitmap | bit-test loop with AVX2 PSHUFB output materialisation |
//! | run × *  | scalar interval merge |

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::bitmap::containers::{
    ARRAY_MAX_CARDINALITY, ArrayContainer, BITMAP_WORDS, BitmapContainer, Container, RunContainer,
};
use crate::bitmap::kernels;

/// Galloping-search threshold: gallop when `large >= small * GALLOP_FACTOR`.
///
/// Schlegel et al. recommend roughly `4 * log2(large)`; CRoaring uses a
/// factor of about 64. We pick a conservative middle: gallop only when
/// the ratio is at least 64x, where the linear-merge oracle starts
/// losing measurable cycles.
const GALLOP_FACTOR: usize = 64;

/// Top-level dispatch.
#[must_use]
pub fn intersect(a: &Container, b: &Container) -> Container {
    match (a, b) {
        (Container::Bitmap(a), Container::Bitmap(b)) => intersect_bitmap_bitmap(a, b),
        (Container::Array(a), Container::Array(b)) => intersect_array_array(a, b),
        (Container::Array(a), Container::Bitmap(b)) => intersect_array_bitmap(a, b),
        (Container::Bitmap(a), Container::Array(b)) => intersect_array_bitmap(b, a),
        (Container::Run(a), Container::Run(b)) => intersect_run_run(a, b),
        (Container::Run(a), Container::Array(b)) => intersect_run_array(a, b),
        (Container::Array(a), Container::Run(b)) => intersect_run_array(b, a),
        (Container::Run(a), Container::Bitmap(b)) => intersect_run_bitmap(a, b),
        (Container::Bitmap(a), Container::Run(b)) => intersect_run_bitmap(b, a),
    }
}

/// Top-level dispatch — cardinality-only variant.
#[must_use]
pub fn intersect_cardinality(a: &Container, b: &Container) -> u32 {
    match (a, b) {
        (Container::Bitmap(a), Container::Bitmap(b)) => bitmap_x_bitmap_and_card(a, b),
        (Container::Array(a), Container::Array(b)) => array_x_array_card(&a.data, &b.data),
        (Container::Array(a), Container::Bitmap(b)) => array_x_bitmap_card(&a.data, &b.words),
        (Container::Bitmap(a), Container::Array(b)) => array_x_bitmap_card(&b.data, &a.words),
        (Container::Run(a), Container::Run(b)) => intersect_run_run(a, b).cardinality(),
        (Container::Run(a), Container::Array(b)) => intersect_run_array(a, b).cardinality(),
        (Container::Array(a), Container::Run(b)) => intersect_run_array(b, a).cardinality(),
        (Container::Run(a), Container::Bitmap(b)) => intersect_run_bitmap(a, b).cardinality(),
        (Container::Bitmap(a), Container::Run(b)) => intersect_run_bitmap(b, a).cardinality(),
    }
}

/// bitmap × bitmap intersect with `_card` (returns Container).
fn intersect_bitmap_bitmap(a: &BitmapContainer, b: &BitmapContainer) -> Container {
    let mut out = BitmapContainer::empty();
    let card = bitmap_x_bitmap_and_into(a, b, &mut out);
    promote_or_keep(out, card)
}

/// bitmap × bitmap intersect cardinality dispatch.
fn bitmap_x_bitmap_and_card(a: &BitmapContainer, b: &BitmapContainer) -> u32 {
    #[cfg(all(
        feature = "std",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if kernels::bitmap_x_bitmap_avx512::is_available() {
            // SAFETY: availability checked.
            return unsafe { kernels::bitmap_x_bitmap_avx512::and_cardinality(&a.words, &b.words) };
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
            return unsafe { kernels::bitmap_x_bitmap_avx2::and_cardinality(&a.words, &b.words) };
        }
    }
    #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe { kernels::bitmap_x_bitmap_neon::and_cardinality(&a.words, &b.words) };
    }
    #[allow(unreachable_code)]
    kernels::bitmap_x_bitmap_scalar::and_cardinality(&a.words, &b.words)
}

/// bitmap × bitmap intersect into provided output, returning cardinality.
fn bitmap_x_bitmap_and_into(
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
                kernels::bitmap_x_bitmap_avx512::and_into(&a.words, &b.words, &mut out.words)
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
                kernels::bitmap_x_bitmap_avx2::and_into(&a.words, &b.words, &mut out.words)
            };
        }
    }
    #[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
    {
        // SAFETY: NEON is mandatory on AArch64.
        return unsafe {
            kernels::bitmap_x_bitmap_neon::and_into(&a.words, &b.words, &mut out.words)
        };
    }
    #[allow(unreachable_code)]
    kernels::bitmap_x_bitmap_scalar::and_into(&a.words, &b.words, &mut out.words)
}

/// array × array intersect.
fn intersect_array_array(a: &ArrayContainer, b: &ArrayContainer) -> Container {
    let mut out = Vec::new();
    array_x_array_into(&a.data, &b.data, &mut out);
    Container::Array(ArrayContainer::from_sorted(out))
}

/// Array × array intersect dispatch (Schlegel SSE4.2 + galloping fallback).
fn array_x_array_into(a: &[u16], b: &[u16], out: &mut Vec<u16>) {
    let small_len = a.len().min(b.len());
    let large_len = a.len().max(b.len());

    // Galloping when one side is much smaller than the other.
    if small_len > 0 && large_len >= small_len.saturating_mul(GALLOP_FACTOR) {
        let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        kernels::array_x_array_scalar::intersect_galloping(small, large, out);
        return;
    }

    #[cfg(all(
        feature = "std",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if kernels::array_x_array_sse42::is_available() {
            // SAFETY: availability checked.
            unsafe {
                kernels::array_x_array_sse42::intersect(a, b, out);
            }
            return;
        }
    }

    kernels::array_x_array_scalar::intersect(a, b, out);
}

/// Array × array intersect cardinality dispatch.
fn array_x_array_card(a: &[u16], b: &[u16]) -> u32 {
    let small_len = a.len().min(b.len());
    let large_len = a.len().max(b.len());

    if small_len > 0 && large_len >= small_len.saturating_mul(GALLOP_FACTOR) {
        let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let mut out = Vec::with_capacity(small.len());
        kernels::array_x_array_scalar::intersect_galloping(small, large, &mut out);
        return out.len() as u32;
    }

    #[cfg(all(
        feature = "std",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if kernels::array_x_array_sse42::is_available() {
            // SAFETY: availability checked.
            return unsafe { kernels::array_x_array_sse42::intersect_cardinality(a, b) };
        }
    }

    kernels::array_x_array_scalar::intersect_cardinality(a, b)
}

/// Array × bitmap intersect.
fn intersect_array_bitmap(array: &ArrayContainer, bitmap: &BitmapContainer) -> Container {
    let mut out = Vec::new();
    array_x_bitmap_into(&array.data, &bitmap.words, &mut out);
    Container::Array(ArrayContainer::from_sorted(out))
}

/// Array × bitmap intersect dispatch.
///
/// Empirically the scalar bit-test loop ties or beats the AVX2 PSHUFB
/// compress kernel at the array sizes we care about (≤ 4096): the
/// per-chunk PSHUFB + scratch-buffer + push has higher overhead than
/// the unrolled scalar form when most chunks emit only a few survivors.
/// We keep the AVX2 kernel exposed at the [`kernels::array_x_bitmap_avx2`]
/// level for callers that want it, but prefer scalar in dispatch.
/// See `docs/v0.2_planning/11_BITMAP.md` § 5 — the AVX2 win materialises
/// only with the AVX-512 VPCOMPRESSD primitive (deferred to v0.3).
fn array_x_bitmap_into(array: &[u16], bitmap: &[u64; BITMAP_WORDS], out: &mut Vec<u16>) {
    kernels::array_x_bitmap_scalar::intersect_array_bitmap(array, bitmap, out);
}

/// Array × bitmap intersect cardinality dispatch.
fn array_x_bitmap_card(array: &[u16], bitmap: &[u64; BITMAP_WORDS]) -> u32 {
    kernels::array_x_bitmap_scalar::intersect_cardinality_array_bitmap(array, bitmap)
}

/// Run × run intersect via scalar interval walk.
fn intersect_run_run(a: &RunContainer, b: &RunContainer) -> Container {
    let mut out_runs: Vec<(u16, u16)> = Vec::new();
    let mut ia = 0_usize;
    let mut ib = 0_usize;
    while ia < a.runs.len() && ib < b.runs.len() {
        let (start_a, len_a_m1) = a.runs[ia];
        let (start_b, len_b_m1) = b.runs[ib];
        let end_a = u32::from(start_a) + u32::from(len_a_m1);
        let end_b = u32::from(start_b) + u32::from(len_b_m1);
        // Overlap is `[max(start), min(end)]` if non-empty.
        let lo = u32::from(start_a).max(u32::from(start_b));
        let hi = end_a.min(end_b);
        if lo <= hi {
            out_runs.push((lo as u16, (hi - lo) as u16));
        }
        // Advance the run that ends first.
        if end_a < end_b {
            ia += 1;
        } else if end_b < end_a {
            ib += 1;
        } else {
            ia += 1;
            ib += 1;
        }
    }
    let rc = RunContainer::from_runs(out_runs);
    let card = rc.cardinality();
    // After intersection the result may be sparse enough to convert to
    // an array, or dense enough to convert to a bitmap.
    optimise_run(rc, card)
}

/// Run × array intersect via scalar interval walk.
fn intersect_run_array(run: &RunContainer, array: &ArrayContainer) -> Container {
    let mut out: Vec<u16> = Vec::new();
    let mut ri = 0_usize;
    for &v in &array.data {
        // Advance through runs whose end is below `v`.
        while ri < run.runs.len() {
            let (start, len_m1) = run.runs[ri];
            let end = u32::from(start) + u32::from(len_m1);
            if end < u32::from(v) {
                ri += 1;
                continue;
            }
            if u32::from(start) > u32::from(v) {
                break;
            }
            // start ≤ v ≤ end → present.
            out.push(v);
            break;
        }
    }
    let card = out.len() as u32;
    let array = ArrayContainer::from_sorted(out);
    promote_or_keep_array(array, card)
}

/// Run × bitmap intersect via scalar walk over runs + per-bit test.
fn intersect_run_bitmap(run: &RunContainer, bitmap: &BitmapContainer) -> Container {
    let mut out_bm = BitmapContainer::empty();
    for &(start, len_m1) in &run.runs {
        let start = u32::from(start);
        let end = start + u32::from(len_m1);
        for v in start..=end {
            let v = v as u16;
            let vi = v as usize;
            if (bitmap.words[vi >> 6] >> (vi & 63)) & 1 == 1 {
                out_bm.insert(v);
            }
        }
    }
    let card = out_bm.cardinality();
    promote_or_keep(out_bm, card)
}

/// Convert a populated bitmap container to an array if it is sparse enough.
fn promote_or_keep(bm: BitmapContainer, card: u32) -> Container {
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        Container::Array(ArrayContainer {
            data: bm.to_array(),
        })
    } else {
        Container::Bitmap(bm)
    }
}

/// Convert a populated array container to a bitmap if it overflowed.
fn promote_or_keep_array(arr: ArrayContainer, card: u32) -> Container {
    if (card as usize) > ARRAY_MAX_CARDINALITY {
        let mut bm = BitmapContainer::empty();
        for &v in &arr.data {
            bm.insert(v);
        }
        Container::Bitmap(bm)
    } else {
        Container::Array(arr)
    }
}

/// Run-container post-intersect: convert to array if sparse, bitmap if
/// dense, else keep as a run container.
fn optimise_run(rc: RunContainer, card: u32) -> Container {
    if (card as usize) <= ARRAY_MAX_CARDINALITY {
        let mut data = Vec::with_capacity(card as usize);
        for &(start, len_m1) in &rc.runs {
            let start = u32::from(start);
            let end = start + u32::from(len_m1);
            for v in start..=end {
                data.push(v as u16);
            }
        }
        Container::Array(ArrayContainer { data })
    } else {
        Container::Run(rc)
    }
}

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
    fn bitmap_x_bitmap_intersect_correct() {
        let a = fill_bitmap(&[0, 5, 10, 100, 4096, 65535]);
        let b = fill_bitmap(&[5, 10, 200, 4096]);
        let result = intersect(&Container::Bitmap(a), &Container::Bitmap(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![5, 10, 4096]),
            other => panic!("expected sparse array, got {other:?}"),
        }
    }

    #[test]
    fn array_x_array_intersect_correct() {
        let a = ArrayContainer::from_sorted(vec![1, 3, 5, 7, 9]);
        let b = ArrayContainer::from_sorted(vec![2, 3, 5, 8, 10]);
        let result = intersect(&Container::Array(a), &Container::Array(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![3, 5]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn array_x_bitmap_intersect_correct() {
        let a = ArrayContainer::from_sorted(vec![1, 5, 10, 100]);
        let b = fill_bitmap(&[5, 10, 100, 200]);
        let result = intersect(&Container::Array(a), &Container::Bitmap(b));
        match result {
            Container::Array(arr) => assert_eq!(arr.data, vec![5, 10, 100]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn run_x_run_intersect_correct() {
        let a = RunContainer::from_runs(vec![(0, 9), (100, 9)]);
        // a covers 0..=9, 100..=109.
        let b = RunContainer::from_runs(vec![(5, 9), (105, 9)]);
        // b covers 5..=14, 105..=114.
        let result = intersect(&Container::Run(a), &Container::Run(b));
        // Intersection: 5..=9 (5 vals), 105..=109 (5 vals) = 10 elements.
        let card = match &result {
            Container::Array(a) => a.cardinality(),
            Container::Bitmap(b) => b.cardinality(),
            Container::Run(r) => r.cardinality(),
        };
        assert_eq!(card, 10);
    }

    #[test]
    fn run_x_array_intersect_correct() {
        let run = RunContainer::from_runs(vec![(10, 9), (100, 0)]);
        // run covers 10..=19, 100..=100.
        let array = ArrayContainer::from_sorted(vec![5, 12, 15, 100, 200]);
        let result = intersect(&Container::Run(run), &Container::Array(array));
        match result {
            Container::Array(a) => assert_eq!(a.data, vec![12, 15, 100]),
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn intersect_cardinality_matches_full_intersect_card() {
        let a = fill_bitmap(&(0..1000).step_by(7).map(|i| i as u16).collect::<Vec<_>>());
        let b = fill_bitmap(&(0..1000).step_by(11).map(|i| i as u16).collect::<Vec<_>>());
        let result = intersect(&Container::Bitmap(a.clone()), &Container::Bitmap(b.clone()));
        let card = match &result {
            Container::Bitmap(bm) => bm.cardinality(),
            Container::Array(arr) => arr.cardinality(),
            Container::Run(r) => r.cardinality(),
        };
        let card_only = intersect_cardinality(&Container::Bitmap(a), &Container::Bitmap(b));
        assert_eq!(card, card_only);
    }

    #[test]
    fn promote_keeps_dense_as_bitmap() {
        // A bitmap with 5000 set bits should stay as a bitmap container.
        let mut bm = BitmapContainer::empty();
        for v in 0..5000 {
            bm.insert(v as u16);
        }
        let card = bm.cardinality();
        let result = promote_or_keep(bm, card);
        assert!(matches!(result, Container::Bitmap(_)));
    }

    #[test]
    fn promote_converts_sparse_to_array() {
        let bm = fill_bitmap(&[0, 100, 65535]);
        let card = bm.cardinality();
        let result = promote_or_keep(bm, card);
        assert!(matches!(result, Container::Array(_)));
    }

    #[test]
    fn galloping_path_engages_when_imbalanced() {
        // Adversarial: 5 needles vs 5000 hay values — well past the
        // GALLOP_FACTOR threshold.
        let small: Vec<u16> = (0..5).map(|i| i * 1000).collect();
        let large: Vec<u16> = (0..5000).map(|i| i as u16).collect();
        let mut out = Vec::new();
        array_x_array_into(&small, &large, &mut out);
        // Each needle (0, 1000, 2000, 3000, 4000) is in the large set.
        assert_eq!(out, small);
    }
}

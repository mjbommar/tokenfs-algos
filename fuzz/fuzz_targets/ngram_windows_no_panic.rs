//! Fuzz target: NGramWindows iterator must not panic and must report the
//! correct number of windows.

#![no_main]

use libfuzzer_sys::fuzz_target;
use tokenfs_algos::windows::ngrams;

fuzz_target!(|data: &[u8]| {
    // N=1, 2, 4, 8 — the four canonical n-gram widths.
    let count1 = ngrams::<1>(data).count();
    assert_eq!(count1, data.len(), "N=1 window count");

    let count2 = ngrams::<2>(data).count();
    let expected2 = data.len().saturating_sub(1);
    assert_eq!(count2, expected2, "N=2 window count");

    let count4 = ngrams::<4>(data).count();
    let expected4 = data.len().saturating_sub(3);
    assert_eq!(count4, expected4, "N=4 window count");

    let count8 = ngrams::<8>(data).count();
    let expected8 = data.len().saturating_sub(7);
    assert_eq!(count8, expected8, "N=8 window count");

    // Each emitted window must equal the slice it claims to cover.
    if data.len() >= 4 {
        for (offset, window) in ngrams::<4>(data).enumerate() {
            assert_eq!(&window[..], &data[offset..offset + 4]);
            if offset > 256 {
                break; // Cap iterations on large inputs.
            }
        }
    }
});

#![allow(missing_docs)]
#![cfg(feature = "bench-internals")]

use proptest::prelude::*;
use tokenfs_algos::histogram::{
    ByteHistogram,
    bench_internals::{HistogramKernel, byte_histogram_with_kernel},
};

fn synthetic_cases() -> Vec<Vec<u8>> {
    let mut cases = vec![
        Vec::new(),
        vec![0],
        vec![0; 4096],
        vec![0xff; 4096],
        (0_u8..=255).collect(),
        (0_u8..=255).cycle().take(4096).collect(),
    ];

    for len in [
        1_usize, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257,
        4095, 4096, 4097,
    ] {
        cases.push(
            (0..len)
                .map(|i| (i.wrapping_mul(17) ^ (i >> 2).wrapping_mul(31)) as u8)
                .collect(),
        );
    }

    cases.push(run_heavy(8192));
    cases.push(repeated_text(8192));
    cases
}

#[test]
fn experimental_kernels_match_public_reference_on_synthetic_cases() {
    for bytes in synthetic_cases() {
        let expected = ByteHistogram::from_block(&bytes);

        for kernel in HistogramKernel::all() {
            let actual = byte_histogram_with_kernel(&bytes, kernel);
            assert_eq!(
                actual,
                expected,
                "kernel {kernel} diverged on length {}",
                bytes.len()
            );
        }
    }
}

#[test]
fn experimental_kernels_match_public_reference_on_unaligned_subslices() {
    let bytes = (0_usize..8192)
        .map(|i| (i.wrapping_mul(29) ^ (i >> 3).wrapping_mul(37)) as u8)
        .collect::<Vec<_>>();

    for start in 0..128 {
        for len in [0_usize, 1, 15, 16, 17, 31, 32, 33, 255, 256, 257, 1023] {
            let end = (start + len).min(bytes.len());
            let slice = &bytes[start..end];
            let expected = ByteHistogram::from_block(slice);

            for kernel in HistogramKernel::all() {
                let actual = byte_histogram_with_kernel(slice, kernel);
                assert_eq!(
                    actual,
                    expected,
                    "kernel {kernel} diverged on start {start}, length {}",
                    slice.len()
                );
            }
        }
    }
}

proptest! {
    #[test]
    fn experimental_kernels_match_public_reference_for_random_inputs(
        bytes in proptest::collection::vec(any::<u8>(), 0..16384)
    ) {
        let expected = ByteHistogram::from_block(&bytes);

        for kernel in HistogramKernel::all() {
            prop_assert_eq!(&byte_histogram_with_kernel(&bytes, kernel), &expected);
        }
    }
}

fn run_heavy(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let pattern = [(0, 64_usize), (0xff, 31), (b'A', 17), (b'\n', 1), (0x7f, 9)];

    while out.len() < size {
        for (byte, run) in pattern {
            let take = (size - out.len()).min(run);
            out.extend(std::iter::repeat_n(byte, take));
            if out.len() == size {
                break;
            }
        }
    }

    out
}

fn repeated_text(size: usize) -> Vec<u8> {
    const TEXT: &[u8] = b"tokenfs-algos measures byte streams: ascii text, paths, symbols, and structured records.\n";
    TEXT.iter().copied().cycle().take(size).collect()
}

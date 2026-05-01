//! Public histogram planner and pinned-kernel API tests.

use tokenfs_algos::{
    dispatch::{HistogramStrategy, ProcessorProfile},
    histogram::{self, kernels},
};

#[test]
fn planned_histogram_matches_reference_counts() {
    for input in cases() {
        let histogram = histogram::block(&input);
        assert_eq!(histogram.counts(), &reference_counts(&input));
        assert_eq!(histogram.total(), input.len() as u64);
    }
}

#[test]
fn explain_block_returns_the_plan_used() {
    let bytes = vec![0_u8; 1024 * 1024];
    let profile = ProcessorProfile::portable();
    let explained = histogram::explain_block(&bytes, &profile);

    assert_eq!(
        explained.plan.strategy,
        HistogramStrategy::AdaptiveLowEntropyFast
    );
    assert_eq!(
        explained.signals.entropy,
        tokenfs_algos::dispatch::EntropyClass::Low
    );
    assert_eq!(explained.signals.top_ratio_q8(), 255);
    assert_eq!(explained.histogram.counts(), &reference_counts(&bytes));
}

#[test]
fn pinned_kernels_match_reference_counts() {
    let input = mixed_case(128 * 1024);
    let reference = reference_counts(&input);

    let histograms = [
        kernels::direct_u64::block(&input),
        kernels::local_u32::block(&input),
        kernels::stripe4_u32::block(&input),
        kernels::stripe8_u32::block(&input),
        kernels::run_length_u64::block(&input),
        #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
        kernels::avx2_palette_u32::block(&input),
        kernels::adaptive_prefix_1k::block(&input),
        kernels::adaptive_prefix_4k::block(&input),
        kernels::adaptive_spread_4k::block(&input),
        kernels::adaptive_run_sentinel_4k::block(&input),
        kernels::adaptive_chunked_64k::block(&input),
        kernels::adaptive_sequential_online_64k::block(&input),
        kernels::adaptive_file_cached_64k::block(&input),
        kernels::adaptive_low_entropy_fast::block(&input),
        kernels::adaptive_ascii_fast::block(&input),
        kernels::adaptive_high_entropy_skip::block(&input),
        kernels::adaptive_meso_detector::block(&input),
    ];

    for histogram in histograms {
        assert_eq!(histogram.counts(), &reference);
        assert_eq!(histogram.total(), input.len() as u64);
    }
}

fn cases() -> [Vec<u8>; 5] {
    [
        Vec::new(),
        b"abbccc".to_vec(),
        vec![0; 4096],
        (0_u8..=255).cycle().take(16 * 1024).collect(),
        mixed_case(64 * 1024),
    ]
}

fn mixed_case(size: usize) -> Vec<u8> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut bytes = Vec::with_capacity(size);
    while bytes.len() < size {
        let take = (size - bytes.len()).min(1024);
        match (bytes.len() / 1024) % 4 {
            0 => bytes.extend(std::iter::repeat_n(0, take)),
            1 => {
                for _ in 0..take {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    bytes.push(state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8);
                }
            }
            2 => bytes.extend(
                b"tokenfs-algos histogram planner\n"
                    .iter()
                    .copied()
                    .cycle()
                    .take(take),
            ),
            _ => bytes.extend((0_u8..=15).cycle().take(take)),
        }
    }
    bytes
}

fn reference_counts(bytes: &[u8]) -> [u64; 256] {
    let mut counts = [0_u64; 256];
    for &byte in bytes {
        counts[byte as usize] += 1;
    }
    counts
}

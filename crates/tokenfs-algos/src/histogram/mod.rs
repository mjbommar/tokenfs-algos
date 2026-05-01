//! Histograms over byte and n-gram distributions.

pub mod byte;
pub mod kernels;
pub mod ngram;
pub mod pair;

#[cfg(feature = "bench-internals")]
pub mod bench_internals;

use crate::dispatch::{
    ApiContext, ContentKind, EntropyClass, EntropyScale, HistogramPlan, HistogramStrategy,
    ProcessorProfile, WorkloadShape, plan_histogram,
};

pub use byte::ByteHistogram;
pub use pair::BytePairHistogram;

/// Result of a planned histogram call with the selected plan attached.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedByteHistogram {
    /// Histogram result.
    pub histogram: ByteHistogram,
    /// Plan used to produce the result.
    pub plan: HistogramPlan,
    /// Input signals sampled by the planner.
    pub signals: HistogramBlockSignals,
}

/// Cheap block-level signals used by the histogram planner.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HistogramBlockSignals {
    /// Number of sampled bytes.
    pub sample_bytes: usize,
    /// Distinct byte values observed in the sample.
    pub distinct_bytes: u16,
    /// Count of the most frequent sampled byte.
    pub top_count: u32,
    /// Count of sampled bytes that look like ASCII text.
    pub ascii_text_bytes: u32,
    /// Number of adjacent equal-byte pairs in the sample.
    pub adjacent_equal_pairs: u32,
    /// Longest equal-byte run in the sample.
    pub longest_run: u32,
    /// Coarse content family inferred from the sample.
    pub content: ContentKind,
    /// Coarse entropy class inferred from the sample.
    pub entropy: EntropyClass,
    /// Entropy scale inferred from the input size and sample.
    pub scale: EntropyScale,
}

impl HistogramBlockSignals {
    /// ASCII/text ratio on a 0..=255 scale.
    #[must_use]
    pub fn ascii_ratio_q8(self) -> u8 {
        ratio_q8(self.ascii_text_bytes, self.sample_bytes)
    }

    /// Largest-symbol ratio on a 0..=255 scale.
    #[must_use]
    pub fn top_ratio_q8(self) -> u8 {
        ratio_q8(self.top_count, self.sample_bytes)
    }

    /// Adjacent-equality ratio on a 0..=255 scale.
    #[must_use]
    pub fn adjacent_equal_ratio_q8(self) -> u8 {
        ratio_q8(
            self.adjacent_equal_pairs,
            self.sample_bytes.saturating_sub(1),
        )
    }
}

/// Builds a byte histogram using the public planner.
///
/// This is the ergonomic default. It chooses a strategy from the current
/// processor profile and block-shaped workload metadata. Users who need pinned
/// reproducibility can call [`kernels`] directly.
#[must_use]
pub fn block(bytes: &[u8]) -> ByteHistogram {
    let profile = ProcessorProfile::detect();
    block_with_profile(bytes, &profile)
}

/// Builds a byte histogram using an explicit processor profile.
#[must_use]
pub fn block_with_profile(bytes: &[u8], profile: &ProcessorProfile) -> ByteHistogram {
    let plan = plan_block(bytes, profile);
    block_with_plan(bytes, &plan)
}

/// Builds a byte histogram and returns the plan used.
#[must_use]
pub fn explain_block(bytes: &[u8], profile: &ProcessorProfile) -> PlannedByteHistogram {
    let signals = block_signals(bytes);
    let plan = plan_block_with_signals(bytes.len(), &signals, profile);
    let histogram = block_with_plan(bytes, &plan);
    PlannedByteHistogram {
        histogram,
        plan,
        signals,
    }
}

/// Plans a block histogram without running it.
#[must_use]
pub fn plan_block(bytes: &[u8], profile: &ProcessorProfile) -> HistogramPlan {
    let signals = block_signals(bytes);
    plan_block_with_signals(bytes.len(), &signals, profile)
}

/// Computes the cheap block signals used by [`plan_block`].
#[must_use]
pub fn block_signals(bytes: &[u8]) -> HistogramBlockSignals {
    classify_block_sample(bytes)
}

fn plan_block_with_signals(
    len: usize,
    signals: &HistogramBlockSignals,
    profile: &ProcessorProfile,
) -> HistogramPlan {
    let mut workload = WorkloadShape::new(ApiContext::Block, len);
    workload.content = signals.content;
    workload.entropy = signals.entropy;
    workload.scale = signals.scale;
    plan_histogram(profile, &workload)
}

/// Builds a byte histogram using an already computed plan.
#[must_use]
pub fn block_with_plan(bytes: &[u8], plan: &HistogramPlan) -> ByteHistogram {
    match plan.strategy {
        HistogramStrategy::DirectU64 => kernels::direct_u64::block(bytes),
        HistogramStrategy::LocalU32 => kernels::local_u32::block(bytes),
        HistogramStrategy::Stripe4U32 => kernels::stripe4_u32::block(bytes),
        HistogramStrategy::Stripe8U32 => kernels::stripe8_u32::block(bytes),
        HistogramStrategy::RunLengthU64 => kernels::run_length_u64::block(bytes),
        HistogramStrategy::Avx2Stripe4U32 => {
            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                kernels::avx2_stripe4_u32::block(bytes)
            }
            #[cfg(not(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64"))))]
            {
                kernels::stripe4_u32::block(bytes)
            }
        }
        HistogramStrategy::Avx2PaletteU32 => {
            #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                kernels::avx2_palette_u32::block(bytes)
            }
            #[cfg(not(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64"))))]
            {
                kernels::local_u32::block(bytes)
            }
        }
        HistogramStrategy::AdaptivePrefix1K => kernels::adaptive_prefix_1k::block(bytes),
        HistogramStrategy::AdaptivePrefix4K => kernels::adaptive_prefix_4k::block(bytes),
        HistogramStrategy::AdaptiveSpread4K => kernels::adaptive_spread_4k::block(bytes),
        HistogramStrategy::AdaptiveRunSentinel4K => kernels::adaptive_run_sentinel_4k::block(bytes),
        HistogramStrategy::AdaptiveLowEntropyFast => {
            kernels::adaptive_low_entropy_fast::block(bytes)
        }
        HistogramStrategy::AdaptiveAsciiFast => kernels::adaptive_ascii_fast::block(bytes),
        HistogramStrategy::AdaptiveHighEntropySkip => {
            kernels::adaptive_high_entropy_skip::block(bytes)
        }
        HistogramStrategy::AdaptiveMesoDetector => kernels::adaptive_meso_detector::block(bytes),
        HistogramStrategy::AdaptiveChunked64K => kernels::adaptive_chunked_64k::block(bytes),
        HistogramStrategy::AdaptiveSequentialOnline64K | HistogramStrategy::StatefulSequential => {
            kernels::adaptive_sequential_online_64k::block(bytes)
        }
        HistogramStrategy::AdaptiveFileCached64K => kernels::adaptive_file_cached_64k::block(bytes),
    }
}

fn classify_block_sample(bytes: &[u8]) -> HistogramBlockSignals {
    let sample_len = bytes.len().min(4 * 1024);
    if sample_len == 0 {
        return HistogramBlockSignals {
            sample_bytes: 0,
            distinct_bytes: 0,
            top_count: 0,
            ascii_text_bytes: 0,
            adjacent_equal_pairs: 0,
            longest_run: 0,
            content: ContentKind::Unknown,
            entropy: EntropyClass::Unknown,
            scale: EntropyScale::Unknown,
        };
    }

    let sample = &bytes[..sample_len];
    let mut counts = [0_usize; 256];
    let mut ascii_text = 0_usize;
    let mut longest_run = 0_usize;
    let mut current_run = 0_usize;
    let mut adjacent_equal = 0_usize;
    let mut previous = None;

    for &byte in sample {
        counts[byte as usize] += 1;
        if byte.is_ascii_graphic() || matches!(byte, b' ' | b'\n' | b'\r' | b'\t') {
            ascii_text += 1;
        }

        if Some(byte) == previous {
            current_run += 1;
            adjacent_equal += 1;
        } else {
            current_run = 1;
            previous = Some(byte);
        }
        longest_run = longest_run.max(current_run);
    }

    let unique = counts.iter().filter(|&&count| count > 0).count();
    let max_count = counts.iter().copied().max().unwrap_or(0);

    let content = if ascii_text * 100 >= sample_len * 90 {
        ContentKind::Text
    } else {
        ContentKind::Binary
    };

    let entropy =
        if unique <= 4 || max_count * 100 >= sample_len * 80 || longest_run >= sample_len / 2 {
            EntropyClass::Low
        } else if unique >= 192 && max_count * 100 < sample_len * 4 {
            EntropyClass::High
        } else {
            EntropyClass::Medium
        };

    let scale = if bytes.len() >= 64 * 1024 {
        EntropyScale::Flat
    } else {
        EntropyScale::Unknown
    };

    HistogramBlockSignals {
        sample_bytes: sample_len,
        distinct_bytes: unique as u16,
        top_count: max_count as u32,
        ascii_text_bytes: ascii_text as u32,
        adjacent_equal_pairs: adjacent_equal as u32,
        longest_run: longest_run as u32,
        content,
        entropy,
        scale,
    }
}

fn ratio_q8(count: u32, total: usize) -> u8 {
    if total == 0 {
        return 0;
    }
    ((u128::from(count) * 255) / total as u128).min(255) as u8
}

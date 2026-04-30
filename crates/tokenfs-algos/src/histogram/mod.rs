//! Histograms over byte and n-gram distributions.

pub mod byte;
pub mod kernels;

#[cfg(feature = "bench-internals")]
pub mod bench_internals;

use crate::dispatch::{
    ApiContext, ContentKind, EntropyClass, EntropyScale, HistogramPlan, HistogramStrategy,
    ProcessorProfile, WorkloadShape, plan_histogram,
};

pub use byte::ByteHistogram;

/// Result of a planned histogram call with the selected plan attached.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlannedByteHistogram {
    /// Histogram result.
    pub histogram: ByteHistogram,
    /// Plan used to produce the result.
    pub plan: HistogramPlan,
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
    let plan = plan_block(bytes, profile);
    let histogram = block_with_plan(bytes, &plan);
    PlannedByteHistogram { histogram, plan }
}

/// Plans a block histogram without running it.
#[must_use]
pub fn plan_block(bytes: &[u8], profile: &ProcessorProfile) -> HistogramPlan {
    let mut workload = WorkloadShape::new(ApiContext::Block, bytes.len());
    workload.content = ContentKind::Unknown;
    workload.entropy = EntropyClass::Unknown;
    workload.scale = EntropyScale::Unknown;
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
        HistogramStrategy::AdaptivePrefix1K => kernels::adaptive_prefix_1k::block(bytes),
        HistogramStrategy::AdaptivePrefix4K => kernels::adaptive_prefix_4k::block(bytes),
        HistogramStrategy::AdaptiveSpread4K => kernels::adaptive_spread_4k::block(bytes),
        HistogramStrategy::AdaptiveRunSentinel4K => kernels::adaptive_run_sentinel_4k::block(bytes),
        HistogramStrategy::AdaptiveChunked64K | HistogramStrategy::StatefulSequential => {
            kernels::adaptive_chunked_64k::block(bytes)
        }
    }
}

//! Convenient imports for common `tokenfs-algos` APIs.

pub use crate::dispatch::{
    ApiContext, Backend, CacheProfile, CacheState, ContentKind, EntropyClass, EntropyScale,
    HistogramKernelInfo, HistogramPlan, HistogramStrategy, KernelIsa, KernelStatefulness,
    PlanContext, PrimitiveFamily, ProcessorProfile, ReadPattern, SourceHint, WorkingSetClass,
    WorkloadShape, clear_forced_backend, detected_backend, detected_cache_profile,
    detected_processor_profile, force_backend, histogram_kernel_catalog, plan_histogram,
};
pub use crate::entropy;
pub use crate::histogram::{
    ByteHistogram, PlannedByteHistogram, block as histogram_block,
    block_with_profile as histogram_block_with_profile, explain_block as explain_histogram_block,
    plan_block as plan_histogram_block,
};

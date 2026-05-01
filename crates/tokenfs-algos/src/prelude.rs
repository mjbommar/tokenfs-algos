//! Convenient imports for common `tokenfs-algos` APIs.

pub use crate::dispatch::{
    ApiContext, Backend, CacheProfile, CacheState, ContentKind, EntropyClass, EntropyScale,
    HistogramKernelInfo, HistogramPlan, HistogramStrategy, KernelIsa, KernelStatefulness,
    PlanContext, PlannerConfidenceSource, PrimitiveFamily, ProcessorProfile, ReadPattern,
    SourceHint, WorkingSetClass, WorkloadShape, clear_forced_backend, detected_backend,
    detected_cache_profile, detected_processor_profile, force_backend, histogram_kernel_catalog,
    plan_histogram,
};
pub use crate::distribution::{
    ByteDistribution, ByteDistributionDistances, ByteDistributionMetric, ByteDistributionReference,
    NearestByteDistribution, nearest_byte_distribution,
};
pub use crate::entropy;
pub use crate::fingerprint::{
    BLOCK_SIZE as FINGERPRINT_BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, FingerprintKernel,
    FingerprintKernelInfo, block as fingerprint_block, extent as fingerprint_extent,
    kernel_catalog as fingerprint_kernel_catalog,
};
pub use crate::histogram::{
    ByteHistogram, HistogramBlockSignals, PlannedByteHistogram, block as histogram_block,
    block_with_profile as histogram_block_with_profile, explain_block as explain_histogram_block,
    plan_block as plan_histogram_block,
};
pub use crate::selector::{RepresentationHint, SelectorSignals};
pub use crate::sketch::{CLog2Lut, CountMinSketch, HashBinSketch, MisraGries};
pub use crate::structure::StructureSummary;

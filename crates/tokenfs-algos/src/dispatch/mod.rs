//! Runtime backend detection, kernel catalogs, and planner support.

use core::sync::atomic::{AtomicU8, Ordering};

const AUTO_BACKEND: u8 = u8::MAX;

static FORCED_BACKEND: AtomicU8 = AtomicU8::new(AUTO_BACKEND);

/// Compute backend selected for runtime-dispatched kernels.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Backend {
    /// AVX-512 backend.
    Avx512 = 1,
    /// AVX2 backend.
    Avx2 = 2,
    /// Arm SVE2 backend.
    Sve2 = 3,
    /// Arm SVE backend.
    Sve = 4,
    /// Arm NEON backend.
    Neon = 5,
    /// Portable scalar backend.
    Scalar = 6,
}

/// Basic cache facts used by planner heuristics.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct CacheProfile {
    /// Cache-line size in bytes.
    pub line_bytes: Option<usize>,
    /// L1 data-cache size in bytes.
    pub l1d_bytes: Option<usize>,
    /// Private or nearest L2-cache size in bytes.
    pub l2_bytes: Option<usize>,
    /// Last-level cache size in bytes.
    pub l3_bytes: Option<usize>,
}

impl CacheProfile {
    /// Returns a cache profile with no known cache sizes.
    #[must_use]
    pub const fn unknown() -> Self {
        Self {
            line_bytes: None,
            l1d_bytes: None,
            l2_bytes: None,
            l3_bytes: None,
        }
    }
}

/// Processor facts used by runtime planning.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ProcessorProfile {
    /// Runtime compute backend selected from instruction-set detection.
    pub backend: Backend,
    /// Known cache facts.
    pub cache: CacheProfile,
    /// Logical CPU count when available.
    pub logical_cpus: Option<usize>,
}

impl ProcessorProfile {
    /// Builds a portable scalar processor profile.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            backend: Backend::Scalar,
            cache: CacheProfile::unknown(),
            logical_cpus: None,
        }
    }

    /// Detects a processor profile for the current process.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            backend: detected_backend(),
            cache: detected_cache_profile(),
            logical_cpus: detect_logical_cpus(),
        }
    }
}

/// Primitive family for cataloged kernels.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PrimitiveFamily {
    /// Byte histogram kernels.
    ByteHistogram,
    /// F22/content fingerprint kernels.
    Fingerprint,
    /// Sketch and approximate counting kernels.
    Sketch,
    /// Byte classification kernels.
    ByteClass,
    /// Run-length and structural detectors.
    Structure,
    /// Entropy reduction kernels.
    Entropy,
    /// Selector feature extraction kernels.
    Selector,
}

/// ISA or backend class required by a cataloged kernel.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelIsa {
    /// Runtime-dispatched kernel; inspect backend/catalog docs for candidates.
    RuntimeDispatch,
    /// Portable scalar kernel.
    PortableScalar,
    /// Portable scalar kernel with manual unrolling/chunking.
    PortableScalarChunked,
    /// x86 SSE2 kernel.
    X86Sse2,
    /// x86 SSSE3 kernel.
    X86Ssse3,
    /// x86 SSE4.2 kernel.
    X86Sse42,
    /// x86 AVX2 kernel.
    X86Avx2,
    /// x86 AVX-512 kernel.
    X86Avx512,
    /// AArch64 NEON kernel.
    Aarch64Neon,
    /// AArch64 SVE kernel.
    Aarch64Sve,
    /// AArch64 SVE2 kernel.
    Aarch64Sve2,
}

/// Approximate working-set class for planner and benchmark interpretation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WorkingSetClass {
    /// Tiny fixed-overhead path.
    Tiny,
    /// Fits comfortably in private L1 data cache.
    L1,
    /// Fits better as an L2-sized chunk.
    L2,
    /// Streaming over inputs larger than private caches.
    Streaming,
    /// Designed for private per-thread state and reduction.
    Parallel,
}

/// Whether a kernel keeps state across calls.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelStatefulness {
    /// No state is retained across calls.
    Stateless,
    /// Reuses state or decisions across sequential calls.
    Stateful,
    /// Applies independent plans to chunks before reducing.
    Chunked,
}

/// Catalog metadata for one histogram kernel.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HistogramKernelInfo {
    /// Primitive family.
    pub family: PrimitiveFamily,
    /// Stable strategy identifier.
    pub strategy: HistogramStrategy,
    /// ISA or backend class required to run the kernel.
    pub isa: KernelIsa,
    /// Approximate working-set class.
    pub working_set: WorkingSetClass,
    /// State model.
    pub statefulness: KernelStatefulness,
    /// Minimum input size where this kernel is worth considering.
    pub min_bytes: usize,
    /// Preferred chunk size in bytes. Zero means caller chunking is retained.
    pub preferred_chunk_bytes: usize,
    /// Classifier sample size in bytes.
    pub sample_bytes: usize,
    /// Private table bytes touched by the core loop, excluding public output.
    pub private_table_bytes: usize,
    /// Short planner-facing description.
    pub description: &'static str,
}

/// High-level API context for a primitive call.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ApiContext {
    /// One whole in-memory block.
    Block,
    /// Whole file or large extent.
    File,
    /// Sequential read stream.
    Sequential,
    /// Random-access reads.
    Random,
    /// Parallel file or extent scan.
    Parallel,
}

/// More specific read/access pattern supplied to the planner.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReadPattern {
    /// One whole in-memory block.
    WholeBlock,
    /// Sequential reads without explicit readahead.
    Sequential,
    /// Sequential reads with caller/file-system readahead.
    Readahead,
    /// Random reads with non-trivial chunks.
    Random,
    /// Random single-byte or tiny reads.
    RandomTiny,
    /// Hot/cold random reads with a skewed access distribution.
    ZipfianHotCold,
    /// Repeated reads from hot cache.
    HotRepeat,
    /// Large scans intended to defeat cache reuse.
    ColdSweep,
    /// Same file or region repeatedly planned across calls.
    SameFileRepeat,
    /// Sequential chunks processed by multiple workers.
    ParallelSequential,
}

impl ReadPattern {
    /// Returns the default read pattern implied by a high-level API context.
    #[must_use]
    pub const fn from_context(context: ApiContext) -> Self {
        match context {
            ApiContext::Block => Self::WholeBlock,
            ApiContext::File | ApiContext::Sequential => Self::Sequential,
            ApiContext::Random => Self::Random,
            ApiContext::Parallel => Self::ParallelSequential,
        }
    }
}

/// Coarse content family used by dispatch planning.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ContentKind {
    /// Content family is not known.
    Unknown,
    /// Mostly text.
    Text,
    /// Mostly binary.
    Binary,
    /// Mixed text and binary regions.
    Mixed,
}

/// Coarse entropy class used by dispatch planning.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EntropyClass {
    /// Entropy class is not known.
    Unknown,
    /// Low entropy.
    Low,
    /// Medium entropy.
    Medium,
    /// High entropy.
    High,
    /// Mixed entropy across regions.
    Mixed,
}

/// Entropy scale used by dispatch planning.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EntropyScale {
    /// Entropy scale is not known.
    Unknown,
    /// Roughly uniform across the full input.
    Flat,
    /// Fine-grained repetition or structure.
    Micro,
    /// Block-sized structure changes.
    Meso,
    /// Large region-sized structure changes.
    Macro,
}

/// Expected cache state for this call or stream.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum CacheState {
    /// Cache behavior is not known.
    Unknown,
    /// Caller expects a hot-cache repeat.
    Hot,
    /// Caller expects cold-ish scan behavior.
    Cold,
    /// Caller expects repeated calls over the same file or region.
    Reused,
}

/// Source/data hint supplied to the planner.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SourceHint {
    /// Source is not known.
    Unknown,
    /// Synthetic/generated benchmark data.
    Synthetic,
    /// Real file or file slice.
    RealFile,
    /// Parsed paper/corpus extent.
    PaperExtent,
}

/// Full context supplied to a primitive planner.
///
/// This is the first-class public planner surface. `WorkloadShape` remains as a
/// compatibility alias.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct PlanContext {
    /// API context.
    pub context: ApiContext,
    /// Specific read/access pattern.
    pub read_pattern: ReadPattern,
    /// Content family.
    pub content: ContentKind,
    /// Entropy class.
    pub entropy: EntropyClass,
    /// Entropy scale.
    pub scale: EntropyScale,
    /// Total bytes in the logical workload.
    pub total_bytes: usize,
    /// Bytes per call or chunk. Zero means no fixed chunk.
    pub chunk_bytes: usize,
    /// Thread count requested by the caller. One means single-threaded.
    pub threads: usize,
    /// Byte alignment offset from the preferred/cache-line boundary.
    pub alignment_offset: usize,
    /// Expected cache state.
    pub cache_state: CacheState,
    /// Source/data hint.
    pub source_hint: SourceHint,
}

/// Backward-compatible name for planner context.
pub type WorkloadShape = PlanContext;

impl PlanContext {
    /// Builds a planner context with unknown content and entropy.
    #[must_use]
    pub const fn new(context: ApiContext, total_bytes: usize) -> Self {
        Self {
            context,
            read_pattern: ReadPattern::from_context(context),
            content: ContentKind::Unknown,
            entropy: EntropyClass::Unknown,
            scale: EntropyScale::Unknown,
            total_bytes,
            chunk_bytes: 0,
            threads: 1,
            alignment_offset: 0,
            cache_state: CacheState::Unknown,
            source_hint: SourceHint::Unknown,
        }
    }
}

/// Named histogram implementation strategy selected by planning.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum HistogramStrategy {
    /// Direct scalar count into one `u64` table.
    DirectU64,
    /// Local `u32` table reduced into public `u64` counts.
    LocalU32,
    /// Four striped `u32` tables.
    Stripe4U32,
    /// Eight striped `u32` tables.
    Stripe8U32,
    /// Run-length count.
    RunLengthU64,
    /// Cheap adaptive classifier using a 1 KiB prefix.
    AdaptivePrefix1K,
    /// Adaptive classifier using a 4 KiB prefix.
    AdaptivePrefix4K,
    /// Spread sampling across a block.
    AdaptiveSpread4K,
    /// Conservative adaptive classifier that detects only obvious long runs.
    AdaptiveRunSentinel4K,
    /// Per-64 KiB chunk adaptation.
    AdaptiveChunked64K,
    /// Adaptive sequential planner updated at 64 KiB boundaries.
    AdaptiveSequentialOnline64K,
    /// File-level planner that samples once and reuses that choice.
    AdaptiveFileCached64K,
    /// Low-entropy fast path.
    AdaptiveLowEntropyFast,
    /// ASCII/text-biased fast path.
    AdaptiveAsciiFast,
    /// High-entropy path that skips specialized logic.
    AdaptiveHighEntropySkip,
    /// Meso-pattern detector for block-palette-like data.
    AdaptiveMesoDetector,
    /// Stateful sequential planner that reuses decisions across reads.
    StatefulSequential,
}

impl HistogramStrategy {
    /// Returns a stable strategy identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DirectU64 => "direct-u64",
            Self::LocalU32 => "local-u32",
            Self::Stripe4U32 => "stripe4-u32",
            Self::Stripe8U32 => "stripe8-u32",
            Self::RunLengthU64 => "run-length-u64",
            Self::AdaptivePrefix1K => "adaptive-prefix-1k",
            Self::AdaptivePrefix4K => "adaptive-prefix-4k",
            Self::AdaptiveSpread4K => "adaptive-spread-4k",
            Self::AdaptiveRunSentinel4K => "adaptive-run-sentinel-4k",
            Self::AdaptiveChunked64K => "adaptive-chunked-64k",
            Self::AdaptiveSequentialOnline64K => "adaptive-sequential-online-64k",
            Self::AdaptiveFileCached64K => "adaptive-file-cached-64k",
            Self::AdaptiveLowEntropyFast => "adaptive-low-entropy-fast",
            Self::AdaptiveAsciiFast => "adaptive-ascii-fast",
            Self::AdaptiveHighEntropySkip => "adaptive-high-entropy-skip",
            Self::AdaptiveMesoDetector => "adaptive-meso-detector",
            Self::StatefulSequential => "stateful-sequential",
        }
    }

    /// Returns catalog metadata for this strategy when available.
    #[must_use]
    pub fn kernel_info(self) -> Option<&'static HistogramKernelInfo> {
        histogram_kernel_catalog()
            .iter()
            .find(|info| info.strategy == self)
    }
}

/// Returns catalog metadata for currently known histogram strategies.
#[must_use]
pub const fn histogram_kernel_catalog() -> &'static [HistogramKernelInfo] {
    &HISTOGRAM_KERNEL_CATALOG
}

const HISTOGRAM_KERNEL_CATALOG: [HistogramKernelInfo; 17] = [
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::DirectU64,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Tiny,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 0,
        preferred_chunk_bytes: 0,
        sample_bytes: 0,
        private_table_bytes: 0,
        description: "single public u64 table; reference behavior and tiny-call fallback",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::LocalU32,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 256,
        preferred_chunk_bytes: 0,
        sample_bytes: 0,
        private_table_bytes: 256 * 4,
        description: "one private u32 table reduced into public u64 counts",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::Stripe4U32,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 0,
        private_table_bytes: 4 * 256 * 4,
        description: "four private u32 tables to break counter dependency chains",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::Stripe8U32,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 0,
        private_table_bytes: 8 * 256 * 4,
        description: "eight private u32 tables for low-cardinality or repeated data",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::RunLengthU64,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Tiny,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 64,
        preferred_chunk_bytes: 0,
        sample_bytes: 0,
        private_table_bytes: 0,
        description: "run scanner that increments one counter per equal-byte run",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptivePrefix1K,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 1024,
        private_table_bytes: 8 * 256 * 4,
        description: "cheap prefix classifier that chooses among scalar table shapes",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptivePrefix4K,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 4 * 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4 * 1024,
        private_table_bytes: 8 * 256 * 4,
        description: "larger prefix classifier for inputs where 1 KiB is too noisy",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveSpread4K,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 16 * 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4 * 1024,
        private_table_bytes: 8 * 256 * 4,
        description: "spread samples across a block to catch meso-scale variation",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveRunSentinel4K,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Tiny,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 4 * 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4 * 1024,
        private_table_bytes: 0,
        description: "conservative long-run sentinel before falling back to direct counts",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveChunked64K,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L2,
        statefulness: KernelStatefulness::Chunked,
        min_bytes: 64 * 1024,
        preferred_chunk_bytes: 64 * 1024,
        sample_bytes: 1024,
        private_table_bytes: 8 * 256 * 4,
        description: "per-64 KiB adaptive chunks for macro-scale mixed files",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveSequentialOnline64K,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Streaming,
        statefulness: KernelStatefulness::Stateful,
        min_bytes: 64 * 1024,
        preferred_chunk_bytes: 64 * 1024,
        sample_bytes: 1024,
        private_table_bytes: 8 * 256 * 4,
        description: "online sequential planner that can change choices at chunk boundaries",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveFileCached64K,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Streaming,
        statefulness: KernelStatefulness::Stateful,
        min_bytes: 64 * 1024,
        preferred_chunk_bytes: 64 * 1024,
        sample_bytes: 4096,
        private_table_bytes: 8 * 256 * 4,
        description: "file-level planner that samples once and reuses the choice",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveLowEntropyFast,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Streaming,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 4 * 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4096,
        private_table_bytes: 0,
        description: "low-entropy detector that promotes obvious long-run inputs",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveAsciiFast,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Streaming,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 4 * 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4096,
        private_table_bytes: 8 * 256 * 4,
        description: "ASCII/text-biased path that avoids extra sampling when text dominates",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveHighEntropySkip,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Streaming,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 64 * 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4096,
        private_table_bytes: 256 * 4,
        description: "high-entropy path that avoids run/text-specific probes",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::AdaptiveMesoDetector,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::L2,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 64 * 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4096,
        private_table_bytes: 8 * 256 * 4,
        description: "meso-pattern detector for block-palette-like data",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::StatefulSequential,
        isa: KernelIsa::PortableScalar,
        working_set: WorkingSetClass::Streaming,
        statefulness: KernelStatefulness::Stateful,
        min_bytes: 4 * 1024,
        preferred_chunk_bytes: 4 * 1024,
        sample_bytes: 1024,
        private_table_bytes: 8 * 256 * 4,
        description: "future sequential planner that reuses decisions across reads",
    },
];

/// Planned histogram strategy and explanation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct HistogramPlan {
    /// Selected strategy.
    pub strategy: HistogramStrategy,
    /// Planned chunk size in bytes. Zero means caller chunking is retained.
    pub chunk_bytes: usize,
    /// Planned sample size in bytes.
    pub sample_bytes: usize,
    /// Planner confidence on a 0..=255 scale.
    pub confidence_q8: u8,
    /// Stable human-readable reason.
    pub reason: &'static str,
}

/// Plans a histogram strategy from processor and workload metadata.
#[must_use]
pub fn plan_histogram(profile: &ProcessorProfile, workload: &WorkloadShape) -> HistogramPlan {
    let _backend = profile.backend;
    let total_bytes = workload.total_bytes;
    let call_bytes = if workload.chunk_bytes == 0 {
        total_bytes
    } else {
        workload.chunk_bytes.min(total_bytes.max(1))
    };
    let threads = workload.threads.max(1);
    let oversubscribed = profile
        .logical_cpus
        .is_some_and(|logical| threads > logical.max(1));

    if threads <= 1 && call_bytes <= 4 * 1024 {
        return plan(
            HistogramStrategy::DirectU64,
            workload.chunk_bytes,
            0,
            255,
            "micro reads should avoid classifier overhead",
        );
    }

    if matches!(
        workload.read_pattern,
        ReadPattern::RandomTiny | ReadPattern::Random | ReadPattern::ZipfianHotCold
    ) && call_bytes <= 4 * 1024
    {
        return plan(
            HistogramStrategy::DirectU64,
            workload.chunk_bytes,
            0,
            255,
            "tiny random reads are dominated by call overhead; avoid adaptive sampling",
        );
    }

    if workload.entropy == EntropyClass::Low && total_bytes >= 64 * 1024 {
        return plan(
            HistogramStrategy::AdaptiveLowEntropyFast,
            workload.chunk_bytes,
            4 * 1024,
            235,
            "large low-entropy inputs need the dedicated low-entropy fast path",
        );
    }

    if workload.cache_state == CacheState::Reused && total_bytes >= 64 * 1024 {
        return plan(
            HistogramStrategy::AdaptiveFileCached64K,
            64 * 1024,
            4 * 1024,
            215,
            "repeated file access can amortize one file-level decision",
        );
    }

    if threads > 1 || workload.context == ApiContext::Parallel {
        if workload.content == ContentKind::Text {
            return plan(
                HistogramStrategy::LocalU32,
                64 * 1024,
                0,
                if oversubscribed { 150 } else { 205 },
                "parallel text scans favored simple private tables over chunked adaptation",
            );
        }

        if workload.entropy == EntropyClass::High && workload.scale == EntropyScale::Flat {
            return plan(
                HistogramStrategy::DirectU64,
                64 * 1024,
                0,
                if oversubscribed { 145 } else { 210 },
                "parallel high-entropy scans should avoid adaptive chunk over-selection",
            );
        }

        return plan(
            HistogramStrategy::AdaptiveRunSentinel4K,
            64 * 1024,
            4 * 1024,
            if oversubscribed { 145 } else { 185 },
            "parallel mixed scans use a cheap run sentinel instead of default chunked planning",
        );
    }

    if workload.alignment_offset >= 16
        && workload.entropy == EntropyClass::High
        && total_bytes >= 64 * 1024
    {
        return plan(
            HistogramStrategy::LocalU32,
            workload.chunk_bytes,
            0,
            185,
            "large unaligned high-entropy inputs avoid the direct u64 path",
        );
    }

    if workload.scale == EntropyScale::Macro && total_bytes >= 256 * 1024 {
        if workload.source_hint == SourceHint::RealFile
            && workload.entropy == EntropyClass::High
            && workload.content == ContentKind::Binary
        {
            return plan(
                HistogramStrategy::DirectU64,
                workload.chunk_bytes,
                0,
                175,
                "real high-entropy file slices favored direct/simple kernels in calibration",
            );
        }

        return plan(
            HistogramStrategy::AdaptiveChunked64K,
            64 * 1024,
            1024,
            205,
            "macro-scale variation benefits from per-region kernel choice",
        );
    }

    if workload.scale == EntropyScale::Meso
        && (workload.context == ApiContext::Block || workload.chunk_bytes >= 64 * 1024)
    {
        return plan(
            HistogramStrategy::AdaptiveMesoDetector,
            workload.chunk_bytes,
            4 * 1024,
            205,
            "meso-scale structure needs a block-pattern detector",
        );
    }

    if workload.content == ContentKind::Text {
        if total_bytes >= 64 * 1024 * 1024 || workload.context == ApiContext::File {
            return plan(
                HistogramStrategy::AdaptiveFileCached64K,
                64 * 1024,
                4 * 1024,
                205,
                "large text/file inputs can amortize file-level text detection",
            );
        }

        if total_bytes >= 16 * 1024 * 1024 {
            return plan(
                HistogramStrategy::AdaptiveAsciiFast,
                workload.chunk_bytes,
                4 * 1024,
                210,
                "large text inputs benefit from the ASCII-biased fast path",
            );
        }

        if total_bytes >= 8 * 1024 {
            return plan(
                HistogramStrategy::LocalU32,
                workload.chunk_bytes,
                0,
                200,
                "medium text inputs favored local private tables in size sweeps",
            );
        }
    }

    if workload.entropy == EntropyClass::High {
        if total_bytes <= 16 * 1024 {
            return plan(
                HistogramStrategy::DirectU64,
                workload.chunk_bytes,
                0,
                220,
                "small high-entropy inputs favored the direct path",
            );
        }

        if total_bytes <= 1024 * 1024 {
            return plan(
                HistogramStrategy::LocalU32,
                workload.chunk_bytes,
                0,
                190,
                "mid-sized high-entropy inputs favored local private tables",
            );
        }

        return plan(
            HistogramStrategy::AdaptiveHighEntropySkip,
            workload.chunk_bytes,
            4 * 1024,
            175,
            "large high-entropy inputs should skip low-entropy/text probes",
        );
    }

    if matches!(
        workload.read_pattern,
        ReadPattern::Sequential | ReadPattern::Readahead
    ) && total_bytes >= 64 * 1024
    {
        return plan(
            HistogramStrategy::AdaptiveSequentialOnline64K,
            64 * 1024,
            1024,
            170,
            "sequential reads can update choices at region boundaries",
        );
    }

    plan(
        HistogramStrategy::AdaptivePrefix1K,
        workload.chunk_bytes,
        1024,
        150,
        "general-purpose fallback with a bounded prefix classifier",
    )
}

fn plan(
    strategy: HistogramStrategy,
    chunk_bytes: usize,
    sample_bytes: usize,
    confidence_q8: u8,
    reason: &'static str,
) -> HistogramPlan {
    HistogramPlan {
        strategy,
        chunk_bytes,
        sample_bytes,
        confidence_q8,
        reason,
    }
}

/// Returns the backend that public kernels should use on this process.
///
/// A forced backend, set with [`force_backend`], takes precedence over runtime
/// CPU detection. This is intended for tests and benchmarks.
#[must_use]
pub fn detected_backend() -> Backend {
    if let Some(backend) = decode_backend(FORCED_BACKEND.load(Ordering::Relaxed)) {
        return backend;
    }

    detect_backend()
}

/// Forces runtime-dispatched kernels to use `backend`.
///
/// This function is primarily for tests and benchmarks. Production callers
/// should normally rely on [`detected_backend`].
pub fn force_backend(backend: Backend) {
    FORCED_BACKEND.store(backend as u8, Ordering::Relaxed);
}

/// Clears a previously forced backend.
pub fn clear_forced_backend() {
    FORCED_BACKEND.store(AUTO_BACKEND, Ordering::Relaxed);
}

/// Returns a processor profile for the current process.
#[must_use]
pub fn detected_processor_profile() -> ProcessorProfile {
    ProcessorProfile::detect()
}

/// Detects cache information for the current process when the platform exposes it.
#[must_use]
pub fn detected_cache_profile() -> CacheProfile {
    detect_cache_profile()
}

fn decode_backend(value: u8) -> Option<Backend> {
    match value {
        value if value == Backend::Avx512 as u8 => Some(Backend::Avx512),
        value if value == Backend::Avx2 as u8 => Some(Backend::Avx2),
        value if value == Backend::Sve2 as u8 => Some(Backend::Sve2),
        value if value == Backend::Sve as u8 => Some(Backend::Sve),
        value if value == Backend::Neon as u8 => Some(Backend::Neon),
        value if value == Backend::Scalar as u8 => Some(Backend::Scalar),
        _ => None,
    }
}

fn detect_backend() -> Backend {
    cfg_if::cfg_if! {
        if #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))] {
            detect_x86_backend()
        } else if #[cfg(all(feature = "std", target_arch = "aarch64"))] {
            detect_aarch64_backend()
        } else {
            Backend::Scalar
        }
    }
}

fn detect_logical_cpus() -> Option<usize> {
    cfg_if::cfg_if! {
        if #[cfg(feature = "std")] {
            std::thread::available_parallelism()
                .ok()
                .map(core::num::NonZeroUsize::get)
        } else {
            None
        }
    }
}

fn detect_cache_profile() -> CacheProfile {
    cfg_if::cfg_if! {
        if #[cfg(all(feature = "std", target_os = "linux"))] {
            cached_linux_cache_profile()
        } else {
            CacheProfile::unknown()
        }
    }
}

#[cfg(all(feature = "std", target_os = "linux"))]
fn cached_linux_cache_profile() -> CacheProfile {
    static CACHE_PROFILE: std::sync::OnceLock<CacheProfile> = std::sync::OnceLock::new();
    *CACHE_PROFILE.get_or_init(detect_linux_cache_profile)
}

#[cfg(all(feature = "std", target_os = "linux"))]
fn detect_linux_cache_profile() -> CacheProfile {
    let root = std::path::Path::new("/sys/devices/system/cpu/cpu0/cache");
    let Ok(entries) = std::fs::read_dir(root) else {
        return CacheProfile::unknown();
    };

    let mut profile = CacheProfile::unknown();

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("index") {
            continue;
        }

        let level = read_sysfs_trimmed(path.join("level")).and_then(|value| value.parse().ok());
        let kind = read_sysfs_trimmed(path.join("type"));
        let size = read_sysfs_trimmed(path.join("size")).and_then(|value| parse_size_bytes(&value));
        let line = read_sysfs_trimmed(path.join("coherency_line_size"))
            .and_then(|value| value.parse().ok());

        if profile.line_bytes.is_none() {
            profile.line_bytes = line;
        }

        let Some(size) = size else {
            continue;
        };

        match (level, kind.as_deref()) {
            (Some(1_u8), Some(kind)) if kind.eq_ignore_ascii_case("Data") => {
                profile.l1d_bytes = min_some(profile.l1d_bytes, size);
            }
            (Some(1_u8), Some(kind)) if kind.eq_ignore_ascii_case("Unified") => {
                profile.l1d_bytes = min_some(profile.l1d_bytes, size);
            }
            (Some(2_u8), Some(kind)) if kind.eq_ignore_ascii_case("Unified") => {
                profile.l2_bytes = min_some(profile.l2_bytes, size);
            }
            (Some(3_u8), _) => {
                profile.l3_bytes = max_some(profile.l3_bytes, size);
            }
            _ => {}
        }
    }

    profile
}

#[cfg(all(feature = "std", target_os = "linux"))]
fn read_sysfs_trimmed(path: std::path::PathBuf) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(all(feature = "std", target_os = "linux"))]
fn parse_size_bytes(value: &str) -> Option<usize> {
    let split = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, suffix) = value.split_at(split);
    let number = number.parse::<usize>().ok()?;
    let multiplier = match suffix.trim().to_ascii_uppercase().as_str() {
        "" => 1,
        "K" | "KB" | "KIB" => 1024,
        "M" | "MB" | "MIB" => 1024 * 1024,
        "G" | "GB" | "GIB" => 1024 * 1024 * 1024,
        _ => return None,
    };
    number.checked_mul(multiplier)
}

#[cfg(all(feature = "std", target_os = "linux"))]
fn min_some(current: Option<usize>, candidate: usize) -> Option<usize> {
    Some(current.map_or(candidate, |value| value.min(candidate)))
}

#[cfg(all(feature = "std", target_os = "linux"))]
fn max_some(current: Option<usize>, candidate: usize) -> Option<usize> {
    Some(current.map_or(candidate, |value| value.max(candidate)))
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_x86_backend() -> Backend {
    #[cfg(all(feature = "avx512", feature = "nightly"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return Backend::Avx512;
        }
    }

    #[cfg(feature = "avx2")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return Backend::Avx2;
        }
    }

    Backend::Scalar
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
fn detect_aarch64_backend() -> Backend {
    #[cfg(feature = "sve2")]
    {
        if std::arch::is_aarch64_feature_detected!("sve2") {
            return Backend::Sve2;
        }
    }

    #[cfg(feature = "sve")]
    {
        if std::arch::is_aarch64_feature_detected!("sve") {
            return Backend::Sve;
        }
    }

    #[cfg(feature = "neon")]
    {
        return Backend::Neon;
    }

    #[allow(unreachable_code)]
    Backend::Scalar
}

#[cfg(test)]
mod tests {
    use super::{
        ApiContext, CacheState, ContentKind, EntropyClass, EntropyScale, HistogramStrategy,
        ProcessorProfile, ReadPattern, SourceHint, WorkloadShape, histogram_kernel_catalog,
        plan_histogram,
    };

    #[test]
    fn planner_avoids_adaptive_sampling_for_tiny_random_reads() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Random, 16 * 1024);
        workload.chunk_bytes = 1;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::DirectU64);
        assert_eq!(plan.sample_bytes, 0);
    }

    #[test]
    fn planner_avoids_adaptive_sampling_for_tiny_blocks() {
        let profile = ProcessorProfile::portable();
        let workload = WorkloadShape::new(ApiContext::Block, 256);

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::DirectU64);
        assert_eq!(plan.sample_bytes, 0);
    }

    #[test]
    fn planner_uses_chunked_strategy_for_macro_inputs() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 1024 * 1024);
        workload.content = ContentKind::Mixed;
        workload.entropy = EntropyClass::Mixed;
        workload.scale = EntropyScale::Macro;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::AdaptiveChunked64K);
        assert_eq!(plan.chunk_bytes, 64 * 1024);
    }

    #[test]
    fn planner_avoids_sampling_for_small_sequential_reads() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Sequential, 1024 * 1024);
        workload.chunk_bytes = 4 * 1024;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::DirectU64);
        assert_eq!(plan.sample_bytes, 0);
    }

    #[test]
    fn planner_uses_low_entropy_fast_path_for_large_low_entropy_inputs() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 256 * 1024 * 1024);
        workload.entropy = EntropyClass::Low;
        workload.scale = EntropyScale::Flat;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::AdaptiveLowEntropyFast);
        assert_ne!(plan.strategy, HistogramStrategy::DirectU64);
    }

    #[test]
    fn planner_does_not_blindly_chunk_parallel_high_entropy_scans() {
        let profile = ProcessorProfile {
            logical_cpus: Some(24),
            ..ProcessorProfile::portable()
        };
        let mut workload = WorkloadShape::new(ApiContext::Parallel, 64 * 1024 * 1024);
        workload.read_pattern = ReadPattern::ParallelSequential;
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Flat;
        workload.chunk_bytes = 64 * 1024;
        workload.threads = 4;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::DirectU64);
    }

    #[test]
    fn planner_uses_local_table_for_medium_text() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 64 * 1024);
        workload.content = ContentKind::Text;
        workload.entropy = EntropyClass::Medium;
        workload.scale = EntropyScale::Micro;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::LocalU32);
        assert_eq!(plan.sample_bytes, 0);
    }

    #[test]
    fn planner_uses_cached_file_strategy_for_repeated_large_regions() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::File, 256 * 1024 * 1024);
        workload.read_pattern = ReadPattern::SameFileRepeat;
        workload.content = ContentKind::Text;
        workload.entropy = EntropyClass::Medium;
        workload.cache_state = CacheState::Reused;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::AdaptiveFileCached64K);
    }

    #[test]
    fn planner_uses_alignment_context_for_large_unaligned_high_entropy_inputs() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 1024 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Flat;
        workload.alignment_offset = 31;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::LocalU32);
    }

    #[test]
    fn planner_avoids_chunked_for_real_high_entropy_file_slices() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::File, 4 * 1024 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Macro;
        workload.source_hint = SourceHint::RealFile;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::DirectU64);
    }

    #[test]
    fn histogram_catalog_uses_stable_strategy_ids() {
        let catalog = histogram_kernel_catalog();

        assert!(catalog.len() >= 10);
        assert!(
            catalog
                .iter()
                .any(|info| info.strategy == HistogramStrategy::AdaptiveChunked64K)
        );

        for info in catalog {
            let catalog_info = info.strategy.kernel_info();
            assert!(catalog_info.is_some());
            assert_eq!(
                info.strategy.as_str(),
                catalog_info
                    .map(|info| info.strategy.as_str())
                    .unwrap_or_default()
            );
        }
    }

    #[cfg(all(feature = "std", target_os = "linux"))]
    #[test]
    fn parses_linux_cache_sizes() {
        assert_eq!(super::parse_size_bytes("48K"), Some(48 * 1024));
        assert_eq!(super::parse_size_bytes("1280K"), Some(1280 * 1024));
        assert_eq!(super::parse_size_bytes("30M"), Some(30 * 1024 * 1024));
        assert_eq!(super::parse_size_bytes("n/a"), None);
    }
}

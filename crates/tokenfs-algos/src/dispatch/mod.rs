//! Runtime backend detection, kernel catalogs, and planner support.

pub mod planner;

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
    /// Intel AMX (Advanced Matrix Extensions) backend (Sapphire Rapids+).
    ///
    /// Detection-only: no kernels are wired to this backend yet. The
    /// variant exists so [`ProcessorProfile`] can advertise AMX
    /// availability and so future tile-based primitives have a stable
    /// label in dispatch traces.
    Amx = 7,
    /// Arm SME (Scalable Matrix Extension) backend (ARMv9.2+).
    ///
    /// Detection-only: see [`Backend::Amx`].
    Sme = 8,
    /// Arm SME2 backend (ARMv9.3+).
    ///
    /// Detection-only: see [`Backend::Amx`].
    Sme2 = 9,
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

/// Matrix/tile accelerator availability for the current process.
///
/// This is purely detection metadata; the crate does not yet ship
/// AMX/SME kernels. The fields exist so the planner and
/// `dispatch_explain` example can surface what each runner advertises,
/// and so future tile-based primitives can gate on real CPU support.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub struct AcceleratorProfile {
    /// Intel AMX tile-config support (`amx-tile`).
    pub amx_tile: bool,
    /// Intel AMX INT8 tile mul (`amx-int8`).
    pub amx_int8: bool,
    /// Intel AMX BF16 tile mul (`amx-bf16`).
    pub amx_bf16: bool,
    /// Arm SME (Scalable Matrix Extension) base feature.
    pub sme: bool,
    /// Arm SME2.
    pub sme2: bool,
}

impl AcceleratorProfile {
    /// Returns the all-false accelerator profile.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            amx_tile: false,
            amx_int8: false,
            amx_bf16: false,
            sme: false,
            sme2: false,
        }
    }

    /// Returns true if any AMX feature is detected.
    #[must_use]
    pub const fn has_any_amx(self) -> bool {
        self.amx_tile || self.amx_int8 || self.amx_bf16
    }

    /// Returns true if SME or SME2 is detected.
    #[must_use]
    pub const fn has_any_sme(self) -> bool {
        self.sme || self.sme2
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
    /// AMX/SME accelerator availability (detection-only).
    pub accelerators: AcceleratorProfile,
}

impl ProcessorProfile {
    /// Builds a portable scalar processor profile.
    #[must_use]
    pub const fn portable() -> Self {
        Self {
            backend: Backend::Scalar,
            cache: CacheProfile::unknown(),
            logical_cpus: None,
            accelerators: AcceleratorProfile::none(),
        }
    }

    /// Detects a processor profile for the current process.
    #[must_use]
    pub fn detect() -> Self {
        Self {
            backend: detected_backend(),
            cache: detected_cache_profile(),
            logical_cpus: detect_logical_cpus(),
            accelerators: detect_accelerators(),
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

/// Implementation status for one primitive family on one backend.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum KernelAvailability {
    /// There is a first-class implementation for this backend.
    Native,
    /// The backend is detected, but this primitive currently uses scalar code.
    ScalarFallback,
    /// The backend is not compiled or not applicable for this primitive.
    NotAvailable,
}

/// Current implementation coverage for a backend.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BackendKernelSupport {
    /// Backend being described.
    pub backend: Backend,
    /// Byte-histogram implementation status.
    pub byte_histogram: KernelAvailability,
    /// F22/content fingerprint implementation status.
    pub fingerprint: KernelAvailability,
    /// Byte-class (classify + UTF-8 validation) implementation status.
    pub byte_class: KernelAvailability,
    /// Sketch / hash-bin (CRC32C-driven) implementation status.
    pub sketch: KernelAvailability,
    /// Run-length transition counter implementation status.
    pub runlength: KernelAvailability,
    /// Similarity f32 distance kernels (dot, L2², cosine).
    pub similarity: KernelAvailability,
    /// SHA-256 hash implementation status for **this `Backend` alone**.
    ///
    /// Hardware SHA-256 lives in separate ISA extensions (SHA-NI on
    /// x86, FEAT_SHA2 on aarch64) that are orthogonal to the SIMD
    /// `Backend` enum. Notable — many AVX2-capable Intel CPUs
    /// (Haswell through Rocket Lake) lack SHA-NI; FEAT_SHA2 is
    /// optional on ARMv8-A. So every SIMD-`Backend` row reports
    /// `ScalarFallback` here even though the runtime dispatch in
    /// `crate::hash::sha256::sha256` does correctly pick the
    /// hardware path when its own probe (`is_x86_feature_detected!
    /// ("sha")` / `is_aarch64_feature_detected!("sha2")`) returns
    /// true. To detect SHA hardware availability, query
    /// `crate::hash::sha256::kernels::x86_shani::is_available` /
    /// `kernels::aarch64_sha2::is_available` directly.
    pub hash_sha256: KernelAvailability,
}

/// Returns current primitive implementation coverage for `backend`.
///
/// This is intentionally conservative. Feature-shaped backends such as NEON,
/// AVX-512, SVE, and SVE2 are reported as scalar fallback until a real,
/// parity-tested kernel exists for the primitive family.
#[must_use]
pub const fn backend_kernel_support(backend: Backend) -> BackendKernelSupport {
    match backend {
        Backend::Avx2 => BackendKernelSupport {
            backend,
            // Native: histogram (avx2_stripe4_u32, avx2_palette_u32,
            // avx2_rle_stripe4_u32), fingerprint (kernels::avx2),
            // byteclass classify + utf8 (kernels::avx2 + utf8_avx2),
            // sketch CRC32C (kernels::sse42 — runs on every CPU that
            // has AVX2 in practice), runlength (kernels::avx2 with
            // BMI2/LZCNT gating), similarity dot/l2/cosine f32
            // (kernels::avx2). hash_sha256 stays ScalarFallback —
            // SHA-NI is OPTIONAL on x86 (Haswell..Rocket Lake have
            // AVX2 but no SHA-NI); see the field doc.
            byte_histogram: KernelAvailability::Native,
            fingerprint: KernelAvailability::Native,
            byte_class: KernelAvailability::Native,
            sketch: KernelAvailability::Native,
            runlength: KernelAvailability::Native,
            similarity: KernelAvailability::Native,
            // SHA hardware (SHA-NI / FEAT_SHA2) is orthogonal to this
            // SIMD backend; see the field's doc comment.
            hash_sha256: KernelAvailability::ScalarFallback,
        },
        Backend::Neon => BackendKernelSupport {
            backend,
            // Native: fingerprint (#37 — fingerprint::neon::block),
            // byteclass classify + utf8 (kernels::neon + utf8_neon),
            // sketch CRC32C (#45 — kernels::neon, hardware __crc32cw),
            // runlength (kernels::neon), similarity f32 distance
            // (kernels::neon), hash_sha256 (FEAT_SHA2 on aarch64,
            // dispatched separately by sha256::aarch64_sha2). Histogram
            // doesn't have a NEON kernel yet — `svhistseg_u8` would be
            // the natural SVE2 candidate; standalone NEON byte-histogram
            // is still future work.
            byte_histogram: KernelAvailability::ScalarFallback,
            fingerprint: KernelAvailability::Native,
            byte_class: KernelAvailability::Native,
            sketch: KernelAvailability::Native,
            runlength: KernelAvailability::Native,
            similarity: KernelAvailability::Native,
            // SHA hardware (SHA-NI / FEAT_SHA2) is orthogonal to this
            // SIMD backend; see the field's doc comment.
            hash_sha256: KernelAvailability::ScalarFallback,
        },
        Backend::Avx512 => BackendKernelSupport {
            backend,
            // Native: byteclass classify + utf8 (kernels::avx512 +
            // utf8_avx512 from #38; VBMI variant added in #47/#60),
            // sketch CRC32C (SSE4.2 path — every AVX-512 CPU has it),
            // runlength (AVX-512BW could replace AVX2 here but the
            // existing AVX2+BMI2 path already wins; reported as Native
            // because the AVX2 kernel runs unchanged on AVX-512 hosts),
            // similarity (AVX2 kernel, same logic), hash_sha256 (SHA-NI).
            // BITALG/GFNI bit-marginals (#51/#52) emit 8 marginals, NOT
            // a 256-bin histogram, so byte_histogram stays ScalarFallback.
            // Native AVX-512 fingerprint is future work.
            byte_histogram: KernelAvailability::ScalarFallback,
            fingerprint: KernelAvailability::ScalarFallback,
            byte_class: KernelAvailability::Native,
            sketch: KernelAvailability::Native,
            runlength: KernelAvailability::Native,
            similarity: KernelAvailability::Native,
            // SHA hardware (SHA-NI / FEAT_SHA2) is orthogonal to this
            // SIMD backend; see the field's doc comment.
            hash_sha256: KernelAvailability::ScalarFallback,
        },
        Backend::Sve | Backend::Sve2 => BackendKernelSupport {
            backend,
            // Native (SVE2): byteclass classify + utf8 (#41 —
            // kernels::sve2), runlength (#41 — kernels::sve2 with
            // svcntp_b8 popcount-of-mask). Native (SVE base ISA):
            // similarity dot/l2/cosine f32 (#41 — kernels::sve, runs on
            // both SVE and SVE2 hardware). NEON-mandatory paths
            // (fingerprint + sketch CRC32C) compose cleanly because
            // every SVE-capable AArch64 CPU also exposes NEON; reported
            // as Native via NEON delegation. Histogram is the same gap
            // as Backend::Neon.
            byte_histogram: KernelAvailability::ScalarFallback,
            fingerprint: KernelAvailability::Native,
            byte_class: KernelAvailability::Native,
            sketch: KernelAvailability::Native,
            runlength: KernelAvailability::Native,
            similarity: KernelAvailability::Native,
            // SHA hardware (SHA-NI / FEAT_SHA2) is orthogonal to this
            // SIMD backend; see the field's doc comment.
            hash_sha256: KernelAvailability::ScalarFallback,
        },
        Backend::Scalar => BackendKernelSupport {
            backend,
            byte_histogram: KernelAvailability::Native,
            fingerprint: KernelAvailability::Native,
            byte_class: KernelAvailability::Native,
            sketch: KernelAvailability::Native,
            runlength: KernelAvailability::Native,
            similarity: KernelAvailability::Native,
            // The scalar SHA-256 reference implementation always works
            // — it IS the FIPS 180-4 reference. Hardware extensions on
            // SIMD backends are reported separately; see the field doc.
            hash_sha256: KernelAvailability::Native,
        },
        // Detection-only accelerators: no kernels are wired in yet, so
        // every primitive falls back to scalar code today.
        Backend::Amx | Backend::Sme | Backend::Sme2 => BackendKernelSupport {
            backend,
            byte_histogram: KernelAvailability::ScalarFallback,
            fingerprint: KernelAvailability::ScalarFallback,
            byte_class: KernelAvailability::ScalarFallback,
            sketch: KernelAvailability::ScalarFallback,
            runlength: KernelAvailability::ScalarFallback,
            similarity: KernelAvailability::ScalarFallback,
            hash_sha256: KernelAvailability::ScalarFallback,
        },
    }
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
    /// AVX2-dispatched general four-stripe counter with exact scalar fallback.
    Avx2Stripe4U32,
    /// AVX2 palette counter with exact scalar fallback.
    Avx2PaletteU32,
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
            Self::Avx2Stripe4U32 => "avx2-stripe4-u32",
            Self::Avx2PaletteU32 => "avx2-palette-u32",
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

const HISTOGRAM_KERNEL_CATALOG: [HistogramKernelInfo; 19] = [
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
        strategy: HistogramStrategy::Avx2Stripe4U32,
        isa: KernelIsa::X86Avx2,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 0,
        private_table_bytes: 4 * 256 * 4,
        // Planner placeholder: the body of this kernel is currently scalar
        // four-stripe counting under an AVX2 target_feature gate. It exists
        // so the planner has a feature-dispatched general-cardinality slot
        // distinct from the palette specialization. A real AVX2 byte
        // histogram (gather-free / pshufb-table / radix) will replace it
        // and exact parity is enforced by tests/avx2_parity.rs today.
        description: "AVX2-dispatched general four-stripe counter (currently scalar body, real AVX2 pending)",
    },
    HistogramKernelInfo {
        family: PrimitiveFamily::ByteHistogram,
        strategy: HistogramStrategy::Avx2PaletteU32,
        isa: KernelIsa::X86Avx2,
        working_set: WorkingSetClass::L1,
        statefulness: KernelStatefulness::Stateless,
        min_bytes: 1024,
        preferred_chunk_bytes: 0,
        sample_bytes: 4096,
        private_table_bytes: 16 * 8,
        description: "AVX2 movemask/popcnt palette counter with exact scalar fallback",
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
    /// Source of the confidence score.
    pub confidence_source: PlannerConfidenceSource,
    /// Stable human-readable reason.
    pub reason: &'static str,
}

/// Planner confidence provenance.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum PlannerConfidenceSource {
    /// Static rule from API/context and coarse workload metadata.
    StaticRule,
    /// Rule anchored in benchmark or paper-data calibration.
    CalibrationRule,
    /// Conservative fallback rule with intentionally low confidence.
    Fallback,
}

impl PlannerConfidenceSource {
    /// Stable identifier.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StaticRule => "static-rule",
            Self::CalibrationRule => "calibration-rule",
            Self::Fallback => "fallback",
        }
    }
}

/// Plans a histogram strategy from processor and workload metadata.
///
/// The planner is implemented as a rule table — see [`planner`] for the
/// architecture. Adding a rule, threshold, or confidence band goes through
/// the planner submodule, not this function.
#[must_use]
pub fn plan_histogram(profile: &ProcessorProfile, workload: &WorkloadShape) -> HistogramPlan {
    planner::rules::plan_histogram(profile, workload)
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
        value if value == Backend::Amx as u8 => Some(Backend::Amx),
        value if value == Backend::Sme as u8 => Some(Backend::Sme),
        value if value == Backend::Sme2 as u8 => Some(Backend::Sme2),
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

/// Detects matrix/tile accelerator availability for the current process.
///
/// Returns the all-false profile when the platform doesn't expose the
/// relevant feature flags (e.g. `std` is off, or the target arch is
/// neither x86 nor aarch64).
#[must_use]
pub fn detect_accelerators() -> AcceleratorProfile {
    // Detection stubs only. AMX (`is_x86_feature_detected!("amx-*")`)
    // and SME (`is_aarch64_feature_detected!("sme"|"sme2")`) macro
    // arms are gated behind unstable `stdarch_*_feature_detection`
    // nightly features that are still in flux. The CI nightly
    // (2026-04-30+) no longer recognises the `feature` attribute
    // for them. Until those macros stabilise, `detect_accelerators`
    // returns the all-false profile on every host. Re-enabling is a
    // single-callsite swap once the toolchain ships them stable; the
    // surrounding `Backend::Amx`/`Sme`/`Sme2` enum + `KernelSupport`
    // wiring stay live so the swap doesn't ripple.
    AcceleratorProfile::none()
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
        AcceleratorProfile, ApiContext, Backend, CacheState, ContentKind, EntropyClass,
        EntropyScale, HistogramStrategy, KernelAvailability, PlannerConfidenceSource,
        ProcessorProfile, ReadPattern, SourceHint, WorkloadShape, backend_kernel_support,
        detect_accelerators, histogram_kernel_catalog, plan_histogram,
    };

    #[test]
    fn portable_profile_advertises_no_accelerators() {
        let profile = ProcessorProfile::portable();
        assert_eq!(profile.accelerators, AcceleratorProfile::none());
        assert!(!profile.accelerators.has_any_amx());
        assert!(!profile.accelerators.has_any_sme());
    }

    #[test]
    fn detect_accelerators_returns_a_concrete_profile() {
        // We can't assert the bits — they depend on the host CPU — but
        // we can at least guarantee the call returns and that, when an
        // AMX bit is reported, the umbrella `has_any_amx` agrees.
        let profile = detect_accelerators();
        if profile.amx_tile || profile.amx_int8 || profile.amx_bf16 {
            assert!(profile.has_any_amx());
        }
        if profile.sme || profile.sme2 {
            assert!(profile.has_any_sme());
        }
    }

    #[test]
    fn backend_support_is_honest_about_amx_and_sme() {
        for backend in [Backend::Amx, Backend::Sme, Backend::Sme2] {
            let support = backend_kernel_support(backend);
            // No real kernels are wired in yet; every primitive must
            // honestly report scalar fallback for the new backends.
            assert_eq!(support.byte_histogram, KernelAvailability::ScalarFallback);
            assert_eq!(support.fingerprint, KernelAvailability::ScalarFallback);
            assert_eq!(support.byte_class, KernelAvailability::ScalarFallback);
            assert_eq!(support.sketch, KernelAvailability::ScalarFallback);
        }
    }

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
    fn planner_uses_run_length_for_small_low_entropy_reads() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Sequential, 1024 * 1024);
        workload.chunk_bytes = 4 * 1024;
        workload.entropy = EntropyClass::Low;
        workload.scale = EntropyScale::Flat;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::RunLengthU64);
        assert_eq!(plan.sample_bytes, 0);
    }

    #[test]
    fn planner_uses_avx2_palette_for_large_low_entropy_inputs() {
        let profile = ProcessorProfile {
            backend: Backend::Avx2,
            ..ProcessorProfile::portable()
        };
        let mut workload = WorkloadShape::new(ApiContext::Block, 1024 * 1024);
        workload.entropy = EntropyClass::Low;
        workload.scale = EntropyScale::Flat;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Avx2PaletteU32);
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
    fn planner_avoids_adaptive_wrappers_for_parallel_high_entropy_meso_scans() {
        let profile = ProcessorProfile {
            logical_cpus: Some(24),
            ..ProcessorProfile::portable()
        };
        let mut workload = WorkloadShape::new(ApiContext::Parallel, 64 * 1024);
        workload.read_pattern = ReadPattern::ParallelSequential;
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Meso;
        workload.chunk_bytes = 64 * 1024;
        workload.threads = 2;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::DirectU64);

        workload.threads = 4;
        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::LocalU32);
    }

    #[test]
    fn planner_uses_stripe8_for_high_entropy_meso_blocks() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 64 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Meso;
        workload.source_hint = SourceHint::RealFile;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Stripe8U32);
    }

    #[test]
    fn planner_uses_stripe4_for_high_entropy_meso_random_4k_reads() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Random, 64 * 1024);
        workload.read_pattern = ReadPattern::Random;
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Meso;
        workload.chunk_bytes = 4 * 1024;
        workload.source_hint = SourceHint::RealFile;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Stripe4U32);
    }

    #[test]
    fn planner_uses_direct_for_real_high_entropy_flat_files() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 64 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Flat;
        workload.source_hint = SourceHint::RealFile;

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
    fn planner_uses_stripe8_for_large_paper_mixed_extents() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 1024 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::Mixed;
        workload.scale = EntropyScale::Macro;
        workload.source_hint = SourceHint::PaperExtent;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Stripe8U32);
    }

    #[test]
    fn planner_uses_direct_for_random_paper_4k_reads() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Random, 1024 * 1024);
        workload.chunk_bytes = 4 * 1024;
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::Mixed;
        workload.scale = EntropyScale::Macro;
        workload.source_hint = SourceHint::PaperExtent;
        workload.read_pattern = ReadPattern::Random;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::DirectU64);
        assert_eq!(plan.sample_bytes, 0);
        assert_eq!(
            plan.confidence_source,
            PlannerConfidenceSource::CalibrationRule
        );
    }

    #[test]
    fn planner_confidence_source_marks_calibration_and_fallback_rules() {
        let profile = ProcessorProfile::portable();
        let mut paper = WorkloadShape::new(ApiContext::Block, 1024 * 1024);
        paper.content = ContentKind::Binary;
        paper.entropy = EntropyClass::Mixed;
        paper.scale = EntropyScale::Macro;
        paper.source_hint = SourceHint::PaperExtent;
        let paper_plan = plan_histogram(&profile, &paper);

        assert_eq!(
            paper_plan.confidence_source,
            PlannerConfidenceSource::CalibrationRule
        );

        let fallback = WorkloadShape::new(ApiContext::Block, 32 * 1024);
        let fallback_plan = plan_histogram(&profile, &fallback);

        assert_eq!(
            fallback_plan.confidence_source,
            PlannerConfidenceSource::Fallback
        );
    }

    #[test]
    fn planner_can_select_avx2_palette_for_medium_micro_binary_inputs() {
        let profile = ProcessorProfile {
            backend: Backend::Avx2,
            ..ProcessorProfile::portable()
        };
        let mut workload = WorkloadShape::new(ApiContext::Block, 64 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::Medium;
        workload.scale = EntropyScale::Micro;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Avx2PaletteU32);
    }

    #[test]
    fn planner_uses_stripe8_for_sequential_meso_structured_inputs() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Sequential, 1024 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::Mixed;
        workload.scale = EntropyScale::Meso;
        workload.chunk_bytes = 64 * 1024;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Stripe8U32);
    }

    #[test]
    fn planner_uses_striped_buckets_for_parallel_meso_structured_inputs() {
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Parallel, 1024 * 1024);
        workload.content = ContentKind::Binary;
        workload.entropy = EntropyClass::Mixed;
        workload.scale = EntropyScale::Meso;
        workload.chunk_bytes = 64 * 1024;
        workload.threads = 4;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Stripe4U32);
    }

    #[test]
    fn planner_uses_avx2_palette_for_parallel_macro_mixed_inputs() {
        let profile = ProcessorProfile {
            backend: Backend::Avx2,
            ..ProcessorProfile::portable()
        };
        let mut workload = WorkloadShape::new(ApiContext::Parallel, 1024 * 1024);
        workload.content = ContentKind::Mixed;
        workload.entropy = EntropyClass::Mixed;
        workload.scale = EntropyScale::Macro;
        workload.chunk_bytes = 64 * 1024;
        workload.threads = 4;

        let plan = plan_histogram(&profile, &workload);

        assert_eq!(plan.strategy, HistogramStrategy::Avx2PaletteU32);
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

    #[test]
    fn planner_does_not_select_the_avx2_stripe4_placeholder() {
        // Per docs/PRIMITIVE_KERNEL_BUFFET.md: the Avx2Stripe4U32 strategy
        // is a planner placeholder whose body is currently scalar (no real
        // x86 byte-histogram win exists per published literature — see
        // Powturbo TurboHist, Yann Collet, Lemire). It survives only as a
        // benchmark history label so a future genuine AVX2 implementation
        // can replace it without re-plumbing the bench schema. No active
        // planner rule should emit it.
        //
        // This sweep exercises a representative cross-product of profiles
        // and workloads. If a future planner rule starts selecting the
        // placeholder, this test will flag it.
        let profiles = [
            ProcessorProfile::portable(),
            {
                let mut p = ProcessorProfile::portable();
                p.backend = Backend::Avx2;
                p
            },
            {
                let mut p = ProcessorProfile::portable();
                p.backend = Backend::Neon;
                p
            },
        ];
        let api_contexts = [
            ApiContext::Block,
            ApiContext::Sequential,
            ApiContext::Random,
            ApiContext::File,
            ApiContext::Parallel,
        ];
        let entropies = [EntropyClass::Low, EntropyClass::Mixed, EntropyClass::High];
        let scales = [EntropyScale::Micro, EntropyScale::Meso, EntropyScale::Macro];
        let totals = [256_u64, 4 * 1024, 64 * 1024, 1024 * 1024, 64 * 1024 * 1024];

        for profile in profiles {
            for api in api_contexts {
                for entropy in entropies {
                    for scale in scales {
                        for total in totals {
                            let mut workload = WorkloadShape::new(api, total as usize);
                            workload.entropy = entropy;
                            workload.scale = scale;
                            let plan = plan_histogram(&profile, &workload);
                            assert_ne!(
                                plan.strategy,
                                HistogramStrategy::Avx2Stripe4U32,
                                "Avx2Stripe4U32 selected by planner: backend={:?}, api={:?}, \
                                 entropy={:?}, scale={:?}, total={total}",
                                profile.backend,
                                api,
                                entropy,
                                scale,
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn backend_support_is_honest_about_future_backends() {
        // AVX2 has full coverage across every primitive family today.
        let avx2 = backend_kernel_support(Backend::Avx2);
        assert_eq!(avx2.byte_histogram, KernelAvailability::Native);
        assert_eq!(avx2.fingerprint, KernelAvailability::Native);
        assert_eq!(avx2.byte_class, KernelAvailability::Native);
        assert_eq!(avx2.sketch, KernelAvailability::Native);
        assert_eq!(avx2.runlength, KernelAvailability::Native);
        assert_eq!(avx2.similarity, KernelAvailability::Native);
        // SHA-NI is OPTIONAL on x86 — Haswell through Rocket Lake have
        // AVX2 but no SHA-NI. The matrix reports ScalarFallback to
        // avoid overclaim; the runtime sha256 dispatch still picks
        // SHA-NI when is_x86_feature_detected!("sha") returns true.
        assert_eq!(avx2.hash_sha256, KernelAvailability::ScalarFallback);

        // NEON ships native byte-class (kernels::neon), fingerprint
        // (#37), sketch CRC32C via __crc32cw (#45), runlength,
        // similarity. Byte-histogram is the remaining gap on NEON.
        // hash_sha256 is ScalarFallback because FEAT_SHA2 is OPTIONAL
        // on ARMv8-A (the runtime dispatcher still picks it when
        // is_aarch64_feature_detected!("sha2") returns true).
        let neon = backend_kernel_support(Backend::Neon);
        assert_eq!(neon.byte_class, KernelAvailability::Native);
        assert_eq!(neon.fingerprint, KernelAvailability::Native);
        assert_eq!(neon.sketch, KernelAvailability::Native);
        assert_eq!(neon.runlength, KernelAvailability::Native);
        assert_eq!(neon.similarity, KernelAvailability::Native);
        assert_eq!(neon.hash_sha256, KernelAvailability::ScalarFallback);
        assert_eq!(neon.byte_histogram, KernelAvailability::ScalarFallback);

        // AVX-512 ships native byte-class (#38, #47, #60). The bit-marginal
        // BITALG/GFNI kernels (#51/#52) emit 8 marginals, NOT a 256-bin
        // histogram, so byte_histogram stays ScalarFallback. fingerprint
        // is the remaining gap. sketch/runlength/similarity/sha256
        // composition: SHA-NI is OPTIONAL even on AVX-512 hosts (Skylake-X
        // notably lacks it), so hash_sha256 stays ScalarFallback here too.
        let avx512 = backend_kernel_support(Backend::Avx512);
        assert_eq!(avx512.byte_class, KernelAvailability::Native);
        assert_eq!(avx512.byte_histogram, KernelAvailability::ScalarFallback);
        assert_eq!(avx512.fingerprint, KernelAvailability::ScalarFallback);
        assert_eq!(avx512.hash_sha256, KernelAvailability::ScalarFallback);

        // SVE / SVE2 ship native byte-class classify + UTF-8 (#41),
        // runlength (#41 svcntp_b8), similarity (#41 svmla_f32_m). NEON-
        // mandatory paths (fingerprint, sketch CRC32C via FEAT_CRC32 —
        // mandatory on ARMv8.1+, which SVE always implies) compose cleanly.
        // Byte-histogram is the remaining gap (svhistseg_u8 is future
        // work). hash_sha256 stays ScalarFallback because FEAT_SHA2 is
        // independent of SVE/NEON; runtime dispatcher handles it.
        for backend in [Backend::Sve, Backend::Sve2] {
            let support = backend_kernel_support(backend);
            assert_eq!(support.byte_class, KernelAvailability::Native);
            assert_eq!(support.fingerprint, KernelAvailability::Native);
            assert_eq!(support.sketch, KernelAvailability::Native);
            assert_eq!(support.runlength, KernelAvailability::Native);
            assert_eq!(support.similarity, KernelAvailability::Native);
            assert_eq!(support.hash_sha256, KernelAvailability::ScalarFallback);
            assert_eq!(support.byte_histogram, KernelAvailability::ScalarFallback);
        }

        // Scalar always reports Native for everything — it's the
        // reference implementation set, including SHA-256.
        let scalar = backend_kernel_support(Backend::Scalar);
        assert_eq!(scalar.hash_sha256, KernelAvailability::Native);
    }

    #[cfg(all(feature = "std", target_os = "linux"))]
    #[test]
    fn parses_linux_cache_sizes() {
        assert_eq!(super::parse_size_bytes("48K"), Some(48 * 1024));
        assert_eq!(super::parse_size_bytes("1280K"), Some(1280 * 1024));
        assert_eq!(super::parse_size_bytes("30M"), Some(30 * 1024 * 1024));
        assert_eq!(super::parse_size_bytes("n/a"), None);
    }

    // ============================================================================
    // Rule-table architecture tests (added by the #28 redesign).
    // ============================================================================

    #[test]
    fn planner_rule_names_are_unique() {
        use crate::dispatch::planner::rules::RULES;
        let mut names: Vec<&'static str> = RULES.iter().map(|r| r.name).collect();
        names.sort_unstable();
        let total = names.len();
        names.dedup();
        assert_eq!(
            names.len(),
            total,
            "duplicate rule names in RULES — names must be unique for telemetry to work"
        );
    }

    #[test]
    fn planner_rule_table_has_a_terminal_match() {
        // The last rule in RULES must be the general fallback; otherwise
        // some workload could fall off the end and trigger the unreachable
        // path in plan_histogram.
        use crate::dispatch::planner::rules::{RULE_GENERAL_FALLBACK, RULES};
        let last = RULES.last().expect("RULES is empty");
        assert_eq!(
            last.name, RULE_GENERAL_FALLBACK.name,
            "last rule must be the general fallback"
        );
    }

    #[test]
    fn planner_general_fallback_predicate_always_matches() {
        use crate::dispatch::planner::rules::RULE_GENERAL_FALLBACK;
        use crate::dispatch::planner::signals::Signals;
        let profile = ProcessorProfile::portable();
        let workload = WorkloadShape::new(ApiContext::Block, 0);
        let signals = Signals::derive(&profile, &workload);
        assert!((RULE_GENERAL_FALLBACK.predicate)(
            &profile, &workload, &signals
        ));
    }

    #[test]
    fn planner_fallback_source_implies_low_confidence() {
        // Rules tagged Fallback should emit a confidence at or below the
        // documented fallback floor; rules above the floor should never
        // be tagged Fallback. This keeps the bench-history confidence
        // distribution coherent with the source classification.
        use crate::dispatch::planner::consts::CONFIDENCE_FALLBACK_FLOOR;
        use crate::dispatch::planner::rules::RULES;
        use crate::dispatch::planner::signals::Signals;

        let profile = ProcessorProfile::portable();
        let workload = WorkloadShape::new(ApiContext::Block, 0);
        let signals = Signals::derive(&profile, &workload);
        for rule in RULES {
            if rule.source == PlannerConfidenceSource::Fallback {
                let plan = (rule.builder)(&profile, &workload, &signals);
                assert!(
                    plan.confidence_q8 <= CONFIDENCE_FALLBACK_FLOOR,
                    "rule {} is tagged Fallback but emits confidence {} > floor {}",
                    rule.name,
                    plan.confidence_q8,
                    CONFIDENCE_FALLBACK_FLOOR,
                );
            }
        }
    }

    #[test]
    fn planner_traced_returns_winner_in_trace() {
        use crate::dispatch::planner::plan_histogram_traced;
        let profile = ProcessorProfile::portable();
        let workload = WorkloadShape::new(ApiContext::Block, 0);
        let (plan, trace) = plan_histogram_traced(&profile, &workload);
        let winner = trace.last().expect("trace must be non-empty");
        assert!(winner.matched, "trace's last entry must be the winner");
        // The winning plan's reason must equal the winning rule's reason.
        // Look up the winning rule by name.
        use crate::dispatch::planner::rules::RULES;
        let winning_rule = RULES
            .iter()
            .find(|r| r.name == winner.name)
            .expect("winner not in RULES");
        assert_eq!(plan.reason, winning_rule.reason);
    }

    #[test]
    fn planner_traced_records_misses_before_winner() {
        // For a workload that should reach a late rule, the trace should
        // contain at least one false entry before the winning true.
        use crate::dispatch::planner::plan_histogram_traced;
        let profile = ProcessorProfile::portable();
        let workload = WorkloadShape::new(ApiContext::Block, 32 * 1024);
        // 32 KiB block is below the LARGE threshold, so it falls through
        // most rules and lands in the general fallback.
        let (_plan, trace) = plan_histogram_traced(&profile, &workload);
        assert!(trace.len() > 1, "trace must show fall-through misses");
        let winner_pos = trace
            .iter()
            .position(|d| d.matched)
            .expect("winner present");
        // All entries before the winner must be misses.
        for entry in &trace[..winner_pos] {
            assert!(
                !entry.matched,
                "rule {} matched before the winner",
                entry.name
            );
        }
    }

    // ============================================================================
    // Tunes (#27) — persisted planner overrides
    // ============================================================================

    #[test]
    fn tunes_default_equals_consts() {
        // Tunes::DEFAULT is built from the const values; this test catches
        // accidental drift if a const is updated without updating the
        // corresponding Tunes::DEFAULT field (or vice versa).
        use crate::dispatch::planner::consts;
        use crate::dispatch::planner::tunes::Tunes;
        let t = Tunes::DEFAULT;
        assert_eq!(
            t.block_threshold_micro_bytes,
            consts::BLOCK_THRESHOLD_MICRO_BYTES
        );
        assert_eq!(
            t.block_threshold_large_bytes,
            consts::BLOCK_THRESHOLD_LARGE_BYTES
        );
        assert_eq!(
            t.total_threshold_macro_bytes,
            consts::TOTAL_THRESHOLD_MACRO_BYTES
        );
        assert_eq!(t.confidence_deterministic, consts::CONFIDENCE_DETERMINISTIC);
        assert_eq!(
            t.confidence_fallback_floor,
            consts::CONFIDENCE_FALLBACK_FLOOR
        );
        assert_eq!(
            t.confidence_general_fallback,
            consts::CONFIDENCE_GENERAL_FALLBACK
        );
    }

    #[test]
    fn tuned_planner_default_matches_untuned() {
        // plan_histogram(profile, workload) and plan_histogram_tuned(profile,
        // workload, &Tunes::DEFAULT) must be bit-identical.
        use crate::dispatch::planner::plan_histogram_tuned;
        use crate::dispatch::planner::tunes::Tunes;
        let profile = ProcessorProfile::portable();
        let workloads = [
            WorkloadShape::new(ApiContext::Block, 32 * 1024),
            WorkloadShape::new(ApiContext::Random, 1024 * 1024),
            WorkloadShape::new(ApiContext::Sequential, 256 * 1024 * 1024),
        ];
        for w in workloads {
            let untuned = plan_histogram(&profile, &w);
            let tuned = plan_histogram_tuned(&profile, &w, &Tunes::DEFAULT);
            assert_eq!(untuned, tuned, "default tunes must match untuned path");
        }
    }

    #[test]
    fn tunes_override_can_flip_micro_threshold_to_change_strategy() {
        // A 6 KiB block on a single thread normally falls past the micro
        // ladder (call_bytes > 4 KiB) and lands further down the table.
        // Overriding block_threshold_micro_bytes to 8 KiB pulls it into the
        // micro-default branch (DirectU64).
        use crate::dispatch::planner::plan_histogram_tuned;
        use crate::dispatch::planner::tunes::Tunes;
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 6 * 1024);
        workload.entropy = EntropyClass::High;
        workload.scale = EntropyScale::Flat;

        let baseline = plan_histogram(&profile, &workload);
        let overridden = plan_histogram_tuned(
            &profile,
            &workload,
            &Tunes::DEFAULT.with_block_threshold_micro_bytes(8 * 1024),
        );

        // The override widens the micro band; the workload now lands on
        // DirectU64 with deterministic confidence.
        assert_eq!(overridden.strategy, HistogramStrategy::DirectU64);
        assert_eq!(overridden.confidence_q8, 255);

        // Baseline (default 4 KiB micro threshold) routes a 6 KiB
        // high-entropy block through the high-entropy ladder (small high
        // entropy -> DirectU64 with confidence_calibrated_boundary=220).
        // The strategies happen to agree here, but the confidences differ
        // because the override hit the deterministic micro rule, not the
        // calibrated-boundary rule.
        assert_ne!(baseline.confidence_q8, overridden.confidence_q8);
        assert_ne!(baseline.reason, overridden.reason);
    }

    #[test]
    fn tunes_override_can_promote_a_workload_into_repeated_regions() {
        // A 64 KiB total at the LARGE threshold normally fires the
        // repeated-regions rule on cache=Reused. Cutting the threshold
        // to 32 KiB pulls a 48 KiB workload into the same rule.
        use crate::dispatch::planner::plan_histogram_tuned;
        use crate::dispatch::planner::tunes::Tunes;
        let profile = ProcessorProfile::portable();
        let mut workload = WorkloadShape::new(ApiContext::Block, 48 * 1024);
        workload.cache_state = CacheState::Reused;
        workload.entropy = EntropyClass::Mixed;

        let baseline = plan_histogram(&profile, &workload);
        let overridden = plan_histogram_tuned(
            &profile,
            &workload,
            &Tunes::DEFAULT.with_block_threshold_large_bytes(32 * 1024),
        );

        // Default LARGE = 64 KiB → 48 KiB total doesn't hit
        // repeated-regions; falls to the general fallback.
        assert_ne!(baseline.strategy, HistogramStrategy::AdaptiveFileCached64K);
        // Overridden LARGE = 32 KiB → 48 KiB ≥ 32 KiB → repeated-regions fires.
        assert_eq!(
            overridden.strategy,
            HistogramStrategy::AdaptiveFileCached64K
        );
    }

    #[cfg(feature = "tunes-json")]
    #[test]
    fn tunes_from_json_overrides_named_fields() {
        use crate::dispatch::planner::tunes::Tunes;
        let json = r#"{
            "block_threshold_large_bytes": 32768,
            "confidence_general_fallback": 100
        }"#;
        let tunes = Tunes::from_json(json).expect("valid JSON");
        assert_eq!(tunes.block_threshold_large_bytes, 32 * 1024);
        assert_eq!(tunes.confidence_general_fallback, 100);
        // Untouched fields keep their default value.
        assert_eq!(
            tunes.confidence_deterministic,
            Tunes::DEFAULT.confidence_deterministic
        );
    }

    #[cfg(feature = "tunes-json")]
    #[test]
    fn tunes_from_json_rejects_unknown_fields() {
        use crate::dispatch::planner::tunes::{TuneLoadError, Tunes};
        let json = r#"{ "not_a_real_field": 42 }"#;
        let err = Tunes::from_json(json).expect_err("must reject unknown field");
        assert!(matches!(err, TuneLoadError::UnknownField(name) if name == "not_a_real_field"));
    }

    #[cfg(feature = "tunes-json")]
    #[test]
    fn tunes_from_json_rejects_out_of_range_u8() {
        use crate::dispatch::planner::tunes::{TuneLoadError, Tunes};
        let json = r#"{ "confidence_general_fallback": 9999 }"#;
        let err = Tunes::from_json(json).expect_err("must reject out-of-range u8");
        assert!(matches!(err, TuneLoadError::OutOfRange { .. }));
    }

    #[cfg(feature = "tunes-json")]
    #[test]
    fn tunes_from_json_empty_object_yields_default() {
        use crate::dispatch::planner::tunes::Tunes;
        let tunes = Tunes::from_json("{}").expect("empty object is valid");
        assert_eq!(tunes, Tunes::DEFAULT);
    }

    #[cfg(feature = "tunes-json")]
    #[test]
    fn tunes_example_json_in_docs_parses_and_matches_default() {
        // The example file at docs/examples/planner-tunes.json is the
        // canonical reference for callers building bench-calibrate output.
        // Keep it loadable and aligned with the current Tunes schema by
        // testing it on every change.
        use crate::dispatch::planner::tunes::Tunes;
        // Path is relative to the crate root (the test runner's CWD).
        let json = std::fs::read_to_string("../../docs/examples/planner-tunes.json")
            .expect("docs/examples/planner-tunes.json must exist");
        let tunes = Tunes::from_json(&json).expect("example tunes file must parse cleanly");
        // The example file lists every field at its compile-time default;
        // loading it should be byte-identical to Tunes::DEFAULT.
        assert_eq!(tunes, Tunes::DEFAULT);
    }
}

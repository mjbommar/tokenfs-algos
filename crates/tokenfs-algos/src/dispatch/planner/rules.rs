//! Rule registry for byte-histogram planning.
//!
//! Every rule is a `pub(crate) const Rule` item. The [`RULES`] slice
//! lists them in priority order — first match wins. To add a rule:
//!
//! 1. Pick a kebab-case `name`. It will appear in trace output and bench
//!    history; choose something stable.
//! 2. Pick a `PlannerConfidenceSource` value: `CalibrationRule` if the
//!    rule fires only on workload classes the bench history has measured
//!    (most paper/F22/rootfs rules); `StaticRule` for hand-tuned heuristics
//!    that have at least one bench sweep behind them; `Fallback` for
//!    catch-all rules with `confidence ≤ tunes.confidence_fallback_floor`.
//! 3. Write a `predicate` that is a pure function of `(profile, workload,
//!    signals)`.
//! 4. Write a `builder` that constructs the plan body using the active
//!    tune table — `s.tunes.field_name` instead of `consts::FIELD_NAME`.
//!    Avoid raw integer literals in the builder; if you need a new
//!    threshold, add it to [`super::tunes::Tunes`] (and its
//!    [`super::consts`] default) so it stays override-able.
//! 5. Append `RULE_FOO` to [`RULES`] at the right precedence position.
//! 6. If the rule names a calibration class, add a corresponding regression
//!    test in `crate::dispatch::tests` that pins its expected output.
//!
//! Rule order in this file matches rule order in [`RULES`] which matches
//! the priority order of the legacy `plan_histogram` chain. This is
//! deliberate — the redesign is a no-op against the existing 24 planner
//! regression tests; behaviour preservation is asserted by those tests.

use crate::dispatch::{
    ApiContext, Backend, CacheState, ContentKind, EntropyClass, EntropyScale, HistogramPlan,
    HistogramStrategy, PlannerConfidenceSource, ProcessorProfile, ReadPattern, SourceHint,
    WorkloadShape,
};

use super::rule::{Rule, build};
use super::signals::Signals;
use super::tunes::Tunes;

// ============================================================================
// Calibration-rooted rules
// ============================================================================

/// F22/rootfs random 4 KiB calibration: direct counting wins over
/// stripe8 on this specific workload class.
pub(crate) const RULE_PAPER_RANDOM_4K_MIXED: Rule = Rule {
    name: "paper-random-4k-mixed",
    reason: "F22/rootfs random 4K calibration favored direct counting over stripe8",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |_p, w, s| {
        w.source_hint == SourceHint::PaperExtent
            && s.random_like
            && s.call_bytes <= s.tunes.block_threshold_micro_bytes
            && w.total_bytes >= s.tunes.block_threshold_large_bytes
            && s.mixedish_entropy
    },
    builder: |_p, w, s| {
        build(
            &RULE_PAPER_RANDOM_4K_MIXED,
            HistogramStrategy::DirectU64,
            w.chunk_bytes,
            0,
            s.tunes.confidence_rule_normal,
        )
    },
};

// ----------------------------------------------------------------------------
// Single-thread micro reads (call_bytes ≤ tunes.block_threshold_micro_bytes)
// ----------------------------------------------------------------------------

pub(crate) const RULE_MICRO_LOW_ENTROPY: Rule = Rule {
    name: "micro-low-entropy",
    reason: "small low-entropy reads favored the dedicated run-length kernel",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        s.threads <= 1
            && s.call_bytes <= s.tunes.block_threshold_micro_bytes
            && w.entropy == EntropyClass::Low
            && s.call_bytes > 1
    },
    builder: |_p, w, s| {
        build(
            &RULE_MICRO_LOW_ENTROPY,
            HistogramStrategy::RunLengthU64,
            w.chunk_bytes,
            0,
            s.tunes.confidence_calibrated,
        )
    },
};

pub(crate) const RULE_MICRO_STRUCTURED_4K: Rule = Rule {
    name: "micro-structured-4k",
    reason: "small structured 4K reads favored pinned structured-data kernels",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, _w, s| {
        s.threads <= 1
            && s.call_bytes <= s.tunes.block_threshold_micro_bytes
            && s.call_bytes >= s.tunes.block_threshold_structured_floor_bytes
            && s.mixedish_entropy
            && s.structured_scale
    },
    builder: |p, w, s| {
        let strategy = if p.backend == Backend::Avx2 && w.scale == EntropyScale::Meso {
            HistogramStrategy::Avx2PaletteU32
        } else if w.scale == EntropyScale::Macro {
            HistogramStrategy::Stripe8U32
        } else {
            HistogramStrategy::Stripe4U32
        };
        build(
            &RULE_MICRO_STRUCTURED_4K,
            strategy,
            w.chunk_bytes,
            0,
            s.tunes.confidence_calibrated_normal,
        )
    },
};

pub(crate) const RULE_MICRO_HIGH_ENTROPY_RANDOM: Rule = Rule {
    name: "micro-high-entropy-random",
    reason: "small high-entropy meso random reads favored striped private buckets",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        s.threads <= 1
            && s.call_bytes <= s.tunes.block_threshold_micro_bytes
            && s.random_like
            && s.call_bytes >= s.tunes.block_threshold_structured_floor_bytes
            && w.entropy == EntropyClass::High
            && w.scale == EntropyScale::Meso
    },
    builder: |_p, w, s| {
        build(
            &RULE_MICRO_HIGH_ENTROPY_RANDOM,
            HistogramStrategy::Stripe4U32,
            w.chunk_bytes,
            0,
            s.tunes.confidence_rule_lower,
        )
    },
};

pub(crate) const RULE_MICRO_DEFAULT: Rule = Rule {
    name: "micro-default",
    reason: "micro reads should avoid classifier overhead",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, _w, s| s.threads <= 1 && s.call_bytes <= s.tunes.block_threshold_micro_bytes,
    builder: |_p, w, s| {
        build(
            &RULE_MICRO_DEFAULT,
            HistogramStrategy::DirectU64,
            w.chunk_bytes,
            0,
            s.tunes.confidence_deterministic,
        )
    },
};

// ----------------------------------------------------------------------------
// Tiny random reads
// ----------------------------------------------------------------------------

pub(crate) const RULE_TINY_RANDOM: Rule = Rule {
    name: "tiny-random",
    reason: "tiny random reads are dominated by call overhead; avoid adaptive sampling",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, _w, s| s.random_like && s.call_bytes <= s.tunes.block_threshold_micro_bytes,
    builder: |_p, w, s| {
        build(
            &RULE_TINY_RANDOM,
            HistogramStrategy::DirectU64,
            w.chunk_bytes,
            0,
            s.tunes.confidence_deterministic,
        )
    },
};

// ----------------------------------------------------------------------------
// Large low-entropy
// ----------------------------------------------------------------------------

pub(crate) const RULE_LARGE_LOW_ENTROPY_AVX2: Rule = Rule {
    name: "large-low-entropy-avx2",
    reason: "AVX2 palette counting dominated large low-entropy calibration slices",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |p, w, s| {
        w.entropy == EntropyClass::Low
            && p.backend == Backend::Avx2
            && w.total_bytes >= s.tunes.block_threshold_large_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_LARGE_LOW_ENTROPY_AVX2,
            HistogramStrategy::Avx2PaletteU32,
            w.chunk_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_high_calibrated,
        )
    },
};

pub(crate) const RULE_LARGE_LOW_ENTROPY: Rule = Rule {
    name: "large-low-entropy",
    reason: "large low-entropy inputs need the dedicated low-entropy fast path",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.entropy == EntropyClass::Low && w.total_bytes >= s.tunes.block_threshold_large_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_LARGE_LOW_ENTROPY,
            HistogramStrategy::AdaptiveLowEntropyFast,
            w.chunk_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_calibrated,
        )
    },
};

pub(crate) const RULE_PAPER_LARGE_MIXED: Rule = Rule {
    name: "paper-large-mixed",
    reason: "F22/rootfs sidecar calibration favored stripe8 on large mixed extents",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |_p, w, s| {
        w.source_hint == SourceHint::PaperExtent
            && w.total_bytes >= s.tunes.block_threshold_large_bytes
            && matches!(w.entropy, EntropyClass::Mixed | EntropyClass::Medium)
    },
    builder: |_p, w, s| {
        build(
            &RULE_PAPER_LARGE_MIXED,
            HistogramStrategy::Stripe8U32,
            w.chunk_bytes,
            0,
            s.tunes.confidence_calibrated_normal,
        )
    },
};

pub(crate) const RULE_AVX2_PALETTE_MEDIUM_MICRO_BINARY: Rule = Rule {
    name: "avx2-palette-medium-micro-binary",
    reason: "AVX2 palette counting is worth trying on medium-entropy micro-pattern binary inputs",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |p, w, s| {
        p.backend == Backend::Avx2
            && w.content == ContentKind::Binary
            && w.entropy == EntropyClass::Medium
            && w.scale == EntropyScale::Micro
            && w.total_bytes >= s.tunes.block_threshold_avx2_palette_micro_binary_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_AVX2_PALETTE_MEDIUM_MICRO_BINARY,
            HistogramStrategy::Avx2PaletteU32,
            w.chunk_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_tentative,
        )
    },
};

pub(crate) const RULE_REPEATED_REGIONS: Rule = Rule {
    name: "repeated-regions",
    reason: "repeated file access can amortize one file-level decision",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.cache_state == CacheState::Reused && w.total_bytes >= s.tunes.block_threshold_large_bytes
    },
    builder: |_p, _w, s| {
        build(
            &RULE_REPEATED_REGIONS,
            HistogramStrategy::AdaptiveFileCached64K,
            s.tunes.chunk_parallel_default_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_cache_amortized,
        )
    },
};

// ----------------------------------------------------------------------------
// Parallel rules
// ----------------------------------------------------------------------------

pub(crate) const RULE_PARALLEL_MACRO_MIXED_AVX2: Rule = Rule {
    name: "parallel-macro-mixed-avx2",
    reason: "parallel macro-mixed calibration favored AVX2 palette counting",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |p, w, s| {
        (s.threads > 1 || w.context == ApiContext::Parallel)
            && s.mixedish_entropy
            && w.content != ContentKind::Text
            && w.scale == EntropyScale::Macro
            && p.backend == Backend::Avx2
    },
    builder: |_p, _w, s| {
        let confidence = if s.oversubscribed {
            s.tunes.confidence_parallel_oversubscribed
        } else {
            s.tunes.confidence_rule_normal
        };
        build(
            &RULE_PARALLEL_MACRO_MIXED_AVX2,
            HistogramStrategy::Avx2PaletteU32,
            s.tunes.chunk_parallel_default_bytes,
            s.tunes.sample_adaptive_default_bytes,
            confidence,
        )
    },
};

pub(crate) const RULE_PARALLEL_MESO_STRUCTURED: Rule = Rule {
    name: "parallel-meso-structured",
    reason: "parallel meso-structured calibration favored striped private buckets",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |_p, w, s| {
        (s.threads > 1 || w.context == ApiContext::Parallel)
            && s.mixedish_entropy
            && w.content != ContentKind::Text
            && w.scale == EntropyScale::Meso
    },
    builder: |_p, _w, s| {
        let strategy = if s.threads >= s.tunes.parallel_stripe4_thread_floor {
            HistogramStrategy::Stripe4U32
        } else {
            HistogramStrategy::Stripe8U32
        };
        let confidence = if s.oversubscribed {
            s.tunes.confidence_parallel_oversubscribed
        } else {
            s.tunes.confidence_rule_normal
        };
        build(
            &RULE_PARALLEL_MESO_STRUCTURED,
            strategy,
            s.tunes.chunk_parallel_default_bytes,
            0,
            confidence,
        )
    },
};

pub(crate) const RULE_PARALLEL_TINY_TEXT: Rule = Rule {
    name: "parallel-tiny-text",
    reason: "tiny parallel text scans favored direct counting",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        (s.threads > 1 || w.context == ApiContext::Parallel)
            && w.content == ContentKind::Text
            && w.total_bytes <= s.tunes.block_threshold_micro_bytes
    },
    builder: |_p, _w, s| {
        let confidence = if s.oversubscribed {
            s.tunes.confidence_parallel_oversubscribed
        } else {
            s.tunes.confidence_rule_normal
        };
        build(
            &RULE_PARALLEL_TINY_TEXT,
            HistogramStrategy::DirectU64,
            s.tunes.chunk_parallel_default_bytes,
            0,
            confidence,
        )
    },
};

pub(crate) const RULE_PARALLEL_TEXT: Rule = Rule {
    name: "parallel-text",
    reason: "parallel text scans favored simple private tables over chunked adaptation",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        (s.threads > 1 || w.context == ApiContext::Parallel) && w.content == ContentKind::Text
    },
    builder: |_p, _w, s| {
        let confidence = if s.oversubscribed {
            s.tunes.confidence_parallel_oversubscribed
        } else {
            s.tunes.confidence_rule_normal
        };
        build(
            &RULE_PARALLEL_TEXT,
            HistogramStrategy::LocalU32,
            s.tunes.chunk_parallel_default_bytes,
            0,
            confidence,
        )
    },
};

pub(crate) const RULE_PARALLEL_HIGH_ENTROPY: Rule = Rule {
    name: "parallel-high-entropy",
    reason: "parallel high-entropy scans should avoid adaptive chunk over-selection",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        (s.threads > 1 || w.context == ApiContext::Parallel) && w.entropy == EntropyClass::High
    },
    builder: |_p, w, s| {
        let strategy = if w.scale == EntropyScale::Meso
            && s.threads >= s.tunes.parallel_stripe4_thread_floor
        {
            HistogramStrategy::LocalU32
        } else {
            HistogramStrategy::DirectU64
        };
        let confidence = if s.oversubscribed {
            s.tunes.confidence_parallel_oversubscribed_tentative
        } else {
            s.tunes.confidence_calibrated_normal
        };
        build(
            &RULE_PARALLEL_HIGH_ENTROPY,
            strategy,
            s.tunes.chunk_parallel_default_bytes,
            0,
            confidence,
        )
    },
};

pub(crate) const RULE_PARALLEL_DEFAULT: Rule = Rule {
    name: "parallel-default",
    reason: "parallel mixed scans use a cheap run sentinel instead of default chunked planning",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| s.threads > 1 || w.context == ApiContext::Parallel,
    builder: |_p, _w, s| {
        let confidence = if s.oversubscribed {
            s.tunes.confidence_parallel_oversubscribed_tentative
        } else {
            s.tunes.confidence_real_file
        };
        build(
            &RULE_PARALLEL_DEFAULT,
            HistogramStrategy::AdaptiveRunSentinel4K,
            s.tunes.chunk_parallel_default_bytes,
            s.tunes.sample_adaptive_default_bytes,
            confidence,
        )
    },
};

// ----------------------------------------------------------------------------
// Real-file rules
// ----------------------------------------------------------------------------

pub(crate) const RULE_REAL_HIGH_ENTROPY_FLAT: Rule = Rule {
    name: "real-high-entropy-flat",
    reason: "real high-entropy flat files favored direct counting in calibration",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |_p, w, s| {
        w.source_hint == SourceHint::RealFile
            && w.content == ContentKind::Binary
            && w.entropy == EntropyClass::High
            && w.scale == EntropyScale::Flat
            && w.total_bytes >= s.tunes.block_threshold_large_bytes
            && !s.random_like
    },
    builder: |_p, w, s| {
        build(
            &RULE_REAL_HIGH_ENTROPY_FLAT,
            HistogramStrategy::DirectU64,
            w.chunk_bytes,
            0,
            s.tunes.confidence_real_file,
        )
    },
};

pub(crate) const RULE_LARGE_UNALIGNED_HIGH_ENTROPY: Rule = Rule {
    name: "large-unaligned-high-entropy",
    reason: "large unaligned high-entropy inputs avoid the direct u64 path",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.alignment_offset >= s.tunes.alignment_penalty_offset_bytes
            && w.entropy == EntropyClass::High
            && w.total_bytes >= s.tunes.block_threshold_large_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_LARGE_UNALIGNED_HIGH_ENTROPY,
            HistogramStrategy::LocalU32,
            w.chunk_bytes,
            0,
            s.tunes.confidence_real_file,
        )
    },
};

pub(crate) const RULE_SEQUENTIAL_MACRO_MIXED_AVX2: Rule = Rule {
    name: "sequential-macro-mixed-avx2",
    reason: "sequential macro-mixed calibration favored AVX2 palette counting",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |p, w, s| {
        p.backend == Backend::Avx2
            && w.scale == EntropyScale::Macro
            && w.entropy == EntropyClass::Mixed
            && s.sequential_like
            && w.total_bytes >= s.tunes.block_threshold_large_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_SEQUENTIAL_MACRO_MIXED_AVX2,
            HistogramStrategy::Avx2PaletteU32,
            w.chunk_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_rule_normal,
        )
    },
};

// ----------------------------------------------------------------------------
// Macro-scale (≥ macro threshold)
// ----------------------------------------------------------------------------

pub(crate) const RULE_REAL_HIGH_ENTROPY_MACRO: Rule = Rule {
    name: "real-high-entropy-macro",
    reason: "real high-entropy file slices favored direct/simple kernels in calibration",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |_p, w, s| {
        w.scale == EntropyScale::Macro
            && w.total_bytes >= s.tunes.total_threshold_macro_bytes
            && w.source_hint == SourceHint::RealFile
            && w.entropy == EntropyClass::High
            && w.content == ContentKind::Binary
    },
    builder: |_p, w, s| {
        build(
            &RULE_REAL_HIGH_ENTROPY_MACRO,
            HistogramStrategy::DirectU64,
            w.chunk_bytes,
            0,
            s.tunes.confidence_tentative,
        )
    },
};

pub(crate) const RULE_MACRO_CHUNKED: Rule = Rule {
    name: "macro-chunked",
    reason: "macro-scale variation benefits from per-region kernel choice",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.scale == EntropyScale::Macro && w.total_bytes >= s.tunes.total_threshold_macro_bytes
    },
    builder: |_p, _w, s| {
        build(
            &RULE_MACRO_CHUNKED,
            HistogramStrategy::AdaptiveChunked64K,
            s.tunes.chunk_parallel_default_bytes,
            s.tunes.sample_prefix_default_bytes,
            s.tunes.confidence_rule_normal,
        )
    },
};

// ----------------------------------------------------------------------------
// Meso-scale rules
// ----------------------------------------------------------------------------

pub(crate) const RULE_HIGH_MESO_BLOCK: Rule = Rule {
    name: "high-meso-block",
    reason: "high-entropy meso real blocks favored stripe8 over adaptive probes",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.entropy == EntropyClass::High
            && w.scale == EntropyScale::Meso
            && w.total_bytes >= s.tunes.block_threshold_large_bytes
            && w.context == ApiContext::Block
    },
    builder: |_p, w, s| {
        build(
            &RULE_HIGH_MESO_BLOCK,
            HistogramStrategy::Stripe8U32,
            w.chunk_bytes,
            0,
            s.tunes.confidence_rule_normal,
        )
    },
};

pub(crate) const RULE_SEQUENTIAL_MESO: Rule = Rule {
    name: "sequential-meso",
    reason: "sequential meso-structured calibration favored stripe8 private buckets",
    source: PlannerConfidenceSource::CalibrationRule,
    predicate: |_p, w, s| {
        w.scale == EntropyScale::Meso
            && s.sequential_like
            && s.call_bytes >= s.tunes.block_threshold_large_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_SEQUENTIAL_MESO,
            HistogramStrategy::Stripe8U32,
            w.chunk_bytes,
            0,
            s.tunes.confidence_calibrated_normal,
        )
    },
};

pub(crate) const RULE_MESO_DETECTOR: Rule = Rule {
    name: "meso-detector",
    reason: "meso-scale structure needs a block-pattern detector",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.scale == EntropyScale::Meso
            && (w.context == ApiContext::Block
                || w.chunk_bytes >= s.tunes.block_threshold_large_bytes)
    },
    builder: |_p, w, s| {
        build(
            &RULE_MESO_DETECTOR,
            HistogramStrategy::AdaptiveMesoDetector,
            w.chunk_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_rule_normal,
        )
    },
};

// ----------------------------------------------------------------------------
// Text rules
// ----------------------------------------------------------------------------

pub(crate) const RULE_TEXT_VERY_LARGE: Rule = Rule {
    name: "text-very-large",
    reason: "large text/file inputs can amortize file-level text detection",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.content == ContentKind::Text
            && (w.total_bytes >= s.tunes.total_threshold_file_cache_bytes
                || w.context == ApiContext::File)
    },
    builder: |_p, _w, s| {
        build(
            &RULE_TEXT_VERY_LARGE,
            HistogramStrategy::AdaptiveFileCached64K,
            s.tunes.chunk_parallel_default_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_rule_normal,
        )
    },
};

pub(crate) const RULE_TEXT_LARGE: Rule = Rule {
    name: "text-large",
    reason: "large text inputs benefit from the ASCII-biased fast path",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.content == ContentKind::Text && w.total_bytes >= s.tunes.total_threshold_ascii_fast_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_TEXT_LARGE,
            HistogramStrategy::AdaptiveAsciiFast,
            w.chunk_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_calibrated_normal,
        )
    },
};

pub(crate) const RULE_TEXT_MEDIUM: Rule = Rule {
    name: "text-medium",
    reason: "medium text inputs favored local private tables in size sweeps",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.content == ContentKind::Text
            && w.total_bytes >= s.tunes.block_threshold_avx2_palette_micro_binary_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_TEXT_MEDIUM,
            HistogramStrategy::LocalU32,
            w.chunk_bytes,
            0,
            s.tunes.confidence_text_probe,
        )
    },
};

// ----------------------------------------------------------------------------
// High-entropy fallback ladder
// ----------------------------------------------------------------------------

pub(crate) const RULE_HIGH_ENTROPY_SMALL: Rule = Rule {
    name: "high-entropy-small",
    reason: "small high-entropy inputs favored the direct path",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.entropy == EntropyClass::High
            && w.total_bytes <= s.tunes.block_threshold_high_entropy_direct_ceiling_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_HIGH_ENTROPY_SMALL,
            HistogramStrategy::DirectU64,
            w.chunk_bytes,
            0,
            s.tunes.confidence_calibrated_boundary,
        )
    },
};

pub(crate) const RULE_HIGH_ENTROPY_MID: Rule = Rule {
    name: "high-entropy-mid",
    reason: "mid-sized high-entropy inputs favored local private tables",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        w.entropy == EntropyClass::High
            && w.total_bytes <= s.tunes.total_threshold_high_entropy_local_ceiling_bytes
    },
    builder: |_p, w, s| {
        build(
            &RULE_HIGH_ENTROPY_MID,
            HistogramStrategy::LocalU32,
            w.chunk_bytes,
            0,
            s.tunes.confidence_rule_low,
        )
    },
};

pub(crate) const RULE_HIGH_ENTROPY_LARGE: Rule = Rule {
    name: "high-entropy-large",
    reason: "large high-entropy inputs should skip low-entropy/text probes",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, _s| w.entropy == EntropyClass::High,
    builder: |_p, w, s| {
        build(
            &RULE_HIGH_ENTROPY_LARGE,
            HistogramStrategy::AdaptiveHighEntropySkip,
            w.chunk_bytes,
            s.tunes.sample_adaptive_default_bytes,
            s.tunes.confidence_tentative,
        )
    },
};

// ----------------------------------------------------------------------------
// Sequential default
// ----------------------------------------------------------------------------

pub(crate) const RULE_SEQUENTIAL_LARGE: Rule = Rule {
    name: "sequential-large",
    reason: "sequential reads can update choices at region boundaries",
    source: PlannerConfidenceSource::StaticRule,
    predicate: |_p, w, s| {
        matches!(
            w.read_pattern,
            ReadPattern::Sequential | ReadPattern::Readahead
        ) && w.total_bytes >= s.tunes.block_threshold_large_bytes
    },
    builder: |_p, _w, s| {
        build(
            &RULE_SEQUENTIAL_LARGE,
            HistogramStrategy::AdaptiveSequentialOnline64K,
            s.tunes.chunk_parallel_default_bytes,
            s.tunes.sample_prefix_default_bytes,
            s.tunes.confidence_sequential_probe,
        )
    },
};

// ----------------------------------------------------------------------------
// General fallback
// ----------------------------------------------------------------------------

pub(crate) const RULE_GENERAL_FALLBACK: Rule = Rule {
    name: "general-fallback",
    reason: "general-purpose fallback with a bounded prefix classifier",
    source: PlannerConfidenceSource::Fallback,
    predicate: |_p, _w, _s| true,
    builder: |_p, w, s| {
        build(
            &RULE_GENERAL_FALLBACK,
            HistogramStrategy::AdaptivePrefix1K,
            w.chunk_bytes,
            s.tunes.sample_prefix_default_bytes,
            s.tunes.confidence_general_fallback,
        )
    },
};

// ============================================================================
// Rule registry
// ============================================================================

pub(crate) const RULES: &[&Rule] = &[
    &RULE_PAPER_RANDOM_4K_MIXED,
    &RULE_MICRO_LOW_ENTROPY,
    &RULE_MICRO_STRUCTURED_4K,
    &RULE_MICRO_HIGH_ENTROPY_RANDOM,
    &RULE_MICRO_DEFAULT,
    &RULE_TINY_RANDOM,
    &RULE_LARGE_LOW_ENTROPY_AVX2,
    &RULE_LARGE_LOW_ENTROPY,
    &RULE_PAPER_LARGE_MIXED,
    &RULE_AVX2_PALETTE_MEDIUM_MICRO_BINARY,
    &RULE_REPEATED_REGIONS,
    &RULE_PARALLEL_MACRO_MIXED_AVX2,
    &RULE_PARALLEL_MESO_STRUCTURED,
    &RULE_PARALLEL_TINY_TEXT,
    &RULE_PARALLEL_TEXT,
    &RULE_PARALLEL_HIGH_ENTROPY,
    &RULE_PARALLEL_DEFAULT,
    &RULE_REAL_HIGH_ENTROPY_FLAT,
    &RULE_LARGE_UNALIGNED_HIGH_ENTROPY,
    &RULE_SEQUENTIAL_MACRO_MIXED_AVX2,
    &RULE_REAL_HIGH_ENTROPY_MACRO,
    &RULE_MACRO_CHUNKED,
    &RULE_HIGH_MESO_BLOCK,
    &RULE_SEQUENTIAL_MESO,
    &RULE_MESO_DETECTOR,
    &RULE_TEXT_VERY_LARGE,
    &RULE_TEXT_LARGE,
    &RULE_TEXT_MEDIUM,
    &RULE_HIGH_ENTROPY_SMALL,
    &RULE_HIGH_ENTROPY_MID,
    &RULE_HIGH_ENTROPY_LARGE,
    &RULE_SEQUENTIAL_LARGE,
    &RULE_GENERAL_FALLBACK,
];

/// Plans a histogram by walking the rule table with the default tune table.
#[must_use]
pub fn plan_histogram(profile: &ProcessorProfile, workload: &WorkloadShape) -> HistogramPlan {
    plan_histogram_tuned(profile, workload, &Tunes::DEFAULT)
}

/// Plans a histogram against a caller-supplied [`Tunes`] table.
///
/// Use this when host-specific calibration overrides are loaded into a
/// [`Tunes`] value (typically via [`Tunes::from_json`]). Supplying
/// `&Tunes::DEFAULT` reproduces the behavior of [`plan_histogram`].
#[must_use]
pub fn plan_histogram_tuned(
    profile: &ProcessorProfile,
    workload: &WorkloadShape,
    tunes: &Tunes,
) -> HistogramPlan {
    let signals = Signals::derive_with(profile, workload, tunes);
    for rule in RULES {
        if (rule.predicate)(profile, workload, &signals) {
            return (rule.builder)(profile, workload, &signals);
        }
    }
    (RULE_GENERAL_FALLBACK.builder)(profile, workload, &signals)
}

/// Plans a histogram and returns the per-rule trace (default tunes).
#[cfg(feature = "std")]
#[must_use]
pub fn plan_histogram_traced(
    profile: &ProcessorProfile,
    workload: &WorkloadShape,
) -> (
    HistogramPlan,
    Vec<crate::dispatch::planner::rule::RuleDecision>,
) {
    plan_histogram_traced_tuned(profile, workload, &Tunes::DEFAULT)
}

/// Plans a histogram against a custom [`Tunes`] and returns the trace.
#[cfg(feature = "std")]
#[must_use]
pub fn plan_histogram_traced_tuned(
    profile: &ProcessorProfile,
    workload: &WorkloadShape,
    tunes: &Tunes,
) -> (
    HistogramPlan,
    Vec<crate::dispatch::planner::rule::RuleDecision>,
) {
    use crate::dispatch::planner::rule::RuleDecision;
    let signals = Signals::derive_with(profile, workload, tunes);
    let mut trace = Vec::new();
    for (index, rule) in RULES.iter().enumerate() {
        let matched = (rule.predicate)(profile, workload, &signals);
        trace.push(RuleDecision {
            index: index as u16,
            name: rule.name,
            matched,
        });
        if matched {
            return ((rule.builder)(profile, workload, &signals), trace);
        }
    }
    let plan = (RULE_GENERAL_FALLBACK.builder)(profile, workload, &signals);
    (plan, trace)
}

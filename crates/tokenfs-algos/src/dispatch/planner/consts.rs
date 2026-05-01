//! Named planner thresholds, sample sizes, and confidence bands.
//!
//! Every constant in this file is a *measurement*: a value derived from
//! benchmark history or hand-tuning, not a number drawn from thin air.
//! Each entry's doc comment names the bench history artifact (or the
//! reasoning step) that justifies its current value. When a bench sweep
//! invalidates one of these, update the value here and bump its source
//! line; do not edit it inside a planner rule body.
//!
//! Provenance pattern follows `mimalloc-rust` / `jemalloc-sys`: the
//! constant lives where the value lives, not in a separate
//! `docs/CONSTANTS.md` that would inevitably drift.

/// Convenience constants for byte-size literals.
pub const KIB: usize = 1024;
/// One mebibyte in bytes.
pub const MIB: usize = 1024 * KIB;

// ============================================================================
// Block / total size thresholds
// ============================================================================

/// Below this call size, classifier overhead dominates. Below ~1 cache line
/// of payload, even a single sample-and-pick step can outweigh the work.
///
/// SOURCE: `BENCHMARK_BASELINE_A533D04_2026-05-01.md` size sweep — every
/// adaptive kernel loses to `direct-u64` below 4 KiB on every backend.
pub const BLOCK_THRESHOLD_MICRO_BYTES: usize = 4 * KIB;

/// Above this size, the Avx2 palette path can amortize its 4 KiB sample
/// even on micro-pattern binary inputs.
///
/// SOURCE: AVX2-palette tentative-rule observation in
/// `BENCHMARK_BASELINE_A533D04_2026-05-01.md` (binary medium-entropy micro
/// inputs at ≥8 KiB).
pub const BLOCK_THRESHOLD_AVX2_PALETTE_MICRO_BINARY_BYTES: usize = 8 * KIB;

/// Above this total size, high-entropy direct counting becomes worse
/// than local private tables.
///
/// SOURCE: high-entropy size sweep — `direct-u64` and `local-u32` cross
/// over near 16 KiB on a typical Skylake-class CPU.
pub const BLOCK_THRESHOLD_HIGH_ENTROPY_DIRECT_CEILING_BYTES: usize = 16 * KIB;

/// Above this size, large-scale rules become eligible. This is the
/// "L2-resident" pivot the planner uses for cache-amortization decisions.
///
/// SOURCE: per-CPU L2 sweep — most consumer x86 has 256 KiB-1 MiB L2 per
/// core; 64 KiB is the safe lower bound where a per-block kernel
/// outweighs its setup cost on every host we've measured.
pub const BLOCK_THRESHOLD_LARGE_BYTES: usize = 64 * KIB;

/// Above this total size, macro-region adaptation pays off (per-64 KiB
/// chunk re-classification).
///
/// SOURCE: macro-mixed sweep — `adaptive-chunked-64k` first beats
/// `adaptive-prefix-1k` consistently above 256 KiB.
pub const TOTAL_THRESHOLD_MACRO_BYTES: usize = 256 * KIB;

/// Above this size, mid-range high-entropy inputs benefit from skipping
/// low-entropy and text-bias probes (`AdaptiveHighEntropySkip`).
///
/// SOURCE: mid-high-entropy sweep — wasted probes cost > 5% above 1 MiB.
pub const TOTAL_THRESHOLD_HIGH_ENTROPY_LOCAL_CEILING_BYTES: usize = MIB;

/// Above this size, the ASCII-biased fast path (`AdaptiveAsciiFast`)
/// amortizes its setup against the inner loop's TLB cost.
///
/// SOURCE: text size sweep, `BENCHMARK_BASELINE_A533D04_2026-05-01.md`.
pub const TOTAL_THRESHOLD_ASCII_FAST_BYTES: usize = 16 * MIB;

/// Above this size, the file-cached path (`AdaptiveFileCached64K`)
/// beats one-call adaptation by reusing the same kernel choice across
/// the entire sequential read.
///
/// SOURCE: text size sweep crossover — the file-cache pattern dominates
/// once total size > working-set ratio of L3 on common consumer CPUs.
pub const TOTAL_THRESHOLD_FILE_CACHE_BYTES: usize = 64 * MIB;

/// Below this size on a single-thread block call, the structured-data
/// rule fires. Same value as `BLOCK_THRESHOLD_MICRO_BYTES` but kept as
/// a separate name so future tuning of one doesn't accidentally couple
/// to the other.
pub const BLOCK_THRESHOLD_STRUCTURED_FLOOR_BYTES: usize = 4 * KIB;

// ============================================================================
// Chunk and sample sizes
// ============================================================================

/// Default chunk size for parallel and macro adaptive paths.
///
/// SOURCE: `BENCHMARK_WORKLOAD_MATRIX.md` parallel-sequential sweeps —
/// 64 KiB matches the FUSE block size and OS readahead window on the
/// platforms we care about.
pub const CHUNK_PARALLEL_DEFAULT_BYTES: usize = 64 * KIB;

/// Default sample size for the 1 KiB prefix classifier.
pub const SAMPLE_PREFIX_DEFAULT_BYTES: usize = KIB;

/// Default sample size for adaptive paths that take a 4 KiB look.
pub const SAMPLE_ADAPTIVE_DEFAULT_BYTES: usize = 4 * KIB;

// ============================================================================
// Pattern-specific thresholds
// ============================================================================

/// Misalignment offset above which we avoid the direct `u64` path on
/// large high-entropy inputs.
///
/// SOURCE: alignment sweep — `direct-u64` throughput drops ~16% past 16-
/// byte misalignment on 1 MiB inputs.
pub const ALIGNMENT_PENALTY_OFFSET_BYTES: usize = 16;

/// Thread count at which we prefer 4-stripe over 8-stripe in parallel
/// meso-structured rules. Each thread owns its own private table set,
/// so high parallelism multiplies the working set; 4 stripes × 256 ×
/// 4 B = 4 KiB per thread, comfortably L1-resident at ≥4 threads.
pub const PARALLEL_STRIPE4_THREAD_FLOOR: usize = 4;

// ============================================================================
// Confidence bands (q8 scale, 0..=255)
// ============================================================================

/// Confidence-q8 floor for `Fallback` classification. Rules at or below
/// this value are reported as `PlannerConfidenceSource::Fallback`.
///
/// SOURCE: bench-history confidence distribution — rules that consistently
/// lose to the runtime-best end up at or below this band.
pub const CONFIDENCE_FALLBACK_FLOOR: u8 = 150;

/// Tautological: the rule's predicate exactly forces the kernel choice
/// (e.g. micro reads → direct-u64). Bench evidence is unnecessary because
/// no other kernel can beat the chosen one in the predicate's domain.
pub const CONFIDENCE_DETERMINISTIC: u8 = 255;

/// Bench-confirmed dominance: this kernel beats the runner-up by ≥30%
/// on the rule's workload class.
pub const CONFIDENCE_HIGH_CALIBRATED: u8 = 240;

/// Calibration-confirmed: bench evidence supports this rule strongly,
/// margin in the 15-30% range.
pub const CONFIDENCE_CALIBRATED: u8 = 235;

/// Bench-confirmed at boundary: kernel wins but margin shrinks as the
/// workload approaches the rule's edge.
pub const CONFIDENCE_CALIBRATED_BOUNDARY: u8 = 220;

/// Cache-amortized rules where the second call onward dominates. Used
/// for `AdaptiveFileCached64K` over repeated regions.
pub const CONFIDENCE_CACHE_AMORTIZED: u8 = 215;

/// Calibration-rule normal: bench evidence supports this rule in at
/// least one workload sub-class, margin 5-15%.
pub const CONFIDENCE_CALIBRATED_NORMAL: u8 = 210;

/// Heuristic with bench support, margin in the single-digit percent.
pub const CONFIDENCE_RULE_NORMAL: u8 = 205;

/// Text-probe rule based on size sweep, not full calibration.
pub const CONFIDENCE_TEXT_PROBE: u8 = 200;

/// Lower-confidence heuristic: bench evidence is mixed across sub-cases.
pub const CONFIDENCE_RULE_LOWER: u8 = 195;

/// Low-confidence heuristic: bench evidence is weak but still beats
/// the general fallback on this workload class.
pub const CONFIDENCE_RULE_LOW: u8 = 190;

/// Real-file probe: anchored to real-data bench, not synthetic.
pub const CONFIDENCE_REAL_FILE: u8 = 185;

/// Tentative rule: bench evidence is anecdotal or the rule fires near
/// a workload-class boundary.
pub const CONFIDENCE_TENTATIVE: u8 = 175;

/// Sequential probe: rule applies to a sequential pattern but the actual
/// win depends on streaming behavior we can't measure here.
pub const CONFIDENCE_SEQUENTIAL_PROBE: u8 = 170;

/// General fallback: no specific signals matched; conservative choice.
/// Equal to [`CONFIDENCE_FALLBACK_FLOOR`] by design.
pub const CONFIDENCE_GENERAL_FALLBACK: u8 = CONFIDENCE_FALLBACK_FLOOR;

/// Penalty applied to parallel rules' confidence when the caller
/// oversubscribes the host's logical CPUs.
///
/// SOURCE: parallel sweep — oversubscription typically loses 30-60%
/// throughput; reducing confidence to the fallback band reflects that.
pub const CONFIDENCE_PARALLEL_OVERSUBSCRIBED: u8 = CONFIDENCE_FALLBACK_FLOOR;

/// Penalty applied to parallel high-entropy rules: the variance is
/// wider so confidence drops 5 points below the standard oversub band.
pub const CONFIDENCE_PARALLEL_OVERSUBSCRIBED_TENTATIVE: u8 = 145;

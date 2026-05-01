//! Derived workload signals.
//!
//! `Signals` is a small bag of booleans + scalars computed once per
//! `plan_histogram` call and then read by every rule predicate. Pulling
//! these out of the rule bodies means each predicate is a pure function
//! of `(ProcessorProfile, WorkloadShape, Signals)` — trivially testable
//! in isolation.

use crate::dispatch::EntropyClass;
use crate::dispatch::EntropyScale;
use crate::dispatch::{ProcessorProfile, ReadPattern, WorkloadShape};

use super::tunes::Tunes;

/// Pre-computed predicates derived from the planner inputs.
///
/// All fields are documented because each one is a primitive that rules
/// read directly. Adding a new field here is the right way to introduce
/// a new derived signal that multiple rules need.
///
/// `tunes` is a borrowed reference to the active [`Tunes`] table. Rules
/// read tune-able values via `signals.tunes.field_name` so a host-specific
/// override file can adjust planner behavior without recompiling.
#[derive(Copy, Clone, Debug)]
pub struct Signals<'tunes> {
    /// Effective per-call byte count: `chunk_bytes` clamped against
    /// `total_bytes`, with `chunk_bytes == 0` meaning "use total".
    pub call_bytes: usize,

    /// Effective thread count, clamped to ≥1.
    pub threads: usize,

    /// True when the caller asked for more threads than the host's
    /// logical CPU count.
    pub oversubscribed: bool,

    /// Read pattern is one of the random-like patterns
    /// (`RandomTiny | Random | ZipfianHotCold`).
    pub random_like: bool,

    /// Read pattern is one of the sequential-like patterns
    /// (`Sequential | Readahead`).
    pub sequential_like: bool,

    /// Entropy class is `Mixed` or `Medium` (used by several rules to
    /// distinguish "noisy mixed" from pure low/high).
    pub mixedish_entropy: bool,

    /// Scale class is `Meso` or `Macro` (used by the structured-data
    /// rules to gate striped vs. adaptive paths).
    pub structured_scale: bool,

    /// Active tune table. Defaults to [`Tunes::DEFAULT`] via
    /// [`Signals::derive`]; override per-call via [`Signals::derive_with`].
    pub tunes: &'tunes Tunes,
}

impl Signals<'static> {
    /// Derives `Signals` from the planner inputs using the compile-time
    /// [`Tunes::DEFAULT`] tune table.
    #[must_use]
    pub fn derive(profile: &ProcessorProfile, workload: &WorkloadShape) -> Self {
        Self::derive_with(profile, workload, &Tunes::DEFAULT)
    }
}

impl<'tunes> Signals<'tunes> {
    /// Derives `Signals` from the planner inputs against a custom tune table.
    #[must_use]
    pub fn derive_with(
        profile: &ProcessorProfile,
        workload: &WorkloadShape,
        tunes: &'tunes Tunes,
    ) -> Self {
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
        let random_like = matches!(
            workload.read_pattern,
            ReadPattern::RandomTiny | ReadPattern::Random | ReadPattern::ZipfianHotCold
        );
        let sequential_like = matches!(
            workload.read_pattern,
            ReadPattern::Sequential | ReadPattern::Readahead
        );
        let mixedish_entropy =
            matches!(workload.entropy, EntropyClass::Mixed | EntropyClass::Medium);
        let structured_scale = matches!(workload.scale, EntropyScale::Meso | EntropyScale::Macro);

        Self {
            call_bytes,
            threads,
            oversubscribed,
            random_like,
            sequential_like,
            mixedish_entropy,
            structured_scale,
            tunes,
        }
    }
}

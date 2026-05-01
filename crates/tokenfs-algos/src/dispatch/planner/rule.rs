//! `Rule` data type and rule-evaluation infrastructure.
//!
//! A rule is `(name, predicate, builder, source)`:
//!
//! - `name` is a stable telemetry identifier (kebab-case).
//! - `predicate` is a pure function over `(ProcessorProfile, WorkloadShape,
//!   Signals)` returning `true` when the rule should fire.
//! - `builder` constructs the [`HistogramPlan`] body when the predicate
//!   matches. It receives the same arguments because some rules emit
//!   different strategies based on backend / thread count / etc.
//! - `source` declares the [`PlannerConfidenceSource`] without inferring
//!   it from substring matching on the reason text.
//!
//! Rules live in [`super::rules`] as `pub(crate) const` items in source
//! order. Adding a rule is: append a `pub(crate) const RULE_FOO: Rule`
//! item with the right predicate, then add it to [`super::rules::RULES`]
//! at the appropriate precedence position.

use crate::dispatch::{HistogramPlan, PlannerConfidenceSource, ProcessorProfile, WorkloadShape};

use super::signals::Signals;

/// One planner rule. See module docs for the design contract.
pub struct Rule {
    /// Stable kebab-case identifier — used in telemetry, traces, and the
    /// optional planner-explain output.
    pub name: &'static str,

    /// Stable human-readable reason. Mirrored into [`HistogramPlan::reason`]
    /// so end users see the same string the rule documents itself with.
    pub reason: &'static str,

    /// Where the rule's confidence comes from. Set explicitly per rule;
    /// **not** inferred from substring matching on `reason`.
    pub source: PlannerConfidenceSource,

    /// True iff this rule should fire for the given inputs.
    pub predicate: for<'t> fn(&ProcessorProfile, &WorkloadShape, &Signals<'t>) -> bool,

    /// Builds the plan body once the predicate matches. The function is
    /// expected to use the active [`super::tunes::Tunes`] table (via
    /// `signals.tunes`) for any numeric thresholds, sample sizes, or
    /// confidence values it emits — never raw integer literals.
    pub builder: for<'t> fn(&ProcessorProfile, &WorkloadShape, &Signals<'t>) -> HistogramPlan,
}

/// Per-rule trace entry returned by [`super::plan_histogram_traced`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct RuleDecision {
    /// Index of the rule in `RULES`.
    pub index: u16,
    /// Stable rule name.
    pub name: &'static str,
    /// True iff the rule's predicate matched.
    pub matched: bool,
}

/// Helper used by every rule builder to assemble a [`HistogramPlan`].
///
/// Centralizing the construction here means every rule's plan body is
/// constructed the same way; the rule itself supplies only the strategy,
/// chunk/sample sizes, and confidence quantum. The reason and source
/// come from the `Rule` declaration itself, eliminating the substring
/// match in the legacy planner.
#[inline]
#[must_use]
pub(crate) fn build(
    rule: &Rule,
    strategy: crate::dispatch::HistogramStrategy,
    chunk_bytes: usize,
    sample_bytes: usize,
    confidence_q8: u8,
) -> HistogramPlan {
    HistogramPlan {
        strategy,
        chunk_bytes,
        sample_bytes,
        confidence_q8,
        confidence_source: rule.source,
        reason: rule.reason,
    }
}

//! Rule-table planner architecture.
//!
//! The legacy `plan_histogram` was a 400-line if/else chain with hardcoded
//! confidence quanta and substring-matched confidence sources. This module
//! replaces it with:
//!
//! - [`consts`] — every threshold, sample size, and confidence value gets
//!   a name and a documented bench-history provenance. Adding a new tuned
//!   value means adding a named constant; rule bodies don't carry magic
//!   numbers.
//!
//! - [`signals::Signals`] — derived workload predicates (random-like,
//!   structured-scale, oversubscribed, etc.) computed once per planning
//!   call so each rule predicate stays a pure function.
//!
//! - [`rule::Rule`] — the rule data type. A rule is `(name, reason, source,
//!   predicate, builder)`. The `source` is declared explicitly per rule;
//!   no substring inference of the reason text.
//!
//! - [`rules`]'s `RULES` registry — the rule registry, in priority order. First match
//!   wins. Adding a rule = appending to the list.
//!
//! - [`rules::plan_histogram_traced`] — returns the per-rule trace
//!   alongside the plan, for explainability.
//!
//! ## Behaviour preservation
//!
//! This redesign is a no-op against the 24 planner regression tests in
//! `crate::dispatch::tests`. Every existing `(profile, workload) →
//! HistogramPlan` mapping is preserved bit-exact: same strategy, same
//! chunk_bytes, sample_bytes, confidence_q8, confidence_source, reason.
//!
//! ## Adding a new rule
//!
//! 1. If the rule needs a new threshold or confidence quantum, add it to
//!    [`consts`] with a doc comment naming the bench-history artifact.
//! 2. If the rule needs a new derived predicate that several rules will
//!    share, add a field to [`signals::Signals`] and populate it in
//!    [`signals::Signals::derive`].
//! 3. Add a `pub(crate) const RULE_FOO: Rule` item in [`rules`] with the
//!    predicate and builder. Use the named constants for any numeric
//!    thresholds.
//! 4. Append `&RULE_FOO` to [`rules`]'s `RULES` registry at the right precedence
//!    position. Rules earlier in the list win over later ones.
//! 5. Add a regression test in `crate::dispatch::tests` that pins the
//!    new rule's expected output for at least one canonical workload.

pub mod consts;
pub mod rule;
pub mod rules;
pub mod signals;
pub mod tunes;

pub use rule::{Rule, RuleDecision};
pub use signals::Signals;
pub use tunes::Tunes;

#[cfg(feature = "std")]
pub use rules::plan_histogram_traced;

pub use rules::plan_histogram_tuned;

#[cfg(feature = "tunes-json")]
pub use tunes::TuneLoadError;

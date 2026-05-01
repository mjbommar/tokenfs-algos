# Planner Design

The histogram planner — `dispatch::plan_histogram` — chooses a kernel
strategy from `(ProcessorProfile, WorkloadShape)` and emits a
`HistogramPlan { strategy, chunk_bytes, sample_bytes, confidence_q8,
confidence_source, reason }`. This document describes the architecture
landed in #28 and the rules for evolving it.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                  dispatch::plan_histogram (5 LOC)                    │
│             delegates to planner::rules::plan_histogram              │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│  planner::rules::plan_histogram                                      │
│    1. Compute Signals once (random_like, structured_scale, ...)      │
│    2. Walk RULES in order; first matching rule wins                  │
│    3. Returns the rule's HistogramPlan via build()                   │
└──────────────────────────────────────────────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       ▼                           ▼                           ▼
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│  consts.rs   │           │  signals.rs  │           │   rule.rs    │
│              │           │              │           │              │
│ Named        │           │ Derived      │           │ Rule struct  │
│ thresholds,  │           │ predicates   │           │ + build()    │
│ confidence   │           │ computed     │           │ + RuleDecision│
│ bands, with  │           │ once per call│           │ for trace    │
│ provenance   │           │              │           │              │
└──────────────┘           └──────────────┘           └──────────────┘
                                   ▲
                                   │
┌──────────────────────────────────────────────────────────────────────┐
│  planner::rules                                                      │
│    32 pub(crate) const Rule items in priority order                  │
│    pub(crate) const RULES: &[&Rule] = &[ ... ]                       │
│    pub fn plan_histogram_traced -> (Plan, Vec<RuleDecision>)         │
└──────────────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `src/dispatch/mod.rs` | Public types (`Backend`, `HistogramPlan`, `PlannerConfidenceSource`, `HistogramStrategy`, `WorkloadShape`/`PlanContext`, `ProcessorProfile`, kernel catalog). `plan_histogram` is a 5-line delegation. |
| `src/dispatch/planner/mod.rs` | Module index. Re-exports `Rule`, `RuleDecision`, `Signals`, `plan_histogram_traced`. |
| `src/dispatch/planner/consts.rs` | Every threshold, sample size, and confidence band. Each constant has a doc-comment naming its bench-history provenance. |
| `src/dispatch/planner/signals.rs` | `Signals` struct + `Signals::derive(profile, workload)`. Pulls common predicates out of rule bodies. |
| `src/dispatch/planner/rule.rs` | The `Rule` data type, `RuleDecision` (trace entry), and the `build()` helper that constructs a `HistogramPlan` from a rule + outputs. |
| `src/dispatch/planner/rules.rs` | 32 `pub(crate) const RULE_*: Rule` items in priority order; `RULES: &[&Rule]`; `plan_histogram` (the rule-walker) and `plan_histogram_traced`. |

## Why this shape

The `dispatch` module previously had a 400-line if/else chain inside one
function. Three problems with that:

1. **Magic numbers without provenance.** `confidence_q8 = 235`, threshold
   `64 * 1024`, sample `4 * 1024` — all literal, no link back to the
   bench session that justified the value. Updating one was an act of
   faith.
2. **Substring-matched confidence sources.** `confidence_source_for(reason)`
   inferred `CalibrationRule` by `reason.contains("calibration")`. Renaming
   a reason string in good faith silently flipped the telemetry category.
3. **Implicit precedence.** Rule order = source-line order. Adding a rule
   meant reading the chain top-to-bottom to find the right insertion point.

The redesign solves these in three steps:

1. **`consts.rs`**: every numeric value has a name and a `SOURCE:` comment
   pointing at the bench history artifact that justifies it. When a sweep
   invalidates a value, update it in `consts.rs` and bump the source line.
   Rule bodies use the named constants; raw integer literals in rule
   builders are a code smell.
2. **Explicit `source` field on each `Rule`**: `PlannerConfidenceSource`
   is set per-rule via the struct, not derived from the reason text. The
   substring matcher is gone.
3. **`RULES: &[&Rule]`**: precedence is the slice's index. Adding a rule
   means appending an entry; reordering is a one-line change. Rules can
   be inspected, iterated, exported, and tested as data.

## Data flow

`plan_histogram(profile, workload)`:

1. `Signals::derive(profile, workload)` computes `call_bytes`, `threads`,
   `oversubscribed`, `random_like`, `sequential_like`, `mixedish_entropy`,
   `structured_scale` once.
2. Walk `RULES` in order.
3. For each `rule`, call `(rule.predicate)(profile, workload, signals)`.
4. On the first `true`, call `(rule.builder)(profile, workload, signals)`
   to construct the `HistogramPlan`. The builder uses `rule::build()`,
   which fills `confidence_source` from `rule.source` and `reason` from
   `rule.reason`.
5. Return.

The general fallback rule has `predicate: |_,_,_| true` so the loop
always terminates.

## Performance

Microbench (5 representative workloads, criterion `--quick`):

| Workload | Dispatch latency |
|----------|------------------|
| `paper-extent-random-4k` (rule #1) | 3.5 ns |
| `micro-random` (rule #5) | 4.3 ns |
| `parallel-avx2-macro-mixed` (rule #12) | 5.7 ns |
| `text-large` (rule #28) | 7.8 ns |
| `fallback-32k-block` (rule #33) | 9.0 ns |

Even fall-through to the general fallback (32 rules tested) takes 9 ns.
At ≤1% of any kernel's runtime, the planner is dispatch-cost-free.

## Adding a rule

1. Pick a kebab-case `name`. Names are stable telemetry identifiers; the
   `planner_rule_names_are_unique` test enforces uniqueness.
2. If the rule needs a new threshold or confidence quantum:
   - Add a `pub const` to `planner/consts.rs` with a doc comment naming
     the bench-history artifact (or "hand-tuned" if no bench yet).
   - Use the named constant in the rule builder. Avoid raw integer literals.
3. If the rule needs a new derived predicate that several rules will share:
   - Add a field to `Signals` and populate it in `Signals::derive`.
4. Write a `pub(crate) const RULE_FOO: Rule` item in `planner/rules.rs`:
   - `name`: kebab-case, telemetry identifier
   - `reason`: the user-facing explanation that ends up in
     `HistogramPlan::reason`
   - `source`: `CalibrationRule` if backed by bench evidence on this
     workload class; `StaticRule` for hand-tuned heuristics with at least
     one bench sweep; `Fallback` for catch-alls with confidence ≤ floor
   - `predicate`: pure function of `(profile, workload, signals)`
   - `builder`: emits `HistogramPlan` via `build(...)`
5. Append `&RULE_FOO` to `RULES` at the right precedence position.
   Rules earlier in the list win over later ones.
6. Add a regression test in `dispatch::tests` that pins the new rule's
   expected output for at least one canonical workload.
7. Run `cargo test -p tokenfs-algos --lib planner` — must show all 30+
   tests pass before merging.

## Adding a new constant

Same pattern as adding a rule's threshold:

```rust
/// Above this size, the foo-bar kernel beats the baz-quux kernel by ≥10%.
///
/// SOURCE: bench-history/runs/2026-05-15-foobar-sweep.jsonl row #34.
pub const FOO_BAR_THRESHOLD_BYTES: usize = 256 * KIB;
```

The `SOURCE:` line is mandatory. If you can't name a bench artifact,
write "SOURCE: hand-tuned, no bench evidence yet" — at least the
provenance is honest.

## Tests

Three tiers, all in `crate::dispatch::tests`:

1. **Regression tests (24)**: pin specific `(workload → strategy)` mappings
   that have shipped behaviour. Examples:
   `planner_uses_run_length_for_small_low_entropy_reads`,
   `planner_uses_avx2_palette_for_large_low_entropy_inputs`. Touch a rule
   in a way that breaks one of these → CI fails immediately.

2. **Architecture tests (6, added in #28)**:
   - `planner_rule_names_are_unique` — no duplicate `name` fields
   - `planner_rule_table_has_a_terminal_match` — last rule is the fallback
   - `planner_general_fallback_predicate_always_matches` — the fallback
     predicate returns `true` for arbitrary input
   - `planner_fallback_source_implies_low_confidence` — every rule tagged
     `Fallback` emits `confidence_q8 ≤ CONFIDENCE_FALLBACK_FLOOR`
   - `planner_traced_returns_winner_in_trace` — trace mode exposes the
     winner correctly
   - `planner_traced_records_misses_before_winner` — trace shows misses

3. **Sweep test (`planner_does_not_select_the_avx2_stripe4_placeholder`)**:
   675 (profile × API × entropy × scale × total) combinations confirm no
   active rule emits the documented placeholder strategy.

Total: 31 planner tests gating every change.

## Trace mode

`planner::plan_histogram_traced(profile, workload) -> (HistogramPlan,
Vec<RuleDecision>)` returns the same plan as `plan_histogram` plus a
trace of every rule examined up to and including the winner. Useful for:

- Explaining "why did the planner pick X" in bench-comparison reports
- Detecting silent rule-precedence regressions when the bench-history
  comparison flags a kernel switch
- Generating planner-explain output in `examples/dispatch_explain.rs`

The `RuleDecision { index, name, matched }` shape is small and
allocation-free in the hot path is preserved by the non-trace
`plan_histogram` (no `Vec`).

## Tunes — host-specific overrides

The `dispatch::planner::tunes::Tunes` struct mirrors every value in
`consts.rs`. `Tunes::DEFAULT` is the compile-time default; rules access
values via `signals.tunes.field_name` rather than touching the consts
directly. This lets a host-specific calibration override individual
thresholds and confidence bands without recompiling.

Two planner entry points:

```rust
// Compile-time defaults (zero overhead).
plan_histogram(&profile, &workload)
plan_histogram_traced(&profile, &workload)

// Host-tuned (caller owns the Tunes value).
plan_histogram_tuned(&profile, &workload, &tunes)
plan_histogram_traced_tuned(&profile, &workload, &tunes)
```

Constructing tunes:

```rust
// Programmatic — chainable builder.
let tunes = Tunes::DEFAULT
    .with_block_threshold_large_bytes(128 * 1024)
    .with_confidence_fallback_floor(120);

// JSON (requires `tunes-json` feature → pulls in serde_json).
let tunes = Tunes::from_json(&std::fs::read_to_string("tunes.json")?)?;
```

The JSON schema is a flat object:

```json
{
  "block_threshold_large_bytes": 131072,
  "total_threshold_macro_bytes": 524288,
  "confidence_high_calibrated": 245
}
```

Field names match `Tunes` struct fields exactly. Unknown fields produce
`TuneLoadError::UnknownField`; out-of-range u8 values produce
`TuneLoadError::OutOfRange`. Missing fields keep their `Tunes::DEFAULT`
value, so a cache file only needs to list the fields that diverge from
the upstream defaults.

The intended workflow per `docs/AUTOTUNING_AND_BENCH_HISTORY.md`:

1. `bench-calibrate` runs the workload matrix and writes JSON tunes
   per `(crate version, rustc, backend, cpu_model, cache_profile)`
   tuple into `target/bench-history/tunes/`.
2. The application loads the tune file matching the running host:
   `Tunes::from_json(&std::fs::read_to_string(path)?)?`.
3. The planner uses `plan_histogram_tuned(&profile, &workload, &tunes)`.

Tests under `dispatch::tests::tunes_*` cover: default-equals-consts,
default-tunes path matches the untuned path bit-exact, threshold
overrides flip the chosen strategy, JSON round-trip, and JSON error
handling for unknown fields and out-of-range values.

## Roadmap

The redesign + tunes layer unblocks three remaining items:

- **#26** — Similarity-distance planner integration. New rules for
  byte-distribution nearest-reference dispatch get added to `rules.rs`
  alongside the histogram rules; the rule machinery generalizes to
  per-family rule registries.
- **Trained classifier exploration**. With trace data captured per call,
  it becomes feasible to train a small decision tree on
  `(workload signals, observed best kernel)` rows from `bench-history/`
  and compare its picks against the hand-tuned rules. Don't ship; just
  measure how far off the hand-tuning is.
- **Multi-primitive planner**. The `Rule` type is generic in shape over
  the primitive family; today it's specialized to `HistogramPlan` /
  `HistogramStrategy`. Generalizing to `Plan<Strategy>` is a
  straightforward refactor when a second family (fingerprint?
  similarity?) needs the same machinery.

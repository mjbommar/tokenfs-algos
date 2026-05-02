# Benchmark history snapshots

This directory stores per-release snapshots of `target/criterion/` JSON results
so we can detect cross-release performance regressions without re-running
benchmarks against historical commits.

## Layout

```
benches/_history/
  <label>/
    <bench-binary>/<group>/<param>/base/estimates.json
    <bench-binary>/<group>/<param>/new/estimates.json
    <bench-binary>/<group>/<param>/new/sample.json
```

Each snapshot preserves the directory structure produced by Criterion. Only the
canonical measurement files (`estimates.json` from `base/` and `new/`, plus
`sample.json` from `new/`) are stored; rendered HTML, PNG plots, raw CSV, and
the `change/` and `report/` subdirectories are skipped.

`<label>` is typically a release tag (`v0.2.0`) or a short SHA. The
`bench-history` xtask defaults to `git rev-parse --short HEAD` when no
`--label` is supplied, so post-tag invocations produce SHA-tagged snapshots
automatically.

## Producing a snapshot

```bash
cargo bench --workspace                          # populate target/criterion/
cargo xtask bench-history --label v0.2.0          # copy the canonical files
```

Rerun `bench-history` with a different label after each release. Snapshots are
checked into git so future contributors can diff against any tagged baseline.

## Comparing snapshots manually

The Criterion `estimates.json` schema records mean, median, std-dev, and
percentile estimates as nanoseconds. To eyeball a single benchmark:

```bash
jq '.mean.point_estimate' \
  benches/_history/v0.2.0/<bench>/<group>/<param>/new/estimates.json
jq '.mean.point_estimate' \
  benches/_history/v0.2.1/<bench>/<group>/<param>/new/estimates.json
```

A perf regression appears as a higher `mean.point_estimate` (nanoseconds per
iteration) at the newer label. Use `cargo xtask bench-compare` for richer
comparisons over the JSONL history (`target/bench-history/runs/*.jsonl`).

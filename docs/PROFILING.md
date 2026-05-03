# Profiling Workflow

This document records the host-environment requirements and concrete commands
for profiling and cross-testing `tokenfs-algos`. It exists because the audit
flagged that several profiling and cross-arch workflows were either undocumented
or blocked by unrelated host configuration.

## Summary

| Want                                      | Command                                                                                  |
| ----------------------------------------- | ---------------------------------------------------------------------------------------- |
| Synthetic primitive bench                 | `cargo xtask bench-workloads`                                                            |
| Adaptive-only primitive bench             | `cargo xtask bench-workloads-adaptive`                                                   |
| Real-data bench (Ubuntu ISO etc.)         | `cargo xtask bench-real ~/ubuntu-26.04-desktop-amd64.iso`                                |
| Compare two run JSONLs                    | `cargo xtask bench-compare <old.jsonl> <new.jsonl>`                                      |
| `perf stat` over the primitive driver    | `cargo xtask profile`                                                                    |
| Flamegraph over the primitive driver      | `cargo xtask profile-flamegraph -- --profile-time 10 workload_matrix/adaptive-prefix-1k` |
| Flamegraph over `examples/profile_primitives` | `cargo xtask profile-primitives-flamegraph`                                          |
| Hardware-counter bench (deterministic)    | `cargo xtask bench-iai`  (requires `valgrind`)                                           |
| Cross-test on AArch64 under QEMU          | `cargo test -p tokenfs-algos --target aarch64-unknown-linux-gnu`                         |

Output artifacts live under `target/bench-history/` (`runs/`, `reports/`,
`comparisons/`, `index.tsv`) and `target/profiles/` for perf stat / flamegraph.

## Bench history layout

```
target/bench-history/
├── index.tsv                         # All runs (timestamp, sha, label, n_rows)
├── latest.md                         # Latest comparison summary
├── runs/                             # JSONL rows per benchmark run
│   └── <unix-ts>-<sha>[-dirty].jsonl
├── reports/                          # Per-run rendered artifacts
│   └── <unix-ts>-<sha>[-dirty]/
│       ├── summary.md
│       ├── timing.csv
│       ├── throughput-heatmap.html
│       ├── *-throughput.svg
│       ├── *-planner.svg
│       ├── *-winners.svg
│       └── thread-scaling.svg
└── comparisons/                      # bench-compare outputs
    └── <ts>-<old>-vs-<new>/
        ├── deltas.csv
        ├── deltas.md
        └── *.svg
```

Use `bench-compare` for durable interpretation. Criterion's inline `change` line
compares only against `target/criterion`'s local baseline, not the project
baseline — it lies if you run benches across branches without resetting.

## Hardware-counter benches with iai-callgrind

`cargo xtask bench-iai` runs a focused subset of the hot primitives
(popcount, streamvbyte encode/decode, sha256, vector dot/l2) under
`valgrind --tool=callgrind` via the `iai-callgrind` harness. The
output is **deterministic across runs on the same binary** —
instruction count, branch misses, L1/L2/LL cache references — which
lets CI catch regressions at much tighter thresholds than the
wall-clock criterion benches can support.

| Bench harness          | Threshold | Noise source                                                       |
| ---------------------- | --------- | ------------------------------------------------------------------ |
| criterion (wall-clock) | 15%       | runner CPU contention, frequency scaling, cache state across runs  |
| iai-callgrind (counters)| 1%       | none on the same binary; recompilation can shift counts ±0.5%      |

This means a 3% iai-callgrind regression is **signal**, not noise. By
contrast, a 5% criterion regression on a GitHub-hosted runner usually
isn't reproducible.

Local invocation:

```bash
sudo apt install valgrind            # Debian/Ubuntu; equivalent on other distros
cargo install iai-callgrind-runner --version 0.16
cargo xtask bench-iai
```

The runner version must match the `iai-callgrind` dev-dep in
`crates/tokenfs-algos/Cargo.toml`, otherwise the bench errors at start.

CI uses `.github/workflows/iai-bench.yml`, which:

1. installs valgrind on the runner,
2. downloads the most recent `iai-baseline-main` artifact (the
   `target/iai/` directory snapshotted from the last green main run),
3. runs the iai benches with `--baseline=main --regression=ir=1.0` so
   the job fails on any benchmark whose instruction count regresses by
   more than 1.0% vs that baseline,
4. on `main`, re-runs with `--save-baseline=main` and re-uploads
   `target/iai/` as the next baseline artifact (90-day retention).

If you don't have `valgrind` installed, skip this bench — there is no
fallback path. The criterion suite (`cargo xtask bench-workloads`) is
the everyday wall-clock companion; the iai suite is the regression
canary.

## Bench-history publication (gh-pages)

Every successful `bench-regression.yml` run on `main` is archived into
the `gh-pages` branch by `.github/workflows/bench-history.yml`, which
republishes a static site at
[mjbommar.github.io/tokenfs-algos/](https://mjbommar.github.io/tokenfs-algos/)
(audit-R10 T3.5).

The gh-pages branch is the canonical archive: each main push appends a
new `data/runs/<unix-ts>-<short-sha>.jsonl` file, and `cargo xtask
bench-history-site` regenerates `index.html` + `groups/<group>.html`
with sparkline + per-kernel timeline SVGs computed from the full
historical set.

`.github/workflows/bench-history-prune.yml` runs weekly (Sunday 05:17
UTC) and drops JSONL runs older than 365 days, so the gh-pages branch
stays bounded.

To bootstrap on a fresh fork: run `tools/scripts/init-gh-pages.sh`
once to create the orphan `gh-pages` branch, then in repo settings
flip Pages to serve from that branch. The workflows take over from
the next main push.

To preview the site locally before pushing: run any benches, then

```bash
cargo xtask bench-history-site target/bench-history/runs /tmp/bench-site
python3 -m http.server -d /tmp/bench-site 8000
# browse http://localhost:8000/
```

## perf hardware counters require `perf_event_paranoid` ≤ 1

`cargo xtask profile` and `profile-flamegraph` invoke `perf record`. On most
fresh Ubuntu installs `kernel.perf_event_paranoid` defaults to **4**, which
blocks userspace `perf record` even for the calling process's own threads.
Symptoms: empty flamegraphs, perf reporting "no samples", or
`PERF_EVENT_OPEN failed: Permission denied`.

Check:

```bash
cat /proc/sys/kernel/perf_event_paranoid
```

Temporarily relax for a single profiling session (does not persist across
reboot):

```bash
sudo sysctl -w kernel.perf_event_paranoid=1
# ...run profile commands...
sudo sysctl -w kernel.perf_event_paranoid=4   # restore
```

Persistently relax (only on dev hosts you own):

```bash
echo 'kernel.perf_event_paranoid = 1' | sudo tee /etc/sysctl.d/99-perf.conf
sudo sysctl --system
```

Setting `0` allows kernel-level samples (call stacks into the kernel); `1`
allows user samples and is sufficient for our flamegraph pipeline. `-1` is
"trust everyone" and should not be used on shared hosts.

## Flamegraph for individual primitives

`examples/profile_primitives` is a long-running driver that takes a kernel
label and an iteration count, then spins on that one kernel. It is the right
target when you want a tight, focused flamegraph for one primitive instead of
the full bench matrix.

Direct invocation (without xtask):

```bash
cargo flamegraph \
  --output target/profiles/primitive-driver-flamegraph.svg \
  -p tokenfs-algos \
  --example profile_primitives \
  --features bench-internals \
  -- all
```

Replace `all` with one of: `histogram-avx2-stripe4`, `fingerprint-avx2`,
`entropy-h8-exact`. See `examples/profile_primitives.rs` for the current label
list.

## Targeting one primitive in the criterion matrix

The xtask bench commands accept an optional Criterion filter as the trailing
argument:

```bash
cargo xtask bench-workloads-adaptive '^primitive_matrix/runlength-transitions-(scalar|avx2)/'
cargo xtask profile-flamegraph -- --profile-time 10 workload_matrix/adaptive-prefix-1k
```

The `^...$`-style anchored regex is recommended — Criterion treats the filter
as a substring by default, which catches more than you usually want.

## Cross-testing on AArch64 under QEMU

The repo's `tests/neon_parity.rs` is `cfg(target_arch = "aarch64")`-gated and
exercises the NEON kernels in `byteclass::kernels::neon` and
`runlength::kernels::neon`. To run them on an x86 dev host:

```bash
# One-time setup (Debian/Ubuntu):
sudo apt install gcc-aarch64-linux-gnu libc6-dev-arm64-cross qemu-user
rustup target add aarch64-unknown-linux-gnu

# Then:
cargo test -p tokenfs-algos --target aarch64-unknown-linux-gnu
```

The repo's `.cargo/config.toml` already declares the linker
(`aarch64-linux-gnu-gcc`) and the runner (`qemu-aarch64 -L
/usr/aarch64-linux-gnu`) for that target, so `cargo test` will transparently
build aarch64 binaries, link against the cross sysroot, and execute them under
QEMU user-mode emulation. Expect a 2-5× slowdown vs. native for proptest
suites; the deterministic-corpus suites complete in seconds.

QEMU user-mode emulates NEON correctly. Some advanced AArch64 features
(SVE/SVE2) are emulated only with explicit CPU model flags
(`qemu-aarch64 -cpu max`); the runner above uses the default which exposes
NEON but not SVE.

For real SVE/SVE2 work or any wall-clock-meaningful AArch64 numbers, run on
real hardware (Ampere Altra, Apple Silicon under Asahi, AWS Graviton, etc.) —
QEMU is only a correctness gate, not a performance reference.

## Cross-testing on AArch64 with no cross sysroot

If you cannot install the cross-toolchain (e.g. on a locked-down host), you
can still verify the lib compiles for aarch64:

```bash
cargo build -p tokenfs-algos --target aarch64-unknown-linux-gnu
```

This builds the library `.rlib` and surfaces any `target_arch = "aarch64"`
type or borrow errors. Test executables won't link without the cross sysroot,
so this only catches code-level issues, not runtime correctness.

## Profiling caveats

- **AVX-512 downclock**: AVX-512 kernels can downclock the core (and on older
  Intel, the package). Don't draw conclusions from a 100 ms benchmark — let
  the planner's measurement-time settings run long enough for clock recovery.
- **Thread topology**: parallel-sequential benchmarks at high thread counts
  (`xtask bench-thread-topology`) are sensitive to CPU pinning, hyper-threading
  policy, and thermal throttling. Note the host's tuned profile (`tuned-adm
  active`) and frequency governor (`cpupower frequency-info`) when comparing
  runs.
- **Cache hot/cold modes**: the bench harness exposes hot, cold, and
  same-file-repeat access patterns (see `benches/support/mod.rs`). A "win" on
  hot but a "loss" on cold usually means the kernel is L1-bound — check the
  working-set class in `dispatch::HISTOGRAM_KERNEL_CATALOG` against the
  observed cache profile.
- **Bench history rows include planner choice**: every JSONL row carries
  `planned_kernel`, `planned_chunk_bytes`, `planned_sample_bytes`,
  `planned_confidence_q8`, `planned_confidence_source`, and `plan_reason`.
  When the planner picks a kernel that loses, the comparison report flags it
  as a "miss" — this is the primary input to planner tuning.

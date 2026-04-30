# tokenfs-algos

Low-level Rust algorithms for hardware-accelerated byte-stream analysis.

This repository is being built from `PLAN.md` and `AGENTS.md`. The core crate
is intended to stay content-agnostic, deterministic across backends, and usable
from TokenFS tooling, future kernel/FUSE integrations, and separate Python
bindings.

## Development

Use the checked-in nightly toolchain:

```bash
cargo xtask check
cargo xtask test
cargo xtask bench
cargo xtask bench-kernels
cargo xtask bench-adaptive
cargo xtask bench-workloads
cargo xtask bench-workloads-adaptive
cargo xtask bench-calibrate
cargo xtask bench-compare target/bench-history/runs/<old>.jsonl target/bench-history/runs/<new>.jsonl
cargo xtask profile
cargo run -p tokenfs-algos --example dispatch_explain
```

Real-data benchmarks are opt-in and keep large corpora out of git:

```bash
cargo xtask bench-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-kernels-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-adaptive-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-adaptive-contexts-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask bench-workloads-adaptive-real ~/ubuntu-26.04-desktop-amd64.iso
cargo xtask profile-real ~/ubuntu-26.04-desktop-amd64.iso
```

If no path is provided, the real-data benchmark/profile tasks use
`TOKENFS_ALGOS_REAL_DATA`, then fall back to
`~/ubuntu-26.04-desktop-amd64.iso`.

The first implementation target is the scalar byte histogram, followed by
entropy, run-length statistics, byte classification, and the F22 fingerprint
kernel.

The workload benchmark matrix and logged result format are documented in
`docs/BENCHMARK_WORKLOAD_MATRIX.md`. Processor-aware dispatch and kernel
promotion strategy are documented in `docs/PROCESSOR_AWARE_DISPATCH.md`,
`docs/PRIMITIVE_KERNEL_BUFFET.md`, and
`docs/AUTOTUNING_AND_BENCH_HISTORY.md`. Paper-linked primitive migration and
consumer latency budgets are tracked in `docs/PAPER_PRIMITIVE_MIGRATION.md` and
`docs/CONSUMER_LATENCY_BUDGETS.md`.

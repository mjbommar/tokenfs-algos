#!/usr/bin/env bash
# Cargo runner for `aarch64-unknown-linux-gnu` cross-tests.
#
# Wraps `qemu-aarch64` and exports environment knobs that opt out of host-CPU
# wall-clock-sensitive gates so an emulated NEON test run does not fail on
# legitimate emulator overhead.
#
# Configured via `.cargo/config.toml`:
#   [target.aarch64-unknown-linux-gnu]
#   runner = "tools/cross-runner/qemu-aarch64.sh"

set -euo pipefail

# QEMU needs the cross sysroot to resolve the dynamic linker.
export QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/aarch64-linux-gnu}"

# Throughput gates assert wall-clock budgets that QEMU emulation cannot meet
# (~10x slowdown vs native). Parity, correctness, and bound checks still run.
export TOKENFS_ALGOS_SKIP_THROUGHPUT_GATE=1

exec qemu-aarch64 "$@"

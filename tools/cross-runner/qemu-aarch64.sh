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

# When the host architecture matches the test binary (i.e. we're on a
# native aarch64 runner, e.g. ubuntu-24.04-arm), skip QEMU entirely and
# exec the binary directly. The .cargo/config.toml runner clause fires
# whenever `target == aarch64-unknown-linux-gnu`, regardless of host —
# this guard keeps native runs from looking for a non-existent qemu.
if [ "$(uname -m)" = "aarch64" ]; then
    exec "$@"
fi

# QEMU needs the cross sysroot to resolve the dynamic linker.
export QEMU_LD_PREFIX="${QEMU_LD_PREFIX:-/usr/aarch64-linux-gnu}"

# Throughput gates assert wall-clock budgets that QEMU emulation cannot meet
# (~10x slowdown vs native). Parity, correctness, and bound checks still run.
export TOKENFS_ALGOS_SKIP_THROUGHPUT_GATE=1

exec qemu-aarch64 "$@"

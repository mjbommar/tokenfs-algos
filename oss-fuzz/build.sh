#!/usr/bin/env bash
# OSS-Fuzz build script for tokenfs-algos.
#
# Submitted upstream to https://github.com/google/oss-fuzz/projects/tokenfs-algos
# along with project.yaml + Dockerfile — this copy is in-tree for
# auditability / reproducibility.
#
# OSS-Fuzz invokes this script in the base-builder container and expects
# every compiled fuzz target to land in $OUT/<target_name>.

set -eu

cd "$SRC/tokenfs-algos/fuzz"

# Build all 13 declared fuzz targets at once via cargo-fuzz. The build
# pulls in the same userspace + arch-pinned-kernels + parallel + blake3
# feature set we use in CI (audit-R10 T0.2).
PATH=$PATH:/usr/local/cargo/bin
cargo +nightly fuzz build -O --debug-assertions

TARGETS=$(cargo +nightly fuzz list)
for t in $TARGETS; do
  cp "$SRC/tokenfs-algos/target/x86_64-unknown-linux-gnu/release/$t" "$OUT/$t"
done

# Seed corpus is committed under fuzz/corpus/<target>/. OSS-Fuzz expects
# `<target>_seed_corpus.zip` for each declared target.
for t in $TARGETS; do
  if [ -d "$SRC/tokenfs-algos/fuzz/corpus/$t" ]; then
    (cd "$SRC/tokenfs-algos/fuzz/corpus/$t" && zip -r "$OUT/${t}_seed_corpus.zip" . > /dev/null)
  fi
done

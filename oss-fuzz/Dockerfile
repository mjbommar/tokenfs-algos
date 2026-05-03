# Dockerfile for OSS-Fuzz integration of tokenfs-algos.
# Submitted upstream to https://github.com/google/oss-fuzz/projects/tokenfs-algos
# along with project.yaml + build.sh — this copy lives in-tree so the
# integration is auditable / reproducible.

FROM gcr.io/oss-fuzz-base/base-builder-rust

# Pin a recent Rust nightly so cargo-fuzz works.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

# OSS-Fuzz sets WORKDIR=/src by convention.
RUN git clone --depth=1 https://github.com/mjbommar/tokenfs-algos.git tokenfs-algos
COPY build.sh $SRC/

WORKDIR $SRC/tokenfs-algos

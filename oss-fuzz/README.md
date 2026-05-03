# OSS-Fuzz integration for `tokenfs-algos`

This directory tracks the project files submitted upstream to
[google/oss-fuzz](https://github.com/google/oss-fuzz) so OSS-Fuzz can
fuzz the 13 declared `cargo-fuzz` targets continuously.

## Submission status

PR to `google/oss-fuzz`: TODO (file the PR after the repo goes public).

The following files mirror the upstream
`projects/tokenfs-algos/{Dockerfile,build.sh,project.yaml}` so every
audit reviewer sees the exact integration we requested:

* `project.yaml` — language, sanitizers (ASan + MSan), primary contact.
* `Dockerfile` — pinned to `gcr.io/oss-fuzz-base/base-builder-rust`,
  clones the public repo at submission time.
* `build.sh` — `cargo +nightly fuzz build` then copy each binary to
  `$OUT/<target_name>` and zip the seed corpus.

## Per-target seed corpus

Seed corpora live under `fuzz/corpus/<target>/` (gitignored except for
the directory placeholder). OSS-Fuzz expects each target to ship with
`<target>_seed_corpus.zip`; the build script auto-generates these from
the local corpus directories.

## Why OSS-Fuzz on top of nightly fuzz CI?

* Local nightly fuzz CI (`.github/workflows/fuzz-nightly.yml`) gives us
  ~10 minutes per target per night — sufficient to catch shallow bugs
  but not the multi-day deep-state-space bugs OSS-Fuzz is built for.
* OSS-Fuzz also provides ClusterFuzz coverage tracking, automatic
  bisection of regressions, and a dedicated Monorail issue tracker —
  none of which we want to re-implement.

## Re-running locally

The `build.sh` script can be invoked outside the OSS-Fuzz base-builder
container by exporting `$SRC` and `$OUT`:

```bash
export SRC=/path/to/checkout
export OUT=/tmp/oss-fuzz-out
mkdir -p "$OUT"
bash oss-fuzz/build.sh
```

This produces a `$OUT/<target>` binary for every declared fuzz target,
plus a `<target>_seed_corpus.zip` if the local corpus directory exists.

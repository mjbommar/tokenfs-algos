# Reference Projects

The subdirectories here are ignored by git and used only for local design
review. Do not vendor code from them into `tokenfs-algos` without an explicit
license and provenance review.

| Directory | Why it is here | What it taught us |
|---|---|---|
| `memchr` | Rust byte-search crate by BurntSushi. | Keep scalar behavior as the oracle; combine runtime feature detection with compile-time fallbacks. |
| `aho-corasick` | Production automata and packed SIMD search. | Gate target-feature paths explicitly and fuzz tiny byte-oriented APIs. |
| `simdutf` | UTF validation/transcoding with many SIMD backends. | Expose active implementation introspection and runtime support checks. |
| `simdjson` | SIMD JSON parser with dispatch and benchmark culture. | Separate stable API from backend implementation details; benchmark real parser workloads. |
| `StringZilla` | Cross-language string/memory kernels. | Treat cache-line, L1/L2/L3 residency, and CPU counters as benchmark axes. |
| `StringZilla-CLI` | CLI packaging and user-facing benchmark surface. | Capability reporting is product documentation, not just developer tooling. |
| `NumKong` | Multi-backend numerical kernels and dispatch. | Distinguish compile-time dispatch from runtime dispatch and report compiled-vs-runtime capabilities. |
| `wide` | Portable Rust SIMD types. | Useful abstraction candidate once the first concrete SIMD histogram kernel shape is known. |
| `safe_arch` | Safer Rust wrappers around architecture intrinsics. | Keep target-feature availability visible at the API boundary. |
| `criterion.rs` | Benchmark harness and stored statistical outputs. | Mine `target/criterion` JSON, but keep our own workload/hardware history. |
| `maturin` | PyO3 packaging and build frontend. | Keep Python binding as a later workspace member wrapping stable Rust APIs. |

Local paper code that still needs migration lives outside this directory:

```text
../tokenfs-paper/tools/rust/entropy_primitives/
```

That crate contains the F22 fingerprint kernel, CRC32C hash-bin counter,
`c * log2(c)` entropy lookup table, top-K coverage, and calibration tests
against F21 sidecar data.

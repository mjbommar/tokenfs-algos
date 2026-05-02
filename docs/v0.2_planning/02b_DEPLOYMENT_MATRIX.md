# Deployment matrix

**Status:** consumer-environment reference, 2026-05-02. Companion to `02_CACHE_RESIDENCY.md`. Maps each near-term consumer of `tokenfs-algos` to its operating constraints, so module specs (`10_*` through `14_*`) can call out which APIs fit which environments.

## Why this matters

Per `crates/tokenfs-algos/AGENTS.md` and `_references/README.md`, this crate is **content-agnostic byte-slice compute**. The named near-term consumers go beyond TokenFS:

> *"... kernel filesystem, FUSE, future Python data-pipeline tooling, compression-codec dispatchers, forensics tools, columnar databases."*

Each of those operates under different constraints. `00_BOTTOM_UP_ANALYSIS.md` and `02_CACHE_RESIDENCY.md` were initially written through a TokenFS-userspace lens. This doc fills in the rest of the consumer surface and lets each module's API surface be evaluated against multiple environments at design time, not after a kernel patch comes back rejected.

## The matrix

| Consumer | Feature config | Stack budget | Threading | Allocation | Latency budget | SIMD posture | Determinism |
|---|---|---|---|---|---|---|---|
| **Linux kernel module** (Rust-for-Linux) | `default-features=false, features=["alloc"]` (no `panicking-shape-apis`) | **8-16 KB** | single thread / softirq context; **no rayon** | `gfp_t`-flagged Vec wrapper; **caller-provided slices preferred** | per-op µs | `kernel_fpu_begin/end` brackets per SIMD section (~hundred-cycle cost) | required for verified module |
| **FUSE userspace daemon** (libfuse) | `default-features=false, features=["alloc","std","avx2","neon"]` (no `panicking-shape-apis`) | ~8 MB user stack | thread-per-request from libfuse worker pool | `Vec` / `Box` | per-request 10-50 µs warm; cold-mmap 200 µs+ | full | not required |
| **Userspace tool / build pipeline** (`tokenfs_writer`-class) | `default-features=true, features=["parallel","blake3"]` | ~8 MB | rayon multi-thread | `Vec` / `Box` | throughput-only | full | preferred for reproducibility |
| **PostgreSQL extension** (Rust extension via `pgrx`-style) | `default-features=true, features=["parallel","blake3"]` | postgres backend ~2 MB | per-query, multi-thread within bg workers | postgres memory contexts; **caller-provided buffers preferred** | per-query latency varies | full but FPU brackets unnecessary outside aggregates | preferred for query plans |
| **MinIO / Go service via cgo** | `default-features=true, features=["parallel","blake3"]` | Go goroutine ~8 KB initial (grows); cgo holds OS thread | Go scheduler; cgo bridges to OS thread | Go-managed buffers passed through cgo | cgo per-call ~200 ns overhead | full | not required |
| **CDN edge cache** | `default-features=true, features=["parallel","blake3"]` | ample | high-fanout multi-thread per-request | per-conn allocators; jemalloc-class | µs per-request | full | not required |
| **Forensics / batch analytics** | `default-features=true, features=["parallel"]` | ample | rayon | `Vec` | throughput | full | required for chain-of-custody |
| **Python / research via PyO3** | `default-features=true, features=["python","parallel"]` | Python ~8 MB | GIL serialized; release GIL during Rust calls | Python heap; **numpy zerocopy preferred** | µs per Python call | full | required for paper artifacts |

## What each constraint means for primitive design

### Stack budget

Kernel: **8-16 KB total stack** including the kernel call chain. Anything that puts large `[T; N]` on the stack is a hazard. The §77 finding (CRC binning's 64 KB on-stack scratch) is the canonical example. The `_with_scratch` caller-provided pattern is **kernel-mandatory**; the convenience in-place variant is **userspace-only**.

Postgres: ~2 MB backend stack — much larger than kernel but smaller than typical userspace. Anything > 100 KB on stack starts to crowd the postgres call chain.

Go via cgo: Go's growable goroutine stack is initially small. cgo calls hold an OS thread, which uses the OS thread's much larger stack. Stack-usage during the cgo call itself is on the OS thread, so the ~8 MB userspace budget applies. Returning to Go drops back to the goroutine stack.

### Threading

Kernel: single-thread within a syscall path; rayon dependency is fatal. Any "parallel batched" API needs a single-thread variant.

FUSE: libfuse spawns a worker thread per request (configurable). Primitives must be **re-entrant** and not rely on shared mutable state without explicit `Arc`/lock.

Postgres: per-query bg workers can use multi-thread, but most SQL functions execute in a single backend. SIMD primitives don't need to do their own parallelization; postgres parallel-query mechanism handles fan-out.

Go via cgo: cgo calls hold the OS thread; calling rayon from cgo is OK (rayon uses its own thread pool internally) but the goroutine that initiated cgo blocks until rayon returns.

### Allocation

Kernel: needs explicit GFP flag (`GFP_KERNEL`, `GFP_ATOMIC`, etc.) for every allocation; the Rust-for-Linux `Vec` wrapper takes the flag at construction. Our APIs that take pre-allocated slices (`_with_scratch` pattern) are **kernel-ideal**; APIs that allocate internally need a `&Allocator` parameter or a feature-gated kernel-shim variant.

Postgres: `palloc` allocates from the current memory context. C extensions get a `MemoryContext*`; Rust extensions via `pgrx` get a Rust wrapper. Like kernel, **caller-provided buffer pattern** lets postgres put output in the right context.

Go via cgo: simplest is to have Go allocate the output buffer and pass it to the cgo function. Avoids any cross-boundary allocation overhead.

### Latency budget

FUSE per-request: 10-50 µs warm-cache is the bar to feel like a normal FS. Cold-mmap-page is unavoidable ~200 µs. Per-primitive single-call latency must be sub-µs for the primitive to compose into a budget.

Postgres per-query: function-call latency varies from µs (predicate) to ms (aggregate). A SIMD primitive used in a per-tuple predicate needs ns-µs latency.

cgo per-call: ~200 ns overhead per call. Any primitive called per-element from Go is dead; **batch APIs are mandatory**.

CDN edge per-request: µs budget for cache lookups; ms for content fetches.

### Determinism

Kernel-module verification (`fs-verity`-class workflows) requires bit-exact reproducibility across kernel versions and CPU generations. Postgres query plans benefit from deterministic ordering (e.g., GROUP BY result order). Research / paper artifacts demand bit-exact reproducibility.

Two practical implications:
- **SIMD reduction order** must be documented and held stable across versions. The Higham §3 dot-product tolerance model (just landed in `7eb0621`) makes "reduction order is part of the public contract" explicit.
- **Tie-breaking in permutations / sorts** must be deterministic. RCM frontier ties on equal degree → break by vertex ID (lowest first).

### Panicking entry points (audit-R5 #157)

A panic in a Linux kernel softirq is fatal to the kernel; in libfuse it kills the FUSE handler. Several primitive entry points (`BitPacker::encode_u32_slice`, `streamvbyte_decode_u32`, `dot_f32_one_to_many`, `RankSelectDict::build`, `sha256_batch_st`, `signature_batch_simd`, etc.) historically asserted on caller-supplied shape mismatches and panicked on failure. Each now has a fallible `try_*` parallel that returns a typed error.

The `panicking-shape-apis` Cargo feature gates the panicking variants behind `#[cfg(feature = "panicking-shape-apis")]`. The feature is **on by default** (back-compat for existing userspace consumers). Kernel and FUSE consumers should disable it:

```toml
tokenfs-algos = { version = "0.2", default-features = false, features = ["alloc"] }
# add "std", "avx2", "neon" as appropriate; the FUSE row above is the
# canonical "everything except the panicking shape wrappers" config.
```

Under that build, only the `try_*` wrappers are reachable on the public API; calls to the panicking constructors fail to compile. The kernel build is verified by `cargo check -p tokenfs-algos --no-default-features --features alloc --lib` (run via `cargo xtask security`).

## Implications for current planning

Re-examining the v0.2 module specs through this matrix:

### `bits` (10_BITS.md)

- **Kernel-safe**: ✅ all of it. Static rodata for Stream-VByte shuffle table (4 KiB) and popcount nibble LUT (16 B). `RankSelectDict` borrows the bit slice; caller controls allocation of superblock/block index arrays (or use the `_borrowed` constructor over a pre-built index).
- **Postgres**: ✅ `BitPacker<W>` is a clean per-tuple primitive. Stream-VByte for varint-coded posting columns.
- **cgo**: ✅ all batched APIs. Single-element `popcount_u64_slice` is a candidate for batch wrappers.

### `bitmap` (11_BITMAP.md)

- **Kernel-safe**: ✅ bitmap container (8 KB stack), array container (caller-allocated Vec). Cardinality popcount is the highest-leverage AVX-512 use; kernel modules with VPOPCNTQ would benefit measurably.
- **Postgres**: ✅ Roaring set ops are exactly the GIN-bitmap-scan inner loop. A pgvector-style extension could consume.
- **cgo**: batch APIs only (per-pair set op called from Go is fine; per-element is not).

### `hash` batched (12_HASH_BATCHED.md) — **revise needed**

- **Kernel-safe**: ⚠️ as currently spec'd, `blake3_batch` uses rayon; kernel can't. **Action**: spec must include a `blake3_batch_st` (single-thread, internal-multi-stream-SIMD via `blake3::guts::ChunkState`) variant that's the kernel-default. The rayon-parallel variant is the userspace optimization. Same for `sha256_batch`.
- **No `blake3` in no_std**: per current `Cargo.toml`, `blake3 = ["dep:blake3", "std"]` — the blake3 crate needs std. Kernel modules can't link blake3. The kernel content-addressing answer is **SHA-256 only** via `sha2` crate (which has ARMv8 SHA-2 acceleration via `cpufeatures` crate). Update the module spec to be explicit.
- **Postgres / cgo**: ✅ all batch APIs.

### `vector` (13_VECTOR.md)

- **Kernel-safe**: ✅ stateless kernels, no allocation. Single-pair dot/L2/cosine in a kernel SIMD section: ~few µs even for 1024-element vectors. Batched many-vs-one is fine if the output buffer is caller-provided.
- **Postgres**: pgvector is the closest analog — exact same kernel surface. Could be a real upstream contribution path.
- **cgo**: batched APIs are the sweet spot; per-pair calls have cgo overhead.

### `permutation` (14_PERMUTATION.md)

- **Permutation construction (RCM, Hilbert, Rabbit Order)**: ❌ build-time only. These algorithms allocate large work buffers (BFS queue, dendrogram) that can't be made stack-only. Never runs in kernel.
- **`Permutation::apply` (using a precomputed permutation)**: ✅ kernel-safe. Stateless, borrowed slices, SIMD-friendly gather. This IS the runtime-hot-path API.
- **Postgres / cgo / userspace**: ✅ for both construction and apply.
- **Open question — online clustering for CDN/MinIO promotion patterns**: out of scope for v0.2. Different algorithm class (online k-means, streaming community detection). Would land as `cluster::online` in v0.3+ if a consumer asks.

### Deferred items (20_DEFERRED.md) — **revise needed**

The trigger conditions in the deferred list are mostly written as "TokenFS asks." Multi-consumer reframing: trigger conditions become "any consumer asks."

Concrete examples:

- **MinHash SIMD**: CDN edge dedup (Cloudflare, Fastly) actively uses MinHash. Real consumer outside TokenFS.
- **Bloom SIMD**: Postgres bloom filter index, MinIO content-fingerprint pre-checks. Real consumers.
- **HyperLogLog merge**: Postgres `approx_count_distinct` extensions, OLAP databases. Real consumers.
- **CSR walk + BFS**: Graph databases (Neo4j-class), recommendation systems. Real consumers.
- **Top-K SIMD heap**: ANN top-K, Postgres `ORDER BY ... LIMIT`. Real consumers.
- **Levenshtein SIMD**: Postgres `pg_trgm`-class fuzzy match, search systems. Real consumers.
- **xxh3/wyhash SIMD**: high-throughput hash table workloads, content-addressed systems. Real consumers.

The deferral discipline still holds (don't build without a documented bottleneck), but the **scope of "documented bottleneck" widens to all consumer environments**, not just TokenFS.

## What this matrix does NOT cover

- **Hardware capability matrix** — that's `docs/AVX512_HARDWARE.md` + `docs/PROCESSOR_AWARE_DISPATCH.md`.
- **Build target matrix** — that's the existing CI matrix (linux-x86_64 / linux-aarch64 / macos / windows / windows-aarch64 / cross-aarch64-qemu / AVX-512 self-hosted).
- **API stability** — that's `docs/PRIMITIVE_CONTRACTS.md`.

## Open questions

1. **Should we have explicit `kernel`-safety tests in CI?** Current `xtask security` verifies the no_std + alloc lib build (which now also implicitly verifies that no panicking shape wrappers are reachable when `panicking-shape-apis` is off — see audit-R5 #157 above); doesn't verify "no rayon," "no blake3" (those are forbidden but absence of usage isn't directly tested in lib code). **Tentative: add a `--features alloc` test target that exercises every kernel-claimed-safe primitive and asserts they don't reach for std/rayon/blake3.**

2. **Is there a real near-term Postgres extension consumer?** If yes, the matrix shapes API choices (caller-provided buffers throughout); if no, design for kernel + FUSE first and let Postgres fit when a consumer materializes.

3. **PyO3 bindings — scope?** The `python` feature flag doesn't yet exist in the workspace. **Tentative: out of scope for v0.2; add when the Python-facing batch APIs stabilize.**

4. **Which primitives need caller-provided-scratch as the default API**, not the parallel variant? **Tentative**: anything that allocates > 4 KB on the stack (CRC bins were 64 KB, fixed in §77; check Stream-VByte shuffle work-buffers, `RankSelectDict::build` working memory).

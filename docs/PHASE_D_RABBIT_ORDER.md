# Phase D: Rabbit Order graph reordering

Status: shipped in v0.3.0 (Sprint 47-49 sequential baseline,
Sprint 50-52 SIMD modularity-gain kernels, Sprint 53-55 round-based
concurrent variant).

This note explains the user-facing problem Rabbit Order solves, when to
reach for it instead of the other orderings shipped by
`tokenfs_algos::permutation`, and the limitations callers should be
aware of. The full algorithmic spec lives in
[`v0.2_planning/14_PERMUTATION.md`](v0.2_planning/14_PERMUTATION.md)
§ 3; the deployment matrix lives in
[`v0.2_planning/02b_DEPLOYMENT_MATRIX.md`](v0.2_planning/02b_DEPLOYMENT_MATRIX.md).

## Problem

Sparse-graph workloads — BFS, PageRank, connected-components, neighbour
aggregation in a GNN, TokenFS extent-following on a deduped image — are
overwhelmingly **memory-latency bound**. The asymptotic time complexity
suggests `O(|V| + |E|)`, but the realised wall-clock cost is dominated
by L3 misses and TLB misses on each neighbour gather. The vertex
*ordering* — i.e., the permutation from logical vertex IDs to physical
storage positions — controls how often the per-neighbour gather lands
on a cache line that is already hot or nearby.

Rabbit Order produces a permutation that places vertices belonging to
the same **community** (cluster of densely interconnected vertices)
contiguously in the output. On graphs with genuine community structure
— social networks, web crawls, dedup clusters in a content-addressed
filesystem, software call-graphs — this is the best published cache
locality of any one-shot reordering algorithm. The IPDPS 2016 paper
reports 1.25-2.4× speedups over natural-order graphs across PageRank,
BFS, and CC across the SNAP benchmark suite.

## When to use Rabbit Order vs the alternatives

`tokenfs_algos::permutation` ships four orderings. Pick by input shape
and quality / cost trade-off.

| Ordering | API | Input shape | Build cost | Quality | Use when |
|---|---|---|---|---|---|
| Identity | `Permutation::identity(n)` | none | trivial | none | a permutation slot is required but no reordering is desired (placeholder, baseline, fixture) |
| RCM | `permutation::rcm(graph)` | sparse undirected graph (`CsrGraph`) | very cheap (`O(\|V\| + \|E\| log Δ)`, ~10 ms / 228 K vertices) | bandwidth-minimising; modest cache-locality win on traversal | sparse-matrix solvers, bandwidth-driven workloads, or any "good enough" graph reorder where build time matters |
| Hilbert | `permutation::hilbert_2d(points)` / `hilbert_nd(points, dim)` | point cloud in 2D or N-D (`&[(f32,f32)]` / `&[Vec<f32>]`) | `O(n log n)` sort | preserves metric locality on the embedding | data has a true low-dimensional point embedding (PCA-projected fingerprints, t-SNE/UMAP outputs) |
| Rabbit Order | `permutation::rabbit_order(graph)` / `rabbit_order_par(graph)` | sparse undirected graph (`CsrGraph`) | heavy (`O(\|E\| log \|V\|)`, 1-5 s / 228 K vertices) | best published cache-locality; community-aware | community-structured graphs feeding locality-sensitive workloads (BFS, PageRank, neighbour scans, TokenFS dedup-cluster reads) |

Rules of thumb:

- If the input is **not** a graph — it's a point cloud or a label cloud
  with a Euclidean embedding — use Hilbert. Rabbit Order requires CSR
  adjacency.
- If the workload is **bandwidth-driven** (sparse linear algebra,
  Kaczmarz/SOR sweep solvers, banded matrix kernels) use RCM. Rabbit
  Order optimises modularity, not bandwidth.
- If build time is **the** constraint — interactive tooling, recurring
  per-request reorder, anything where shaving 1-5 seconds matters more
  than shaving a few percent off downstream cache-miss rates — use RCM.
  Build cost differs by 100-500×.
- Otherwise, when the graph has community structure and the workload
  is locality-sensitive, use Rabbit Order. TokenFS images are sealed
  once and read many times, so paying seconds at build time for
  percent-level cache improvements at every read is the right
  trade-off.

## Performance characteristics

### Sequential vs parallel

The crate exposes two entry points:

- `permutation::rabbit_order(graph) -> Permutation`. Single-threaded,
  always available, the canonical reference implementation.
- `permutation::rabbit_order_par(graph) -> Permutation`. Round-based
  concurrent variant, gated on the `parallel` Cargo feature, available
  when the input has at least
  `permutation::RABBIT_PARALLEL_EDGE_THRESHOLD` directed edges (default
  ~200 000). Below that threshold the function transparently delegates
  to the sequential path — rayon's per-task overhead exceeds the
  agglomeration cost on small graphs.

Both produce **valid** Rabbit Order permutations and both group
communities contiguously, but the merge sequences differ on graphs that
admit multiple positive-modularity merges (the sequential path uses a
heap that re-evaluates each vertex against the *current* state; the
round-based parallel path locks per-round proposals against the
*snapshot* taken at round start). The two functions are bit-exact with
each other only on degenerate inputs that admit no merges (empty,
n=1, fully disconnected, edgeless).

The parallel variant typically runs at **wall-clock parity or modestly
slower** than the sequential baseline on TokenFS-typical sparse inputs
at 100 K - 1 M vertices. The bottleneck is the sequential apply phase,
which preserves determinism by applying merges in canonical order. The
function exists primarily to provide a deterministic API surface for
callers who want to participate in a rayon-driven pipeline, and to
anchor future work that swaps the apply-phase strategy
(colouring-based conflict-free batching, hand-rolled lock-free
adjacency, etc.) without churning the public API.

### SIMD modularity-gain kernel

The per-pair modularity-gain inner loop is a hot path under the
agglomeration loop. Sprint 50-52 lifted it into
`permutation::rabbit::kernels` with scalar / AVX2 / AVX-512 / NEON
backends, runtime-dispatched via `kernels::auto::modularity_gains_neighbor_batch`.
All backends are bit-exact with the scalar reference (integer
arithmetic; no floating-point reduction order to track). The SIMD
backends use a 32-bit-input widening multiply (`_mm256_mul_epu32`,
`_mm512_mul_epu32`, `vmull_u32`) and operate in `i64` lanes when the
input magnitudes fit; for adversarial inputs whose products would
overflow `i64`, the kernels fall back to the `i128` scalar path. The
fast-path eligibility predicate is exposed as
`kernels::scalar::fast_path_eligible` so external benchmarks can
pre-classify their inputs.

## Limitations

- **Undirected only.** The algorithm treats every edge as undirected.
  Callers with a directed adjacency must symmetrise upstream (typically
  by emitting both `(u, v)` and `(v, u)` in the CSR). Self-loops are
  tolerated and contribute weight to the vertex's own degree per
  Newman's modularity convention; duplicate edges fold into a combined
  weight per neighbour.
- **Modularity-greedy heuristic.** Modularity maximisation is
  NP-hard; Rabbit Order is a *greedy* heuristic, not an exact solver.
  Two graphs that are isomorphic up to a vertex relabelling may
  produce different community structures because the lowest-degree-
  first iteration order — and the modularity-gain tie-break — depends
  on vertex IDs. The crate fixes the tie-break to ascending neighbour
  ID, so the implementation is deterministic *given a fixed CSR*, but
  the algorithm's output is not invariant under re-labelling.
- **Build-time only.** Rabbit Order allocates `O(|V| + |E|)` working
  state and runs for seconds on realistic graphs. It cannot be made
  stack-only and is **never** safe to run inside a kernel or FUSE
  request handler. Per
  [`v0.2_planning/02b_DEPLOYMENT_MATRIX.md`](v0.2_planning/02b_DEPLOYMENT_MATRIX.md),
  `permutation::rabbit_order` is permitted only in the userspace
  build-pipeline tier (`tokenfs_writer`, batch analytics, research).
  The kernel and FUSE consumers load a precomputed `[u32; n]`
  permutation from the sealed image manifest and apply it via
  `Permutation::apply` / `Permutation::apply_into`, both of which are
  kernel-safe.
- **Single-pass agglomeration.** The shipped implementation runs one
  bottom-up pass and emits the dendrogram-DFS visit order. The
  reference C++ from Arai et al. recurses on the merged super-graph
  (multi-level Louvain-style) for further quality gains; that
  refinement is a follow-on sprint and is not in v0.3.0.
- **Heuristic build-time vs locality trade-off is not bench-pinned in
  v0.3.0.** The shipped agglomeration matches the IPDPS 2016
  pseudocode, but the locality vs build-cost ratio on TokenFS-class
  workloads is not yet measured against a sealed dedup image. The
  expected operating point is documented as "1-5 s build for percent-
  level read-time wins"; the actual ratio for any given image will
  depend on the cluster structure of its content.

## Worked example

Two triangles connected by a bridge edge — Rabbit Order should place
each triangle contiguously in the output:

```rust
use tokenfs_algos::permutation::{CsrGraph, rabbit_order};

// Vertices 0-1-2 form one triangle, 3-4-5 another, with a single
// bridge edge 2-3 holding the two communities together.
// CSR layout (undirected, both directions present):
//   0 -> [1, 2]
//   1 -> [0, 2]
//   2 -> [0, 1, 3]
//   3 -> [2, 4, 5]
//   4 -> [3, 5]
//   5 -> [3, 4]
let offsets = [0_u32, 2, 4, 7, 10, 12, 14];
let neighbors = [1_u32, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4];
let graph = CsrGraph { n: 6, offsets: &offsets, neighbors: &neighbors };

let perm = rabbit_order(graph);
assert_eq!(perm.len(), 6);

// The result is a valid permutation: every id 0..n appears exactly once.
let mut seen = [false; 6];
for &new_id in perm.as_slice() {
    seen[new_id as usize] = true;
}
assert!(seen.iter().all(|b| *b));

// Apply the permutation to a payload (e.g., per-vertex feature vector).
let features: [u32; 6] = [100, 101, 102, 200, 201, 202];
let reordered = perm.apply(&features);
assert_eq!(reordered.len(), 6);
```

For larger inputs above
`permutation::RABBIT_PARALLEL_EDGE_THRESHOLD`, the same example with
the `parallel` Cargo feature enabled would call
`permutation::rabbit_order_par(graph)` for the same deterministic
result with the proposal phase parallelised across the rayon thread
pool.

## Reference

J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, S. Iwamura, "Rabbit
Order: Just-in-time Parallel Reordering for Fast Graph Analysis",
*Proc. IEEE International Parallel and Distributed Processing
Symposium (IPDPS) 2016*, pp. 22-31. [DOI: 10.1109/IPDPS.2016.110](https://doi.org/10.1109/IPDPS.2016.110)

Reference C++ implementation: `araij/rabbit_order` on GitHub
(C++/Boost; uses concurrent hash maps and per-thread merge buffers
for the multi-level recursion). The Rust port shipped here is the
first published Rust implementation and tracks the IPDPS 2016
pseudocode for the single-pass sequential baseline.

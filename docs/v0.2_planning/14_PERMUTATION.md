# `permutation` module — locality-improving orderings

**Status:** spec, 2026-05-02. Phase B4/B5 + D1 of `01_PHASES.md`.

This is the missing-from-prior-docs primitive class. TokenFS images are read-only and built once, so the layout permutation can be computed once and stored as part of the manifest. The downstream gain is uniform across every other primitive: a 1.5-3x cache-miss reduction on graph traversals and metadata scans.

## Goal & scope

Three permutation primitives, by quality/cost tradeoff:

1. **§ 2 Reverse Cuthill-McKee (RCM)** — classic, cheap, deterministic. The baseline. Phase B4.
2. **§ 4 Hilbert curve order** — for 2D-embedded data (PCA-projected fingerprints, t-SNE embeddings). Trivial implementation. Phase B5.
3. **§ 3 Rabbit Order** — community-detection-driven. Best published quality. Hard to implement (no Rust port). Phase D1.

GOrder, METIS, and minLA are explicitly **not in scope**. See § 5.

## § 1 Module surface

```
permutation/
├── mod.rs
├── rcm.rs                    // Reverse Cuthill-McKee
├── hilbert.rs                // 2D/N-D Hilbert curve sort
├── rabbit.rs                 // Rabbit Order (Phase D)
└── tests.rs
```

Public types:

```rust
pub mod permutation {
    /// A permutation array: `perm[old_id] = new_id`.
    /// Length equals the number of vertices/items.
    pub struct Permutation(pub Vec<u32>);

    impl Permutation {
        pub fn identity(n: usize) -> Self;
        pub fn inverse(&self) -> Self;
        pub fn apply<T: Copy>(&self, src: &[T]) -> Vec<T>;
        pub fn apply_into<T: Copy>(&self, src: &[T], dst: &mut [T]);
        pub fn as_slice(&self) -> &[u32];
    }

    /// CSR adjacency (input format for graph-based permutations).
    pub struct CsrGraph<'a> {
        pub n: u32,
        pub offsets: &'a [u32],   // length n + 1
        pub neighbors: &'a [u32], // length offsets[n]
    }

    /// Reverse Cuthill-McKee ordering.
    pub fn rcm(graph: CsrGraph<'_>) -> Permutation;

    /// Hilbert curve ordering of points in 2D or N-D.
    pub fn hilbert_2d(points: &[(f32, f32)]) -> Permutation;
    pub fn hilbert_nd(points: &[Vec<f32>], dim: usize) -> Permutation;

    /// Rabbit Order (Phase D — community-detection ordering).
    pub fn rabbit_order(graph: CsrGraph<'_>) -> Permutation;
}
```

## § 2 Reverse Cuthill-McKee

### Algorithm

Cuthill-McKee, 1969: BFS the graph from a low-degree start vertex, sorting each level's frontier by ascending degree. Reverse the resulting visit order. The reversal step (Liu and Sherman 1976) is what makes "RCM" — pure CM has worse profile.

Steps:
1. Pick a **pseudoperipheral** start vertex via the GPS algorithm: BFS once from any vertex, find the deepest level, BFS from one of those, repeat until depth doesn't increase.
2. BFS from the start. At each level, sort the frontier by ascending vertex degree.
3. Record the visit order. Reverse it. That's the permutation.

### Complexity

- Time: O(|V| + |E| log Δ) where Δ is max degree (the log is from sorting frontiers).
- Space: O(|V|) for the queue + visit-order array.

For 228K vertices with average degree 5 (TokenFS-typical), this is ~10 ms.

### Quality

- Bandwidth reduction: typical 10-100x on sparse-matrix graphs.
- Locality on graph algorithms: ~20-30% worse than Rabbit Order or GOrder on real graphs (Flickr/Twitter benchmarks, per Faldu et al. IISWC 2019).
- Deterministic: same graph → same permutation, modulo tie-breaking on equal-degree neighbors.

### Existing Rust impls

- **`sprs::linalg::ordering::reverse_cuthill_mckee`** — production-quality, pseudoperipheral start vertex, returns `Permutation` over a `CsMatViewI`. Use as oracle.

### Why ship our own when `sprs` exists

Three reasons:
1. We don't want to drag in `sprs`'s sparse-matrix data model just for one ordering function.
2. We want the CSR input format to match our `CsrGraph` (which is the universal input shape across `permutation`, future `graph`, etc.).
3. The argsort step in BFS frontier-by-degree benefits from `bits::popcount`-aware radix sort. Vendoring `sprs` doesn't expose that knob.

### SIMD opportunities

- **Degree lookup**: `vpgatherdd`-based gather for sorting the frontier. Modest win (~20-30%).
- **Frontier sort**: AVX-512 bitonic sort for small frontiers (≤ 64 elements); radix sort for larger. Modest win.

The kernel is fundamentally pointer-chasing-bound. SIMD is the cherry on top, not the win. **Phase B4 ships scalar first**; SIMD opportunistically.

### Test plan

- Property: `rcm(graph)` is a permutation (each id appears once).
- Property: bandwidth (max distance between connected vertices in the new order) is ≤ original bandwidth.
- Parity vs `sprs::reverse_cuthill_mckee` on small test graphs.
- Edge cases: empty graph, isolated vertices (no edges), disconnected components, self-loops.

### Bench plan

- 10K, 100K, 1M vertex sparse graphs (synthetic Erdős–Rényi + scale-free).
- Wall-clock build cost.
- Downstream: time a sequential BFS over the original-order graph vs the RCM-permuted graph; report the cache-miss reduction.

## § 3 Rabbit Order (Phase D)

### Algorithm (Arai et al., IPDPS 2016)

Bottom-up agglomerative community detection followed by dendrogram-DFS visit order. High level:

1. **Initialize**: every vertex is its own community.
2. **Repeat**: for each vertex `u` in low-degree-first order:
   - Find the neighbor `v` whose merge maximizes modularity gain `dQ`:
     ```
     dQ(u, v) = w(u,v)/m - deg(u)*deg(v)/(2*m^2)
     ```
   - Merge `u` into `v`'s community. Record the merge in a dendrogram.
3. **Recurse**: treat each community as a super-vertex; iterate.
4. **Final**: DFS the dendrogram. Leaves visited in DFS order are the new vertex IDs.

The reference C++ uses parallel agglomerative merging via concurrent hash maps and a per-thread merge buffer.

### Complexity

- Time: roughly Louvain-class (~|E| log |V|).
- Space: O(|V|) for the dendrogram + O(|E|) for the working adjacency.

For 228K vertices, expect ~1-5 seconds. Heavy compared to RCM's ~10 ms.

### Quality

- Best published locality on graph algorithms across published benchmarks (PageRank, BFS, Connected Components).
- 1.25-2.4× speedup over natural-order on standard benchmarks.
- Good fit for TokenFS specifically because dedup clusters *are* communities (vertices sharing extents form natural cliques).

### Why ship our own

**No Rust port exists.** The reference is `araij/rabbit_order` (C++/Boost). Building a Rust port is a real contribution to the ecosystem.

### Implementation notes

- The concurrent hash map for adjacency merging is the trickiest part. Options: `dashmap`, hand-rolled atomic-bucket structure, or rayon-parallel batch merging (sequence-then-merge).
- Reference Boost uses `boost::container::flat_map`. Equivalent: pre-sorted Vec with binary insert.
- Modularity-gain inner loop is dot-product-shaped — viable AVX2/AVX-512 win.

### Risk

This is the only Phase-D item that's a multi-week effort. Defer until image-layout quality becomes a measured bottleneck. **RCM (B4) covers the 80% case at 1% of the implementation cost.**

### Test plan

- Property: result is a valid permutation.
- Property: modularity computed on the resulting community structure ≥ 0.3 on benchmark graphs (loose check).
- Parity against `araij/rabbit_order` C++ output on the same input. (Ship a generator script.)

## § 4 Hilbert curve order

### Algorithm

Skilling 2004 algorithm for N-dimensional Hilbert curves: bit-interleave the coordinates (after quantizing to integer grid) and apply the Skilling rotation lemma. For 2D specifically, much simpler: pre-quantize to u32, interleave bits via PEXT/PDEP or Morton code, and the resulting integer is a Z-order (Morton) curve key. The Hilbert key is similar but with corner-flip rotations applied per quad.

For TokenFS use case (sort fingerprint-projected-to-2D points by locality):
1. PCA-project F22 fingerprints to 2D (one-time at build time, in `tokenfs-paper`'s writer; output to `permutation::hilbert_2d`).
2. Quantize to u32 grid (e.g., scale to 0..2^16-1).
3. Compute Hilbert key per point.
4. Argsort by Hilbert key.

### Complexity

O(n log n) sort.

### When to use this vs RCM

- Use **Hilbert** when data has true 2D/N-D embedding structure. Best for spatial-locality-driven workloads (similarity-scan).
- Use **RCM** when data is graph-shaped (CSR adjacency).
- They're complementary, not competitive.

### Existing Rust impls

- `hilbert` (Skilling N-D, has `Permutation` type). https://docs.rs/hilbert
- `fast_hilbert` (LUT-based 2D). https://crates.io/crates/fast_hilbert
- `hilbert_2d`, `hilbert_index`, `moore-hilbert` (FFI to Doug Moore).

**Tentative: depend on `fast_hilbert` for 2D and `hilbert` for N-D**, behind feature flags. Don't reimplement Skilling.

### SIMD

The bit-interleave step vectorizes cleanly (AVX2 PEXT/PDEP equivalent or VPSHUFB-based). Argsort can use AVX-512 radix or AVX2 chunked sort. **But the kernel is small and the input data is small** (one pair of f32s per point); SIMD wins are <2x. Don't over-engineer.

### Test plan

- Property: result is a valid permutation.
- Property: locality preserved — points within a small bounding box have keys within a contiguous range.
- Parity against `fast_hilbert` for the 2D case.

## § 5 Why GOrder / METIS / minLA are not shipped

| Algorithm | Reason for exclusion |
|---|---|
| **GOrder** (Wei et al. SIGMOD 2016) | Best query-locality of all 4 candidates, but reordering cost is ~592x slower than RCM/HashOrder. Only worth it for graphs processed thousands of times. TokenFS images don't see that traffic; RCM + Rabbit cover the regime. |
| **METIS multi-level partition** | Heavy C dependency (`metis-sys` FFI). Rabbit Order with community structure is the better fit since dedup clusters *are* communities. Add later if explicit k-way partitioning for sharding becomes a use case. |
| **minLA (Minimum Linear Arrangement)** | NP-hard. Spectral approximations exist but no production Rust impl. The 1-2% additional quality vs Rabbit isn't worth it. |
| **Degree-sort baseline** | Cheap (O(n log n) sort by degree) but ~2x worse cache-miss rate than GOrder. Useful as a benchmark baseline only; not a primitive worth shipping. |

## § 6 Test scaffolding

We need benchmark graphs to evaluate ordering quality. Suggestions:

- **Scale-free**: SNAP small examples (`as-skitter`, `wiki-Talk`) — public, well-known.
- **Synthetic Erdős–Rényi**: easy to generate, gives baseline.
- **TokenFS sample image**: the actual workload. Generate from a real Ubuntu rootfs or similar.

The bench compares `original-order BFS` vs `RCM-order BFS` vs `Rabbit-order BFS` cache-miss rates and wall-clock time. Use perf-counters to capture LLC misses.

## § 7 Open questions

1. **Should `Permutation` be its own type, or just `Vec<u32>` everywhere?** The wrapper type lets us add invariant checks (every id appears exactly once) and inverse / apply helpers. **Tentative: yes, a wrapper.**

2. **Inverse permutation caching**: applying a permutation needs the array; inverting it needs an O(n) pass. Cache the inverse on the type? **Tentative: lazy, computed on demand.**

3. **Where does the cache-miss benchmark live?** Outside the unit tests (uses perf events). **Tentative: in `benches/permutation_quality.rs` with a flag to skip if perf events unavailable.**

4. **Phase ordering**: Hilbert is independent of RCM/Rabbit and trivial — could ship before RCM. **Tentative: yes — Hilbert in B5 alongside RCM in B4 because they're independent.**

5. **Multi-level RCM** (split graph into k components, RCM each)? **Tentative: skip; Rabbit covers the multi-level quality regime.**

## § 8 Environment fitness

Per [`02b_DEPLOYMENT_MATRIX.md`](02b_DEPLOYMENT_MATRIX.md):

| API | Kernel module | FUSE | Userspace (build pipeline) | Postgres ext | cgo (Go) | Python (PyO3) |
|---|---|---|---|---|---|---|
| `rcm(graph)` | ❌ build-time only | ❌ | ✅ | ⚠️ uncommon | ⚠️ uncommon | ✅ research |
| `hilbert_2d(points)` | ❌ build-time only | ❌ | ✅ | ⚠️ for spatial indexes | ⚠️ batch only | ✅ |
| `rabbit_order(graph)` | ❌ build-time only | ❌ | ✅ | ❌ too heavy | ❌ | ✅ research |
| `Permutation::apply` | ✅ stateless | ✅ | ✅ | ✅ | ✅ batch | ✅ |
| `Permutation::inverse` | ✅ allocates output Vec | ✅ | ✅ | ⚠️ palloc-aware | ✅ | ✅ |

**Critical note:** the `permutation` module is **inherently a build-time / batch-analytics primitive**, not a hot-path one. RCM, Hilbert, and Rabbit Order all operate on graph adjacency or point sets to produce a permutation that is then *applied once* and stored as part of the image / dataset layout.

- **TokenFS use:** invoked by `tokenfs_writer` at image build time to compute the inode/extent ordering that makes `tokenfs_reader` faster. Output is part of the sealed image manifest.
- **Postgres use:** uncommon but possible for offline index-build-time clustering (BRIN summary range optimization, pgvector index construction).
- **Research use:** common for graph experiments — RCM/Rabbit/Hilbert as preprocessing for downstream graph algorithms.
- **Kernel module use:** never. The permutation is consumed (via `apply`) at read time, but the permutation itself was computed offline.
- **cgo use:** the permutation arrays are typically large (millions of u32s); keep at batch granularity — Go calls `rcm(adjacency)` once per graph, not per-vertex.

`Permutation::apply` is the only API in this module that's hot-path-capable; it's stateless and SIMD-friendly (sequential gather for sorted permutations, scatter for arbitrary).

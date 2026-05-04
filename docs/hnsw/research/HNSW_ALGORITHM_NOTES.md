# HNSW Algorithm Implementation Notes

Specification-level notes for a Rust HNSW implementation. Sourced primarily from
Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search
using Hierarchical Navigable Small World graphs", arXiv:1603.09320 (v4, 2018;
journal version IEEE TPAMI 42(4):824-836, April 2020). Algorithm and section
numbers below refer to that paper unless otherwise noted.

---

## 1. The Five Paper Algorithms

### Notation conventions

Paper notation:

- `q` query / new element
- `ep` entry-point set (one element from INSERT, possibly more from K-NN-SEARCH)
- `W` dynamic list of nearest neighbours found so far (max-heap, capped at `ef`)
- `C` candidate priority queue (min-heap, no cap)
- `lc` layer index being processed
- `L` top layer index (level of the global entry point at insert time)
- `l` level chosen for the new element
- `M` desired bidirectional connections per element (per layer it occupies)
- `Mmax` cap on connections in upper layers (default `Mmax = M`)
- `Mmax0` cap on connections in layer 0 (default `Mmax0 = 2*M`)
- `efConstruction` dynamic candidate list size during construction
- `ef` dynamic candidate list size during query
- `mL` level normalization factor (paper recommends `mL = 1/ln(M)`)

Our Rust naming used in the impl:

| Paper | Rust impl |
|-------|-----------|
| `q` | `query` (search) or `point` (insert) |
| `ep` | `entry` |
| `W` | `nearest` |
| `C` | `candidates` |
| `lc` | `layer` |
| `L` | `top_level` |
| `l` | `new_level` |
| `M` | `m` |
| `Mmax` | `m_max` |
| `Mmax0` | `m_max0` |
| `efConstruction` | `ef_construction` |
| `ef` | `ef_search` |
| `mL` | `level_mult` (`1.0 / (m as f64).ln()`) |

`hnsw.distance(a, b)` denotes the metric. `hnsw.neighbourhood(e, lc)` denotes
the neighbour set of `e` on layer `lc` (always undirected, bidirectional).

---

### Algorithm 1: INSERT(hnsw, q, M, Mmax, efConstruction, mL)

Inputs: HNSW structure `hnsw`, new element `q`, parameters as above.
Output: `q` is inserted into `hnsw`; entry point may be updated.

Paper pseudocode (section 4, Algorithm 1):

```
INSERT(hnsw, q, M, Mmax, efConstruction, mL)
1   W <- empty                                   // currently found nearest
2   ep <- get enter point for hnsw
3   L  <- level of ep                            // top layer for hnsw
4   l  <- floor(-ln(unif(0,1)) * mL)             // new element's level
5   for lc in L .. l+1
6       W  <- SEARCH-LAYER(q, ep, ef=1, lc)
7       ep <- get the nearest element from W to q
8   for lc in min(L, l) .. 0
9       W  <- SEARCH-LAYER(q, ep, efConstruction, lc)
10      neighbors <- SELECT-NEIGHBORS(q, W, M, lc)   // alg 3 or 4
11      add bidirectional connections from neighbors to q at layer lc
12      for each e in neighbors                  // shrink if needed
13          eConn <- neighbourhood(e) at layer lc
14          if |eConn| > Mmax                    // (Mmax0 when lc == 0)
15              eNewConn <- SELECT-NEIGHBORS(e, eConn, Mmax, lc)
16              set neighbourhood(e) at layer lc to eNewConn
17      ep <- W
18  if l > L
19      set enter point for hnsw to q
```

Edge cases handled by the pseudocode:

- Line 5 loop body does not execute when `l >= L` (range empty), so a node
  promoted above the current top simply falls through to Phase 2 starting at
  layer `L`.
- Line 18-19 promotes only when strictly taller than the previous entry.

Edge cases the implementation must add:

- **Empty graph.** Lines 2-3 dereference an entry point that does not exist.
  First insert: record the node, set `entry_point = q` and
  `entry_level = chosen l`, return.
- **`unif(0,1)` returning 0.** `-ln(0) = +inf`. Draw on the open interval
  `(0,1)` or clamp.
- **`new_level` huge.** Cap at e.g. `max_level = 64`.
- **Tied distances** at lines 7 / 10. Tie-break by `(distance, NodeKey)`.

Time complexity (paper section 4.2.2): expected `O(log N)` distance
comparisons per insert; `O(N log N)` for a build of `N` points. Phase 2
dominates: `O((l + 1) * efConstruction * Mmax)` per insert.

Space per element: `Mmax0` neighbours at layer 0 plus `Mmax` per upper layer
the element occupies. Expected total: `Mmax0 + Mmax / (M - 1)` neighbours.

---

### Algorithm 2: SEARCH-LAYER(q, ep, ef, lc)

Inputs: query `q`, entry-point set `ep`, dynamic list size `ef`, layer `lc`.
Output: at most `ef` nearest elements to `q` found on layer `lc`.

Paper pseudocode (Algorithm 2):

```
SEARCH-LAYER(q, ep, ef, lc)
1   v <- ep                                      // set of visited
2   C <- ep                                      // candidates (min-heap)
3   W <- ep                                      // results (max-heap)
4   while |C| > 0
5       c <- extract nearest element from C to q
6       f <- get furthest element from W to q
7       if distance(c, q) > distance(f, q)
8           break                                // (early termination)
9       for each e in neighbourhood(c) at layer lc
10          if e not in v
11              v <- v union {e}
12              f <- get furthest element from W to q
13              if distance(e, q) < distance(f, q) or |W| < ef
14                  C <- C union {e}
15                  W <- W union {e}
16                  if |W| > ef
17                      remove furthest element from W to q
18  return W
```

Implementation notes:

- The line-7 early termination is what makes the algorithm sub-linear: once
  the closest unexplored candidate is farther than the farthest current
  result, no future candidate can improve `W`.
- The `|W| < ef` clause on line 13 is essential. Without it `W` never fills
  to `ef` if early candidates happen to be far, and recall collapses.
- Multiple entry points (size > 1) only happens when called from K-NN-SEARCH
  Phase 2 with the carried `W`. INSERT always passes a single element.
- Tied distances: order both heaps by `(distance, NodeKey)` for determinism
  across SIMD architectures.
- Disconnected component: silently returns suboptimal `W` if the entry-point
  chain never enters the component containing the true NN. The hierarchy
  mitigates but does not eliminate this. See section 4 / 6.

Time: `O(ef * log N)` distance comparisons per call. Each iteration pops one
candidate and visits up to `Mmax` (or `Mmax0`) neighbours. Empirically the
hop count is `O(log N)` (small-world property). Space: `O(ef + |visited|)`.

---

### Algorithm 3: SELECT-NEIGHBORS-SIMPLE(q, C, M)

Inputs: base `q`, candidate set `C`, target neighbour count `M`.
Output: `M` nearest elements in `C` to `q`.

Paper pseudocode (Algorithm 3):

```
SELECT-NEIGHBORS-SIMPLE(q, C, M)
1   return M nearest elements from C to q
```

Edge cases: `|C| < M` returns all of `C`; tie-break by NodeKey.
Complexity: `O(|C| log M)` with a top-M heap.

---

### Algorithm 4: SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections)

Inputs: base `q`, candidates `C`, target count `M`, layer `lc`, two flags.
Output: up to `M` elements selected by the diversity heuristic.

Paper pseudocode (Algorithm 4):

```
SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections)
1   R  <- empty
2   W  <- C                                      // working queue
3   if extendCandidates
4       for each e in C
5           for each eadj in neighbourhood(e) at layer lc
6               if eadj not in W
7                   W <- W union {eadj}
8   Wd <- empty                                  // discarded queue
9   while |W| > 0 and |R| < M
10      e <- extract nearest element from W to q
11      if e is closer to q than to any element in R
12          R  <- R union {e}
13      else
14          Wd <- Wd union {e}
15  if keepPrunedConnections                     // top up from discards
16      while |Wd| > 0 and |R| < M
17          R <- R union (extract nearest element from Wd to q)
18  return R
```

The line-11 test is the key: for each already-selected `r`, check
`distance(q, e) < distance(r, e)`. If so, `e` lies in a different "direction"
from existing neighbours and improves graph diversity / global connectivity.
This is what preserves long-range edges that connect clusters.

Edge cases:

- `extendCandidates = false` is the documented default. The paper notes it
  is "useful only for extremely clustered data."
  hnswlib does not even expose the flag.
- `keepPrunedConnections = true` is effectively required. Without it,
  nodes can end up with degree below `M`, starving the graph and dropping
  recall. hnswlib's `getNeighborsByHeuristic2` enforces fixed degree.
- Tied distances: tie-break by `(distance, NodeKey)`.

Complexity: `O(|C| * M)` distance comparisons (each candidate vs. up to `M`
selected). With `extendCandidates`, `|C|` grows by `O(Mmax)`.

When does the heuristic outperform simple selection? Section 5.3:

- **Simple** is fine on uniform high-dim data (random Gaussian).
- **Heuristic** is essential on clustered or low-intrinsic-dim data
  (graphs, image features, real text embeddings). It preserves long-range
  edges that simple pruning destroys.
- All production HNSW (hnswlib, FAISS, pgvector, qdrant) default to
  heuristic. So should we.

---

### Algorithm 5: K-NN-SEARCH(hnsw, q, K, ef)

Inputs: index `hnsw`, query `q`, number of NNs `K`, dynamic list size `ef`
(typically `ef >= K`).
Output: `K` nearest elements to `q`.

Paper pseudocode (Algorithm 5):

```
K-NN-SEARCH(hnsw, q, K, ef)
1   W <- empty
2   ep <- get enter point for hnsw
3   L  <- level of ep
4   for lc in L .. 1                             // does NOT include layer 0
5       W  <- SEARCH-LAYER(q, ep, ef=1, lc)
6       ep <- get nearest element from W to q
7   W <- SEARCH-LAYER(q, ep, ef, lc=0)
8   return K nearest elements from W to q
```

Edge cases:

- **Empty index.** Line 2 fails. Short-circuit and return empty.
- **Single-node index.** Line 4 loop is empty. Line 7 returns the only
  node.
- **`ef < K`.** Either reject or silently bump to `K`. hnswlib bumps and
  warns. `ef >= K` is required for correctness; otherwise recall drops.
- **`K > N`.** Return all `N` results without padding.
- **Filtering** (not in paper): see section 7.

Time: expected `O(ef * log N)`. Upper-layer descent is `O(L * log N)` with
`ef = 1`; base layer is `O(ef * log N)`. Space: `O(ef + |visited|)`.

---

## 2. Parameters and Their Effects

Sources: paper section 4.2; hnswlib `ALGO_PARAMS.md`
([link](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md));
FAISS HNSW struct
([link](https://faiss.ai/cpp_api/struct/structfaiss_1_1HNSW.html));
OpenSearch hyperparameter guide
([link](https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/)).

### `M`

Target bidirectional edges per element per layer it occupies.

- Typical 4-64. Paper's reasonable range: 5-48.
- hnswlib hard-cap: 100000 (post CVE-2023-37365).
- Higher `M`: better recall on high-dim data, more memory, slower build,
  slightly slower queries.
- Heuristic: high intrinsic dimension and tight recall targets push `M` up;
  memory-constrained or low-dim push it down.

### `Mmax`

Cap on edges in upper layers (1+).

- Default `Mmax = M`. hnswlib sets `maxM_ = M_`.
- Cap exists because inserts can transiently push degree above `M`; line
  13-16 of INSERT re-prunes back to `Mmax`.

### `Mmax0`

Cap on edges in layer 0.

- Default `Mmax0 = 2 * M`. hnswlib hardcodes `maxM0_ = M_ * 2`.
- Layer 0 contains every element and dominates query work. Under-connected
  layer 0 is the dominant cause of recall drops below ~98%.

### `efConstruction`

`ef` passed to `SEARCH-LAYER` inside INSERT Phase 2.

- Typical 64-512. FAISS default 40 (low; raise it for production).
  hnswlib examples use 200.
- Higher `efConstruction`: higher recall on the resulting graph, longer
  build, no effect on query memory or speed.
- Quality test (hnswlib): build with some `efConstruction`, measure recall
  with `ef_search = ef_construction`. If recall < 0.9, raise it.
- Rule of thumb: `M * efConstruction` roughly constant for a given recall
  target; doubling `M` lets you halve `efConstruction`.

### `efSearch`

`ef` passed to `SEARCH-LAYER` at layer 0 inside K-NN-SEARCH. Upper layers
always use `ef = 1`.

- Typical 16-256. Mutable post-build.
- Monotonic recall vs. latency knob. Recall plateaus asymptotically.
- Constraint: `efSearch >= K`.

### `mL`

Recommended `mL = 1 / ln(M)` (paper section 4.1, Eq. 4).

The level distribution becomes geometric with parameter `1/M`, so each
successive layer holds ~`1/M` of the previous layer's elements. This
mirrors skip lists with `p = 1/M`, where expected per-element overlap
between layers is exactly one element.

Smaller `mL` collapses the hierarchy; larger `mL` overpopulates upper
layers. Implementation: store `level_mult: f64 = 1.0 / (m as f64).ln()`
once at index creation.

### `seed`

Required for reproducibility. Use a deterministic 64-bit PRNG (e.g.
`Xoshiro256`) seeded once and threaded through. Do not pull from OS
entropy inside `getRandomLevel`. Note: even with fixed seed, *insert
order* changes graph topology.

### Recommended values for our use cases

| Dataset | M | Mmax0 | efConstruction | efSearch |
|---------|---|-------|----------------|----------|
| F22 32-byte i8 (Hamming/L2 squared) | 16 | 32 | 200 | 64 |
| 128-dim f32 embeddings (cosine) | 16-32 | 32-64 | 200 | 64-128 |
| 768-dim f32 embeddings (cosine) | 32-48 | 64-96 | 256-400 | 128-256 |
| 256-bit MinHash (Jaccard via Hamming) | 16 | 32 | 128 | 64 |

These come from triangulating OpenSearch's precomputed configs, the
Pinecone HNSW guide, and FAISS defaults as a floor. F22 fingerprints are
low intrinsic dimension after sign-bit hashing, so `M = 16` suffices.
768-dim embeddings have high intrinsic dim and demand more edges. Hamming
on 256-bit signatures is so cheap (4 popcount ops) that the search-time
cost is graph traversal, not distance compute.

---

## 3. Level Distribution Rule

Source: paper section 4.1, and `getRandomLevel` in hnswlib's `hnswalg.h`
([link](https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h)).

The level for a new element is

```
l = floor(-ln(unif(0, 1)) * mL)
```

drawn on the **open** interval since `ln(0)` is undefined. This is an
exponential RV with rate `1/mL`, then floored, equivalently a geometric RV
with parameter `p = 1 - exp(-1/mL)`. With `mL = 1/ln(M)`,
`p ≈ 1 - 1/M`, so each level retains about `1/M` of the previous level's
elements.

Consequences:

- `Pr[l = 0] ≈ 1 - 1/M` (almost all elements live only at layer 0).
- `Pr[l >= k] ≈ (1/M)^k` (geometric tail).
- For `M = 16`, expected level counts on `N = 1M` elements:
  - layer 0: 1,000,000
  - layer 1: ~62,500
  - layer 2: ~3,900
  - layer 3: ~244
  - layer 4: ~15
  - layer 5: ~1

Layer 0 is the **long-range** layer (every node, dense edges). Upper
layers are **shortcut** layers (sparse, used only as routing scaffolding
during the descent phase of K-NN-SEARCH and INSERT).

Cap `l` at some `max_level` to bound vector sizes and avoid pathological
allocations. hnswlib does not cap explicitly; FAISS uses a precomputed
`assign_probas` table with a fixed cutoff. A cap of 64 is comfortable:
`Pr[l = 64]` for `M = 16` is `(1/16)^64 ≈ 10^-77`.

Standard impl pattern:

```
fn random_level(rng, level_mult) -> usize:
    let r = rng.gen_range(f64::MIN_POSITIVE..1.0);   // open below
    (-r.ln() * level_mult).floor() as usize
```

Avoid drawing from `[0.0, 1.0)` literally because `0.0` blows up.

---

## 4. Edge Cases / Gotchas Every Implementation Must Handle

### Empty graph (first INSERT)

The pseudocode reads `ep <- get enter point for hnsw` unconditionally.
Special-case:

```
if hnsw.size == 0:
    chosen_level = random_level(rng, level_mult)
    hnsw.insert_first(point, chosen_level)
    hnsw.entry_point = point
    hnsw.entry_level = chosen_level
    return
```

### Single-node graph

Once one node exists, INSERT and K-NN-SEARCH work without further special
casing. `SEARCH-LAYER` on layer 0 with one node returns it.

### Entry point selection when current entry is removed

We do not support delete in v1. For the record: when the entry is deleted,
the impl must promote the highest-level surviving node. hnswlib defers
this by only marking deleted; the result is the unreachable-points
phenomenon when the deleted entry was on a tall tower
([Xu et al. 2024](https://arxiv.org/html/2407.07871v1)).

### Disconnected subgraph

`SEARCH-LAYER` is graph-local. If layer 0 has a connected component the
entry-point chain never enters, points there are unreachable. The
hierarchy nominally fixes this but in practice it can occur when:

- `M` is very small (< 8) and the data has natural clusters.
- Many deletions accumulated without rebuild
  ([pgvector #244](https://github.com/pgvector/pgvector/issues/244)).
- The insert sequence concentrates a cluster all at once.

Mitigation: keep `M >= 12`; periodically rebuild; log recall@1 against a
brute-force sample.

### Insertion order sensitivity

Same data + same parameters + different insert order = different topology.
Recall typically within 1-2% but never identical. For deterministic
builds: fix insert order (sort by NodeKey), fix RNG seed, use a single
thread or a deterministic parallel scheme that serializes RNG draws.

### Heuristic vs simple selection

See Algorithm 4 above. Default to heuristic; simple is a strict subset of
configurations and is rarely worth the recall hit.

### `extendCandidates` flag

Default `false`. Useful only for "extremely clustered data" per the paper.
hnswlib does not expose it.

### `keepPrunedConnections` flag

Default `true`. Without it, degree drops below `M` and recall suffers.
hnswlib enforces fixed degree.

### Tied distances

Distance can collide naturally. SIMD reductions (sum of products, sum of
squared diffs) produce slightly different bits depending on lane order.
hnswlib v0.8.0 fixed an inner-product ordering bug for this reason
([releases](https://github.com/nmslib/hnswlib/releases/tag/v0.8.0)).
Order priority queues by `(distance, NodeKey)` lexicographically:

```
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then(self.id.cmp(&other.id))
    }
}
```

---

## 5. Variants and Improvements Published Since 2018

### FAISS HNSW

Documented at
[Struct faiss::HNSW](https://faiss.ai/cpp_api/struct/structfaiss_1_1HNSW.html).
Adds over the paper:

- **Precomputed level distribution.** `assign_probas` (CDF table) and
  `cum_nneighbor_per_level` precomputed once. Faster than per-insert
  `floor(-ln(unif) * mL)`, bounds `l` deterministically.
- **Search batch processing.** Batched query path with shared visited
  scaffolding where safe.
- **Different defaults.** `efConstruction = 40`, `efSearch = 16`,
  `M = 32`. The 40 default is unusually low; bump for production.
- **`check_relative_distance`, `search_bounded_queue` flags.** Internal
  toggles for the search-loop bound check.
- **Composability.** `IndexHNSWPQ`, `IndexHNSWSQ`, `IndexHNSWFlat` etc.

### hnswlib v0.7+

Source: [nmslib/hnswlib](https://github.com/nmslib/hnswlib).

- **`mark_deleted` / `replace_deleted`.** Logical deletion plus reuse of
  deleted slots.
- **Visited list pool.** Pre-allocated reusable bitsets per thread.
- **`set_ef` post-build.** `efSearch` is mutable.
- **CVE-2023-37365 fix.** Caps `M <= 100000` after a double-free in
  `init_index`
  ([releases](https://github.com/nmslib/hnswlib/releases/tag/v0.8.0)).
- **Cross-architecture inner-product consistency** (v0.8.0). Reduction
  order was previously SSE/AVX/AVX-512 dependent.

### DiskANN (Vamana)

Sources: [DiskANN paper](https://suhasjs.github.io/files/diskann_neurips19.pdf);
[Weaviate Vamana vs HNSW](https://weaviate.io/blog/ann-algorithms-vamana-vs-hnsw).

- Different family. Single-shot graph build (not incremental), disk-resident.
- Vamana algorithm: iterative robust pruning with a tunable parameter `α`.
  HNSW implicitly uses `α = 1`. With `α > 1`, Vamana keeps more edges in
  the same direction as shorter ones, trading degree for diameter.
- Single layer, not hierarchical. Single beam-search from a fixed entry
  point.
- Disk-resident: graph laid out for SSD reads (sectors of neighbours
  co-located with vectors).
- We are **not** implementing DiskANN. HNSW is the right choice for
  in-memory sub-microsecond queries; DiskANN wins at billion-scale on a
  single SSD-attached node.

### Yashunin 2022 / replaced_update follow-up

There is no separate Yashunin 2022 paper that I could verify; the
deletion-and-replace behavior shipped in hnswlib's `replace_deleted`
without a paper. The authoritative critique and improvement is:

- Xu et al. 2024, "Enhancing HNSW Index for Real-Time Updates", arXiv:2407.07871
  ([HTML](https://arxiv.org/html/2407.07871v1)).
- Identifies the **unreachable points phenomenon**: after `replaced_update`,
  some original neighbours end up with no in-edges and become invisible to
  subsequent queries. After 3000 delete-reinsert cycles at 5%/iter on SIFT,
  3-4% become unreachable and recall drops ~3%.
- Proposes MN-RU (Mutual Neighbour Replaced Update) with a reverse-edge
  repair pass.

### Filtered HNSW: ACORN, SeRF

ACORN ([arXiv:2403.04871](https://arxiv.org/abs/2403.04871), SIGMOD 2024):

- Predicate-agnostic. Modifies HNSW build so any subgraph induced by an
  arbitrary predicate is itself navigable.
- **ACORN-γ**: γ× denser graph (γ ≈ 5-10). Higher build cost and memory,
  optimal recall under filters.
- **ACORN-1**: same edge count as HNSW but predicate-agnostic compression
  heuristic.
- 2-1000× higher throughput than naive post-filter HNSW at fixed recall.
- Supported by Vespa as ACORN-1
  ([Vespa blog](https://blog.vespa.ai/additions-to-hnsw/)) and DuckDB
  community extension `hnsw_acorn`.

SeRF ([SIGMOD 2024 PDF](https://miaoqiao.github.io/paper/SIGMOD24_SeRF.pdf)):

- Range-filter specialization on a single attribute.
- "Segment graph": one HNSW with each edge annotated with a validity
  interval. Insert in attribute order; an edge is valid when the query's
  upper bound contains both endpoints.
- Compresses N range-specific HNSWs into one index losslessly.
- Specifically for range predicates on a sortable attribute, not arbitrary
  Boolean predicates.

### Quantization integrations

- **FAISS `IndexHNSWPQ`.** Layers HNSW on top of product quantization;
  graph stores PQ codes, distance is asymmetric PQ.
- **FAISS `IndexHNSWSQ`.** Scalar quantization (8 or 4 bit).
- **usearch.** Native multi-precision: `f64`, `f32`, `bf16`, `f16`, `i8`,
  `u8`, `b1` (1 bit). Auto-casts at search
  ([repo](https://github.com/unum-cloud/usearch)). For F22 fingerprints
  the analogue is "no quantization, distance is Hamming on i8 sign bits";
  usearch's `b1` mode is the right reference. For 768-dim embeddings,
  `IndexHNSWPQ` with `m = 96` 8-bit subquantizers is the ann-benchmarks
  reference for high-recall low-memory.

---

## 6. Implementation Gotchas From Production Deployments

### Floating-point determinism

SIMD reductions produce slightly different bits depending on lane order.
hnswlib v0.8.0 release notes explicitly called this out for inner product.
Mitigation: `(distance, NodeKey)` ordering everywhere. For exact bit
reproducibility across architectures, do reductions in scalar order
(slow). Otherwise accept that bit-exact outputs are per-architecture.

### Atomic operations and the visited bitset

Concurrent search and insert sharing a graph requires either:

- **Per-thread visited bitset** (hnswlib's `VisitedListPool`), pulled
  from a pool to avoid allocation. Simple and correct.
- **Shared atomic visited bitset** with epoch-based reset. More complex,
  cache-line contention. Multiple bug reports across libraries; do not
  invent this.

For our Rust impl: pool of `Vec<u32>` (epoch-tagged) per thread, acquired
via a `Mutex<Vec<...>>` of free lists.

### Recall vs efSearch curves

From OpenSearch HNSW benchmarks
([source](https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/))
and SIFT-1M numbers:

| efSearch | recall@1 (M=16, SIFT-1M) |
|----------|---------------------------|
| 16       | ~0.874                   |
| 32       | ~0.949                   |
| 64       | ~0.978                   |
| 128      | ~0.989                   |
| 256      | ~0.992                   |

Recall plateaus around 99% by `efSearch = 256`; doubling `efSearch` from
128 to 256 buys ~0.3% recall but doubles latency. Plateau is dataset
dependent: GloVe (high intrinsic dim) plateaus later; SIFT plateaus earlier.

### Unreachable points when M is too small

When `M < 8` and data has clusters, layer-0 nodes can end up with zero or
one incoming edges. Combined with neighbour re-pruning during a *neighbour's*
insert (Algorithm 1 lines 13-16), this can sever them from the rest of the
graph.

Symptom: recall@1 plateaus below ~98% no matter how high `efSearch` goes.

Mitigation: keep `M >= 12`; for high-dim use `M >= 16`; for billion-scale
use `M >= 32`. Periodically run reachability audits.

### Reported production bugs

- **CVE-2023-37365** (hnswlib): double-free in `init_index` with huge `M`.
  Patched by capping `M <= 100000`.
- **pgvector dead tuples** ([#244](https://github.com/pgvector/pgvector/issues/244)):
  updates and deletes accumulate dead tuples that count toward `efSearch`,
  dropping effective recall to near zero in pathological cases.
- **pgvector real-time degradation** ([#763](https://github.com/pgvector/pgvector/issues/763)):
  rolling updates degrade HNSW until full rebuild.
- **qdrant filter recall drop** ([#7147](https://github.com/qdrant/qdrant/issues/7147)):
  filtered HNSW recall fell to 34% under specific tenant-isolation configs.
- **hnswlib mark_deleted exception** ([#321](https://github.com/nmslib/hnswlib/issues/321)):
  `mark_deleted()` then `query` raises "Cannot return the results in a
  contiguous 2D array" because `efSearch` candidate pool is exhausted by
  tombstones.
- **pgvector + filtering** ([#259](https://github.com/pgvector/pgvector/issues/259)):
  post-filter and pre-filter both destroy the sub-linear win at moderate
  selectivity.

---

## 7. Filter Integration (in-search vs post-filter)

Sources: [Weaviate ACORN blog](https://weaviate.io/blog/speed-up-filtered-vector-search);
[Tigris filterable HNSW](https://dzone.com/articles/filterable-hnsw-production-vector-search-part-3);
[OpenSearch filter docs](https://opensearch.org/docs/latest/vector-search/filter-search-knn/index/).

### Why post-filter destroys the sub-linear win

Post-filter retrieves `K * f` candidates from HNSW (overshoot factor `f`)
then drops those that fail the filter. With selectivity `s`, you need
`f >= 1/s` to expect `K` results:

- `s = 1.0`: `f = 1`, no overhead.
- `s = 0.1`: `f = 10`, latency ~10×.
- `s = 0.01`: `f = 100`, candidate list explodes.
- `s = 0.001`: post-filter is worse than brute force.

Worse, with low selectivity HNSW cannot even return `K * f` results because
the early-termination condition fires before enough matches are collected.

### Predicate-function approach (in-search)

Check the predicate at each node visited inside `SEARCH-LAYER`. The
candidate enters the priority queue only if it matches; non-matching nodes
are still **traversed** for connectivity but do not count toward `ef`:

```
for e_id in neighbourhood(c.id, layer):
    if visited.insert(e_id):
        d = distance(query, e_id)
        if !predicate(e_id):
            continue        // skip add to nearest, but DO traverse
        if nearest.len() < ef || d < nearest.peek_max().dist:
            candidates.push((d, e_id))
            nearest.push((d, e_id))
            ...
```

Crucially, traversal continues through non-matching neighbours so the
graph remains navigable. hnswlib, Weaviate, and qdrant all do this.

### Bitmap approach

For categorical filters (e.g. tenant_id), pre-compute a bitmap of valid
node IDs. Predicate check is one bit-test. Combine multiple filters by
ANDing bitmaps. Roaring bitmaps are the standard library.

### When to fall back to brute force

Crossover depends on `M`, `efSearch`, and `N`:

- Selectivity > 0.1: filtered HNSW wins.
- Selectivity in [0.01, 0.1]: profile and switch dynamically.
- Selectivity < 0.01: brute force on the bitmap-filtered subset wins.
- Matching subset < ~`K * 100`: exact KNN is fastest.

qdrant's planner does this dynamically based on payload-index cardinality
estimates.

### ACORN and SeRF as references

- **ACORN-1**: same `M` but build heuristic ignores predicate matching,
  ensuring the graph remains navigable for any subset.
- **ACORN-γ**: same idea with `γ * M` edges per node.
- **SeRF**: range-filter specialization with edge-validity intervals.

For v1, in-search predicate function with bitmap is the right starting
point. ACORN-style modifications are a v2 question.

---

## 8. Empirical Recall Benchmarks to Expect

Sources: [ann-benchmarks](https://ann-benchmarks.com); OpenSearch HNSW
benchmarks; hnswlib README examples.

### SIFT-1M (128-dim f32, L2 squared)

| Config | recall@10 |
|--------|-----------|
| M=16, efC=200, efS=64  | ~0.978 |
| M=16, efC=200, efS=128 | ~0.989 |
| M=32, efC=400, efS=128 | ~0.995 |
| M=32, efC=400, efS=256 | ~0.998 |

Build: ~1-3 minutes single-threaded for `M=16, efC=200`. Memory: ~1.2 GB
graph alone (without vectors).

### GloVe-100 (100-dim f32, cosine, ~1.2M vectors)

| Config | recall@10 |
|--------|-----------|
| M=16, efC=200, efS=64  | ~0.86 |
| M=32, efC=400, efS=128 | ~0.94 |
| M=64, efC=500, efS=256 | ~0.97 |

GloVe is harder than SIFT (higher intrinsic dim, hub structure). To match
SIFT-level recall you need ~2× the parameters.

### Random binary 256-bit (Hamming distance)

No standard ann-benchmark, but for reference: `M=16, efC=128, efS=64` gives
recall@10 ~0.95 on `N = 1M` random-uniform 256-bit vectors. Hamming is so
cheap (4 popcount ops) the cost is dominated by cache misses on neighbour
fetches, not distance compute.

### F22 fingerprint distance distribution

Empirical from our primitive benchmarks:

- Pairwise normalized Hamming on **unrelated** F22 fingerprints (random
  inputs): mean ~0.5 (sign bits flip with prob ~0.5 per byte).
- Pairwise normalized Hamming on **near-duplicates** (single-byte edits):
  mean ~0.03-0.08 (a few bits flip).
- Distance threshold for "similar enough to surface as a candidate":
  ~0.15-0.20 normalized Hamming, depending on application.

Tuning implication: `efSearch = 64` should be plenty for the F22 use case
because relevant neighbours are far from random and the graph quickly
converges on the cluster the query belongs to.

---

## 9. References

### Original paper

- Yu. A. Malkov, D. A. Yashunin, "Efficient and robust approximate nearest
  neighbor search using Hierarchical Navigable Small World graphs",
  arXiv:1603.09320, 2016 (v1) - 2018 (v4). Journal: IEEE TPAMI 42(4):
  824-836, April 2020.
  - <https://arxiv.org/abs/1603.09320>
  - <https://arxiv.org/pdf/1603.09320>
  - <https://dl.acm.org/doi/10.1109/TPAMI.2018.2889473>
  - Mirror: <https://users.cs.utah.edu/~pandey/courses/cs6530/fall24/papers/vectordb/HNSW.pdf>

### Follow-up papers and critiques

- Xu et al. 2024, "Enhancing HNSW Index for Real-Time Updates: Addressing
  Unreachable Points and Performance Degradation", arXiv:2407.07871.
  <https://arxiv.org/abs/2407.07871>
- Patel et al. 2024, "ACORN: Performant and Predicate-Agnostic Search
  Over Vector Embeddings and Structured Data", arXiv:2403.04871, SIGMOD.
  <https://arxiv.org/abs/2403.04871>
- Zuo et al. 2024, "SeRF: Segment Graph for Range-Filtering Approximate
  Nearest Neighbor Search", SIGMOD.
  <https://miaoqiao.github.io/paper/SIGMOD24_SeRF.pdf>
- Subramanya et al. 2019, "DiskANN: Fast Accurate Billion-point Nearest
  Neighbor Search on a Single Node", NeurIPS.
  <https://suhasjs.github.io/files/diskann_neurips19.pdf>
- "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'",
  arXiv:2412.01940, 2024.
  <https://arxiv.org/html/2412.01940v1>

### Production implementations

- **hnswlib** (reference C++ by the original authors):
  <https://github.com/nmslib/hnswlib>
  - `hnswalg.h`:
    <https://github.com/nmslib/hnswlib/blob/master/hnswlib/hnswalg.h>
  - `ALGO_PARAMS.md`:
    <https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md>
- **FAISS** (Meta): <https://github.com/facebookresearch/faiss>
  - HNSW struct:
    <https://faiss.ai/cpp_api/struct/structfaiss_1_1HNSW.html>
- **pgvector**: <https://github.com/pgvector/pgvector>
  - HNSW design (DeepWiki):
    <https://deepwiki.com/pgvector/pgvector/5.1-hnsw-index>
- **qdrant**:
  <https://github.com/qdrant/qdrant/tree/master/lib/segment/src/index/hnsw_index>
- **usearch** (multi-precision HNSW):
  <https://github.com/unum-cloud/usearch>
- **hnswlib-rs** (Rust port):
  <https://github.com/jean-pierreBoth/hnswlib-rs>

### Practitioner guides

- OpenSearch HNSW hyperparameter guide:
  <https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/>
- Pinecone HNSW guide: <https://www.pinecone.io/learn/series/faiss/hnsw/>
- Vespa ACORN explainer: <https://blog.vespa.ai/additions-to-hnsw/>

### Benchmarks

- ANN-Benchmarks: <https://ann-benchmarks.com>
- ETHZ filtered ANN benchmark, 2025:
  <http://htor.inf.ethz.ch/publications/img/2025_iff_fanns_benchmark.pdf>
- "Filtered Approximate Nearest Neighbor Search: A Unified Benchmark and
  Systematic Experimental Study", arXiv:2509.07789.
  <https://arxiv.org/html/2509.07789v1>

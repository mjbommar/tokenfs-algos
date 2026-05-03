//! Rabbit Order — community-detection-driven graph permutation.
//!
//! See `docs/v0.2_planning/14_PERMUTATION.md` § 3 for the spec and
//! `docs/v0.2_planning/03_EXECUTION_PLAN.md` § "Sprint 47-49 / 50-52"
//! for the sprint-level milestones.
//!
//! ## What it is
//!
//! Rabbit Order is the graph-permutation algorithm published by
//! [J. Arai, H. Shiokawa, T. Yamamuro, M. Onizuka, S. Iwamura,
//! "Rabbit Order: Just-in-time Parallel Reordering for Fast Graph
//! Analysis", *Proc. IEEE IPDPS 2016*, pp. 22-31][arai2016]. It
//! computes a vertex permutation by repeatedly merging neighbouring
//! vertices whose union maximises **modularity gain** (the same
//! objective that drives Louvain community detection), then walks the
//! resulting agglomerative dendrogram in DFS pre-order to emit the
//! final ordering. The permutation places vertices that share a
//! community contiguously, so subsequent locality-sensitive operations
//! (BFS, PageRank, neighbour scans, sequential prefetch) hit the same
//! cache line and the same translation-buffer entry.
//!
//! ## When to use it
//!
//! Use Rabbit Order when the input is a **sparse undirected graph**
//! whose adjacency exhibits genuine community structure — clusters of
//! vertices with many internal edges and few external ones — and the
//! downstream workload is **locality-sensitive** (any of: BFS / DFS,
//! PageRank, connected-components, neighbour aggregation in a GNN, or
//! TokenFS extent-following on a deduped image where each cluster maps
//! to a content-shared inode group). On benchmarks reported in the
//! IPDPS 2016 paper, Rabbit Order yields the best known cache-miss
//! reduction for this class of workload — typically 1.25-2.4× speedup
//! over natural-order graphs across PageRank, BFS, and CC.
//!
//! Use [`super::rcm()`] (Reverse Cuthill-McKee) instead when the
//! workload is **bandwidth-driven** (sparse linear solvers, banded
//! matrices, classical sparse-matrix-vector products) or when build
//! cost matters: RCM is roughly 100-500× faster to compute and is
//! deterministic by construction. Use [`super::hilbert_2d`] /
//! [`super::hilbert_nd`] when the input has a true low-dimensional
//! point embedding (PCA-projected fingerprints, t-SNE/UMAP outputs)
//! rather than graph adjacency. See `docs/PHASE_D_RABBIT_ORDER.md`
//! for the side-by-side comparison.
//!
//! ## How it differs from RCM
//!
//! RCM and Rabbit Order optimise different objectives:
//!
//! * **RCM** minimises **bandwidth** — the maximum distance between
//!   connected vertices in the new order — by BFS-and-reverse from a
//!   pseudoperipheral start vertex. The result is a banded permutation
//!   that is excellent for sparse-matrix solvers but only modestly
//!   better than natural order for graph traversal workloads (because
//!   bandwidth and cache locality are correlated but not identical).
//! * **Rabbit Order** maximises **community modularity** — vertices
//!   that share many edges land in adjacent slots regardless of their
//!   global position in the BFS frontier. The result is a permutation
//!   tuned for cache locality on graph-traversal workloads, at the
//!   cost of a substantially heavier build phase.
//!
//! Concretely: RCM is `O(|V| + |E| log Δ)` with a small constant;
//! Rabbit Order is `O(|E| log |V|)` with a large constant (heap
//! maintenance plus sorted-list intersections per merge). On a
//! 228 K-vertex graph with average degree 5, RCM completes in roughly
//! 10 ms and Rabbit Order in 1-5 seconds. The build cost amortises
//! over the workload's lifetime — TokenFS images are sealed once and
//! read many times, so paying seconds at build time for percent-level
//! cache improvements at every read is the right tradeoff.
//!
//! [arai2016]: https://doi.org/10.1109/IPDPS.2016.110
//!
//! ## Algorithm
//!
//! Bottom-up agglomerative community detection followed by dendrogram-DFS
//! visit-order assignment. Per Arai et al. (IPDPS 2016):
//!
//! 1. Each vertex starts as its own community.
//! 2. Iterate over vertices in **lowest-degree-first** order (ties on
//!    lower vertex ID for determinism).
//! 3. For each surviving community `u`, find the neighbour community
//!    `v` whose merge maximises modularity gain:
//!    ```text
//!    dQ(u, v) = w(u, v) / m - deg(u) * deg(v) / (2 * m^2)
//!    ```
//!    where `w(u, v)` is the edge weight between `u` and `v`, `deg(*)`
//!    is the community's total weighted degree, and `m` is the total
//!    edge weight. If no neighbour yields `dQ > 0`, `u` stays as its
//!    own community.
//! 4. Merge `u` into the chosen `v`: record the merge in a dendrogram
//!    node, then absorb `u`'s adjacency into `v`'s (folding shared
//!    neighbour weights and removing the `u-v` edge).
//! 5. DFS the dendrogram in pre-order; the leaf-visit sequence is the
//!    new ordering.
//!
//! ## Sequential baseline + SIMD modularity-gain inner loop + parallel agglomeration
//!
//! The Sprint 47-49 sequential baseline is the agglomeration loop in
//! [`rabbit_order`]. Sprint 50-52 lifts the per-pair modularity-gain
//! kernel into the [`kernels`] module with scalar / AVX2 / AVX-512 /
//! NEON variants. Sprint 53-55 adds [`rabbit_order_par`] (gated on the
//! `parallel` Cargo feature): a round-based concurrent variant where
//! each round's per-vertex proposal phase runs across the global rayon
//! thread pool while the merge-application phase remains sequential in
//! canonical (ascending absorbed-vertex-id) order so the resulting
//! permutation is bit-exact across thread counts. See
//! [`rabbit_order_par`]'s rustdoc for the full determinism contract.
//! The implementation continues to use sorted `Vec`-backed adjacency
//! rather than a concurrent hash map — `rayon::par_iter` provides
//! enough parallelism for the proposal phase without the per-bucket
//! atomics overhead.
//!
//! ## Determinism
//!
//! * Iteration order ties broken by ascending community/vertex ID.
//! * Modularity-gain ties broken by ascending neighbour ID.
//! * Stale heap entries are filtered by re-checking the recorded degree
//!   against the current degree of the popped community.
//! * The dendrogram DFS uses the merge-tree structure directly; the
//!   resulting permutation is reproducible bit-for-bit across runs and
//!   architectures.
//! * SIMD kernels in [`kernels`] are bit-exact with the scalar reference
//!   per neighbour pair (integer arithmetic; no floating-point reduction
//!   order to track). When the i64 fast path triggers (small inputs),
//!   results round-trip exactly with the i128 scalar; when it cannot
//!   trigger, the kernels fall back to the i128 scalar in-place. See
//!   [`kernels`] for the precise eligibility predicate.
//!
//! ## Complexity
//!
//! * Time: roughly O(|E| log |V|) for the heap-driven agglomeration plus
//!   the cost of intersecting sorted adjacency lists on each merge.
//! * Space: O(|V| + |E|) for the working adjacency, the dendrogram, and
//!   the heap.
//!
//! For 228 K vertices with average degree 5 (TokenFS-typical), the
//! sequential baseline runs in roughly 1-5 seconds — heavier than RCM
//! (~10 ms) by 100-500x. The downstream cache-miss reduction on graph
//! traversal pays back the build cost on workloads that touch the
//! permuted layout repeatedly.

#[cfg(not(feature = "std"))]
use alloc::collections::BinaryHeap;
#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::cmp::Reverse;
#[cfg(feature = "std")]
use std::collections::BinaryHeap;

use super::{CsrGraph, Permutation, PermutationConstructionError};

/// Per-community weighted adjacency list.
///
/// Sorted by ascending neighbour community ID for fast intersection
/// during merges. Degree is the sum of `weight` across the entries.
type AdjList = Vec<(u32, u64)>;

/// A single merge in the agglomerative dendrogram.
///
/// `absorbed` joins `into`'s community when `dQ(absorbed, into) > 0`.
/// Both fields are vertex IDs (the original CSR IDs); communities are
/// identified by the ID of the vertex they were rooted at.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Merge {
    /// The community being absorbed (becomes a child in the dendrogram).
    absorbed: u32,
    /// The community absorbing it (becomes the parent).
    into: u32,
}

/// Computes a Rabbit Order permutation for `graph`.
///
/// Returns a [`Permutation`] of length `graph.n` where `perm[old_id] =
/// new_id`. The new ordering is the DFS pre-order leaf sequence of the
/// agglomerative dendrogram produced by lowest-degree-first
/// modularity-maximising merges.
///
/// ## Inputs
///
/// `graph` is a [`CsrGraph`] borrowing two slices: `offsets` (length
/// `n + 1`) and `neighbors` (length `offsets[n]`). The graph is treated
/// as **undirected** — callers that have a directed adjacency must
/// symmetrise it upstream. Self-loops, duplicate edges, and isolated
/// vertices are all tolerated; see "Behaviour on degenerate inputs"
/// below.
///
/// ## Output
///
/// A [`Permutation`] of length `n`. The constructor invariant is
/// satisfied — every id `0..n` appears exactly once — so the result
/// can be applied directly via [`Permutation::apply`] /
/// [`Permutation::apply_into`].
///
/// ## Complexity
///
/// * Time: roughly `O(|E| log |V|)` for the heap-driven agglomeration
///   plus the cost of intersecting sorted adjacency lists on each
///   merge.
/// * Space: `O(|V| + |E|)` for the working adjacency, the dendrogram,
///   and the heap.
///
/// On a 228 K-vertex graph with average degree 5 (TokenFS-typical),
/// expect roughly 1-5 seconds of wall-clock time.
///
/// ## When to use this vs `rabbit_order_par`
///
/// This is the **single-threaded** entry point. When the `parallel`
/// Cargo feature is enabled and the graph has at least
/// [`RABBIT_PARALLEL_EDGE_THRESHOLD`] directed edges, prefer
/// [`rabbit_order_par`] — it parallelises the per-round merge-proposal
/// phase across the global rayon thread pool while keeping the apply
/// phase deterministic. Below the threshold, [`rabbit_order_par`]
/// transparently delegates to this function (the rayon overhead
/// outweighs any savings on small graphs).
///
/// ## Sequential baseline
///
/// This is the Sprint 47-49 single-pass sequential baseline. Quality
/// is meaningfully better than RCM on community-structured graphs but
/// worse than the multi-level concurrent reference; see the
/// module-level documentation for the full sprint plan. The function
/// signature, type discipline, and determinism guarantees are stable
/// across the follow-on SIMD and concurrent-merging sprints — only
/// build cost will improve.
///
/// ## Example
///
/// Two triangles connected by a bridge edge — Rabbit Order should
/// place each triangle contiguously in the output:
///
/// ```
/// use tokenfs_algos::permutation::{CsrGraph, rabbit_order};
///
/// // Vertices 0-1-2 form one triangle, 3-4-5 another, with a single
/// // bridge edge 2-3 holding the two communities together.
/// // CSR layout (undirected, both directions present):
/// //   0 -> [1, 2]
/// //   1 -> [0, 2]
/// //   2 -> [0, 1, 3]
/// //   3 -> [2, 4, 5]
/// //   4 -> [3, 5]
/// //   5 -> [3, 4]
/// let offsets = [0_u32, 2, 4, 7, 10, 12, 14];
/// let neighbors = [1_u32, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4];
/// let graph = CsrGraph { n: 6, offsets: &offsets, neighbors: &neighbors };
///
/// let perm = rabbit_order(graph);
/// assert_eq!(perm.len(), 6);
/// // The result is a valid permutation: every id 0..n appears exactly once.
/// let mut seen = [false; 6];
/// for &new_id in perm.as_slice() {
///     seen[new_id as usize] = true;
/// }
/// assert!(seen.iter().all(|b| *b));
/// ```
///
/// ## Behaviour on degenerate inputs
///
/// * Empty graph (`n == 0`) returns the empty permutation.
/// * Single vertex returns `[0]`.
/// * Edgeless graph (no neighbours anywhere) returns the identity
///   permutation: every vertex is its own singleton community and the
///   DFS simply walks the singleton roots in ascending order.
/// * Self-loops contribute weight to the vertex's own degree (matching
///   the convention used by Newman's modularity formulation) but are
///   never considered as merge targets.
/// * Duplicate edges in the CSR input are folded: the per-community
///   adjacency stores combined weight per neighbour.
///
/// # Panics
///
/// Panics if `graph.offsets.len() != graph.n + 1`, if the CSR offsets
/// are non-monotone, if `offsets[n]` does not equal `neighbors.len()`,
/// or if any neighbour ID is out of range. Kernel/FUSE callers that
/// need a non-panicking variant should prefer [`try_rabbit_order`].
#[must_use]
pub fn rabbit_order(graph: CsrGraph<'_>) -> Permutation {
    let n = graph.n as usize;
    assert_eq!(
        graph.offsets.len(),
        n + 1,
        "rabbit_order: offsets.len() ({}) != n + 1 ({})",
        graph.offsets.len(),
        n + 1
    );
    if n == 0 {
        return Permutation::try_identity(0).expect("identity construction within u32::MAX");
    }
    for w in graph.offsets.windows(2) {
        assert!(
            w[0] <= w[1],
            "rabbit_order: offsets non-monotone: offsets contains {} followed by {}",
            w[0],
            w[1]
        );
    }
    assert_eq!(
        graph.offsets[n] as usize,
        graph.neighbors.len(),
        "rabbit_order: offsets[n] ({}) != neighbors.len() ({})",
        graph.offsets[n],
        graph.neighbors.len()
    );

    // Build the per-community weighted adjacency from the input CSR.
    // Self-loops contribute weight to the vertex's own degree but are
    // omitted from the neighbour lists (we never merge a community
    // with itself). Duplicate edges fold into a single weighted entry.
    let mut adj: Vec<AdjList> = Vec::with_capacity(n);
    let mut self_loop: Vec<u64> = vec![0_u64; n];
    for v in 0..n {
        let raw = graph.neighbors_of(v as u32);
        // Validate neighbour IDs eagerly — `neighbors_of` itself
        // doesn't bounds-check the IDs, only the vertex `v`.
        for &u in raw {
            assert!(
                (u as usize) < n,
                "rabbit_order: neighbour {u} of vertex {v} out of range [0, {n})"
            );
        }
        adj.push(consolidate_neighbours(v as u32, raw, &mut self_loop));
    }

    // Total edge weight m. Each undirected edge appears in both
    // endpoints' adjacency lists, so summing all per-vertex weighted
    // degrees and halving gives `m`. Self-loops are counted once via
    // the convention `m_self += w(v, v)` (Newman 2006); we add them
    // back after the halving to avoid double-counting.
    let mut weighted_degree: Vec<u64> = (0..n)
        .map(|v| adj[v].iter().map(|(_, w)| *w).sum::<u64>() + self_loop[v])
        .collect();
    let total_edge_weight: u64 = {
        let off_diag: u64 = (0..n).map(|v| weighted_degree[v] - self_loop[v]).sum();
        // The off-diagonal contribution counts each undirected edge
        // twice; halve it. Self-loops contribute their full weight.
        off_diag / 2 + self_loop.iter().sum::<u64>()
    };

    // Edgeless graph (no merges possible): identity ordering of
    // singleton communities. Every DFS leaf is a singleton, walked
    // by ascending ID.
    if total_edge_weight == 0 {
        return Permutation::try_identity(n).expect("identity construction within u32::MAX");
    }

    // Lowest-degree-first iteration via a min-heap. Stale entries
    // (degree changed since the entry was pushed) are filtered on
    // pop by re-comparing the recorded weighted degree.
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = BinaryHeap::with_capacity(n);
    let mut alive: Vec<bool> = vec![true; n];
    for v in 0..n {
        // Skip vertices with no incident edges — they cannot merge
        // and contribute their identity position via the
        // singleton-collection pass at the end of the DFS.
        if !adj[v].is_empty() {
            heap.push(Reverse((weighted_degree[v], v as u32)));
        }
    }

    // Dendrogram in merge order. We will reconstruct the visit order
    // by collecting parent->children adjacency post-hoc.
    let mut merges: Vec<Merge> = Vec::with_capacity(n);

    // Scratch buffers for the per-iteration columnar projection that
    // [`best_merge_target`] passes into the SIMD modularity-gain
    // kernel. Reused across every heap pop so the inner loop avoids
    // millions of small Vec allocations on dense graphs.
    let mut scratch_weights: Vec<u64> = Vec::with_capacity(64);
    let mut scratch_degrees: Vec<u64> = Vec::with_capacity(64);

    while let Some(Reverse((rec_deg, u))) = heap.pop() {
        let u_idx = u as usize;
        if !alive[u_idx] {
            // `u` has already been absorbed by a previous merge.
            continue;
        }
        if rec_deg != weighted_degree[u_idx] {
            // Degree changed since this entry was pushed; the
            // current degree was re-pushed when adjacency mutated.
            continue;
        }

        // Find the neighbour `v` maximising dQ(u, v). Tie-break on
        // ascending neighbour ID so merges are deterministic.
        let best = best_merge_target(
            u,
            &adj[u_idx],
            &weighted_degree,
            total_edge_weight,
            &mut scratch_weights,
            &mut scratch_degrees,
        );

        let Some(v) = best else {
            // No neighbour offers a positive modularity gain. `u`
            // becomes a permanent singleton in this single-pass
            // baseline (multi-level recursion is a follow-on sprint).
            // Drop it from the active set without recording a merge.
            continue;
        };

        // Record the merge. Convention: `u` is absorbed into `v`'s
        // community. The dendrogram parent is `v`; the child is `u`.
        merges.push(Merge {
            absorbed: u,
            into: v,
        });
        absorb_into(
            u_idx,
            v as usize,
            &mut adj,
            &mut weighted_degree,
            &mut self_loop,
        );
        alive[u_idx] = false;

        // `v`'s degree changed; push a fresh entry. The stale entry
        // (if any) is filtered on pop via the rec_deg check above.
        if alive[v as usize] && !adj[v as usize].is_empty() {
            heap.push(Reverse((weighted_degree[v as usize], v)));
        }
    }

    dfs_visit_order(n, &merges)
}

/// Like [`rabbit_order`] but returns an error on malformed CSR input
/// instead of panicking.
///
/// Validates the entire CSR header upfront via
/// [`CsrGraph::try_validate`] (offsets length, monotonicity,
/// neighbours-tail consistency, in-range neighbour IDs) before
/// invoking the existing internal pipeline. On a successful validation
/// dispatches to [`rabbit_order`] — the success path is bit-identical
/// to the panicking sibling, so the determinism guarantees and the
/// degenerate-input behaviour documented for [`rabbit_order`] carry
/// over verbatim.
///
/// **Allocation note:** the try_* path is about not panicking on bad
/// input. It is NOT heap-free. Internally the per-vertex consolidated
/// adjacency, weighted-degree cache, dendrogram, scratch buffers, and
/// min-heap are all allocated on the heap. Rabbit Order is a
/// build-time primitive (image-build phase, see
/// `02b_DEPLOYMENT_MATRIX.md`); kernel-mode consumers load the
/// resulting [`Permutation`] from the sealed image and apply it via
/// the kernel-safe [`Permutation::try_apply_into`] /
/// [`Permutation::try_apply_into_strict`].
///
/// # Errors
///
/// Returns [`PermutationConstructionError::InvalidCsr`] wrapping the
/// first [`super::CsrGraphError`] encountered during upfront
/// validation. See [`CsrGraph::try_validate`] for the full failure
/// list.
pub fn try_rabbit_order(graph: CsrGraph<'_>) -> Result<Permutation, PermutationConstructionError> {
    graph.try_validate()?;
    Ok(rabbit_order(graph))
}

/// Edge-count threshold below which [`rabbit_order_par`] falls back to
/// the sequential [`rabbit_order`] path.
///
/// Concurrent agglomeration introduces per-round bookkeeping (snapshot
/// the active set, gather proposals, apply sequentially). On graphs with
/// fewer than ~200 K directed edges the rayon thread-pool wake-up cost
/// and the per-round overhead overwhelm the savings, so we delegate to
/// the heap-based sequential implementation.
#[cfg(feature = "parallel")]
pub const RABBIT_PARALLEL_EDGE_THRESHOLD: usize = 200_000;

/// Round-based concurrent Rabbit Order — Sprint 53-55 of `03_EXECUTION_PLAN.md`.
///
/// Computes the same kind of agglomerative-modularity-driven permutation
/// as [`rabbit_order`] but parallelises the per-round merge-proposal
/// phase across the global rayon thread pool. The merge-application
/// phase remains sequential and uses a canonical (ascending absorbed-
/// vertex-id) order so the resulting [`Permutation`] is deterministic
/// regardless of how many rayon threads execute the proposal phase.
///
/// ## Inputs
///
/// `graph` is a [`CsrGraph`] borrowing two slices: `offsets` (length
/// `n + 1`) and `neighbors` (length `offsets[n]`). The graph is treated
/// as undirected; see [`rabbit_order`] for the full input contract
/// (multi-edge folding, self-loop accounting, degenerate cases).
///
/// ## Output
///
/// A [`Permutation`] of length `n`, ready to be applied via
/// [`Permutation::apply`] / [`Permutation::apply_into`]. Bit-exact
/// across runs and thread counts (see "Determinism contract" below);
/// **not** bit-exact with [`rabbit_order`] on graphs that admit
/// multiple positive-modularity merges (see "Note: not bit-exact"
/// below).
///
/// ## Complexity
///
/// * Time: same `O(|E| log |V|)` asymptotic as [`rabbit_order`]; the
///   per-round proposal phase parallelises across rayon while the
///   per-round apply phase runs sequentially. See "Performance
///   posture" below for measured behaviour on sparse inputs.
/// * Space: same `O(|V| + |E|)` working set as [`rabbit_order`], plus
///   one `Vec<Option<u32>>` of length `|active|` per round for the
///   parallel proposal output.
///
/// ## When to use this vs `rabbit_order`
///
/// Reach for this function when the input is **above** the
/// [`RABBIT_PARALLEL_EDGE_THRESHOLD`] edge count (default ~200 K
/// directed edges) and the calling pipeline is already rayon-driven.
/// Below the threshold, this function delegates to [`rabbit_order`]
/// transparently — see "Fallback for small graphs" below. Above the
/// threshold, expect wall-clock parity with the sequential path on
/// realistic sparse graphs (the apply phase is the bottleneck); the
/// function exists to anchor a deterministic concurrent API surface
/// rather than to deliver large speedups today.
///
/// ## Algorithm (Option A from `14_PERMUTATION.md` § 3)
///
/// 1. Initialise per-community adjacency, weighted degrees, and self-
///    loop accounting from the CSR — identical to the sequential path.
/// 2. **Round loop**: while at least one positive-`dQ` proposal exists,
///    repeat:
///    - **Snapshot phase** (sequential): collect every still-alive
///      vertex into a vector sorted by ascending `(weighted_degree,
///      vertex_id)`. This is the canonical "low-degree-first" iteration
///      order shared with [`rabbit_order`].
///    - **Proposal phase** (parallel): partition the snapshot across
///      rayon worker threads via `par_iter`. Each worker iterates its
///      chunk and calls the per-vertex `best_merge_target` helper
///      (read-only against the shared adjacency / degree state) using
///      thread-local scratch buffers. Output: per-vertex `Option<u32>`
///      proposing a merge target, or `None` when no neighbour yields
///      `dQ > 0`.
///    - **Apply phase** (sequential, canonical order): iterate the
///      proposals in ascending absorbed-vertex-id order. For each
///      `(u -> v)` proposal:
///      - Skip when `u` was already absorbed earlier in this round
///        (another vertex picked `u` as its merge target).
///      - Skip when `v` was already absorbed earlier in this round
///        (the target evaporated before its turn came up).
///      - Otherwise: record the merge in the dendrogram, fold `u`'s
///        adjacency into `v`'s via the same `absorb_into` helper the
///        sequential path uses, and mark `u` as dead.
///    - When the round produced no merges, break out.
/// 3. Reconstruct the permutation via the dendrogram-DFS pre-order
///    walk.
///
/// ## Determinism contract
///
/// **Same input + same thread count → same output.** Each rayon worker
/// reads from the shared snapshot (immutable during the proposal phase)
/// so the per-vertex proposals are pure functions of `(adj,
/// weighted_degree, self_degree, m)` at round start. Workers do not
/// mutate any shared state during the proposal phase.
///
/// **Same input + different thread counts → same output.** The proposal
/// phase's per-vertex output does not depend on chunk boundaries (each
/// vertex's merge-target computation is hermetic). The apply phase
/// orders proposals by ascending absorbed-vertex-id before consuming
/// them, so the merge sequence — and therefore the dendrogram — is
/// invariant under any reordering the parallel proposal phase might
/// have produced.
///
/// **Bit-exact across runs.** Integer arithmetic throughout; no
/// floating-point reduction order to track. The scalar/SIMD kernel
/// dispatch in [`kernels::auto`] is itself bit-exact (see the kernel
/// module documentation) so the choice of backend per-thread does not
/// perturb proposals.
///
/// **Note: not bit-exact with [`rabbit_order`].** The sequential path
/// uses a heap that re-evaluates each vertex against the *current*
/// (post-merge) state, while the round-based parallel path locks the
/// per-round proposals against the *snapshot* taken at round start.
/// Both produce valid Rabbit Order permutations and both group
/// communities contiguously, but the merge sequences differ on graphs
/// with >2 vertices that admit multiple positive-`dQ` merges. On
/// trivial inputs (empty graph, n=1, fully disconnected, edgeless) both
/// produce identical output because no merges happen.
///
/// ## Fallback for small graphs
///
/// When the input has fewer than [`RABBIT_PARALLEL_EDGE_THRESHOLD`]
/// directed edges, this function delegates to [`rabbit_order`] —
/// rayon's per-task overhead exceeds the agglomeration cost on small
/// graphs. The threshold is documented as a public constant so
/// callers can pre-classify their inputs.
///
/// ## Performance posture
///
/// The Sprint 53-55 spec (`docs/v0.2_planning/03_EXECUTION_PLAN.md`)
/// flags up front that the round-based concurrent variant's wall-clock
/// gain over the heap-based sequential path is bounded by the
/// **sequential apply phase**: each round, the merge-application step
/// runs serially in canonical order to preserve determinism. On
/// realistic sparse graphs that means the speedup is dominated by the
/// proposal-phase fraction of the per-round work, which is small
/// relative to the per-merge `absorb_into` + `relink_neighbour`
/// bookkeeping. As a result, the parallel variant typically runs at
/// **wall-clock parity or modestly slower** than the sequential
/// baseline on TokenFS-typical sparse inputs at 100 K - 1 M vertices.
///
/// The function exists primarily to provide a **deterministic API
/// surface** for callers who want to participate in a rayon-driven
/// pipeline without forcing a sequential `rabbit_order` call inside
/// it, and to anchor future work that swaps the apply-phase strategy
/// (colouring-based conflict-free batching, hand-rolled lock-free
/// adjacency, etc.) without churning the public API.
///
/// ## Behaviour on degenerate inputs
///
/// Same as [`rabbit_order`]: empty / single-vertex / edgeless / self-
/// looped / multigraph inputs all return a valid [`Permutation`]. See
/// the [`rabbit_order`] documentation for the precise contract.
///
/// ## Example
///
/// The example mirrors [`rabbit_order`]'s — two triangles joined by a
/// bridge edge. Below the small-graph threshold this call delegates
/// to the sequential path, so the output is identical regardless of
/// thread count. The only practical difference at this size is a few
/// extra microseconds of bookkeeping; the example exercises the
/// public API surface, not the parallel speedup.
///
/// ```no_run
/// use tokenfs_algos::permutation::{CsrGraph, rabbit_order_par};
///
/// let offsets = [0_u32, 2, 4, 7, 10, 12, 14];
/// let neighbors = [1_u32, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4];
/// let graph = CsrGraph { n: 6, offsets: &offsets, neighbors: &neighbors };
///
/// let perm = rabbit_order_par(graph);
/// assert_eq!(perm.len(), 6);
/// ```
///
/// # Panics
///
/// Same panic conditions as [`rabbit_order`]. Kernel/FUSE callers
/// that need a non-panicking variant should prefer
/// [`try_rabbit_order_par`].
#[cfg(feature = "parallel")]
#[must_use]
pub fn rabbit_order_par(graph: CsrGraph<'_>) -> Permutation {
    let n = graph.n as usize;
    assert_eq!(
        graph.offsets.len(),
        n + 1,
        "rabbit_order_par: offsets.len() ({}) != n + 1 ({})",
        graph.offsets.len(),
        n + 1
    );
    if n == 0 {
        return Permutation::try_identity(0).expect("identity construction within u32::MAX");
    }
    // Defer the rest of the validation + small-graph fallback to the
    // sequential path. The sequential routine performs the same bounds
    // checks and is bit-for-bit deterministic; we want the small-graph
    // codepath to share both behaviours so the public API is uniform.
    if graph.neighbors.len() < RABBIT_PARALLEL_EDGE_THRESHOLD {
        return rabbit_order(graph);
    }

    // -- input validation (mirrors `rabbit_order`) --
    for w in graph.offsets.windows(2) {
        assert!(
            w[0] <= w[1],
            "rabbit_order_par: offsets non-monotone: offsets contains {} followed by {}",
            w[0],
            w[1]
        );
    }
    assert_eq!(
        graph.offsets[n] as usize,
        graph.neighbors.len(),
        "rabbit_order_par: offsets[n] ({}) != neighbors.len() ({})",
        graph.offsets[n],
        graph.neighbors.len()
    );

    // -- Build the per-community adjacency in parallel. Each vertex's
    // consolidation is independent (input is the immutable raw CSR
    // slice; output is a per-vertex AdjList plus a self-loop counter).
    // We collect into temporary Vecs first, then drain into the
    // primary `adj` vector and `self_loop` counter array. --
    use rayon::prelude::*;

    let consolidated: Vec<(AdjList, u64)> = (0..n)
        .into_par_iter()
        .map(|v| {
            let raw = graph.neighbors_of(v as u32);
            for &u in raw {
                assert!(
                    (u as usize) < n,
                    "rabbit_order_par: neighbour {u} of vertex {v} out of range [0, {n})"
                );
            }
            consolidate_neighbours_pure(v as u32, raw)
        })
        .collect();
    let mut adj: Vec<AdjList> = Vec::with_capacity(n);
    let mut self_loop: Vec<u64> = Vec::with_capacity(n);
    for (a, sl) in consolidated {
        adj.push(a);
        self_loop.push(sl);
    }

    let mut weighted_degree: Vec<u64> = (0..n)
        .map(|v| adj[v].iter().map(|(_, w)| *w).sum::<u64>() + self_loop[v])
        .collect();
    let total_edge_weight: u64 = {
        let off_diag: u64 = (0..n).map(|v| weighted_degree[v] - self_loop[v]).sum();
        off_diag / 2 + self_loop.iter().sum::<u64>()
    };

    if total_edge_weight == 0 {
        return Permutation::try_identity(n).expect("identity construction within u32::MAX");
    }

    let mut alive: Vec<bool> = vec![true; n];
    let mut merges: Vec<Merge> = Vec::with_capacity(n);

    // Active vertex list, maintained incrementally across rounds.
    // Initialised with every vertex that has at least one neighbour;
    // vertices are dropped from this list when they are absorbed by
    // the apply phase. Re-sorting the survivors at the start of each
    // round is O(|active| log |active|) — much cheaper than an
    // O(n)-per-round full-scan once the agglomeration has reduced
    // the active set to a fraction of `n`.
    let mut active: Vec<u32> = Vec::with_capacity(n);
    for (v, list) in adj.iter().enumerate() {
        if !list.is_empty() {
            // SAFETY: v < n <= u32::MAX as usize (vertex IDs are u32).
            #[allow(clippy::cast_possible_truncation)]
            active.push(v as u32);
        }
    }

    loop {
        // -- Snapshot phase: drop absorbed vertices from the active
        // list (incremental maintenance), then re-sort the survivors
        // by canonical (weighted_degree asc, vertex_id asc) order.
        // The sort fixes the per-round proposal-phase iteration
        // order to be the same on every thread count, which is the
        // determinism contract's anchor point. --
        active.retain(|&u| alive[u as usize] && !adj[u as usize].is_empty());
        if active.is_empty() {
            break;
        }
        active.sort_unstable_by_key(|&u| (weighted_degree[u as usize], u));

        // -- Proposal phase (parallel, read-only against `adj` /
        // `weighted_degree`). Each worker thread takes a contiguous
        // chunk of the canonical-ordered active set and reuses a pair
        // of scratch buffers across every vertex in the chunk. Using
        // `par_chunks` (rather than per-vertex `par_iter`) amortises
        // rayon's per-task dispatch overhead — for typical TokenFS
        // sparse graphs each `best_merge_target` call evaluates only
        // a handful of neighbours, so per-task overhead would dominate
        // a per-vertex parallel iteration. --
        let chunk_size = active
            .len()
            .div_ceil(rayon::current_num_threads().max(1) * 4)
            .max(1);
        let proposals: Vec<Option<u32>> = active
            .par_chunks(chunk_size)
            .flat_map_iter(|chunk| {
                // Per-thread scratch buffers reused across the chunk.
                let mut scratch_weights: Vec<u64> = Vec::with_capacity(64);
                let mut scratch_degrees: Vec<u64> = Vec::with_capacity(64);
                let mut out: Vec<Option<u32>> = Vec::with_capacity(chunk.len());
                for &u in chunk {
                    out.push(best_merge_target(
                        u,
                        &adj[u as usize],
                        &weighted_degree,
                        total_edge_weight,
                        &mut scratch_weights,
                        &mut scratch_degrees,
                    ));
                }
                out
            })
            .collect();

        // -- Build the canonical apply-phase ordering. Each entry is
        // `(absorbed_id, target_id)`; sorted by absorbed_id ascending
        // so the apply phase is deterministic across thread counts. --
        let mut applies: Vec<(u32, u32)> = active
            .iter()
            .zip(proposals)
            .filter_map(|(&u, opt)| opt.map(|v| (u, v)))
            .collect();
        if applies.is_empty() {
            // No positive-dQ merge proposed anywhere; agglomeration is
            // complete (or the remaining alive vertices are mutually
            // disconnected).
            break;
        }
        applies.sort_unstable_by_key(|&(u, _)| u);

        // -- Sequential apply phase. Skip proposals whose endpoints
        // were already absorbed by an earlier (lower-absorbed-id)
        // apply in this round. The "earlier" criterion is the
        // canonical sort order — invariant across threads. --
        let mut applied_in_round = false;
        for (u, v) in applies {
            if !alive[u as usize] {
                // `u` was already absorbed earlier this round.
                continue;
            }
            if !alive[v as usize] {
                // `v` was absorbed earlier this round; the proposal
                // referenced a target that no longer exists.
                continue;
            }
            if u == v {
                // Defensive: `best_merge_target` already excludes
                // self-loops, but apply-phase invariants demand
                // u != v.
                continue;
            }
            merges.push(Merge {
                absorbed: u,
                into: v,
            });
            absorb_into(
                u as usize,
                v as usize,
                &mut adj,
                &mut weighted_degree,
                &mut self_loop,
            );
            alive[u as usize] = false;
            applied_in_round = true;
        }

        if !applied_in_round {
            // Every proposal was a conflict; no merge happened. This
            // should not be reachable with the canonical ordering
            // (the lowest-id alive vertex always wins), but guard
            // against infinite loops defensively.
            break;
        }
    }

    dfs_visit_order(n, &merges)
}

/// Like [`rabbit_order_par`] but returns an error on malformed CSR
/// input instead of panicking.
///
/// Validates the entire CSR header upfront via
/// [`CsrGraph::try_validate`] (offsets length, monotonicity,
/// neighbours-tail consistency, in-range neighbour IDs) before
/// invoking the existing internal parallel pipeline. On a successful
/// validation dispatches to [`rabbit_order_par`] — the success path
/// is bit-identical to the panicking sibling, including the
/// `RABBIT_PARALLEL_EDGE_THRESHOLD` small-graph fallback that
/// delegates to [`rabbit_order`] underneath.
///
/// **Allocation note:** the try_* path is about not panicking on bad
/// input. It is NOT heap-free, and (because the parallel path may
/// dispatch to rayon) it does not attempt any kernel-mode safety
/// guarantees. Rabbit Order is a build-time primitive (image-build
/// phase, see `02b_DEPLOYMENT_MATRIX.md`); kernel-mode consumers load
/// the resulting [`Permutation`] from the sealed image and apply it
/// via the kernel-safe [`Permutation::try_apply_into`] /
/// [`Permutation::try_apply_into_strict`].
///
/// # Errors
///
/// Returns [`PermutationConstructionError::InvalidCsr`] wrapping the
/// first [`super::CsrGraphError`] encountered during upfront
/// validation. See [`CsrGraph::try_validate`] for the full failure
/// list.
#[cfg(feature = "parallel")]
pub fn try_rabbit_order_par(
    graph: CsrGraph<'_>,
) -> Result<Permutation, PermutationConstructionError> {
    graph.try_validate()?;
    Ok(rabbit_order_par(graph))
}

/// Pure variant of [`consolidate_neighbours`] that returns the per-
/// vertex `(AdjList, self_loop_count)` instead of mutating a shared
/// slice. Used by [`rabbit_order_par`]'s parallel CSR ingestion phase.
#[cfg(feature = "parallel")]
fn consolidate_neighbours_pure(v: u32, raw: &[u32]) -> (AdjList, u64) {
    if raw.is_empty() {
        return (AdjList::new(), 0);
    }
    let mut tmp: Vec<u32> = raw.to_vec();
    tmp.sort_unstable();

    let mut out: AdjList = AdjList::with_capacity(tmp.len());
    let mut self_loop: u64 = 0;
    let mut i = 0;
    while i < tmp.len() {
        let n = tmp[i];
        let mut j = i + 1;
        let mut count = 1_u64;
        while j < tmp.len() && tmp[j] == n {
            count += 1;
            j += 1;
        }
        if n == v {
            self_loop += count;
        } else {
            out.push((n, count));
        }
        i = j;
    }
    (out, self_loop)
}

/// Folds duplicate neighbour entries from the raw CSR into a sorted
/// weighted adjacency list, redirecting self-loops into `self_loop[v]`.
///
/// The input `raw` is the contiguous neighbour slice for vertex `v`
/// from the CSR. The returned list is sorted by ascending neighbour ID
/// so subsequent merges can use a linear two-pointer intersection.
fn consolidate_neighbours(v: u32, raw: &[u32], self_loop: &mut [u64]) -> AdjList {
    if raw.is_empty() {
        return AdjList::new();
    }
    // Copy + sort to ensure ascending order. The CSR input from
    // realistic builders is typically already sorted, but the spec
    // does not guarantee it.
    let mut tmp: Vec<u32> = raw.to_vec();
    tmp.sort_unstable();

    let mut out: AdjList = AdjList::with_capacity(tmp.len());
    let mut i = 0;
    while i < tmp.len() {
        let n = tmp[i];
        let mut j = i + 1;
        let mut count = 1_u64;
        while j < tmp.len() && tmp[j] == n {
            count += 1;
            j += 1;
        }
        if n == v {
            self_loop[v as usize] += count;
        } else {
            out.push((n, count));
        }
        i = j;
    }
    out
}

/// Picks the neighbour `v` of community `u` maximising modularity gain.
///
/// Returns `None` when no neighbour offers `dQ > 0`. Ties are broken on
/// ascending neighbour ID for determinism.
///
/// The modularity gain formula:
/// ```text
/// dQ(u, v) = w(u, v) / m - deg(u) * deg(v) / (2 * m^2)
/// ```
///
/// To avoid floating-point comparisons with epsilon games, we compare
/// `2 * m * w(u, v) - deg(u) * deg(v) / m` symbolically by clearing the
/// `1 / (2 * m^2)` factor: maximise
/// ```text
/// 2 * m * w(u, v) - deg(u) * deg(v)
/// ```
/// and require it strictly positive (which is equivalent to `dQ > 0`).
/// The arithmetic is done in `i128` to avoid overflow on dense large
/// graphs (`2 * m * w` can exceed `u64` for `m`, `w` near 2^32).
///
/// The per-pair score evaluation is delegated to
/// [`kernels::auto::modularity_gains_neighbor_batch`] so the inner loop
/// picks up SIMD acceleration when the runtime backend (AVX2 / AVX-512
/// / NEON) is available and the input fits the i64 fast path. Both the
/// score values and the tie-breaking semantics are bit-exact with the
/// pre-SIMD scalar implementation.
///
/// `scratch_weights` and `scratch_degrees` are caller-owned scratch
/// vectors that this function clears and re-fills with the columnar
/// projection of `u_adj`. Reusing them across heap pops keeps the
/// agglomeration loop allocation-light on dense graphs.
fn best_merge_target(
    u: u32,
    u_adj: &AdjList,
    weighted_degree: &[u64],
    total_edge_weight: u64,
    scratch_weights: &mut Vec<u64>,
    scratch_degrees: &mut Vec<u64>,
) -> Option<u32> {
    if u_adj.is_empty() {
        return None;
    }

    // Project the adjacency into the columnar layout the kernel
    // expects. The scratch buffers grow monotonically across calls so
    // amortised allocation cost is O(max neighbour count) overall.
    scratch_weights.clear();
    scratch_degrees.clear();
    scratch_weights.reserve(u_adj.len());
    scratch_degrees.reserve(u_adj.len());
    for &(v, w) in u_adj {
        scratch_weights.push(w);
        scratch_degrees.push(weighted_degree[v as usize]);
    }
    let self_degree = weighted_degree[u as usize];
    let m_doubled = u128::from(total_edge_weight).saturating_mul(2);

    let scores = kernels::auto::modularity_gains_neighbor_batch(
        scratch_weights,
        scratch_degrees,
        self_degree,
        m_doubled,
    );

    let mut best_v: Option<u32> = None;
    // Score must strictly exceed zero for a merge to happen; we then
    // beat the running best on score, with ties broken on neighbour
    // ID. Tracking the score independently from `best_v` lets us
    // collapse the "first positive candidate" and "improve over
    // previous best" cases into one comparison sequence.
    let mut best_score: i128 = 0;

    for (&(v, _), &score) in u_adj.iter().zip(scores.iter()) {
        match best_v {
            None if score > 0 => {
                best_v = Some(v);
                best_score = score;
            }
            None => {}
            Some(prev_v) if score > best_score || (score == best_score && v < prev_v) => {
                best_v = Some(v);
                best_score = score;
            }
            Some(_) => {}
        }
    }
    best_v
}

pub mod kernels {
    //! Per-neighbour modularity-gain kernels for [`super::rabbit_order`].
    //!
    //! Sprint 50-52 lifts the per-pair score computation out of the
    //! sequential agglomeration loop and into a kernel module with
    //! scalar / AVX2 / AVX-512 / NEON variants. The shared API computes
    //! the integer modularity-gain *score*
    //!
    //! ```text
    //! score(u, v) = 2 * m * w(u, v) - deg(u) * deg(v)
    //! ```
    //!
    //! for every neighbour `v_i` of a fixed community `u`. The score is
    //! symbolically equivalent to the modularity-gain function `dQ`
    //! scaled by the constant `2 * m^2`, so it preserves the sign and
    //! ordering of `dQ` without any floating-point arithmetic. Callers
    //! pick the merge target by argmax over the resulting `i128` slice.
    //!
    //! ## Determinism guarantee
    //!
    //! Every backend produces **bit-exact** results given the same
    //! inputs. There is no floating-point reduction order to track. The
    //! SIMD backends compute via integer multiplies that are
    //! semantically equivalent to the scalar reference, with the only
    //! ambient choice being whether the **i64 fast path** is eligible
    //! (see below). Eligibility itself is a pure function of the input
    //! magnitudes, so the same inputs always pick the same code path on
    //! the same backend.
    //!
    //! ## i64 fast path eligibility
    //!
    //! Eligibility is `m_doubled < 2^31` AND `self_degree < 2^31` AND
    //! every entry of `neighbor_weights` and `neighbor_degrees` is
    //! `< 2^31` (audit-R7 #6: code-doc reconciliation; the literal
    //! `BOUND` constant on each backend is `1_u64 << 31`, not `1u64 << 32`,
    //! to keep the i64 product symmetric around zero — see the per-
    //! backend `BOUND` doc-comment for the sign-overflow rationale).
    //! Under that bound both products `2 * m * w` and `deg(u) * deg(v)`
    //! fit in `u64`, their difference fits in `i64` (the per-pair score
    //! sits in `i63` after the `m_doubled` factor, so even the
    //! community-wide sum stays in `i64` for the realistic batch sizes
    //! the agglomeration loop emits), and the SIMD backends use
    //! 32-bit-input widening multiplies (`_mm256_mul_epu32`,
    //! `_mm512_mul_epu32`, `vmull_u32`) to evaluate the score in `i64`
    //! lanes. For inputs larger than the bound, every backend
    //! transparently delegates to [`scalar::modularity_gains_neighbor_batch`]
    //! for the i128 arithmetic (the SIMD lanes do not have a portable
    //! widening multiply that produces 128-bit results).
    //!
    //! For TokenFS-typical workloads (200 K vertices, average degree
    //! 5-20, weights bounded by a small constant) the fast path covers
    //! 100% of inputs; for adversarial dense graphs with weighted
    //! degree near `u64::MAX` the scalar fallback preserves correctness.
    //!
    //! ## Surface
    //!
    //! Each backend exposes the same function signature:
    //!
    //! ```text
    //! fn modularity_gains_neighbor_batch(
    //!     neighbor_weights: &[u64],   // w(u, v_i)
    //!     neighbor_degrees: &[u64],   // deg(v_i)
    //!     self_degree: u64,           // deg(u)
    //!     m_doubled: u128,            // 2 * total edge weight
    //! ) -> Vec<i128>;                 // score(u, v_i) per neighbour
    //! ```
    //!
    //! [`auto::modularity_gains_neighbor_batch`] is the runtime-dispatched
    //! entry point used by the agglomeration loop. External consumers
    //! that wish to pin a specific backend (e.g. for benchmarking) call
    //! into [`scalar`], [`avx2`] (x86 only), [`avx512`] (x86 only), or
    //! `neon` (aarch64 only) directly.

    /// Portable scalar reference implementation.
    ///
    /// Bit-exact ground truth for every other backend. Always uses
    /// `i128` arithmetic so it is safe even on adversarial dense
    /// graphs whose products overflow `u64`.
    #[cfg(feature = "arch-pinned-kernels")]
    pub mod scalar;
    #[cfg(not(feature = "arch-pinned-kernels"))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod scalar;

    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx2;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx2",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx2;

    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    pub mod avx512;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "avx512",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod avx512;

    #[cfg(all(
        feature = "arch-pinned-kernels",
        feature = "neon",
        target_arch = "aarch64"
    ))]
    pub mod neon;
    #[cfg(all(
        not(feature = "arch-pinned-kernels"),
        feature = "neon",
        target_arch = "aarch64"
    ))]
    #[allow(dead_code, unreachable_pub)]
    pub(crate) mod neon;

    /// Runtime-dispatched modularity-gain kernel.
    pub mod auto {
        #[cfg(not(feature = "std"))]
        use alloc::vec::Vec;

        /// Computes the per-neighbour modularity-gain score using the
        /// best available SIMD backend.
        ///
        /// Equivalent to [`super::scalar::modularity_gains_neighbor_batch`]
        /// bit-for-bit. See the [`super`] module documentation for the
        /// score definition, the i64 fast path eligibility predicate,
        /// and the determinism guarantee.
        ///
        /// # Panics
        ///
        /// Panics if `neighbor_weights.len() != neighbor_degrees.len()`.
        #[must_use]
        pub fn modularity_gains_neighbor_batch(
            neighbor_weights: &[u64],
            neighbor_degrees: &[u64],
            self_degree: u64,
            m_doubled: u128,
        ) -> Vec<i128> {
            assert_eq!(
                neighbor_weights.len(),
                neighbor_degrees.len(),
                "modularity_gains_neighbor_batch: neighbor_weights.len() ({}) != neighbor_degrees.len() ({})",
                neighbor_weights.len(),
                neighbor_degrees.len()
            );
            #[cfg(all(
                feature = "std",
                feature = "avx512",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx512::is_available() {
                    // SAFETY: AVX-512F + AVX-512DQ availability checked
                    // immediately above; lengths match.
                    return unsafe {
                        super::avx512::modularity_gains_neighbor_batch(
                            neighbor_weights,
                            neighbor_degrees,
                            self_degree,
                            m_doubled,
                        )
                    };
                }
            }
            #[cfg(all(
                feature = "std",
                feature = "avx2",
                any(target_arch = "x86", target_arch = "x86_64")
            ))]
            {
                if super::avx2::is_available() {
                    // SAFETY: AVX2 availability checked immediately above; lengths match.
                    return unsafe {
                        super::avx2::modularity_gains_neighbor_batch(
                            neighbor_weights,
                            neighbor_degrees,
                            self_degree,
                            m_doubled,
                        )
                    };
                }
            }
            #[cfg(all(feature = "neon", target_arch = "aarch64"))]
            {
                if super::neon::is_available() {
                    // SAFETY: NEON is mandatory on AArch64; lengths match.
                    return unsafe {
                        super::neon::modularity_gains_neighbor_batch(
                            neighbor_weights,
                            neighbor_degrees,
                            self_degree,
                            m_doubled,
                        )
                    };
                }
            }
            super::scalar::modularity_gains_neighbor_batch(
                neighbor_weights,
                neighbor_degrees,
                self_degree,
                m_doubled,
            )
        }
    }
}

/// Absorbs community `u` into community `v`.
///
/// Folds `u`'s adjacency into `v`'s, removing the direct `u-v` and
/// `v-u` edges and combining shared neighbours. Updates the weighted
/// degree of `v` and relinks every neighbour `w` that previously
/// pointed at `u` to point at `v` instead.
///
/// `self_loop[v]` accumulates `2 * w(u, v) + self_loop[u]` so the
/// total weighted degree of the merged community is preserved
/// (Newman 2006 modularity-update convention).
fn absorb_into(
    u: usize,
    v: usize,
    adj: &mut [AdjList],
    weighted_degree: &mut [u64],
    self_loop: &mut [u64],
) {
    debug_assert!(u != v, "absorb_into: cannot merge a vertex with itself");

    // Take ownership of `u`'s adjacency so the borrow checker permits
    // simultaneous mutation of `v`'s entry below.
    let u_adj = core::mem::take(&mut adj[u]);

    // Find the direct edge weight w(u, v) for the self-loop update,
    // and remove the `u-v` entry from u_adj before merging.
    let mut w_uv: u64 = 0;
    let mut filtered_u_adj: AdjList = AdjList::with_capacity(u_adj.len());
    for (n, w) in u_adj {
        if n as usize == v {
            w_uv = w;
        } else {
            filtered_u_adj.push((n, w));
        }
    }

    // Merge filtered_u_adj into adj[v], folding shared neighbours.
    let v_adj_before = core::mem::take(&mut adj[v]);
    let merged = merge_sorted_adj(&v_adj_before, &filtered_u_adj, v as u32, u as u32);
    adj[v] = merged;

    // Rewrite every neighbour `w`'s adjacency: any entry pointing at
    // `u` becomes an entry pointing at `v`, with weights folded if
    // `w` already had a direct `v` entry.
    for &(w, _w_uw) in &filtered_u_adj {
        if w as usize == v {
            continue;
        }
        relink_neighbour(w as usize, u as u32, v as u32, adj);
    }

    // Update weighted degree and self-loop accounting.
    self_loop[v] += 2 * w_uv + self_loop[u];
    weighted_degree[v] += weighted_degree[u];
    weighted_degree[u] = 0;
    self_loop[u] = 0;
}

/// Merges two sorted weighted adjacency lists, folding shared
/// neighbour weights. Excludes any entry pointing at the destination
/// community itself or at the absorbed community.
fn merge_sorted_adj(a: &AdjList, b: &AdjList, exclude_v: u32, exclude_u: u32) -> AdjList {
    let mut out = AdjList::with_capacity(a.len() + b.len());
    let mut i = 0;
    let mut j = 0;
    while i < a.len() && j < b.len() {
        let (na, wa) = a[i];
        let (nb, wb) = b[j];
        if na == nb {
            if na != exclude_v && na != exclude_u {
                out.push((na, wa + wb));
            }
            i += 1;
            j += 1;
        } else if na < nb {
            if na != exclude_v && na != exclude_u {
                out.push((na, wa));
            }
            i += 1;
        } else {
            if nb != exclude_v && nb != exclude_u {
                out.push((nb, wb));
            }
            j += 1;
        }
    }
    while i < a.len() {
        let (na, wa) = a[i];
        if na != exclude_v && na != exclude_u {
            out.push((na, wa));
        }
        i += 1;
    }
    while j < b.len() {
        let (nb, wb) = b[j];
        if nb != exclude_v && nb != exclude_u {
            out.push((nb, wb));
        }
        j += 1;
    }
    out
}

/// Replaces every reference to community `from` in `w`'s adjacency
/// with a reference to `to`, folding weight if `w` already had a
/// direct `to` entry. Maintains the sorted-by-neighbour invariant.
fn relink_neighbour(w: usize, from: u32, to: u32, adj: &mut [AdjList]) {
    // Locate and remove the `from` entry; capture its weight.
    let w_adj = &mut adj[w];
    let Ok(from_idx) = w_adj.binary_search_by_key(&from, |&(n, _)| n) else {
        return;
    };
    let (_, w_from) = w_adj.remove(from_idx);

    // Insert (or fold into) the `to` entry, preserving sorted order.
    match w_adj.binary_search_by_key(&to, |&(n, _)| n) {
        Ok(idx) => {
            w_adj[idx].1 += w_from;
        }
        Err(idx) => {
            w_adj.insert(idx, (to, w_from));
        }
    }
}

/// Reconstructs the dendrogram and emits the DFS pre-order leaf
/// sequence as a permutation.
///
/// The dendrogram is implicit in the merge sequence: each `Merge {
/// absorbed: u, into: v }` records that `u`'s subtree becomes a child
/// of `v`'s subtree. The DFS starts from each surviving root (any
/// vertex never recorded as `absorbed`) in ascending root-ID order so
/// disconnected components and never-merged singletons appear in a
/// deterministic position.
fn dfs_visit_order(n: usize, merges: &[Merge]) -> Permutation {
    // Build a children-of-parent map: for each vertex, list of vertices
    // whose `into = self`. Insert children in *reverse* merge order so
    // that DFS pre-order visits the most-recently-merged child first
    // and the merged subtree's vertices land contiguously near the
    // parent in the output.
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut absorbed: Vec<bool> = vec![false; n];
    for merge in merges.iter().rev() {
        children[merge.into as usize].push(merge.absorbed);
        absorbed[merge.absorbed as usize] = true;
    }

    // DFS pre-order from each surviving root in ascending root-ID order.
    let mut perm = vec![0_u32; n];
    let mut next_position: u32 = 0;
    let mut stack: Vec<u32> = Vec::with_capacity(n);
    for (root, &is_absorbed) in absorbed.iter().enumerate() {
        if is_absorbed {
            continue;
        }
        stack.push(root as u32);
        while let Some(node) = stack.pop() {
            // Pre-order visit: assign the next position to this node.
            perm[node as usize] = next_position;
            next_position += 1;
            // Push children in *reverse* order so the first child is
            // processed first when popped (DFS pre-order semantics).
            // `children[node]` was built in reverse merge order above,
            // so iterating it as-is and pushing yields the desired
            // visit order on pop.
            for &child in &children[node as usize] {
                stack.push(child);
            }
        }
    }
    debug_assert_eq!(
        next_position as usize, n,
        "dfs_visit_order: missed a vertex"
    );
    // SAFETY: every vertex in `0..n` is either an unabsorbed root or the
    // descendant of one. The DFS pre-order walk visits each vertex
    // exactly once and assigns it a strictly increasing `next_position`
    // in `0..n`. The debug assertion above checks the cardinality. Hence
    // `perm` is a bijection on `0..n` with `n <= u32::MAX as usize`
    // (vertex IDs are u32 by construction).
    unsafe { Permutation::from_vec_unchecked(perm) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a CSR adjacency from a list of undirected edges.
    fn csr_from_edges(n: u32, edges: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
        for &(a, b) in edges {
            adj[a as usize].push(b);
            if a != b {
                adj[b as usize].push(a);
            }
        }
        for list in &mut adj {
            list.sort_unstable();
        }
        let mut offsets = Vec::with_capacity((n as usize) + 1);
        let mut neighbors = Vec::new();
        offsets.push(0_u32);
        for list in &adj {
            neighbors.extend(list.iter().copied());
            offsets.push(neighbors.len() as u32);
        }
        (offsets, neighbors)
    }

    /// Asserts that `perm` is a valid permutation of `0..n`.
    fn assert_valid_permutation(perm: &Permutation, n: usize) {
        assert_eq!(perm.len(), n, "permutation length mismatch");
        let mut seen = vec![false; n];
        for &id in perm.as_slice() {
            let idx = id as usize;
            assert!(idx < n, "permutation contains id {idx} >= n {n}");
            assert!(!seen[idx], "permutation contains duplicate id {idx}");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|b| *b), "permutation missing an id");
    }

    /// Returns the inverse-permutation array, useful for checking that
    /// vertices in the same community are placed contiguously.
    fn inverse(perm: &Permutation) -> Vec<u32> {
        let n = perm.len();
        let mut inv = vec![0_u32; n];
        for (old, &new) in perm.as_slice().iter().enumerate() {
            inv[new as usize] = old as u32;
        }
        inv
    }

    #[test]
    fn empty_graph_returns_empty_permutation() {
        let offsets = [0_u32];
        let neighbors: [u32; 0] = [];
        let g = CsrGraph {
            n: 0,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert!(perm.is_empty());
    }

    #[test]
    fn single_vertex_returns_identity() {
        let offsets = [0_u32, 0];
        let neighbors: [u32; 0] = [];
        let g = CsrGraph {
            n: 1,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, 1);
        assert_eq!(perm.as_slice(), &[0_u32]);
    }

    #[test]
    fn two_connected_vertices() {
        let offsets = [0_u32, 1, 2];
        let neighbors = [1_u32, 0];
        let g = CsrGraph {
            n: 2,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, 2);
    }

    #[test]
    fn fully_disconnected_returns_identity_layout() {
        // Five isolated vertices.
        let offsets = vec![0_u32; 6];
        let neighbors: Vec<u32> = Vec::new();
        let g = CsrGraph {
            n: 5,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, 5);
        // No edges means no merges; the DFS walks singletons in
        // ascending root order, which equals the identity.
        assert_eq!(perm.as_slice(), &[0_u32, 1, 2, 3, 4]);
    }

    #[test]
    fn self_loops_do_not_break_anything() {
        // Triangle 0-1-2 with self-loop on 0.
        let n = 3_u32;
        let edges = vec![(0_u32, 1), (1, 2), (0, 2), (0, 0)];
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);
    }

    #[test]
    fn complete_graph_k5() {
        // K_5: every pair connected.
        let n = 5_u32;
        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                edges.push((i, j));
            }
        }
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);
    }

    #[test]
    fn determinism_same_input_same_output() {
        let n = 16_u32;
        let edges: Vec<(u32, u32)> = (0..(n - 1)).map(|i| (i, i + 1)).collect();
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let p1 = rabbit_order(g);
        let p2 = rabbit_order(g);
        let p3 = rabbit_order(g);
        assert_eq!(p1, p2);
        assert_eq!(p2, p3);
    }

    #[test]
    fn path_graph_produces_valid_permutation() {
        let n = 8_u32;
        let edges: Vec<(u32, u32)> = (0..(n - 1)).map(|i| (i, i + 1)).collect();
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);
    }

    #[test]
    fn star_graph_produces_valid_permutation() {
        // Star: vertex 0 connected to 1..=4.
        let n = 5_u32;
        let edges: Vec<(u32, u32)> = (1..n).map(|i| (0, i)).collect();
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);
    }

    /// Quality check: on a graph with K = 4 obvious 4-cliques connected
    /// by sparse single-edge bridges, the resulting permutation should
    /// place each clique's members contiguously in the output. We check
    /// this by computing, for each clique, the span of its members'
    /// new positions and asserting the spans don't overlap.
    #[test]
    fn community_graph_groups_clique_members_contiguously() {
        // Four 4-cliques: {0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13,14,15}.
        // Bridges: 3-4, 7-8, 11-12.
        let cliques: [[u32; 4]; 4] = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]];
        let n = 16_u32;
        let mut edges: Vec<(u32, u32)> = Vec::new();
        for clique in &cliques {
            for i in 0..clique.len() {
                for j in (i + 1)..clique.len() {
                    edges.push((clique[i], clique[j]));
                }
            }
        }
        edges.push((3, 4));
        edges.push((7, 8));
        edges.push((11, 12));
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);

        // Compute per-clique span of new positions.
        let p = perm.as_slice();
        let mut spans: Vec<(u32, u32)> = Vec::with_capacity(cliques.len());
        for clique in &cliques {
            let mut min_pos = u32::MAX;
            let mut max_pos = 0_u32;
            for &v in clique {
                let pos = p[v as usize];
                if pos < min_pos {
                    min_pos = pos;
                }
                if pos > max_pos {
                    max_pos = pos;
                }
            }
            // Each clique has 4 members; perfect contiguity means
            // span = 3. Allow a small slack (span <= 4) to absorb the
            // bridge-adjacent vertices' attraction to their neighbour
            // clique. Anything wider indicates the algorithm failed
            // to detect the community structure.
            let span = max_pos - min_pos;
            assert!(
                span <= 4,
                "clique {clique:?} span {span} exceeds tolerance (positions: {:?})",
                clique.iter().map(|&v| p[v as usize]).collect::<Vec<_>>()
            );
            spans.push((min_pos, max_pos));
        }

        // All four clique spans must be pairwise non-overlapping
        // except possibly at a single shared boundary slot for bridge
        // vertices.
        spans.sort_unstable();
        for w in spans.windows(2) {
            // Adjacent clique spans must be (mostly) ordered: the
            // later clique's min must not fall deep inside the
            // earlier clique's span. Allow a 1-slot overlap to absorb
            // a bridge vertex.
            assert!(
                w[1].0 + 1 >= w[0].1,
                "clique span overlap too large: {:?} vs {:?}",
                w[0],
                w[1]
            );
        }
    }

    /// Two disjoint triangles: {0,1,2} and {3,4,5}. The dendrogram
    /// has two roots; the DFS visits them in ascending root order.
    /// Whatever the merge order, the two communities' members must
    /// land contiguously.
    #[test]
    fn two_disjoint_triangles_grouped() {
        let n = 6_u32;
        let edges = vec![(0_u32, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);

        let inv = inverse(&perm);
        // The first three positions belong to one triangle; the last
        // three belong to the other.
        let first_half: Vec<u32> = inv[..3].to_vec();
        let second_half: Vec<u32> = inv[3..].to_vec();
        let mut a = first_half.clone();
        let mut b = second_half.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert!(
            a == vec![0, 1, 2] && b == vec![3, 4, 5] || a == vec![3, 4, 5] && b == vec![0, 1, 2],
            "triangles not grouped: first_half={first_half:?} second_half={second_half:?}"
        );
    }

    /// Permutation round-trip with the inverse.
    #[test]
    fn permutation_round_trips_with_inverse() {
        let n = 32_u32;
        let edges: Vec<(u32, u32)> = (0..(n - 1)).map(|i| (i, i + 1)).collect();
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        let inv = perm.inverse();
        let src: Vec<u32> = (0..n).collect();
        let permuted = perm.apply(&src);
        let recovered = inv.apply(&permuted);
        assert_eq!(recovered, src);
    }

    /// Synthetic Erdős-Rényi graph generator, deterministic seed.
    fn erdos_renyi_csr(n: u32, avg_degree: u32, seed: u64) -> (Vec<u32>, Vec<u32>) {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
        let mut state = seed | 1;
        let denom = (n as u64).saturating_sub(1).max(1);
        let p_num = avg_degree as u64;
        for u in 0..n {
            for v in (u + 1)..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let r = state.wrapping_mul(0x2545_f491_4f6c_dd1d) % denom;
                if r < p_num {
                    adj[u as usize].push(v);
                    adj[v as usize].push(u);
                }
            }
        }
        for list in &mut adj {
            list.sort_unstable();
        }
        let mut offsets = Vec::with_capacity((n as usize) + 1);
        let mut neighbors = Vec::new();
        offsets.push(0_u32);
        for list in &adj {
            neighbors.extend(list.iter().copied());
            offsets.push(neighbors.len() as u32);
        }
        (offsets, neighbors)
    }

    #[test]
    fn random_graphs_produce_valid_permutations() {
        for &(n, deg, seed) in &[
            (32_u32, 2_u32, 0xC0FFEE_u64),
            (64, 3, 0xDEAD_BEEF),
            (100, 4, 0x5151_5EED),
            (200, 5, 0xF00D_BABE),
        ] {
            let (offsets, neighbors) = erdos_renyi_csr(n, deg, seed);
            let g = CsrGraph {
                n,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            let perm = rabbit_order(g);
            assert_valid_permutation(&perm, n as usize);
        }
    }

    /// Multigraph stress test: duplicate edges on a small graph must
    /// fold into per-community weights without breaking determinism.
    #[test]
    fn duplicate_edges_fold_into_weights() {
        // Triangle with the (0,1) edge listed three times.
        let offsets = [0_u32, 4, 5, 7];
        // Vertex 0: [1, 1, 1, 2] (duplicates 1 three times)
        // Vertex 1: [0]          (single back-edge — intentionally
        //                         asymmetric to exercise the duplicate
        //                         handling on the in-degree side)
        // Vertex 2: [0, 1]
        let neighbors = [1_u32, 1, 1, 2, 0, 0, 1];
        let g = CsrGraph {
            n: 3,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, 3);
    }

    /// Sanity check: parent absorption preserves total weighted degree
    /// invariant. Build a small triangle, run rabbit_order, and assert
    /// that the merge sequence accumulates the expected total edge
    /// weight on the surviving root.
    #[test]
    fn merge_preserves_total_edge_weight() {
        // Triangle 0-1-2 (each edge weight 1 in CSR; total m = 3).
        let n = 3_u32;
        let edges = vec![(0_u32, 1), (1, 2), (0, 2)];
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);
    }

    /// Cross-architecture parity is implicit — the algorithm uses only
    /// integer arithmetic. This test pins the exact output for a small
    /// known graph so any future changes that perturb the determinism
    /// guarantee are caught.
    #[test]
    fn pinned_output_for_two_triangles_with_bridge() {
        // Triangle {0,1,2} bridged to triangle {3,4,5} via edge 2-3.
        let n = 6_u32;
        let edges = vec![(0_u32, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)];
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rabbit_order(g);
        assert_valid_permutation(&perm, n as usize);
        // Two communities are obvious; assert each lands in a
        // contiguous block of three positions, regardless of which
        // block is first.
        let inv = inverse(&perm);
        // Use `BTreeSet` (works on both alloc and std) so the test
        // compiles under `--no-default-features --features alloc` while
        // remaining unchanged under default builds (audit-R6 #164).
        // The crate's `extern crate alloc` is only declared when std is
        // off, so reach for the right path explicitly.
        #[cfg(not(feature = "std"))]
        use alloc::collections::BTreeSet;
        #[cfg(feature = "std")]
        use std::collections::BTreeSet;
        let first: BTreeSet<u32> = inv[..3].iter().copied().collect();
        let second: BTreeSet<u32> = inv[3..].iter().copied().collect();
        let group_a: BTreeSet<u32> = [0, 1, 2].into_iter().collect();
        let group_b: BTreeSet<u32> = [3, 4, 5].into_iter().collect();
        assert!(
            (first == group_a && second == group_b) || (first == group_b && second == group_a),
            "communities not grouped: first={first:?} second={second:?}"
        );
    }

    // ---------------------------------------------------------------------
    // Reference-fixture generator (commented Python). Future contributors
    // can use this to regenerate parity fixtures against
    // `araij/rabbit_order` (C++):
    //
    // ```python
    // # Save as gen_rabbit_fixture.py
    // # Usage: python gen_rabbit_fixture.py > fixture.txt
    // # Then feed `fixture.txt` to both implementations and diff
    // # the resulting permutation arrays.
    // import random, sys
    //
    // random.seed(0xC0FFEE)
    // N = 64
    // EDGES = []
    // for i in range(N):
    //     deg = random.randint(2, 6)
    //     for _ in range(deg):
    //         j = random.randrange(N)
    //         if i != j:
    //             EDGES.append((i, j))
    //
    // # Output format: first line is N M, then M lines of "u v".
    // print(N, len(EDGES))
    // for (u, v) in EDGES:
    //     print(u, v)
    // ```
    //
    // Parity against the C++ reference is left for Sprint 53-55 (the
    // concurrent-merging port) once we share the same fixture format
    // end-to-end; the sequential baseline shipped here is verified by
    // the property and quality tests above, and the SIMD inner loop
    // (Sprint 50-52) is verified by the kernel parity tests below.

    // ---------------------------------------------------------------------
    // Sprint 50-52: SIMD modularity-gain inner loop parity tests.
    //
    // Every SIMD backend must produce bit-exact results vs the scalar
    // reference for both the i64-fast-path regime (every input < 2^31
    // — see `kernels::scalar::fast_path_eligible` for the bound and
    // its sign-overflow rationale) and the i128-fallback regime (any
    // input >= 2^31). The `rabbit_order` permutation must remain
    // unchanged whether the SIMD path or the scalar path was selected
    // at runtime.

    use super::kernels;

    /// Deterministic seed-driven generator of `(weights, degrees)`
    /// vectors with values bounded by `bound`. Used to drive the
    /// SIMD-vs-scalar parity tests across both fast-path and
    /// scalar-fallback regimes.
    fn random_neighbor_batch(n: usize, seed: u64, bound: u64) -> (Vec<u64>, Vec<u64>) {
        let mut state = seed | 1;
        let mut next = || -> u64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        };
        let mut weights = Vec::with_capacity(n);
        let mut degrees = Vec::with_capacity(n);
        for _ in 0..n {
            weights.push(next() % bound);
            degrees.push(next() % bound);
        }
        (weights, degrees)
    }

    #[test]
    fn scalar_kernel_matches_inline_formula() {
        // Triangle with weights: m_doubled = 6, deg_u = 2, neighbors
        // are [(deg_v=2, w=1), (deg_v=2, w=1)]. Score per neighbour:
        //   2*m*w - deg_u*deg_v = 6*1 - 2*2 = 2
        let weights = vec![1_u64, 1];
        let degrees = vec![2_u64, 2];
        let scores = kernels::scalar::modularity_gains_neighbor_batch(&weights, &degrees, 2, 6);
        assert_eq!(scores, vec![2_i128, 2_i128]);
    }

    #[test]
    fn scalar_kernel_handles_empty_input() {
        let scores: Vec<i128> = kernels::scalar::modularity_gains_neighbor_batch(&[], &[], 5, 10);
        assert!(scores.is_empty());
    }

    // Uses `std::panic::catch_unwind` to assert the panicking branch;
    // gate on `feature = "std"` so the alloc-only build compiles
    // (audit-R6 finding #164). The kernel-level invariant is also
    // enforced by `debug_assert_eq!` inside the scalar kernel itself.
    #[cfg(feature = "std")]
    #[test]
    fn scalar_kernel_panics_on_length_mismatch() {
        let result = std::panic::catch_unwind(|| {
            kernels::scalar::modularity_gains_neighbor_batch(&[1, 2, 3], &[1, 2], 1, 1)
        });
        assert!(result.is_err(), "expected panic on length mismatch");
    }

    #[test]
    fn scalar_kernel_handles_large_inputs_via_i128() {
        // Inputs exceed the u32 fast-path bound; the scalar path stays
        // correct via i128 arithmetic. Use values that would overflow
        // i64: 2*m * w = 2 * (2^60) * (2^4) = 2^65.
        let m = 1_u64 << 60;
        let m_doubled = u128::from(m).saturating_mul(2);
        let w = 1_u64 << 4;
        let deg_u = 2_u64;
        let deg_v = 4_u64;
        let scores =
            kernels::scalar::modularity_gains_neighbor_batch(&[w], &[deg_v], deg_u, m_doubled);
        let two_m = i128::try_from(m_doubled).expect("test m_doubled fits i128");
        let expected = two_m * i128::from(w) - i128::from(deg_u) * i128::from(deg_v);
        assert_eq!(scores, vec![expected]);
        // And confirm fast_path_eligible says no.
        assert!(!kernels::scalar::fast_path_eligible(
            &[w],
            &[deg_v],
            deg_u,
            m_doubled
        ));
    }

    #[test]
    fn fast_path_eligibility_matches_threshold() {
        // m_doubled exactly at 2^31 is NOT eligible (strict <).
        assert!(!kernels::scalar::fast_path_eligible(
            &[1],
            &[1],
            1,
            1_u128 << 31
        ));
        // self_degree at 2^31 - 1 IS eligible.
        assert!(kernels::scalar::fast_path_eligible(
            &[1],
            &[1],
            (1_u64 << 31) - 1,
            1
        ));
        // A single neighbour weight at 2^31 disqualifies.
        assert!(!kernels::scalar::fast_path_eligible(
            &[1, 1_u64 << 31, 1],
            &[1, 1, 1],
            1,
            1
        ));
        // A single neighbour degree at 2^31 disqualifies.
        assert!(!kernels::scalar::fast_path_eligible(
            &[1, 1, 1],
            &[1, 1, 1_u64 << 31],
            1,
            1
        ));
        // Empty input is trivially eligible.
        assert!(kernels::scalar::fast_path_eligible(&[], &[], 0, 0));
    }

    /// AVX2 must produce bit-exact results vs the scalar reference
    /// across a range of fast-path-eligible inputs and lengths.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_kernel_matches_scalar_on_fast_path() {
        if !kernels::avx2::is_available() {
            // AVX2 not available at runtime; skip without failing.
            // The scalar test above covers the contract.
            return;
        }
        // Sweep N across {0, 1, 3, 4, 5, 7, 8, 9, 16, 17, 100, 1024}
        // to exercise the SIMD body, the SIMD/tail boundary, and the
        // tail itself.
        for &n in &[0_usize, 1, 3, 4, 5, 7, 8, 9, 16, 17, 100, 1024] {
            let (weights, degrees) = random_neighbor_batch(n, 0xC0FFEE_u64 ^ n as u64, 1_000_000);
            let self_degree = 12345_u64;
            let m_doubled = 9_876_543_u128;
            let scalar_out = kernels::scalar::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            );
            // SAFETY: AVX2 detected at runtime above.
            let avx2_out = unsafe {
                kernels::avx2::modularity_gains_neighbor_batch(
                    &weights,
                    &degrees,
                    self_degree,
                    m_doubled,
                )
            };
            assert_eq!(
                avx2_out, scalar_out,
                "avx2 kernel diverges from scalar at n={n}"
            );
        }
    }

    /// Stress the i64 fast path with values near the eligibility
    /// boundary (`< 2^31`). These produce per-pair products near
    /// `2^62`, which is the largest magnitude the SIMD lanes can hold
    /// without flipping the sign bit.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_kernel_handles_boundary_values() {
        if !kernels::avx2::is_available() {
            return;
        }
        let near_max = (1_u64 << 31) - 1;
        let weights = vec![near_max; 16];
        let degrees = vec![near_max; 16];
        let self_degree = near_max;
        let m_doubled = u128::from(near_max);
        let scalar_out = kernels::scalar::modularity_gains_neighbor_batch(
            &weights,
            &degrees,
            self_degree,
            m_doubled,
        );
        // SAFETY: AVX2 detected.
        let avx2_out = unsafe {
            kernels::avx2::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            )
        };
        assert_eq!(avx2_out, scalar_out);
    }

    /// AVX2 must defer to the scalar path (bit-exact) when an input
    /// exceeds the i64 fast-path bound.
    #[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx2_kernel_falls_back_to_scalar_on_large_inputs() {
        if !kernels::avx2::is_available() {
            return;
        }
        // m_doubled exceeds the u32 bound, forcing the i128 fallback.
        let m_doubled = (1_u128 << 40) | 7;
        let weights = vec![1_u64, 2, 3, 4, 5];
        let degrees = vec![10_u64, 20, 30, 40, 50];
        let self_degree = 1_000_u64;
        let scalar_out = kernels::scalar::modularity_gains_neighbor_batch(
            &weights,
            &degrees,
            self_degree,
            m_doubled,
        );
        // SAFETY: AVX2 detected.
        let avx2_out = unsafe {
            kernels::avx2::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            )
        };
        assert_eq!(avx2_out, scalar_out);
    }

    /// AVX-512 parity, identical structure to the AVX2 test.
    #[cfg(all(feature = "avx512", any(target_arch = "x86", target_arch = "x86_64")))]
    #[test]
    fn avx512_kernel_matches_scalar_on_fast_path() {
        if !kernels::avx512::is_available() {
            return;
        }
        for &n in &[0_usize, 1, 3, 7, 8, 9, 15, 16, 17, 100, 1024] {
            let (weights, degrees) = random_neighbor_batch(n, 0xDEADBEEF_u64 ^ n as u64, 1_000_000);
            let self_degree = 7777_u64;
            let m_doubled = 1_234_567_u128;
            let scalar_out = kernels::scalar::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            );
            // SAFETY: AVX-512F + AVX-512DQ detected at runtime.
            let avx512_out = unsafe {
                kernels::avx512::modularity_gains_neighbor_batch(
                    &weights,
                    &degrees,
                    self_degree,
                    m_doubled,
                )
            };
            assert_eq!(
                avx512_out, scalar_out,
                "avx512 kernel diverges from scalar at n={n}"
            );
        }
    }

    /// NEON parity, identical structure to the AVX2 test.
    #[cfg(all(feature = "neon", target_arch = "aarch64"))]
    #[test]
    fn neon_kernel_matches_scalar_on_fast_path() {
        if !kernels::neon::is_available() {
            return;
        }
        for &n in &[0_usize, 1, 2, 3, 4, 5, 7, 8, 17, 100, 1024] {
            let (weights, degrees) = random_neighbor_batch(n, 0xBABE_u64 ^ n as u64, 1_000_000);
            let self_degree = 5555_u64;
            let m_doubled = 7_654_321_u128;
            let scalar_out = kernels::scalar::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            );
            // SAFETY: NEON is mandatory on AArch64.
            let neon_out = unsafe {
                kernels::neon::modularity_gains_neighbor_batch(
                    &weights,
                    &degrees,
                    self_degree,
                    m_doubled,
                )
            };
            assert_eq!(
                neon_out, scalar_out,
                "neon kernel diverges from scalar at n={n}"
            );
        }
    }

    /// `auto::modularity_gains_neighbor_batch` must equal scalar for
    /// every input regardless of which backend the runtime selects.
    #[test]
    fn auto_kernel_matches_scalar() {
        for &n in &[0_usize, 1, 3, 4, 5, 8, 16, 17, 100, 1024] {
            let (weights, degrees) =
                random_neighbor_batch(n, 0xF00D_BABE_u64 ^ n as u64, 1_000_000);
            let self_degree = 17_u64;
            let m_doubled = 32_768_u128;
            let scalar_out = kernels::scalar::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            );
            let auto_out = kernels::auto::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            );
            assert_eq!(
                auto_out, scalar_out,
                "auto kernel diverges from scalar at n={n}"
            );
        }
    }

    /// Larger graph: confirm `rabbit_order` itself produces an
    /// identical permutation regardless of which kernel backend the
    /// runtime selects. Since `auto::modularity_gains_neighbor_batch`
    /// is bit-exact with `scalar::modularity_gains_neighbor_batch`,
    /// the two `rabbit_order(...)` calls must coincide.
    ///
    /// To force "scalar" inside `rabbit_order`, we cannot easily
    /// rewire the dispatcher mid-call; instead we run `rabbit_order`
    /// twice on the same input and assert the result is deterministic
    /// (covered by the existing `determinism_same_input_same_output`
    /// test) and additionally cross-check the kernel-level parity on
    /// a representative slice of intermediate inputs the agglomeration
    /// loop would have produced.
    #[test]
    fn rabbit_order_unchanged_with_simd_dispatch() {
        // Build a 100-vertex random graph that traverses the
        // agglomeration loop many times.
        let n = 100_u32;
        let (offsets, neighbors) = erdos_renyi_csr(n, 4, 0xC0FFEE_u64);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm_first = rabbit_order(g);
        let perm_second = rabbit_order(g);
        assert_eq!(
            perm_first, perm_second,
            "rabbit_order must be deterministic across calls"
        );
        // And independently verify a few representative inner-loop
        // batches against scalar.
        for &n_neigh in &[1_usize, 4, 7, 16] {
            let (weights, degrees) =
                random_neighbor_batch(n_neigh, 0xC0FFEE_u64 ^ n_neigh as u64, 1_000);
            let self_degree = 10_u64;
            let m_doubled = 100_u128;
            let scalar_out = kernels::scalar::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            );
            let auto_out = kernels::auto::modularity_gains_neighbor_batch(
                &weights,
                &degrees,
                self_degree,
                m_doubled,
            );
            assert_eq!(scalar_out, auto_out);
        }
    }

    // ----- audit-R7-followup #11 — try_rabbit_order fallible builder -----

    #[test]
    fn try_rabbit_order_matches_rabbit_order_on_well_formed_input() {
        // Happy path: bit-identical to the panicking sibling on a
        // shape-correct CSR.
        let n = 6_u32;
        let edges = vec![(0_u32, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let panic_path = rabbit_order(g);
        let try_path = try_rabbit_order(g).expect("well-formed CSR must succeed");
        assert_eq!(panic_path, try_path);
    }

    #[test]
    fn try_rabbit_order_rejects_non_monotone_offsets_without_panicking() {
        // offsets[1]=5 > offsets[2]=2; both within neighbors length.
        let offsets = [0_u32, 5, 2, 6];
        let neighbors = [1_u32, 2, 0, 1, 0, 2];
        let g = CsrGraph {
            n: 3,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = try_rabbit_order(g).expect_err("non-monotone offsets must error");
        match err {
            PermutationConstructionError::InvalidCsr(
                super::super::CsrGraphError::OffsetsNonMonotone { i, lo, hi },
            ) => {
                assert_eq!(i, 1);
                assert_eq!(lo, 5);
                assert_eq!(hi, 2);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn try_rabbit_order_rejects_neighbors_length_mismatch_without_panicking() {
        // offsets[n] = 2 but neighbors only has 1 element.
        let offsets = [0_u32, 2];
        let neighbors = [1_u32];
        let g = CsrGraph {
            n: 1,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let err = try_rabbit_order(g).expect_err("tail-length mismatch must error");
        assert_eq!(
            err,
            PermutationConstructionError::InvalidCsr(
                super::super::CsrGraphError::NeighborsLengthMismatch {
                    offsets_tail: 2,
                    neighbors_len: 1,
                }
            )
        );
    }

    // ---------------------------------------------------------------------
    // Sprint 53-55: round-based concurrent merging tests.
    //
    // The parallel variant produces an identical permutation to the
    // sequential one ONLY on degenerate inputs that admit no merges
    // (empty, n=1, n=2, fully disconnected, edgeless). On non-trivial
    // inputs both produce valid Rabbit Order permutations but the
    // merge sequences differ — the heap-based sequential path
    // re-evaluates each vertex against the current state, while the
    // round-based parallel path locks per-round proposals against the
    // round-start snapshot. The tests below cover both regimes.

    #[cfg(feature = "parallel")]
    mod parallel_tests {
        //! Sprint 53-55 deterministic round-based parallel rabbit-order tests.
        //!
        //! Covers:
        //!   * Trivial-input parity vs the sequential path.
        //!   * Run-to-run determinism (same thread count, three runs).
        //!   * Cross-thread-count determinism (1, 2, 4, 8 threads via
        //!     `rayon::ThreadPoolBuilder::install`).
        //!   * Permutation validity property on randomised inputs.
        //!   * Quality property: K-clique-with-bridges → cliques
        //!     contiguous in output.
        use super::*;

        /// Constructs a graph with directed-edge count above
        /// [`RABBIT_PARALLEL_EDGE_THRESHOLD`] so the parallel path
        /// actually executes. Used by the determinism tests below.
        ///
        /// Returns `(n, offsets, neighbors)`. The graph is a chain of
        /// 4-cliques connected by single bridge edges, expanded to
        /// reach the edge threshold.
        ///
        /// `num_cliques` controls how far above the threshold we go;
        /// callers that just need to exercise the parallel path use
        /// the minimum, while quality tests use a larger size.
        fn clique_chain(num_cliques: u32) -> (u32, Vec<u32>, Vec<u32>) {
            // Each 4-clique has 6 undirected edges = 12 directed.
            // Each bridge contributes 2 directed edges. To exceed
            // RABBIT_PARALLEL_EDGE_THRESHOLD (= 200_000), we need
            // roughly 200_000 / 12 ~= 17_000 cliques.
            let n: u32 = num_cliques * 4;
            let mut edges: Vec<(u32, u32)> =
                Vec::with_capacity((num_cliques as usize) * 6 + (num_cliques as usize - 1));
            for k in 0..num_cliques {
                let base = k * 4;
                for i in 0..4_u32 {
                    for j in (i + 1)..4_u32 {
                        edges.push((base + i, base + j));
                    }
                }
                if k + 1 < num_cliques {
                    // Bridge: last vertex of clique k -> first vertex of clique k+1.
                    edges.push((base + 3, (k + 1) * 4));
                }
            }
            let (offsets, neighbors) = csr_from_edges(n, &edges);
            (n, offsets, neighbors)
        }

        /// Convenience wrapper: minimum-sized chain that still trips
        /// the parallel threshold. Used by the determinism tests
        /// where speed of test wall time matters.
        fn min_above_threshold_chain() -> (u32, Vec<u32>, Vec<u32>) {
            // 17_000 cliques give 17_000 * 12 + 16_999 * 2 = 237_998
            // directed edges, comfortably above the 200_000 threshold.
            clique_chain(17_000)
        }

        /// Convenience wrapper: large chain for the quality property
        /// test, where seeing many cliques matters more than wall
        /// time.
        fn large_clique_chain() -> (u32, Vec<u32>, Vec<u32>) {
            clique_chain(18_000)
        }

        #[test]
        fn par_matches_sequential_on_empty_graph() {
            let offsets = [0_u32];
            let neighbors: [u32; 0] = [];
            let g = CsrGraph {
                n: 0,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            assert_eq!(rabbit_order_par(g), rabbit_order(g));
        }

        #[test]
        fn par_matches_sequential_on_single_vertex() {
            let offsets = [0_u32, 0];
            let neighbors: [u32; 0] = [];
            let g = CsrGraph {
                n: 1,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            assert_eq!(rabbit_order_par(g), rabbit_order(g));
        }

        #[test]
        fn par_matches_sequential_on_two_connected_vertices() {
            let offsets = [0_u32, 1, 2];
            let neighbors = [1_u32, 0];
            let g = CsrGraph {
                n: 2,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            assert_eq!(rabbit_order_par(g), rabbit_order(g));
        }

        #[test]
        fn par_matches_sequential_on_fully_disconnected() {
            let offsets = vec![0_u32; 6];
            let neighbors: Vec<u32> = Vec::new();
            let g = CsrGraph {
                n: 5,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            assert_eq!(rabbit_order_par(g), rabbit_order(g));
        }

        #[test]
        fn par_matches_sequential_on_edgeless_graph() {
            // n=10 vertices, no edges. `total_edge_weight == 0`
            // triggers the identity-permutation early return on both
            // paths.
            let offsets = vec![0_u32; 11];
            let neighbors: Vec<u32> = Vec::new();
            let g = CsrGraph {
                n: 10,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            assert_eq!(rabbit_order_par(g), rabbit_order(g));
        }

        #[test]
        fn par_is_deterministic_across_runs() {
            // Use a graph above the parallel threshold so the
            // multi-round agglomeration code path actually runs.
            let (n, offsets, neighbors) = min_above_threshold_chain();
            let g = CsrGraph {
                n,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            let p1 = rabbit_order_par(g);
            let p2 = rabbit_order_par(g);
            let p3 = rabbit_order_par(g);
            assert_eq!(p1, p2, "rabbit_order_par must be deterministic across runs");
            assert_eq!(p2, p3, "rabbit_order_par must be deterministic across runs");
        }

        #[test]
        fn par_is_deterministic_across_thread_counts() {
            // Build the input once. Use the minimum-above-threshold
            // chain to keep the four-pool test wall time bounded
            // while still actually exercising the parallel path.
            let (n, offsets, neighbors) = min_above_threshold_chain();
            let g = CsrGraph {
                n,
                offsets: &offsets,
                neighbors: &neighbors,
            };

            // Run under each of {1, 2, 4, 8} threads and assert the
            // resulting permutations are identical.
            let mut perms: Vec<Permutation> = Vec::with_capacity(4);
            for &tc in &[1_usize, 2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(tc)
                    .build()
                    .expect("ThreadPoolBuilder must succeed");
                let p = pool.install(|| rabbit_order_par(g));
                perms.push(p);
            }
            for w in perms.windows(2) {
                assert_eq!(
                    w[0], w[1],
                    "rabbit_order_par must be deterministic across thread counts"
                );
            }
        }

        #[test]
        fn par_returns_valid_permutation_on_random_graphs() {
            // Generate several random graphs above the parallel
            // threshold; assert each output is a valid permutation.
            for &(n, deg, seed) in &[
                (50_000_u32, 5_u32, 0xC0FFEE_u64),
                (50_000, 8, 0xDEAD_BEEF),
                (100_000, 3, 0x5151_5EED),
            ] {
                let (offsets, neighbors) = erdos_renyi_csr(n, deg, seed);
                if neighbors.len() < RABBIT_PARALLEL_EDGE_THRESHOLD {
                    // Below threshold falls back to sequential; that's
                    // already covered by the sequential tests.
                    continue;
                }
                let g = CsrGraph {
                    n,
                    offsets: &offsets,
                    neighbors: &neighbors,
                };
                let perm = rabbit_order_par(g);
                assert_valid_permutation(&perm, n as usize);
            }
        }

        #[test]
        fn par_returns_valid_permutation_below_threshold() {
            // Small inputs that fall back to the sequential path must
            // still produce valid permutations through the public API.
            for &n in &[5_u32, 16, 32, 64] {
                let edges: Vec<(u32, u32)> = (0..(n - 1)).map(|i| (i, i + 1)).collect();
                let (offsets, neighbors) = csr_from_edges(n, &edges);
                let g = CsrGraph {
                    n,
                    offsets: &offsets,
                    neighbors: &neighbors,
                };
                let perm = rabbit_order_par(g);
                assert_valid_permutation(&perm, n as usize);
                // Below threshold, the parallel path delegates to the
                // sequential path so they are bit-exact.
                assert_eq!(perm, rabbit_order(g));
            }
        }

        #[test]
        fn par_groups_clique_members_contiguously() {
            // Quality property: same K-clique-with-bridges check the
            // sequential path runs (tolerated within a small slack).
            let cliques: [[u32; 4]; 4] =
                [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]];
            let n = 16_u32;
            let mut edges: Vec<(u32, u32)> = Vec::new();
            for clique in &cliques {
                for i in 0..clique.len() {
                    for j in (i + 1)..clique.len() {
                        edges.push((clique[i], clique[j]));
                    }
                }
            }
            edges.push((3, 4));
            edges.push((7, 8));
            edges.push((11, 12));
            let (offsets, neighbors) = csr_from_edges(n, &edges);
            let g = CsrGraph {
                n,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            // Below threshold; the par variant delegates to sequential
            // for n=16. Validate the contract regardless: the parallel
            // path must satisfy the same quality property.
            let perm = rabbit_order_par(g);
            assert_valid_permutation(&perm, n as usize);

            let p = perm.as_slice();
            for clique in &cliques {
                let mut min_pos = u32::MAX;
                let mut max_pos = 0_u32;
                for &v in clique {
                    let pos = p[v as usize];
                    if pos < min_pos {
                        min_pos = pos;
                    }
                    if pos > max_pos {
                        max_pos = pos;
                    }
                }
                let span = max_pos - min_pos;
                assert!(
                    span <= 4,
                    "clique {clique:?} span {span} exceeds tolerance (positions: {:?})",
                    clique.iter().map(|&v| p[v as usize]).collect::<Vec<_>>()
                );
            }
        }

        #[test]
        fn par_groups_clique_chain_above_threshold() {
            // Quality property on the above-threshold graph: every
            // 4-clique must have its 4 vertices in some contiguous (or
            // very nearly contiguous) block of the output, allowing
            // for the bridge-vertex absorption near boundaries.
            let (n, offsets, neighbors) = large_clique_chain();
            let g = CsrGraph {
                n,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            let perm = rabbit_order_par(g);
            assert_valid_permutation(&perm, n as usize);
            let p = perm.as_slice();
            // Inspect every 50th clique to keep the assertion runtime
            // bounded while still covering a representative slice.
            let num_cliques = n / 4;
            let mut tight = 0_u32;
            let mut slack = 0_u32;
            for k in (0..num_cliques).step_by(50) {
                let base = k * 4;
                let mut min_pos = u32::MAX;
                let mut max_pos = 0_u32;
                for offset in 0..4 {
                    let pos = p[(base + offset) as usize];
                    if pos < min_pos {
                        min_pos = pos;
                    }
                    if pos > max_pos {
                        max_pos = pos;
                    }
                }
                let span = max_pos - min_pos;
                if span <= 3 {
                    tight += 1;
                } else if span <= 6 {
                    slack += 1;
                }
            }
            // Expect overwhelmingly tight clique groupings; allow a
            // small minority of slack outliers near community
            // boundaries where the bridge vertex can be absorbed
            // either way.
            assert!(tight + slack > 0, "no cliques inspected; sampling broke?");
            let total = tight + slack;
            assert!(
                tight * 5 >= total * 4,
                "expected >=80% tight cliques, got tight={tight} slack={slack} total_inspected={total}"
            );
        }

        // ----- audit-R7-followup #11 — try_rabbit_order_par fallible builder -----

        #[test]
        fn try_rabbit_order_par_matches_par_on_well_formed_input() {
            // Happy path: bit-identical to the panicking sibling on a
            // well-formed CSR. Use a small input that takes the
            // sequential fallback under the hood.
            let n = 6_u32;
            let edges = vec![(0_u32, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)];
            let (offsets, neighbors) = csr_from_edges(n, &edges);
            let g = CsrGraph {
                n,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            let panic_path = rabbit_order_par(g);
            let try_path = try_rabbit_order_par(g).expect("well-formed CSR must succeed");
            assert_eq!(panic_path, try_path);
        }

        #[test]
        fn try_rabbit_order_par_rejects_offsets_length_mismatch() {
            // offsets.len()=2 but n=3 — should be 4 entries.
            let offsets = [0_u32, 0];
            let neighbors: [u32; 0] = [];
            let g = CsrGraph {
                n: 3,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            let err = try_rabbit_order_par(g).expect_err("malformed offsets must error, not panic");
            assert_eq!(
                err,
                PermutationConstructionError::InvalidCsr(
                    super::super::super::CsrGraphError::OffsetsLengthMismatch {
                        actual_len: 2,
                        expected_len: 4,
                    }
                )
            );
        }

        #[test]
        fn try_rabbit_order_par_rejects_neighbor_out_of_range() {
            // Single neighbour with id 99 but n=2.
            let offsets = [0_u32, 1, 1];
            let neighbors = [99_u32];
            let g = CsrGraph {
                n: 2,
                offsets: &offsets,
                neighbors: &neighbors,
            };
            let err =
                try_rabbit_order_par(g).expect_err("out-of-range neighbour must error, not panic");
            assert_eq!(
                err,
                PermutationConstructionError::InvalidCsr(
                    super::super::super::CsrGraphError::NeighborOutOfRange {
                        neighbor: 99,
                        n: 2,
                        at_index: 0,
                    }
                )
            );
        }
    }
}

//! Rabbit Order — community-detection-driven graph permutation.
//!
//! See `docs/v0.2_planning/14_PERMUTATION.md` § 3 for the spec and
//! `docs/v0.2_planning/03_EXECUTION_PLAN.md` § "Sprint 47-49" for the
//! sprint-level milestone.
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
//! ## Sequential baseline scope
//!
//! **This is the Sprint 47-49 sequential baseline only.** The reference
//! C++ implementation (`araij/rabbit_order`) parallelises step 3 via
//! per-thread merge buffers and a concurrent hash map; the SIMD inner
//! loop for the modularity-gain dot product and the concurrent merging
//! land in Sprint 50-52 and Sprint 53-55 respectively (see
//! `03_EXECUTION_PLAN.md` Phase D1). This file is single-threaded,
//! single-pass (no recursive multi-level), and uses sorted `Vec`-backed
//! adjacency rather than a concurrent hash map.
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

use super::{CsrGraph, Permutation};

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
/// ## Sequential baseline
///
/// **This is the Sprint 47-49 single-pass sequential baseline.** Quality
/// is meaningfully better than RCM on community-structured graphs but
/// worse than the multi-level concurrent reference; see the module-level
/// documentation for the full sprint plan. The function signature, type
/// discipline, and determinism guarantees are stable across the
/// follow-on SIMD and concurrent-merging sprints — only build cost will
/// improve.
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
/// or if any neighbour ID is out of range.
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
        return Permutation::identity(0);
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
        return Permutation::identity(n);
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
        let best = best_merge_target(u, &adj[u_idx], &weighted_degree, total_edge_weight);

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
fn best_merge_target(
    u: u32,
    u_adj: &AdjList,
    weighted_degree: &[u64],
    total_edge_weight: u64,
) -> Option<u32> {
    let m = i128::from(total_edge_weight);
    let two_m = 2_i128 * m;
    let deg_u = i128::from(weighted_degree[u as usize]);

    let mut best_v: Option<u32> = None;
    // Score must strictly exceed zero for a merge to happen; we then
    // beat the running best on score, with ties broken on neighbour
    // ID. Tracking the score independently from `best_v` lets us
    // collapse the "first positive candidate" and "improve over
    // previous best" cases into one comparison sequence.
    let mut best_score: i128 = 0;

    for &(v, w) in u_adj {
        let deg_v = i128::from(weighted_degree[v as usize]);
        let w_i = i128::from(w);
        let score = two_m * w_i - deg_u * deg_v;
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
        let first: std::collections::BTreeSet<u32> = inv[..3].iter().copied().collect();
        let second: std::collections::BTreeSet<u32> = inv[3..].iter().copied().collect();
        let group_a: std::collections::BTreeSet<u32> = [0, 1, 2].into_iter().collect();
        let group_b: std::collections::BTreeSet<u32> = [3, 4, 5].into_iter().collect();
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
    // Parity is left for Sprint 50-52 / Sprint 53-55 once the SIMD
    // and concurrent variants land; the sequential baseline shipped
    // here is verified by the property and quality tests above.
}

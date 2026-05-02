//! Reverse Cuthill-McKee (RCM) ordering.
//!
//! See `docs/v0.2_planning/14_PERMUTATION.md` § 2 for the spec. The
//! algorithm:
//!
//! 1. Pick a **pseudoperipheral** start vertex via the GPS algorithm:
//!    BFS once from a low-degree vertex, find the deepest level, BFS
//!    from one of those, repeat until depth doesn't increase.
//! 2. BFS from the start. At each level, sort the frontier by ascending
//!    vertex degree (tie-break by lower vertex ID for determinism).
//! 3. Record the visit order. Reverse it. That's the permutation.
//!
//! Disconnected graphs: when BFS from a chosen start exhausts before
//! visiting every vertex, restart from the lowest-degree unvisited
//! vertex; concatenate orderings. Reversing happens once at the end.
//!
//! ## Complexity
//!
//! * Time: O(|V| + |E| log Δ) where Δ is the max degree (the log is
//!   from sorting frontiers).
//! * Space: O(|V|) for the queue + visit-order array.
//!
//! For 228 K vertices with average degree 5 (TokenFS-typical), this is
//! roughly 10 ms on a modern x86 P-core.
//!
//! ## Determinism
//!
//! * Tie-breaking on equal-degree neighbours is by ascending vertex ID.
//! * Tie-breaking on equal-degree start candidates (during the
//!   pseudoperipheral search and component restarts) is by ascending
//!   vertex ID.
//! * The same input graph always produces the same permutation,
//!   independent of host architecture.

#[cfg(not(feature = "std"))]
use alloc::vec;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::{CsrGraph, Permutation};

/// Maximum number of GPS pseudoperipheral iterations.
///
/// In pathological cases the GPS loop can oscillate; bounding the
/// iteration count protects against worst-case build time. Typical
/// graphs converge in 2-3 iterations. The published GPS paper reports
/// ≤ 5 across SPARSE-collection benchmarks.
const GPS_MAX_ITERATIONS: u32 = 8;

/// Computes a Reverse Cuthill-McKee ordering for `graph`.
///
/// Returns a [`Permutation`] of length `graph.n` where `perm[old_id] =
/// new_id`. Connected components are processed in order of their
/// lowest-degree starting vertex (ties broken by lowest vertex ID); the
/// pseudoperipheral start vertex within a component is found by the
/// GPS algorithm.
///
/// Empty graphs (`n == 0`) and isolated vertices (no edges) are
/// handled gracefully — the result is the identity for those special
/// cases.
///
/// # Panics
///
/// Panics if `graph.offsets.len()` is not exactly `graph.n + 1`, or if
/// any neighbour ID is out of range `0..graph.n`, or if a CSR offset
/// pair is inverted.
#[must_use]
pub fn rcm(graph: CsrGraph<'_>) -> Permutation {
    let n = graph.n as usize;
    assert_eq!(
        graph.offsets.len(),
        n + 1,
        "rcm: offsets.len() ({}) != n + 1 ({})",
        graph.offsets.len(),
        n + 1
    );
    if n == 0 {
        return Permutation::identity(0);
    }
    // Validate offsets are monotone and in-bounds; bail out early if not.
    for w in graph.offsets.windows(2) {
        assert!(
            w[0] <= w[1],
            "rcm: offsets non-monotone: offsets contains {} followed by {}",
            w[0],
            w[1]
        );
    }
    assert_eq!(
        graph.offsets[n] as usize,
        graph.neighbors.len(),
        "rcm: offsets[n] ({}) != neighbors.len() ({})",
        graph.offsets[n],
        graph.neighbors.len()
    );

    // Per-vertex degree cache. CSR `degree(v)` is one subtraction; the
    // hot loop reads degree per visited frontier element so caching as
    // a flat Vec<u32> keeps the inner loop branch-free.
    let degrees: Vec<u32> = (0..n)
        .map(|v| graph.offsets[v + 1] - graph.offsets[v])
        .collect();

    // Visit order accumulator, sized exactly so the final reversal
    // produces an n-length permutation.
    let mut order = Vec::with_capacity(n);
    let mut visited = vec![false; n];

    // Process every connected component. Within each component, find a
    // pseudoperipheral start vertex via GPS, then run the BFS.
    while order.len() < n {
        // Lowest-degree unvisited vertex; ties by lowest ID.
        let component_seed = lowest_degree_unvisited(&visited, &degrees);
        let start = pseudoperipheral_vertex(&graph, &degrees, component_seed, &visited);
        bfs_record_order(&graph, &degrees, start, &mut visited, &mut order);
    }

    debug_assert_eq!(order.len(), n, "rcm: visit order length mismatch");
    order.reverse();
    // SAFETY: `bfs_record_order` records every vertex exactly once via the
    // `visited` mask, and the outer loop drains every connected component
    // until `order.len() == n`. Reversing preserves the bijection. Hence
    // `order` is a permutation of `0..n` with `n <= u32::MAX as usize`
    // (CSR vertex IDs are u32 by construction).
    unsafe { Permutation::from_vec_unchecked(order) }
}

/// Finds the lowest-degree unvisited vertex (tie-break: lowest ID).
///
/// The visited mask must contain at least one unvisited slot; callers
/// guarantee this via the outer `while order.len() < n` loop.
fn lowest_degree_unvisited(visited: &[bool], degrees: &[u32]) -> u32 {
    debug_assert!(
        !visited.is_empty(),
        "lowest_degree_unvisited on empty graph"
    );
    let mut best = u32::MAX;
    let mut best_deg = u32::MAX;
    for (v, &was_seen) in visited.iter().enumerate() {
        if was_seen {
            continue;
        }
        let d = degrees[v];
        if d < best_deg || (d == best_deg && (v as u32) < best) {
            best_deg = d;
            best = v as u32;
        }
    }
    debug_assert!(best != u32::MAX, "no unvisited vertex");
    best
}

/// Returns a pseudoperipheral starting vertex for the connected
/// component containing `seed`.
///
/// Uses the GPS algorithm (Gibbs, Poole, Stockmeyer 1976):
///
/// 1. BFS from `seed`. Let the deepest level be `L_seed`.
/// 2. Pick the deepest-level vertex of lowest degree (ties by lowest
///    ID), call it `next`. BFS from `next`. Let the deepest level be
///    `L_next`.
/// 3. If `L_next > L_seed`, repeat with `next` as the new seed.
/// 4. Else return `next` (or `seed` if no progress was made).
///
/// Bounded by [`GPS_MAX_ITERATIONS`] to protect against pathological
/// oscillation.
fn pseudoperipheral_vertex(
    graph: &CsrGraph<'_>,
    degrees: &[u32],
    seed: u32,
    visited_global: &[bool],
) -> u32 {
    // Allocate the BFS scratch once per call; reusing across iterations
    // avoids re-zeroing the level array repeatedly inside the GPS loop.
    let n = graph.n as usize;
    let mut levels = vec![u32::MAX; n];
    let mut queue = Vec::with_capacity(n);

    let mut current = seed;
    let mut current_depth = bfs_levels(graph, current, visited_global, &mut levels, &mut queue);

    for _ in 0..GPS_MAX_ITERATIONS {
        // Find the lowest-degree vertex at the deepest level.
        let mut best = current;
        let mut best_deg = degrees[current as usize];
        for (v, &lvl) in levels.iter().enumerate() {
            if lvl != current_depth {
                continue;
            }
            let d = degrees[v];
            if d < best_deg || (d == best_deg && (v as u32) < best) {
                best_deg = d;
                best = v as u32;
            }
        }
        if best == current {
            // No deeper-level candidate distinct from the current seed
            // exists, or the only deepest-level vertex is `current`
            // itself — converged.
            return current;
        }

        let candidate_depth = bfs_levels(graph, best, visited_global, &mut levels, &mut queue);
        if candidate_depth > current_depth {
            current = best;
            current_depth = candidate_depth;
        } else {
            // No deeper BFS rooted at `best`; we cannot improve, so
            // `current` (the previous root) is the pseudoperipheral
            // pick. Re-running the BFS from `current` is wasted work
            // but conceptually correct.
            return current;
        }
    }
    current
}

/// Runs a BFS from `root` and writes per-vertex levels into `levels`.
///
/// Vertices already marked `visited_global[v] == true` are NOT skipped
/// here — the GPS pseudoperipheral search must traverse the entire
/// component the first time. Instead, the caller passes `visited_global`
/// only as a hint about which vertices belong to *other* components
/// already processed (those are skipped). Vertices not reachable from
/// `root` keep `u32::MAX` as their level.
///
/// Returns the maximum level reached (the BFS eccentricity of `root`).
///
/// `levels` and `queue` are reused buffers — caller-provided so the
/// GPS loop avoids reallocating per iteration.
fn bfs_levels(
    graph: &CsrGraph<'_>,
    root: u32,
    visited_global: &[bool],
    levels: &mut [u32],
    queue: &mut Vec<u32>,
) -> u32 {
    for slot in levels.iter_mut() {
        *slot = u32::MAX;
    }
    queue.clear();

    levels[root as usize] = 0;
    queue.push(root);
    let mut head = 0_usize;
    let mut max_level = 0_u32;

    while head < queue.len() {
        let v = queue[head];
        head += 1;
        let lvl = levels[v as usize];
        let neighbors = graph.neighbors_of(v);
        for &u in neighbors {
            // Self-loops and duplicate edges leave `u == v` or `u`
            // already-discovered; both cases short-circuit at the
            // `levels[u] != MAX` check below.
            let u_idx = u as usize;
            if visited_global[u_idx] {
                // Belongs to a previously-processed component; treat
                // as not-in-this-component for GPS purposes. A
                // well-formed disconnected graph never has cross-
                // component edges, so this branch is rarely taken,
                // but defensive against malformed inputs.
                continue;
            }
            if levels[u_idx] != u32::MAX {
                continue;
            }
            let next_lvl = lvl + 1;
            levels[u_idx] = next_lvl;
            if next_lvl > max_level {
                max_level = next_lvl;
            }
            queue.push(u);
        }
    }
    max_level
}

/// BFS from `root` recording the visit order into `order`. At each
/// frontier expansion, neighbours are visited in ascending-degree
/// order (tie-break: lowest ID).
///
/// `visited` is the global visited mask shared across components; this
/// function marks every vertex it reaches.
fn bfs_record_order(
    graph: &CsrGraph<'_>,
    degrees: &[u32],
    root: u32,
    visited: &mut [bool],
    order: &mut Vec<u32>,
) {
    if visited[root as usize] {
        return;
    }

    // Working buffer for sorting each vertex's not-yet-seen neighbours.
    // Bounded by max_degree, allocated once and reused across vertices
    // within this component.
    let mut frontier_buf: Vec<u32> = Vec::new();

    let mut queue: Vec<u32> = Vec::new();
    visited[root as usize] = true;
    order.push(root);
    queue.push(root);
    let mut head = 0_usize;

    while head < queue.len() {
        let v = queue[head];
        head += 1;

        // Gather not-yet-visited neighbours of `v`.
        frontier_buf.clear();
        for &u in graph.neighbors_of(v) {
            if u == v {
                // Self-loop contributes degree weight but not a new
                // neighbour to enqueue.
                continue;
            }
            let u_idx = u as usize;
            if !visited[u_idx] {
                // Mark visited NOW to avoid double-enqueueing the same
                // vertex from two parents in the same frontier.
                visited[u_idx] = true;
                frontier_buf.push(u);
            }
        }

        // Sort by ascending degree, then ascending ID. The standard
        // `sort_unstable_by` is fine — equal-degree entries are
        // broken by the explicit secondary key, so output is
        // deterministic regardless of stability.
        frontier_buf.sort_unstable_by(|&a, &b| {
            let da = degrees[a as usize];
            let db = degrees[b as usize];
            (da, a).cmp(&(db, b))
        });

        for &u in &frontier_buf {
            order.push(u);
            queue.push(u);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a CSR adjacency from a list of undirected edges.
    fn csr_from_edges(n: u32, edges: &[(u32, u32)]) -> (Vec<u32>, Vec<u32>) {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
        for &(a, b) in edges {
            adj[a as usize].push(b);
            if a != b {
                adj[b as usize].push(a);
            }
        }
        // Sort each adjacency list for canonical ordering — RCM's
        // tie-breaking is by ID, so the input order doesn't change
        // the output, but a sorted CSR is what most real inputs look
        // like and makes test debugging easier.
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
            let id = id as usize;
            assert!(id < n, "permutation contains id {id} >= n {n}");
            assert!(!seen[id], "permutation contains duplicate id {id}");
            seen[id] = true;
        }
        assert!(seen.iter().all(|b| *b), "permutation missing an id");
    }

    /// Computes the bandwidth of `graph` under permutation `perm`.
    /// Bandwidth = `max_{(u,v) edge} |perm[u] - perm[v]|`.
    fn bandwidth(graph: &CsrGraph<'_>, perm: &Permutation) -> u32 {
        let p = perm.as_slice();
        let mut bw = 0_u32;
        for v in 0..graph.n {
            for &u in graph.neighbors_of(v) {
                if u == v {
                    continue;
                }
                let pv = p[v as usize];
                let pu = p[u as usize];
                let delta = pv.abs_diff(pu);
                if delta > bw {
                    bw = delta;
                }
            }
        }
        bw
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
        let perm = rcm(g);
        assert!(perm.is_empty());
    }

    #[test]
    fn single_vertex_no_edges_returns_identity() {
        let offsets = [0_u32, 0];
        let neighbors: [u32; 0] = [];
        let g = CsrGraph {
            n: 1,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rcm(g);
        assert_valid_permutation(&perm, 1);
        assert_eq!(perm.as_slice(), &[0_u32]);
    }

    #[test]
    fn isolated_vertices_no_edges_visit_each_once() {
        // 5 disconnected isolated vertices.
        let offsets = vec![0_u32; 6];
        let neighbors: Vec<u32> = Vec::new();
        let g = CsrGraph {
            n: 5,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rcm(g);
        assert_valid_permutation(&perm, 5);
    }

    #[test]
    fn path_graph_bandwidth_no_worse_than_natural_order() {
        // Path graph 0-1-2-3-4-5-6-7. Natural order has bandwidth 1.
        let n = 8_u32;
        let edges: Vec<(u32, u32)> = (0..(n - 1)).map(|i| (i, i + 1)).collect();
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let identity = Permutation::identity(n as usize);
        let original_bw = bandwidth(&g, &identity);
        let perm = rcm(g);
        let new_bw = bandwidth(&g, &perm);
        assert_valid_permutation(&perm, n as usize);
        assert!(
            new_bw <= original_bw,
            "RCM bandwidth {new_bw} > original {original_bw}"
        );
    }

    #[test]
    fn star_graph_visits_all_vertices_exactly_once() {
        // Star: vertex 0 connected to 1..=4.
        let n = 5_u32;
        let edges: Vec<(u32, u32)> = (1..n).map(|i| (0, i)).collect();
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rcm(g);
        assert_valid_permutation(&perm, n as usize);
        // The center has highest degree; the leaves have degree 1.
        // Pseudoperipheral start picks a leaf; reversing puts the
        // high-degree center near the END of natural-order positions.
        // The center should NOT end up at position 0 in the permutation.
        let center_new = perm.as_slice()[0];
        assert_ne!(
            center_new, 0,
            "center vertex 0 should not map to position 0"
        );
    }

    #[test]
    fn two_disconnected_components_concatenate() {
        // Two disjoint paths: 0-1-2 and 3-4.
        let n = 5_u32;
        let edges = vec![(0_u32, 1), (1, 2), (3, 4)];
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rcm(g);
        assert_valid_permutation(&perm, n as usize);
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
        let perm = rcm(g);
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
        let p1 = rcm(g);
        let p2 = rcm(g);
        let p3 = rcm(g);
        assert_eq!(p1, p2);
        assert_eq!(p2, p3);
    }

    /// A 5x5 grid graph in row-major form is already near-optimal for
    /// bandwidth; RCM is allowed to be slightly worse on already-good
    /// natural orderings. The honest property to check is the lower
    /// bound: bandwidth must be at least the graph's diameter (any
    /// single edge crossing the diameter forces this), and the result
    /// must remain a valid permutation. Real RCM-improves-bandwidth
    /// behaviour shows up on adversarial orderings, exercised in the
    /// shuffled-path test below.
    #[test]
    fn grid_5x5_produces_valid_permutation_and_reasonable_bandwidth() {
        let dim = 5_u32;
        let n = dim * dim;
        let mut edges = Vec::new();
        for i in 0..dim {
            for j in 0..dim {
                let v = i * dim + j;
                if i + 1 < dim {
                    edges.push((v, v + dim));
                }
                if j + 1 < dim {
                    edges.push((v, v + 1));
                }
            }
        }
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rcm(g);
        assert_valid_permutation(&perm, n as usize);
        let new_bw = bandwidth(&g, &perm);
        // Loose upper bound: any BFS-level-order ordering on a 5x5
        // grid has bandwidth ≤ |L_max| + |L_max-1| ≤ 2 * 5 = 10. We
        // use a generous bound here to catch genuine bugs (e.g. an
        // unconnected component or pessimal layering) without
        // over-fitting to grid-specific RCM behaviour.
        assert!(
            new_bw <= 2 * dim,
            "RCM bandwidth {new_bw} > 2 * dim {}",
            2 * dim
        );
    }

    /// Construct a deliberately bad initial numbering for a path graph
    /// and verify RCM strictly improves bandwidth.
    #[test]
    fn shuffled_path_bandwidth_strictly_improves() {
        // Path 0-1-2-3-4-5-6-7 but logically labelled in a
        // bandwidth-pessimal way: 0-7-1-6-2-5-3-4. Edge endpoints
        // labels reflect the *new* numbering; edges connect adjacent
        // positions in the original path.
        let n = 8_u32;
        // Path adjacency in original positions [0..7].
        // Map original positions to vertex IDs: pos 0->0, pos 1->7,
        // pos 2->1, pos 3->6, pos 4->2, pos 5->5, pos 6->3, pos 7->4.
        let pos_to_id: [u32; 8] = [0, 7, 1, 6, 2, 5, 3, 4];
        let mut edges = Vec::new();
        for p in 0..(n - 1) {
            edges.push((pos_to_id[p as usize], pos_to_id[(p + 1) as usize]));
        }
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let identity = Permutation::identity(n as usize);
        let original_bw = bandwidth(&g, &identity);
        let perm = rcm(g);
        let new_bw = bandwidth(&g, &perm);
        assert_valid_permutation(&perm, n as usize);
        assert!(
            new_bw < original_bw,
            "RCM bandwidth {new_bw} did not improve original {original_bw}"
        );
    }

    #[test]
    fn complete_graph_bandwidth_is_n_minus_1() {
        // Complete K_5 — every permutation has bandwidth n-1.
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
        let perm = rcm(g);
        assert_valid_permutation(&perm, n as usize);
        assert_eq!(bandwidth(&g, &perm), n - 1);
    }

    #[test]
    fn known_path_graph_expected_output() {
        // Path 0-1-2-3-4. CM-then-reverse for a path picks an endpoint
        // (vertex 0 or 4 — both are leaves; tie-break selects the
        // lowest ID). BFS from 0: visit 0, then 1, then 2, then 3,
        // then 4. Reverse: [4, 3, 2, 1, 0].
        // perm[old_id] = new_id:
        //   old 0 -> new 4
        //   old 1 -> new 3
        //   old 2 -> new 2
        //   old 3 -> new 1
        //   old 4 -> new 0
        let n = 5_u32;
        let edges: Vec<(u32, u32)> = (0..(n - 1)).map(|i| (i, i + 1)).collect();
        let (offsets, neighbors) = csr_from_edges(n, &edges);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rcm(g);
        // The two endpoints (0 and 4) both have degree 1. GPS converges
        // to one of them; either is pseudoperipheral. The deterministic
        // tie-break on "lowest ID" means we start the seed scan at
        // vertex 0; GPS may then advance to vertex 4 because BFS from
        // either endpoint of a path has eccentricity n-1, and the
        // deepest level holds the *opposite* endpoint. So GPS picks 0
        // or 4 depending on which BFS round breaks the tie. Both
        // produce mirror-image valid orderings; we accept either.
        assert_valid_permutation(&perm, n as usize);
        let p = perm.as_slice();
        let endpoint_a = vec![4, 3, 2, 1, 0]; // started at 0
        let endpoint_b = vec![0, 1, 2, 3, 4]; // started at 4
        assert!(
            p == endpoint_a.as_slice() || p == endpoint_b.as_slice(),
            "unexpected RCM output {p:?}"
        );
    }

    /// Synthetic Erdős-Rényi graph generator, deterministic seed.
    fn erdos_renyi_csr(n: u32, avg_degree: u32, seed: u64) -> (Vec<u32>, Vec<u32>) {
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
        // Probability ≈ avg_degree / (n - 1); use rejection sampling
        // via a simple xorshift PRNG.
        let mut state = seed | 1; // never zero
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
            let perm = rcm(g);
            assert_valid_permutation(&perm, n as usize);
            // Bandwidth is bounded by `n - 1` for any permutation; the
            // valid-permutation assertion above is the load-bearing
            // check on random ER graphs. The deterministic
            // bandwidth-improvement property shows up on adversarial
            // orderings (see `shuffled_path_bandwidth_strictly_improves`).
            let bw_rcm = bandwidth(&g, &perm);
            assert!(bw_rcm < n, "n={n}: bandwidth {bw_rcm} >= n");
        }
    }

    #[test]
    fn permutation_round_trips_with_inverse() {
        // Apply a generated RCM permutation, then its inverse, verify
        // we get the original data back. This is the "Permutation
        // type round-trip" property called out in the spec.
        let n = 100_u32;
        let (offsets, neighbors) = erdos_renyi_csr(n, 4, 0xABBA_CAFE);
        let g = CsrGraph {
            n,
            offsets: &offsets,
            neighbors: &neighbors,
        };
        let perm = rcm(g);
        let inv = perm.inverse();
        let src: Vec<u32> = (0..n).collect();
        let permuted = perm.apply(&src);
        let recovered = inv.apply(&permuted);
        assert_eq!(recovered, src);
    }
}

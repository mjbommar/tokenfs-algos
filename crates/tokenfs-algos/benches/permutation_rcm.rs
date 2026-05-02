//! `bench-permutation-rcm`: RCM build cost, Rabbit Order build cost,
//! and `Permutation::apply` throughput.
//!
//! Build cost is reported on synthetic Erdős-Rényi graphs at three
//! vertex counts (10K, 100K, 1M for RCM; 10K and 100K for Rabbit Order
//! since the sequential dendrogram baseline is ~100-500x slower than
//! RCM and 1M vertices would dominate bench wall time) and two
//! average-degree settings (5 and 20). The applied permutation
//! throughput is reported across the canonical L1 / L2 / L3 / DRAM
//! cache tiers from [`support::cache_tier_sizes`].
//!
//! Run all: `cargo bench -p tokenfs-algos --bench permutation_rcm`
//! Quick:   `cargo bench -p tokenfs-algos --bench permutation_rcm -- --quick`
//! Filter:  `cargo bench -p tokenfs-algos --bench permutation_rcm -- in-L1`
//! RCM:     `cargo bench -p tokenfs-algos --bench permutation_rcm -- rcm`
//! Rabbit:  `cargo bench -p tokenfs-algos --bench permutation_rcm -- rabbit`

#![allow(missing_docs)]
// `support` is shared with the larger workload-matrix benches; only the
// `cache_tier_sizes` helper is consumed here, which leaves most of the
// module unreferenced from this binary.
#![allow(dead_code)]

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use support::cache_tier_sizes;
use tokenfs_algos::permutation::rabbit::kernels;
use tokenfs_algos::permutation::{CsrGraph, Permutation, rabbit_order, rcm};

/// Vertex counts swept by the build benchmark.
///
/// 1 M vertices at average degree 20 generates ~20 M edges and a
/// CSR neighbours buffer of ~80 MB; the bench runs in milliseconds
/// per iteration but keep an eye on the working set when expanding.
const VERTEX_COUNTS: &[u32] = &[10_000, 100_000, 1_000_000];

/// Rabbit Order vertex counts. Smaller than [`VERTEX_COUNTS`] because
/// the Sprint 47-49 sequential dendrogram baseline is ~100-500x heavier
/// per vertex than RCM; including 1 M would inflate criterion's wall
/// time per run beyond the typical PR-CI budget. The SIMD inner loop
/// (Sprint 50-52) and concurrent merging (Sprint 53-55) follow-on
/// sprints will close most of that gap and we'll add the 1 M tier back
/// at that point.
const RABBIT_VERTEX_COUNTS: &[u32] = &[10_000, 100_000];

/// Average undirected degree settings.
///
/// 5 matches TokenFS-typical extent-similarity graphs; 20 is a heavier
/// workload representative of dedup clusters and richer adjacency.
const AVERAGE_DEGREES: &[u32] = &[5, 20];

/// Generates an Erdős-Rényi-style undirected graph with target average
/// degree using a deterministic PRNG.
///
/// Returns `(offsets, neighbours)` ready to wrap in a [`CsrGraph`].
fn erdos_renyi(n: u32, avg_degree: u32, seed: u64) -> (Vec<u32>, Vec<u32>) {
    // Probability per edge p = avg_degree / (n - 1).
    // We avoid materialising the O(n^2) edge list by streaming
    // candidate edges and rejecting via PRNG.
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
    let mut state = seed | 1;
    let denom = u64::from(n).saturating_sub(1).max(1);
    let p_num = u64::from(avg_degree);

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

/// Faster ER-style generator that scales linearly: each vertex draws
/// `avg_degree` random neighbours via PRNG and emits both directed
/// edges. Avoids the O(n^2) candidate-edge sweep used by [`erdos_renyi`]
/// for the small-n correctness benchmarks.
///
/// At n = 1 M, vertex-per-vertex sampling yields ~5 M-20 M directed
/// edges, totally bench-sized. The generator allows a small number of
/// duplicates and self-loops; RCM is robust to both.
fn linear_random_graph(n: u32, avg_degree: u32, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n as usize];
    let mut state = seed | 1;
    let denom = u64::from(n);

    for u in 0..n {
        for _ in 0..avg_degree {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let target = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) % denom) as u32;
            if target == u {
                continue;
            }
            adj[u as usize].push(target);
            adj[target as usize].push(u);
        }
    }
    // Deduplicate to keep the CSR clean. RCM tolerates duplicates but
    // they inflate degree counters and bias the apparent average.
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
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

fn bench_rcm_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_rcm/build");
    for &n in VERTEX_COUNTS {
        for &deg in AVERAGE_DEGREES {
            let (offsets, neighbors) =
                linear_random_graph(n, deg, 0xF22_C0FFEE_u64 ^ u64::from(n) ^ u64::from(deg));
            let graph_edges = neighbors.len() as u64;
            // Throughput in vertices/sec is the headline metric for
            // RCM build cost; secondary edge throughput is reported
            // via the elements axis.
            group.throughput(Throughput::Elements(u64::from(n)));

            let id = format!("rcm/n={n}/avg_deg={deg}/edges={graph_edges}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let g = CsrGraph {
                        n,
                        offsets: black_box(&offsets),
                        neighbors: black_box(&neighbors),
                    };
                    rcm(g)
                });
            });
        }
    }
    group.finish();
}

fn bench_rabbit_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_rabbit/build");
    // The sequential baseline is heavy enough that larger n per
    // criterion sample inflates run time noticeably; cap the sample
    // count and warmup to keep total bench wall time tractable while
    // still producing statistically meaningful numbers.
    group.sample_size(10);
    group.warm_up_time(core::time::Duration::from_millis(500));
    for &n in RABBIT_VERTEX_COUNTS {
        for &deg in AVERAGE_DEGREES {
            let (offsets, neighbors) =
                linear_random_graph(n, deg, 0xF22_C0FFEE_u64 ^ u64::from(n) ^ u64::from(deg));
            let graph_edges = neighbors.len() as u64;
            // Throughput in vertices/sec is the natural axis to
            // compare against `bench_rcm_build`.
            group.throughput(Throughput::Elements(u64::from(n)));

            let id = format!("rabbit/n={n}/avg_deg={deg}/edges={graph_edges}");
            group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
                b.iter(|| {
                    let g = CsrGraph {
                        n,
                        offsets: black_box(&offsets),
                        neighbors: black_box(&neighbors),
                    };
                    rabbit_order(g)
                });
            });
        }
    }
    group.finish();
}

fn deterministic_u32_payload(byte_size: usize, seed: u64) -> Vec<u32> {
    let n = byte_size / core::mem::size_of::<u32>();
    let mut state = seed;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32
        })
        .collect()
}

/// Builds a `Permutation` whose mapping is the bit-reversal of
/// `0..n`. Bit-reversal is a non-trivial permutation that touches
/// every output slot — perfect for measuring the gather/scatter cost
/// of `Permutation::apply` independent of the construction kernel.
fn deterministic_permutation(n: usize) -> Permutation {
    // For n that's a power of two, bit-reversal yields a valid
    // permutation. For non-power-of-two `n`, fall back to a simple
    // odd-stride mapping that's also bijective: perm[i] = (i * stride)
    // mod n, choosing `stride` coprime to `n`.
    if n == 0 {
        return Permutation::identity(0);
    }
    let stride = pick_coprime_stride(n);
    let perm: Vec<u32> = (0..n)
        .map(|i| ((i as u64 * stride) % n as u64) as u32)
        .collect();
    // SAFETY: `pick_coprime_stride(n)` returns `stride` coprime to `n`
    // (with `gcd(stride, n) == 1`), so `i -> (i * stride) mod n` is a
    // bijection on `0..n`. `n` is a usize from the bench harness and is
    // far below `u32::MAX`. The inner test suite cross-checks this
    // construction against `try_from_vec`.
    unsafe { Permutation::from_vec_unchecked(perm) }
}

/// Returns an odd integer coprime to `n`. For `n >= 1`, 1 is always
/// coprime to `n`; we step up to the smallest odd integer > sqrt(n)
/// that's coprime to keep the permutation visibly non-trivial.
fn pick_coprime_stride(n: usize) -> u64 {
    if n <= 2 {
        return 1;
    }
    let target = (n as f64).sqrt() as u64 | 1; // make it odd
    let mut s = target.max(3);
    while gcd(s, n as u64) != 1 {
        s += 2;
    }
    s
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a
}

fn bench_apply_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_rcm/apply_u32");
    for (tier_label, byte_size) in cache_tier_sizes() {
        let src = deterministic_u32_payload(*byte_size, 0xF22_C0FFEE_u64 ^ (*byte_size as u64));
        let n = src.len();
        if n == 0 {
            continue;
        }
        let perm = deterministic_permutation(n);
        // Throughput in bytes/sec — apply is essentially a u32-strided
        // gather/scatter, so reporting per-byte makes the cache-tier
        // axis directly comparable with the popcount/bit_pack benches.
        let buffer_bytes = n * core::mem::size_of::<u32>();
        group.throughput(Throughput::Bytes(buffer_bytes as u64));

        // `apply_into` is the kernel-safe form (caller-provided buffer);
        // bench it with a pre-allocated dst so we measure the gather/
        // scatter cost without the allocator hidden inside `apply`.
        let id = format!("apply_into/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            let mut dst = vec![0_u32; n];
            b.iter(|| {
                perm.apply_into(black_box(&src), black_box(&mut dst));
            });
        });

        // `apply` reports the allocating form; useful for dataset-build
        // pipelines that build a fresh permuted Vec.
        let id = format!("apply/{tier_label}");
        group.bench_with_input(BenchmarkId::from_parameter(id), &(), |b, _| {
            b.iter(|| perm.apply(black_box(&src)));
        });
    }
    group.finish();
}

/// Neighbour-batch sizes swept by the modularity-gain inner-loop
/// bench. Spans the realistic "few neighbours per merge" regime
/// (10-100, average for TokenFS-typical sparse graphs) up through the
/// stress regime (10K, representative of dense super-vertex
/// adjacency after several merge rounds).
const RABBIT_NEIGHBOR_COUNTS: &[usize] = &[10, 100, 1_000, 10_000];

/// Builds a deterministic neighbour-batch payload for the modularity-
/// gain inner-loop benchmark. Values are bounded by `2^30` so the
/// inputs fit the i64 fast path (`< 2^32`) regardless of which
/// SIMD backend the runtime selects.
fn modularity_gain_inputs(n: usize, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let mut state = seed | 1;
    let mut next = || -> u64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    let bound = 1_u64 << 30;
    let mut weights = Vec::with_capacity(n);
    let mut degrees = Vec::with_capacity(n);
    for _ in 0..n {
        weights.push(next() % bound);
        degrees.push(next() % bound);
    }
    (weights, degrees)
}

/// Benchmarks the modularity-gain SIMD inner loop in isolation,
/// comparing scalar vs the runtime-dispatched best backend across
/// a sweep of neighbour-batch sizes.
///
/// The bench reports throughput in elements/sec (one element = one
/// neighbour scored), which makes the scalar-vs-SIMD speedup ratio
/// visible directly in the criterion output. The scalar arm always
/// runs the i128 reference; the auto arm picks AVX-512 → AVX2 → NEON
/// → scalar at runtime per the dispatcher in
/// [`tokenfs_algos::permutation::rabbit::kernels::auto`].
fn bench_modularity_gain_kernel(c: &mut Criterion) {
    let mut group = c.benchmark_group("permutation_rabbit/modularity_gain");
    let self_degree = 12_345_u64;
    let m_doubled = 9_876_543_u128;

    for &n in RABBIT_NEIGHBOR_COUNTS {
        let (weights, degrees) = modularity_gain_inputs(n, 0xC0FFEE_u64 ^ n as u64);
        group.throughput(Throughput::Elements(n as u64));

        let id_scalar = format!("scalar/n={n}");
        group.bench_with_input(BenchmarkId::from_parameter(id_scalar), &(), |b, _| {
            b.iter(|| {
                kernels::scalar::modularity_gains_neighbor_batch(
                    black_box(&weights),
                    black_box(&degrees),
                    black_box(self_degree),
                    black_box(m_doubled),
                )
            });
        });

        let id_auto = format!("auto/n={n}");
        group.bench_with_input(BenchmarkId::from_parameter(id_auto), &(), |b, _| {
            b.iter(|| {
                kernels::auto::modularity_gains_neighbor_batch(
                    black_box(&weights),
                    black_box(&degrees),
                    black_box(self_degree),
                    black_box(m_doubled),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rcm_build,
    bench_rabbit_build,
    bench_apply_throughput,
    bench_modularity_gain_kernel
);
criterion_main!(benches);

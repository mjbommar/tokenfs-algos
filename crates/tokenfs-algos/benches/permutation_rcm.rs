//! `bench-permutation-rcm`: RCM build cost + Permutation::apply throughput.
//!
//! Build cost is reported on synthetic Erdős-Rényi graphs at three
//! vertex counts (10K, 100K, 1M) and two average-degree settings (5
//! and 20). The applied permutation throughput is reported across the
//! canonical L1 / L2 / L3 / DRAM cache tiers from
//! [`support::cache_tier_sizes`].
//!
//! Run all: `cargo bench -p tokenfs-algos --bench permutation_rcm`
//! Quick:   `cargo bench -p tokenfs-algos --bench permutation_rcm -- --quick`
//! Filter:  `cargo bench -p tokenfs-algos --bench permutation_rcm -- in-L1`

#![allow(missing_docs)]
// `support` is shared with the larger workload-matrix benches; only the
// `cache_tier_sizes` helper is consumed here, which leaves most of the
// module unreferenced from this binary.
#![allow(dead_code)]

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use support::cache_tier_sizes;
use tokenfs_algos::permutation::{CsrGraph, Permutation, rcm};

/// Vertex counts swept by the build benchmark.
///
/// 1 M vertices at average degree 20 generates ~20 M edges and a
/// CSR neighbours buffer of ~80 MB; the bench runs in milliseconds
/// per iteration but keep an eye on the working set when expanding.
const VERTEX_COUNTS: &[u32] = &[10_000, 100_000, 1_000_000];

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
    // Bench-only: trust the construction; the inner test suite verifies
    // it via `try_from_vec`.
    Permutation::from_vec_unchecked(perm)
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

criterion_group!(benches, bench_rcm_build, bench_apply_throughput);
criterion_main!(benches);

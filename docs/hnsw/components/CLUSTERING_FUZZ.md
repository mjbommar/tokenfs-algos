# Component: Clustering-fuzz generator + walker correctness gate

**Status:** spec, 2026-05-04. Designed by user; replaces the original
"libusearch golden fixture" approach in PHASE_1.md / PHASE_4.md.

**Lives in:**
- Generator: `crates/tokenfs-algos/src/similarity/hnsw/testing/clustering.rs` (`#[cfg(test)]`-gated; not part of the public surface)
- Round-trip oracle: `crates/tokenfs-algos/tests/hnsw_clustering.rs` (integration test)
- Continuous-input fuzz target: `fuzz/fuzz_targets/hnsw_walker_clustering.rs` (libfuzzer-driven)

## Problem statement

The HNSW walker needs a correctness oracle. The traditional approach is
"build with libusearch, save bytes, commit, walk our parser, assert
identical k-NN result list." That requires a libusearch dependency at
fixture-generation time and only catches byte-format regressions on a
single corpus.

Per user direction (2026-05-04): no external dependencies, no vendoring.
Survey of `_references/usearch/cpp/test.cpp` confirmed usearch ships no
committed binary fixtures — every test builds an index at runtime,
saves, reloads, re-queries (see lines 318, 538, 561). Same pattern, no
shareable bytes.

The replacement: **construct ground truth.** Generate vectors with a
clustering structure baked in by construction; assert the walker
recovers that structure.

## Generator design

### Inputs

- `M`: number of seed clusters
- `N`: number of variants per seed (so total corpus size is M × N)
- `dim`: vector length in bytes (32 for F22 fingerprints; 256 for binary
  256-bit MinHash signatures)
- `p`: bit-flip probability per byte (0.0 → identical to seed; 0.5 →
  uniformly random; sweet spot for clustering tests is 0.05 – 0.20)
- `seed`: deterministic RNG seed (ChaCha8) per `DETERMINISM.md`

### Output

```
Vec<(NodeKey, Vec<u8>, ClusterId)>
```

Each entry is a vector belonging to a known cluster. NodeKey is
sequential (0..M*N). ClusterId is the seed index (0..M). The vector is
the seed bytes with each bit independently flipped at probability p.

### Properties (by construction)

1. **Variants of seed_i are similar to seed_i.** Hamming distance from
   variant to seed is binomial(dim*8, p), expected p*dim*8 bits.
2. **Variants of distinct seeds are dissimilar.** Two seeds drawn
   uniformly random differ in expected dim*8/2 bits; their variants
   differ by approximately the same amount.
3. **The clustering is recoverable.** For p < ~0.25, the walker should
   find that variant_i_j's nearest neighbors are dominated by other
   variants of seed_i.

### Algorithm

```
function generate_clustering_corpus(M, N, dim, p, seed):
    rng = ChaCha8::seed_from_u64(seed)
    seeds: Vec<Vec<u8>> = (0..M).map(|_| rng.gen_bytes(dim))
    corpus: Vec<(NodeKey, Vec<u8>, ClusterId)> = []
    key = 0
    for cluster_id in 0..M:
        for variant in 0..N:
            v = seeds[cluster_id].clone()
            for byte_idx in 0..dim:
                for bit_idx in 0..8:
                    if rng.gen_f32() < p:
                        v[byte_idx] ^= 1 << bit_idx
            corpus.push((key, v, cluster_id))
            key += 1
    return corpus
```

Pure-functional, deterministic, no allocation outside `Vec` growth.

## Correctness assertions

For a corpus generated with parameters (M, N, dim, p, seed):

1. **Build → walk recovers cluster membership at recall threshold T.**
   For each variant v belonging to cluster c, run `try_search(v, k=N)`.
   At least T*N of the top-N results should belong to cluster c. T is a
   function of p (lower p → higher achievable T).
   - At p=0.05, expect T >= 0.95 (almost-perfect clustering)
   - At p=0.10, expect T >= 0.90
   - At p=0.20, expect T >= 0.70

2. **Brute-force scan agrees with walker.** Compute the brute-force
   top-N from the same corpus using the same metric. Walker results may
   differ in *order* (HNSW is approximate), but the set overlap must be
   >= T_walker (typically 0.95 at efSearch=64).

3. **Determinism.** Same (M, N, dim, p, seed) → byte-identical corpus.
   Same corpus + same query → identical walker result list.

## Round-trip test (Phase 4 oracle)

```rust
// Pseudocode: tests/hnsw_clustering.rs

#[test]
fn walker_recovers_clusters_at_p_05() {
    let corpus = generate_clustering_corpus(M=100, N=20, dim=32, p=0.05, seed=42);
    let mut builder = Builder::try_new(BuildConfig {
        dimensions: 32,
        scalar_kind: ScalarKind::Binary,
        metric: Metric::Hamming,
        seed: 42,
        ..Default::default()
    }).unwrap();
    for (key, vec, _) in &corpus {
        builder.try_insert(*key, vec).unwrap();
    }
    let bytes = builder.try_finish_to_bytes().unwrap();
    let view = HnswView::try_new(&bytes).unwrap();

    let mut total_recall = 0.0_f64;
    for (query_key, query_vec, expected_cluster) in &corpus {
        let results = try_search(&view, query_vec, &SearchConfig {
            k: 20, ef_search: 64,
            metric: Metric::Hamming,
            scalar_kind: ScalarKind::Binary,
            ..Default::default()
        }).unwrap();

        let same_cluster = results.iter()
            .filter(|(k, _)| corpus[*k as usize].2 == *expected_cluster)
            .count();
        total_recall += same_cluster as f64 / 20.0;
    }
    let avg_recall = total_recall / corpus.len() as f64;
    assert!(avg_recall >= 0.95, "expected recall >= 0.95 at p=0.05, got {:.3}", avg_recall);
}

#[test]
fn walker_brute_force_overlap_at_p_10() { ... }
#[test]
fn walker_deterministic_across_runs() { ... }
```

## libfuzzer integration (Phase 4 hardening)

A new `fuzz/fuzz_targets/hnsw_walker_clustering.rs` target feeds the
clustering-fuzz parameters from libfuzzer's input bytes (M, N, p, seed
all derived from the byte stream); builds the corpus; builds the index;
walks; asserts recall is above a *floor* (not the Phase 4 expected
value, just "non-zero recall — not random") for any non-pathological
parameter combination.

```rust
fuzz_target!(|data: &[u8]| {
    if data.len() < 8 { return; }
    let M = (data[0] as usize % 30) + 5;       // 5..35 clusters
    let N = (data[1] as usize % 15) + 5;       // 5..20 variants
    let p_q8 = (data[2] as f32) / 255.0 * 0.3; // 0..0.3 flip probability
    let seed = u64::from_le_bytes(data[..8].try_into().unwrap());
    // ... build, walk, assert recall >= 0.5 (sanity floor)
});
```

This catches: walker returning empty results on non-empty corpus, walker
panicking on edge cases, walker returning unsorted candidates, walker
violating the visited-set invariant.

## Why this is better than libusearch golden fixture

| Concern | libusearch fixture | Clustering-fuzz |
|---|---|---|
| External dep at gen time | Yes (libusearch C++) | No (pure Rust) |
| Catches byte-format regression | Yes | No (Phase 1 toy fixture covers this) |
| Catches algorithmic recall regression | Only on the one corpus | On any corpus shape |
| Reproducible across machines | Yes (binary committed) | Yes (deterministic from seed) |
| Repo size impact | 1-2 MB committed | 0 bytes (generated at test time) |
| Generalizes to new metrics / dims | One fixture per | One generator handles all |
| User can run locally without C++ toolchain | No | Yes |

The hand-crafted toy fixture in Phase 1 covers the byte-format regression
that the libusearch golden fixture would have caught. The clustering-fuzz
covers the algorithmic regression that the golden fixture *couldn't*
catch — recall changes that don't perturb the byte layout.

## Optional libusearch wire-format compat (manual sanity)

For the rare user with `pip install usearch==2.25.x` available locally:

`tests/hnsw_libusearch_compat.rs`, gated on
`#[cfg(feature = "hnsw-libusearch-compat")]`. Spawns Python in a
subprocess, asks libusearch to read our serialized bytes, asserts
libusearch finds the same nearest neighbors we did. Not in CI; opt-in
only. Documented in `tests/README.md` as "if you want to verify
multi-language reader compatibility against libusearch directly, opt
into this feature." Out of scope for v0.7.0 ship; could land in a
later v0.7.x patch.

## Cross-references

- Replaces: prior PHASE_1.md "libusearch fixture" deliverable
- Phases that consume this:
  - [`../phases/PHASE_4.md`](../phases/PHASE_4.md) — adds round-trip + clustering-fuzz tests as primary correctness gate
  - [`../phases/PHASE_5.md`](../phases/PHASE_5.md) — adds the libfuzzer-driven `hnsw_walker_clustering` fuzz target
- Related: [`WALKER.md`](WALKER.md), [`BUILDER.md`](BUILDER.md)
- Pattern reference: existing `fuzz/fuzz_targets/{bitmap_intersect_parity,vector_distance_parity}.rs` (mode-byte input layout, bounded sizes, parity-shaped assertions)
- Determinism: [`../research/DETERMINISM.md`](../research/DETERMINISM.md) (ChaCha8 seeded RNG)

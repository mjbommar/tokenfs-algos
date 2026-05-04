//! HNSW native deterministic builder (Phase 4 of v0.7.0).
//!
//! Composes the in-memory [`super::graph::Graph`], the neighbor-
//! selection heuristics in [`super::select`], the level-assignment
//! RNG in [`level`], and the upcoming `Builder::try_insert`
//! (Algorithm 1) to construct an HNSW index. Output is byte-for-byte
//! usearch v2.25 wire format via the [`serialize`] sub-module.
//!
//! ## Determinism contract
//!
//! Per `docs/hnsw/research/DETERMINISM.md`:
//!
//! - **Single-threaded insert.** No rayon. (An optional `parallel`
//!   feature can land later for non-deterministic builds.)
//! - **Sorted input.** Caller MUST sort vectors by `NodeKey`
//!   ascending before insertion.
//! - **Seeded RNG.** Level assignment uses [`level::random_level`]
//!   driven by a `ChaCha8Rng` seeded from `BuildConfig::seed`.
//! - **Tie-break ordering.** Candidate min-heap and selection pre-
//!   sort by `(distance, NodeKey)` ascending.
//! - **Frozen config.** All hyperparameters (M, M_max, M_max0,
//!   ef_construction, max_level, mL) baked into the spec at
//!   construction; cannot mutate during build.
//!
//! Same `(BuildConfig, input_sequence)` always produces byte-
//! identical wire-format output.
//!
//! ## Phase coverage
//!
//! - **#241 (this commit, Phase 4.1):** [`level::random_level`] +
//!   `BuildConfig` skeleton. Builder API itself comes in 4.2.
//! - **Phase 4.2:** `Builder::try_insert` calling Algorithm 1 over
//!   [`super::graph::Graph`] + [`super::select::select_neighbors_simple`].
//! - **Phase 4.3:** [`serialize::serialize_to_bytes`] —
//!   `Graph` → usearch v2.25 wire-format `Vec<u8>`.
//! - **Phase 4.4:** Round-trip integration test (build → serialize
//!   → walker → brute-force comparison via tokenfs-algos-corpora
//!   clustering presets per `CLUSTERING_FUZZ.md`).
//! - **Phase 4.5:** Algorithm 4 (heuristic neighbor selection).

#![cfg(feature = "hnsw-build")]

pub mod level;

/// Builder hyperparameters per HNSW paper §4.2.
///
/// Frozen at construction; cannot mutate during build. See
/// [`HNSW_ALGORITHM_NOTES.md`](../../../../../docs/hnsw/research/HNSW_ALGORITHM_NOTES.md)
/// §2 for recommended values per use case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildConfig {
    /// Bytes per vector (must equal what the walker expects via
    /// `HnswView::bytes_per_vector`).
    pub bytes_per_vector: usize,
    /// `M` from the paper. Cap on edges per non-base layer. Typical 8-32.
    pub m: u32,
    /// `M_max`: cap on edges in upper layers. Default `M`.
    pub m_max: u32,
    /// `M_max0`: cap on edges in layer 0. Default `2 * M`.
    pub m_max0: u32,
    /// Dynamic candidate-list size during construction. Typical 64-200.
    pub ef_construction: u32,
    /// Hard cap on level value (avoids pathological tail).
    pub max_level: u8,
    /// Level normalization factor `1 / ln(M)` per paper §4.1.
    /// Stored as f64 to avoid recomputing per insert.
    pub level_mult: f64,
    /// RNG seed for level assignment. Per `DETERMINISM.md`, derived
    /// from the TokenFS image_salt by the caller.
    pub seed: u64,
}

impl BuildConfig {
    /// Construct with paper-recommended defaults derived from `m`.
    /// `bytes_per_vector` and `seed` are caller-supplied; everything
    /// else follows the standard ratios.
    pub fn from_m(m: u32, bytes_per_vector: usize, seed: u64) -> Self {
        BuildConfig {
            bytes_per_vector,
            m,
            m_max: m,
            m_max0: m.saturating_mul(2),
            ef_construction: 64.max(m.saturating_mul(4)),
            max_level: 16,
            level_mult: 1.0 / (m as f64).ln().max(f64::MIN_POSITIVE),
            seed,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn build_config_from_m_sets_paper_defaults() {
        let cfg = BuildConfig::from_m(16, 32, 0xCAFE_BABE);
        assert_eq!(cfg.m, 16);
        assert_eq!(cfg.m_max, 16);
        assert_eq!(cfg.m_max0, 32);
        assert_eq!(cfg.ef_construction, 64);
        assert_eq!(cfg.max_level, 16);
        assert_eq!(cfg.bytes_per_vector, 32);
        assert_eq!(cfg.seed, 0xCAFE_BABE);
        // level_mult = 1 / ln(16) ≈ 0.3606
        assert!((cfg.level_mult - 0.3606).abs() < 0.001);
    }

    #[test]
    fn build_config_from_m_handles_m_4() {
        let cfg = BuildConfig::from_m(4, 16, 1);
        assert_eq!(cfg.m, 4);
        assert_eq!(cfg.m_max0, 8);
        assert_eq!(cfg.ef_construction, 64); // floor at 64
    }

    #[test]
    fn build_config_from_m_handles_large_m() {
        let cfg = BuildConfig::from_m(64, 16, 1);
        assert_eq!(cfg.ef_construction, 256); // 4 * m
    }
}

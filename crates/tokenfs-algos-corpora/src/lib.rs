//! Parameterized synthetic corpus generators for `tokenfs-algos`.
//!
//! Three composable layers (per `docs/hnsw/components/CLUSTERING_FUZZ.md`
//! generalized to all 10 use cases):
//!
//! - **Bytes** ([`bytes`]): how raw bytes are generated. uniform-random
//!   / Markov-chain / bit-flip-variants / byte-histogram-targeted.
//! - **Structure** ([`structure`]): how items are arranged. flat /
//!   file-Pareto-sized / time-snapshotted / hierarchical-doc-chunks.
//! - **Vector** ([`vector`]): what fingerprint each item produces.
//!   F22 / MinHash / f32-embedding / i8-embedding / LSH-bands.
//!
//! Composition: [`CorpusSpec`] picks one of each layer; [`Corpus`]
//! holds the generated items plus their [`GroundTruth`] (cluster
//! membership, expected k-NN, contamination labels). Determinism is
//! ChaCha8 — same seed produces byte-identical output across runs and
//! architectures.
//!
//! ## Scale
//!
//! Every preset takes a [`Scale`] enum so the same generator runs at
//! three sizes:
//!
//! ```text
//! Tiny    <  100 items  — for unit tests; runs in <100 ms
//! Small   ~ 10k items   — for criterion benches; runs in seconds
//! Medium  ~100k items   — for nightly regression runs
//! Large   ~  1M items   — for paper-quality bench writeups
//! ```
//!
//! Same generator code, same parameters, just different `n_items`.
//!
//! ## Why this exists
//!
//! Real-world corpora (Ubuntu rootfs, linux-mainline, NIST CFReDS) are
//! non-portable, large, license-encumbered, and slow to bench against.
//! Synthetic data with **constructed ground truth** is more useful for
//! testing because we control the answer — recall measurement becomes
//! `assert!(recall >= 0.95)` instead of "looks plausible." Real corpora
//! become end-to-end validation, synthetic is the day-to-day dev loop.
//!
//! Phase 4 of the HNSW landing uses these generators as the primary
//! correctness gate (build → walk → brute-force scan; recall floor
//! per `CLUSTERING_FUZZ.md` §"Correctness assertions"). Future fuzz
//! targets consume the same generators.

#![warn(missing_docs)]

pub mod bytes;
pub mod ground_truth;
pub mod presets;
pub mod structure;
pub mod vector;

pub use self::bytes::BytesLayer;
pub use self::ground_truth::{Cluster, ExpectedKnn, GroundTruth};
pub use self::structure::StructureLayer;
pub use self::vector::VectorLayer;

/// External caller-supplied key. Matches `tokenfs_algos::similarity::hnsw::NodeKey`.
pub type NodeKey = u64;

/// Cluster identifier within a corpus's ground truth.
pub type ClusterId = u32;

/// Generator scale knob. Same spec runs at any of these sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Scale {
    /// <100 items. Unit-test scale — runs in <100 ms.
    Tiny,
    /// ~10k items. Criterion-bench scale — runs in seconds.
    Small,
    /// ~100k items. Nightly-regression scale.
    Medium,
    /// ~1M items. Paper-quality bench writeup scale.
    Large,
}

impl Scale {
    /// Default item count for this scale. Presets may override per
    /// their use-case shape (e.g. snapshot-dedup at `Tiny` is 5
    /// snapshots × 20 base files = 100 items).
    pub const fn default_item_count(self) -> usize {
        match self {
            Scale::Tiny => 100,
            Scale::Small => 10_000,
            Scale::Medium => 100_000,
            Scale::Large => 1_000_000,
        }
    }
}

/// Top-level generator specification.
///
/// Pick one of each layer, hand to [`CorpusSpec::generate`] to get a
/// [`Corpus`]. Determinism: same `(spec, seed)` → byte-identical output.
#[derive(Debug, Clone)]
pub struct CorpusSpec {
    /// Number of items to generate (top-level item count; each layer
    /// may interpret this differently — e.g. `StructureLayer::TimeSnapshots`
    /// produces `n_items × snapshots` rows).
    pub n_items: usize,
    /// Seed for the ChaCha8 RNG.
    pub seed: u64,
    /// How raw bytes are generated.
    pub bytes_layer: BytesLayer,
    /// How items are organized.
    pub structure_layer: StructureLayer,
    /// What fingerprint each item produces.
    pub vector_layer: VectorLayer,
}

impl CorpusSpec {
    /// Generate the [`Corpus`] for this spec.
    ///
    /// Phase 1 of the corpora crate: layer scaffolds + the
    /// clustering bit-flip-variants preset only. Other layer
    /// combinations panic with "not yet implemented" — to be filled
    /// in across tasks #236 / #237 / #238.
    pub fn generate(&self) -> Corpus {
        // Wire-up lands progressively across the bytes / vector /
        // structure layer tasks. For now this is a stub returning a
        // skeleton Corpus with empty items — the per-layer
        // implementations replace this as they land.
        Corpus {
            items: Vec::new(),
            ground_truth: GroundTruth::empty(),
        }
    }
}

/// Generated corpus.
///
/// Items + ground truth. Ground truth lets correctness tests assert
/// concrete properties ("recall >= 0.95 on cluster i") rather than
/// "results look plausible."
#[derive(Debug, Clone)]
pub struct Corpus {
    /// Generated items in insertion order.
    pub items: Vec<CorpusItem>,
    /// Constructed ground truth — known cluster membership, expected
    /// k-NN, contamination labels.
    pub ground_truth: GroundTruth,
}

/// One generated item.
#[derive(Debug, Clone)]
pub struct CorpusItem {
    /// Sequential key (monotonic from 0).
    pub key: NodeKey,
    /// Raw bytes generated by the [`BytesLayer`].
    pub bytes: Vec<u8>,
    /// Fingerprint generated by the [`VectorLayer`] from the bytes.
    pub vector: Vec<u8>,
    /// Cluster membership if the bytes layer assigns one (bit-flip-
    /// variants does; uniform-random does not).
    pub cluster_id: Option<ClusterId>,
}

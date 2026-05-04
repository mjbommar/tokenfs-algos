//! Per-use-case preset constructors.
//!
//! Each preset is a named function returning a [`crate::CorpusSpec`]
//! pre-configured for one of the 10 downstream use cases. Sized via
//! the [`crate::Scale`] enum so the same preset runs at unit-test
//! and paper-bench scales.
//!
//! Preset roster (per the use-case map in conversation history):
//!
//! | # | Use case | Preset fn |
//! |---|---|---|
//! | 1 | rootfs near-dup | [`rootfs_near_duplicate`] |
//! | 2 | code monorepo | [`monorepo_code_similarity`] |
//! | 3 | snapshot dedup | [`snapshot_dedup`] |
//! | 4 | forensics | [`forensics_planted_clusters`] |
//! | 5 | training-data dedup | [`training_data_near_dup`] |
//! | 6 | RAG | [`rag_embeddings`] |
//! | 7 | token grep | [`token_grep`] |
//! | 8 | Postgres ext | [`postgres_extension`] |
//! | 9 | CDN edge | [`cdn_edge_cache`] |
//! | 10 | log anomaly | [`log_anomaly`] |
//!
//! Phase coverage:
//!
//! - **#235 scaffold:** every preset returns the right
//!   `CorpusSpec` shape with `n_items = 0`.
//! - **#238 (this commit):** Scale-aware `n_items` per preset, so
//!   the same preset spec runs at Tiny/Small/Medium/Large.
//!
//! Preset bodies derive `n_items` from `Scale::default_item_count`
//! by default; specific presets override (e.g. `snapshot_dedup` at
//! `Tiny` is 5 base files × 12 snapshots = 60 items).

use crate::{BytesLayer, CorpusSpec, Scale, StructureLayer, VectorLayer};

/// Per-scale base item count for clustering presets that derive
/// total_items from `clusters * variants_per_cluster`. Returns
/// `(clusters, variants_per_cluster)` such that the product is
/// reasonable for the given scale.
const fn cluster_dims_for_scale(scale: Scale) -> (usize, usize) {
    match scale {
        Scale::Tiny => (10, 5),        // 50 items
        Scale::Small => (200, 50),     // 10k items
        Scale::Medium => (1_000, 100), // 100k items
        Scale::Large => (5_000, 200),  // 1M items
    }
}

/// Use case 1 — rootfs near-duplicate detection (F22 / Hamming).
pub fn rootfs_near_duplicate(scale: Scale, seed: u64) -> CorpusSpec {
    let (clusters, variants_per_cluster) = cluster_dims_for_scale(scale);
    CorpusSpec {
        n_items: clusters * variants_per_cluster,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters,
            variants_per_cluster,
            p_flip: 0.05,
        },
        structure_layer: StructureLayer::FilesParetoSized {
            alpha: 1.5,
            scale: 4096,
        },
        vector_layer: VectorLayer::F22ExtentFingerprint,
    }
}

/// Use case 2 — code monorepo similarity (MinHash / Jaccard).
pub fn monorepo_code_similarity(scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: scale.default_item_count(),
        seed,
        bytes_layer: BytesLayer::MarkovChain {
            bytes_per_item: 4096,
            order: 2,
            transition_seed: seed.wrapping_add(0xC0DE),
        },
        structure_layer: StructureLayer::FilesParetoSized {
            alpha: 1.7,
            scale: 2048,
        },
        vector_layer: VectorLayer::MinHashSignature { k: 64 },
    }
}

/// Use case 3 — snapshot dedup (SHA-256 + F22).
///
/// Per `Scale`: Tiny = 5 base × 12 snapshots = 60; Small = 800 × 12 ≈ 10k; etc.
pub fn snapshot_dedup(scale: Scale, seed: u64) -> CorpusSpec {
    let snapshots = 12;
    let base_files = match scale {
        Scale::Tiny => 5,
        Scale::Small => 800,
        Scale::Medium => 8_000,
        Scale::Large => 80_000,
    };
    CorpusSpec {
        n_items: base_files,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters: base_files,
            variants_per_cluster: 1,
            p_flip: 0.0,
        },
        structure_layer: StructureLayer::TimeSnapshots {
            snapshots,
            p_unchanged: 0.95,
            p_modified: 0.04,
            p_new: 0.01,
        },
        vector_layer: VectorLayer::F22ExtentFingerprint,
    }
}

/// Use case 4 — forensics (planted F22 clusters across drives).
pub fn forensics_planted_clusters(scale: Scale, seed: u64) -> CorpusSpec {
    let (clusters, variants_per_cluster) = cluster_dims_for_scale(scale);
    CorpusSpec {
        n_items: clusters * variants_per_cluster,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters,
            variants_per_cluster,
            p_flip: 0.10,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::F22ExtentFingerprint,
    }
}

/// Use case 5 — training-data near-duplicate filter (MinHash + LSH).
pub fn training_data_near_dup(scale: Scale, seed: u64) -> CorpusSpec {
    let docs = match scale {
        Scale::Tiny => 5,
        Scale::Small => 500,
        Scale::Medium => 5_000,
        Scale::Large => 50_000,
    };
    CorpusSpec {
        n_items: docs * 20, // ~20 chunks per doc on average
        seed,
        bytes_layer: BytesLayer::MarkovChain {
            bytes_per_item: 1024,
            order: 3,
            transition_seed: seed.wrapping_add(0x7E57),
        },
        structure_layer: StructureLayer::HierarchicalDocChunks {
            docs,
            chunks_per_doc_mean: 20.0,
            near_verbatim_rate: 0.05,
        },
        vector_layer: VectorLayer::MinHashSignature { k: 128 },
    }
}

/// Use case 6 — RAG (cosine over i8-quantized 384-dim embeddings).
pub fn rag_embeddings(scale: Scale, seed: u64) -> CorpusSpec {
    let (clusters, variants_per_cluster) = cluster_dims_for_scale(scale);
    CorpusSpec {
        // Cluster structure here is in the EmbeddingI8 vector layer
        // (per-cluster centroid) rather than in the bytes; the bytes
        // layer is a stub generator just to drive item count.
        n_items: clusters * variants_per_cluster,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 1, // bytes ignored; vector layer is synthetic
            clusters,
            variants_per_cluster,
            p_flip: 0.0,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::EmbeddingI8 {
            dim: 384,
            anisotropy: 0.4,
            hub_rate: 0.01,
        },
    }
}

/// Use case 7 — token-aware grep (n-gram-frequency sketches via LSH).
pub fn token_grep(scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: scale.default_item_count(),
        seed,
        bytes_layer: BytesLayer::MarkovChain {
            bytes_per_item: 8192,
            order: 4,
            transition_seed: seed.wrapping_add(0x6792),
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::LshBands {
            bands: 16,
            rows_per_band: 8,
        },
    }
}

/// Use case 8 — Postgres extension benchmark (i8 embedding column).
pub fn postgres_extension(scale: Scale, seed: u64) -> CorpusSpec {
    let (clusters, variants_per_cluster) = cluster_dims_for_scale(scale);
    CorpusSpec {
        n_items: clusters * variants_per_cluster,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 1, // bytes ignored; vector is synthetic
            clusters,
            variants_per_cluster,
            p_flip: 0.0,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::EmbeddingI8 {
            dim: 256,
            anisotropy: 0.3,
            hub_rate: 0.005,
        },
    }
}

/// Use case 9 — CDN edge cache (Hamming on F22, Zipfian access — but the
/// access pattern is a separate concern; this preset just produces the
/// corpus to be cached).
pub fn cdn_edge_cache(scale: Scale, seed: u64) -> CorpusSpec {
    let (clusters, variants_per_cluster) = cluster_dims_for_scale(scale);
    CorpusSpec {
        n_items: clusters * variants_per_cluster,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters,
            variants_per_cluster,
            p_flip: 0.03,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::F22ExtentFingerprint,
    }
}

/// Use case 10 — audit log anomaly detection (cosine over f32 embeddings
/// of log lines, with controllable anomaly-injection rate).
pub fn log_anomaly(scale: Scale, seed: u64) -> CorpusSpec {
    let (clusters, variants_per_cluster) = cluster_dims_for_scale(scale);
    CorpusSpec {
        n_items: clusters * variants_per_cluster,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 1,
            clusters,
            variants_per_cluster,
            p_flip: 0.0,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::EmbeddingF32 {
            dim: 128,
            anisotropy: 0.5,
            hub_rate: 0.02,
        },
    }
}

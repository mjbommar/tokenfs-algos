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
//! Phase coverage for this scaffold (#235): every preset returns a
//! placeholder `CorpusSpec` with the right shape but zero items —
//! `unimplemented!()` is avoided so the workspace compiles green.
//! Preset bodies fill in across #236 / #237 / #238 once each layer
//! lands.

use crate::{BytesLayer, CorpusSpec, Scale, StructureLayer, VectorLayer};

/// Use case 1 — rootfs near-duplicate detection (F22 / Hamming).
pub fn rootfs_near_duplicate(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0, // filled in #236 / #237 / #238
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters: 100,
            variants_per_cluster: 20,
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
pub fn monorepo_code_similarity(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
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
pub fn snapshot_dedup(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters: 50,
            variants_per_cluster: 1,
            p_flip: 0.0,
        },
        structure_layer: StructureLayer::TimeSnapshots {
            snapshots: 12,
            p_unchanged: 0.95,
            p_modified: 0.04,
            p_new: 0.01,
        },
        vector_layer: VectorLayer::F22ExtentFingerprint,
    }
}

/// Use case 4 — forensics (planted F22 clusters across drives).
pub fn forensics_planted_clusters(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters: 30,
            variants_per_cluster: 50,
            p_flip: 0.10,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::F22ExtentFingerprint,
    }
}

/// Use case 5 — training-data near-duplicate filter (MinHash + LSH).
pub fn training_data_near_dup(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
        seed,
        bytes_layer: BytesLayer::MarkovChain {
            bytes_per_item: 1024,
            order: 3,
            transition_seed: seed.wrapping_add(0x7E57),
        },
        structure_layer: StructureLayer::HierarchicalDocChunks {
            docs: 1000,
            chunks_per_doc_mean: 20.0,
            near_verbatim_rate: 0.05,
        },
        vector_layer: VectorLayer::MinHashSignature { k: 128 },
    }
}

/// Use case 6 — RAG (cosine over i8-quantized 384-dim embeddings).
pub fn rag_embeddings(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
        seed,
        bytes_layer: BytesLayer::UniformRandom {
            bytes_per_item: 384, // ignored — VectorLayer::EmbeddingI8 generates direct
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
pub fn token_grep(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
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
pub fn postgres_extension(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
        seed,
        bytes_layer: BytesLayer::UniformRandom {
            bytes_per_item: 256,
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
pub fn cdn_edge_cache(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
        seed,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 4096,
            clusters: 200,
            variants_per_cluster: 10,
            p_flip: 0.03,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::F22ExtentFingerprint,
    }
}

/// Use case 10 — audit log anomaly detection (cosine over f32 embeddings
/// of log lines, with controllable anomaly-injection rate).
pub fn log_anomaly(_scale: Scale, seed: u64) -> CorpusSpec {
    CorpusSpec {
        n_items: 0,
        seed,
        bytes_layer: BytesLayer::MarkovChain {
            bytes_per_item: 256,
            order: 2,
            transition_seed: seed.wrapping_add(0x10C7),
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::EmbeddingF32 {
            dim: 128,
            anisotropy: 0.5,
            hub_rate: 0.02,
        },
    }
}

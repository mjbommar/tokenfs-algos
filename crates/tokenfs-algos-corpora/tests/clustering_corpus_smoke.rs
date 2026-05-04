//! Integration smoke for the bit-flip-variants → Identity vector
//! pipeline. This is the load-bearing path the HNSW Phase 4 builder
//! validation will consume — `Corpus.items[i].vector` is exactly the
//! bytes the walker would search over with Hamming distance.

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use tokenfs_algos_corpora::{BytesLayer, CorpusSpec, StructureLayer, VectorLayer};

#[test]
fn bit_flip_pipeline_yields_clustered_corpus() {
    let spec = CorpusSpec {
        n_items: 0, // bytes layer derives from clusters * variants
        seed: 12345,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 32,
            clusters: 5,
            variants_per_cluster: 4,
            p_flip: 0.05,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::Identity,
    };
    let corpus = spec.generate();

    // 5 × 4 = 20 items.
    assert_eq!(corpus.items.len(), 20);
    // 5 clusters in ground truth.
    assert_eq!(corpus.ground_truth.clusters.len(), 5);
    // Each cluster has 4 members.
    for cluster in &corpus.ground_truth.clusters {
        assert_eq!(cluster.members.len(), 4);
    }

    // Every item has the right cluster id and a 32-byte vector
    // identical to its bytes (Identity vector layer).
    for (i, item) in corpus.items.iter().enumerate() {
        assert_eq!(item.cluster_id, Some((i / 4) as u32));
        assert_eq!(item.vector.len(), 32);
        assert_eq!(item.vector, item.bytes);
    }

    // Sanity: every cluster's first member should be most similar to
    // the rest of its own cluster than to other clusters' members.
    for cluster_idx in 0..5 {
        let base_key = cluster_idx * 4;
        let base = &corpus.items[base_key].vector;
        let mut intra_total = 0u32;
        let mut inter_total = 0u32;
        let mut inter_count = 0u32;
        for variant_idx in 1..4 {
            intra_total += hamming(base, &corpus.items[base_key + variant_idx].vector);
        }
        for other_cluster in 0..5 {
            if other_cluster == cluster_idx {
                continue;
            }
            for variant_idx in 0..4 {
                inter_total += hamming(base, &corpus.items[other_cluster * 4 + variant_idx].vector);
                inter_count += 1;
            }
        }
        let intra_avg = intra_total as f32 / 3.0;
        let inter_avg = inter_total as f32 / inter_count as f32;
        // 3x separation is the "clearly clustered" threshold from the
        // literature; with p=0.05 over 32-byte vectors the expected
        // ratio at the mean is ~24/128 ≈ 5x, but per-pair variance
        // can push individual clusters down to ~3x at this scale.
        // Larger N reduces the variance.
        assert!(
            intra_avg < inter_avg / 3.0,
            "cluster {cluster_idx}: intra ({intra_avg}) should be << inter ({inter_avg})"
        );
    }
}

#[test]
fn deterministic_across_runs() {
    let spec = || CorpusSpec {
        n_items: 0,
        seed: 0xCAFE_BABE,
        bytes_layer: BytesLayer::BitFlipVariants {
            bytes_per_item: 16,
            clusters: 3,
            variants_per_cluster: 5,
            p_flip: 0.10,
        },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::Identity,
    };

    let a = spec().generate();
    let b = spec().generate();
    assert_eq!(a.items.len(), b.items.len());
    for (x, y) in a.items.iter().zip(b.items.iter()) {
        assert_eq!(x.bytes, y.bytes);
        assert_eq!(x.cluster_id, y.cluster_id);
        assert_eq!(x.vector, y.vector);
    }
}

#[test]
fn uniform_random_assigns_no_cluster() {
    let spec = CorpusSpec {
        n_items: 7,
        seed: 1,
        bytes_layer: BytesLayer::UniformRandom { bytes_per_item: 8 },
        structure_layer: StructureLayer::Flat,
        vector_layer: VectorLayer::Identity,
    };
    let corpus = spec.generate();
    assert_eq!(corpus.items.len(), 7);
    for item in &corpus.items {
        assert_eq!(item.cluster_id, None);
    }
    assert!(corpus.ground_truth.clusters.is_empty());
}

fn hamming(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

// ----- Preset smoke tests (one per implemented use-case path) -----

#[test]
fn preset_snapshot_dedup_tiny_yields_dedup_at_floor() {
    use tokenfs_algos_corpora::{Scale, presets};

    let spec = presets::snapshot_dedup(Scale::Tiny, 0xCAFE);
    let corpus = spec.generate();

    // Tiny: 5 base files × 12 snapshots = 60 items.
    assert_eq!(corpus.items.len(), 60);

    // Dedup floor sanity: at least 50% of items should be bit-identical
    // to a base item (preset configures p_unchanged = 0.95).
    let base_byte_set: std::collections::HashSet<Vec<u8>> = corpus
        .items
        .iter()
        .step_by(12) // every 12th item is the snapshot-0 (base) of one file
        .map(|item| item.bytes.clone())
        .collect();
    assert_eq!(base_byte_set.len(), 5);

    let replicas = corpus
        .items
        .iter()
        .filter(|item| base_byte_set.contains(&item.bytes))
        .count();
    let dedup = replicas as f32 / corpus.items.len() as f32;
    assert!(dedup >= 0.5, "snapshot-dedup rate {dedup} below floor 0.5");
}

#[test]
fn preset_rootfs_near_duplicate_tiny_has_clusters() {
    use tokenfs_algos_corpora::{Scale, presets};

    let spec = presets::rootfs_near_duplicate(Scale::Tiny, 42);
    let corpus = spec.generate();

    // Tiny: 10 clusters × 5 variants = 50 items.
    assert_eq!(corpus.items.len(), 50);
    assert_eq!(corpus.ground_truth.clusters.len(), 10);
    for cluster in &corpus.ground_truth.clusters {
        assert_eq!(cluster.members.len(), 5);
    }

    // Vector layer is F22 → 20 bytes per vector.
    for item in &corpus.items {
        assert_eq!(item.vector.len(), 20);
    }
}

#[test]
fn preset_rag_embeddings_tiny_has_clusters_and_i8_vectors() {
    use tokenfs_algos_corpora::{Scale, presets};

    let spec = presets::rag_embeddings(Scale::Tiny, 99);
    let corpus = spec.generate();

    assert_eq!(corpus.items.len(), 50); // 10 × 5
    assert_eq!(corpus.ground_truth.clusters.len(), 10);
    for item in &corpus.items {
        assert_eq!(item.vector.len(), 384); // i8 dim=384 → 384 bytes
    }
}

#[test]
fn preset_forensics_planted_clusters_tiny() {
    use tokenfs_algos_corpora::{Scale, presets};

    let spec = presets::forensics_planted_clusters(Scale::Tiny, 7);
    let corpus = spec.generate();
    assert_eq!(corpus.items.len(), 50);
    assert_eq!(corpus.ground_truth.clusters.len(), 10);
    // F22 vector layer.
    for item in &corpus.items {
        assert_eq!(item.vector.len(), 20);
    }
}

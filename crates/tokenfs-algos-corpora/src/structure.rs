//! Layer 2 — Structure generators.
//!
//! How items are arranged in the corpus.
//!
//! Phase coverage:
//!
//! - **#238 (this commit):** [`StructureLayer::Flat`] (pass-through),
//!   [`StructureLayer::TimeSnapshots`] (snapshot-dedup model).
//! - **Later:** [`StructureLayer::FilesParetoSized`] (variable file
//!   sizes), [`StructureLayer::HierarchicalDocChunks`] (doc → chunks).
//!
//! Composition: `StructureLayer::transform(items, seed)` takes the
//! bytes-layer output and rearranges / replicates / mutates it. Most
//! layers are post-process wrappers around bytes; `Flat` is a no-op.

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

use crate::bytes::GeneratedBytes;

/// One of the four structural patterns.
#[derive(Debug, Clone)]
pub enum StructureLayer {
    /// Flat sequence of items. Each item's bytes are independent.
    Flat,
    /// File-shaped corpus with Pareto-distributed sizes (most files
    /// small, few huge — matches real-world filesystem distribution).
    /// Models use cases 1, 2, 4 (rootfs / monorepo / forensics).
    FilesParetoSized {
        /// Pareto shape parameter α. Typical filesystem: 1.5–2.0.
        alpha: f32,
        /// Scale parameter (median size in bytes).
        scale: u64,
    },
    /// Time-evolving snapshots of a base corpus. Each snapshot is a
    /// mutation of the prior (some files unchanged, some modified,
    /// some new). Models use cases 3 (snapshot dedup) + 9 (CDN edge).
    TimeSnapshots {
        /// Number of snapshots to generate.
        snapshots: usize,
        /// Probability a file is unchanged in the next snapshot.
        p_unchanged: f32,
        /// Probability a file is modified (small bit-flip mutation).
        p_modified: f32,
        /// Probability a file is new in the next snapshot.
        /// (`p_unchanged + p_modified + p_new ≈ 1.0`; remainder = deleted.)
        p_new: f32,
    },
    /// Hierarchical doc → chunk structure. Models use case 5
    /// (training-data near-duplicate filter): documents containing
    /// chunks with controllable cross-doc near-verbatim contamination.
    HierarchicalDocChunks {
        /// Number of documents.
        docs: usize,
        /// Mean chunks per document (Poisson-distributed).
        chunks_per_doc_mean: f32,
        /// Probability a chunk is a near-verbatim copy across docs.
        near_verbatim_rate: f32,
    },
}

impl StructureLayer {
    /// Total row count this structural pattern produces given the
    /// bytes layer's per-item count `n_items`.
    pub fn row_count(&self, n_items: usize) -> usize {
        match self {
            StructureLayer::Flat | StructureLayer::FilesParetoSized { .. } => n_items,
            StructureLayer::TimeSnapshots { snapshots, .. } => n_items * snapshots,
            StructureLayer::HierarchicalDocChunks {
                docs,
                chunks_per_doc_mean,
                ..
            } => (*docs as f32 * chunks_per_doc_mean) as usize,
        }
    }
}

/// Apply the structure layer's transform to the bytes layer's output.
///
/// Phase coverage:
/// - `Flat`: pass-through (identity).
/// - `TimeSnapshots`: replicates the input across N snapshots, applying
///   per-snapshot mutations to model real backup-style data evolution.
/// - `FilesParetoSized` and `HierarchicalDocChunks`: pass-through for
///   now (full implementations defer to later — the use cases that
///   need them are also v0.8+ deferred).
pub fn transform(
    layer: &StructureLayer,
    items: Vec<GeneratedBytes>,
    seed: u64,
) -> Vec<GeneratedBytes> {
    match layer {
        StructureLayer::Flat => items,
        StructureLayer::TimeSnapshots {
            snapshots,
            p_unchanged,
            p_modified,
            p_new,
        } => apply_time_snapshots(items, seed, *snapshots, *p_unchanged, *p_modified, *p_new),
        StructureLayer::FilesParetoSized { .. } | StructureLayer::HierarchicalDocChunks { .. } => {
            // Defer: pass through. Use cases needing these (rootfs at
            // realistic file-size variance, training-data with doc
            // hierarchy) are themselves v0.8+ deferred.
            items
        }
    }
}

/// Time-snapshots structure: replicate the base corpus across N
/// snapshots; each snapshot mutates with the configured probabilities.
///
/// Per-item sequencing (so a file's "lineage" is recoverable):
///   snapshot 0 of file 0, snapshot 1 of file 0, ..., snapshot N-1 of file 0,
///   snapshot 0 of file 1, ...
///
/// This is the load-bearing fixture for use case 3 (snapshot dedup):
/// across 12 monthly snapshots with `p_unchanged = 0.95`, ~95% of
/// items are bit-identical replicas, providing the "expected dedup
/// rate" ground truth the harness measures against.
fn apply_time_snapshots(
    base_items: Vec<GeneratedBytes>,
    seed: u64,
    snapshots: usize,
    p_unchanged: f32,
    p_modified: f32,
    _p_new: f32,
) -> Vec<GeneratedBytes> {
    debug_assert!(p_unchanged + p_modified <= 1.0);
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let p_unchanged_threshold = (p_unchanged * (u32::MAX as f32)) as u32;
    let p_modified_threshold = ((p_unchanged + p_modified) * (u32::MAX as f32)) as u32;

    let mut out = Vec::with_capacity(base_items.len() * snapshots);
    for base in &base_items {
        let mut current = base.bytes.clone();
        out.push(GeneratedBytes {
            bytes: current.clone(),
            cluster_id: base.cluster_id,
        });
        for _ in 1..snapshots {
            let draw = rng.next_u32();
            if draw < p_unchanged_threshold {
                // Replica — unchanged.
                out.push(GeneratedBytes {
                    bytes: current.clone(),
                    cluster_id: base.cluster_id,
                });
            } else if draw < p_modified_threshold {
                // Modified — flip a small percentage of bits (~1%).
                small_mutate(&mut rng, &mut current);
                out.push(GeneratedBytes {
                    bytes: current.clone(),
                    cluster_id: base.cluster_id,
                });
            } else {
                // "New" — replace with fresh random bytes (loses cluster
                // membership). The remainder probability after p_unchanged
                // + p_modified represents files added or replaced.
                let mut fresh = vec![0u8; current.len()];
                rng.fill_bytes(&mut fresh);
                current = fresh.clone();
                out.push(GeneratedBytes {
                    bytes: fresh,
                    cluster_id: None,
                });
            }
        }
    }
    out
}

/// Apply a small mutation (~1% of bits) to `bytes` in place. Models
/// "file edited" mutations at typical commit / diff scale.
fn small_mutate(rng: &mut ChaCha8Rng, bytes: &mut [u8]) {
    let p = 0.01_f32;
    let threshold = (p * (u32::MAX as f32)) as u32;
    for byte in bytes.iter_mut() {
        for bit in 0..8u8 {
            if rng.next_u32() < threshold {
                *byte ^= 1 << bit;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::bytes::GeneratedBytes;

    fn make_items(n: usize) -> Vec<GeneratedBytes> {
        (0..n)
            .map(|i| GeneratedBytes {
                bytes: vec![i as u8; 16],
                cluster_id: Some(0),
            })
            .collect()
    }

    #[test]
    fn flat_is_identity() {
        let items = make_items(3);
        let out = transform(&StructureLayer::Flat, items.clone(), 42);
        assert_eq!(out.len(), 3);
        for (a, b) in items.iter().zip(out.iter()) {
            assert_eq!(a.bytes, b.bytes);
            assert_eq!(a.cluster_id, b.cluster_id);
        }
    }

    #[test]
    fn time_snapshots_replicates_count() {
        let items = make_items(4);
        let out = transform(
            &StructureLayer::TimeSnapshots {
                snapshots: 3,
                p_unchanged: 1.0, // all replicas, no mutation
                p_modified: 0.0,
                p_new: 0.0,
            },
            items,
            7,
        );
        assert_eq!(out.len(), 12); // 4 × 3
    }

    #[test]
    fn time_snapshots_p_unchanged_one_means_all_replicas() {
        let items = make_items(2);
        let out = transform(
            &StructureLayer::TimeSnapshots {
                snapshots: 5,
                p_unchanged: 1.0,
                p_modified: 0.0,
                p_new: 0.0,
            },
            items.clone(),
            42,
        );
        // 2 × 5 = 10. Each row is bit-identical to its base item.
        assert_eq!(out.len(), 10);
        for (i, snapshot) in out.iter().enumerate() {
            let base = &items[i / 5].bytes;
            assert_eq!(&snapshot.bytes, base);
        }
    }

    #[test]
    fn time_snapshots_high_dedup_at_p_unchanged_095() {
        // At p_unchanged = 0.95 over 12 snapshots, expected unchanged
        // rate is ~95% — the snapshot-dedup story.
        let items = make_items(50); // 50 files
        let out = transform(
            &StructureLayer::TimeSnapshots {
                snapshots: 12,
                p_unchanged: 0.95,
                p_modified: 0.04,
                p_new: 0.01,
            },
            items.clone(),
            123,
        );
        assert_eq!(out.len(), 600); // 50 × 12

        // Count how many items are bit-identical to one of the bases.
        let base_byte_set: std::collections::HashSet<Vec<u8>> =
            items.iter().map(|i| i.bytes.clone()).collect();
        let mut replicas = 0;
        for item in &out {
            if base_byte_set.contains(&item.bytes) {
                replicas += 1;
            }
        }
        // First snapshot of each file is always a replica (50 of them);
        // subsequent snapshots inherit at p_unchanged. Per cluster the
        // chain dedup-survives (~0.95)^11 of the time, but cumulative
        // dedup floor is much higher because many "modified" / "new"
        // items still occur. Floor: at least 50% of items should be
        // base-replicas (very loose; mean ~70%).
        let dedup_rate = replicas as f32 / out.len() as f32;
        assert!(
            dedup_rate >= 0.5,
            "expected dedup rate >= 0.5, got {dedup_rate}"
        );
    }

    #[test]
    fn time_snapshots_deterministic_across_runs() {
        let items = make_items(3);
        let a = transform(
            &StructureLayer::TimeSnapshots {
                snapshots: 4,
                p_unchanged: 0.5,
                p_modified: 0.3,
                p_new: 0.2,
            },
            items.clone(),
            777,
        );
        let b = transform(
            &StructureLayer::TimeSnapshots {
                snapshots: 4,
                p_unchanged: 0.5,
                p_modified: 0.3,
                p_new: 0.2,
            },
            items,
            777,
        );
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.bytes, y.bytes);
        }
    }
}

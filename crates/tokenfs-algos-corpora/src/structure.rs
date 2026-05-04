//! Layer 2 — Structure generators.
//!
//! How items are arranged in the corpus. Phase coverage: **#238**
//! lands the four variants below.

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

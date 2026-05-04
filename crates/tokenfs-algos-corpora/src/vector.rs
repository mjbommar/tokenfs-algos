//! Layer 3 — Vector generators.
//!
//! What fingerprint per item. Phase coverage: **#237** lands the
//! five variants below.

/// One of the five fingerprint shapes.
#[derive(Debug, Clone)]
pub enum VectorLayer {
    /// **Pass-through** of the bytes layer's output. Useful when the
    /// bytes ARE the fingerprint (e.g. clustering-fuzz with binary
    /// vectors — bytes go straight into Hamming distance).
    Identity,
    /// F22 ExtentFingerprint (32 bytes per item) computed via
    /// `tokenfs_algos::fingerprint::extent`. Models use cases 1
    /// (rootfs near-dup), 4 (forensics), 9 (CDN edge).
    F22ExtentFingerprint,
    /// MinHash signature (256 bytes for k=64, etc.) computed via
    /// `tokenfs_algos::similarity::minhash`. Models use cases 2
    /// (monorepo code), 5 (training-data dedup).
    MinHashSignature {
        /// Number of MinHash slots (k). 64 → 256 bytes; 128 → 512 bytes.
        k: usize,
    },
    /// Synthetic f32 embedding with controllable cluster geometry,
    /// anisotropy, and hubness. Generated directly (does NOT
    /// transform the bytes — for use cases 6/8/10 where the input
    /// "is" the embedding). Models use cases 6 (RAG), 8 (Postgres
    /// extension), 10 (audit log anomaly).
    EmbeddingF32 {
        /// Vector dimensionality.
        dim: usize,
        /// Anisotropy strength ∈ [0.0, 1.0]. 0 = isotropic Gaussian;
        /// higher values stretch the distribution along principal axes
        /// (matches real embedding-space behavior).
        anisotropy: f32,
        /// Hub-rate ∈ [0.0, 1.0]. Probability a vector becomes a "hub"
        /// near-neighbor to many others — known HNSW failure mode worth
        /// stressing.
        hub_rate: f32,
    },
    /// i8-quantized embedding. Same shape as `EmbeddingF32` but
    /// post-quantization. Models use case 6 (RAG with i8-quantized
    /// MiniLM) + use case 8 (Postgres ext with i8 column).
    EmbeddingI8 {
        /// Vector dimensionality (in i8 elements; bytes-per-vector == dim).
        dim: usize,
        /// Anisotropy strength.
        anisotropy: f32,
        /// Hub-rate.
        hub_rate: f32,
    },
    /// LSH band-hashes of the bytes layer's output. Models the
    /// pre-filter tier in use case 5 (training-data dedup) and
    /// case 7 (token-aware grep).
    LshBands {
        /// Number of LSH bands.
        bands: usize,
        /// Rows per band.
        rows_per_band: usize,
    },
}

impl VectorLayer {
    /// Bytes per generated vector.
    pub fn bytes_per_vector(&self, bytes_per_item_input: usize) -> usize {
        match self {
            VectorLayer::Identity => bytes_per_item_input,
            VectorLayer::F22ExtentFingerprint => 32, // F22 is fixed 32 bytes
            VectorLayer::MinHashSignature { k } => k * 4, // u32 per slot
            VectorLayer::EmbeddingF32 { dim, .. } => dim * 4,
            VectorLayer::EmbeddingI8 { dim, .. } => *dim,
            VectorLayer::LshBands {
                bands,
                rows_per_band,
            } => bands * rows_per_band, // 1 byte per (band, row)
        }
    }
}

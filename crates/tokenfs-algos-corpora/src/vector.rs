//! Layer 3 — Vector generators.
//!
//! What fingerprint per item.
//!
//! Phase coverage:
//!
//! - **#237 (this commit):** [`VectorLayer::Identity`] (pass-through),
//!   [`VectorLayer::F22ExtentFingerprint`] (calls
//!   `tokenfs_algos::fingerprint::extent`),
//!   [`VectorLayer::EmbeddingF32`] (synthetic clustered f32 vectors),
//!   [`VectorLayer::EmbeddingI8`] (quantized i8 from f32).
//! - **#238:** [`VectorLayer::MinHashSignature`] (calls
//!   `tokenfs_algos::similarity::minhash`),
//!   [`VectorLayer::LshBands`].

use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

use crate::ClusterId;
use crate::bytes::GeneratedBytes;

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
    ///
    /// Note: F22 outputs **20 bytes** (5 × f32 fields) — not 32.
    /// "F22" refers to the fingerprint *family*; the
    /// `ExtentFingerprint` struct in `tokenfs_algos::fingerprint`
    /// has 5 f32 fields totaling 20 bytes. The `bytes_per_item`
    /// computation here is what the corpora vector layer actually
    /// emits.
    pub fn bytes_per_vector(&self, bytes_per_item_input: usize) -> usize {
        match self {
            VectorLayer::Identity => bytes_per_item_input,
            VectorLayer::F22ExtentFingerprint => 20, // 5 × f32 fields
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

/// Generate a vector for one item per the layer spec.
///
/// `gen_bytes` is the bytes-layer output for the item (used by F22,
/// MinHash, LSH which derive their vector from the input). `vec_seed`
/// is item-specific RNG seed (used by EmbeddingF32 / EmbeddingI8 to
/// generate synthetic vectors).
///
/// Returns `Vec<u8>` of length `layer.bytes_per_vector(gen_bytes.bytes.len())`.
pub fn generate_one(layer: &VectorLayer, gen_bytes: &GeneratedBytes, vec_seed: u64) -> Vec<u8> {
    match layer {
        VectorLayer::Identity => gen_bytes.bytes.clone(),
        VectorLayer::F22ExtentFingerprint => f22_serialize(&gen_bytes.bytes),
        VectorLayer::EmbeddingF32 {
            dim,
            anisotropy,
            hub_rate,
        } => generate_embedding_f32(*dim, *anisotropy, *hub_rate, vec_seed, gen_bytes.cluster_id),
        VectorLayer::EmbeddingI8 {
            dim,
            anisotropy,
            hub_rate,
        } => generate_embedding_i8(*dim, *anisotropy, *hub_rate, vec_seed, gen_bytes.cluster_id),
        // MinHash + LshBands land in #238.
        VectorLayer::MinHashSignature { k } => vec![0u8; k * 4],
        VectorLayer::LshBands {
            bands,
            rows_per_band,
        } => vec![0u8; bands * rows_per_band],
    }
}

/// Run `tokenfs_algos::fingerprint::extent` on the bytes and serialize
/// the resulting `ExtentFingerprint` struct into the canonical 20-byte
/// LE float layout: `[h1, h4, rl_fraction, top16_coverage, byte_entropy_skew]`.
fn f22_serialize(bytes: &[u8]) -> Vec<u8> {
    let fp = tokenfs_algos::fingerprint::extent(bytes);
    let mut out = Vec::with_capacity(20);
    for &v in &[
        fp.h1,
        fp.h4,
        fp.rl_fraction,
        fp.top16_coverage,
        fp.byte_entropy_skew,
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Generate a synthetic f32 embedding with controllable cluster
/// geometry, anisotropy, and hub-rate. Per docs/hnsw/components/
/// CLUSTERING_FUZZ.md generalized to embedding-shaped data.
///
/// Approach:
/// 1. Pick a cluster centroid (seeded by cluster_id; identical for
///    every item in the same cluster).
/// 2. Sample a perturbation around the centroid, scaled by anisotropy:
///    first half of dimensions get full noise, second half get
///    `(1 - anisotropy) * noise` (stretches distribution along
///    early axes).
/// 3. With probability `hub_rate`, override the embedding with the
///    "hub" vector (mean of all centroids ≈ origin) — known HNSW
///    failure mode worth stressing.
///
/// Returns `dim * 4` bytes (f32 LE).
fn generate_embedding_f32(
    dim: usize,
    anisotropy: f32,
    hub_rate: f32,
    item_seed: u64,
    cluster_id: Option<ClusterId>,
) -> Vec<u8> {
    let mut rng = ChaCha8Rng::seed_from_u64(item_seed);

    // Hub-rate check first.
    if hub_rate > 0.0 && (rng.next_u32() as f32 / u32::MAX as f32) < hub_rate {
        // Hub vector: all-zero centroid + small noise. Becomes a
        // near-neighbor to many cluster centroids.
        let mut out = Vec::with_capacity(dim * 4);
        for _ in 0..dim {
            let v = sample_normal(&mut rng) * 0.1;
            out.extend_from_slice(&v.to_le_bytes());
        }
        return out;
    }

    // Cluster centroid: seeded deterministically by cluster_id so all
    // items in the same cluster have the same centroid.
    let centroid = match cluster_id {
        Some(cid) => generate_centroid(dim, u64::from(cid).wrapping_mul(0x9E37_79B9)),
        None => vec![0.0_f32; dim], // unclustered → centered at origin
    };

    let mut out = Vec::with_capacity(dim * 4);
    for (i, c) in centroid.iter().enumerate().take(dim) {
        // Anisotropy scaling: first half of dims gets full noise,
        // second half gets reduced noise per anisotropy strength.
        let scale = if i < dim / 2 { 1.0 } else { 1.0 - anisotropy };
        let noise = sample_normal(&mut rng) * 0.3 * scale;
        let v = c + noise;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Generate i8-quantized embedding. Internally generates the f32
/// version then quantizes to i8 via linear scaling to ±127.
fn generate_embedding_i8(
    dim: usize,
    anisotropy: f32,
    hub_rate: f32,
    item_seed: u64,
    cluster_id: Option<ClusterId>,
) -> Vec<u8> {
    let f32_bytes = generate_embedding_f32(dim, anisotropy, hub_rate, item_seed, cluster_id);
    let mut out = Vec::with_capacity(dim);
    for chunk in f32_bytes.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        // Clamp to [-1.0, 1.0] then scale to [-127, 127].
        let clamped = v.clamp(-1.0, 1.0);
        let q = (clamped * 127.0).round() as i8;
        out.push(q as u8);
    }
    out
}

/// Generate a deterministic cluster centroid in [-1, 1]^dim.
fn generate_centroid(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        let raw = rng.next_u32();
        let v = (raw as f32 / u32::MAX as f32) * 2.0 - 1.0;
        out.push(v);
    }
    out
}

/// Box-Muller sample from N(0, 1). Two uniform draws → one normal.
fn sample_normal(rng: &mut ChaCha8Rng) -> f32 {
    let u1 = (rng.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0);
    let u2 = rng.next_u32() as f32 / u32::MAX as f32;
    (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).cos()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn make_bytes(b: &[u8], cid: Option<u32>) -> GeneratedBytes {
        GeneratedBytes {
            bytes: b.to_vec(),
            cluster_id: cid,
        }
    }

    #[test]
    fn identity_passes_bytes_through() {
        let g = make_bytes(&[1, 2, 3, 4, 5], Some(0));
        let v = generate_one(&VectorLayer::Identity, &g, 7);
        assert_eq!(v, g.bytes);
    }

    #[test]
    fn f22_emits_twenty_bytes() {
        let g = make_bytes(&vec![0xABu8; 4096], None);
        let v = generate_one(&VectorLayer::F22ExtentFingerprint, &g, 0);
        assert_eq!(v.len(), 20);
        // Decode the 5 f32s; sanity: h1 should be 0 (single-byte stream).
        let h1 = f32::from_le_bytes(v[0..4].try_into().unwrap());
        assert!(
            h1.abs() < 1e-3,
            "h1 of all-0xAB stream should be 0; got {h1}"
        );
    }

    #[test]
    fn f22_distinguishes_different_inputs() {
        let g1 = make_bytes(&vec![0xABu8; 4096], None);
        let g2 = make_bytes(
            &(0..4096).map(|i| (i % 256) as u8).collect::<Vec<u8>>(),
            None,
        );
        let v1 = generate_one(&VectorLayer::F22ExtentFingerprint, &g1, 0);
        let v2 = generate_one(&VectorLayer::F22ExtentFingerprint, &g2, 0);
        // Two extents with very different content distributions
        // produce different fingerprints.
        assert_ne!(v1, v2);
    }

    #[test]
    fn embedding_f32_emits_dim_times_4_bytes() {
        let g = make_bytes(&[], Some(0));
        let v = generate_one(
            &VectorLayer::EmbeddingF32 {
                dim: 8,
                anisotropy: 0.0,
                hub_rate: 0.0,
            },
            &g,
            42,
        );
        assert_eq!(v.len(), 32); // 8 × 4
    }

    #[test]
    fn embedding_i8_emits_dim_bytes() {
        let g = make_bytes(&[], Some(0));
        let v = generate_one(
            &VectorLayer::EmbeddingI8 {
                dim: 16,
                anisotropy: 0.0,
                hub_rate: 0.0,
            },
            &g,
            42,
        );
        assert_eq!(v.len(), 16);
    }

    #[test]
    fn embedding_f32_clusters_by_centroid() {
        // Two items in cluster 0 should be closer to each other than
        // to an item in cluster 1.
        let g0a = make_bytes(&[], Some(0));
        let g0b = make_bytes(&[], Some(0));
        let g1 = make_bytes(&[], Some(1));

        let v0a = decode_f32_vec(&generate_one(
            &VectorLayer::EmbeddingF32 {
                dim: 32,
                anisotropy: 0.0,
                hub_rate: 0.0,
            },
            &g0a,
            1,
        ));
        let v0b = decode_f32_vec(&generate_one(
            &VectorLayer::EmbeddingF32 {
                dim: 32,
                anisotropy: 0.0,
                hub_rate: 0.0,
            },
            &g0b,
            2, // different vec_seed → different perturbation
        ));
        let v1 = decode_f32_vec(&generate_one(
            &VectorLayer::EmbeddingF32 {
                dim: 32,
                anisotropy: 0.0,
                hub_rate: 0.0,
            },
            &g1,
            3,
        ));

        let intra = l2_distance(&v0a, &v0b);
        let inter = l2_distance(&v0a, &v1);
        assert!(
            intra < inter,
            "intra-cluster distance ({intra}) must be < inter-cluster ({inter})"
        );
    }

    #[test]
    fn embedding_f32_deterministic_across_runs() {
        let g = make_bytes(&[], Some(7));
        let v1 = generate_one(
            &VectorLayer::EmbeddingF32 {
                dim: 16,
                anisotropy: 0.3,
                hub_rate: 0.0,
            },
            &g,
            999,
        );
        let v2 = generate_one(
            &VectorLayer::EmbeddingF32 {
                dim: 16,
                anisotropy: 0.3,
                hub_rate: 0.0,
            },
            &g,
            999,
        );
        assert_eq!(v1, v2);
    }

    fn decode_f32_vec(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

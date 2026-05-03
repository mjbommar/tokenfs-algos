//! Convenient imports for common `tokenfs-algos` APIs.

pub use crate::dispatch::{
    ApiContext, Backend, CacheProfile, CacheState, ContentKind, EntropyClass, EntropyScale,
    HistogramKernelInfo, HistogramPlan, HistogramStrategy, KernelIsa, KernelStatefulness,
    PlanContext, PlannerConfidenceSource, PrimitiveFamily, ProcessorProfile, ReadPattern,
    SourceHint, WorkingSetClass, WorkloadShape, detected_backend, detected_cache_profile,
    detected_processor_profile, histogram_kernel_catalog, plan_histogram,
};
// Process-wide mutable backend override; gated on `bench-internals` so
// kernel-adjacent consumers do not get policy-mutating globals through
// the prelude (audit-R9 #7).
#[cfg(feature = "bench-internals")]
pub use crate::dispatch::{clear_forced_backend, force_backend};
pub use crate::distribution::{
    ByteDistribution, ByteDistributionDistances, ByteDistributionMetric, ByteDistributionReference,
    NearestByteDistribution, nearest_byte_distribution, nearest_reference,
};
pub use crate::entropy;
pub use crate::fingerprint::{
    BLOCK_SIZE as FINGERPRINT_BLOCK_SIZE, BlockFingerprint, ExtentFingerprint, FingerprintKernel,
    FingerprintKernelInfo, block as fingerprint_block, extent as fingerprint_extent,
    kernel_catalog as fingerprint_kernel_catalog,
};
#[cfg(any(feature = "std", feature = "alloc"))]
pub use crate::format::{
    Detection as FormatDetection, Format, Sniffer as FormatSniffer, detect as detect_format,
};
#[cfg(feature = "blake3")]
pub use crate::hash::blake3::{
    BLOCK_BYTES as BLAKE3_BLOCK_BYTES, DIGEST_BYTES as BLAKE3_DIGEST_BYTES, Hasher as Blake3Hasher,
    blake3, blake3_derive_key, blake3_keyed, blake3_xof,
};
pub use crate::hash::sha256::{Hasher as Sha256Hasher, HasherBackend as Sha256HasherBackend};
pub use crate::hash::{fnv1a64, mix64};
pub use crate::histogram::{
    ByteHistogram, BytePairHistogram, BytePairScratch, HistogramBlockSignals, PlannedByteHistogram,
    block as histogram_block, block_with_profile as histogram_block_with_profile,
    explain_block as explain_histogram_block, plan_block as plan_histogram_block,
    summary::{ByteValueMoments, byte_value_moments},
    topk::MisraGries as ByteMisraGries,
};
#[cfg(all(feature = "blake3", any(feature = "std", feature = "alloc")))]
pub use crate::identity::blake3_cid;
pub use crate::identity::{
    MAX_VARINT_LEN, MULTIBASE_BASE32_LOWER_PREFIX, Multicodec, MultihashCode, base32_lower_len,
    build_cid_v1, decode_multihash, decode_varint_u64, encode_base32_lower, encode_multihash,
    encode_varint_u64, varint_u64_len,
};
#[cfg(any(feature = "std", feature = "alloc"))]
pub use crate::identity::{
    build_cid_v1_string, build_cid_v1_vec, encode_multihash_vec, sha256_cid,
};
pub use crate::search::bitap::{Bitap16, Bitap64};
#[cfg(any(feature = "std", feature = "alloc"))]
pub use crate::search::packed_dfa::PackedDfa;
pub use crate::search::packed_pair::PackedPair;
pub use crate::search::rabin_karp::RabinKarp;
pub use crate::search::shift_or::ShiftOr;
pub use crate::search::two_way::TwoWay;
pub use crate::selector::{RepresentationHint, SelectorSignals};
pub use crate::similarity::fuzzy::{ctph, tlsh_like};
pub use crate::similarity::minhash::IncrementalSignature as MinHashIncrementalSignature;
pub use crate::sketch::{
    CLog2Lut, CountMinSketch, Crc32cBackend, Crc32cHasher, HashBinSketch, MisraGries, crc32c_bytes,
    crc32c_u32,
};
pub use crate::structure::StructureSummary;

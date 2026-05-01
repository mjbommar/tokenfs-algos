//! Fuzzy fingerprint primitives — locality-sensitive digests for file-level
//! similarity that survive insertion, deletion, and shift.
//!
//! Per `docs/SIMILARITY_APPROXIMATION_ROADMAP.md` Phase 2. Currently only
//! [`tlsh_like`] is implemented; CTPH (ssdeep-style) and sdhash-style
//! variants are tracked as follow-ups in the SIMD/algorithm roadmap.
//!
//! The naming `tlsh_like` (vs. `tlsh`) is intentional: this is a
//! quality-faithful reimplementation following the published TLSH paper
//! and the Apache-2.0 reference, but produces digests that may diverge
//! slightly from upstream byte-for-byte (different Pearson permutation
//! seed, no header/footer chars). Distances are calibrated against the
//! published "weak < 30, related < 100, unrelated > 150" thresholds.

pub mod tlsh_like;

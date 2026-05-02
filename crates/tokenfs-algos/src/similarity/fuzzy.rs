//! Fuzzy fingerprint primitives — locality-sensitive digests for file-level
//! similarity that survive insertion, deletion, and shift.
//!
//! Per `docs/SIMILARITY_APPROXIMATION_ROADMAP.md` Phase 2. Two families are
//! currently implemented:
//!
//! - [`tlsh_like`] — TLSH-style bucket-quartile digest (35 bytes), tuned
//!   for files of >= 50 bytes with diverse byte content. Distance metric
//!   is integer-valued; published thresholds (`< 30` near-duplicate,
//!   `30..100` related, `> 150` unrelated).
//! - [`ctph`] — Context-Triggered Piecewise Hashing (ssdeep-style), a
//!   variable-length printable digest tuned for "did this file change a
//!   little?" workloads. Distance metric is in `[0, 100]` (`0` identical,
//!   `100` unrelated/incomparable).
//!
//! The naming `tlsh_like` (vs. `tlsh`) and the doc-level note on `ctph`
//! are intentional: both are quality-faithful reimplementations of the
//! published algorithms but make no promise of byte-for-byte parity with
//! the canonical reference implementations. See each module for the
//! specific deviations.

pub mod ctph;
pub mod tlsh_like;

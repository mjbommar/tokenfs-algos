//! Hardware-accelerated byte-stream analysis primitives.
//!
//! `tokenfs-algos` provides low-level, content-agnostic algorithms over byte
//! slices. The public API starts with scalar reference implementations and is
//! structured so SIMD backends can be added without changing callers.
//!
//! ## Cargo feature: `panicking-shape-apis`
//!
//! Several primitive entry points (e.g. [`bits::bit_pack::BitPacker::encode_u32_slice`],
//! [`bits::streamvbyte::streamvbyte_decode_u32`], [`vector::batch::dot_f32_one_to_many`],
//! [`bits::rank_select::RankSelectDict::build`], [`hash::set_membership::contains_u32_batch_simd`],
//! [`hash::sha256_batch_st`], and [`similarity::minhash::signature_batch_simd`])
//! validate caller-supplied shapes (lengths, widths, strides) and panic on
//! mismatch. Each has a fallible `try_*` parallel that returns a typed error
//! instead.
//!
//! For kernel- and FUSE-class consumers (per
//! `docs/v0.2_planning/02b_DEPLOYMENT_MATRIX.md`) the panicking branches
//! must not exist on the public surface — a panic in a kernel softirq is
//! fatal. The on-by-default `panicking-shape-apis` Cargo feature gates the
//! panicking variants behind `#[cfg(feature = "panicking-shape-apis")]`.
//!
//! Userspace consumers get the unchanged behaviour by default. Kernel
//! consumers opt out:
//!
//! ```toml
//! tokenfs-algos = { version = "0.2", default-features = false, features = ["alloc"] }
//! ```
//!
//! Under that build, only the `try_*` wrappers are reachable; calls to the
//! panicking constructors fail to compile. See audit-R5 finding #157.

#![cfg_attr(not(feature = "std"), no_std)]
// AArch64 SVE / SVE2 intrinsics (`core::arch::aarch64::sv*`) are gated
// behind the `stdarch_aarch64_sve` unstable feature today (rust-lang
// tracking issue #145052). Opting in is required for any code that
// touches the SVE/SVE2 module under our `sve` / `sve2` cargo features.
// The umbrella `nightly` feature plus the architecture-specific cargo
// flag keeps the attribute scoped to builds that actually want SVE.
#![cfg_attr(
    all(
        feature = "nightly",
        any(feature = "sve", feature = "sve2"),
        target_arch = "aarch64"
    ),
    feature(stdarch_aarch64_sve)
)]

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

pub mod approx;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod bitmap;
pub mod bits;
pub mod byteclass;
pub mod chunk;
pub mod dispatch;
pub mod distribution;
pub mod divergence;
pub mod entropy;
pub mod error;
pub mod fingerprint;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod format;
pub mod hash;
pub mod histogram;
pub mod identity;
pub mod paper;
#[cfg(any(feature = "std", feature = "alloc"))]
pub mod permutation;
pub mod prelude;
pub mod runlength;
pub mod search;
pub mod selector;
pub mod similarity;
pub mod sketch;
pub mod sketch_p2;
pub mod structure;
pub mod vector;
pub mod windows;

pub(crate) mod math;
pub(crate) mod primitives;

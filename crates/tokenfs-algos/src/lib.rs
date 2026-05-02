//! Hardware-accelerated byte-stream analysis primitives.
//!
//! `tokenfs-algos` provides low-level, content-agnostic algorithms over byte
//! slices. The public API starts with scalar reference implementations and is
//! structured so SIMD backends can be added without changing callers.

#![cfg_attr(not(feature = "std"), no_std)]
// Detection-only nightly features for tile/matrix accelerator probing.
// Both `is_x86_feature_detected!("amx-tile")` and
// `is_aarch64_feature_detected!("sme")` are gated under unstable
// `stdarch_*_feature_detection` features. Pin them here so the
// `dispatch::detect_accelerators` path compiles on the workspace's
// nightly toolchain. No actual AMX/SME intrinsics are invoked.
#![cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature(x86_amx_intrinsics)
)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_feature_detection))]

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

pub mod approx;
pub mod byteclass;
pub mod chunk;
pub mod dispatch;
pub mod distribution;
pub mod divergence;
pub mod entropy;
pub mod error;
pub mod fingerprint;
pub mod hash;
pub mod histogram;
pub mod paper;
pub mod prelude;
pub mod runlength;
pub mod search;
pub mod selector;
pub mod similarity;
pub mod sketch;
pub mod sketch_p2;
pub mod structure;
pub mod windows;

pub(crate) mod math;
pub(crate) mod primitives;

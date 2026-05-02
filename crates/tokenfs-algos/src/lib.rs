//! Hardware-accelerated byte-stream analysis primitives.
//!
//! `tokenfs-algos` provides low-level, content-agnostic algorithms over byte
//! slices. The public API starts with scalar reference implementations and is
//! structured so SIMD backends can be added without changing callers.

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

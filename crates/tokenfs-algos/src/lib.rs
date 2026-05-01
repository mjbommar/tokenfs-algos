//! Hardware-accelerated byte-stream analysis primitives.
//!
//! `tokenfs-algos` provides low-level, content-agnostic algorithms over byte
//! slices. The public API starts with scalar reference implementations and is
//! structured so SIMD backends can be added without changing callers.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod byteclass;
pub mod chunk;
pub mod dispatch;
pub mod divergence;
pub mod entropy;
pub mod error;
pub mod fingerprint;
pub mod histogram;
pub mod paper;
pub mod prelude;
pub mod runlength;
pub mod selector;
pub mod sketch;
pub mod structure;
pub mod windows;

pub(crate) mod primitives;

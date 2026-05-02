//! Bit-level primitive surface.
//!
//! Sprint 1 (v0.2 plan, see `docs/v0.2_planning/10_BITS.md` § 4): the
//! foundation kernel is [`popcount_u64_slice`] / [`popcount_u8_slice`],
//! with runtime-dispatched scalar / AVX2 / AVX-512 / NEON backends.
//!
//! Subsequent v0.2 work in this module: [`bit_pack`] (§ 2),
//! [`streamvbyte`] (§ 3), and `rank_select` (§ 5). They all share
//! `popcount` as the inner kernel.
//!
//! ## Public API
//!
//! * [`popcount_u64_slice`] — bit-count over `&[u64]`.
//! * [`popcount_u8_slice`] — bit-count over `&[u8]`.
//! * [`BitPacker`] / [`DynamicBitPacker`] — pack/unpack 1-32 bit u32 values.
//! * [`streamvbyte_encode_u32`] / [`streamvbyte_decode_u32`] — Stream-VByte
//!   variable-byte codec for `u32` streams.
//!
//! Pinned per-backend kernels live under [`kernels`] for reproducible
//! benchmarks and forensic comparisons.

pub mod bit_pack;
pub mod popcount;
pub mod streamvbyte;

pub use bit_pack::{BitPacker, DynamicBitPacker};
pub use popcount::{popcount_u8_slice, popcount_u64_slice};
pub use streamvbyte::{
    streamvbyte_control_len, streamvbyte_data_max_len, streamvbyte_decode_u32,
    streamvbyte_encode_u32,
};

/// Pinned popcount kernels.
///
/// Re-exports the `kernels` submodule of [`popcount`] so callers can
/// reach the per-backend primitives via the conventional
/// `bits::kernels::scalar::popcount_u64_slice` path used by the rest of
/// the crate.
pub mod kernels {
    pub use crate::bits::popcount::kernels::*;
}

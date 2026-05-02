//! Pinned per-backend kernels for the [`crate::bitmap`] module.
//!
//! Each submodule corresponds to one (container-pair, ISA) combination.
//! The dispatch wrappers in [`crate::bitmap::intersect`] and friends pick
//! the best available kernel at runtime; benchmarks and parity tests
//! address the kernels directly.
//!
//! Modules:
//!
//! * [`bitmap_x_bitmap_scalar`] — portable fallback for AND/OR/XOR/ANDNOT.
//! * [`bitmap_x_bitmap_avx2`] — AVX2 256-bit AND/OR/XOR/ANDNOT.
//! * [`bitmap_x_bitmap_avx512`] — AVX-512 512-bit + VPOPCNTQ.
//! * `bitmap_x_bitmap_neon` — NEON 128-bit AND/OR/XOR/ANDNOT (AArch64 only).
//! * [`array_x_array_scalar`] — merge-based intersect oracle.
//! * [`array_x_array_sse42`] — Schlegel pcmpistrm-based intersect.
//! * [`array_x_bitmap_scalar`] — bit-test loop.
//! * [`array_x_bitmap_avx2`] — bit-test loop with vectorised test/compress.

pub mod array_x_array_scalar;
pub mod array_x_bitmap_scalar;
pub mod bitmap_x_bitmap_scalar;

// SIMD kernels are gated on std because the shuffle-table implementations
// use `std::sync::OnceLock`. AVX2 / AVX-512 / NEON kernels with no
// allocator-bound state could in principle work in alloc-only no_std,
// but mixing the two adds complexity for marginal benefit.

#[cfg(all(
    feature = "std",
    feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub mod array_x_array_sse42;

#[cfg(all(
    feature = "std",
    feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub mod array_x_bitmap_avx2;

#[cfg(all(
    feature = "std",
    feature = "avx2",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub mod bitmap_x_bitmap_avx2;

#[cfg(all(
    feature = "std",
    feature = "avx512",
    any(target_arch = "x86", target_arch = "x86_64")
))]
pub mod bitmap_x_bitmap_avx512;

#[cfg(all(feature = "std", feature = "neon", target_arch = "aarch64"))]
pub mod bitmap_x_bitmap_neon;

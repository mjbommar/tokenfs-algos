//! Internal primitive kernels.

pub(crate) mod histogram_scalar;

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) mod histogram_avx2;

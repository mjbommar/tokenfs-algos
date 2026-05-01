//! Floating-point helpers that work in `std` and `no_std` builds.

#[inline]
pub(crate) fn ln_f64(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log(value)
    }
}

#[inline]
pub(crate) fn log2_f64(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.log2()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log2(value)
    }
}

#[inline]
pub(crate) fn powf_f64(base: f64, exponent: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        base.powf(exponent)
    }
    #[cfg(not(feature = "std"))]
    {
        libm::pow(base, exponent)
    }
}

#[inline]
pub(crate) fn sqrt_f64(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(value)
    }
}

#[inline]
pub(crate) fn round_f32(value: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        value.round()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::roundf(value)
    }
}

//! Error types for fallible algorithms.

use core::fmt;

/// Error type used by fallible algorithms in this crate.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AlgoError {
    /// A caller supplied an invalid parameter.
    InvalidParameter(&'static str),
    /// The requested backend is not available for this target or build.
    UnsupportedBackend,
}

impl fmt::Display for AlgoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter(name) => write!(f, "invalid parameter: {name}"),
            Self::UnsupportedBackend => f.write_str("unsupported backend"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for AlgoError {}

/// Crate-local result type.
pub type Result<T> = core::result::Result<T, AlgoError>;

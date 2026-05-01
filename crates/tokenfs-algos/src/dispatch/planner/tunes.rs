//! Override-able planner tunes.
//!
//! `Tunes` is a value type holding every numeric value the planner reads:
//! byte thresholds, sample sizes, confidence quanta. It defaults to the
//! compile-time constants in [`super::consts`]; callers can override
//! individual fields per-host via [`Tunes::with`] or load a JSON override
//! file via [`Tunes::from_json`] (when the `tunes-json` feature is on).
//!
//! ## Why a struct, not just constants
//!
//! Constants are great for the v0 case where every host shares the same
//! tune table, but they prevent host-specific calibration: a CPU with a
//! 256 KiB L2 cache and a CPU with 1 MiB L2 might want different values
//! for `BLOCK_THRESHOLD_LARGE_BYTES`. The `Tunes` struct lets `bench-calibrate`
//! emit a JSON override file per `(crate version, rustc, backend, cpu_model,
//! cache_profile)` tuple — see `docs/AUTOTUNING_AND_BENCH_HISTORY.md` —
//! and lets the planner pick the right table at startup.
//!
//! ## API contract
//!
//! - [`Tunes::DEFAULT`] is the compile-time default; identical to the
//!   constants in [`super::consts`]. Equality with the const-based
//!   defaults is asserted by `tunes_default_equals_consts` in tests.
//! - [`Tunes::with`] returns a new `Tunes` with one field replaced.
//!   Chainable builder.
//! - [`Tunes::from_json`] (feature `tunes-json`) parses a flat JSON object
//!   on top of `DEFAULT`; missing keys keep their default value, unknown
//!   keys are an error.
//! - The [`super::rules::plan_histogram`] entry point uses
//!   `Tunes::DEFAULT`. The new [`super::rules::plan_histogram_tuned`]
//!   takes `&Tunes` for host-specific overrides.

use super::consts;

/// Override table for planner-tunable values.
///
/// Construct via [`Tunes::DEFAULT`] for the compile-time defaults, then
/// override individual fields with [`Tunes::with`]. JSON loading via
/// [`Tunes::from_json`] requires the `tunes-json` feature.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Tunes {
    // ---- byte thresholds ----
    /// See [`super::consts::BLOCK_THRESHOLD_MICRO_BYTES`].
    pub block_threshold_micro_bytes: usize,
    /// See [`super::consts::BLOCK_THRESHOLD_AVX2_PALETTE_MICRO_BINARY_BYTES`].
    pub block_threshold_avx2_palette_micro_binary_bytes: usize,
    /// See [`super::consts::BLOCK_THRESHOLD_HIGH_ENTROPY_DIRECT_CEILING_BYTES`].
    pub block_threshold_high_entropy_direct_ceiling_bytes: usize,
    /// See [`super::consts::BLOCK_THRESHOLD_LARGE_BYTES`].
    pub block_threshold_large_bytes: usize,
    /// See [`super::consts::BLOCK_THRESHOLD_STRUCTURED_FLOOR_BYTES`].
    pub block_threshold_structured_floor_bytes: usize,
    /// See [`super::consts::TOTAL_THRESHOLD_MACRO_BYTES`].
    pub total_threshold_macro_bytes: usize,
    /// See [`super::consts::TOTAL_THRESHOLD_HIGH_ENTROPY_LOCAL_CEILING_BYTES`].
    pub total_threshold_high_entropy_local_ceiling_bytes: usize,
    /// See [`super::consts::TOTAL_THRESHOLD_ASCII_FAST_BYTES`].
    pub total_threshold_ascii_fast_bytes: usize,
    /// See [`super::consts::TOTAL_THRESHOLD_FILE_CACHE_BYTES`].
    pub total_threshold_file_cache_bytes: usize,

    // ---- chunk and sample sizes ----
    /// See [`super::consts::CHUNK_PARALLEL_DEFAULT_BYTES`].
    pub chunk_parallel_default_bytes: usize,
    /// See [`super::consts::SAMPLE_PREFIX_DEFAULT_BYTES`].
    pub sample_prefix_default_bytes: usize,
    /// See [`super::consts::SAMPLE_ADAPTIVE_DEFAULT_BYTES`].
    pub sample_adaptive_default_bytes: usize,

    // ---- pattern thresholds ----
    /// See [`super::consts::ALIGNMENT_PENALTY_OFFSET_BYTES`].
    pub alignment_penalty_offset_bytes: usize,
    /// See [`super::consts::PARALLEL_STRIPE4_THREAD_FLOOR`].
    pub parallel_stripe4_thread_floor: usize,

    // ---- confidence bands (q8 0..=255) ----
    /// See [`super::consts::CONFIDENCE_FALLBACK_FLOOR`].
    pub confidence_fallback_floor: u8,
    /// See [`super::consts::CONFIDENCE_DETERMINISTIC`].
    pub confidence_deterministic: u8,
    /// See [`super::consts::CONFIDENCE_HIGH_CALIBRATED`].
    pub confidence_high_calibrated: u8,
    /// See [`super::consts::CONFIDENCE_CALIBRATED`].
    pub confidence_calibrated: u8,
    /// See [`super::consts::CONFIDENCE_CALIBRATED_BOUNDARY`].
    pub confidence_calibrated_boundary: u8,
    /// See [`super::consts::CONFIDENCE_CACHE_AMORTIZED`].
    pub confidence_cache_amortized: u8,
    /// See [`super::consts::CONFIDENCE_CALIBRATED_NORMAL`].
    pub confidence_calibrated_normal: u8,
    /// See [`super::consts::CONFIDENCE_RULE_NORMAL`].
    pub confidence_rule_normal: u8,
    /// See [`super::consts::CONFIDENCE_TEXT_PROBE`].
    pub confidence_text_probe: u8,
    /// See [`super::consts::CONFIDENCE_RULE_LOWER`].
    pub confidence_rule_lower: u8,
    /// See [`super::consts::CONFIDENCE_RULE_LOW`].
    pub confidence_rule_low: u8,
    /// See [`super::consts::CONFIDENCE_REAL_FILE`].
    pub confidence_real_file: u8,
    /// See [`super::consts::CONFIDENCE_TENTATIVE`].
    pub confidence_tentative: u8,
    /// See [`super::consts::CONFIDENCE_SEQUENTIAL_PROBE`].
    pub confidence_sequential_probe: u8,
    /// See [`super::consts::CONFIDENCE_GENERAL_FALLBACK`].
    pub confidence_general_fallback: u8,
    /// See [`super::consts::CONFIDENCE_PARALLEL_OVERSUBSCRIBED`].
    pub confidence_parallel_oversubscribed: u8,
    /// See [`super::consts::CONFIDENCE_PARALLEL_OVERSUBSCRIBED_TENTATIVE`].
    pub confidence_parallel_oversubscribed_tentative: u8,
}

impl Tunes {
    /// Default tunes drawn from the compile-time constants in
    /// [`super::consts`].
    pub const DEFAULT: Self = Self {
        block_threshold_micro_bytes: consts::BLOCK_THRESHOLD_MICRO_BYTES,
        block_threshold_avx2_palette_micro_binary_bytes:
            consts::BLOCK_THRESHOLD_AVX2_PALETTE_MICRO_BINARY_BYTES,
        block_threshold_high_entropy_direct_ceiling_bytes:
            consts::BLOCK_THRESHOLD_HIGH_ENTROPY_DIRECT_CEILING_BYTES,
        block_threshold_large_bytes: consts::BLOCK_THRESHOLD_LARGE_BYTES,
        block_threshold_structured_floor_bytes: consts::BLOCK_THRESHOLD_STRUCTURED_FLOOR_BYTES,
        total_threshold_macro_bytes: consts::TOTAL_THRESHOLD_MACRO_BYTES,
        total_threshold_high_entropy_local_ceiling_bytes:
            consts::TOTAL_THRESHOLD_HIGH_ENTROPY_LOCAL_CEILING_BYTES,
        total_threshold_ascii_fast_bytes: consts::TOTAL_THRESHOLD_ASCII_FAST_BYTES,
        total_threshold_file_cache_bytes: consts::TOTAL_THRESHOLD_FILE_CACHE_BYTES,
        chunk_parallel_default_bytes: consts::CHUNK_PARALLEL_DEFAULT_BYTES,
        sample_prefix_default_bytes: consts::SAMPLE_PREFIX_DEFAULT_BYTES,
        sample_adaptive_default_bytes: consts::SAMPLE_ADAPTIVE_DEFAULT_BYTES,
        alignment_penalty_offset_bytes: consts::ALIGNMENT_PENALTY_OFFSET_BYTES,
        parallel_stripe4_thread_floor: consts::PARALLEL_STRIPE4_THREAD_FLOOR,
        confidence_fallback_floor: consts::CONFIDENCE_FALLBACK_FLOOR,
        confidence_deterministic: consts::CONFIDENCE_DETERMINISTIC,
        confidence_high_calibrated: consts::CONFIDENCE_HIGH_CALIBRATED,
        confidence_calibrated: consts::CONFIDENCE_CALIBRATED,
        confidence_calibrated_boundary: consts::CONFIDENCE_CALIBRATED_BOUNDARY,
        confidence_cache_amortized: consts::CONFIDENCE_CACHE_AMORTIZED,
        confidence_calibrated_normal: consts::CONFIDENCE_CALIBRATED_NORMAL,
        confidence_rule_normal: consts::CONFIDENCE_RULE_NORMAL,
        confidence_text_probe: consts::CONFIDENCE_TEXT_PROBE,
        confidence_rule_lower: consts::CONFIDENCE_RULE_LOWER,
        confidence_rule_low: consts::CONFIDENCE_RULE_LOW,
        confidence_real_file: consts::CONFIDENCE_REAL_FILE,
        confidence_tentative: consts::CONFIDENCE_TENTATIVE,
        confidence_sequential_probe: consts::CONFIDENCE_SEQUENTIAL_PROBE,
        confidence_general_fallback: consts::CONFIDENCE_GENERAL_FALLBACK,
        confidence_parallel_oversubscribed: consts::CONFIDENCE_PARALLEL_OVERSUBSCRIBED,
        confidence_parallel_oversubscribed_tentative:
            consts::CONFIDENCE_PARALLEL_OVERSUBSCRIBED_TENTATIVE,
    };

    /// Returns the compile-time default tunes.
    #[must_use]
    pub const fn default_const() -> &'static Self {
        &Self::DEFAULT
    }

    /// Loads tunes overrides from a flat JSON object.
    ///
    /// The JSON schema is `{ "field_name": value, ... }` where field names
    /// are exact matches for [`Tunes`] fields and values are unsigned
    /// integers (`usize` for byte thresholds; `u8` for confidence values).
    /// Missing fields keep their [`Tunes::DEFAULT`] value. Unknown fields
    /// produce a [`TuneLoadError::UnknownField`].
    ///
    /// # Errors
    ///
    /// Returns [`TuneLoadError`] when the JSON is malformed, references an
    /// unknown field, supplies a value outside the field's range, or
    /// supplies a non-integer value for a numeric field.
    #[cfg(feature = "tunes-json")]
    pub fn from_json(json: &str) -> Result<Self, TuneLoadError> {
        json_loader::from_json(json)
    }

    /// Returns a new `Tunes` with the same defaults as `self` but with
    /// `block_threshold_micro_bytes` overridden. Chainable.
    #[must_use]
    pub const fn with_block_threshold_micro_bytes(mut self, value: usize) -> Self {
        self.block_threshold_micro_bytes = value;
        self
    }

    /// Override `block_threshold_large_bytes`.
    #[must_use]
    pub const fn with_block_threshold_large_bytes(mut self, value: usize) -> Self {
        self.block_threshold_large_bytes = value;
        self
    }

    /// Override `total_threshold_macro_bytes`.
    #[must_use]
    pub const fn with_total_threshold_macro_bytes(mut self, value: usize) -> Self {
        self.total_threshold_macro_bytes = value;
        self
    }

    /// Override `confidence_fallback_floor`.
    #[must_use]
    pub const fn with_confidence_fallback_floor(mut self, value: u8) -> Self {
        self.confidence_fallback_floor = value;
        self
    }
}

impl Default for Tunes {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Error from JSON tune loading.
#[cfg(feature = "tunes-json")]
#[derive(Debug)]
pub enum TuneLoadError {
    /// The JSON could not be parsed.
    Parse(serde_json::Error),
    /// The JSON top-level value was not an object.
    NotAnObject,
    /// An unknown field name appeared in the JSON.
    UnknownField(String),
    /// A field was supplied with a non-integer value.
    NotAnInteger(&'static str),
    /// A `u8` confidence field was supplied a value outside `0..=255`.
    OutOfRange {
        /// Name of the offending tune field.
        field: &'static str,
        /// Out-of-range value supplied by the caller.
        value: i64,
    },
}

#[cfg(feature = "tunes-json")]
impl core::fmt::Display for TuneLoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Parse(err) => write!(f, "tune-json parse error: {err}"),
            Self::NotAnObject => write!(f, "tune-json must be a JSON object"),
            Self::UnknownField(name) => write!(f, "tune-json unknown field: {name}"),
            Self::NotAnInteger(field) => {
                write!(f, "tune-json field `{field}` must be an integer")
            }
            Self::OutOfRange { field, value } => {
                write!(
                    f,
                    "tune-json field `{field}` value {value} is outside u8 range 0..=255"
                )
            }
        }
    }
}

#[cfg(all(feature = "tunes-json", feature = "std"))]
impl std::error::Error for TuneLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(err) => Some(err),
            _ => None,
        }
    }
}

#[cfg(feature = "tunes-json")]
mod json_loader {
    use super::{TuneLoadError, Tunes};

    pub(super) fn from_json(json: &str) -> Result<Tunes, TuneLoadError> {
        let value: serde_json::Value = serde_json::from_str(json).map_err(TuneLoadError::Parse)?;
        let serde_json::Value::Object(map) = value else {
            return Err(TuneLoadError::NotAnObject);
        };
        let mut tunes = Tunes::DEFAULT;
        for (key, value) in map {
            apply_field(&mut tunes, key.as_str(), &value)?;
        }
        Ok(tunes)
    }

    fn apply_field(
        tunes: &mut Tunes,
        field: &str,
        value: &serde_json::Value,
    ) -> Result<(), TuneLoadError> {
        macro_rules! u_field {
            ($name:ident, $field:literal) => {
                if field == $field {
                    let n = as_u64(value, $field)?;
                    tunes.$name = n as usize;
                    return Ok(());
                }
            };
        }
        macro_rules! u8_field {
            ($name:ident, $field:literal) => {
                if field == $field {
                    let n = as_u64(value, $field)?;
                    if n > u8::MAX as u64 {
                        return Err(TuneLoadError::OutOfRange {
                            field: $field,
                            value: n as i64,
                        });
                    }
                    tunes.$name = n as u8;
                    return Ok(());
                }
            };
        }

        u_field!(block_threshold_micro_bytes, "block_threshold_micro_bytes");
        u_field!(
            block_threshold_avx2_palette_micro_binary_bytes,
            "block_threshold_avx2_palette_micro_binary_bytes"
        );
        u_field!(
            block_threshold_high_entropy_direct_ceiling_bytes,
            "block_threshold_high_entropy_direct_ceiling_bytes"
        );
        u_field!(block_threshold_large_bytes, "block_threshold_large_bytes");
        u_field!(
            block_threshold_structured_floor_bytes,
            "block_threshold_structured_floor_bytes"
        );
        u_field!(total_threshold_macro_bytes, "total_threshold_macro_bytes");
        u_field!(
            total_threshold_high_entropy_local_ceiling_bytes,
            "total_threshold_high_entropy_local_ceiling_bytes"
        );
        u_field!(
            total_threshold_ascii_fast_bytes,
            "total_threshold_ascii_fast_bytes"
        );
        u_field!(
            total_threshold_file_cache_bytes,
            "total_threshold_file_cache_bytes"
        );
        u_field!(chunk_parallel_default_bytes, "chunk_parallel_default_bytes");
        u_field!(sample_prefix_default_bytes, "sample_prefix_default_bytes");
        u_field!(
            sample_adaptive_default_bytes,
            "sample_adaptive_default_bytes"
        );
        u_field!(
            alignment_penalty_offset_bytes,
            "alignment_penalty_offset_bytes"
        );
        u_field!(
            parallel_stripe4_thread_floor,
            "parallel_stripe4_thread_floor"
        );

        u8_field!(confidence_fallback_floor, "confidence_fallback_floor");
        u8_field!(confidence_deterministic, "confidence_deterministic");
        u8_field!(confidence_high_calibrated, "confidence_high_calibrated");
        u8_field!(confidence_calibrated, "confidence_calibrated");
        u8_field!(
            confidence_calibrated_boundary,
            "confidence_calibrated_boundary"
        );
        u8_field!(confidence_cache_amortized, "confidence_cache_amortized");
        u8_field!(confidence_calibrated_normal, "confidence_calibrated_normal");
        u8_field!(confidence_rule_normal, "confidence_rule_normal");
        u8_field!(confidence_text_probe, "confidence_text_probe");
        u8_field!(confidence_rule_lower, "confidence_rule_lower");
        u8_field!(confidence_rule_low, "confidence_rule_low");
        u8_field!(confidence_real_file, "confidence_real_file");
        u8_field!(confidence_tentative, "confidence_tentative");
        u8_field!(confidence_sequential_probe, "confidence_sequential_probe");
        u8_field!(confidence_general_fallback, "confidence_general_fallback");
        u8_field!(
            confidence_parallel_oversubscribed,
            "confidence_parallel_oversubscribed"
        );
        u8_field!(
            confidence_parallel_oversubscribed_tentative,
            "confidence_parallel_oversubscribed_tentative"
        );

        Err(TuneLoadError::UnknownField(field.to_string()))
    }

    fn as_u64(value: &serde_json::Value, field: &'static str) -> Result<u64, TuneLoadError> {
        value.as_u64().ok_or(TuneLoadError::NotAnInteger(field))
    }
}

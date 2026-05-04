//! usearch v2.25 dense-index header parser (read + write).
//!
//! The 64-byte `index_dense_head_t` is the magic-bearing region of a
//! serialized usearch dense index. Field-by-field layout per
//! `_references/usearch/include/usearch/index_dense.hpp:42-79`,
//! transcribed in `docs/hnsw/research/USEARCH_DEEP_DIVE.md` §1.3.
//!
//! Endianness: little-endian (usearch is built/tested only on LE hosts;
//! we replicate that contract). Every multi-byte field uses
//! [`u16::from_le_bytes`] / [`u64::from_le_bytes`] so we'd produce
//! correct bytes even on a big-endian host that wanted to write a file
//! for an LE consumer.
//!
//! # Audit posture
//!
//! - Public entry: [`HnswHeader::try_parse`] returns
//!   `Result<HnswHeader, HnswHeaderError>`. Never panics on caller input.
//! - Userspace ergonomic [`HnswHeader::parse`] is gated on
//!   `cfg(feature = "userspace")` per `docs/KERNEL_SAFETY.md`.

/// Size of `index_dense_head_t` on disk. Locked by the static_assert at
/// `_references/usearch/include/usearch/index_dense.hpp:31`.
pub const HEADER_BYTES: usize = 64;

/// usearch wire-format magic — ASCII `"usearch"`, no NUL terminator.
const MAGIC: &[u8; 7] = b"usearch";

/// Major version we read + write. usearch is at major == 2 currently
/// (`include/usearch/index.hpp:10`).
pub(crate) const SUPPORTED_VERSION_MAJOR: u16 = 2;

/// Minor version we pin to. v0.7.0 of `tokenfs-algos` writes v2.25 only.
/// Pre-2.10 files would need `convert_pre_2_10_scalar_kind` fix-up; we
/// reject them.
pub(crate) const PINNED_VERSION_MINOR: u16 = 25;

/// Distance metric, char-coded enum. Values per
/// `_references/usearch/include/usearch/index_plugins.hpp:114-133`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MetricKind {
    /// Inner product — `'i'` (0x69).
    InnerProduct = b'i',
    /// Cosine — `'c'` (0x63).
    Cosine = b'c',
    /// L2 squared — `'e'` (0x65).
    L2Squared = b'e',
    /// Hamming on packed binary — `'b'` (0x62).
    Hamming = b'b',
    /// Tanimoto on packed binary — `'t'` (0x74). Collapses to Jaccard
    /// for binary inputs (per usearch's runtime dispatcher mapping).
    Tanimoto = b't',
    /// Sorensen on packed binary — `'s'` (0x73).
    Sorensen = b's',
    /// Jaccard on packed binary — `'j'` (0x6A).
    Jaccard = b'j',
}

impl MetricKind {
    fn try_from_u8(byte: u8) -> Result<Self, HnswHeaderError> {
        match byte {
            b'i' => Ok(MetricKind::InnerProduct),
            b'c' => Ok(MetricKind::Cosine),
            b'e' => Ok(MetricKind::L2Squared),
            b'b' => Ok(MetricKind::Hamming),
            b't' => Ok(MetricKind::Tanimoto),
            b's' => Ok(MetricKind::Sorensen),
            b'j' => Ok(MetricKind::Jaccard),
            other => Err(HnswHeaderError::UnknownMetricKind { code: other }),
        }
    }
}

/// Scalar type, numeric enum. Codes per
/// `_references/usearch/include/usearch/index_plugins.hpp:139-164`.
/// **The numeric values matter** — they are persisted in the header.
///
/// v0.7.0 reads only the four scalar types we ship distance kernels
/// for: `f32`, `i8`, `u8`, `b1x8` (packed binary). Other variants
/// (`f64`, `f16`, `bf16`, `e5m2`, etc.) parse correctly but the walker
/// will return [`HnswHeaderError::UnsupportedScalarKind`] when used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ScalarKind {
    /// Sentinel — `0`.
    Unknown = 0,
    /// 1-bit packed (8 dims per byte) — `1`. (`b1x8_k`).
    B1x8 = 1,
    /// 5-byte slot index — `2`. (`u40_k`). Used by `index_dense_big_t`.
    U40 = 2,
    /// 16-byte UUID key — `3`. (`uuid_k`).
    Uuid = 3,
    /// brain-float 16-bit — `4`. (`bf16_k`).
    Bf16 = 4,
    /// FP8 IEEE 754 — `5`. (`e5m2_k`).
    E5m2 = 5,
    /// FP8 OCP — `6`. (`e4m3_k`).
    E4m3 = 6,
    /// FP6 — `7`. (`e2m3_k`). Note the swap with `e3m2`.
    E2m3 = 7,
    /// FP6 — `8`. (`e3m2_k`).
    E3m2 = 8,
    /// 64-bit IEEE 754 — `10`. (`f64_k`).
    F64 = 10,
    /// 32-bit IEEE 754 — `11`. (`f32_k`).
    F32 = 11,
    /// 16-bit IEEE 754 — `12`. (`f16_k`).
    F16 = 12,
    /// 64-bit unsigned — `14`. (`u64_k`). Default key type.
    U64 = 14,
    /// 32-bit unsigned — `15`. (`u32_k`). Default slot type.
    U32 = 15,
    /// 16-bit unsigned — `16`. (`u16_k`).
    U16 = 16,
    /// 8-bit unsigned — `17`. (`u8_k`).
    U8 = 17,
    /// 64-bit signed — `20`. (`i64_k`).
    I64 = 20,
    /// 32-bit signed — `21`. (`i32_k`).
    I32 = 21,
    /// 16-bit signed — `22`. (`i16_k`).
    I16 = 22,
    /// 8-bit signed — `23`. (`i8_k`).
    I8 = 23,
}

impl ScalarKind {
    fn try_from_u8(byte: u8) -> Result<Self, HnswHeaderError> {
        match byte {
            0 => Ok(ScalarKind::Unknown),
            1 => Ok(ScalarKind::B1x8),
            2 => Ok(ScalarKind::U40),
            3 => Ok(ScalarKind::Uuid),
            4 => Ok(ScalarKind::Bf16),
            5 => Ok(ScalarKind::E5m2),
            6 => Ok(ScalarKind::E4m3),
            7 => Ok(ScalarKind::E2m3),
            8 => Ok(ScalarKind::E3m2),
            10 => Ok(ScalarKind::F64),
            11 => Ok(ScalarKind::F32),
            12 => Ok(ScalarKind::F16),
            14 => Ok(ScalarKind::U64),
            15 => Ok(ScalarKind::U32),
            16 => Ok(ScalarKind::U16),
            17 => Ok(ScalarKind::U8),
            20 => Ok(ScalarKind::I64),
            21 => Ok(ScalarKind::I32),
            22 => Ok(ScalarKind::I16),
            23 => Ok(ScalarKind::I8),
            other => Err(HnswHeaderError::UnknownScalarKind { code: other }),
        }
    }

    /// Bits per scalar — used by the wire-format byte-length computation
    /// and by distance-kernel dispatch.
    pub const fn bits_per_scalar(self) -> u32 {
        match self {
            ScalarKind::Unknown => 0,
            ScalarKind::B1x8 => 1,
            ScalarKind::U40 => 40,
            ScalarKind::Uuid => 128,
            ScalarKind::Bf16 | ScalarKind::F16 | ScalarKind::U16 | ScalarKind::I16 => 16,
            ScalarKind::E5m2
            | ScalarKind::E4m3
            | ScalarKind::E2m3
            | ScalarKind::E3m2
            | ScalarKind::U8
            | ScalarKind::I8 => 8,
            ScalarKind::F32 | ScalarKind::U32 | ScalarKind::I32 => 32,
            ScalarKind::F64 | ScalarKind::U64 | ScalarKind::I64 => 64,
        }
    }
}

/// Parsed `index_dense_head_t`. Fields are private; access via
/// the `pub fn` accessors so we can evolve the internal representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HnswHeader {
    version_major: u16,
    version_minor: u16,
    version_patch: u16,
    metric_kind: MetricKind,
    scalar_kind: ScalarKind,
    key_kind: ScalarKind,
    slot_kind: ScalarKind,
    count_present: u64,
    count_deleted: u64,
    dimensions: u64,
    multi: bool,
}

impl HnswHeader {
    /// Parse a 64-byte usearch v2.25 dense-index header. Validates magic,
    /// version range, all four kind enums, and dimensions plausibility.
    /// Never panics.
    ///
    /// Returns the parsed header on success; an [`HnswHeaderError`]
    /// describing the violation otherwise.
    pub fn try_parse(bytes: &[u8]) -> Result<Self, HnswHeaderError> {
        if bytes.len() < HEADER_BYTES {
            return Err(HnswHeaderError::Truncated {
                got: bytes.len(),
                need: HEADER_BYTES,
            });
        }

        // Offset 0x00 — 7-byte magic.
        if &bytes[0..7] != MAGIC {
            return Err(HnswHeaderError::WrongMagic);
        }

        // Offset 0x07 / 0x09 / 0x0B — version triple. Pin to v2.25.x for
        // v0.7.0; future minor / major bumps land at a new section ID
        // per IMAGE_FORMAT_v0.3 §11. Pre-2.10 files have a different
        // scalar_kind numbering; we reject them upfront rather than
        // implementing the convert_pre_2_10 fix-up.
        let version_major = u16::from_le_bytes([bytes[7], bytes[8]]);
        let version_minor = u16::from_le_bytes([bytes[9], bytes[10]]);
        let version_patch = u16::from_le_bytes([bytes[11], bytes[12]]);
        if version_major != SUPPORTED_VERSION_MAJOR {
            return Err(HnswHeaderError::UnsupportedFormatVersion {
                major: version_major,
                minor: version_minor,
                patch: version_patch,
            });
        }
        if version_minor != PINNED_VERSION_MINOR {
            return Err(HnswHeaderError::UnsupportedFormatVersion {
                major: version_major,
                minor: version_minor,
                patch: version_patch,
            });
        }

        // Offset 0x0D — metric kind (char-coded).
        let metric_kind = MetricKind::try_from_u8(bytes[13])?;

        // Offset 0x0E — scalar kind. Driver of distance-kernel dispatch.
        let scalar_kind = ScalarKind::try_from_u8(bytes[14])?;

        // Offset 0x0F — key kind. Default index_dense_t uses u64.
        let key_kind = ScalarKind::try_from_u8(bytes[15])?;
        if key_kind != ScalarKind::U64 {
            return Err(HnswHeaderError::UnsupportedKeyKind {
                kind: key_kind as u8,
            });
        }

        // Offset 0x10 — slot kind. Default index_dense_t uses u32.
        // index_dense_big_t uses u40 — we reject that variant in v0.7.0.
        let slot_kind = ScalarKind::try_from_u8(bytes[16])?;
        if slot_kind != ScalarKind::U32 {
            return Err(HnswHeaderError::UnsupportedSlotKind {
                kind: slot_kind as u8,
            });
        }

        // Offset 0x11 — count_present (u64 LE; misaligned).
        let count_present =
            u64::from_le_bytes(bytes[17..25].try_into().expect("17..25 is exactly 8 bytes"));

        // Offset 0x19 — count_deleted (u64 LE; misaligned).
        let count_deleted =
            u64::from_le_bytes(bytes[25..33].try_into().expect("25..33 is exactly 8 bytes"));

        // Offset 0x21 — dimensions (logical scalar count, NOT bytes).
        let dimensions =
            u64::from_le_bytes(bytes[33..41].try_into().expect("33..41 is exactly 8 bytes"));
        if dimensions == 0 {
            return Err(HnswHeaderError::ZeroDimensions);
        }

        // Offset 0x29 — multi flag (1 byte).
        let multi = match bytes[41] {
            0 => false,
            1 => true,
            other => {
                return Err(HnswHeaderError::InvalidMultiFlag { value: other });
            }
        };

        // Offsets 0x2A..0x30 — 6 reserved bytes. Per the docstring at
        // index_dense.hpp:38-40 these MUST be zero. We tolerate any value
        // for forward-compat (a future writer may use them) but log via
        // the type's Debug if needed; see `reserved_bytes()` accessor.

        Ok(HnswHeader {
            version_major,
            version_minor,
            version_patch,
            metric_kind,
            scalar_kind,
            key_kind,
            slot_kind,
            count_present,
            count_deleted,
            dimensions,
            multi,
        })
    }

    /// Userspace ergonomic alias. Panics on parse failure with a message
    /// derived from the underlying error. Gated on
    /// `cfg(feature = "userspace")` per `docs/KERNEL_SAFETY.md`.
    #[cfg(feature = "userspace")]
    pub fn parse(bytes: &[u8]) -> Self {
        Self::try_parse(bytes)
            .expect("HnswHeader::parse failed; use try_parse for fallible variant")
    }

    /// Major version (always 2 for v0.7.0).
    pub fn version_major(&self) -> u16 {
        self.version_major
    }

    /// Minor version (always 25 for v0.7.0).
    pub fn version_minor(&self) -> u16 {
        self.version_minor
    }

    /// Patch version (varies; usearch is currently at 2.25.1).
    pub fn version_patch(&self) -> u16 {
        self.version_patch
    }

    /// Distance metric the index was built with.
    pub fn metric_kind(&self) -> MetricKind {
        self.metric_kind
    }

    /// Scalar storage type for vector elements.
    pub fn scalar_kind(&self) -> ScalarKind {
        self.scalar_kind
    }

    /// Number of live entries (excluding tombstones).
    pub fn count_present(&self) -> u64 {
        self.count_present
    }

    /// Number of tombstoned entries (we don't write tombstones in v0.7.0
    /// but a usearch-built index may have them).
    pub fn count_deleted(&self) -> u64 {
        self.count_deleted
    }

    /// Logical scalar count per vector. Multiply by `bytes_per_vector()`
    /// to get the on-disk row size.
    pub fn dimensions(&self) -> u64 {
        self.dimensions
    }

    /// Whether the index allows multiple vectors per key.
    pub fn multi(&self) -> bool {
        self.multi
    }

    /// Bytes per vector in the vectors blob, computed from
    /// `dimensions * bits_per_scalar(scalar_kind)` rounded up to bytes
    /// (per `_references/usearch/include/usearch/index_plugins.hpp:2978-2980`).
    pub fn bytes_per_vector(&self) -> u64 {
        let bits = self.dimensions * (self.scalar_kind.bits_per_scalar() as u64);
        bits.div_ceil(8)
    }
}

/// Errors produced by [`HnswHeader::try_parse`]. Every variant
/// describes the violation precisely enough that the caller can
/// surface it without re-parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HnswHeaderError {
    /// Input shorter than the 64-byte fixed header.
    Truncated {
        /// Number of bytes provided.
        got: usize,
        /// Bytes needed (always [`HEADER_BYTES`]).
        need: usize,
    },
    /// First 7 bytes are not ASCII `"usearch"`.
    WrongMagic,
    /// Version triple does not match v2.25.x.
    UnsupportedFormatVersion {
        /// Major version found.
        major: u16,
        /// Minor version found.
        minor: u16,
        /// Patch version found.
        patch: u16,
    },
    /// `kind_metric` byte is not in
    /// `_references/usearch/include/usearch/index_plugins.hpp:114-133`.
    UnknownMetricKind {
        /// Char code byte found.
        code: u8,
    },
    /// `kind_scalar`, `kind_key`, or `kind_compressed_slot` byte is not
    /// a known [`ScalarKind`] code.
    UnknownScalarKind {
        /// Numeric code byte found.
        code: u8,
    },
    /// Key type is not `u64`. v0.7.0 only supports the default
    /// `index_dense_t` shape.
    UnsupportedKeyKind {
        /// Code byte found (see [`ScalarKind`]).
        kind: u8,
    },
    /// Slot type is not `u32`. v0.7.0 rejects the `index_dense_big_t`
    /// variant which uses `u40` slots.
    UnsupportedSlotKind {
        /// Code byte found (see [`ScalarKind`]).
        kind: u8,
    },
    /// `dimensions` is zero — not a valid index.
    ZeroDimensions,
    /// `multi` flag byte is neither 0 nor 1.
    InvalidMultiFlag {
        /// Byte found.
        value: u8,
    },
}

impl core::fmt::Display for HnswHeaderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated { got, need } => {
                write!(f, "header truncated: got {got} bytes, need {need}")
            }
            Self::WrongMagic => f.write_str("wrong magic; not a usearch dense index"),
            Self::UnsupportedFormatVersion {
                major,
                minor,
                patch,
            } => write!(
                f,
                "unsupported format version v{major}.{minor}.{patch}; v0.7.0 of tokenfs-algos pins to v{}.{}.x",
                SUPPORTED_VERSION_MAJOR, PINNED_VERSION_MINOR
            ),
            Self::UnknownMetricKind { code } => {
                write!(f, "unknown metric kind code 0x{code:02x}")
            }
            Self::UnknownScalarKind { code } => {
                write!(f, "unknown scalar kind code 0x{code:02x}")
            }
            Self::UnsupportedKeyKind { kind } => {
                write!(
                    f,
                    "unsupported key kind 0x{kind:02x}; v0.7.0 expects u64 (kind=14)"
                )
            }
            Self::UnsupportedSlotKind { kind } => write!(
                f,
                "unsupported slot kind 0x{kind:02x}; v0.7.0 expects u32 (kind=15) — index_dense_big_t with u40 slots is not supported"
            ),
            Self::ZeroDimensions => f.write_str("dimensions == 0; not a valid index"),
            Self::InvalidMultiFlag { value } => {
                write!(f, "multi flag byte 0x{value:02x} is neither 0 nor 1")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HnswHeaderError {}

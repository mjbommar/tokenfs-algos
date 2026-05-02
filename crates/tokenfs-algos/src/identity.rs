//! Content identifiers (CID) and multihash encoding.
//!
//! This module implements just enough of the [multihash] and [CIDv1] specs to
//! let callers turn a hash digest into an interoperable IPFS/IPLD content
//! identifier without pulling in the upstream `multihash` / `cid` crates.
//!
//! [multihash]: https://github.com/multiformats/multihash
//! [CIDv1]: https://github.com/multiformats/cid
//!
//! # Layered shape
//!
//! - **multihash**: `<varint hash-code><varint digest-length><digest bytes>`.
//!   Hash codes follow the [multicodec] table — only `sha2-256` (`0x12`) and
//!   `blake3` (`0x1e`, 256-bit) are wired in here. Anything else is a decode
//!   error.
//! - **CIDv1 binary**: `<varint cid-version=1><varint multicodec><multihash>`.
//!   The CID version is always `1`, so the leading byte is always `0x01`.
//! - **CIDv1 text**: `<multibase prefix><base-encoded binary CID>`. We emit
//!   `'b' + base32-lower-no-padding` (RFC 4648 §6, lower-cased), which is the
//!   default text form that IPFS / kubo / `ipfs cid` produce.
//!
//! [multicodec]: https://github.com/multiformats/multicodec/blob/master/table.csv
//!
//! # `no_std` posture
//!
//! All byte-array encode / decode entry points (`encode_varint_u64`,
//! `decode_varint_u64`, `encode_multihash`, `decode_multihash`,
//! `build_cid_v1`, `encode_base32_lower`) are pure `no_std` — they take and
//! return `&[u8]` / `&mut [u8]` only. The `Vec` / `String` returners
//! (`encode_multihash_vec`, `build_cid_v1_vec`, `build_cid_v1_string`,
//! `sha256_cid`, `blake3_cid`) are gated on either the `std` or `alloc`
//! cargo feature.
//!
//! # Example
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "alloc"))] {
//! use tokenfs_algos::identity::{Multicodec, MultihashCode, build_cid_v1_string,
//!     encode_multihash_vec};
//! use tokenfs_algos::hash::sha256::sha256;
//!
//! let digest = sha256(b"");
//! let mh = encode_multihash_vec(MultihashCode::Sha2_256, &digest);
//! let cid = build_cid_v1_string(Multicodec::Raw, &mh);
//! assert_eq!(cid, "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku");
//! # }
//! ```

#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::string::String;
#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use core::fmt;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Failure modes for the byte-array encoding entry points.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EncodeError {
    /// The output buffer was too small to hold the encoded value.
    BufferTooSmall {
        /// Bytes the encoder needed to write.
        needed: usize,
        /// Bytes that the caller provided.
        got: usize,
    },
    /// The supplied digest length did not match the multihash code's fixed
    /// digest size.
    DigestLengthMismatch {
        /// Multihash code that was being encoded.
        code: MultihashCode,
        /// Digest length the multihash code requires.
        expected: usize,
        /// Digest length the caller actually supplied.
        got: usize,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferTooSmall { needed, got } => {
                write!(f, "output buffer too small: need {needed} bytes, got {got}")
            }
            Self::DigestLengthMismatch {
                code,
                expected,
                got,
            } => write!(
                f,
                "digest length mismatch for {}: expected {expected} bytes, got {got}",
                code.name()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EncodeError {}

/// Failure modes for the byte-array decoding entry points.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeError {
    /// Input ended in the middle of a varint or digest.
    Truncated,
    /// The multihash code does not match any value in [`MultihashCode`].
    UnknownMultihashCode(u64),
    /// The multihash declared a digest length larger than the remaining input.
    DigestLengthMismatch {
        /// Length declared in the multihash header.
        declared: usize,
        /// Bytes actually available after the header.
        available: usize,
    },
    /// A varint exceeded the maximum 10-byte encoding for `u64`.
    VarintOverflow,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("input truncated"),
            Self::UnknownMultihashCode(code) => write!(f, "unknown multihash code: 0x{code:x}"),
            Self::DigestLengthMismatch {
                declared,
                available,
            } => write!(
                f,
                "multihash declared {declared} digest bytes but only {available} are available"
            ),
            Self::VarintOverflow => f.write_str("varint exceeds 10-byte u64 limit"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DecodeError {}

// ---------------------------------------------------------------------------
// Multihash codec table
// ---------------------------------------------------------------------------

/// Subset of the multicodec table that we recognize as multihash codes.
///
/// Only fixed-output 256-bit hash families are wired in; adding a new code
/// requires extending [`MultihashCode::raw`], [`MultihashCode::digest_len`],
/// [`MultihashCode::name`], and [`MultihashCode::from_raw`] together.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultihashCode {
    /// SHA-256 (FIPS 180-4), multicodec `0x12`.
    Sha2_256,
    /// BLAKE3 256-bit, multicodec `0x1e`.
    Blake3,
}

impl MultihashCode {
    /// Multicodec table value for this hash code.
    #[must_use]
    pub const fn raw(self) -> u64 {
        match self {
            Self::Sha2_256 => 0x12,
            Self::Blake3 => 0x1e,
        }
    }

    /// Fixed digest length, in bytes, that this code emits.
    #[must_use]
    pub const fn digest_len(self) -> usize {
        match self {
            Self::Sha2_256 | Self::Blake3 => 32,
        }
    }

    /// Spec name (matches the multicodec table column).
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sha2_256 => "sha2-256",
            Self::Blake3 => "blake3",
        }
    }

    /// Reverse lookup from a raw multicodec value.
    #[must_use]
    pub fn from_raw(code: u64) -> Option<Self> {
        match code {
            0x12 => Some(Self::Sha2_256),
            0x1e => Some(Self::Blake3),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Multicodec (content type) table
// ---------------------------------------------------------------------------

/// Subset of the multicodec table for CID content types.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Multicodec {
    /// Opaque bytes, multicodec `0x55`.
    Raw,
    /// MerkleDAG protobuf, multicodec `0x70`.
    DagPb,
    /// MerkleDAG CBOR, multicodec `0x71`.
    DagCbor,
    /// MerkleDAG JSON, multicodec `0x0129` (multi-byte varint).
    DagJson,
}

impl Multicodec {
    /// Multicodec table value.
    #[must_use]
    pub const fn raw(self) -> u64 {
        match self {
            Self::Raw => 0x55,
            Self::DagPb => 0x70,
            Self::DagCbor => 0x71,
            Self::DagJson => 0x0129,
        }
    }
}

// ---------------------------------------------------------------------------
// Varint codec (LEB128 unsigned, multiformats unsigned-varint subset)
// ---------------------------------------------------------------------------

/// Maximum length, in bytes, of an unsigned varint encoding a `u64`.
///
/// 10 bytes is the worst case: `u64::MAX` needs nine full 7-bit groups plus
/// one bit in the tenth byte.
pub const MAX_VARINT_LEN: usize = 10;

/// Computes the byte length of the unsigned varint encoding of `value`.
#[must_use]
pub const fn varint_u64_len(value: u64) -> usize {
    let mut bits: u32 = 64 - value.leading_zeros();
    if bits == 0 {
        bits = 1;
    }
    // ceil(bits / 7).
    bits.div_ceil(7) as usize
}

/// Encodes `value` as an unsigned varint into `out`.
///
/// Returns the number of bytes written. Errors with
/// [`EncodeError::BufferTooSmall`] when `out` is shorter than the encoding.
pub fn encode_varint_u64(value: u64, out: &mut [u8]) -> Result<usize, EncodeError> {
    let needed = varint_u64_len(value);
    if out.len() < needed {
        return Err(EncodeError::BufferTooSmall {
            needed,
            got: out.len(),
        });
    }
    let mut v = value;
    let mut i = 0;
    while v >= 0x80 {
        out[i] = ((v as u8) & 0x7f) | 0x80;
        v >>= 7;
        i += 1;
    }
    out[i] = v as u8;
    Ok(i + 1)
}

/// Decodes an unsigned varint from `bytes`.
///
/// Returns the decoded `u64` and the number of bytes consumed.
pub fn decode_varint_u64(bytes: &[u8]) -> Result<(u64, usize), DecodeError> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    for (i, &byte) in bytes.iter().enumerate() {
        if i >= MAX_VARINT_LEN {
            return Err(DecodeError::VarintOverflow);
        }
        let chunk = u64::from(byte & 0x7f);
        // Tenth byte (i == 9) of a u64 varint may carry only 1 payload bit.
        if i == MAX_VARINT_LEN - 1 && (byte & 0x7f) > 0x01 {
            return Err(DecodeError::VarintOverflow);
        }
        value |= chunk << shift;
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
    }
    // Either empty input or every byte had the continuation bit set.
    Err(DecodeError::Truncated)
}

// ---------------------------------------------------------------------------
// Multihash encoding
// ---------------------------------------------------------------------------

/// Encodes a multihash (`<varint code><varint length><digest>`) into `out`.
///
/// Returns the number of bytes written.
pub fn encode_multihash(
    code: MultihashCode,
    digest: &[u8],
    out: &mut [u8],
) -> Result<usize, EncodeError> {
    if digest.len() != code.digest_len() {
        return Err(EncodeError::DigestLengthMismatch {
            code,
            expected: code.digest_len(),
            got: digest.len(),
        });
    }
    let code_len = varint_u64_len(code.raw());
    let len_len = varint_u64_len(digest.len() as u64);
    let needed = code_len + len_len + digest.len();
    if out.len() < needed {
        return Err(EncodeError::BufferTooSmall {
            needed,
            got: out.len(),
        });
    }
    let n1 = encode_varint_u64(code.raw(), &mut out[..code_len])?;
    let n2 = encode_varint_u64(digest.len() as u64, &mut out[n1..n1 + len_len])?;
    out[n1 + n2..n1 + n2 + digest.len()].copy_from_slice(digest);
    Ok(n1 + n2 + digest.len())
}

/// Decodes a multihash. Returns `(code, digest)` where `digest` borrows from
/// `bytes`.
///
/// The decoder is strict: the declared digest length must equal the digest
/// length expected by the multihash code, and there must be exactly that many
/// bytes available after the header.
pub fn decode_multihash(bytes: &[u8]) -> Result<(MultihashCode, &[u8]), DecodeError> {
    if bytes.is_empty() {
        return Err(DecodeError::Truncated);
    }
    let (raw_code, n1) = decode_varint_u64(bytes)?;
    let code =
        MultihashCode::from_raw(raw_code).ok_or(DecodeError::UnknownMultihashCode(raw_code))?;
    let rest = &bytes[n1..];
    let (declared_len_u64, n2) = decode_varint_u64(rest)?;
    let declared = declared_len_u64 as usize;
    let digest = &rest[n2..];
    if declared != code.digest_len() || declared > digest.len() {
        return Err(DecodeError::DigestLengthMismatch {
            declared,
            available: digest.len(),
        });
    }
    Ok((code, &digest[..declared]))
}

/// Allocates a `Vec<u8>` and writes the multihash into it.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn encode_multihash_vec(code: MultihashCode, digest: &[u8]) -> Vec<u8> {
    debug_assert_eq!(
        digest.len(),
        code.digest_len(),
        "digest length must match multihash code"
    );
    let needed = varint_u64_len(code.raw()) + varint_u64_len(digest.len() as u64) + digest.len();
    let mut out = vec![0u8; needed];
    let written =
        encode_multihash(code, digest, &mut out).expect("multihash encode into sized buffer");
    debug_assert_eq!(written, needed);
    out
}

// ---------------------------------------------------------------------------
// CIDv1 binary encoding
// ---------------------------------------------------------------------------

/// CID version varint. Always `1` in this module.
const CID_VERSION_V1: u64 = 1;

/// Encodes a CIDv1 binary form: `<varint version=1><varint codec><multihash>`.
///
/// `multihash` is taken as opaque bytes — pass the output of
/// [`encode_multihash`] (or [`encode_multihash_vec`]). Returns the number of
/// bytes written.
pub fn build_cid_v1(
    codec: Multicodec,
    multihash: &[u8],
    out: &mut [u8],
) -> Result<usize, EncodeError> {
    let v_len = varint_u64_len(CID_VERSION_V1);
    let c_len = varint_u64_len(codec.raw());
    let needed = v_len + c_len + multihash.len();
    if out.len() < needed {
        return Err(EncodeError::BufferTooSmall {
            needed,
            got: out.len(),
        });
    }
    let n1 = encode_varint_u64(CID_VERSION_V1, &mut out[..v_len])?;
    let n2 = encode_varint_u64(codec.raw(), &mut out[n1..n1 + c_len])?;
    out[n1 + n2..n1 + n2 + multihash.len()].copy_from_slice(multihash);
    Ok(n1 + n2 + multihash.len())
}

/// Allocates a `Vec<u8>` and writes the binary CIDv1 into it.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn build_cid_v1_vec(codec: Multicodec, multihash: &[u8]) -> Vec<u8> {
    let needed = varint_u64_len(CID_VERSION_V1) + varint_u64_len(codec.raw()) + multihash.len();
    let mut out = vec![0u8; needed];
    let written = build_cid_v1(codec, multihash, &mut out).expect("CID encode into sized buffer");
    debug_assert_eq!(written, needed);
    out
}

// ---------------------------------------------------------------------------
// Base32-lower (RFC 4648 §6, no padding)
// ---------------------------------------------------------------------------

/// Multibase prefix for `base32-lower` text encodings.
pub const MULTIBASE_BASE32_LOWER_PREFIX: char = 'b';

/// RFC 4648 §6 alphabet, lower-cased.
const BASE32_LOWER_ALPHABET: &[u8; 32] = b"abcdefghijklmnopqrstuvwxyz234567";

/// Length of the base32-lower output for `input_bytes` bytes of input
/// (no padding).
#[must_use]
pub const fn base32_lower_len(input_bytes: usize) -> usize {
    // ceil(input_bytes * 8 / 5)
    (input_bytes * 8).div_ceil(5)
}

/// Encodes `bytes` as RFC 4648 §6 base32 with the lower-case alphabet and no
/// padding. Returns the number of bytes written into `out`.
pub fn encode_base32_lower(bytes: &[u8], out: &mut [u8]) -> Result<usize, EncodeError> {
    let needed = base32_lower_len(bytes.len());
    if out.len() < needed {
        return Err(EncodeError::BufferTooSmall {
            needed,
            got: out.len(),
        });
    }
    let mut buffer: u64 = 0;
    let mut bits: u32 = 0;
    let mut written = 0;
    for &b in bytes {
        buffer = (buffer << 8) | u64::from(b);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let idx = ((buffer >> bits) & 0x1f) as usize;
            out[written] = BASE32_LOWER_ALPHABET[idx];
            written += 1;
        }
    }
    if bits > 0 {
        let idx = ((buffer << (5 - bits)) & 0x1f) as usize;
        out[written] = BASE32_LOWER_ALPHABET[idx];
        written += 1;
    }
    debug_assert_eq!(written, needed);
    Ok(written)
}

/// Builds the canonical `base32-lower` text form of a CIDv1: `'b' + base32`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn build_cid_v1_string(codec: Multicodec, multihash: &[u8]) -> String {
    let binary = build_cid_v1_vec(codec, multihash);
    let needed = base32_lower_len(binary.len());
    let mut out = vec![0u8; 1 + needed];
    out[0] = MULTIBASE_BASE32_LOWER_PREFIX as u8;
    let written =
        encode_base32_lower(&binary, &mut out[1..]).expect("base32 encode into sized buffer");
    debug_assert_eq!(written, needed);
    // Safety: alphabet is ASCII, prefix is ASCII.
    String::from_utf8(out).expect("base32-lower output is ASCII")
}

// ---------------------------------------------------------------------------
// Convenience wrappers
// ---------------------------------------------------------------------------

/// Computes the SHA-256 CIDv1 (raw codec, base32-lower text) of `bytes`.
#[cfg(any(feature = "std", feature = "alloc"))]
#[must_use]
pub fn sha256_cid(bytes: &[u8]) -> String {
    let digest = crate::hash::sha256::sha256(bytes);
    let mh = encode_multihash_vec(MultihashCode::Sha2_256, &digest);
    build_cid_v1_string(Multicodec::Raw, &mh)
}

/// Computes the BLAKE3 CIDv1 (raw codec, base32-lower text) of `bytes`.
#[cfg(all(feature = "blake3", any(feature = "std", feature = "alloc")))]
#[must_use]
pub fn blake3_cid(bytes: &[u8]) -> String {
    let digest = crate::hash::blake3::blake3(bytes);
    let mh = encode_multihash_vec(MultihashCode::Blake3, &digest);
    build_cid_v1_string(Multicodec::Raw, &mh)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

    use super::*;

    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec;
    #[cfg(all(feature = "alloc", not(feature = "std")))]
    use alloc::vec::Vec;

    // ---- varint --------------------------------------------------------

    #[test]
    fn varint_round_trip_known_values() {
        for value in [0u64, 1, 0x7f, 0x80, 0x4000, 0x129, u64::MAX] {
            let mut buf = [0u8; MAX_VARINT_LEN];
            let n = encode_varint_u64(value, &mut buf).unwrap();
            assert_eq!(n, varint_u64_len(value));
            let (decoded, m) = decode_varint_u64(&buf[..n]).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(m, n);
        }
    }

    #[test]
    fn varint_len_known_widths() {
        assert_eq!(varint_u64_len(0), 1);
        assert_eq!(varint_u64_len(0x7f), 1);
        assert_eq!(varint_u64_len(0x80), 2);
        assert_eq!(varint_u64_len(0x129), 2);
        assert_eq!(varint_u64_len(0x3fff), 2);
        assert_eq!(varint_u64_len(0x4000), 3);
        assert_eq!(varint_u64_len(u64::MAX), MAX_VARINT_LEN);
    }

    #[test]
    fn varint_encode_dagjson_byte_pattern() {
        // 0x129 should encode as 0xa9, 0x02.
        let mut buf = [0u8; 2];
        let n = encode_varint_u64(0x129, &mut buf).unwrap();
        assert_eq!(n, 2);
        assert_eq!(buf, [0xa9, 0x02]);
    }

    #[test]
    fn varint_decode_truncated_continuation() {
        // Single byte with continuation bit set, no follow-up.
        let result = decode_varint_u64(&[0x80]);
        assert_eq!(result, Err(DecodeError::Truncated));
    }

    #[test]
    fn varint_decode_empty_is_truncated() {
        assert_eq!(decode_varint_u64(&[]), Err(DecodeError::Truncated));
    }

    #[test]
    fn varint_decode_eleven_bytes_overflows() {
        // 11 continuation bytes should hit VarintOverflow before reading the
        // 11th byte's payload.
        let bytes = [0xff_u8; 11];
        assert_eq!(decode_varint_u64(&bytes), Err(DecodeError::VarintOverflow));
    }

    #[test]
    fn varint_decode_tenth_byte_payload_overflow() {
        // Nine 0x80 bytes (carry only zeros) followed by a 10th byte whose
        // payload is 0x02 — that bit would shift past bit 63, so reject.
        let bytes = [0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02];
        assert_eq!(decode_varint_u64(&bytes), Err(DecodeError::VarintOverflow));
    }

    #[test]
    fn varint_decode_tenth_byte_payload_one_is_ok() {
        // u64::MAX = 0xffffffffffffffff. Encoded as 9× 0xff then 0x01.
        let bytes = [0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        let (value, n) = decode_varint_u64(&bytes).unwrap();
        assert_eq!(value, u64::MAX);
        assert_eq!(n, 10);
    }

    #[test]
    fn varint_encode_buffer_too_small() {
        let mut buf = [0u8; 1];
        let err = encode_varint_u64(0x129, &mut buf).unwrap_err();
        assert_eq!(err, EncodeError::BufferTooSmall { needed: 2, got: 1 });
    }

    // ---- multihash -----------------------------------------------------

    #[test]
    fn multihash_round_trip_sha256() {
        let digest = [0x42u8; 32];
        let mut buf = [0u8; 64];
        let n = encode_multihash(MultihashCode::Sha2_256, &digest, &mut buf).unwrap();
        assert_eq!(n, 34);
        assert_eq!(buf[0], 0x12);
        assert_eq!(buf[1], 0x20);
        assert_eq!(&buf[2..n], &digest);

        let (code, decoded) = decode_multihash(&buf[..n]).unwrap();
        assert_eq!(code, MultihashCode::Sha2_256);
        assert_eq!(decoded, &digest);
    }

    #[test]
    fn multihash_round_trip_blake3() {
        let digest = [0xa5u8; 32];
        let mut buf = [0u8; 64];
        let n = encode_multihash(MultihashCode::Blake3, &digest, &mut buf).unwrap();
        assert_eq!(buf[0], 0x1e);
        assert_eq!(buf[1], 0x20);

        let (code, decoded) = decode_multihash(&buf[..n]).unwrap();
        assert_eq!(code, MultihashCode::Blake3);
        assert_eq!(decoded, &digest);
    }

    #[test]
    fn multihash_encode_rejects_wrong_digest_len() {
        let mut buf = [0u8; 64];
        let err = encode_multihash(MultihashCode::Sha2_256, &[0; 16], &mut buf).unwrap_err();
        assert_eq!(
            err,
            EncodeError::DigestLengthMismatch {
                code: MultihashCode::Sha2_256,
                expected: 32,
                got: 16,
            }
        );
    }

    #[test]
    fn multihash_encode_rejects_short_buffer() {
        let mut buf = [0u8; 8];
        let err = encode_multihash(MultihashCode::Sha2_256, &[0; 32], &mut buf).unwrap_err();
        assert!(matches!(
            err,
            EncodeError::BufferTooSmall { needed: 34, .. }
        ));
    }

    #[test]
    fn multihash_decode_rejects_empty() {
        assert_eq!(decode_multihash(&[]), Err(DecodeError::Truncated));
    }

    #[test]
    fn multihash_decode_rejects_unknown_code() {
        // 0x55 (raw) is a valid multicodec but NOT in our multihash table.
        // The high bit is clear so it decodes as a single-byte varint = 0x55.
        let bytes = [
            0x55, 0x20, 0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(
            decode_multihash(&bytes),
            Err(DecodeError::UnknownMultihashCode(0x55))
        );
    }

    #[test]
    fn multihash_decode_rejects_wrong_declared_length() {
        // sha2-256 (0x12) but declared 16-byte digest.
        let mut bytes = [0u8; 18];
        bytes[0] = 0x12;
        bytes[1] = 0x10; // 16
        let err = decode_multihash(&bytes).unwrap_err();
        assert_eq!(
            err,
            DecodeError::DigestLengthMismatch {
                declared: 16,
                available: 16,
            }
        );
    }

    #[test]
    fn multihash_decode_rejects_truncated_digest() {
        // sha2-256 with declared 32-byte digest but only 8 bytes available.
        let mut bytes = [0u8; 10];
        bytes[0] = 0x12;
        bytes[1] = 0x20;
        let err = decode_multihash(&bytes).unwrap_err();
        assert_eq!(
            err,
            DecodeError::DigestLengthMismatch {
                declared: 32,
                available: 8,
            }
        );
    }

    // ---- base32-lower (RFC 4648 §10 vectors, lower-cased) -------------

    fn b32(input: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; base32_lower_len(input.len())];
        let n = encode_base32_lower(input, &mut out).unwrap();
        out.truncate(n);
        out
    }

    #[test]
    fn base32_lower_rfc4648_vectors() {
        // Lower-cased RFC 4648 §10 reference vectors, no padding.
        assert_eq!(b32(b""), b"");
        assert_eq!(b32(b"f"), b"my");
        assert_eq!(b32(b"fo"), b"mzxq");
        assert_eq!(b32(b"foo"), b"mzxw6");
        assert_eq!(b32(b"foob"), b"mzxw6yq");
        assert_eq!(b32(b"fooba"), b"mzxw6ytb");
        assert_eq!(b32(b"foobar"), b"mzxw6ytboi");
    }

    #[test]
    fn base32_lower_buffer_too_small() {
        let mut out = [0u8; 1];
        let err = encode_base32_lower(b"foob", &mut out).unwrap_err();
        assert_eq!(err, EncodeError::BufferTooSmall { needed: 7, got: 1 });
    }

    #[test]
    fn base32_lower_len_known_widths() {
        assert_eq!(base32_lower_len(0), 0);
        assert_eq!(base32_lower_len(1), 2);
        assert_eq!(base32_lower_len(2), 4);
        assert_eq!(base32_lower_len(3), 5);
        assert_eq!(base32_lower_len(4), 7);
        assert_eq!(base32_lower_len(5), 8);
        assert_eq!(base32_lower_len(36), 58);
    }

    // ---- CIDv1 binary layout ------------------------------------------

    #[test]
    fn cid_v1_binary_layout_for_sha256_raw() {
        let digest = [0xaau8; 32];
        let mh = encode_multihash_vec(MultihashCode::Sha2_256, &digest);
        let cid = build_cid_v1_vec(Multicodec::Raw, &mh);
        // 01 (cid version) | 55 (raw codec) | 12 (sha2-256) | 20 (32 bytes) | digest
        assert_eq!(cid[0], 0x01);
        assert_eq!(cid[1], 0x55);
        assert_eq!(cid[2], 0x12);
        assert_eq!(cid[3], 0x20);
        assert_eq!(&cid[4..], &digest);
        assert_eq!(cid.len(), 4 + 32);
    }

    #[test]
    fn cid_v1_binary_layout_for_dagjson_multibyte_codec() {
        // DagJson is 0x0129 → varint encodes as 0xa9 0x02.
        let digest = [0xbb_u8; 32];
        let mh = encode_multihash_vec(MultihashCode::Sha2_256, &digest);
        let cid = build_cid_v1_vec(Multicodec::DagJson, &mh);
        assert_eq!(cid[0], 0x01);
        assert_eq!(cid[1], 0xa9);
        assert_eq!(cid[2], 0x02);
        assert_eq!(cid[3], 0x12);
        assert_eq!(cid[4], 0x20);
        assert_eq!(&cid[5..], &digest);
        assert_eq!(cid.len(), 5 + 32);
    }

    #[test]
    fn cid_v1_buffer_too_small() {
        let digest = [0u8; 32];
        let mh = encode_multihash_vec(MultihashCode::Sha2_256, &digest);
        let mut out = [0u8; 8];
        let err = build_cid_v1(Multicodec::Raw, &mh, &mut out).unwrap_err();
        assert!(matches!(err, EncodeError::BufferTooSmall { .. }));
    }

    // ---- canonical CID strings ----------------------------------------

    #[test]
    fn sha256_cid_empty_string() {
        assert_eq!(
            sha256_cid(b""),
            "bafkreihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku"
        );
    }

    #[test]
    fn sha256_cid_hello() {
        assert_eq!(
            sha256_cid(b"hello"),
            "bafkreibm6jg3ux5qumhcn2b3flc3tyu6dmlb4xa7u5bf44yegnrjhc4yeq"
        );
    }

    #[test]
    fn sha256_cid_abc() {
        assert_eq!(
            sha256_cid(b"abc"),
            "bafkreif2pall7dybz7vecqka3zo24irdwabwdi4wc55jznaq75q7eaavvu"
        );
    }

    #[test]
    fn sha256_cid_one_thousand_a() {
        let input = vec![b'a'; 1000];
        assert_eq!(
            sha256_cid(&input),
            "bafkreicb5xwoilld5dm36uk2totjglq4edf4t5nf2e2giww3lwy3s436um"
        );
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_cid_empty_string() {
        assert_eq!(
            blake3_cid(b""),
            "bafkr4ifpcne3t5pzugtkaqcn5i3nzskjtpfslsnnyejlpte2spfoihzsmi"
        );
    }

    #[cfg(feature = "blake3")]
    #[test]
    fn blake3_cid_abc() {
        assert_eq!(
            blake3_cid(b"abc"),
            "bafkr4ideg6z2yocgkez77nr3outtvdnvjdcvqrs5phnqh7jvtrwnlpm5qu"
        );
    }

    // ---- MultihashCode helpers ----------------------------------------

    #[test]
    fn multihash_code_from_raw_round_trip() {
        for code in [MultihashCode::Sha2_256, MultihashCode::Blake3] {
            assert_eq!(MultihashCode::from_raw(code.raw()), Some(code));
        }
        assert_eq!(MultihashCode::from_raw(0x00), None);
        assert_eq!(MultihashCode::from_raw(0x99), None);
    }

    #[test]
    fn multihash_code_metadata() {
        assert_eq!(MultihashCode::Sha2_256.raw(), 0x12);
        assert_eq!(MultihashCode::Sha2_256.digest_len(), 32);
        assert_eq!(MultihashCode::Sha2_256.name(), "sha2-256");
        assert_eq!(MultihashCode::Blake3.raw(), 0x1e);
        assert_eq!(MultihashCode::Blake3.digest_len(), 32);
        assert_eq!(MultihashCode::Blake3.name(), "blake3");
    }

    // ---- error display -------------------------------------------------

    #[test]
    fn error_display_smoke() {
        #[cfg(all(feature = "alloc", not(feature = "std")))]
        use alloc::format;
        let s = format!("{}", EncodeError::BufferTooSmall { needed: 4, got: 1 });
        assert!(s.contains("too small"));
        let s = format!("{}", DecodeError::UnknownMultihashCode(0x99));
        assert!(s.contains("99"));
    }
}

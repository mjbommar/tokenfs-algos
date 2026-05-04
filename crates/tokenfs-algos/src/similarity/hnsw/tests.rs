//! Unit tests for the v0.7.0 HNSW Phase 1 surface.
//!
//! Phase 1's correctness gate is byte-format parsing against a
//! hand-crafted toy fixture. The fixture bytes are constructed directly
//! from `docs/hnsw/research/USEARCH_DEEP_DIVE.md` §1.3 (header) — the
//! authoritative annotation of `_references/usearch/include/usearch/index_dense.hpp:42-79`.
//!
//! Phase 4 introduces the clustering-fuzz + brute-force correctness
//! gate (see `docs/hnsw/components/CLUSTERING_FUZZ.md`); this file
//! grows alongside the walker and builder.

#![allow(clippy::unwrap_used)] // Test code — panic on Err is the desired failure mode.

use super::header::{HEADER_BYTES, HnswHeader, HnswHeaderError, MetricKind, ScalarKind};

/// Hand-crafted toy `index_dense_head_t` for a 4-node, 8-dimensional,
/// `u8` scalar, `l2sq` metric index.
///
/// Byte-by-byte breakdown (see `USEARCH_DEEP_DIVE.md` §1.3):
///
/// ```text
/// 0x00..0x07  magic               b"usearch"            7 bytes
/// 0x07..0x09  version_major = 2   le_u16                2 bytes
/// 0x09..0x0B  version_minor = 25  le_u16                2 bytes
/// 0x0B..0x0D  version_patch = 1   le_u16                2 bytes
/// 0x0D..0x0E  metric = 'e' (l2sq) u8                    1 byte
/// 0x0E..0x0F  scalar = 17 (u8)    u8                    1 byte
/// 0x0F..0x10  key = 14 (u64)      u8                    1 byte
/// 0x10..0x11  slot = 15 (u32)     u8                    1 byte
/// 0x11..0x19  count_present = 4   le_u64                8 bytes
/// 0x19..0x21  count_deleted = 0   le_u64                8 bytes
/// 0x21..0x29  dimensions = 8      le_u64                8 bytes
/// 0x29..0x2A  multi = 0           u8                    1 byte
/// 0x2A..0x40  reserved (zero)                           22 bytes
/// ```
///
/// Total: 64 bytes (= [`HEADER_BYTES`]).
const TOY_HEADER_V2_25: [u8; HEADER_BYTES] = [
    // magic "usearch" (7 bytes)
    b'u', b's', b'e', b'a', b'r', b'c', b'h', // version_major = 2 (LE u16)
    0x02, 0x00, // version_minor = 25 (LE u16)
    0x19, 0x00, // version_patch = 1 (LE u16)
    0x01, 0x00, // metric = 'e' (l2sq)
    b'e', // scalar = 17 (u8_k)
    17,   // key = 14 (u64_k)
    14,   // slot = 15 (u32_k)
    15,   // count_present = 4 (LE u64)
    0x04, 0, 0, 0, 0, 0, 0, 0, // count_deleted = 0 (LE u64)
    0, 0, 0, 0, 0, 0, 0, 0, // dimensions = 8 (LE u64)
    0x08, 0, 0, 0, 0, 0, 0, 0, // multi = false
    0, // reserved (22 zero bytes — bytes 0x2A through 0x3F inclusive)
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

#[test]
fn header_parses_canonical_v2_25() {
    let header = HnswHeader::try_parse(&TOY_HEADER_V2_25)
        .expect("toy header constructed by hand from the spec must parse");

    assert_eq!(header.version_major(), 2);
    assert_eq!(header.version_minor(), 25);
    assert_eq!(header.version_patch(), 1);
    assert_eq!(header.metric_kind(), MetricKind::L2Squared);
    assert_eq!(header.scalar_kind(), ScalarKind::U8);
    assert_eq!(header.count_present(), 4);
    assert_eq!(header.count_deleted(), 0);
    assert_eq!(header.dimensions(), 8);
    assert!(!header.multi());
    // 8 dims × 8 bits per scalar = 64 bits = 8 bytes
    assert_eq!(header.bytes_per_vector(), 8);
}

#[test]
fn header_truncated_input_fails_clean() {
    let too_short = [0u8; 32];
    let err = HnswHeader::try_parse(&too_short).unwrap_err();
    assert_eq!(
        err,
        HnswHeaderError::Truncated {
            got: 32,
            need: HEADER_BYTES,
        }
    );
}

#[test]
fn header_empty_input_fails_clean() {
    let err = HnswHeader::try_parse(&[]).unwrap_err();
    assert!(matches!(
        err,
        HnswHeaderError::Truncated { got: 0, need: 64 }
    ));
}

#[test]
fn header_rejects_wrong_magic() {
    let mut bad = TOY_HEADER_V2_25;
    // Corrupt one magic byte.
    bad[0] = b'X';
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert_eq!(err, HnswHeaderError::WrongMagic);
}

#[test]
fn header_rejects_v1_format() {
    let mut bad = TOY_HEADER_V2_25;
    // version_major = 1
    bad[7] = 0x01;
    bad[8] = 0x00;
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert!(matches!(
        err,
        HnswHeaderError::UnsupportedFormatVersion { major: 1, .. }
    ));
}

#[test]
fn header_rejects_v3_future_format() {
    let mut bad = TOY_HEADER_V2_25;
    // version_major = 3
    bad[7] = 0x03;
    bad[8] = 0x00;
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert!(matches!(
        err,
        HnswHeaderError::UnsupportedFormatVersion { major: 3, .. }
    ));
}

#[test]
fn header_rejects_v2_20_pre_2_25() {
    // pre-2.25 minor is allowed by usearch but our parser pins to v2.25.x.
    // Specifically pre-2.10 had a different scalar_kind numbering; rejecting
    // anything < v2.25 means we never silently mis-parse those files.
    let mut bad = TOY_HEADER_V2_25;
    // version_minor = 20 (LE u16: 0x14, 0x00)
    bad[9] = 0x14;
    bad[10] = 0x00;
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert!(matches!(
        err,
        HnswHeaderError::UnsupportedFormatVersion {
            major: 2,
            minor: 20,
            ..
        }
    ));
}

#[test]
fn header_rejects_unknown_metric_kind() {
    let mut bad = TOY_HEADER_V2_25;
    // 'X' is not in the metric_kind_t enum.
    bad[13] = b'X';
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert_eq!(err, HnswHeaderError::UnknownMetricKind { code: b'X' });
}

#[test]
fn header_rejects_unknown_scalar_kind() {
    let mut bad = TOY_HEADER_V2_25;
    // 99 is not in the scalar_kind_t enum.
    bad[14] = 99;
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert_eq!(err, HnswHeaderError::UnknownScalarKind { code: 99 });
}

#[test]
fn header_rejects_non_u64_key() {
    let mut bad = TOY_HEADER_V2_25;
    // Use u32 (15) as the key kind — valid scalar, wrong for default index_dense_t.
    bad[15] = 15;
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert_eq!(err, HnswHeaderError::UnsupportedKeyKind { kind: 15 });
}

#[test]
fn header_rejects_u40_slot() {
    let mut bad = TOY_HEADER_V2_25;
    // Slot kind = u40 (2) — index_dense_big_t variant. v0.7.0 rejects.
    bad[16] = 2;
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert_eq!(err, HnswHeaderError::UnsupportedSlotKind { kind: 2 });
}

#[test]
fn header_rejects_zero_dimensions() {
    let mut bad = TOY_HEADER_V2_25;
    // dimensions = 0 (already zero in bytes 0x21..0x29 for an empty index;
    // we explicitly set them here for clarity).
    for byte in bad.iter_mut().skip(0x21).take(8) {
        *byte = 0;
    }
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert_eq!(err, HnswHeaderError::ZeroDimensions);
}

#[test]
fn header_rejects_invalid_multi_flag() {
    let mut bad = TOY_HEADER_V2_25;
    // multi byte = 0xFF — neither 0 nor 1.
    bad[0x29] = 0xFF;
    let err = HnswHeader::try_parse(&bad).unwrap_err();
    assert_eq!(err, HnswHeaderError::InvalidMultiFlag { value: 0xFF });
}

#[test]
fn header_metric_kinds_round_trip_through_byte_codes() {
    // Spot-check every supported MetricKind discriminant matches the
    // documented char code (USEARCH_DEEP_DIVE §1.3).
    for (byte, expected) in [
        (b'i', MetricKind::InnerProduct),
        (b'c', MetricKind::Cosine),
        (b'e', MetricKind::L2Squared),
        (b'b', MetricKind::Hamming),
        (b't', MetricKind::Tanimoto),
        (b's', MetricKind::Sorensen),
        (b'j', MetricKind::Jaccard),
    ] {
        let mut bytes = TOY_HEADER_V2_25;
        bytes[13] = byte;
        let header = HnswHeader::try_parse(&bytes).expect("valid metric byte");
        assert_eq!(header.metric_kind(), expected, "metric byte 0x{byte:02x}");
    }
}

#[test]
fn header_scalar_kinds_round_trip_for_v1_supported_set() {
    // Spot-check the four scalar types v1 ships distance kernels for.
    for (code, expected, expected_bits) in [
        (1u8, ScalarKind::B1x8, 1u32),
        (11, ScalarKind::F32, 32),
        (17, ScalarKind::U8, 8),
        (23, ScalarKind::I8, 8),
    ] {
        let mut bytes = TOY_HEADER_V2_25;
        bytes[14] = code;
        let header = HnswHeader::try_parse(&bytes).expect("valid scalar code");
        assert_eq!(header.scalar_kind(), expected, "scalar code {code}");
        assert_eq!(
            expected.bits_per_scalar(),
            expected_bits,
            "bits per scalar for {expected:?}"
        );
    }
}

#[test]
fn header_bytes_per_vector_for_packed_binary() {
    // 256 dims × 1 bit per scalar = 256 bits = 32 bytes. This is the
    // F22 fingerprint cell that motivates Hamming distance kernels.
    let mut bytes = TOY_HEADER_V2_25;
    // scalar_kind = b1x8 (1)
    bytes[14] = 1;
    // dimensions = 256 (LE u64)
    bytes[0x21] = 0x00;
    bytes[0x22] = 0x01;
    bytes[0x23] = 0x00;
    bytes[0x24] = 0x00;
    let header = HnswHeader::try_parse(&bytes).expect("256-dim b1x8 header");
    assert_eq!(header.dimensions(), 256);
    assert_eq!(header.bytes_per_vector(), 32);
}

#[test]
fn header_does_not_panic_on_random_garbage() {
    // Sanity smoke: 1024 random-ish byte patterns must never panic.
    // This is a crude proxy for the Phase 5 cargo-fuzz target.
    let seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut state = seed;
    for _ in 0..1024 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut bytes = [0u8; HEADER_BYTES];
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = ((state >> (i % 56)) & 0xFF) as u8;
        }
        // Never panics; result is irrelevant.
        let _ = HnswHeader::try_parse(&bytes);
    }
}

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
#[cfg(any(feature = "std", feature = "alloc"))]
use super::view::{HnswView, HnswViewError};

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

// ---------------------------------------------------------------------
// Full-file toy fixture for HnswView tests
// ---------------------------------------------------------------------
//
// Per `docs/hnsw/research/USEARCH_DEEP_DIVE.md` §1.8 — a 4-node, max-
// level-2, 8-dim u8 L2² index. Total 308 bytes. Layout:
//
//   0x0000  vectors header         8       [u32 rows=4][u32 cols=8]
//   0x0008  vectors data           32      4 × 8 raw bytes
//   0x0028  dense head             64      see §1.3
//   0x0068  graph header           40      see §1.5
//   0x0090  levels[0..3]           8       [i16 2][i16 1][i16 0][i16 0]
//   0x0098  node-0 tape (level 2)  54
//   0x00CE  node-1 tape (level 1)  42
//   0x00F8  node-2 tape (level 0)  30
//   0x0116  node-3 tape (level 0)  30
//   EOF     0x0134
//
// Graph topology (slot IDs):
//   node 0 (level 2 entry): base→{1,2,3}, level-1→{1}, level-2→{}
//   node 1 (level 1):       base→{0,2,3}, level-1→{0}
//   node 2 (level 0):       base→{0,1,3}
//   node 3 (level 0):       base→{0,1,2}

const TOY_KEY_BASE: u64 = 0x1000_0000_0000_0000;

fn toy_vector(slot: usize) -> [u8; 8] {
    let base = 0x10 + (slot as u8) * 0x10;
    [
        base,
        base + 1,
        base + 2,
        base + 3,
        base + 4,
        base + 5,
        base + 6,
        base + 7,
    ]
}

/// Build the full 308-byte toy fixture. M0=4 (`connectivity_base`), M=2
/// (`connectivity`), 8-dim u8, L2² metric.
fn build_toy_fixture() -> Vec<u8> {
    let mut buf = Vec::with_capacity(308);

    // ---- vectors header (32-bit rows + cols)
    buf.extend_from_slice(&4u32.to_le_bytes()); // rows
    buf.extend_from_slice(&8u32.to_le_bytes()); // cols (= bytes_per_vector)

    // ---- vectors data (4 × 8 raw bytes)
    for slot in 0..4 {
        buf.extend_from_slice(&toy_vector(slot));
    }
    assert_eq!(buf.len(), 0x28); // dense head starts at 0x28

    // ---- dense head (64 bytes)
    buf.extend_from_slice(b"usearch"); // magic (7 bytes)
    buf.extend_from_slice(&2u16.to_le_bytes()); // version_major
    buf.extend_from_slice(&25u16.to_le_bytes()); // version_minor
    buf.extend_from_slice(&1u16.to_le_bytes()); // version_patch
    buf.push(b'e'); // metric_kind = L2²
    buf.push(17); // scalar_kind = u8_k
    buf.push(14); // key_kind = u64_k
    buf.push(15); // slot_kind = u32_k
    buf.extend_from_slice(&4u64.to_le_bytes()); // count_present
    buf.extend_from_slice(&0u64.to_le_bytes()); // count_deleted
    buf.extend_from_slice(&8u64.to_le_bytes()); // dimensions
    buf.push(0); // multi = false
    // 22 reserved zero bytes (header is 64 bytes total per static_assert in
    // index_dense.hpp:31; the §1.3 byte table only documents 6 of these but
    // the actual reserved tail spans bytes 0x2A..0x40 of the header).
    buf.extend_from_slice(&[0u8; 22]);
    assert_eq!(buf.len(), 0x68); // graph header starts at 0x68

    // ---- graph header (40 bytes)
    buf.extend_from_slice(&4u64.to_le_bytes()); // size
    buf.extend_from_slice(&2u64.to_le_bytes()); // connectivity (M)
    buf.extend_from_slice(&4u64.to_le_bytes()); // connectivity_base (M0)
    buf.extend_from_slice(&2u64.to_le_bytes()); // max_level
    buf.extend_from_slice(&0u64.to_le_bytes()); // entry_slot
    assert_eq!(buf.len(), 0x90); // levels array starts at 0x90

    // ---- levels[0..3] (i16 LE)
    buf.extend_from_slice(&2i16.to_le_bytes());
    buf.extend_from_slice(&1i16.to_le_bytes());
    buf.extend_from_slice(&0i16.to_le_bytes());
    buf.extend_from_slice(&0i16.to_le_bytes());
    assert_eq!(buf.len(), 0x98); // node-0 tape starts at 0x98

    // ---- node-0 tape (level 2; 54 bytes total: 10 head + 20 base + 12 + 12)
    write_node_head(&mut buf, TOY_KEY_BASE, 2);
    write_slab(&mut buf, &[1, 2, 3], 4); // base, cap=M0=4
    write_slab(&mut buf, &[1], 2); // level-1, cap=M=2
    write_slab(&mut buf, &[], 2); // level-2 (entry-only level), cap=M=2
    assert_eq!(buf.len(), 0xCE); // node-1 tape at 0xCE

    // ---- node-1 tape (level 1; 42 bytes: 10 + 20 + 12)
    write_node_head(&mut buf, TOY_KEY_BASE | 1, 1);
    write_slab(&mut buf, &[0, 2, 3], 4);
    write_slab(&mut buf, &[0], 2);
    assert_eq!(buf.len(), 0xF8); // node-2 tape at 0xF8

    // ---- node-2 tape (level 0; 30 bytes: 10 + 20)
    write_node_head(&mut buf, TOY_KEY_BASE | 2, 0);
    write_slab(&mut buf, &[0, 1, 3], 4);
    assert_eq!(buf.len(), 0x116); // node-3 tape at 0x116

    // ---- node-3 tape (level 0; 30 bytes)
    write_node_head(&mut buf, TOY_KEY_BASE | 3, 0);
    write_slab(&mut buf, &[0, 1, 2], 4);
    assert_eq!(buf.len(), 0x134); // EOF

    buf
}

fn write_node_head(buf: &mut Vec<u8>, key: u64, level: i16) {
    buf.extend_from_slice(&key.to_le_bytes());
    buf.extend_from_slice(&level.to_le_bytes());
}

fn write_slab(buf: &mut Vec<u8>, neighbors: &[u32], cap: u32) {
    assert!(
        neighbors.len() <= cap as usize,
        "slab with {} neighbors exceeds cap {cap}",
        neighbors.len()
    );
    buf.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
    for &n in neighbors {
        buf.extend_from_slice(&n.to_le_bytes());
    }
    // Pad the slab out to its cap with zero slots.
    for _ in neighbors.len()..cap as usize {
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}

#[test]
fn view_parses_full_toy_fixture() {
    let fixture = build_toy_fixture();
    assert_eq!(fixture.len(), 0x134);
    let view = HnswView::try_new(&fixture).expect("toy fixture must parse");

    // Dense head
    assert_eq!(view.dimensions(), 8);
    assert_eq!(view.bytes_per_vector(), 8);
    assert_eq!(view.scalar_kind(), ScalarKind::U8);
    assert_eq!(view.header().metric_kind(), MetricKind::L2Squared);
    assert_eq!(view.header().count_present(), 4);

    // Graph header
    assert_eq!(view.node_count(), 4);
    assert_eq!(view.connectivity(), 2);
    assert_eq!(view.connectivity_base(), 4);
    assert_eq!(view.max_level(), 2);
    assert_eq!(view.entry_point(), Some(0));
}

#[test]
fn view_node_lookup_in_bounds() {
    let fixture = build_toy_fixture();
    let view = HnswView::try_new(&fixture).unwrap();

    for slot in 0..4 {
        let node = view.try_node(slot).unwrap();
        assert_eq!(node.slot(), slot);
        assert_eq!(node.key(), TOY_KEY_BASE | (slot as u64));
        assert_eq!(node.vector_bytes(), &toy_vector(slot as usize));
    }

    assert_eq!(view.try_node(0).unwrap().level(), 2);
    assert_eq!(view.try_node(1).unwrap().level(), 1);
    assert_eq!(view.try_node(2).unwrap().level(), 0);
    assert_eq!(view.try_node(3).unwrap().level(), 0);
}

#[test]
fn view_node_lookup_out_of_bounds_returns_error() {
    let fixture = build_toy_fixture();
    let view = HnswView::try_new(&fixture).unwrap();
    let err = view.try_node(4).unwrap_err();
    assert_eq!(
        err,
        HnswViewError::NodeIdOutOfRange {
            slot: 4,
            node_count: 4,
        }
    );
}

#[test]
fn view_neighbor_iteration_per_level() {
    let fixture = build_toy_fixture();
    let view = HnswView::try_new(&fixture).unwrap();

    // Node 0 (entry, level 2)
    let n0 = view.try_node(0).unwrap();
    let n0_base = n0.try_neighbors(0).unwrap();
    assert_eq!(n0_base.len(), 3);
    assert_eq!(n0_base.iter().collect::<Vec<u32>>(), vec![1, 2, 3]);

    let n0_l1 = n0.try_neighbors(1).unwrap();
    assert_eq!(n0_l1.len(), 1);
    assert_eq!(n0_l1.iter().collect::<Vec<u32>>(), vec![1]);

    let n0_l2 = n0.try_neighbors(2).unwrap();
    assert_eq!(n0_l2.len(), 0);
    assert!(n0_l2.is_empty());

    // Node 1 (level 1)
    let n1 = view.try_node(1).unwrap();
    let n1_base = n1.try_neighbors(0).unwrap();
    assert_eq!(n1_base.iter().collect::<Vec<u32>>(), vec![0, 2, 3]);
    let n1_l1 = n1.try_neighbors(1).unwrap();
    assert_eq!(n1_l1.iter().collect::<Vec<u32>>(), vec![0]);
}

#[test]
fn view_neighbor_level_out_of_range_returns_error() {
    let fixture = build_toy_fixture();
    let view = HnswView::try_new(&fixture).unwrap();
    // Node 2 is at level 0; querying level 1 must fail.
    let n2 = view.try_node(2).unwrap();
    let err = n2.try_neighbors(1).unwrap_err();
    assert_eq!(
        err,
        HnswViewError::NeighborLevelOutOfRange {
            slot: 2,
            level: 1,
            node_level: 0,
        }
    );
}

#[test]
fn view_neighbor_get_returns_none_past_count() {
    let fixture = build_toy_fixture();
    let view = HnswView::try_new(&fixture).unwrap();
    let n0 = view.try_node(0).unwrap();
    let base = n0.try_neighbors(0).unwrap();
    assert_eq!(base.get(0), Some(1));
    assert_eq!(base.get(2), Some(3));
    assert_eq!(base.get(3), None); // count = 3, position 3 is past live range
    assert_eq!(base.get(99), None);
}

#[test]
fn view_rejects_truncated_input() {
    let fixture = build_toy_fixture();
    // Truncate at every plausible boundary; each should fail closed.
    for cut_at in [0, 4, 7, 8, 16, 0x28, 0x68, 0x90, 0x98, 0xC0, 0x130] {
        let _ = HnswView::try_new(&fixture[..cut_at]).unwrap_err();
    }
}

#[test]
fn view_rejects_vectors_cols_mismatch() {
    let mut bad = build_toy_fixture();
    // Corrupt the dense-head dimensions field rather than the vectors-
    // header cols field: that keeps the file layout addressable (dense
    // head still parses at 0x28) while making the cross-check trip.
    // Dimensions field is at dense_head_offset + 0x21 = 0x28 + 0x21 = 0x49.
    bad[0x49..0x51].copy_from_slice(&16u64.to_le_bytes()); // claim dim=16, bytes_per_vector=16
    let err = HnswView::try_new(&bad).unwrap_err();
    assert!(
        matches!(
            err,
            HnswViewError::VectorsColsMismatch {
                got: 8,
                expected: 16,
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn view_rejects_vectors_rows_mismatch() {
    let mut bad = build_toy_fixture();
    // Set vectors-header rows to 8 (graph header has size=4).
    bad[0..4].copy_from_slice(&8u32.to_le_bytes());
    // Tail bytes won't be reachable — make the fixture longer to satisfy
    // the vectors-blob length check first, then the cross-check fires.
    let extra = vec![0u8; 4 * 8]; // 4 more 8-byte vectors so vectors_blob is satisfied
    let mut padded = bad[..0x08].to_vec();
    padded.extend_from_slice(&[0u8; 8 * 8]); // 8 vectors total now
    padded.extend_from_slice(&bad[0x28..]); // dense head onwards
    drop(extra);
    let err = HnswView::try_new(&padded).unwrap_err();
    assert!(matches!(
        err,
        HnswViewError::VectorsRowsMismatch {
            got: 8,
            expected: 4,
        }
    ));
}

#[test]
fn view_rejects_invalid_connectivity() {
    let mut bad = build_toy_fixture();
    // graph header connectivity is at file offset 0x68 + 8 = 0x70.
    bad[0x70..0x78].copy_from_slice(&1u64.to_le_bytes());
    let err = HnswView::try_new(&bad).unwrap_err();
    assert!(matches!(err, HnswViewError::InvalidConnectivity { got: 1 }));
}

#[test]
fn view_rejects_connectivity_base_below_connectivity() {
    let mut bad = build_toy_fixture();
    // connectivity = 4 (offset 0x70), connectivity_base = 2 (offset 0x78).
    bad[0x70..0x78].copy_from_slice(&4u64.to_le_bytes());
    bad[0x78..0x80].copy_from_slice(&2u64.to_le_bytes());
    let err = HnswView::try_new(&bad).unwrap_err();
    assert!(matches!(
        err,
        HnswViewError::InvalidConnectivityBase {
            got: 2,
            connectivity: 4,
        }
    ));
}

#[test]
fn view_rejects_entry_point_out_of_range() {
    let mut bad = build_toy_fixture();
    // entry_slot at file offset 0x68 + 0x20 = 0x88.
    bad[0x88..0x90].copy_from_slice(&99u64.to_le_bytes());
    let err = HnswView::try_new(&bad).unwrap_err();
    assert!(matches!(
        err,
        HnswViewError::EntryPointOutOfRange {
            entry_slot: 99,
            size: 4,
        }
    ));
}

#[test]
fn view_rejects_negative_level() {
    let mut bad = build_toy_fixture();
    // levels[2] is at file offset 0x90 + 4 = 0x94. Set it to -1.
    bad[0x94..0x96].copy_from_slice(&(-1i16).to_le_bytes());
    let err = HnswView::try_new(&bad).unwrap_err();
    assert!(matches!(
        err,
        HnswViewError::LevelOutOfRange { slot: 2, value: -1 }
    ));
}

#[test]
fn view_rejects_entry_level_disagreement() {
    let mut bad = build_toy_fixture();
    // levels[0] is at file offset 0x90; entry_slot=0 has level=2 in
    // graph header but we set the on-disk level to 1.
    bad[0x90..0x92].copy_from_slice(&1i16.to_le_bytes());
    let err = HnswView::try_new(&bad).unwrap_err();
    assert!(matches!(
        err,
        HnswViewError::EntryLevelMismatch {
            max_level: 2,
            entry_level: 1,
        }
    ));
}

#[test]
fn view_rejects_neighbor_count_exceeds_cap() {
    let mut bad = build_toy_fixture();
    // Node 0's base slab starts at 0x98 + 10 = 0xA2. Set count=99.
    bad[0xA2..0xA6].copy_from_slice(&99u32.to_le_bytes());
    let view = HnswView::try_new(&bad).unwrap();
    let n0 = view.try_node(0).unwrap();
    let err = n0.try_neighbors(0).unwrap_err();
    assert!(matches!(
        err,
        HnswViewError::NeighborCountExceedsCap {
            slot: 0,
            level: 0,
            count: 99,
            cap: 4,
        }
    ));
}

#[test]
fn view_rejects_neighbor_slot_out_of_range() {
    let mut bad = build_toy_fixture();
    // Node 0's base slab: count at 0xA2, first neighbor slot at 0xA6.
    // Set first neighbor to slot 99 (>= node_count=4).
    bad[0xA6..0xAA].copy_from_slice(&99u32.to_le_bytes());
    let view = HnswView::try_new(&bad).unwrap();
    let n0 = view.try_node(0).unwrap();
    let err = n0.try_neighbors(0).unwrap_err();
    assert!(matches!(
        err,
        HnswViewError::NeighborSlotOutOfRange {
            slot: 0,
            level: 0,
            position: 0,
            neighbor: 99,
            node_count: 4,
        }
    ));
}

#[test]
fn view_does_not_panic_on_random_garbage() {
    // Same shape as the header smoke: 1024 random byte sequences must
    // never panic. Crude proxy for the Phase 5 cargo-fuzz target.
    let seed = 0x1234_5678_9ABC_DEF0u64;
    let mut state = seed;
    for size_pick in [16, 64, 200, 308, 512, 1024] {
        for _ in 0..256 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let mut bytes = vec![0u8; size_pick];
            for (i, b) in bytes.iter_mut().enumerate() {
                *b = ((state >> (i % 56)) & 0xFF) as u8;
            }
            let _ = HnswView::try_new(&bytes);
        }
    }
}

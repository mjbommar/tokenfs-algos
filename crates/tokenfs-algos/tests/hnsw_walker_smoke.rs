//! HNSW walker integration smoke test.
//!
//! Exercises the v0.7.0 Phase 1 public API
//! (`HnswView::try_new` + `try_search`) end-to-end against three
//! hand-crafted toy fixtures covering the load-bearing edge cases for
//! Algorithm 5 / Algorithm 2:
//!
//! - **Single-node** (entry-point-only). Algorithm 5's upper-layer
//!   descent loop must do nothing; the base-layer search must return
//!   the entry point.
//! - **Flat multi-node** (level 0 only). All four nodes at layer 0;
//!   no upper-layer descent at all.
//! - **Two-level** (the same 4-node, max-level-2 fixture used by the
//!   in-crate unit tests). Exercises the descent loop properly.
//!
//! Because integration tests can only see the crate's `pub` API, all
//! fixture construction is inlined here; we cannot import the
//! `pub(crate) fn build_toy_fixture` helper from the in-crate tests
//! module.

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)] // Integration test code; panic on Err is the desired failure mode.

use tokenfs_algos::similarity::hnsw::{HnswView, MetricKind, ScalarKind, SearchConfig, try_search};

const TOY_KEY_BASE: u64 = 0x1000_0000_0000_0000;

// ---------------------------------------------------------------------
// Fixture construction primitives — same shape as the in-crate
// `tests.rs` helpers, but reachable from this external integration
// test file.
// ---------------------------------------------------------------------

/// Common 64-byte dense head with the supplied metric / scalar /
/// dimensions / count / multi flag. Patterns for tests below.
fn write_dense_head(buf: &mut Vec<u8>, metric: u8, scalar: u8, dimensions: u64, count: u64) {
    buf.extend_from_slice(b"usearch");
    buf.extend_from_slice(&2u16.to_le_bytes()); // version_major
    buf.extend_from_slice(&25u16.to_le_bytes()); // version_minor
    buf.extend_from_slice(&1u16.to_le_bytes()); // version_patch
    buf.push(metric);
    buf.push(scalar);
    buf.push(14); // key_kind = u64_k
    buf.push(15); // slot_kind = u32_k
    buf.extend_from_slice(&count.to_le_bytes()); // count_present
    buf.extend_from_slice(&0u64.to_le_bytes()); // count_deleted
    buf.extend_from_slice(&dimensions.to_le_bytes());
    buf.push(0); // multi = false
    buf.extend_from_slice(&[0u8; 22]); // 22 reserved zero bytes
}

fn write_graph_header(
    buf: &mut Vec<u8>,
    size: u64,
    connectivity: u64,
    connectivity_base: u64,
    max_level: u64,
    entry_slot: u64,
) {
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&connectivity.to_le_bytes());
    buf.extend_from_slice(&connectivity_base.to_le_bytes());
    buf.extend_from_slice(&max_level.to_le_bytes());
    buf.extend_from_slice(&entry_slot.to_le_bytes());
}

fn write_node_head(buf: &mut Vec<u8>, key: u64, level: i16) {
    buf.extend_from_slice(&key.to_le_bytes());
    buf.extend_from_slice(&level.to_le_bytes());
}

fn write_slab(buf: &mut Vec<u8>, neighbors: &[u32], cap: u32) {
    assert!(neighbors.len() <= cap as usize);
    buf.extend_from_slice(&(neighbors.len() as u32).to_le_bytes());
    for &n in neighbors {
        buf.extend_from_slice(&n.to_le_bytes());
    }
    for _ in neighbors.len()..cap as usize {
        buf.extend_from_slice(&0u32.to_le_bytes());
    }
}

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

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

/// Single-node, max-level-0, 8-dim u8 L2² index.
///
/// File layout: vectors header (8) + 1 vector (8) + dense head (64) +
/// graph header (40) + levels[0] (2) + node-0 tape (10 head + 20 base
/// slab) = 152 bytes. Base slab cap = 4 (M0); count = 0 (no edges).
fn build_single_node_fixture() -> Vec<u8> {
    let mut buf = Vec::with_capacity(152);

    // Vectors header + 1 vector.
    buf.extend_from_slice(&1u32.to_le_bytes()); // rows = 1
    buf.extend_from_slice(&8u32.to_le_bytes()); // cols = 8
    buf.extend_from_slice(&toy_vector(0));
    assert_eq!(buf.len(), 16);

    // Dense head.
    write_dense_head(&mut buf, b'e', 17, 8, 1); // L2², u8, dim=8, count=1
    assert_eq!(buf.len(), 80);

    // Graph header.
    write_graph_header(&mut buf, 1, 2, 4, 0, 0); // size=1 M=2 M0=4 max_level=0 entry=0
    assert_eq!(buf.len(), 120);

    // levels[0] = 0
    buf.extend_from_slice(&0i16.to_le_bytes());

    // Node-0 tape: key + level + base slab (cap 4).
    write_node_head(&mut buf, TOY_KEY_BASE, 0);
    write_slab(&mut buf, &[], 4); // no neighbors
    assert_eq!(buf.len(), 152);

    buf
}

/// Multi-node flat (max-level-0), 4 nodes, 8-dim u8 L2² index.
///
/// All four nodes at layer 0; entry point is slot 0. Each node's base
/// slab references the other three. M0=4 (slab cap 4). Tests
/// Algorithm 5 with no upper-layer descent.
///
/// Total: 8 + 32 + 64 + 40 + 8 + 4 × 30 = 272 bytes.
fn build_flat_4node_fixture() -> Vec<u8> {
    let mut buf = Vec::with_capacity(272);

    // Vectors header + 4 vectors.
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    for slot in 0..4 {
        buf.extend_from_slice(&toy_vector(slot));
    }
    assert_eq!(buf.len(), 40);

    // Dense head.
    write_dense_head(&mut buf, b'e', 17, 8, 4);
    assert_eq!(buf.len(), 104);

    // Graph header.
    write_graph_header(&mut buf, 4, 2, 4, 0, 0); // max_level = 0 → flat
    assert_eq!(buf.len(), 144);

    // levels[0..3] all 0
    for _ in 0..4 {
        buf.extend_from_slice(&0i16.to_le_bytes());
    }
    assert_eq!(buf.len(), 152);

    // Node tapes: each node connects to all others at layer 0.
    let neighbors_per_slot: [&[u32]; 4] = [&[1, 2, 3], &[0, 2, 3], &[0, 1, 3], &[0, 1, 2]];
    for (slot, &neighbors) in neighbors_per_slot.iter().enumerate() {
        write_node_head(&mut buf, TOY_KEY_BASE | slot as u64, 0);
        write_slab(&mut buf, neighbors, 4);
    }
    assert_eq!(buf.len(), 272);

    buf
}

/// Same 4-node, max-level-2 fixture as the in-crate `build_toy_fixture()`.
fn build_two_level_fixture() -> Vec<u8> {
    let mut buf = Vec::with_capacity(308);

    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());
    for slot in 0..4 {
        buf.extend_from_slice(&toy_vector(slot));
    }
    write_dense_head(&mut buf, b'e', 17, 8, 4);
    write_graph_header(&mut buf, 4, 2, 4, 2, 0);

    // levels: node 0=2, node 1=1, node 2=0, node 3=0
    for level in [2i16, 1, 0, 0] {
        buf.extend_from_slice(&level.to_le_bytes());
    }

    // Node 0 (level 2): base→{1,2,3}; level-1→{1}; level-2→{}
    write_node_head(&mut buf, TOY_KEY_BASE, 2);
    write_slab(&mut buf, &[1, 2, 3], 4);
    write_slab(&mut buf, &[1], 2);
    write_slab(&mut buf, &[], 2);

    // Node 1 (level 1): base→{0,2,3}; level-1→{0}
    write_node_head(&mut buf, TOY_KEY_BASE | 1, 1);
    write_slab(&mut buf, &[0, 2, 3], 4);
    write_slab(&mut buf, &[0], 2);

    // Node 2 (level 0): base→{0,1,3}
    write_node_head(&mut buf, TOY_KEY_BASE | 2, 0);
    write_slab(&mut buf, &[0, 1, 3], 4);

    // Node 3 (level 0): base→{0,1,2}
    write_node_head(&mut buf, TOY_KEY_BASE | 3, 0);
    write_slab(&mut buf, &[0, 1, 2], 4);

    assert_eq!(buf.len(), 308);
    buf
}

// ---------------------------------------------------------------------
// Smoke tests
// ---------------------------------------------------------------------

#[test]
fn single_node_index_returns_entry_point() {
    let fixture = build_single_node_fixture();
    let view = HnswView::try_new(&fixture).expect("single-node fixture parses");
    assert_eq!(view.node_count(), 1);
    assert_eq!(view.max_level(), 0);
    assert_eq!(view.entry_point(), Some(0));
    assert_eq!(view.header().metric_kind(), MetricKind::L2Squared);
    assert_eq!(view.header().scalar_kind(), ScalarKind::U8);

    let cfg = SearchConfig::new(1, 4);
    let result = try_search(&view, &toy_vector(0), &cfg).expect("search");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, TOY_KEY_BASE); // exact match
    assert_eq!(result[0].1, 0); // distance 0
}

#[test]
fn single_node_index_query_other_returns_only_node() {
    let fixture = build_single_node_fixture();
    let view = HnswView::try_new(&fixture).unwrap();

    // Query bytes don't match any vector — single-node index still
    // returns the only node it has.
    let query = [0xFFu8; 8];
    let cfg = SearchConfig::new(5, 8);
    let result = try_search(&view, &query, &cfg).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, TOY_KEY_BASE);
}

#[test]
fn flat_4node_index_returns_top_k_in_distance_order() {
    let fixture = build_flat_4node_fixture();
    let view = HnswView::try_new(&fixture).expect("flat fixture parses");
    assert_eq!(view.max_level(), 0);
    assert_eq!(view.node_count(), 4);

    // Query = node 0's vector → node 0 first.
    let cfg = SearchConfig::new(4, 8);
    let result = try_search(&view, &toy_vector(0), &cfg).unwrap();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].0, TOY_KEY_BASE);
    assert_eq!(result[0].1, 0);
    for window in result.windows(2) {
        assert!(window[0].1 <= window[1].1);
    }
}

#[test]
fn flat_4node_index_returns_only_three_when_k_equals_three() {
    let fixture = build_flat_4node_fixture();
    let view = HnswView::try_new(&fixture).unwrap();

    let cfg = SearchConfig::new(3, 8);
    let result = try_search(&view, &toy_vector(2), &cfg).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].0, TOY_KEY_BASE | 2); // exact match first
}

#[test]
fn two_level_index_returns_top_k() {
    let fixture = build_two_level_fixture();
    let view = HnswView::try_new(&fixture).unwrap();
    assert_eq!(view.max_level(), 2);
    assert_eq!(view.node_count(), 4);

    let cfg = SearchConfig::new(2, 4);
    let result = try_search(&view, &toy_vector(0), &cfg).unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].0, TOY_KEY_BASE);
    assert_eq!(result[0].1, 0);
}

#[test]
fn two_level_index_walker_matches_brute_force() {
    let fixture = build_two_level_fixture();
    let view = HnswView::try_new(&fixture).unwrap();

    // Query that doesn't exactly match any vector.
    let query = [0x18u8, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F];

    // Brute-force oracle: compute L2² to every node, sort.
    let mut brute: Vec<(u64, u32)> = (0..4u32)
        .map(|slot| {
            let n = view.try_node(slot).unwrap();
            let v = n.vector_bytes();
            let d: u32 = query
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| {
                    let d = (a as i32) - (b as i32);
                    (d * d) as u32
                })
                .sum();
            (n.key(), d)
        })
        .collect();
    brute.sort_by_key(|(_, d)| *d);

    let cfg = SearchConfig::new(4, 8);
    let walker_result = try_search(&view, &query, &cfg).unwrap();

    assert_eq!(walker_result, brute);
}

#[test]
fn cross_fixture_queries_dont_panic_on_mismatched_dim() {
    // Smoke: any well-formed fixture + any query with the right
    // dimension must return without panicking. Loop over the three
    // fixtures and a handful of synthetic queries.
    let fixtures = [
        build_single_node_fixture(),
        build_flat_4node_fixture(),
        build_two_level_fixture(),
    ];
    for fixture in &fixtures {
        let view = HnswView::try_new(fixture).unwrap();
        for seed in 0u8..16 {
            let query = [seed; 8];
            let cfg = SearchConfig::new(2, 4);
            let _ = try_search(&view, &query, &cfg).unwrap();
        }
    }
}

#[test]
fn wrong_query_length_returns_error_not_panic() {
    let fixture = build_two_level_fixture();
    let view = HnswView::try_new(&fixture).unwrap();
    let cfg = SearchConfig::new(2, 4);
    let err = try_search(&view, &[0u8; 16], &cfg).unwrap_err();
    // Exact variant content is checked in the in-crate unit tests; here
    // we just confirm the error path doesn't panic.
    let msg = format!("{err}");
    assert!(msg.contains("length") || msg.contains("query"));
}

# Component: Wire format

**Status:** Phase 1 read path landed 2026-05-04 (commits f985337, dac8e3c). Write path lands in Phase 4 (`build/serialize.rs`).

**Lives in:** `crates/tokenfs-algos/src/similarity/hnsw/{header.rs, view.rs, build/serialize.rs}`

## Phase 1 implementation summary

Two modules carry the read path; write path arrives Phase 4.

### `header.rs` (~380 LOC, 17 tests)

Parses + validates the 64-byte `index_dense_head_t` per
`USEARCH_DEEP_DIVE.md` §1.3. Surface:

- `HEADER_BYTES = 64` (locked by usearch's `static_assert`).
- `MetricKind` enum — char-coded (`'i'`/`'c'`/`'e'`/`'b'`/`'t'`/`'s'`/`'j'`).
- `ScalarKind` enum — numeric codes 0..23 per usearch's
  `index_plugins.hpp:139-164`. v0.7.0 ships kernels for B1x8, F32,
  I8, U8 only; other variants parse correctly but the walker
  rejects them with `UnsupportedMetricScalar`.
- `HnswHeader::try_parse(&[u8])` — kernel-safe, never panics.
- `HnswHeaderError` covers every failure mode:
  `Truncated`, `WrongMagic`, `UnsupportedFormatVersion` (rejects
  pre-v2.25), `UnknownMetricKind`, `UnknownScalarKind`,
  `UnsupportedKeyKind` (rejects non-u64), `UnsupportedSlotKind`
  (rejects u40 / `index_dense_big_t`), `ZeroDimensions`,
  `InvalidMultiFlag`.

The `#[cfg(feature = "userspace")] HnswHeader::parse` userspace
shim is gated; kernel-default builds only see `try_parse`.

### `view.rs` (~660 LOC, 17 tests)

Zero-copy view over the full serialized index (vectors blob +
dense head + graph blob) per `USEARCH_DEEP_DIVE.md` §1.5–§1.8. Surface:

- `GRAPH_HEADER_BYTES = 40` (locked by `index_serialized_header_t`).
- `GraphHeader { size, connectivity, connectivity_base, max_level, entry_slot }`.
- `HnswView::try_new(bytes)` — full structural validation in one
  O(N) pass. Catches CVE-2023-37365-class bounds bugs at
  construction time.
- `HnswView::{node_count, dimensions, scalar_kind, bytes_per_vector,
  max_level, entry_point, connectivity, connectivity_base, header,
  graph_header}`.
- `HnswView::try_node(NodeId) → Result<NodeRef<'a>, HnswViewError>`.
- `NodeRef::{slot, key, level, vector_bytes, try_neighbors(level) → NeighborSlice}`.
- `NeighborSlice::{len, is_empty, get(i), iter()}` — exposes u32
  slot IDs without a misaligned `&[u32]` transmute (slabs sit at
  non-4-byte-aligned offsets within tapes).

`HnswViewError` covers every failure mode at every layer of the
parse, including:
- structural truncation (vectors header / vectors blob / graph
  header / levels array / node tape),
- dense-head cross-checks (cols vs `bytes_per_vector`, rows vs
  graph-header `size`),
- graph-header invariants (connectivity ≥ 2, connectivity_base ≥
  connectivity, entry_slot < size, max_level fits u8),
- per-node `levels[i]` out-of-range,
- `levels[entry_slot] != max_level` mismatch,
- `try_neighbors` validation: count ≤ cap, every live neighbor
  slot < node_count.

Allocation footprint: only the precomputed per-node offset table
(`Vec<u32>`) and per-node level table (`Vec<u8>`), both sized to
`node_count` at construction. After `try_new`, every accessor is
O(1) byte arithmetic.

## Role

Adopt the usearch v2.25 serialized index format byte-for-byte. The walker reads this format from mmapped section bytes; the builder writes this format. Multi-language reader compatibility (DuckDB, ClickHouse, pgvector) flows from this choice.

## Required research input

- [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md) §1 (header layout) + §2 (graph topology + addressing) + §7 (save/load/view)

## Sections to fill in (during Phase 1)

1. **64-byte header layout.** Annotated field-by-field per `_references/usearch/include/usearch/index_dense.hpp:42-79`. Endianness, alignment, padding semantics.
2. **Body layout.** Per-node tape format. Where vectors live (interleaved / separate blob). How neighbor lists are packed per level. Where uint40_t is used and why.
3. **TokenFS section wrapper.** 8-byte prefix per `IMAGE_FORMAT_v0.3 §0x203`: `magic[4] = "HNSW"`, `section_version = 0x01`, `reserved[3] = 0`. usearch blob starts at byte 8 of the section.
4. **Reader: zero-copy view.** `HnswView<'a>` wraps `&'a [u8]`, computes node-table offsets at construction, supports `node_count()` / `dimensions()` / `entry_point()` / `try_node(NodeId)` with bounds checks.
5. **Writer: serialize.** `serialize_to_bytes(&Graph, &BuildConfig) -> Vec<u8>` writes the header + per-node tape in one allocation pass.
6. **Forward-compat strategy.** Pin to v2.25.x. Future format versions land at a new section ID per `IMAGE_FORMAT_v0.3 §11`. Header parser fails closed on unknown versions.
7. **Determinism.** Wire-format byte ordering is fully specified by usearch; no implementation freedom. Round-trip test asserts byte-identical between our writer and libusearch v2.25 single-thread.
8. **Concrete byte-level diagram.** A 4-node, 2-level, 8-byte-vector toy index annotated end-to-end.

## API skeleton

```rust
// header.rs
pub struct HnswHeader { /* opaque */ }
pub enum HnswHeaderError { WrongMagic, UnsupportedFormatVersion { version }, ... }
pub fn try_parse_header(bytes: &[u8]) -> Result<HnswHeader, HnswHeaderError>;
pub const HEADER_BYTES: usize = 64;

// view.rs
pub struct HnswView<'a> { /* opaque, borrows &'a [u8] */ }
pub enum HnswViewError { Header(HnswHeaderError), TruncatedBody, NodeIdOutOfBounds, ... }
impl<'a> HnswView<'a> {
    pub fn try_new(bytes: &'a [u8]) -> Result<Self, HnswViewError>;
    pub fn node_count(&self) -> usize;
    pub fn dimensions(&self) -> usize;
    pub fn entry_point(&self) -> Option<NodeId>;
    pub fn try_node(&self, id: NodeId) -> Result<NodeRef<'a>, HnswViewError>;
}
pub struct NodeRef<'a> { /* opaque */ }
impl<'a> NodeRef<'a> {
    pub fn try_neighbors(&self, level: u8) -> Result<&'a [NodeId], HnswViewError>;
    pub fn vector_bytes(&self) -> &'a [u8];
    pub fn level(&self) -> u8;
}

// build/serialize.rs (cfg(feature = "std"))
pub fn serialize_graph_to_bytes(
    graph: &Graph,
    config: &BuildConfig,
) -> Result<Vec<u8>, HnswBuildError>;
```

## Cross-references

- Phase: [`../phases/PHASE_1.md`](../phases/PHASE_1.md) (delivers parser + view + writer skeleton)
- Research: [`../research/USEARCH_DEEP_DIVE.md`](../research/USEARCH_DEEP_DIVE.md)
- Spec: `tokenfs-paper/docs/IMAGE_FORMAT_v0.3.md` §0x203 (the section ID this format lives at)
- Source: `_references/usearch/include/usearch/index_dense.hpp:42-79` (header struct definition)

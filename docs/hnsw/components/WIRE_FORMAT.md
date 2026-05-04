# Component: Wire format

**Status:** skeleton, 2026-05-03. Filled in during Phase 1.

**Lives in:** `crates/tokenfs-algos/src/similarity/hnsw/{header.rs, view.rs, build/serialize.rs}`

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

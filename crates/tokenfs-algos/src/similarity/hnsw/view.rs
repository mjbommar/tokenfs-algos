//! Zero-copy view over a usearch v2.25 serialized dense index.
//!
//! Layout per `docs/hnsw/research/USEARCH_DEEP_DIVE.md` §1.5–1.8 and
//! `_references/usearch/include/usearch/index.hpp:3411-3452, 3638-3707`.
//!
//! ```text
//! [ vectors blob ]  [ 64-byte dense head ]  [ HNSW graph blob ]
//!        |                  |                         |
//!     §1.2              §1.3 (header.rs)        §1.5–1.6 (here)
//! ```
//!
//! Validates the entire structure up-front (graph header, levels array,
//! every node tape, every neighbor slab fits within the byte slice).
//! After successful [`HnswView::try_new`], every accessor is
//! O(1) byte arithmetic with no further validation needed.
//!
//! # Audit posture
//!
//! - Public entry [`HnswView::try_new`] returns `Result`. Never panics
//!   on caller input. Catches CVE-2023-37365-class bounds bugs at
//!   construction.
//! - All node / neighbor lookups return `Result`. Out-of-bounds is an
//!   error variant, not a panic.
//! - `no_std + alloc`-clean: only allocation is the precomputed
//!   per-node offset table (`Vec<u32>` of length `node_count`).

// Module-level cfg gate (in mod.rs) ensures we're only compiled when
// `feature = "std"` or `feature = "alloc"` is enabled, so `Vec` is
// always available via either prelude or the crate-root extern.
#[cfg(all(feature = "alloc", not(feature = "std")))]
use alloc::vec::Vec;

use super::header::{HnswHeader, HnswHeaderError, ScalarKind};
use super::{NodeId, NodeKey};

/// Size of the graph header (`index_serialized_header_t`).
/// `_references/usearch/include/usearch/index.hpp:1990-1996`.
pub const GRAPH_HEADER_BYTES: usize = 40;

/// Size of the vectors-blob shape header when 32-bit dimensions are used
/// (the default; see `serialization_config_t::use_64_bit_dimensions`).
const VECTORS_HEADER_32BIT_BYTES: usize = 8;

/// Size of `node_head_bytes_()` for the default `index_dense_t`:
/// `sizeof(u64 key) + sizeof(i16 level)` = 10 bytes.
const NODE_HEAD_BYTES: usize = 10;

/// Size of the slot count prefix at the start of every neighbor slab.
const NEIGHBORS_COUNT_BYTES: usize = 4;

/// Size of one slot ID in the v0.7.0 wire format (u32; `index_dense_t`).
const SLOT_BYTES: usize = 4;

/// Parsed graph header — the 40 bytes that immediately follow the
/// dense head. See `index.hpp:1990-1996`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphHeader {
    /// Number of node entries (must equal vectors blob row count).
    pub size: u64,
    /// `M` from the HNSW paper. Cap on edges per non-base layer.
    pub connectivity: u64,
    /// `M0`. Cap on edges in the base layer (typically `2 * M`).
    pub connectivity_base: u64,
    /// Top level (`level_t`) of the entry node. 0 means a flat graph.
    pub max_level: u64,
    /// Slot ID of the entry node. May reference any slot in `0..size`.
    pub entry_slot: u64,
}

/// Errors produced by [`HnswView::try_new`] and accessor methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HnswViewError {
    /// Dense-head parse failure (delegated to [`HnswHeader::try_parse`]).
    Header(HnswHeaderError),
    /// Vectors blob shorter than the 8-byte rows/cols header.
    VectorsHeaderTruncated {
        /// Bytes provided.
        got: usize,
    },
    /// Vectors blob's `cols` field disagrees with the dense head's
    /// `bytes_per_vector` (`dimensions * bits_per_scalar / 8`).
    VectorsColsMismatch {
        /// `cols` field read from the vectors blob.
        got: u32,
        /// Expected from dense-head dimensions and scalar kind.
        expected: u64,
    },
    /// Vectors blob's `rows` field disagrees with the graph header's `size`.
    VectorsRowsMismatch {
        /// `rows` field read from the vectors blob.
        got: u32,
        /// Expected from graph header.
        expected: u64,
    },
    /// Vectors blob ends before all `rows * cols` bytes are available.
    VectorsBlobTruncated {
        /// Bytes available after the vectors header.
        got: usize,
        /// Bytes needed.
        need: u64,
    },
    /// File ends before the graph header is reachable.
    GraphHeaderTruncated {
        /// Bytes available where the graph header should start.
        got: usize,
    },
    /// `connectivity` is zero — invalid index (would imply zero edges
    /// per non-base layer).
    InvalidConnectivity {
        /// Value found.
        got: u64,
    },
    /// `connectivity_base` is less than `connectivity` — invalid index.
    InvalidConnectivityBase {
        /// `connectivity_base` found.
        got: u64,
        /// `connectivity` found.
        connectivity: u64,
    },
    /// Graph header size is too large to address (slots would overflow [`NodeId`]).
    NodeCountOverflow {
        /// `size` field from graph header.
        got: u64,
    },
    /// `entry_slot >= size` — entry point out of range.
    EntryPointOutOfRange {
        /// `entry_slot` value.
        entry_slot: u64,
        /// `size` field from graph header.
        size: u64,
    },
    /// `max_level` is too large to fit in `u8` (we cap at 255 per
    /// `BuildConfig::max_level` future field).
    MaxLevelTooLarge {
        /// Value found.
        got: u64,
    },
    /// `levels[i]` for some `i` is negative or > `u8::MAX`.
    LevelOutOfRange {
        /// Slot index.
        slot: NodeId,
        /// Value found (i16 from disk).
        value: i16,
    },
    /// `levels[entry_slot] != max_level` — the entry node's recorded
    /// level disagrees with the graph header.
    EntryLevelMismatch {
        /// `max_level` from graph header.
        max_level: u8,
        /// Level read from `levels[entry_slot]`.
        entry_level: u8,
    },
    /// File truncated mid-levels-array.
    LevelsArrayTruncated {
        /// Bytes available where the levels array should start.
        got: usize,
        /// Bytes needed (`size * 2`).
        need: u64,
    },
    /// File truncated mid-node-tape.
    NodeTapeTruncated {
        /// Slot whose tape would extend past EOF.
        slot: NodeId,
        /// Tape's expected start offset.
        tape_offset: u64,
        /// Bytes available from `tape_offset`.
        got: u64,
        /// Bytes needed for this node tape.
        need: u64,
    },
    /// Total graph blob byte length would exceed `usize::MAX` on the host.
    OffsetOverflow,
    /// [`HnswView::try_node`] called with a slot >= `node_count`.
    NodeIdOutOfRange {
        /// Slot requested.
        slot: NodeId,
        /// Number of nodes in the index.
        node_count: usize,
    },
    /// [`NodeRef::try_neighbors`] called with `level > node.level()`.
    NeighborLevelOutOfRange {
        /// Slot whose neighbors were queried.
        slot: NodeId,
        /// Level requested.
        level: u8,
        /// Node's max level.
        node_level: u8,
    },
    /// A neighbor slab's `count` exceeds its `cap` (corrupt / hostile input).
    NeighborCountExceedsCap {
        /// Slot whose neighbor list violated the invariant.
        slot: NodeId,
        /// Level whose slab violated the invariant.
        level: u8,
        /// Count read from disk.
        count: u32,
        /// Slab cap (M0 for base, M for upper levels).
        cap: u32,
    },
    /// A neighbor slot ID is >= `node_count`.
    NeighborSlotOutOfRange {
        /// Slot whose neighbor list contained the bad reference.
        slot: NodeId,
        /// Level of the slab.
        level: u8,
        /// Position within the neighbor list.
        position: usize,
        /// Bad slot ID found.
        neighbor: u32,
        /// Number of nodes in the index.
        node_count: usize,
    },
}

impl From<HnswHeaderError> for HnswViewError {
    fn from(value: HnswHeaderError) -> Self {
        HnswViewError::Header(value)
    }
}

impl core::fmt::Display for HnswViewError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Header(inner) => write!(f, "header: {inner}"),
            Self::VectorsHeaderTruncated { got } => {
                write!(f, "vectors blob truncated; got {got} bytes, need 8")
            }
            Self::VectorsColsMismatch { got, expected } => write!(
                f,
                "vectors blob cols={got} disagrees with dense-head bytes-per-vector={expected}"
            ),
            Self::VectorsRowsMismatch { got, expected } => write!(
                f,
                "vectors blob rows={got} disagrees with graph header size={expected}"
            ),
            Self::VectorsBlobTruncated { got, need } => write!(
                f,
                "vectors blob payload truncated; got {got} bytes, need {need}"
            ),
            Self::GraphHeaderTruncated { got } => write!(
                f,
                "graph header truncated; got {got} bytes, need {GRAPH_HEADER_BYTES}"
            ),
            Self::InvalidConnectivity { got } => write!(f, "connectivity={got} is invalid"),
            Self::InvalidConnectivityBase { got, connectivity } => {
                write!(f, "connectivity_base={got} < connectivity={connectivity}")
            }
            Self::NodeCountOverflow { got } => write!(f, "node count {got} overflows NodeId"),
            Self::EntryPointOutOfRange { entry_slot, size } => {
                write!(f, "entry_slot={entry_slot} >= size={size}")
            }
            Self::MaxLevelTooLarge { got } => write!(f, "max_level={got} exceeds u8::MAX"),
            Self::LevelOutOfRange { slot, value } => {
                write!(f, "levels[{slot}]={value} is negative or exceeds u8::MAX")
            }
            Self::EntryLevelMismatch {
                max_level,
                entry_level,
            } => write!(
                f,
                "graph header max_level={max_level} disagrees with levels[entry_slot]={entry_level}"
            ),
            Self::LevelsArrayTruncated { got, need } => {
                write!(f, "levels array truncated; got {got} bytes, need {need}")
            }
            Self::NodeTapeTruncated {
                slot,
                tape_offset,
                got,
                need,
            } => write!(
                f,
                "node {slot} tape truncated at offset {tape_offset}; got {got} bytes, need {need}"
            ),
            Self::OffsetOverflow => f.write_str("computed graph blob offset overflows usize"),
            Self::NodeIdOutOfRange { slot, node_count } => {
                write!(f, "node id {slot} out of range; node_count={node_count}")
            }
            Self::NeighborLevelOutOfRange {
                slot,
                level,
                node_level,
            } => write!(
                f,
                "node {slot} has level {node_level}; cannot query level {level}"
            ),
            Self::NeighborCountExceedsCap {
                slot,
                level,
                count,
                cap,
            } => write!(
                f,
                "node {slot} level {level}: neighbor count {count} > slab cap {cap}"
            ),
            Self::NeighborSlotOutOfRange {
                slot,
                level,
                position,
                neighbor,
                node_count,
            } => write!(
                f,
                "node {slot} level {level} neighbor[{position}]={neighbor} >= node_count={node_count}"
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HnswViewError {}

/// Zero-copy view over a serialized usearch v2.25 dense index.
///
/// Constructed with [`HnswView::try_new`]. After construction every
/// accessor is O(1) byte arithmetic.
#[derive(Debug)]
pub struct HnswView<'a> {
    bytes: &'a [u8],
    header: HnswHeader,
    graph_header: GraphHeader,
    /// Offset of the first vector's raw bytes in `bytes`. (Skips the
    /// 8-byte rows/cols vectors-header.)
    vectors_data_offset: usize,
    /// Bytes per vector row. Equals
    /// `header.bytes_per_vector()` (we cache as `usize`).
    bytes_per_vector: usize,
    /// Per-node tape start offsets, precomputed from the on-disk
    /// `levels[]` exclusive scan. Length == `node_count`.
    node_offsets: Vec<u32>,
    /// Per-node level (cached from on-disk `levels[]`).
    node_levels: Vec<u8>,
    /// `connectivity_base * SLOT_BYTES + 4` — base-layer slab size.
    neighbors_base_bytes: usize,
    /// `connectivity * SLOT_BYTES + 4` — non-base-layer slab size.
    neighbors_bytes: usize,
}

impl<'a> HnswView<'a> {
    /// Parse + validate a full usearch v2.25 serialized dense index.
    ///
    /// Validates: vectors blob shape header, dense head, graph header,
    /// levels array, every node tape's existence within `bytes`. Does
    /// NOT validate neighbor-list contents (those are checked lazily
    /// per-access in [`NodeRef::try_neighbors`]).
    pub fn try_new(bytes: &'a [u8]) -> Result<Self, HnswViewError> {
        // ---- 1. vectors blob shape header (8 bytes, 32-bit dims default)
        if bytes.len() < VECTORS_HEADER_32BIT_BYTES {
            return Err(HnswViewError::VectorsHeaderTruncated { got: bytes.len() });
        }
        let vectors_rows =
            u32::from_le_bytes(bytes[0..4].try_into().expect("0..4 is exactly 4 bytes"));
        let vectors_cols =
            u32::from_le_bytes(bytes[4..8].try_into().expect("4..8 is exactly 4 bytes"));
        let vectors_data_offset = VECTORS_HEADER_32BIT_BYTES;

        // ---- 2. dense head (parses + validates magic / version / kinds)
        let dense_head_offset = vectors_data_offset
            .checked_add(
                (vectors_rows as usize)
                    .checked_mul(vectors_cols as usize)
                    .ok_or(HnswViewError::OffsetOverflow)?,
            )
            .ok_or(HnswViewError::OffsetOverflow)?;
        if bytes.len() < dense_head_offset {
            return Err(HnswViewError::VectorsBlobTruncated {
                got: bytes.len().saturating_sub(vectors_data_offset),
                need: (vectors_rows as u64) * (vectors_cols as u64),
            });
        }
        let dense_head_end = dense_head_offset
            .checked_add(super::HEADER_BYTES)
            .ok_or(HnswViewError::OffsetOverflow)?;
        if bytes.len() < dense_head_end {
            return Err(HnswViewError::Header(HnswHeaderError::Truncated {
                got: bytes.len().saturating_sub(dense_head_offset),
                need: super::HEADER_BYTES,
            }));
        }
        let header = HnswHeader::try_parse(&bytes[dense_head_offset..dense_head_end])?;

        // ---- 3. cross-check vectors blob against dense-head shape
        let expected_bytes_per_vector = header.bytes_per_vector();
        if vectors_cols as u64 != expected_bytes_per_vector {
            return Err(HnswViewError::VectorsColsMismatch {
                got: vectors_cols,
                expected: expected_bytes_per_vector,
            });
        }
        let bytes_per_vector = usize::try_from(expected_bytes_per_vector)
            .map_err(|_| HnswViewError::OffsetOverflow)?;

        // ---- 4. graph header (40 bytes after dense head)
        let graph_header_offset = dense_head_end;
        let graph_header_end = graph_header_offset
            .checked_add(GRAPH_HEADER_BYTES)
            .ok_or(HnswViewError::OffsetOverflow)?;
        if bytes.len() < graph_header_end {
            return Err(HnswViewError::GraphHeaderTruncated {
                got: bytes.len().saturating_sub(graph_header_offset),
            });
        }
        let graph_header = parse_graph_header(&bytes[graph_header_offset..graph_header_end]);

        // ---- 5. validate graph header invariants
        if graph_header.connectivity < 2 {
            return Err(HnswViewError::InvalidConnectivity {
                got: graph_header.connectivity,
            });
        }
        if graph_header.connectivity_base < graph_header.connectivity {
            return Err(HnswViewError::InvalidConnectivityBase {
                got: graph_header.connectivity_base,
                connectivity: graph_header.connectivity,
            });
        }
        let node_count =
            usize::try_from(graph_header.size).map_err(|_| HnswViewError::NodeCountOverflow {
                got: graph_header.size,
            })?;
        if vectors_rows as u64 != graph_header.size {
            return Err(HnswViewError::VectorsRowsMismatch {
                got: vectors_rows,
                expected: graph_header.size,
            });
        }
        if node_count > 0 && graph_header.entry_slot >= graph_header.size {
            return Err(HnswViewError::EntryPointOutOfRange {
                entry_slot: graph_header.entry_slot,
                size: graph_header.size,
            });
        }
        let max_level =
            u8::try_from(graph_header.max_level).map_err(|_| HnswViewError::MaxLevelTooLarge {
                got: graph_header.max_level,
            })?;

        // ---- 6. precompute slab byte sizes
        let neighbors_base_bytes = (graph_header.connectivity_base as usize)
            .checked_mul(SLOT_BYTES)
            .and_then(|x| x.checked_add(NEIGHBORS_COUNT_BYTES))
            .ok_or(HnswViewError::OffsetOverflow)?;
        let neighbors_bytes = (graph_header.connectivity as usize)
            .checked_mul(SLOT_BYTES)
            .and_then(|x| x.checked_add(NEIGHBORS_COUNT_BYTES))
            .ok_or(HnswViewError::OffsetOverflow)?;

        // ---- 7. levels array (i16 LE per slot, immediately after graph header)
        let levels_offset = graph_header_end;
        let levels_byte_len = node_count
            .checked_mul(core::mem::size_of::<i16>())
            .ok_or(HnswViewError::OffsetOverflow)?;
        let levels_end = levels_offset
            .checked_add(levels_byte_len)
            .ok_or(HnswViewError::OffsetOverflow)?;
        if bytes.len() < levels_end {
            return Err(HnswViewError::LevelsArrayTruncated {
                got: bytes.len().saturating_sub(levels_offset),
                need: levels_byte_len as u64,
            });
        }

        // ---- 8. precompute per-node tape offsets via exclusive scan
        let mut node_offsets = Vec::with_capacity(node_count);
        let mut node_levels = Vec::with_capacity(node_count);
        let mut cursor: u64 = levels_end as u64;
        for slot_idx in 0..node_count {
            let level_off = levels_offset + slot_idx * 2;
            let raw = i16::from_le_bytes(
                bytes[level_off..level_off + 2]
                    .try_into()
                    .expect("2 bytes from in-range slice"),
            );
            if raw < 0 || raw > u8::MAX as i16 {
                return Err(HnswViewError::LevelOutOfRange {
                    slot: slot_idx as NodeId,
                    value: raw,
                });
            }
            let level = raw as u8;
            node_levels.push(level);
            let cursor_u32 = u32::try_from(cursor).map_err(|_| HnswViewError::OffsetOverflow)?;
            node_offsets.push(cursor_u32);
            let body = node_body_bytes(level, neighbors_base_bytes, neighbors_bytes);
            cursor = cursor
                .checked_add(body as u64)
                .ok_or(HnswViewError::OffsetOverflow)?;
        }
        if (bytes.len() as u64) < cursor {
            // Identify which slot's tape is the first to overflow so the
            // error can pinpoint the violation.
            let mut probe: u64 = levels_end as u64;
            for (slot_idx, &level) in node_levels.iter().enumerate() {
                let body = node_body_bytes(level, neighbors_base_bytes, neighbors_bytes);
                let need_end = probe.saturating_add(body as u64);
                if (bytes.len() as u64) < need_end {
                    return Err(HnswViewError::NodeTapeTruncated {
                        slot: slot_idx as NodeId,
                        tape_offset: probe,
                        got: (bytes.len() as u64).saturating_sub(probe),
                        need: body as u64,
                    });
                }
                probe = need_end;
            }
            return Err(HnswViewError::OffsetOverflow);
        }

        // ---- 9. validate entry-level matches levels[entry_slot]
        if node_count > 0 {
            let entry_level = node_levels[graph_header.entry_slot as usize];
            if entry_level != max_level {
                return Err(HnswViewError::EntryLevelMismatch {
                    max_level,
                    entry_level,
                });
            }
        }

        Ok(HnswView {
            bytes,
            header,
            graph_header,
            vectors_data_offset,
            bytes_per_vector,
            node_offsets,
            node_levels,
            neighbors_base_bytes,
            neighbors_bytes,
        })
    }

    /// Parsed dense-index header (the magic-bearing 64-byte region).
    pub fn header(&self) -> &HnswHeader {
        &self.header
    }

    /// Parsed graph header (the 40 bytes immediately after the dense head).
    pub fn graph_header(&self) -> GraphHeader {
        self.graph_header
    }

    /// Number of nodes in the index.
    pub fn node_count(&self) -> usize {
        self.node_offsets.len()
    }

    /// Logical scalar count per vector. Same as `header().dimensions()`
    /// but exposed here for ergonomic calls.
    pub fn dimensions(&self) -> u64 {
        self.header.dimensions()
    }

    /// Storage scalar type for vector elements.
    pub fn scalar_kind(&self) -> ScalarKind {
        self.header.scalar_kind()
    }

    /// Bytes per vector row in the vectors blob.
    pub fn bytes_per_vector(&self) -> usize {
        self.bytes_per_vector
    }

    /// Top level in the graph (0 means a flat graph).
    pub fn max_level(&self) -> u8 {
        self.graph_header.max_level as u8
    }

    /// Entry node slot. Returns `None` for an empty index.
    pub fn entry_point(&self) -> Option<NodeId> {
        if self.node_offsets.is_empty() {
            None
        } else {
            Some(self.graph_header.entry_slot as NodeId)
        }
    }

    /// `M` from the HNSW paper.
    pub fn connectivity(&self) -> u32 {
        self.graph_header.connectivity as u32
    }

    /// `M0` (base-layer connectivity cap; typically `2 * M`).
    pub fn connectivity_base(&self) -> u32 {
        self.graph_header.connectivity_base as u32
    }

    /// Look up a node by slot index. Returns `Err` if the slot is out
    /// of range.
    pub fn try_node(&self, slot: NodeId) -> Result<NodeRef<'a>, HnswViewError> {
        let idx = slot as usize;
        if idx >= self.node_offsets.len() {
            return Err(HnswViewError::NodeIdOutOfRange {
                slot,
                node_count: self.node_offsets.len(),
            });
        }
        let tape_offset = self.node_offsets[idx] as usize;
        let level = self.node_levels[idx];
        Ok(NodeRef {
            bytes: self.bytes,
            slot,
            level,
            tape_offset,
            vectors_data_offset: self.vectors_data_offset,
            bytes_per_vector: self.bytes_per_vector,
            neighbors_base_bytes: self.neighbors_base_bytes,
            neighbors_bytes: self.neighbors_bytes,
            connectivity: self.graph_header.connectivity as u32,
            connectivity_base: self.graph_header.connectivity_base as u32,
            node_count: self.node_offsets.len(),
        })
    }
}

/// Borrowed view of a single node in an [`HnswView`].
#[derive(Debug, Clone, Copy)]
pub struct NodeRef<'a> {
    bytes: &'a [u8],
    slot: NodeId,
    level: u8,
    tape_offset: usize,
    vectors_data_offset: usize,
    bytes_per_vector: usize,
    neighbors_base_bytes: usize,
    neighbors_bytes: usize,
    connectivity: u32,
    connectivity_base: u32,
    node_count: usize,
}

impl<'a> NodeRef<'a> {
    /// Slot index of this node in the graph.
    pub fn slot(&self) -> NodeId {
        self.slot
    }

    /// External caller-supplied key (the first 8 bytes of the tape).
    pub fn key(&self) -> NodeKey {
        let off = self.tape_offset;
        u64::from_le_bytes(
            self.bytes[off..off + 8]
                .try_into()
                .expect("validated at try_new"),
        )
    }

    /// Top level the node participates in (`levels[slot]` from the
    /// on-disk levels array). 0 means base-layer-only.
    pub fn level(&self) -> u8 {
        self.level
    }

    /// Raw vector bytes from the vectors blob (`bytes_per_vector`
    /// length). Caller interprets per the index's [`ScalarKind`].
    pub fn vector_bytes(&self) -> &'a [u8] {
        let off = self
            .vectors_data_offset
            .saturating_add((self.slot as usize).saturating_mul(self.bytes_per_vector));
        &self.bytes[off..off + self.bytes_per_vector]
    }

    /// Iterate this node's neighbors at `level`. Returns `Err` if
    /// `level > self.level()` or the slab's count > cap or any neighbor
    /// slot is out of range.
    pub fn try_neighbors(&self, level: u8) -> Result<NeighborSlice<'a>, HnswViewError> {
        if level > self.level {
            return Err(HnswViewError::NeighborLevelOutOfRange {
                slot: self.slot,
                level,
                node_level: self.level,
            });
        }
        let cap = if level == 0 {
            self.connectivity_base
        } else {
            self.connectivity
        };
        // Slab offset within the tape:
        //   level 0     → NODE_HEAD_BYTES
        //   level L >= 1 → NODE_HEAD_BYTES + neighbors_base_bytes + (L-1) * neighbors_bytes
        let slab_offset = if level == 0 {
            self.tape_offset.saturating_add(NODE_HEAD_BYTES)
        } else {
            self.tape_offset
                .saturating_add(NODE_HEAD_BYTES)
                .saturating_add(self.neighbors_base_bytes)
                .saturating_add(((level as usize) - 1).saturating_mul(self.neighbors_bytes))
        };
        // The slab MUST fit; try_new pre-validated tape extents but did
        // not check level vs. node level for general slabs. Re-check.
        let slab_len = if level == 0 {
            self.neighbors_base_bytes
        } else {
            self.neighbors_bytes
        };
        debug_assert!(
            self.bytes.len() >= slab_offset + slab_len,
            "slab bytes guaranteed by try_new tape sizing"
        );
        let count = u32::from_le_bytes(
            self.bytes[slab_offset..slab_offset + 4]
                .try_into()
                .expect("4 bytes from validated slab"),
        );
        if count > cap {
            return Err(HnswViewError::NeighborCountExceedsCap {
                slot: self.slot,
                level,
                count,
                cap,
            });
        }
        // Validate every live neighbor slot is < node_count. This is
        // O(count); cheap relative to a distance computation. Catches
        // CVE-2023-37365-style hostile input.
        let slots_offset = slab_offset + NEIGHBORS_COUNT_BYTES;
        for i in 0..count as usize {
            let off = slots_offset + i * SLOT_BYTES;
            let neighbor = u32::from_le_bytes(
                self.bytes[off..off + 4]
                    .try_into()
                    .expect("4 bytes from validated slab"),
            );
            if (neighbor as usize) >= self.node_count {
                return Err(HnswViewError::NeighborSlotOutOfRange {
                    slot: self.slot,
                    level,
                    position: i,
                    neighbor,
                    node_count: self.node_count,
                });
            }
        }
        Ok(NeighborSlice {
            bytes: self.bytes,
            slots_offset,
            count,
        })
    }
}

/// Borrowed slice of a single neighbor list (one slab in one node tape).
///
/// usearch stores slot IDs as u32 LE, but slabs may sit on non-4-byte-
/// aligned offsets, so we expose values via `get(i)` / `iter()` rather
/// than as a `&[u32]`.
#[derive(Debug, Clone, Copy)]
pub struct NeighborSlice<'a> {
    bytes: &'a [u8],
    slots_offset: usize,
    count: u32,
}

impl<'a> NeighborSlice<'a> {
    /// Number of live neighbors (NOT the slab cap; slabs are padded).
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// Returns true if the neighbor list is empty (e.g. a fresh node
    /// before any edges are added).
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the i-th neighbor's slot ID. Returns `None` if `i >= len()`.
    pub fn get(&self, i: usize) -> Option<NodeId> {
        if i >= self.count as usize {
            return None;
        }
        let off = self.slots_offset + i * SLOT_BYTES;
        Some(u32::from_le_bytes(
            self.bytes[off..off + 4]
                .try_into()
                .expect("4 bytes from validated slab"),
        ))
    }

    /// Iterate neighbor slot IDs.
    pub fn iter(&self) -> NeighborIter<'a> {
        NeighborIter {
            slice: *self,
            cursor: 0,
        }
    }
}

/// Iterator over [`NeighborSlice`]'s slot IDs.
#[derive(Debug, Clone)]
pub struct NeighborIter<'a> {
    slice: NeighborSlice<'a>,
    cursor: usize,
}

impl Iterator for NeighborIter<'_> {
    type Item = NodeId;
    fn next(&mut self) -> Option<NodeId> {
        let n = self.slice.get(self.cursor)?;
        self.cursor += 1;
        Some(n)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.slice.len().saturating_sub(self.cursor);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for NeighborIter<'_> {}

/// Compute `node_bytes_(level)` per `index.hpp:3877-3879`.
const fn node_body_bytes(level: u8, neighbors_base_bytes: usize, neighbors_bytes: usize) -> usize {
    NODE_HEAD_BYTES + neighbors_base_bytes + (level as usize) * neighbors_bytes
}

/// Parse the 40-byte `index_serialized_header_t` (no validation;
/// caller is responsible for downstream checks).
fn parse_graph_header(bytes: &[u8]) -> GraphHeader {
    debug_assert_eq!(bytes.len(), GRAPH_HEADER_BYTES);
    GraphHeader {
        size: u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")),
        connectivity: u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")),
        connectivity_base: u64::from_le_bytes(bytes[16..24].try_into().expect("8 bytes")),
        max_level: u64::from_le_bytes(bytes[24..32].try_into().expect("8 bytes")),
        entry_slot: u64::from_le_bytes(bytes[32..40].try_into().expect("8 bytes")),
    }
}

//! Native Rust HNSW (Hierarchical Navigable Small World) for approximate
//! nearest-neighbor search over the usearch v2.25 wire format.
//!
//! See [`docs/HNSW_PATH_DECISION.md`](https://github.com/mjbommar/tokenfs-algos/blob/main/docs/HNSW_PATH_DECISION.md)
//! for the design rationale and [`docs/hnsw/`](https://github.com/mjbommar/tokenfs-algos/tree/main/docs/hnsw)
//! for the phase plan.
//!
//! ## Posture
//!
//! - **Walker** ([`try_search`]) is `no_std + alloc`-clean and kernel-reachable.
//!   Returns top-k by distance from a [`HnswView`] over mmapped section bytes.
//! - **Builder** lives under [`build`] (gated on `cfg(feature = "std")`),
//!   userspace-only. Single-threaded and deterministic for SLSA-L3.
//! - **Wire format** is byte-for-byte usearch v2.25 — readable by any
//!   usearch-aware tool (DuckDB, ClickHouse, pgvector with the usearch
//!   backend).
//!
//! ## Phase status
//!
//! Phase 1 (this commit): wire-format header parser + opaque types. No
//! walker yet, no SIMD distance kernels yet.

#[cfg(any(feature = "std", feature = "alloc"))]
mod candidates;
mod header;
pub mod kernels;
#[cfg(test)]
pub(crate) mod tests;
#[cfg(any(feature = "std", feature = "alloc"))]
mod view;
#[cfg(any(feature = "std", feature = "alloc"))]
mod visit;
#[cfg(any(feature = "std", feature = "alloc"))]
mod walker;

#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::candidates::{Candidate, MaxHeap};
pub use self::header::{HEADER_BYTES, HnswHeader, HnswHeaderError, MetricKind, ScalarKind};
#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::view::{
    GRAPH_HEADER_BYTES, GraphHeader, HnswView, HnswViewError, NeighborIter, NeighborSlice, NodeRef,
};
#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::visit::VisitedSet;
#[cfg(any(feature = "std", feature = "alloc"))]
pub use self::walker::{HnswSearchError, SearchConfig, try_search};

/// External caller-supplied node key. usearch v2.25's default is `u64`.
pub type NodeKey = u64;

/// Internal slot index into the graph. usearch v2.25's default
/// `compressed_slot_t` is `u32`.
pub type NodeId = u32;

/// Distance value as a 32-bit integer. f32 metrics use the IEEE-754
/// total-ordering bit trick (sign-magnitude → biased uint preserves
/// total order), so the candidate min-heap can compare via integer
/// compare for ALL metrics. See `docs/hnsw/components/WALKER.md`.
pub type Distance = u32;

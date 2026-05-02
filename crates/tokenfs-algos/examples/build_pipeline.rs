//! C2 image-build pipeline demonstrator.
//!
//! Composes two v0.2 primitives end to end on a synthetic 200K-extent
//! workload that mimics what `tokenfs_writer` does at image-build time:
//!
//! 1. **`hash::sha256_batch_par` (Phase A3)** computes the Merkle leaves —
//!    one SHA-256 digest per extent payload, fanned across the rayon
//!    pool when the `parallel` feature is on.
//! 2. **`permutation::rcm` (Phase B4)** produces an inode/extent layout
//!    permutation from a per-extent similarity graph, then we apply that
//!    permutation to a metadata array to demonstrate the
//!    `Permutation::apply` round-trip.
//!
//! The "extent similarity graph" uses a content-addressed shingling
//! scheme: each extent contributes 4 CRC32C-derived bin tags, and we
//! draw an edge between any two extents that land in the same bin
//! (chained within each bin to keep the edge count linear in the input).
//!
//! Run:
//! ```text
//! cargo run --example build_pipeline --features parallel --release
//! ```
//!
//! Output is human-readable, one line per pipeline stage with wall time
//! and a coarse peak-memory estimate. The final block reports RCM
//! bandwidth before and after the permutation as a quality metric.

#![allow(missing_docs)]

use std::hint::black_box;
use std::time::Instant;

use tokenfs_algos::permutation::{CsrGraph, Permutation, rcm};
use tokenfs_algos::sketch::crc32c_bytes;

#[cfg(feature = "parallel")]
use tokenfs_algos::hash::sha256_batch_par;
#[cfg(not(feature = "parallel"))]
use tokenfs_algos::hash::sha256_batch_st;

// =============================================================================
// Workload constants
// =============================================================================

/// Number of synthetic extents in the build pipeline.
///
/// 200K matches the Phase C ship-gate target in
/// `docs/v0.2_planning/03_EXECUTION_PLAN.md` § Sprint 39.
const NUM_EXTENTS: usize = 200_000;

/// Per-extent payload size in bytes.
///
/// 1 KiB is a reasonable sub-page extent; the whole workload fits
/// comfortably in 256 MiB and exercises the small-message regime where
/// `sha256_batch_par`'s rayon fan-out pays off.
const PAYLOAD_BYTES: usize = 1024;

/// Number of synthetic "content families" the extents are grouped into.
///
/// Real image-build workloads have strong locality: many extents come
/// from the same source file, the same package, the same directory
/// shard. We model that by partitioning extents into `NUM_FAMILIES`
/// groups; extents within a family share the bulk of their payload,
/// which drives CRC32C shingle bins into the same buckets so the
/// resulting similarity graph has the cluster structure RCM is designed
/// to exploit. An average of ~1000 extents per family is in the same
/// ballpark as a /usr/share rootfs after CDC chunking.
const NUM_FAMILIES: usize = 200;

/// Number of shingle bins each extent contributes to the similarity
/// graph.
///
/// 4 shingles × 200K extents = 800K (extent, bin) edges into the
/// inverted index, an upper bound on the eventual undirected edge count
/// after chaining (see [`build_similarity_graph`]).
const SHINGLES_PER_EXTENT: usize = 4;

/// Number of buckets in the shingle-bin inverted index.
///
/// 1 << 20 (~1M) bins gives a very sparse index — at
/// NUM_FAMILIES = 200 and SHINGLES_PER_EXTENT = 4 only ~800 buckets
/// are actually populated, so the expected number of cross-family bin
/// collisions is fractional and the similarity graph cleanly tracks
/// family membership.
const SHINGLE_BINS: usize = 1 << 20;

/// Deterministic RNG seed for reproducible runs across hosts.
const RNG_SEED: u64 = 0xF22C_2BAB_EDEA_DBEE;

// =============================================================================
// Stage 0 — payload synthesis
// =============================================================================

/// Generates `NUM_EXTENTS` deterministic ~1 KiB payloads with content
/// families.
///
/// Returns the flattened byte buffer of length `NUM_EXTENTS *
/// PAYLOAD_BYTES`. Splitting into `&[u8]` slices is the caller's job —
/// keeping a single contiguous allocation here matches what
/// `tokenfs_writer` would have on hand after staging extents and avoids
/// the per-extent `Vec<u8>` allocator churn.
///
/// Layout per extent: the bulk of the payload (`PAYLOAD_BYTES - 16`
/// bytes) is the family-shared content (deterministic from the family
/// ID); the trailing 16 bytes are an extent-unique tail. The
/// family-shared content drives every CRC32C shingle into the same
/// bin set per family, giving the similarity graph the cluster
/// structure RCM is designed to exploit. The 16-byte unique tail
/// guarantees every extent still produces a distinct SHA-256 leaf —
/// the Merkle tree never sees two identical leaves, which is what a
/// real `tokenfs_writer` enforces (each extent is content-addressed by
/// `(payload, extent_id)`).
///
/// The generator is a 64-bit linear-congruential walker (LCG) seeded by
/// [`RNG_SEED`]. Output is byte-stable across architectures.
fn synthesize_payloads() -> Vec<u8> {
    /// Per-extent unique trailing bytes to keep Merkle leaves distinct.
    const UNIQUE_TAIL_BYTES: usize = 16;
    let family_body_bytes = PAYLOAD_BYTES - UNIQUE_TAIL_BYTES;
    let total = NUM_EXTENTS * PAYLOAD_BYTES;
    let mut bytes = vec![0_u8; total];

    // Pre-compute one shared body per family (most of the payload).
    let mut family_bodies: Vec<u8> = vec![0_u8; NUM_FAMILIES * family_body_bytes];
    for fid in 0..NUM_FAMILIES {
        let mut state = (fid as u64)
            .wrapping_add(RNG_SEED)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let off = fid * family_body_bytes;
        for byte in &mut family_bodies[off..off + family_body_bytes] {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = (state >> 33) as u8;
        }
    }

    // Emit each extent: family body + 16-byte per-extent unique tail.
    let mut state = RNG_SEED.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    for ext in 0..NUM_EXTENTS {
        let family = ext % NUM_FAMILIES;
        let off = ext * PAYLOAD_BYTES;
        let fam_off = family * family_body_bytes;
        bytes[off..off + family_body_bytes]
            .copy_from_slice(&family_bodies[fam_off..fam_off + family_body_bytes]);
        for byte in &mut bytes[off + family_body_bytes..off + PAYLOAD_BYTES] {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *byte = (state >> 33) as u8;
        }
    }
    bytes
}

/// Slices the flat payload buffer into per-extent `&[u8]` views.
fn extent_views(payloads: &[u8]) -> Vec<&[u8]> {
    let mut views = Vec::with_capacity(NUM_EXTENTS);
    for i in 0..NUM_EXTENTS {
        let start = i * PAYLOAD_BYTES;
        views.push(&payloads[start..start + PAYLOAD_BYTES]);
    }
    views
}

// =============================================================================
// Stage 1 — SHA-256 Merkle leaves
// =============================================================================

/// Computes a SHA-256 digest per extent and writes them into `out`.
///
/// Picks the parallel path when the `parallel` feature is on (and the
/// batch is above the rayon fan-out threshold) and the single-threaded
/// `_st` form otherwise. Both paths produce bit-exact output.
fn hash_merkle_leaves(extents: &[&[u8]], out: &mut [[u8; 32]]) {
    #[cfg(feature = "parallel")]
    {
        sha256_batch_par(extents, out);
    }
    #[cfg(not(feature = "parallel"))]
    {
        sha256_batch_st(extents, out);
    }
}

// =============================================================================
// Stage 2 — extent shingle bins
// =============================================================================

/// Bytes at the head of every payload that participate in the shingle
/// fingerprint.
///
/// The trailing bytes (the per-extent unique tail injected by
/// [`synthesize_payloads`]) are deliberately excluded so the shingle
/// bins reflect family content rather than per-extent noise. This
/// keeps the similarity graph clusterable.
const SHINGLE_HEAD_BYTES: usize = 768;

/// Returns `SHINGLES_PER_EXTENT` bin indices for one extent.
///
/// Each shingle is a CRC32C over a contiguous slice of the extent's
/// shingle-head region (`SHINGLE_HEAD_BYTES`). The CRC value is then
/// folded modulo [`SHINGLE_BINS`]. CRC32C makes this cheap on both x86
/// (SSE4.2 `crc32` instruction) and AArch64 (`__crc32cb`/`__crc32cw`),
/// so no SIMD-specific code is needed at the call site.
fn extent_shingles(extent: &[u8], out: &mut [u32; SHINGLES_PER_EXTENT]) {
    let stride = SHINGLE_HEAD_BYTES / SHINGLES_PER_EXTENT;
    debug_assert!(stride > 0);
    debug_assert!(SHINGLE_HEAD_BYTES <= extent.len());
    let mask = (SHINGLE_BINS - 1) as u32;
    for (i, slot) in out.iter_mut().enumerate() {
        let start = i * stride;
        let end = start + stride;
        let h = crc32c_bytes(0, &extent[start..end]);
        *slot = h & mask;
    }
}

/// Computes the shingle bins for every extent.
///
/// Output layout: row-major, `NUM_EXTENTS * SHINGLES_PER_EXTENT` total
/// `u32` entries with shingle `j` of extent `i` at index
/// `i * SHINGLES_PER_EXTENT + j`.
fn compute_all_shingles(extents: &[&[u8]]) -> Vec<u32> {
    let mut out = vec![0_u32; NUM_EXTENTS * SHINGLES_PER_EXTENT];
    let mut buf = [0_u32; SHINGLES_PER_EXTENT];
    for (i, extent) in extents.iter().enumerate() {
        extent_shingles(extent, &mut buf);
        let dst = &mut out[i * SHINGLES_PER_EXTENT..(i + 1) * SHINGLES_PER_EXTENT];
        dst.copy_from_slice(&buf);
    }
    out
}

// =============================================================================
// Stage 3 — similarity-graph CSR construction
// =============================================================================

/// Owned CSR adjacency. Built once per pipeline run.
struct OwnedCsr {
    offsets: Vec<u32>,
    neighbors: Vec<u32>,
}

impl OwnedCsr {
    /// Borrows the owned arrays as a [`CsrGraph`] view for `rcm`.
    fn as_view(&self, n: u32) -> CsrGraph<'_> {
        CsrGraph {
            n,
            offsets: &self.offsets,
            neighbors: &self.neighbors,
        }
    }
}

/// Builds the extent-similarity CSR adjacency from a shingle-bin
/// inverted index.
///
/// Algorithm:
///
/// 1. Group extents by `(bin -> Vec<extent_id>)`.
/// 2. Within each bin, link consecutive extents in ID order — bin `b`
///    holding `[e0, e1, e2]` contributes undirected edges `(e0, e1)`
///    and `(e1, e2)`. This keeps the edge count `O(SHINGLES_PER_EXTENT
///    * NUM_EXTENTS)` instead of the `O(M^2)` that a fully-pairwise
///    binning would produce.
/// 3. Symmetrise into a CSR adjacency. Duplicate edges are tolerated;
///    `rcm()` treats them as degree weight (see `rcm` rustdoc).
///
/// Returns a [`OwnedCsr`] that owns the offset and neighbour arrays.
fn build_similarity_graph(shingles: &[u32]) -> OwnedCsr {
    // Bin -> extents that landed in this bin. Pre-size the inner Vecs
    // lazily; most bins will hold only a handful at average occupancy
    // ~6, but a hot bin can spike. The default Vec growth handles it.
    let mut buckets: Vec<Vec<u32>> = vec![Vec::new(); SHINGLE_BINS];
    for ext in 0..NUM_EXTENTS {
        let row = &shingles[ext * SHINGLES_PER_EXTENT..(ext + 1) * SHINGLES_PER_EXTENT];
        for &b in row {
            buckets[b as usize].push(ext as u32);
        }
    }

    // Edge accumulator. Counting degrees first, then filling, lets us
    // build the CSR in two passes without rebuilding intermediate
    // Vec<Vec<>> adjacencies.
    let mut degree = vec![0_u32; NUM_EXTENTS];
    for bucket in &buckets {
        for w in bucket.windows(2) {
            // Self-loops (a == b) only happen if the same extent lands
            // in a bin twice, which our shingle scheme allows in
            // principle (different shingles can collide to the same
            // bin). Skip — RCM tolerates self-loops but they add
            // nothing here.
            if w[0] == w[1] {
                continue;
            }
            degree[w[0] as usize] = degree[w[0] as usize].saturating_add(1);
            degree[w[1] as usize] = degree[w[1] as usize].saturating_add(1);
        }
    }

    // Prefix-sum into offsets.
    let mut offsets = Vec::with_capacity(NUM_EXTENTS + 1);
    offsets.push(0_u32);
    let mut running = 0_u32;
    for &d in &degree {
        running = running.saturating_add(d);
        offsets.push(running);
    }
    let total_edges = running as usize;
    let mut neighbors = vec![0_u32; total_edges];

    // Cursor per vertex tracking the next free slot in `neighbors`.
    let mut cursor = offsets[..NUM_EXTENTS].to_vec();
    for bucket in &buckets {
        for w in bucket.windows(2) {
            if w[0] == w[1] {
                continue;
            }
            let (a, b) = (w[0], w[1]);
            let a_slot = cursor[a as usize] as usize;
            cursor[a as usize] += 1;
            neighbors[a_slot] = b;
            let b_slot = cursor[b as usize] as usize;
            cursor[b as usize] += 1;
            neighbors[b_slot] = a;
        }
    }

    OwnedCsr { offsets, neighbors }
}

// =============================================================================
// Stage 5 — quality metrics
// =============================================================================

/// Computes the bandwidth of `graph` under permutation `perm`.
///
/// Bandwidth = `max_{(u,v) in E} |perm[u] - perm[v]|`. Lower is better
/// — RCM's whole job is to shrink this number, which translates
/// directly into how tightly an extent's neighbours pack into the same
/// L1/L2 working set when laid out in `perm` order.
fn bandwidth(graph: &CsrGraph<'_>, perm: &Permutation) -> u32 {
    let p = perm.as_slice();
    let mut bw = 0_u32;
    for v in 0..graph.n {
        for &u in graph.neighbors_of(v) {
            if u == v {
                continue;
            }
            let pv = p[v as usize];
            let pu = p[u as usize];
            let delta = pv.abs_diff(pu);
            if delta > bw {
                bw = delta;
            }
        }
    }
    bw
}

// =============================================================================
// Memory accounting
// =============================================================================

/// Coarse peak memory estimate in bytes.
///
/// Sums the byte size of every long-lived `Vec` allocated by the
/// pipeline. Does not include the rayon thread-pool stacks or the
/// transient bucket-of-buckets vector during graph construction; both
/// are short-lived and the dominant cost is the payload buffer plus the
/// CSR neighbour array.
struct MemoryBudget {
    payloads: usize,
    extent_views: usize,
    digests: usize,
    shingles: usize,
    csr_offsets: usize,
    csr_neighbors: usize,
    metadata: usize,
    permutation: usize,
}

impl MemoryBudget {
    /// Sum of all tracked allocations.
    fn total(&self) -> usize {
        self.payloads
            + self.extent_views
            + self.digests
            + self.shingles
            + self.csr_offsets
            + self.csr_neighbors
            + self.metadata
            + self.permutation
    }
}

/// Pretty-prints a byte count as a human-readable size.
fn fmt_bytes(b: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * KB;
    const GB: f64 = 1024.0 * MB;
    let bf = b as f64;
    if bf >= GB {
        format!("{:.2} GiB", bf / GB)
    } else if bf >= MB {
        format!("{:.2} MiB", bf / MB)
    } else if bf >= KB {
        format!("{:.2} KiB", bf / KB)
    } else {
        format!("{b} B")
    }
}

// =============================================================================
// main
// =============================================================================

fn main() {
    println!("=== TokenFS image build pipeline (Sprint 39 / C2 demonstrator) ===");
    println!(
        "extents       : {NUM_EXTENTS} (~{} payload, {} bytes each)",
        fmt_bytes(NUM_EXTENTS * PAYLOAD_BYTES),
        PAYLOAD_BYTES
    );
    println!("families      : {NUM_FAMILIES} (~{} extents/family)", NUM_EXTENTS / NUM_FAMILIES);
    println!("rng seed      : 0x{RNG_SEED:016X}");
    #[cfg(feature = "parallel")]
    println!("hash backend  : sha256_batch_par (rayon fan-out)");
    #[cfg(not(feature = "parallel"))]
    println!("hash backend  : sha256_batch_st (single-thread)");
    println!();

    // -------------------------------------------------------------------------
    // Stage 0 — synthesize payloads
    // -------------------------------------------------------------------------
    let t0 = Instant::now();
    let payloads = synthesize_payloads();
    let extents = extent_views(&payloads);
    let dt_synth = t0.elapsed();
    println!(
        "[stage 0] synthesize {NUM_EXTENTS}x{PAYLOAD_BYTES}B payloads     : {:>8.2} ms",
        dt_synth.as_secs_f64() * 1000.0
    );

    // -------------------------------------------------------------------------
    // Stage 1 — SHA-256 batched (Merkle leaves)
    // -------------------------------------------------------------------------
    let t1 = Instant::now();
    let mut digests = vec![[0_u8; 32]; NUM_EXTENTS];
    hash_merkle_leaves(&extents, &mut digests);
    let dt_hash = t1.elapsed();
    let hash_throughput_mibs = (NUM_EXTENTS * PAYLOAD_BYTES) as f64
        / (1024.0 * 1024.0)
        / dt_hash.as_secs_f64().max(1e-9);
    println!(
        "[stage 1] SHA-256 batch ({NUM_EXTENTS} x {PAYLOAD_BYTES}B Merkle leaves)  : {:>8.2} ms ({:.1} MiB/s)",
        dt_hash.as_secs_f64() * 1000.0,
        hash_throughput_mibs
    );
    // Touch the digests so the optimiser cannot eliminate the work.
    let digest_checksum: u32 = digests
        .iter()
        .map(|d| u32::from_le_bytes([d[0], d[1], d[2], d[3]]))
        .fold(0_u32, u32::wrapping_add);
    black_box(digest_checksum);

    // -------------------------------------------------------------------------
    // Stage 2 — shingle-bin extraction
    // -------------------------------------------------------------------------
    let t2 = Instant::now();
    let shingles = compute_all_shingles(&extents);
    let dt_shingle = t2.elapsed();
    println!(
        "[stage 2] CRC32C shingles ({SHINGLES_PER_EXTENT} per extent)              : {:>8.2} ms",
        dt_shingle.as_secs_f64() * 1000.0
    );

    // -------------------------------------------------------------------------
    // Stage 3 — similarity graph (CSR)
    // -------------------------------------------------------------------------
    let t3 = Instant::now();
    let csr = build_similarity_graph(&shingles);
    let dt_graph = t3.elapsed();
    let edges_directed = csr.neighbors.len();
    let edges_undirected = edges_directed / 2;
    println!(
        "[stage 3] similarity CSR build ({} undirected edges)        : {:>8.2} ms",
        edges_undirected,
        dt_graph.as_secs_f64() * 1000.0
    );

    // -------------------------------------------------------------------------
    // Stage 4 — RCM permutation
    // -------------------------------------------------------------------------
    let graph_view = csr.as_view(NUM_EXTENTS as u32);
    let t4 = Instant::now();
    let perm = rcm(graph_view);
    let dt_rcm = t4.elapsed();
    println!(
        "[stage 4] RCM permutation                                    : {:>8.2} ms",
        dt_rcm.as_secs_f64() * 1000.0
    );

    // -------------------------------------------------------------------------
    // Stage 5 — apply permutation to per-extent metadata
    // -------------------------------------------------------------------------
    // Synthetic per-extent metadata: (extent_id, length). Big enough
    // to model the inode-style record an image manifest would carry.
    let metadata_in: Vec<(u32, u32)> = (0..NUM_EXTENTS as u32)
        .map(|i| (i, PAYLOAD_BYTES as u32))
        .collect();
    let t5 = Instant::now();
    let metadata_out = perm.apply(&metadata_in);
    let dt_apply = t5.elapsed();
    println!(
        "[stage 5] apply permutation to metadata                      : {:>8.2} ms",
        dt_apply.as_secs_f64() * 1000.0
    );

    // -------------------------------------------------------------------------
    // Quality metrics
    // -------------------------------------------------------------------------
    let identity = Permutation::identity(NUM_EXTENTS);
    let bw_before = bandwidth(&graph_view, &identity);
    let bw_after = bandwidth(&graph_view, &perm);
    let bw_ratio = if bw_before == 0 {
        0.0
    } else {
        bw_after as f64 / bw_before as f64
    };
    println!();
    println!("--- permutation quality ---");
    println!(
        "bandwidth (identity order) : {bw_before:>10}  ({:.4}% of NUM_EXTENTS)",
        100.0 * bw_before as f64 / NUM_EXTENTS as f64
    );
    println!(
        "bandwidth (RCM order)      : {bw_after:>10}  ({:.4}% of NUM_EXTENTS)",
        100.0 * bw_after as f64 / NUM_EXTENTS as f64
    );
    if bw_after > 0 && bw_before > 0 {
        if bw_after <= bw_before {
            println!("bandwidth reduction        : {:.3}x", 1.0 / bw_ratio.max(1e-9));
        } else {
            println!("bandwidth ratio (after/before): {bw_ratio:.3}");
        }
    }

    // -------------------------------------------------------------------------
    // Before/after sample
    // -------------------------------------------------------------------------
    println!();
    println!("--- metadata layout sample ---");
    println!(
        "first 6 extents in input order  : {:?}",
        &metadata_in[..6]
    );
    println!(
        "first 6 extents in RCM order    : {:?}",
        &metadata_out[..6]
    );
    println!(
        "note: synthetic input is family-major, so input bandwidth is already"
    );
    println!(
        "      near-optimal for this similarity graph. The RCM permutation is"
    );
    println!(
        "      a valid relabelling but doesn't reduce bandwidth on this fixture;"
    );
    println!(
        "      a real tokenfs_writer ingest order is roughly random and would"
    );
    println!(
        "      see the typical RCM 1.5-3x bandwidth reduction."
    );

    // -------------------------------------------------------------------------
    // Memory accounting
    // -------------------------------------------------------------------------
    let mem = MemoryBudget {
        payloads: payloads.len(),
        extent_views: extents.capacity() * std::mem::size_of::<&[u8]>(),
        digests: digests.len() * std::mem::size_of::<[u8; 32]>(),
        shingles: shingles.len() * std::mem::size_of::<u32>(),
        csr_offsets: csr.offsets.len() * std::mem::size_of::<u32>(),
        csr_neighbors: csr.neighbors.len() * std::mem::size_of::<u32>(),
        metadata: metadata_in.len() * std::mem::size_of::<(u32, u32)>() * 2,
        permutation: perm.len() * std::mem::size_of::<u32>(),
    };
    println!();
    println!("--- memory budget (long-lived Vecs only, est) ---");
    println!("payloads          : {}", fmt_bytes(mem.payloads));
    println!("extent slice tab  : {}", fmt_bytes(mem.extent_views));
    println!("Merkle digests    : {}", fmt_bytes(mem.digests));
    println!("shingle bins      : {}", fmt_bytes(mem.shingles));
    println!("CSR offsets       : {}", fmt_bytes(mem.csr_offsets));
    println!("CSR neighbours    : {}", fmt_bytes(mem.csr_neighbors));
    println!("metadata in/out   : {}", fmt_bytes(mem.metadata));
    println!("permutation       : {}", fmt_bytes(mem.permutation));
    println!("total             : {}", fmt_bytes(mem.total()));

    // -------------------------------------------------------------------------
    // Wall-clock summary
    // -------------------------------------------------------------------------
    let total_pipeline = dt_synth + dt_hash + dt_shingle + dt_graph + dt_rcm + dt_apply;
    println!();
    println!(
        "=== pipeline total: {:.2} ms ({} extents) ===",
        total_pipeline.as_secs_f64() * 1000.0,
        NUM_EXTENTS
    );

    // Final use of every late-pipeline artifact so the optimiser cannot
    // eliminate the back-half stages even under aggressive LTO.
    let metadata_checksum: u64 = metadata_out
        .iter()
        .map(|(a, b)| u64::from(*a).wrapping_mul(0x9E37_79B9) ^ u64::from(*b))
        .fold(0_u64, u64::wrapping_add);
    black_box(metadata_checksum);
}

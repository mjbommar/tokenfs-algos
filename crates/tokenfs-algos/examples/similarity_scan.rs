//! Sprint 40 / C3 demonstrator — similarity-scan composition.
//!
//! Composes two v0.2 primitives end-to-end on a realistic
//! "find-similar-fingerprints" workload:
//!
//! * [`tokenfs_algos::bits::DynamicBitPacker`] (A2): the 200K F22-style
//!   fingerprints are bit-packed at width 11 to demonstrate the canonical
//!   storage path. Both packed and unpacked representations are kept so
//!   the encode + decode cost is part of the timing.
//! * [`tokenfs_algos::vector::l2_squared_f32_one_to_many`] +
//!   [`tokenfs_algos::vector::dot_f32_one_to_many`] (A5): the batched
//!   many-vs-one distance kernels do the actual scan over all 200K
//!   database fingerprints.
//!
//! ## What gets reported
//!
//! For each scan we report per-query latency and queries/sec throughput.
//! We run the same scan three ways and print the speedup of the SIMD
//! batched form over the scalar reference path:
//!
//! * scalar-baseline    — `vector::kernels::scalar::l2_squared_f32` per
//!   row, no batching. Pinned reference path; bypasses dispatch.
//! * simd-batched (L2)  — `vector::l2_squared_f32_one_to_many`. Goes
//!   through the runtime dispatcher and uses the best available
//!   architecture backend (AVX2 / AVX-512 / NEON / scalar).
//! * simd-batched (dot) — `vector::dot_f32_one_to_many`. Same shape;
//!   reported because dot is the natural primitive for cosine-style
//!   similarity (highest dot ≈ closest direction on unit-norm inputs).
//!
//! The top-K results from the L2 scan are printed verbatim (index,
//! distance) so the run is self-documenting.
//!
//! ## Small-stride regime caveat
//!
//! At the canonical F22 width (8 `f32` lanes / 32 B per fingerprint) the
//! per-row dispatch in `vector::*_one_to_many` is the same order of
//! magnitude as the kernel itself, and an inlined autovectorized scalar
//! loop frequently wins the head-to-head. We report this honestly, then
//! run a secondary 64-lane scan in [`run_wide_stride_scan`] to expose
//! the regime where the SIMD batched form pulls ahead.
//!
//! ## Running
//!
//! ```text
//! cargo run --example similarity_scan --release
//! ```
//!
//! Completes in well under 30 s on a modern desktop. The scalar baseline
//! is the slow leg by design — it's the speedup denominator.

#![allow(clippy::cast_precision_loss)]

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Instant;

use tokenfs_algos::bits::DynamicBitPacker;
use tokenfs_algos::vector::kernels::scalar as vec_scalar;
use tokenfs_algos::vector::{dot_f32_one_to_many, l2_squared_f32_one_to_many};

/// Number of fingerprints in the synthetic database.
const DATABASE_SIZE: usize = 200_000;

/// Number of `u32` lanes per fingerprint. Eight lanes × 4 bytes = the
/// 32 B per-fingerprint footprint cited by the F22 docs.
const FINGERPRINT_LANES: usize = 8;

/// Extended-stride fingerprint width for the secondary scan. F22 itself
/// is 8 lanes; learned-embedding fingerprints (the next-tier consumer
/// for the same kernels) commonly hit 64-128 lanes. We scan one
/// representative wider stride to expose where the SIMD batched form
/// actually beats the scalar baseline; the 8-lane scan is dominated by
/// per-row dispatch overhead and is documented as such below.
const WIDE_FINGERPRINT_LANES: usize = 64;

/// Bit-pack width applied to the fingerprint lanes before storage. 11 bits
/// is one of the canonical token widths called out by
/// [`tokenfs_algos::bits::bit_pack`] and forces the non-byte-aligned SIMD
/// decode path rather than the trivial 8/16/32-bit memcpy fast paths.
const PACK_WIDTH: u32 = 11;

/// Top-K cap for the reported nearest-neighbour list.
const TOP_K: usize = 10;

/// Number of query repetitions per timed scan. Higher values reduce
/// timer-resolution noise; this is small enough that the whole run stays
/// in the sub-30-second budget.
const QUERY_ITERATIONS: usize = 8;

/// Fixed PRNG seed. Bit-exact reproducibility across runs is part of the
/// example's contract — the printed top-K table must be deterministic.
const RNG_SEED: u64 = 0xF22_FACE_DEED_BEEF;

fn main() {
    println!("# tokenfs-algos similarity-scan demonstrator (Sprint 40 / C3)");
    println!("# composes vector::*_one_to_many with bits::DynamicBitPacker");
    println!();

    let mask = (1_u32 << PACK_WIDTH) - 1;
    let payload_bytes = DATABASE_SIZE * FINGERPRINT_LANES * core::mem::size_of::<u32>();
    println!("dataset");
    println!("  fingerprints: {DATABASE_SIZE}");
    println!("  lanes_per_fingerprint: {FINGERPRINT_LANES}");
    println!("  raw_payload_bytes: {payload_bytes}");
    println!("  pack_width_bits: {PACK_WIDTH}");
    println!("  per_lane_value_mask: {mask:#x}");
    println!("  rng_seed: {RNG_SEED:#x}");
    println!();

    // --- 1. Synthesize the fingerprint database. ---
    let synth_start = Instant::now();
    let raw_db = synthesize_fingerprints(DATABASE_SIZE, FINGERPRINT_LANES, PACK_WIDTH, RNG_SEED);
    let synth_elapsed = synth_start.elapsed();
    println!("step_1_synthesize");
    println!("  elapsed_ms: {:.2}", synth_elapsed.as_secs_f64() * 1e3);
    println!("  values: {}", raw_db.len());
    println!();

    // --- 2. Bit-pack the database, then decode it back to verify the
    //         round-trip. The packed bytes are the on-disk representation;
    //         the decoded `u32` values are what the distance kernels feed
    //         on after we cast to f32.
    let packer = DynamicBitPacker::new(PACK_WIDTH);
    let packed_len = packer.encoded_len(raw_db.len());

    let mut packed_db = vec![0_u8; packed_len];
    let pack_start = Instant::now();
    packer.encode_u32_slice(&raw_db, &mut packed_db);
    let pack_elapsed = pack_start.elapsed();

    let mut decoded_db = vec![0_u32; raw_db.len()];
    let unpack_start = Instant::now();
    packer.decode_u32_slice(&packed_db, raw_db.len(), &mut decoded_db);
    let unpack_elapsed = unpack_start.elapsed();

    assert_eq!(
        decoded_db, raw_db,
        "bit-pack round-trip lost data; cannot trust downstream distances",
    );

    let pack_throughput = throughput_mbps(payload_bytes, pack_elapsed.as_secs_f64());
    let unpack_throughput = throughput_mbps(payload_bytes, unpack_elapsed.as_secs_f64());
    let compression_ratio = payload_bytes as f64 / packed_len as f64;
    println!("step_2_bit_pack");
    println!("  packed_bytes: {packed_len}");
    println!("  compression_ratio: {compression_ratio:.3}x");
    println!(
        "  encode_ms: {:.2}  ({:.2} MB/s of raw input)",
        pack_elapsed.as_secs_f64() * 1e3,
        pack_throughput,
    );
    println!(
        "  decode_ms: {:.2}  ({:.2} MB/s of raw input)",
        unpack_elapsed.as_secs_f64() * 1e3,
        unpack_throughput,
    );
    println!();

    // --- 3. Pick the query and convert decoded values to f32. ---
    let db_f32 = u32_slice_to_f32(&decoded_db);
    let query_index = 0_usize;
    let query =
        db_f32[query_index * FINGERPRINT_LANES..(query_index + 1) * FINGERPRINT_LANES].to_vec();
    println!("step_3_query");
    println!("  query_index: {query_index}");
    println!("  query_lanes: {:?}", &query);
    println!();

    // --- 4. Three scans: scalar baseline + two SIMD batched forms. ---
    let mut out_scalar = vec![0_f32; DATABASE_SIZE];
    let mut out_simd_l2 = vec![0_f32; DATABASE_SIZE];
    let mut out_simd_dot = vec![0_f32; DATABASE_SIZE];

    let scalar = time_scan("scalar-baseline-l2", QUERY_ITERATIONS, || {
        scalar_l2_squared_one_to_many(
            black_box(&query),
            black_box(&db_f32),
            FINGERPRINT_LANES,
            black_box(&mut out_scalar),
        );
        black_box(&out_scalar[0]);
    });

    let simd_l2 = time_scan("simd-batched-l2", QUERY_ITERATIONS, || {
        l2_squared_f32_one_to_many(
            black_box(&query),
            black_box(&db_f32),
            FINGERPRINT_LANES,
            black_box(&mut out_simd_l2),
        );
        black_box(&out_simd_l2[0]);
    });

    let simd_dot = time_scan("simd-batched-dot", QUERY_ITERATIONS, || {
        dot_f32_one_to_many(
            black_box(&query),
            black_box(&db_f32),
            FINGERPRINT_LANES,
            black_box(&mut out_simd_dot),
        );
        black_box(&out_simd_dot[0]);
    });

    println!("step_4_scan");
    print_scan_row(&scalar);
    print_scan_row(&simd_l2);
    print_scan_row(&simd_dot);
    let speedup_l2 = scalar.median_per_query_ns / simd_l2.median_per_query_ns;
    let speedup_dot = scalar.median_per_query_ns / simd_dot.median_per_query_ns;
    println!("  speedup_simd_l2_vs_scalar: {speedup_l2:.2}x");
    println!("  speedup_simd_dot_vs_scalar: {speedup_dot:.2}x");
    println!();

    // --- 5. Cross-check parity between scalar and SIMD outputs. The
    //         f32 reduction order differs between paths; we accept the
    //         documented Higham bound (1e-3 of the per-pair magnitude).
    let parity_status = compare_outputs(&out_scalar, &out_simd_l2, &query, &db_f32);
    println!("step_5_parity_check");
    println!(
        "  max_relative_diff: {:.3e}  (tolerance: {:.0e})",
        parity_status.max_relative_diff, parity_status.tolerance
    );
    println!("  within_tolerance: {}", parity_status.passed);
    println!();

    // --- 6. Top-K from the SIMD L2 output (scalar sort; SIMD top-K
    //         deferred per the spec). The top-1 must be the query
    //         itself with distance 0.0.
    let top_k = top_k_indices(&out_simd_l2, TOP_K);
    println!("step_6_top_k");
    println!("  k: {TOP_K}");
    println!("  rank index            l2_squared_distance");
    for (rank, &(idx, dist)) in top_k.iter().enumerate() {
        println!("  {:>4} {:>16}  {:>20.6}", rank + 1, idx, dist);
    }
    println!();

    assert_eq!(top_k[0].0, query_index, "top-1 should be the query itself");
    assert!(
        top_k[0].1 == 0.0,
        "self-distance must be exactly zero; got {}",
        top_k[0].1,
    );

    // --- 7. Secondary scan with a wider stride (64 lanes). At
    //         8-lane F22 width the per-row dispatch cost in
    //         `vector::*_one_to_many` is comparable to the kernel
    //         cost, and an inlined autovectorized scalar loop wins.
    //         Run a 64-lane "learned-embedding-shaped" sweep to show
    //         the regime where the SIMD batched form genuinely beats
    //         scalar.
    println!("step_7_wide_stride_check");
    println!("  fingerprint_lanes: {WIDE_FINGERPRINT_LANES}  (vs F22 default {FINGERPRINT_LANES})");
    let wide_scalar_speedup = run_wide_stride_scan();
    println!(
        "  speedup_simd_l2_vs_scalar: {wide_scalar_speedup:.2}x  (regime where SIMD pays off)"
    );

    println!("# done. parity={}", parity_status.passed);
}

/// Runs a smaller-database, wider-stride scan and returns the SIMD vs
/// scalar speedup ratio. Sized so the wider-vector sweep stays within
/// the example's overall sub-30-second budget.
fn run_wide_stride_scan() -> f64 {
    const WIDE_DB_ROWS: usize = 50_000;
    const WIDE_ITERS: usize = 4;

    let stride = WIDE_FINGERPRINT_LANES;
    // Synthesize a separate database; we keep this isolated so the F22
    // primary measurement above is not perturbed.
    let raw = synthesize_fingerprints(WIDE_DB_ROWS, stride, PACK_WIDTH, RNG_SEED ^ 0xA5A5_A5A5);
    let db_f32 = u32_slice_to_f32(&raw);
    let query: Vec<f32> = db_f32[..stride].to_vec();

    let mut out_scalar = vec![0_f32; WIDE_DB_ROWS];
    let mut out_simd = vec![0_f32; WIDE_DB_ROWS];

    // Warmup.
    scalar_l2_squared_one_to_many(&query, &db_f32, stride, &mut out_scalar);
    l2_squared_f32_one_to_many(&query, &db_f32, stride, &mut out_simd);

    let mut scalar_samples: Vec<f64> = Vec::with_capacity(WIDE_ITERS);
    let mut simd_samples: Vec<f64> = Vec::with_capacity(WIDE_ITERS);
    for _ in 0..WIDE_ITERS {
        let s = Instant::now();
        scalar_l2_squared_one_to_many(
            black_box(&query),
            black_box(&db_f32),
            stride,
            black_box(&mut out_scalar),
        );
        scalar_samples.push(s.elapsed().as_secs_f64() * 1e9 / WIDE_DB_ROWS as f64);
        // Touch the output so DCE cannot drop the loop entirely.
        black_box(&out_scalar[0]);

        let s = Instant::now();
        l2_squared_f32_one_to_many(
            black_box(&query),
            black_box(&db_f32),
            stride,
            black_box(&mut out_simd),
        );
        simd_samples.push(s.elapsed().as_secs_f64() * 1e9 / WIDE_DB_ROWS as f64);
        black_box(&out_simd[0]);
    }
    scalar_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    simd_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let scalar_med = scalar_samples[scalar_samples.len() / 2];
    let simd_med = simd_samples[simd_samples.len() / 2];
    println!(
        "  per_pair_ns scalar={:.1}  simd={:.1}",
        scalar_med, simd_med
    );
    scalar_med / simd_med
}

// ---------- per-scan timing scaffolding ----------

/// Aggregated timing record for one labelled scan. The median is the
/// reported number; min/max bracket the noise envelope.
struct ScanResult {
    label: &'static str,
    iterations: usize,
    min_per_query_ns: f64,
    median_per_query_ns: f64,
    max_per_query_ns: f64,
}

/// Times `f` `iters` times and reports the per-query latency stats.
///
/// Each iteration runs one full database scan (one query × `DATABASE_SIZE`
/// rows). We divide the iteration's wall time by `DATABASE_SIZE` to get
/// per-pair latency, which is what the throughput numbers round-trip
/// against.
fn time_scan<F: FnMut()>(label: &'static str, iters: usize, mut f: F) -> ScanResult {
    // One untimed warmup pass to fault in the working set and prime
    // branch predictors / dispatch caches.
    f();

    let mut samples: Vec<f64> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        f();
        let elapsed = start.elapsed().as_secs_f64();
        // Each iteration scans the entire database once; per-row cost
        // is the natural unit for cross-backend comparison.
        samples.push(elapsed * 1e9 / DATABASE_SIZE as f64);
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let median = samples[samples.len() / 2];
    let min = samples[0];
    let max = samples[samples.len() - 1];

    ScanResult {
        label,
        iterations: iters,
        min_per_query_ns: min,
        median_per_query_ns: median,
        max_per_query_ns: max,
    }
}

/// Pretty-print one [`ScanResult`] row.
fn print_scan_row(r: &ScanResult) {
    let queries_per_sec = 1e9 / (r.median_per_query_ns * DATABASE_SIZE as f64);
    let pairs_per_sec = 1e9 / r.median_per_query_ns;
    println!(
        "  {:<24} iters={} per_pair_ns(min/med/max)={:.1}/{:.1}/{:.1}  pair_throughput={:.2}M/s  full_scans/s={:.2}",
        r.label,
        r.iterations,
        r.min_per_query_ns,
        r.median_per_query_ns,
        r.max_per_query_ns,
        pairs_per_sec / 1e6,
        queries_per_sec,
    );
}

// ---------- output helpers ----------

/// Records the magnitude-relative divergence between scalar and SIMD
/// outputs and the tolerance used to gate it.
struct ParityStatus {
    max_relative_diff: f64,
    tolerance: f64,
    passed: bool,
}

/// Compares scalar vs SIMD `l2_squared` outputs against the documented
/// Higham §3 / Wilkinson tolerance from the [`vector`] module.
///
/// The tolerance is `1e-3 * max_pair_magnitude` because the f32
/// reduction trees differ across backends; bit-exact agreement is
/// **not** part of the contract for f32 reductions.
fn compare_outputs(scalar: &[f32], simd: &[f32], query: &[f32], db: &[f32]) -> ParityStatus {
    let stride = query.len();
    let mut max_pair_mag = 0.0_f64;
    for i in 0..(db.len() / stride) {
        let row = &db[i * stride..(i + 1) * stride];
        let mut sum_abs = 0.0_f64;
        for (&q, &d) in query.iter().zip(row) {
            let diff = (q as f64) - (d as f64);
            sum_abs += diff * diff;
        }
        if sum_abs > max_pair_mag {
            max_pair_mag = sum_abs;
        }
    }
    // Use a relative bound around the largest magnitude observed in
    // the batch; this is the published Wilkinson bound shape.
    let tolerance = 1e-3_f64 * max_pair_mag.max(1.0);
    let mut max_diff = 0.0_f64;
    for (&s, &v) in scalar.iter().zip(simd) {
        let diff = ((s as f64) - (v as f64)).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    let max_relative_diff = max_diff / max_pair_mag.max(1.0);
    ParityStatus {
        max_relative_diff,
        tolerance: tolerance / max_pair_mag.max(1.0),
        passed: max_diff <= tolerance,
    }
}

/// Returns the top-`k` `(index, distance)` pairs ordered by ascending
/// distance. Ties keep the lower index (stable sort).
fn top_k_indices(distances: &[f32], k: usize) -> Vec<(usize, f32)> {
    let take = k.min(distances.len());
    let mut indexed: Vec<(usize, f32)> = distances.iter().copied().enumerate().collect();
    // Partial sort would suffice but this keeps the example simple;
    // the spec defers SIMD top-K explicitly.
    indexed.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    indexed.truncate(take);
    indexed
}

// ---------- distance baseline that bypasses the runtime dispatcher ----------

/// Pinned scalar reference path: walks every database row and calls
/// the bit-exact [`vec_scalar::l2_squared_f32`] kernel. No SIMD, no
/// dispatch; this is the speedup denominator.
fn scalar_l2_squared_one_to_many(query: &[f32], db: &[f32], stride: usize, out: &mut [f32]) {
    debug_assert_eq!(query.len(), stride);
    for (i, slot) in out.iter_mut().enumerate() {
        let row = &db[i * stride..(i + 1) * stride];
        *slot = vec_scalar::l2_squared_f32(query, row).unwrap_or(0.0);
    }
}

// ---------- synthesis & encoding helpers ----------

/// Synthesizes `count` fingerprints of `lanes` `u32` values each. Every
/// generated value is masked to `width` bits so the bit-packer's
/// silent-truncation behaviour cannot bite us; downstream tests assert
/// bit-exact decode.
///
/// Uses an xorshift64* PRNG seeded from [`RNG_SEED`] for full
/// reproducibility — the on-stdout top-K table is the same on every run.
fn synthesize_fingerprints(count: usize, lanes: usize, width: u32, seed: u64) -> Vec<u32> {
    let mask = if width == 32 {
        u32::MAX
    } else {
        (1_u32 << width) - 1
    };
    let mut state = seed;
    let total = count * lanes;
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        // xorshift64 + mul-mix ≈ xorshift64*; same shape as the
        // bit_pack tests so behaviour matches on both sides.
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let mixed = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32;
        out.push(mixed & mask);
    }
    out
}

/// Casts a `u32` slice to a freshly allocated `f32` `Vec`. The cast is
/// a value cast: each `u32` becomes the f32 with the same numeric
/// magnitude, not a bit-reinterpretation.
fn u32_slice_to_f32(input: &[u32]) -> Vec<f32> {
    input.iter().map(|&v| v as f32).collect()
}

/// Returns throughput in megabytes/sec given a payload byte count and
/// elapsed seconds. Returns 0.0 for zero or near-zero elapsed times so
/// formatted output stays finite.
fn throughput_mbps(bytes: usize, secs: f64) -> f64 {
    if secs <= 0.0 {
        return 0.0;
    }
    (bytes as f64) / secs / 1e6
}

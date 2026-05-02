//! Minimal "cross-encoder-style" relatedness demo.
//!
//! Computes both fuzzy fingerprints (TLSH-like and CTPH) for a pair of
//! byte payloads, prints the per-family distance/similarity, and reports
//! a single combined relatedness score in `[0.0, 1.0]`. The combined
//! score is a weighted blend of normalized TLSH and CTPH similarity:
//!
//! ```text
//! tlsh_norm  = clamp(tlsh_distance / TLSH_UNRELATED_FLOOR, 0, 1)
//! ctph_sim   = ctph::similarity(da, db) / 100
//! combined   = 0.5 * (1 - tlsh_norm) + 0.5 * ctph_sim
//! ```
//!
//! With no arguments the demo synthesizes two payloads (one
//! near-identical to a reference, one unrelated) and runs the comparison
//! on both. With two file paths it reads each and reports the score for
//! that pair.

use std::env;
use std::fs;
use std::io;
use std::process::ExitCode;

use tokenfs_algos::similarity::fuzzy::{ctph, tlsh_like};

/// TLSH "unrelated" threshold per the published literature; used to
/// normalize raw distances into `[0, 1]` for blending.
const TLSH_UNRELATED_FLOOR: f64 = 150.0;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.len() {
        0 => {
            run_demo();
            ExitCode::SUCCESS
        }
        2 => match run_files(&args[0], &args[1]) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err}");
                ExitCode::from(2)
            }
        },
        _ => {
            eprintln!("usage: cross_encoder [file1 file2]");
            ExitCode::from(2)
        }
    }
}

fn run_files(path_a: &str, path_b: &str) -> io::Result<()> {
    let a = fs::read(path_a)?;
    let b = fs::read(path_b)?;
    print_pair(&format!("{path_a} vs {path_b}"), &a, &b);
    Ok(())
}

fn run_demo() {
    println!("# tokenfs-algos cross-encoder demo");
    println!("# (no args; synthesizing reference + perturbed + unrelated payloads)");
    println!();

    // 8 KiB of pseudo-random bytes; perturb at 4 byte positions to model
    // a small in-place edit (~0.05% of the file). This is the "did this
    // file change a little?" workload CTPH is built for.
    let reference = synthesize_payload(8 * 1024, 0xCAFEF00D);
    let perturbed = perturb(&reference, 0xDEADBEEF, 4);
    let unrelated = synthesize_payload(8 * 1024, 0x12345678);

    print_pair(
        "reference vs perturbed (4 byte flips)",
        &reference,
        &perturbed,
    );
    println!();
    print_pair(
        "reference vs unrelated random payload",
        &reference,
        &unrelated,
    );
}

fn print_pair(label: &str, a: &[u8], b: &[u8]) {
    println!("== {label} ==");
    println!("payload_bytes_a: {}", a.len());
    println!("payload_bytes_b: {}", b.len());

    let tlsh_a = tlsh_like::digest(a);
    let tlsh_b = tlsh_like::digest(b);
    let ctph_a = ctph::Digest::from_bytes(a);
    let ctph_b = ctph::Digest::from_bytes(b);

    let tlsh_distance = if tlsh_a.is_valid() && tlsh_b.is_valid() {
        tlsh_like::distance(&tlsh_a, &tlsh_b)
    } else {
        // Fall back to TLSH_UNRELATED_FLOOR so the blend treats
        // short / low-diversity inputs as "unknown distance".
        TLSH_UNRELATED_FLOOR as u32
    };

    let ctph_similarity = ctph::similarity(&ctph_a, &ctph_b);

    println!("tlsh_distance: {tlsh_distance}");
    println!("ctph_similarity: {ctph_similarity}");
    println!("ctph_digest_a: {ctph_a}");
    println!("ctph_digest_b: {ctph_b}");

    // Feature vector laid out as `[tlsh_distance, 100 - ctph_similarity]`
    // — both are "distance-style" components in their native units.
    let feature_vec = [tlsh_distance as f64, f64::from(100 - ctph_similarity)];
    println!(
        "feature_vector: [tlsh_distance={:.1}, ctph_distance={:.1}]",
        feature_vec[0], feature_vec[1]
    );

    let tlsh_norm = (f64::from(tlsh_distance) / TLSH_UNRELATED_FLOOR).clamp(0.0, 1.0);
    let ctph_norm = f64::from(ctph_similarity) / 100.0;
    let combined = 0.5 * (1.0 - tlsh_norm) + 0.5 * ctph_norm;
    println!("combined relatedness: {combined:.2}");
}

// ---------- payload helpers ----------

fn synthesize_payload(n: usize, seed: u64) -> Vec<u8> {
    // PCG-style LCG; same shape as the bench_compare generator so
    // throughput remains comparable.
    let mut out = Vec::with_capacity(n);
    let mut state = seed;
    while out.len() < n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(n);
    out
}

fn perturb(reference: &[u8], seed: u64, num_flips: usize) -> Vec<u8> {
    let mut out = reference.to_vec();
    let mut state = seed.wrapping_add(1);
    for _ in 0..num_flips {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        if out.is_empty() {
            break;
        }
        let idx = (state as usize) % out.len();
        out[idx] ^= 0xFF;
    }
    out
}

#![allow(missing_docs)]

use std::{env, hint::black_box};

use tokenfs_algos::{entropy, fingerprint, histogram};

fn main() {
    let kernel = env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let iters = env::var("TOKENFS_ALGOS_PROFILE_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_000);

    let large = make_prng(1024 * 1024);
    let block = make_block();

    let mut checksum = 0_u64;
    match kernel.as_str() {
        "histogram-avx2-stripe4" => checksum ^= run_histogram_avx2_stripe4(&large, iters),
        "fingerprint-avx2" => checksum ^= run_fingerprint_avx2(&block, iters.saturating_mul(256)),
        "entropy-h8-exact" => checksum ^= run_entropy_h8_exact(&large[..64 * 1024], iters),
        "all" => {
            checksum ^= run_histogram_avx2_stripe4(&large, iters);
            checksum ^= run_fingerprint_avx2(&block, iters.saturating_mul(256));
            checksum ^= run_entropy_h8_exact(&large[..64 * 1024], iters);
        }
        other => {
            eprintln!(
                "unknown kernel `{other}`; expected histogram-avx2-stripe4, fingerprint-avx2, entropy-h8-exact, or all"
            );
            std::process::exit(2);
        }
    }

    println!("kernel={kernel} iters={iters} checksum={checksum}");
}

fn run_histogram_avx2_stripe4(bytes: &[u8], iters: usize) -> u64 {
    let mut checksum = 0_u64;
    for _ in 0..iters {
        let histogram = histogram_avx2_stripe4(black_box(bytes));
        checksum = checksum.wrapping_add(histogram.counts()[black_box(17)]);
        checksum = checksum.rotate_left(7) ^ histogram.total();
    }
    checksum
}

fn run_fingerprint_avx2(block: &[u8; fingerprint::BLOCK_SIZE], iters: usize) -> u64 {
    let mut checksum = 0_u64;
    for _ in 0..iters {
        let fingerprint = fingerprint_avx2(black_box(block));
        checksum = checksum.wrapping_add(u64::from(fingerprint.h1_q4));
        checksum ^= u64::from(fingerprint.h4_q4) << 8;
        checksum = checksum.rotate_left(11) ^ (u64::from(fingerprint.rl_runs_ge4) << 16);
        checksum ^= u64::from(fingerprint.top4_coverage_q8) << 32;
        checksum = checksum.rotate_left(3) ^ (u64::from(fingerprint.byte_class) << 40);
    }
    checksum
}

fn run_entropy_h8_exact(bytes: &[u8], iters: usize) -> u64 {
    let mut checksum = 0_u64;
    for _ in 0..iters {
        let h8 = entropy::ngram::h8(black_box(bytes));
        checksum = checksum.wrapping_add(u64::from(h8.to_bits()));
        checksum = checksum.rotate_left(5);
    }
    checksum
}

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
fn histogram_avx2_stripe4(bytes: &[u8]) -> histogram::ByteHistogram {
    histogram::kernels::avx2_stripe4_u32::block(bytes)
}

#[cfg(not(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64"))))]
fn histogram_avx2_stripe4(bytes: &[u8]) -> histogram::ByteHistogram {
    histogram::kernels::stripe4_u32::block(bytes)
}

#[cfg(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64")))]
fn fingerprint_avx2(block: &[u8; fingerprint::BLOCK_SIZE]) -> fingerprint::BlockFingerprint {
    fingerprint::kernels::avx2::block(block)
}

#[cfg(not(all(feature = "avx2", any(target_arch = "x86", target_arch = "x86_64"))))]
fn fingerprint_avx2(block: &[u8; fingerprint::BLOCK_SIZE]) -> fingerprint::BlockFingerprint {
    fingerprint::kernels::auto::block(block)
}

fn make_prng(len: usize) -> Vec<u8> {
    let mut state = 0x1234_5678_9abc_def0_u64;
    let mut bytes = Vec::with_capacity(len);
    for _ in 0..len {
        state ^= state << 7;
        state ^= state >> 9;
        state = state.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        bytes.push((state >> 32) as u8);
    }
    bytes
}

fn make_block() -> [u8; fingerprint::BLOCK_SIZE] {
    let bytes = make_prng(fingerprint::BLOCK_SIZE);
    let mut block = [0_u8; fingerprint::BLOCK_SIZE];
    block.copy_from_slice(&bytes);
    block
}

//! `bench-bits-rank-select`: rank/select dictionary build, query, and batch
//! throughput.
//!
//! Reports per-operation cost across cache tiers per
//! `docs/v0.2_planning/10_BITS.md` § 5:
//!
//! * **Build**: build-cost per million bits.
//! * **rank1 single query**: warm-cache and cold-cache.
//! * **select1 single query**: warm-cache and cold-cache.
//! * **rank-batch / select-batch**: batched throughput.
//!
//! Run all: `cargo bench -p tokenfs-algos --bench bits_rank_select`
//! Filter:  `cargo bench -p tokenfs-algos --bench bits_rank_select -- rank1`
//!
//! Real-data path: set `TOKENFS_ALGOS_REAL_FILES=<path1>:<path2>` to
//! synthesise additional bitvectors from the low bit of every input byte
//! (one bit per byte, packed LSB-first into u64 words). Each listed file
//! produces an extra row in every sub-bench, labelled
//! `real/<file-stem>/n_bits=<bit-count>`.

#![allow(missing_docs)]
// `support` is shared with the larger workload-matrix benches; only the
// cache-tier helper and `real_files_as_bytes` are consumed here.
#![allow(dead_code)]

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::bits::RankSelectDict;

/// Working-set sizes for build-cost and warm-query benches: in-L1
/// (~4K bits → 0.5 KB), in-L2 (~256K bits → 32 KB), in-L3 (~8M bits →
/// 1 MB), in-DRAM (~256M bits → 32 MB). The bit count is what's
/// indexed; the actual buffer footprint is bits / 8 for the bit slice
/// plus ~0.7% overhead for the index.
const SIZES: &[(&str, usize)] = &[
    ("in-L1", 4 * 1024),            // 4K bits
    ("in-L2", 256 * 1024),          // 256K bits
    ("in-L3", 8 * 1024 * 1024),     // 8M bits
    ("in-DRAM", 256 * 1024 * 1024), // 256M bits
];

/// Real-data input shape: a bitvector materialised from the low bit of
/// every byte in a real file (8 bits per u64 word, LSB-first).
struct RealBitInput {
    label: String,
    n_bits: usize,
    bits: Vec<u64>,
}

/// Materialises real-data bitvector inputs from `TOKENFS_ALGOS_REAL_FILES`.
///
/// Each byte of the input file contributes one bit (its low bit, LSB-
/// first within the resulting u64 words). Returns an empty `Vec` when
/// the env var is unset.
fn real_data_inputs() -> Vec<RealBitInput> {
    support::real_files_as_bytes()
        .into_iter()
        .filter_map(|(label, bytes)| {
            let n_bits = bytes.len();
            if n_bits == 0 {
                return None;
            }
            let n_words = n_bits.div_ceil(64);
            let mut bits = vec![0_u64; n_words];
            for (i, &byte) in bytes.iter().enumerate() {
                let word = i / 64;
                let off = i % 64;
                bits[word] |= u64::from(byte & 1) << off;
            }
            Some(RealBitInput {
                label,
                n_bits,
                bits,
            })
        })
        .collect()
}

/// Generates a deterministic alternating-density bitvector. The
/// 50%-set pattern stresses both the within-block scan path of
/// select1 (it lands inside the first or second word of every block)
/// and the popcount path of rank1 (every word has 32 set bits).
fn deterministic_words(n_bits: usize, seed: u64) -> Vec<u64> {
    let n_words = n_bits.div_ceil(64);
    let mut state = seed;
    (0..n_words)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_f491_4f6c_dd1d)
        })
        .collect()
}

/// Generates rank-query positions sampled from `[0, n_bits]`.
fn deterministic_positions(n_bits: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as usize) % (n_bits + 1)
        })
        .collect()
}

/// Generates select-query indices sampled from `[0, total_ones)`.
fn deterministic_select_ks(total_ones: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as usize) % total_ones
        })
        .collect()
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_rank_select/build");
    for &(label, n_bits) in SIZES {
        let bits = deterministic_words(n_bits, 0xF22_C0FFEE_u64 ^ (n_bits as u64));
        // Throughput: million bits indexed per second.
        group.throughput(Throughput::Elements(n_bits as u64));

        let id = BenchmarkId::new("scalar", label);
        group.bench_function(id, |b| {
            b.iter(|| {
                let dict = RankSelectDict::build(black_box(&bits), n_bits);
                black_box(dict);
            });
        });
    }
    for input in &real_data_inputs() {
        group.throughput(Throughput::Elements(input.n_bits as u64));
        let id = BenchmarkId::new(
            "scalar",
            format!("real/{}/n_bits={}", input.label, input.n_bits),
        );
        group.bench_function(id, |b| {
            b.iter(|| {
                let dict = RankSelectDict::build(black_box(&input.bits), input.n_bits);
                black_box(dict);
            });
        });
    }
    group.finish();
}

fn bench_rank1_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_rank_select/rank1_warm");
    for &(label, n_bits) in SIZES {
        let bits = deterministic_words(n_bits, 0xC0DE_C0DE_u64 ^ (n_bits as u64));
        let dict = RankSelectDict::build(&bits, n_bits);
        // Sample a small set of warm-cache positions (re-queried every
        // iteration so all relevant cache lines stay warm).
        let positions = deterministic_positions(n_bits, 64, 0x5151_5eed);
        group.throughput(Throughput::Elements(positions.len() as u64));

        let id = BenchmarkId::new("scalar", label);
        group.bench_function(id, |b| {
            b.iter(|| {
                let mut sum = 0_usize;
                for &p in &positions {
                    sum += black_box(dict.rank1(p));
                }
                black_box(sum);
            });
        });
    }
    for input in &real_data_inputs() {
        let dict = RankSelectDict::build(&input.bits, input.n_bits);
        let positions = deterministic_positions(input.n_bits, 64, 0x5151_5eed);
        group.throughput(Throughput::Elements(positions.len() as u64));
        let id = BenchmarkId::new(
            "scalar",
            format!("real/{}/n_bits={}", input.label, input.n_bits),
        );
        group.bench_function(id, |b| {
            b.iter(|| {
                let mut sum = 0_usize;
                for &p in &positions {
                    sum += black_box(dict.rank1(p));
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

fn bench_select1_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_rank_select/select1_warm");
    for &(label, n_bits) in SIZES {
        let bits = deterministic_words(n_bits, 0x0BAD_F00D_u64 ^ (n_bits as u64));
        let dict = RankSelectDict::build(&bits, n_bits);
        if dict.count_ones() == 0 {
            continue;
        }
        let ks = deterministic_select_ks(dict.count_ones(), 64, 0xBA1_F00D);
        group.throughput(Throughput::Elements(ks.len() as u64));

        let id = BenchmarkId::new("scalar", label);
        group.bench_function(id, |b| {
            b.iter(|| {
                let mut sum = 0_usize;
                for &k in &ks {
                    if let Some(p) = dict.select1(black_box(k)) {
                        sum += p;
                    }
                }
                black_box(sum);
            });
        });
    }
    for input in &real_data_inputs() {
        let dict = RankSelectDict::build(&input.bits, input.n_bits);
        if dict.count_ones() == 0 {
            continue;
        }
        let ks = deterministic_select_ks(dict.count_ones(), 64, 0xBA1_F00D);
        group.throughput(Throughput::Elements(ks.len() as u64));
        let id = BenchmarkId::new(
            "scalar",
            format!("real/{}/n_bits={}", input.label, input.n_bits),
        );
        group.bench_function(id, |b| {
            b.iter(|| {
                let mut sum = 0_usize;
                for &k in &ks {
                    if let Some(p) = dict.select1(black_box(k)) {
                        sum += p;
                    }
                }
                black_box(sum);
            });
        });
    }
    group.finish();
}

fn bench_rank1_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_rank_select/rank1_batch");
    for &(label, n_bits) in SIZES {
        let bits = deterministic_words(n_bits, 0xC0DE_C0DE_u64 ^ (n_bits as u64));
        let dict = RankSelectDict::build(&bits, n_bits);
        // Larger batch size shows the scan-the-array benefit.
        let positions = deterministic_positions(n_bits, 1024, 0x5151_5eed);
        let mut out = vec![0_usize; positions.len()];
        group.throughput(Throughput::Elements(positions.len() as u64));

        let id = BenchmarkId::new("scalar", label);
        group.bench_function(id, |b| {
            b.iter(|| {
                dict.rank1_batch(black_box(&positions), black_box(&mut out));
            });
        });
    }
    for input in &real_data_inputs() {
        let dict = RankSelectDict::build(&input.bits, input.n_bits);
        let positions = deterministic_positions(input.n_bits, 1024, 0x5151_5eed);
        let mut out = vec![0_usize; positions.len()];
        group.throughput(Throughput::Elements(positions.len() as u64));
        let id = BenchmarkId::new(
            "scalar",
            format!("real/{}/n_bits={}", input.label, input.n_bits),
        );
        group.bench_function(id, |b| {
            b.iter(|| {
                dict.rank1_batch(black_box(&positions), black_box(&mut out));
            });
        });
    }
    group.finish();
}

fn bench_select1_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_rank_select/select1_batch");
    for &(label, n_bits) in SIZES {
        let bits = deterministic_words(n_bits, 0x0BAD_F00D_u64 ^ (n_bits as u64));
        let dict = RankSelectDict::build(&bits, n_bits);
        if dict.count_ones() == 0 {
            continue;
        }
        let ks = deterministic_select_ks(dict.count_ones(), 1024, 0xBA1_F00D);
        let mut out = vec![None; ks.len()];
        group.throughput(Throughput::Elements(ks.len() as u64));

        let id = BenchmarkId::new("scalar", label);
        group.bench_function(id, |b| {
            b.iter(|| {
                dict.select1_batch(black_box(&ks), black_box(&mut out));
            });
        });
    }
    for input in &real_data_inputs() {
        let dict = RankSelectDict::build(&input.bits, input.n_bits);
        if dict.count_ones() == 0 {
            continue;
        }
        let ks = deterministic_select_ks(dict.count_ones(), 1024, 0xBA1_F00D);
        let mut out = vec![None; ks.len()];
        group.throughput(Throughput::Elements(ks.len() as u64));
        let id = BenchmarkId::new(
            "scalar",
            format!("real/{}/n_bits={}", input.label, input.n_bits),
        );
        group.bench_function(id, |b| {
            b.iter(|| {
                dict.select1_batch(black_box(&ks), black_box(&mut out));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_build,
    bench_rank1_warm,
    bench_select1_warm,
    bench_rank1_batch,
    bench_select1_batch
);
criterion_main!(benches);

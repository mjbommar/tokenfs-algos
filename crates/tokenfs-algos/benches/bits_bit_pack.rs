//! Benchmarks for `bits::bit_pack` per `docs/v0.2_planning/10_BITS.md` § 2.
//!
//! Reports per-width × per-tier × per-backend throughput on encode + decode.

#![allow(missing_docs)]
// Only consumes `cache_tier_sizes` from the shared support module; the rest
// is shared scaffolding for the workload-matrix benches.
#![allow(dead_code)]

mod support;

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use tokenfs_algos::bits::bit_pack::{
    DynamicBitPacker,
    kernels::{auto, scalar},
};

const WIDTHS: [u32; 7] = [1, 4, 8, 11, 12, 16, 32];

fn deterministic_values(n: usize, width: u32) -> Vec<u32> {
    let mask: u32 = if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    let mut state = 0x9e37_79b9_u32;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(0x0019_660d).wrapping_add(0x3c6e_f35f);
            state & mask
        })
        .collect()
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_bit_pack/decode");

    for (tier_label, tier_bytes) in support::cache_tier_sizes() {
        for width in WIDTHS {
            // Fit roughly `tier_bytes` of decoded output (n * 4 ~= tier_bytes).
            let n = (tier_bytes / 4).max(64);
            let values = deterministic_values(n, width);

            let packer = DynamicBitPacker::new(width);
            let encoded_len = packer.encoded_len(n);
            let mut encoded = vec![0_u8; encoded_len];
            packer.encode_u32_slice(&values, &mut encoded);

            group.throughput(Throughput::Bytes((n * 4) as u64));

            // auto-dispatched
            let id = BenchmarkId::new(format!("auto/w={width}/{tier_label}"), n);
            let mut out = vec![0_u32; n];
            group.bench_function(id, |b| {
                b.iter(|| {
                    auto::decode_u32_slice(
                        black_box(width),
                        black_box(&encoded),
                        black_box(n),
                        black_box(&mut out),
                    );
                });
            });

            // scalar reference
            let id = BenchmarkId::new(format!("scalar/w={width}/{tier_label}"), n);
            let mut out = vec![0_u32; n];
            group.bench_function(id, |b| {
                b.iter(|| {
                    scalar::decode_u32_slice(
                        black_box(width),
                        black_box(&encoded),
                        black_box(n),
                        black_box(&mut out),
                    );
                });
            });
        }
    }

    group.finish();
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("bits_bit_pack/encode");

    for (tier_label, tier_bytes) in support::cache_tier_sizes() {
        for width in WIDTHS {
            let n = (tier_bytes / 4).max(64);
            let values = deterministic_values(n, width);
            let packer = DynamicBitPacker::new(width);
            let encoded_len = packer.encoded_len(n);
            let mut out = vec![0_u8; encoded_len];

            group.throughput(Throughput::Bytes((n * 4) as u64));

            let id = BenchmarkId::new(format!("auto/w={width}/{tier_label}"), n);
            group.bench_function(id, |b| {
                b.iter(|| {
                    auto::encode_u32_slice(
                        black_box(width),
                        black_box(&values),
                        black_box(&mut out),
                    );
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_decode, bench_encode);
criterion_main!(benches);

#![allow(missing_docs)]

mod support;

use std::env;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use support::{
    AccessPattern, WorkloadInput, context_inputs_from_env, real_inputs_from_env, synthetic_inputs,
    workload_matrix_inputs_from_env, write_workload_manifest,
};
use tokenfs_algos::histogram::{
    ByteHistogram,
    bench_internals::{HistogramKernel, add_block_with_kernel, byte_histogram_with_kernel},
};

#[derive(Clone, Copy)]
enum BenchKernel {
    PublicDefault,
    Experimental(HistogramKernel),
}

impl BenchKernel {
    fn id(self) -> &'static str {
        match self {
            Self::PublicDefault => "public-default",
            Self::Experimental(kernel) => kernel.id(),
        }
    }

    fn histogram(self, bytes: &[u8]) -> ByteHistogram {
        match self {
            Self::PublicDefault => ByteHistogram::from_block(bytes),
            Self::Experimental(kernel) => byte_histogram_with_kernel(bytes, kernel),
        }
    }

    fn add_block(self, bytes: &[u8], histogram: &mut ByteHistogram) {
        match self {
            Self::PublicDefault => histogram.add_block(bytes),
            Self::Experimental(kernel) => add_block_with_kernel(bytes, histogram, kernel),
        }
    }
}

fn bench_byte_histogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_histogram");
    let mut inputs = synthetic_inputs();
    inputs.extend(real_inputs_from_env());
    let kernels = bench_kernels();

    for input in &inputs {
        group.throughput(Throughput::Bytes(input.bytes.len() as u64));

        for kernel in &kernels {
            let id = format!("{}/{}", kernel.id(), input.id);
            group.bench_with_input(BenchmarkId::from_parameter(id), &input.bytes, |b, bytes| {
                b.iter(|| black_box(kernel.histogram(black_box(bytes))));
            });
        }
    }

    group.finish();

    if env::var_os("TOKENFS_ALGOS_CONTEXT_SWEEP").is_some() {
        bench_context_histogram(c, "file_histogram_64k", 64 * 1024, &kernels);
        bench_context_histogram(c, "sequential_histogram_4k", 4 * 1024, &kernels);
    }

    if env::var_os("TOKENFS_ALGOS_WORKLOAD_MATRIX").is_some() {
        bench_workload_matrix(c, &kernels);
    }
}

fn bench_kernels() -> Vec<BenchKernel> {
    if env::var_os("TOKENFS_ALGOS_ADAPTIVE_ONLY").is_some() {
        let mut kernels = HistogramKernel::adaptive()
            .into_iter()
            .map(BenchKernel::Experimental)
            .collect::<Vec<_>>();
        if env::var_os("TOKENFS_ALGOS_INCLUDE_DIRECT").is_some() {
            kernels.insert(0, BenchKernel::Experimental(HistogramKernel::DirectU64));
        }
        kernels
    } else if env::var_os("TOKENFS_ALGOS_KERNEL_SWEEP").is_some() {
        HistogramKernel::all()
            .into_iter()
            .map(BenchKernel::Experimental)
            .collect()
    } else {
        vec![BenchKernel::PublicDefault]
    }
}

fn bench_context_histogram(
    c: &mut Criterion,
    group_name: &'static str,
    chunk_size: usize,
    kernels: &[BenchKernel],
) {
    let mut group = c.benchmark_group(group_name);
    let inputs = context_inputs_from_env();

    for input in &inputs {
        group.throughput(Throughput::Bytes(input.bytes.len() as u64));

        for kernel in kernels {
            let id = format!("{}/{}", kernel.id(), input.id);
            group.bench_with_input(BenchmarkId::from_parameter(id), &input.bytes, |b, bytes| {
                b.iter(|| {
                    let mut histogram = ByteHistogram::new();
                    for chunk in black_box(bytes).chunks(chunk_size) {
                        kernel.add_block(chunk, &mut histogram);
                    }
                    black_box(histogram)
                });
            });
        }
    }

    group.finish();
}

fn bench_workload_matrix(c: &mut Criterion, kernels: &[BenchKernel]) {
    let inputs = workload_matrix_inputs_from_env();
    write_workload_manifest(&inputs);

    let mut group = c.benchmark_group("workload_matrix");

    for input in &inputs {
        group.throughput(Throughput::Bytes(input.processed_bytes as u64));

        for kernel in kernels {
            let id = format!("{}/{}", kernel.id(), input.id);
            group.bench_with_input(BenchmarkId::from_parameter(id), input, |b, workload| {
                b.iter(|| black_box(run_workload(*kernel, black_box(workload))));
            });
        }
    }

    group.finish();
}

fn run_workload(kernel: BenchKernel, workload: &WorkloadInput) -> ByteHistogram {
    let bytes = workload.bytes();
    match &workload.access {
        AccessPattern::WholeBlock => kernel.histogram(bytes),
        AccessPattern::Sequential { chunk_size } => histogram_by_chunks(kernel, bytes, *chunk_size),
        AccessPattern::ReadAhead { chunk_size } => histogram_by_chunks(kernel, bytes, *chunk_size),
        AccessPattern::Random {
            chunk_size,
            offsets,
        }
        | AccessPattern::ZipfianHotCold {
            chunk_size,
            offsets,
        } => histogram_by_offsets(kernel, bytes, *chunk_size, offsets),
        AccessPattern::HotRepeat {
            chunk_size,
            repeats,
        } => histogram_repeated_chunks(kernel, bytes, *chunk_size, *repeats),
        AccessPattern::ColdSweep { chunk_size } => histogram_by_chunks(kernel, bytes, *chunk_size),
        AccessPattern::SameFileRepeat { repeats } => {
            histogram_repeated_block(kernel, bytes, *repeats)
        }
        AccessPattern::ParallelSequential {
            chunk_size,
            threads,
        } => histogram_parallel(kernel, bytes, *chunk_size, *threads),
    }
}

fn histogram_by_chunks(kernel: BenchKernel, bytes: &[u8], chunk_size: usize) -> ByteHistogram {
    let mut histogram = ByteHistogram::new();
    for chunk in bytes.chunks(chunk_size) {
        kernel.add_block(chunk, &mut histogram);
    }
    histogram
}

fn histogram_by_offsets(
    kernel: BenchKernel,
    bytes: &[u8],
    chunk_size: usize,
    offsets: &[usize],
) -> ByteHistogram {
    let mut histogram = ByteHistogram::new();
    for &offset in offsets {
        let end = offset + chunk_size;
        kernel.add_block(&bytes[offset..end], &mut histogram);
    }
    histogram
}

fn histogram_repeated_chunks(
    kernel: BenchKernel,
    bytes: &[u8],
    chunk_size: usize,
    repeats: usize,
) -> ByteHistogram {
    let mut histogram = ByteHistogram::new();
    for _ in 0..repeats {
        for chunk in bytes.chunks(chunk_size) {
            kernel.add_block(chunk, &mut histogram);
        }
    }
    histogram
}

fn histogram_repeated_block(kernel: BenchKernel, bytes: &[u8], repeats: usize) -> ByteHistogram {
    let mut histogram = ByteHistogram::new();
    for _ in 0..repeats {
        kernel.add_block(bytes, &mut histogram);
    }
    histogram
}

fn histogram_parallel(
    kernel: BenchKernel,
    bytes: &[u8],
    chunk_size: usize,
    threads: usize,
) -> ByteHistogram {
    if threads <= 1 || bytes.len() < chunk_size {
        return histogram_by_chunks(kernel, bytes, chunk_size);
    }

    let mut partials = Vec::with_capacity(threads);

    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);

        for index in 0..threads {
            let start = bytes.len() * index / threads;
            let end = bytes.len() * (index + 1) / threads;
            let shard = &bytes[start..end];
            handles.push(scope.spawn(move || histogram_by_chunks(kernel, shard, chunk_size)));
        }

        for handle in handles {
            match handle.join() {
                Ok(partial) => partials.push(partial),
                Err(payload) => std::panic::resume_unwind(payload),
            }
        }
    });

    let mut histogram = ByteHistogram::new();
    for partial in partials {
        histogram += &partial;
    }
    histogram
}

criterion_group!(benches, bench_byte_histogram);
criterion_main!(benches);

#![allow(missing_docs)]

use std::{env, sync::OnceLock};

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use tokenfs_algos::{
    byteclass, entropy, fingerprint, histogram::ByteHistogram, runlength, selector, sketch,
    structure,
};

#[derive(Clone)]
struct PrimitiveInput {
    case: &'static str,
    source: &'static str,
    content: &'static str,
    entropy: &'static str,
    pattern: &'static str,
    bytes: Vec<u8>,
}

#[derive(Clone, Copy)]
enum PrimitiveKernel {
    FingerprintBlockAuto,
    FingerprintBlockScalar,
    FingerprintExtentAuto,
    FingerprintExtentScalar,
    SketchMisraGriesK16,
    SketchCountMin4x1024,
    SketchCrc32Hash4Auto,
    SketchCrc32Hash4Sse42,
    SketchCrc32Hash4Scalar,
    SketchEntropyLut,
    ByteClassClassify,
    RunLengthSummarize,
    StructureSummarize,
    EntropyH1FromHistogram,
    SelectorSignals,
}

impl PrimitiveKernel {
    fn all() -> Vec<Self> {
        let mut kernels = vec![
            Self::FingerprintBlockAuto,
            Self::FingerprintBlockScalar,
            Self::FingerprintExtentAuto,
            Self::FingerprintExtentScalar,
            Self::SketchMisraGriesK16,
            Self::SketchCountMin4x1024,
            Self::SketchCrc32Hash4Auto,
            Self::SketchEntropyLut,
            Self::ByteClassClassify,
            Self::RunLengthSummarize,
            Self::StructureSummarize,
            Self::EntropyH1FromHistogram,
            Self::SelectorSignals,
        ];
        #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
        if sketch::kernels::sse42::is_available() {
            kernels.push(Self::SketchCrc32Hash4Sse42);
        }
        kernels.push(Self::SketchCrc32Hash4Scalar);
        kernels
    }

    fn id(self) -> &'static str {
        match self {
            Self::FingerprintBlockAuto => "fingerprint-block-auto",
            Self::FingerprintBlockScalar => "fingerprint-block-scalar",
            Self::FingerprintExtentAuto => "fingerprint-extent-auto",
            Self::FingerprintExtentScalar => "fingerprint-extent-scalar",
            Self::SketchMisraGriesK16 => "sketch-misra-gries-k16",
            Self::SketchCountMin4x1024 => "sketch-count-min-4x1024",
            Self::SketchCrc32Hash4Auto => "sketch-crc32-hash4-auto",
            Self::SketchCrc32Hash4Sse42 => "sketch-crc32-hash4-sse42",
            Self::SketchCrc32Hash4Scalar => "sketch-crc32-hash4-scalar",
            Self::SketchEntropyLut => "sketch-entropy-lut",
            Self::ByteClassClassify => "byteclass-classify",
            Self::RunLengthSummarize => "runlength-summarize",
            Self::StructureSummarize => "structure-summarize",
            Self::EntropyH1FromHistogram => "entropy-h1-from-histogram",
            Self::SelectorSignals => "selector-signals",
        }
    }

    fn primitive(self) -> &'static str {
        match self {
            Self::FingerprintBlockAuto | Self::FingerprintBlockScalar => "fingerprint-block",
            Self::FingerprintExtentAuto | Self::FingerprintExtentScalar => "fingerprint-extent",
            Self::SketchMisraGriesK16
            | Self::SketchCountMin4x1024
            | Self::SketchCrc32Hash4Auto
            | Self::SketchCrc32Hash4Sse42
            | Self::SketchCrc32Hash4Scalar
            | Self::SketchEntropyLut => "sketch",
            Self::ByteClassClassify => "byteclass",
            Self::RunLengthSummarize => "runlength",
            Self::StructureSummarize => "structure",
            Self::EntropyH1FromHistogram => "entropy",
            Self::SelectorSignals => "selector",
        }
    }
}

fn bench_primitive_matrix(c: &mut Criterion) {
    let inputs = primitive_inputs();
    let kernels = primitive_kernels_from_env();
    let mut group = c.benchmark_group("primitive_matrix");

    for input in &inputs {
        group.throughput(Throughput::Bytes(input.bytes.len() as u64));

        for kernel in &kernels {
            let id = format!(
                "{}/primitive={}/case={}/source={}/content={}/entropy={}/pattern={}/bytes={}",
                kernel.id(),
                kernel.primitive(),
                input.case,
                input.source,
                input.content,
                input.entropy,
                input.pattern,
                input.bytes.len()
            );
            group.bench_with_input(BenchmarkId::from_parameter(id), input, |b, input| {
                b.iter(|| black_box(run_kernel(*kernel, black_box(&input.bytes))));
            });
        }
    }

    group.finish();
}

fn primitive_kernels_from_env() -> Vec<PrimitiveKernel> {
    let Some(filter) = env::var("TOKENFS_ALGOS_PRIMITIVE_FILTER").ok() else {
        return PrimitiveKernel::all();
    };
    let tokens = filter
        .split([',', ';', ':', ' '])
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();

    PrimitiveKernel::all()
        .into_iter()
        .filter(|kernel| {
            tokens
                .iter()
                .any(|token| kernel.id().contains(token) || kernel.primitive().contains(token))
        })
        .collect()
}

fn primitive_inputs() -> Vec<PrimitiveInput> {
    let sizes = [256_usize, 4 * 1024, 64 * 1024, 1024 * 1024];
    let mut inputs = Vec::new();
    for size in sizes {
        inputs.extend([
            PrimitiveInput {
                case: "zeros",
                source: "synthetic",
                content: "binary",
                entropy: "low",
                pattern: "zeros",
                bytes: vec![0; size],
            },
            PrimitiveInput {
                case: "prng",
                source: "synthetic",
                content: "binary",
                entropy: "high",
                pattern: "prng",
                bytes: deterministic_prng(size, size as u64 ^ 0xA17E_A17E),
            },
            PrimitiveInput {
                case: "text",
                source: "synthetic",
                content: "text",
                entropy: "medium",
                pattern: "ascii-text",
                bytes: repeated_text(size),
            },
            PrimitiveInput {
                case: "runs",
                source: "synthetic",
                content: "binary",
                entropy: "low",
                pattern: "long-runs",
                bytes: run_heavy(size),
            },
            PrimitiveInput {
                case: "motif-64",
                source: "synthetic",
                content: "binary",
                entropy: "medium",
                pattern: "repeated-motif",
                bytes: repeated_random_motif(size, 64),
            },
        ]);
    }
    inputs
}

fn run_kernel(kernel: PrimitiveKernel, bytes: &[u8]) -> u64 {
    match kernel {
        PrimitiveKernel::FingerprintBlockAuto => fingerprint_blocks(bytes, fingerprint::block),
        PrimitiveKernel::FingerprintBlockScalar => {
            fingerprint_blocks(bytes, fingerprint::kernels::scalar::block)
        }
        PrimitiveKernel::FingerprintExtentAuto => fold_extent(fingerprint::extent(bytes)),
        PrimitiveKernel::FingerprintExtentScalar => {
            fold_extent(fingerprint::kernels::scalar::extent(bytes))
        }
        PrimitiveKernel::SketchMisraGriesK16 => {
            let mut sketch = sketch::MisraGries::<16>::new();
            sketch.update_iter(bytes.iter().copied().map(u32::from));
            sketch
                .candidates()
                .iter()
                .fold(sketch.observations(), |acc, (item, count)| {
                    acc ^ u64::from(*item) ^ u64::from(*count)
                })
        }
        PrimitiveKernel::SketchCountMin4x1024 => {
            let mut sketch = sketch::CountMinSketch::<4, 1024>::new();
            for window in bytes.windows(4) {
                let value = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
                sketch.update(value);
            }
            sketch.estimate(0) as u64 ^ sketch.observations()
        }
        PrimitiveKernel::SketchCrc32Hash4Auto => {
            let mut bins = [0_u32; 4096];
            sketch::crc32_hash4_bins(bytes, &mut bins);
            bins.iter()
                .fold(0_u64, |acc, count| acc + u64::from(*count))
        }
        PrimitiveKernel::SketchCrc32Hash4Sse42 => {
            let mut bins = [0_u32; 4096];
            #[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
            {
                if sketch::kernels::sse42::is_available() {
                    // SAFETY: availability was checked immediately above.
                    unsafe { sketch::kernels::sse42::crc32_hash4_bins(bytes, &mut bins) };
                }
            }
            bins.iter()
                .fold(0_u64, |acc, count| acc + u64::from(*count))
        }
        PrimitiveKernel::SketchCrc32Hash4Scalar => {
            let mut bins = [0_u32; 4096];
            sketch::kernels::scalar::crc32_hash4_bins(bytes, &mut bins);
            bins.iter()
                .fold(0_u64, |acc, count| acc + u64::from(*count))
        }
        PrimitiveKernel::SketchEntropyLut => {
            let mut counts = [0_u32; 256];
            for &byte in bytes {
                counts[byte as usize] += 1;
            }
            let lut = entropy_lut();
            sketch::entropy_from_counts_u32_lut(&counts, bytes.len() as u64, lut).to_bits() as u64
        }
        PrimitiveKernel::ByteClassClassify => {
            let counts = byteclass::classify(bytes);
            counts.printable_ascii ^ counts.whitespace ^ counts.control ^ counts.high_bit
        }
        PrimitiveKernel::RunLengthSummarize => {
            let summary = runlength::summarize(bytes);
            summary.transitions ^ summary.bytes_in_runs_ge4 ^ summary.runs_ge4
        }
        PrimitiveKernel::StructureSummarize => {
            let summary = structure::summarize(bytes);
            summary.zero_bytes
                ^ u64::from(summary.unique_bytes)
                ^ summary.repeated_period.unwrap_or(0) as u64
                ^ summary.zero_pages_4k
        }
        PrimitiveKernel::EntropyH1FromHistogram => {
            let histogram = ByteHistogram::from_block(bytes);
            entropy::shannon::h1(&histogram).to_bits() as u64
        }
        PrimitiveKernel::SelectorSignals => {
            let signals = selector::signals(bytes);
            signals.fingerprint.h1.to_bits() as u64
                ^ u64::from(signals.structure.unique_bytes)
                ^ u64::from(signals.skip_compression_candidate)
        }
    }
}

fn fingerprint_blocks(
    bytes: &[u8],
    block: fn(&[u8; fingerprint::BLOCK_SIZE]) -> fingerprint::BlockFingerprint,
) -> u64 {
    let mut acc = 0_u64;
    for chunk in bytes.chunks_exact(fingerprint::BLOCK_SIZE) {
        let block_bytes: &[u8; fingerprint::BLOCK_SIZE] = chunk
            .try_into()
            .expect("chunks_exact must yield fingerprint-sized blocks");
        let fp = block(block_bytes);
        acc ^= u64::from(fp.h1_q4)
            ^ (u64::from(fp.h4_q4) << 8)
            ^ (u64::from(fp.rl_runs_ge4) << 16)
            ^ (u64::from(fp.top4_coverage_q8) << 32)
            ^ (u64::from(fp.byte_class) << 40);
    }
    acc
}

fn fold_extent(fp: fingerprint::ExtentFingerprint) -> u64 {
    fp.h1.to_bits() as u64
        ^ ((fp.h4.to_bits() as u64) << 1)
        ^ ((fp.rl_fraction.to_bits() as u64) << 2)
        ^ ((fp.top16_coverage.to_bits() as u64) << 3)
}

fn entropy_lut() -> &'static sketch::CLog2Lut<257> {
    static LUT: OnceLock<sketch::CLog2Lut<257>> = OnceLock::new();
    LUT.get_or_init(sketch::CLog2Lut::new)
}

fn deterministic_prng(size: usize, seed: u64) -> Vec<u8> {
    let mut state = seed;
    let mut out = Vec::with_capacity(size);
    for _ in 0..size {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        out.push(state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8);
    }
    out
}

fn repeated_random_motif(size: usize, motif_len: usize) -> Vec<u8> {
    deterministic_prng(motif_len, 0x243f_6a88_85a3_08d3)
        .into_iter()
        .cycle()
        .take(size)
        .collect()
}

fn run_heavy(size: usize) -> Vec<u8> {
    let pattern = [(0x00, 64_usize), (0xff, 31), (b'A', 17), (b'\n', 1)];
    let mut out = Vec::with_capacity(size);
    while out.len() < size {
        for (byte, run) in pattern {
            let take = (size - out.len()).min(run);
            out.extend(std::iter::repeat_n(byte, take));
            if out.len() == size {
                break;
            }
        }
    }
    out
}

fn repeated_text(size: usize) -> Vec<u8> {
    const TEXT: &[u8] =
        b"tokenfs-algos measures byte streams, entropy, fingerprints, and selectors.\n";
    TEXT.iter().copied().cycle().take(size).collect()
}

criterion_group!(benches, bench_primitive_matrix);
criterion_main!(benches);

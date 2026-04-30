#![allow(missing_docs)]

use std::{
    env,
    fmt::Write as _,
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write as _},
    path::{Path, PathBuf},
};

use tokenfs_algos::dispatch::{
    ApiContext, ContentKind, EntropyClass, EntropyScale, ProcessorProfile, WorkloadShape,
    plan_histogram,
};

pub(crate) struct BenchInput {
    pub(crate) id: String,
    pub(crate) bytes: Vec<u8>,
}

impl BenchInput {
    pub(crate) fn new(id: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            id: id.into(),
            bytes,
        }
    }
}

#[derive(Clone)]
pub(crate) enum AccessPattern {
    WholeBlock,
    Sequential {
        chunk_size: usize,
    },
    Random {
        chunk_size: usize,
        offsets: Vec<usize>,
    },
    ParallelSequential {
        chunk_size: usize,
        threads: usize,
    },
}

impl AccessPattern {
    fn name(&self) -> &'static str {
        match self {
            Self::WholeBlock => "block",
            Self::Sequential { .. } => "sequential",
            Self::Random { .. } => "random",
            Self::ParallelSequential { .. } => "parallel-sequential",
        }
    }

    fn chunk_size(&self) -> usize {
        match self {
            Self::WholeBlock => 0,
            Self::Sequential { chunk_size }
            | Self::Random { chunk_size, .. }
            | Self::ParallelSequential { chunk_size, .. } => *chunk_size,
        }
    }

    fn threads(&self) -> usize {
        match self {
            Self::ParallelSequential { threads, .. } => *threads,
            _ => 1,
        }
    }

    fn processed_bytes(&self, buffer_bytes: usize) -> usize {
        match self {
            Self::WholeBlock | Self::Sequential { .. } | Self::ParallelSequential { .. } => {
                buffer_bytes
            }
            Self::Random {
                chunk_size,
                offsets,
            } => chunk_size.saturating_mul(offsets.len()),
        }
    }
}

pub(crate) struct WorkloadInput {
    pub(crate) id: String,
    pub(crate) label: String,
    pub(crate) source: &'static str,
    pub(crate) content: &'static str,
    pub(crate) entropy: &'static str,
    pub(crate) scale: &'static str,
    pub(crate) pattern: &'static str,
    pub(crate) bytes: Vec<u8>,
    pub(crate) access: AccessPattern,
    pub(crate) processed_bytes: usize,
    h1_bits_per_byte: f64,
    h1_4k_mean: f64,
    h1_4k_min: f64,
    h1_4k_max: f64,
    planned_kernel: &'static str,
    planned_chunk_bytes: usize,
    planned_sample_bytes: usize,
    plan_reason: &'static str,
}

struct Payload {
    label: String,
    source: &'static str,
    content: &'static str,
    entropy: &'static str,
    scale: &'static str,
    pattern: &'static str,
    bytes: Vec<u8>,
}

impl Payload {
    fn new(
        label: impl Into<String>,
        source: &'static str,
        content: &'static str,
        entropy: &'static str,
        scale: &'static str,
        pattern: &'static str,
        bytes: Vec<u8>,
    ) -> Self {
        Self {
            label: label.into(),
            source,
            content,
            entropy,
            scale,
            pattern,
            bytes,
        }
    }
}

impl WorkloadInput {
    fn new(payload: &Payload, access: AccessPattern, profile: &ProcessorProfile) -> Self {
        let processed_bytes = access.processed_bytes(payload.bytes.len());
        let h1_bits_per_byte = shannon_entropy(&payload.bytes);
        let (h1_4k_mean, h1_4k_min, h1_4k_max) = chunk_entropy_stats(&payload.bytes, 4 * 1024);
        let workload = workload_shape(payload, &access, processed_bytes);
        let plan = plan_histogram(profile, &workload);
        let id = format!(
            "case={}/source={}/content={}/entropy={}/scale={}/pattern={}/access={}/chunk={}/threads={}/bytes={}",
            payload.label,
            payload.source,
            payload.content,
            payload.entropy,
            payload.scale,
            payload.pattern,
            access.name(),
            access.chunk_size(),
            access.threads(),
            processed_bytes,
        );

        Self {
            id,
            label: payload.label.clone(),
            source: payload.source,
            content: payload.content,
            entropy: payload.entropy,
            scale: payload.scale,
            pattern: payload.pattern,
            bytes: payload.bytes.clone(),
            access,
            processed_bytes,
            h1_bits_per_byte,
            h1_4k_mean,
            h1_4k_min,
            h1_4k_max,
            planned_kernel: plan.strategy.as_str(),
            planned_chunk_bytes: plan.chunk_bytes,
            planned_sample_bytes: plan.sample_bytes,
            plan_reason: plan.reason,
        }
    }
}

pub(crate) fn synthetic_inputs() -> Vec<BenchInput> {
    let mut inputs = Vec::new();

    for size in [256_usize, 4 * 1024, 64 * 1024, 1024 * 1024] {
        inputs.push(BenchInput::new(
            format!("synthetic/zeros/{size}"),
            vec![0; size],
        ));
        inputs.push(BenchInput::new(
            format!("synthetic/uniform-cycle/{size}"),
            uniform_cycle(size),
        ));
        inputs.push(BenchInput::new(
            format!("synthetic/prng/{size}"),
            deterministic_prng(size, 0x9e37_79b9_7f4a_7c15),
        ));
        inputs.push(BenchInput::new(
            format!("synthetic/runs/{size}"),
            run_heavy(size),
        ));
        inputs.push(BenchInput::new(
            format!("synthetic/text/{size}"),
            repeated_text(size),
        ));
    }

    inputs
}

pub(crate) fn workload_matrix_inputs_from_env() -> Vec<WorkloadInput> {
    let level = env::var("TOKENFS_ALGOS_MATRIX_LEVEL").unwrap_or_else(|_| "quick".into());
    let full = level == "full";
    let threaded = full || env::var_os("TOKENFS_ALGOS_THREAD_SWEEP").is_some();
    let base_size = if full { 4 * 1024 * 1024 } else { 1024 * 1024 };
    let mut payloads = synthetic_workload_payloads(base_size);
    let profile = ProcessorProfile::detect();

    if let Some(path) = env::var_os("TOKENFS_ALGOS_REAL_DATA") {
        match real_workload_payloads_from_path(Path::new(&path), base_size) {
            Ok(real_payloads) => payloads.extend(real_payloads),
            Err(error) => {
                eprintln!(
                    "skipping real workload matrix payloads for `{}`: {error}",
                    PathBuf::from(path).display()
                );
            }
        }
    }

    let mut inputs = Vec::new();
    for payload in &payloads {
        for access in workload_access_patterns(payload.bytes.len(), full, threaded) {
            inputs.push(WorkloadInput::new(payload, access, &profile));
        }
    }

    inputs
}

pub(crate) fn write_workload_manifest(inputs: &[WorkloadInput]) {
    let path = env::var_os("TOKENFS_ALGOS_WORKLOAD_MANIFEST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/tokenfs-algos/workload-manifest.jsonl"));

    if let Some(parent) = path.parent()
        && let Err(error) = fs::create_dir_all(parent)
    {
        eprintln!(
            "failed to create workload manifest directory `{}`: {error}",
            parent.display()
        );
        return;
    }

    let mut file = match File::create(&path) {
        Ok(file) => file,
        Err(error) => {
            eprintln!(
                "failed to create workload manifest `{}`: {error}",
                path.display()
            );
            return;
        }
    };

    for input in inputs {
        let line = workload_manifest_line(input);
        if let Err(error) = file.write_all(line.as_bytes()) {
            eprintln!(
                "failed to write workload manifest `{}`: {error}",
                path.display()
            );
            return;
        }
        if let Err(error) = file.write_all(b"\n") {
            eprintln!(
                "failed to write workload manifest `{}`: {error}",
                path.display()
            );
            return;
        }
    }
}

pub(crate) fn real_inputs_from_env() -> Vec<BenchInput> {
    let Some(path) = env::var_os("TOKENFS_ALGOS_REAL_DATA") else {
        return Vec::new();
    };

    match real_inputs_from_path(Path::new(&path)) {
        Ok(inputs) => inputs,
        Err(error) => {
            eprintln!(
                "skipping real-data benchmarks for `{}`: {error}",
                PathBuf::from(path).display()
            );
            Vec::new()
        }
    }
}

pub(crate) fn context_inputs_from_env() -> Vec<BenchInput> {
    let mut inputs = synthetic_context_inputs();

    let Some(path) = env::var_os("TOKENFS_ALGOS_REAL_DATA") else {
        return inputs;
    };

    match real_context_inputs_from_path(Path::new(&path)) {
        Ok(real_inputs) => inputs.extend(real_inputs),
        Err(error) => {
            eprintln!(
                "skipping real context benchmarks for `{}`: {error}",
                PathBuf::from(path).display()
            );
        }
    }

    inputs
}

fn synthetic_workload_payloads(size: usize) -> Vec<Payload> {
    vec![
        Payload::new(
            "zeros",
            "synthetic",
            "binary",
            "low",
            "flat",
            "zeros",
            vec![0; size],
        ),
        Payload::new(
            "prng",
            "synthetic",
            "binary",
            "high",
            "flat",
            "prng",
            deterministic_prng(size, 0xe703_7ed1_a0b4_28db),
        ),
        Payload::new(
            "ascii-text",
            "synthetic",
            "text",
            "medium",
            "micro",
            "ascii-text",
            varied_text(size),
        ),
        Payload::new(
            "repeated-random-256",
            "synthetic",
            "binary",
            "medium",
            "micro",
            "repeated-random-256",
            repeated_random_motif(size, 256),
        ),
        Payload::new(
            "block-palette-4k",
            "synthetic",
            "binary",
            "mixed",
            "meso",
            "block-palette-4k",
            block_palette(size, 4 * 1024),
        ),
        Payload::new(
            "macro-regions",
            "synthetic",
            "mixed",
            "mixed",
            "macro",
            "macro-regions",
            macro_regions(size),
        ),
        Payload::new(
            "binary-words",
            "synthetic",
            "binary",
            "medium",
            "micro",
            "binary-words",
            binary_words(size),
        ),
    ]
}

fn real_workload_payloads_from_path(path: &Path, size: usize) -> std::io::Result<Vec<Payload>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    if len < size as u64 {
        return Ok(Vec::new());
    }

    let label = path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("real-data");
    let mut payloads = Vec::new();

    for (where_, offset) in fixed_offsets(len, size as u64) {
        let bytes = read_at(&mut file, offset, size)?;
        payloads.push(Payload::new(
            format!("{}-{where_}", sanitize_id_fragment(label)),
            "real",
            "binary",
            "mixed",
            "macro",
            match where_ {
                "first" => "real-first",
                "middle" => "real-middle",
                "last" => "real-last",
                _ => "real-slice",
            },
            bytes,
        ));
    }

    Ok(payloads)
}

fn workload_access_patterns(buffer_bytes: usize, full: bool, threaded: bool) -> Vec<AccessPattern> {
    let mut patterns = vec![
        AccessPattern::WholeBlock,
        AccessPattern::Sequential {
            chunk_size: 4 * 1024,
        },
        AccessPattern::Sequential {
            chunk_size: 64 * 1024,
        },
        AccessPattern::Random {
            chunk_size: 1,
            offsets: random_offsets(buffer_bytes, 1, if full { 65_536 } else { 16_384 }, 1, 1),
        },
        AccessPattern::Random {
            chunk_size: 4 * 1024,
            offsets: random_offsets(
                buffer_bytes,
                4 * 1024,
                if full { 256 } else { 64 },
                0x517c_c1b7_2722_0a95,
                4 * 1024,
            ),
        },
    ];

    if full {
        patterns.extend([
            AccessPattern::Sequential { chunk_size: 1024 },
            AccessPattern::Sequential {
                chunk_size: 8 * 1024,
            },
            AccessPattern::Sequential {
                chunk_size: 16 * 1024,
            },
        ]);
    }

    if threaded {
        patterns.extend([
            AccessPattern::ParallelSequential {
                chunk_size: 64 * 1024,
                threads: 2,
            },
            AccessPattern::ParallelSequential {
                chunk_size: 64 * 1024,
                threads: 4,
            },
        ]);
    }

    patterns
}

fn real_inputs_from_path(path: &Path) -> std::io::Result<Vec<BenchInput>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let label = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("real-data");
    let mut inputs = Vec::new();

    for size in [4 * 1024_usize, 64 * 1024, 1024 * 1024] {
        if len < size as u64 {
            continue;
        }

        for (where_, offset) in fixed_offsets(len, size as u64) {
            let bytes = read_at(&mut file, offset, size)?;
            inputs.push(BenchInput::new(
                format!("real/{label}/{where_}/{size}"),
                bytes,
            ));
        }
    }

    Ok(inputs)
}

fn synthetic_context_inputs() -> Vec<BenchInput> {
    let mut mixed = Vec::new();
    mixed.extend(vec![0; 1024 * 1024]);
    mixed.extend(deterministic_prng(1024 * 1024, 0xa076_1d64_78bd_642f));
    mixed.extend(run_heavy(1024 * 1024));
    mixed.extend(repeated_text(1024 * 1024));

    vec![
        BenchInput::new("synthetic-file/mixed-4m", mixed),
        BenchInput::new(
            "synthetic-file/prng-4m",
            deterministic_prng(4 * 1024 * 1024, 13),
        ),
        BenchInput::new("synthetic-file/zeros-4m", vec![0; 4 * 1024 * 1024]),
    ]
}

fn real_context_inputs_from_path(path: &Path) -> std::io::Result<Vec<BenchInput>> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    let label = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("real-data");
    let size = 16 * 1024 * 1024_usize;

    if len < size as u64 {
        return Ok(Vec::new());
    }

    let mut inputs = Vec::new();
    for (where_, offset) in fixed_offsets(len, size as u64) {
        let bytes = read_at(&mut file, offset, size)?;
        inputs.push(BenchInput::new(
            format!("real-file/{label}/{where_}/{size}"),
            bytes,
        ));
    }

    Ok(inputs)
}

fn fixed_offsets(len: u64, size: u64) -> [(&'static str, u64); 3] {
    [
        ("first", 0),
        ("middle", len.saturating_sub(size) / 2),
        ("last", len.saturating_sub(size)),
    ]
}

fn read_at(file: &mut File, offset: u64, size: usize) -> std::io::Result<Vec<u8>> {
    let mut bytes = vec![0; size];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn uniform_cycle(size: usize) -> Vec<u8> {
    (0_u8..=255).cycle().take(size).collect()
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

fn block_palette(size: usize, block_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let mut block_index = 0_usize;

    while out.len() < size {
        let remaining = size - out.len();
        let take = remaining.min(block_size);
        match block_index % 4 {
            0 => out.extend(std::iter::repeat_n(0, take)),
            1 => out.extend(deterministic_prng(take, block_index as u64 + 11)),
            2 => out.extend(repeated_random_motif(take, 64)),
            _ => out.extend(repeated_text(take)),
        }
        block_index += 1;
    }

    out
}

fn macro_regions(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let quarter = size / 4;
    out.extend(std::iter::repeat_n(0, quarter));
    out.extend(deterministic_prng(quarter, 0xa076_1d64_78bd_642f));
    out.extend(run_heavy(quarter));
    out.extend(varied_text(size.saturating_sub(out.len())));
    out
}

fn binary_words(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let mut value = 0x9e37_79b9_7f4a_7c15_u64;

    while out.len() < size {
        value = value.wrapping_mul(6364136223846793005).wrapping_add(1);
        let word = (value & 0x0000_ffff_ffff_ffff).to_le_bytes();
        let take = (size - out.len()).min(word.len());
        out.extend_from_slice(&word[..take]);
    }

    out
}

fn varied_text(size: usize) -> Vec<u8> {
    const LINES: &[&[u8]] = &[
        b"tokenfs records paths, offsets, extents, cache lines, and byte histograms.\n",
        b"src/lib.rs:42 measures sequential access against random access with fixed seeds.\n",
        b"{\"level\":\"info\",\"target\":\"tokenfs\",\"event\":\"window\",\"bytes\":4096}\n",
        b"Entropy changes by scale: micro motifs, meso blocks, macro file regions.\n",
    ];

    let mut out = Vec::with_capacity(size);
    let mut index = 0_usize;

    while out.len() < size {
        let line = LINES[index % LINES.len()];
        let take = (size - out.len()).min(line.len());
        out.extend_from_slice(&line[..take]);
        index += 1;
    }

    out
}

fn random_offsets(
    buffer_bytes: usize,
    chunk_size: usize,
    count: usize,
    seed: u64,
    alignment: usize,
) -> Vec<usize> {
    if chunk_size == 0 || buffer_bytes < chunk_size {
        return Vec::new();
    }

    let align = alignment.max(1);
    let max_start = buffer_bytes - chunk_size;
    let slots = (max_start / align) + 1;
    let mut state = seed;
    let mut offsets = Vec::with_capacity(count);

    for _ in 0..count {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let slot = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as usize) % slots;
        offsets.push(slot * align);
    }

    offsets
}

fn run_heavy(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let pattern = [
        (0x00, 64_usize),
        (0xff, 31),
        (b'A', 17),
        (b'\n', 1),
        (0x7f, 9),
    ];

    while out.len() < size {
        for (byte, run) in pattern {
            let remaining = size - out.len();
            let take = remaining.min(run);
            out.extend(std::iter::repeat_n(byte, take));
            if out.len() == size {
                break;
            }
        }
    }

    out
}

fn repeated_text(size: usize) -> Vec<u8> {
    const TEXT: &[u8] = b"tokenfs-algos measures byte streams: ascii text, paths, symbols, and structured records.\n";
    TEXT.iter().copied().cycle().take(size).collect()
}

fn shannon_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }

    let mut counts = [0_u64; 256];
    for &byte in bytes {
        counts[byte as usize] += 1;
    }

    let total = bytes.len() as f64;
    let entropy = counts
        .into_iter()
        .filter(|count| *count != 0)
        .map(|count| {
            let p = count as f64 / total;
            -p * p.log2()
        })
        .sum::<f64>();

    if entropy == 0.0 { 0.0 } else { entropy }
}

fn chunk_entropy_stats(bytes: &[u8], chunk_size: usize) -> (f64, f64, f64) {
    if bytes.is_empty() || chunk_size == 0 {
        return (0.0, 0.0, 0.0);
    }

    let mut chunks = 0_usize;
    let mut sum = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;

    for chunk in bytes.chunks(chunk_size) {
        let entropy = shannon_entropy(chunk);
        chunks += 1;
        sum += entropy;
        min = min.min(entropy);
        max = max.max(entropy);
    }

    (sum / chunks as f64, min, max)
}

fn workload_manifest_line(input: &WorkloadInput) -> String {
    let mut line = String::new();
    line.push('{');
    write_json_str(&mut line, "workload_id", &input.id);
    write_json_str(&mut line, "case", &input.label);
    write_json_str(&mut line, "source", input.source);
    write_json_str(&mut line, "content", input.content);
    write_json_str(&mut line, "entropy_class", input.entropy);
    write_json_str(&mut line, "entropy_scale", input.scale);
    write_json_str(&mut line, "pattern", input.pattern);
    write_json_str(&mut line, "access", input.access.name());
    write_json_num(&mut line, "chunk_size", input.access.chunk_size());
    write_json_num(&mut line, "threads", input.access.threads());
    write_json_num(&mut line, "buffer_bytes", input.bytes.len());
    write_json_num(&mut line, "processed_bytes", input.processed_bytes);
    write_json_float(&mut line, "h1_bits_per_byte", input.h1_bits_per_byte);
    write_json_float(&mut line, "h1_4k_mean", input.h1_4k_mean);
    write_json_float(&mut line, "h1_4k_min", input.h1_4k_min);
    write_json_float(&mut line, "h1_4k_max", input.h1_4k_max);
    write_json_str(&mut line, "planned_kernel", input.planned_kernel);
    write_json_num(&mut line, "planned_chunk_bytes", input.planned_chunk_bytes);
    write_json_num(
        &mut line,
        "planned_sample_bytes",
        input.planned_sample_bytes,
    );
    write_json_str(&mut line, "plan_reason", input.plan_reason);
    line.push('}');
    line
}

fn workload_shape(
    payload: &Payload,
    access: &AccessPattern,
    processed_bytes: usize,
) -> WorkloadShape {
    WorkloadShape {
        context: match access {
            AccessPattern::WholeBlock => ApiContext::Block,
            AccessPattern::Sequential { .. } => ApiContext::Sequential,
            AccessPattern::Random { .. } => ApiContext::Random,
            AccessPattern::ParallelSequential { .. } => ApiContext::Parallel,
        },
        content: match payload.content {
            "text" => ContentKind::Text,
            "binary" => ContentKind::Binary,
            "mixed" => ContentKind::Mixed,
            _ => ContentKind::Unknown,
        },
        entropy: match payload.entropy {
            "low" => EntropyClass::Low,
            "medium" => EntropyClass::Medium,
            "high" => EntropyClass::High,
            "mixed" => EntropyClass::Mixed,
            _ => EntropyClass::Unknown,
        },
        scale: match payload.scale {
            "flat" => EntropyScale::Flat,
            "micro" => EntropyScale::Micro,
            "meso" => EntropyScale::Meso,
            "macro" => EntropyScale::Macro,
            _ => EntropyScale::Unknown,
        },
        total_bytes: processed_bytes,
        chunk_bytes: access.chunk_size(),
        threads: access.threads(),
    }
}

fn write_json_str(line: &mut String, key: &str, value: &str) {
    if line.len() > 1 {
        line.push(',');
    }
    let _ = write!(line, "\"{key}\":\"{}\"", json_escape(value));
}

fn write_json_num(line: &mut String, key: &str, value: usize) {
    if line.len() > 1 {
        line.push(',');
    }
    let _ = write!(line, "\"{key}\":{value}");
}

fn write_json_float(line: &mut String, key: &str, value: f64) {
    if line.len() > 1 {
        line.push(',');
    }
    let _ = write!(line, "\"{key}\":{value:.6}");
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch.is_control() => {
                let _ = write!(escaped, "\\u{:04x}", ch as u32);
            }
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn sanitize_id_fragment(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

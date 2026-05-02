#![allow(missing_docs)]

use std::{
    collections::BTreeSet,
    env,
    fmt::Write as _,
    fs::{self, File},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write as _},
    path::{Path, PathBuf},
    sync::Arc,
};

use serde_json::Value;
use tokenfs_algos::dispatch::{
    ApiContext, CacheState, ContentKind, EntropyClass, EntropyScale, ProcessorProfile,
    ReadPattern as PlannerReadPattern, SourceHint, WorkloadShape, plan_histogram,
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
    ReadAhead {
        chunk_size: usize,
    },
    Random {
        chunk_size: usize,
        offsets: Vec<usize>,
    },
    ZipfianHotCold {
        chunk_size: usize,
        offsets: Vec<usize>,
    },
    HotRepeat {
        chunk_size: usize,
        repeats: usize,
    },
    ColdSweep {
        chunk_size: usize,
    },
    SameFileRepeat {
        repeats: usize,
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
            Self::ReadAhead { .. } => "readahead",
            Self::Random { .. } => "random",
            Self::ZipfianHotCold { .. } => "zipfian-hot-cold",
            Self::HotRepeat { .. } => "hot-repeat",
            Self::ColdSweep { .. } => "cold-sweep",
            Self::SameFileRepeat { .. } => "same-file-repeat",
            Self::ParallelSequential { .. } => "parallel-sequential",
        }
    }

    fn chunk_size(&self) -> usize {
        match self {
            Self::WholeBlock => 0,
            Self::Sequential { chunk_size }
            | Self::ReadAhead { chunk_size }
            | Self::Random { chunk_size, .. }
            | Self::ZipfianHotCold { chunk_size, .. }
            | Self::HotRepeat { chunk_size, .. }
            | Self::ColdSweep { chunk_size }
            | Self::ParallelSequential { chunk_size, .. } => *chunk_size,
            Self::SameFileRepeat { .. } => 0,
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
            Self::ReadAhead { .. } | Self::ColdSweep { .. } => buffer_bytes,
            Self::Random {
                chunk_size,
                offsets,
            }
            | Self::ZipfianHotCold {
                chunk_size,
                offsets,
            } => chunk_size.saturating_mul(offsets.len()),
            Self::HotRepeat { repeats, .. } | Self::SameFileRepeat { repeats } => {
                buffer_bytes.saturating_mul(*repeats)
            }
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
    bytes: Arc<[u8]>,
    byte_offset: usize,
    byte_len: usize,
    pub(crate) access: AccessPattern,
    pub(crate) processed_bytes: usize,
    h1_bits_per_byte: f64,
    h1_4k_mean: f64,
    h1_4k_min: f64,
    h1_4k_max: f64,
    planned_kernel: &'static str,
    planned_chunk_bytes: usize,
    planned_sample_bytes: usize,
    planned_confidence_q8: u8,
    planned_confidence_source: &'static str,
    plan_reason: &'static str,
}

struct Payload {
    label: String,
    source: &'static str,
    content: &'static str,
    entropy: &'static str,
    scale: &'static str,
    pattern: &'static str,
    bytes: Arc<[u8]>,
    byte_offset: usize,
    byte_len: usize,
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
            bytes: Arc::from(bytes),
            byte_offset: 0,
            byte_len: 0,
        }
        .normalize_len()
    }

    fn normalize_len(mut self) -> Self {
        self.byte_len = self.bytes.len();
        self
    }

    fn bytes(&self) -> &[u8] {
        &self.bytes[self.byte_offset..self.byte_offset + self.byte_len]
    }

    fn len(&self) -> usize {
        self.byte_len
    }
}

impl WorkloadInput {
    fn new(payload: &Payload, access: AccessPattern, profile: &ProcessorProfile) -> Self {
        let processed_bytes = access.processed_bytes(payload.len());
        let bytes = payload.bytes();
        let h1_bits_per_byte = shannon_entropy(bytes);
        let (h1_4k_mean, h1_4k_min, h1_4k_max) = chunk_entropy_stats(bytes, 4 * 1024);
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
            byte_offset: payload.byte_offset,
            byte_len: payload.byte_len,
            access,
            processed_bytes,
            h1_bits_per_byte,
            h1_4k_mean,
            h1_4k_min,
            h1_4k_max,
            planned_kernel: plan.strategy.as_str(),
            planned_chunk_bytes: plan.chunk_bytes,
            planned_sample_bytes: plan.sample_bytes,
            planned_confidence_q8: plan.confidence_q8,
            planned_confidence_source: plan.confidence_source.as_str(),
            plan_reason: plan.reason,
        }
    }

    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes[self.byte_offset..self.byte_offset + self.byte_len]
    }

    fn buffer_bytes(&self) -> usize {
        self.byte_len
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
    let suite = WorkloadSuite::from_env(full);
    let only_magic_bpe = env::var_os("TOKENFS_ALGOS_ONLY_MAGIC_BPE").is_some();
    let mut payloads = if only_magic_bpe {
        Vec::new()
    } else {
        synthetic_workload_payloads(base_size)
    };
    let profile = ProcessorProfile::detect();

    if suite.mixed_regions {
        payloads.extend(mixed_region_payloads(base_size));
    }
    if suite.motifs {
        payloads.extend(motif_payloads(base_size));
    }
    if suite.alignment {
        payloads.extend(alignment_payloads(base_size.min(1024 * 1024)));
    }
    if suite.size {
        payloads.extend(size_sweep_payloads());
    }

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
    for path in real_paths_from_env() {
        match real_workload_payloads_from_path(&path, base_size) {
            Ok(real_payloads) => payloads.extend(real_payloads),
            Err(error) => {
                eprintln!(
                    "skipping real workload matrix payloads for `{}`: {error}",
                    path.display()
                );
            }
        }
    }
    for path in paper_paths_from_env(suite.paper) {
        match paper_workload_payloads_from_path(&path, base_size.min(1024 * 1024)) {
            Ok(real_payloads) => payloads.extend(real_payloads),
            Err(error) => {
                eprintln!(
                    "skipping paper-data workload matrix payloads for `{}`: {error}",
                    path.display()
                );
            }
        }
    }
    if let Some(path) = env::var_os("TOKENFS_ALGOS_MAGIC_BPE_DATA") {
        match magic_bpe_payloads_from_root(Path::new(&path)) {
            Ok(real_payloads) => payloads.extend(real_payloads),
            Err(error) => {
                eprintln!(
                    "skipping Magic-BPE workload matrix payloads for `{}`: {error}",
                    PathBuf::from(path).display()
                );
            }
        }
    }

    filter_payloads_from_env(&mut payloads);
    let access_filter = env_tokens("TOKENFS_ALGOS_WORKLOAD_ACCESS");

    let mut inputs = Vec::new();
    for payload in &payloads {
        for access in workload_access_patterns(payload.len(), full, threaded, &suite) {
            if !access_matches_filter(&access, access_filter.as_deref()) {
                continue;
            }
            if access.processed_bytes(payload.len()) == 0 {
                continue;
            }
            inputs.push(WorkloadInput::new(payload, access, &profile));
        }
    }

    if let Some(max_inputs) = env_usize("TOKENFS_ALGOS_WORKLOAD_MAX_INPUTS") {
        inputs.truncate(max_inputs);
    }

    inputs
}

fn filter_payloads_from_env(payloads: &mut Vec<Payload>) {
    if let Some(tokens) = env_tokens("TOKENFS_ALGOS_WORKLOAD_CASES") {
        payloads.retain(|payload| {
            matches_any_token(&payload.label, &tokens)
                || matches_any_token(payload.source, &tokens)
                || matches_any_token(payload.content, &tokens)
                || matches_any_token(payload.entropy, &tokens)
                || matches_any_token(payload.scale, &tokens)
                || matches_any_token(payload.pattern, &tokens)
        });
    }

    if let Some(max_payloads) = env_usize("TOKENFS_ALGOS_WORKLOAD_MAX_PAYLOADS") {
        payloads.truncate(max_payloads);
    }
}

#[derive(Clone, Copy)]
struct WorkloadSuite {
    size: bool,
    alignment: bool,
    mixed_regions: bool,
    motifs: bool,
    access: bool,
    cache: bool,
    paper: bool,
}

impl WorkloadSuite {
    fn from_env(full: bool) -> Self {
        let suite = env::var("TOKENFS_ALGOS_WORKLOAD_SUITE").unwrap_or_default();
        let has = |needle: &str| {
            suite
                .split([',', ';', ':', ' '])
                .any(|token| token == needle)
        };
        let all = has("all");
        let synthetic_full = has("synthetic-full");
        Self {
            size: all
                || synthetic_full
                || has("size")
                || env::var_os("TOKENFS_ALGOS_SIZE_SWEEP").is_some(),
            alignment: all
                || synthetic_full
                || has("alignment")
                || env::var_os("TOKENFS_ALGOS_ALIGNMENT_SWEEP").is_some(),
            mixed_regions: full
                || all
                || synthetic_full
                || has("mixed")
                || has("mixed-regions")
                || env::var_os("TOKENFS_ALGOS_MIXED_REGION_SWEEP").is_some(),
            motifs: full
                || all
                || synthetic_full
                || has("motif")
                || env::var_os("TOKENFS_ALGOS_MOTIF_SWEEP").is_some(),
            access: full
                || all
                || synthetic_full
                || has("access")
                || env::var_os("TOKENFS_ALGOS_ACCESS_SWEEP").is_some(),
            cache: all || has("cache") || env::var_os("TOKENFS_ALGOS_CACHE_SWEEP").is_some(),
            paper: all
                || has("paper")
                || has("real-f21")
                || env::var_os("TOKENFS_ALGOS_PAPER_DATA").is_some()
                || env::var_os("TOKENFS_ALGOS_F21_DATA").is_some()
                || env::var_os("TOKENFS_ALGOS_F22_DATA").is_some(),
        }
    }
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
        Payload::new(
            "json-lines",
            "synthetic",
            "text",
            "medium",
            "micro",
            "json-lines",
            json_lines(size),
        ),
        Payload::new(
            "csv-records",
            "synthetic",
            "text",
            "medium",
            "micro",
            "csv-records",
            csv_records(size),
        ),
        Payload::new(
            "source-like",
            "synthetic",
            "text",
            "medium",
            "micro",
            "source-like",
            source_like(size),
        ),
        Payload::new(
            "sqlite-like-pages",
            "synthetic",
            "binary",
            "mixed",
            "meso",
            "sqlite-like-pages",
            sqlite_like_pages(size),
        ),
        Payload::new(
            "compressed-like",
            "synthetic",
            "binary",
            "high",
            "flat",
            "compressed-like",
            compressed_like(size),
        ),
    ]
}

fn mixed_region_payloads(size: usize) -> Vec<Payload> {
    [4 * 1024, 64 * 1024, 1024 * 1024]
        .into_iter()
        .filter(|region| *region <= size)
        .map(|region| {
            Payload::new(
                format!("mixed-regions-{region}"),
                "synthetic",
                "mixed",
                "mixed",
                if region >= 1024 * 1024 {
                    "macro"
                } else {
                    "meso"
                },
                "alternating-regions",
                alternating_regions(size, region),
            )
        })
        .collect()
}

fn motif_payloads(size: usize) -> Vec<Payload> {
    vec![
        Payload::new(
            "motif-short-8",
            "synthetic",
            "binary",
            "low",
            "micro",
            "short-motif",
            repeated_random_motif(size, 8),
        ),
        Payload::new(
            "motif-long-4096",
            "synthetic",
            "binary",
            "medium",
            "meso",
            "long-motif",
            repeated_random_motif(size, 4 * 1024),
        ),
        Payload::new(
            "periodic-byte-classes",
            "synthetic",
            "binary",
            "medium",
            "micro",
            "periodic-byte-classes",
            periodic_byte_classes(size),
        ),
    ]
}

fn alignment_payloads(size: usize) -> Vec<Payload> {
    let base = deterministic_prng(size, 0xA11_6A11);
    [0_usize, 1, 3, 7, 31]
        .into_iter()
        .map(|offset| {
            let mut bytes = vec![0xA5; offset];
            bytes.extend_from_slice(&base);
            let mut payload = Payload::new(
                format!("alignment-plus-{offset}"),
                "synthetic",
                "binary",
                "high",
                "flat",
                "alignment-sweep",
                bytes,
            );
            payload.byte_offset = offset;
            payload.byte_len = base.len();
            payload
        })
        .collect()
}

fn size_sweep_payloads() -> Vec<Payload> {
    let max_size = env::var("TOKENFS_ALGOS_MAX_SWEEP_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(256 * 1024 * 1024);
    let sizes = [
        64_usize,
        256,
        1024,
        4 * 1024,
        8 * 1024,
        16 * 1024,
        64 * 1024,
        1024 * 1024,
        16 * 1024 * 1024,
        256 * 1024 * 1024,
    ];
    let mut payloads = Vec::new();

    for size in sizes.into_iter().filter(|size| *size <= max_size) {
        payloads.push(Payload::new(
            format!("size-prng-{size}"),
            "synthetic",
            "binary",
            "high",
            "flat",
            "size-sweep-prng",
            deterministic_prng(size, size as u64 ^ 0x5151_5eed),
        ));
        payloads.push(Payload::new(
            format!("size-zeros-{size}"),
            "synthetic",
            "binary",
            "low",
            "flat",
            "size-sweep-zeros",
            vec![0; size],
        ));
        payloads.push(Payload::new(
            format!("size-text-{size}"),
            "synthetic",
            "text",
            "medium",
            "micro",
            "size-sweep-text",
            varied_text(size),
        ));
    }

    payloads
}

fn real_workload_payloads_from_path(path: &Path, size: usize) -> std::io::Result<Vec<Payload>> {
    if path.is_dir() {
        return real_workload_payloads_from_dir(path, size);
    }

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

fn paper_workload_payloads_from_path(path: &Path, size: usize) -> std::io::Result<Vec<Payload>> {
    let mut payloads = real_workload_payloads_from_path(path, size)?;
    for payload in &mut payloads {
        payload.source = "paper";
        payload.content = "binary";
        payload.entropy = "mixed";
        payload.scale = "macro";
        payload.pattern = "paper-extents";
    }
    Ok(payloads)
}

fn magic_bpe_payloads_from_root(root: &Path) -> std::io::Result<Vec<Payload>> {
    let index = root.join("processed-index.jsonl");
    let processed = root.join("processed");
    let file = File::open(&index)?;
    let limit = env_usize("TOKENFS_ALGOS_MAGIC_BPE_LIMIT").unwrap_or(64);
    let per_mime_limit = env_usize("TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT").unwrap_or(4);
    let seed = env_usize("TOKENFS_ALGOS_MAGIC_BPE_SEED").unwrap_or(0x51f1_5eed) as u64;
    let shuffle = env_bool("TOKENFS_ALGOS_MAGIC_BPE_SHUFFLE").unwrap_or(true);
    let mut candidates = Vec::new();

    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let sample_bytes = value
            .get("sample_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if sample_bytes == 0 {
            continue;
        }
        let Some(sample_relpath) = value.get("sample_relpath").and_then(Value::as_str) else {
            continue;
        };
        let mime = value
            .get("mime_type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let extension = value.get("extension").and_then(Value::as_str);
        let original_path = value.get("path").and_then(Value::as_str).unwrap_or("");
        let key = if shuffle {
            mix_u64(seed ^ index as u64 ^ stable_str_hash(mime) ^ stable_str_hash(original_path))
        } else {
            index as u64
        };
        candidates.push(MagicBpeCandidate {
            key,
            sample_relpath: sample_relpath.to_owned(),
            mime: mime.to_owned(),
            extension: extension.map(ToOwned::to_owned),
            original_path: original_path.to_owned(),
        });
    }

    candidates.sort_by_key(|candidate| candidate.key);

    let mut payloads = Vec::new();
    let mut per_mime = std::collections::BTreeMap::<String, usize>::new();
    for candidate in candidates {
        if limit != 0 && payloads.len() >= limit {
            break;
        }
        let count = per_mime.entry(candidate.mime.clone()).or_default();
        if per_mime_limit != 0 && *count >= per_mime_limit {
            continue;
        }

        let path = processed.join(&candidate.sample_relpath);
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        if bytes.is_empty() {
            continue;
        }

        let pattern = magic_bpe_pattern(&candidate.mime, candidate.extension.as_deref());
        let content = magic_bpe_content(&candidate.mime, candidate.extension.as_deref());
        let entropy = entropy_label(shannon_entropy(&bytes));
        let scale = scale_label(&bytes);
        let label = format!(
            "magic-{}-{}-{}",
            pattern,
            sanitize_id_fragment(&candidate.mime),
            payloads.len()
        );
        let mut payload = Payload::new(label, "magic-bpe", content, entropy, scale, pattern, bytes);
        let sample_bytes = env_usize("TOKENFS_ALGOS_MAGIC_BPE_SAMPLE_BYTES").unwrap_or(64 * 1024);
        if sample_bytes != 0 {
            payload.byte_len = payload.byte_len.min(sample_bytes);
        }
        let _ = &candidate.original_path;
        payloads.push(payload);
        *count += 1;
    }

    Ok(payloads)
}

struct MagicBpeCandidate {
    key: u64,
    sample_relpath: String,
    mime: String,
    extension: Option<String>,
    original_path: String,
}

fn real_workload_payloads_from_dir(path: &Path, size: usize) -> std::io::Result<Vec<Payload>> {
    let mut payloads = Vec::new();
    let mut files = Vec::new();
    let limit = env_usize("TOKENFS_ALGOS_REAL_DIR_LIMIT").unwrap_or(64);
    collect_representative_files(path, limit, &mut files)?;

    for file_path in files {
        payloads.extend(real_workload_payloads_from_path(
            &file_path,
            size.min(1024 * 1024),
        )?);
    }

    Ok(payloads)
}

fn collect_representative_files(
    path: &Path,
    limit: usize,
    out: &mut Vec<PathBuf>,
) -> std::io::Result<()> {
    if out.len() >= limit {
        return Ok(());
    }

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let child = entry.path();
        if child.is_dir() {
            let name = child
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("");
            if matches!(name, ".git" | "target" | "node_modules" | "__pycache__") {
                continue;
            }
            collect_representative_files(&child, limit, out)?;
        } else if is_representative_real_file(&child) {
            out.push(child);
        }

        if out.len() >= limit {
            break;
        }
    }

    Ok(())
}

fn is_representative_real_file(path: &Path) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return true;
    };
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "rs" | "c"
            | "h"
            | "cpp"
            | "hpp"
            | "py"
            | "txt"
            | "log"
            | "json"
            | "csv"
            | "sqlite"
            | "db"
            | "parquet"
            | "tar"
            | "gz"
            | "xz"
            | "zst"
            | "zip"
            | "so"
            | "a"
            | "o"
            | "bin"
            | "dict"
    )
}

fn real_paths_from_env() -> Vec<PathBuf> {
    env::var_os("TOKENFS_ALGOS_REAL_PATHS")
        .map(|paths| env::split_paths(&paths).collect())
        .unwrap_or_default()
}

fn paper_paths_from_env(include_defaults: bool) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for key in ["TOKENFS_ALGOS_F21_DATA", "TOKENFS_ALGOS_F22_DATA"] {
        if let Some(path) = env::var_os(key) {
            paths.push(PathBuf::from(path));
        }
    }
    if !include_defaults {
        return paths;
    }

    let paper_root = env::var_os("TOKENFS_ALGOS_PAPER_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../tokenfs-paper"));
    for relative in [
        "data/tokenizers-corpus-matched/rootfs-4096.json",
        "data/tokenizers-corpus-matched/rootfs-65536.json",
        "data/tokenizers-corpus-matched/rootfs-65536-latin1.json",
        "data/zstd-dicts/rootfs-natural-64k.dict",
        "data/zstd-dicts/rootfs-bpe-64k.dict",
    ] {
        let path = paper_root.join(relative);
        if path.exists() {
            paths.push(path);
        }
    }

    paths
}

fn workload_access_patterns(
    buffer_bytes: usize,
    full: bool,
    threaded: bool,
    suite: &WorkloadSuite,
) -> Vec<AccessPattern> {
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
            AccessPattern::ReadAhead {
                chunk_size: 128 * 1024,
            },
        ]);
    }

    if suite.access {
        patterns.extend([
            AccessPattern::ReadAhead {
                chunk_size: 64 * 1024,
            },
            AccessPattern::ReadAhead {
                chunk_size: 128 * 1024,
            },
            AccessPattern::ZipfianHotCold {
                chunk_size: 4 * 1024,
                offsets: zipfian_offsets(buffer_bytes, 4 * 1024, if full { 512 } else { 128 }),
            },
        ]);
    }

    if suite.cache {
        patterns.extend([
            AccessPattern::HotRepeat {
                chunk_size: 64 * 1024,
                repeats: 8,
            },
            AccessPattern::ColdSweep {
                chunk_size: 1024 * 1024,
            },
            AccessPattern::SameFileRepeat { repeats: 8 },
        ]);
    }

    if threaded {
        for threads in thread_sweep_counts(full) {
            patterns.push(AccessPattern::ParallelSequential {
                chunk_size: 64 * 1024,
                threads,
            });
        }
    }

    patterns
}

fn thread_sweep_counts(full: bool) -> Vec<usize> {
    let logical = std::thread::available_parallelism().map_or(1, usize::from);
    let physical = physical_core_count().unwrap_or(logical).clamp(1, logical);
    let saturated = logical.saturating_mul(2).max(logical);
    let value = env::var("TOKENFS_ALGOS_THREAD_SWEEP").unwrap_or_default();
    let tokens = value
        .split([',', ';', ':', ' '])
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let mut counts = BTreeSet::new();

    if tokens.is_empty() {
        if full {
            insert_full_thread_sweep(&mut counts, physical, logical, saturated);
        } else {
            insert_quick_thread_sweep(&mut counts);
        }
    }

    for token in tokens {
        match token.to_ascii_lowercase().as_str() {
            "1" | "true" | "quick" | "basic" => insert_quick_thread_sweep(&mut counts),
            "full" | "all" | "topology" => {
                insert_full_thread_sweep(&mut counts, physical, logical, saturated);
            }
            "physical" | "core" | "cores" => {
                counts.insert(physical);
            }
            "logical" | "proc" | "processor" | "processors" | "available" => {
                counts.insert(logical);
            }
            "saturated" | "oversubscribe" | "oversubscribed" => {
                counts.insert(saturated);
            }
            numeric => {
                if let Ok(threads) = numeric.parse::<usize>()
                    && threads > 1
                {
                    counts.insert(threads);
                }
            }
        }
    }

    counts.into_iter().filter(|threads| *threads > 1).collect()
}

fn access_matches_filter(access: &AccessPattern, tokens: Option<&[String]>) -> bool {
    let Some(tokens) = tokens else {
        return true;
    };
    if tokens.is_empty() {
        return true;
    }

    let access_name = access.name();
    if matches_any_token(access_name, tokens) {
        return true;
    }

    let chunk_key = format!("{access_name}-{}", access.chunk_size());
    if matches_any_token(&chunk_key, tokens) {
        return true;
    }

    let threads = access.threads();
    threads > 1
        && (matches_any_token("parallel", tokens)
            || matches_any_token("threaded", tokens)
            || matches_any_token(&format!("{access_name}-{threads}"), tokens)
            || matches_any_token(&format!("threads-{threads}"), tokens))
}

fn insert_quick_thread_sweep(counts: &mut BTreeSet<usize>) {
    counts.insert(2);
    counts.insert(4);
}

fn insert_full_thread_sweep(
    counts: &mut BTreeSet<usize>,
    physical: usize,
    logical: usize,
    saturated: usize,
) {
    insert_quick_thread_sweep(counts);
    counts.insert(physical);
    counts.insert(logical);
    counts.insert(saturated);
}

fn physical_core_count() -> Option<usize> {
    let mut cores = BTreeSet::new();
    let entries = fs::read_dir("/sys/devices/system/cpu").ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(cpu_id) = name.strip_prefix("cpu") else {
            continue;
        };
        if cpu_id.is_empty() || !cpu_id.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }

        let topology = entry.path().join("topology");
        let package = fs::read_to_string(topology.join("physical_package_id"))
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let core = fs::read_to_string(topology.join("core_id"))
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or_else(|| cpu_id.parse::<usize>().unwrap_or(0));
        cores.insert((package, core));
    }

    Some(cores.len()).filter(|count| *count > 0)
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

fn alternating_regions(size: usize, region_size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let mut region = 0_usize;

    while out.len() < size {
        let take = (size - out.len()).min(region_size);
        match region % 4 {
            0 => out.extend(std::iter::repeat_n(0, take)),
            1 => out.extend(deterministic_prng(take, region as u64 ^ 0xA17E_A17E)),
            2 => out.extend(varied_text(take)),
            _ => out.extend(repeated_random_motif(take, 256)),
        }
        region += 1;
    }

    out
}

fn periodic_byte_classes(size: usize) -> Vec<u8> {
    const CLASSES: &[&[u8]] = &[
        b"ABCDEF0123456789",
        b"\n\r\t    ",
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[0x80, 0x91, 0xfe, 0xff],
    ];
    let mut out = Vec::with_capacity(size);
    let mut index = 0_usize;
    while out.len() < size {
        let class = CLASSES[index % CLASSES.len()];
        out.push(class[index % class.len()]);
        index += 1;
    }
    out
}

fn json_lines(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let mut id = 0_u64;
    while out.len() < size {
        let line = format!(
            "{{\"ts\":{},\"level\":\"info\",\"path\":\"/usr/lib/tokenfs/{:08x}\",\"bytes\":{},\"entropy\":\"medium\"}}\n",
            1_777_000_000_u64 + id,
            id,
            4096 + (id % 65536)
        );
        append_truncated(&mut out, line.as_bytes(), size);
        id += 1;
    }
    out
}

fn csv_records(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    append_truncated(&mut out, b"inode,extent,offset,length,policy,h1,h4\n", size);
    let mut id = 0_u64;
    while out.len() < size {
        let line = format!(
            "{},{},{},{},policy-{},{:.3},{:.3}\n",
            id % 65_537,
            id,
            id * 4096,
            4096 + (id % 16) * 1024,
            id % 5,
            (id % 8000) as f64 / 1000.0,
            (id % 7900) as f64 / 1000.0
        );
        append_truncated(&mut out, line.as_bytes(), size);
        id += 1;
    }
    out
}

fn source_like(size: usize) -> Vec<u8> {
    const LINES: &[&[u8]] = &[
        b"pub fn add_block(bytes: &[u8], counts: &mut [u64; 256]) {\n",
        b"    for &byte in bytes { counts[byte as usize] += 1; }\n",
        b"}\n\n",
        b"#[inline(always)]\n",
        b"let entropy = histogram.counts().iter().filter(|c| **c != 0);\n",
    ];
    repeat_lines(size, LINES)
}

fn sqlite_like_pages(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let mut page = 0_usize;
    while out.len() < size {
        let remaining = size - out.len();
        let take = remaining.min(4096);
        let mut bytes = vec![0; take];
        if take >= 16 {
            bytes[..16].copy_from_slice(b"SQLite format 3\0");
        }
        for (i, byte) in bytes.iter_mut().enumerate().skip(16) {
            *byte = ((page * 131 + i * 17) & 0xff) as u8;
        }
        out.extend(bytes);
        page += 1;
    }
    out
}

fn compressed_like(size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    append_truncated(&mut out, b"\x28\xb5\x2f\xfd", size);
    out.extend(deterministic_prng(
        size.saturating_sub(out.len()),
        0xC0DE_C0DE,
    ));
    out
}

fn repeat_lines(size: usize, lines: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(size);
    let mut index = 0_usize;
    while out.len() < size {
        append_truncated(&mut out, lines[index % lines.len()], size);
        index += 1;
    }
    out
}

fn append_truncated(out: &mut Vec<u8>, bytes: &[u8], target_len: usize) {
    let take = (target_len - out.len()).min(bytes.len());
    out.extend_from_slice(&bytes[..take]);
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

fn zipfian_offsets(buffer_bytes: usize, chunk_size: usize, count: usize) -> Vec<usize> {
    if chunk_size == 0 || buffer_bytes < chunk_size {
        return Vec::new();
    }

    let hot_span = buffer_bytes.min(1024 * 1024).max(chunk_size);
    let mut offsets = Vec::with_capacity(count);
    let mut state = 0x51f1_5eed_u64;
    for index in 0..count {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let hot = index % 5 != 0;
        let span = if hot { hot_span } else { buffer_bytes };
        let max_start = span - chunk_size;
        let slots = (max_start / chunk_size) + 1;
        let slot = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) as usize) % slots;
        offsets.push(slot * chunk_size);
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
    write_json_num(&mut line, "buffer_bytes", input.buffer_bytes());
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
    write_json_num(
        &mut line,
        "planned_confidence_q8",
        usize::from(input.planned_confidence_q8),
    );
    write_json_str(
        &mut line,
        "planned_confidence_source",
        input.planned_confidence_source,
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
            AccessPattern::Sequential { .. }
            | AccessPattern::ReadAhead { .. }
            | AccessPattern::HotRepeat { .. }
            | AccessPattern::ColdSweep { .. }
            | AccessPattern::SameFileRepeat { .. } => ApiContext::Sequential,
            AccessPattern::Random { .. } | AccessPattern::ZipfianHotCold { .. } => {
                ApiContext::Random
            }
            AccessPattern::ParallelSequential { .. } => ApiContext::Parallel,
        },
        read_pattern: match access {
            AccessPattern::WholeBlock => PlannerReadPattern::WholeBlock,
            AccessPattern::Sequential { .. } => PlannerReadPattern::Sequential,
            AccessPattern::ReadAhead { .. } => PlannerReadPattern::Readahead,
            AccessPattern::Random { chunk_size, .. } if *chunk_size <= 1 => {
                PlannerReadPattern::RandomTiny
            }
            AccessPattern::Random { .. } => PlannerReadPattern::Random,
            AccessPattern::ZipfianHotCold { .. } => PlannerReadPattern::ZipfianHotCold,
            AccessPattern::HotRepeat { .. } => PlannerReadPattern::HotRepeat,
            AccessPattern::ColdSweep { .. } => PlannerReadPattern::ColdSweep,
            AccessPattern::SameFileRepeat { .. } => PlannerReadPattern::SameFileRepeat,
            AccessPattern::ParallelSequential { .. } => PlannerReadPattern::ParallelSequential,
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
        alignment_offset: payload.byte_offset,
        cache_state: match access {
            AccessPattern::HotRepeat { .. } => CacheState::Hot,
            AccessPattern::ColdSweep { .. } => CacheState::Cold,
            AccessPattern::SameFileRepeat { .. } => CacheState::Reused,
            _ => CacheState::Unknown,
        },
        source_hint: match payload.source {
            "synthetic" => SourceHint::Synthetic,
            "real" | "magic-bpe" => SourceHint::RealFile,
            "paper" | "f21" | "f22" => SourceHint::PaperExtent,
            _ => SourceHint::Unknown,
        },
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

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|value| value.parse().ok())
}

fn env_tokens(name: &str) -> Option<Vec<String>> {
    env::var(name).ok().map(|value| {
        value
            .split([',', ';', ':', ' '])
            .map(str::trim)
            .filter(|token| !token.is_empty())
            .map(|token| token.to_ascii_lowercase())
            .collect()
    })
}

fn matches_any_token(value: &str, tokens: &[String]) -> bool {
    let value = value.to_ascii_lowercase();
    tokens
        .iter()
        .any(|token| token == "*" || value == *token || value.contains(token))
}

fn env_bool(name: &str) -> Option<bool> {
    env::var(name).ok().map(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "shuffle"
        )
    })
}

fn magic_bpe_content(mime: &str, extension: Option<&str>) -> &'static str {
    let mime = mime.to_ascii_lowercase();
    let extension = extension.unwrap_or("").to_ascii_lowercase();
    if mime.starts_with("text/")
        || mime.contains("json")
        || mime.contains("xml")
        || mime.contains("javascript")
        || matches!(
            extension.as_str(),
            "rs" | "c"
                | "h"
                | "cpp"
                | "hpp"
                | "py"
                | "js"
                | "ts"
                | "json"
                | "csv"
                | "html"
                | "svg"
                | "md"
                | "toml"
        )
    {
        "text"
    } else {
        "binary"
    }
}

fn magic_bpe_pattern(mime: &str, extension: Option<&str>) -> &'static str {
    let mime = mime.to_ascii_lowercase();
    let extension = extension.unwrap_or("").to_ascii_lowercase();

    if mime.contains("gzip")
        || mime.contains("zstd")
        || mime.contains("zlib")
        || mime.contains("zip")
        || mime.contains("7z")
        || matches!(
            extension.as_str(),
            "gz" | "zst" | "zip" | "7z" | "xz" | "bz2"
        )
    {
        "archive-compressed"
    } else if mime.starts_with("image/") {
        "image"
    } else if mime.starts_with("audio/") {
        "audio"
    } else if mime.contains("font")
        || matches!(extension.as_str(), "ttf" | "otf" | "woff" | "woff2" | "tfm")
    {
        "font"
    } else if mime.contains("sqlite") || matches!(extension.as_str(), "sqlite" | "db") {
        "database"
    } else if mime.contains("executable")
        || mime.contains("sharedlib")
        || matches!(extension.as_str(), "exe" | "so" | "a" | "o")
    {
        "executable-library"
    } else if mime.starts_with("text/")
        || mime.contains("json")
        || mime.contains("javascript")
        || matches!(
            extension.as_str(),
            "rs" | "c"
                | "h"
                | "cpp"
                | "hpp"
                | "py"
                | "js"
                | "ts"
                | "json"
                | "csv"
                | "html"
                | "svg"
                | "md"
                | "toml"
        )
    {
        "text-source"
    } else if mime.contains("powerpoint")
        || mime.contains("excel")
        || mime.contains("ole")
        || matches!(
            extension.as_str(),
            "ppt" | "pptx" | "xls" | "xlsx" | "doc" | "docx"
        )
    {
        "document"
    } else {
        "binary-other"
    }
}

fn entropy_label(h1: f64) -> &'static str {
    if h1 < 2.5 {
        "low"
    } else if h1 >= 7.25 {
        "high"
    } else {
        "medium"
    }
}

fn scale_label(bytes: &[u8]) -> &'static str {
    if bytes.len() < 8 * 1024 {
        return "micro";
    }
    let (_, min, max) = chunk_entropy_stats(bytes, 4 * 1024);
    if max - min >= 3.0 { "meso" } else { "flat" }
}

fn stable_str_hash(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in value.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Cache-tier reporting axis for v0.2 benches per
/// `docs/v0.2_planning/02_CACHE_RESIDENCY.md` § "What this means for benchmarking".
///
/// Returns the canonical (label, size_bytes) pairs to bench at: in-L1
/// (4 KB), in-L2 (256 KB), in-L3 (8 MB), in-DRAM (64 MB). Per-tier
/// sizes chosen to fit in a "typical" P-core's cache hierarchy with
/// margin (modern x86 L1d 48-64 KB, L2 1-4 MB, L3 16-128 MB).
///
/// Use as the third bench axis after (kernel, input shape):
///
/// ```ignore
/// for (tier_label, size) in cache_tier_sizes() {
///     let input = vec![0_u8; size];
///     group.throughput(Throughput::Bytes(size as u64));
///     group.bench_with_input(
///         BenchmarkId::new(tier_label, size),
///         &input,
///         |b, bytes| b.iter(|| kernel(black_box(bytes))),
///     );
/// }
/// ```
#[allow(dead_code)]
pub(crate) fn cache_tier_sizes() -> &'static [(&'static str, usize)] {
    &[
        ("in-L1", 4 * 1024),           // 4 KB — fits in L1d
        ("in-L2", 256 * 1024),         // 256 KB — exceeds L1, fits in L2
        ("in-L3", 8 * 1024 * 1024),    // 8 MB — exceeds L2, fits in L3
        ("in-DRAM", 64 * 1024 * 1024), // 64 MB — exceeds L3, DRAM-bound
    ]
}

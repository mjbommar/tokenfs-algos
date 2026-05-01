#![allow(missing_docs)]

use std::{
    collections::HashMap,
    env,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    time::Instant,
};

use tokenfs_algos::{entropy, fingerprint, histogram::ByteHistogram, paper};

const F21_H1_MAX_ABS_DIFF: f32 = 0.05;
const F21_H2_MAX_ABS_DIFF: f32 = 0.08;
const F21_H3_MAX_ABS_DIFF: f32 = 0.10;
const F21_TOP16_MAX_ABS_DIFF: f32 = 0.02;
const F21_RL_MAX_ABS_DIFF: f32 = 0.002;
const F21_SKEW_MAX_ABS_DIFF: f32 = 0.01;
const F21_ANALYSIS_MIN_SELECTOR_ACCURACY: f64 = 0.91;
const F21_ANALYSIS_MIN_TOP2_ACCURACY: f64 = 0.96;
const F22_RELEASE_BLOCK_MAX_NS: f64 = 1_800.0;
const F22_RELEASE_EXTENT_MIN_GIB_S: f64 = 0.93;
const F22_DEBUG_BLOCK_MAX_NS: f64 = 100_000.0;
const F22_DEBUG_EXTENT_MIN_GIB_S: f64 = 0.01;

fn deterministic_block(seed: u64) -> [u8; fingerprint::BLOCK_SIZE] {
    let mut state = seed;
    let mut block = [0_u8; fingerprint::BLOCK_SIZE];
    for byte in &mut block {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *byte = state.wrapping_mul(0x2545_f491_4f6c_dd1d) as u8;
    }
    block
}

#[test]
fn public_f22_aliases_match_product_api() {
    let block = deterministic_block(0xF22);

    assert_eq!(
        fingerprint::block(&block),
        paper::f22::fingerprint_block(&block)
    );
    assert_eq!(
        fingerprint::kernels::scalar::block(&block),
        paper::f22::scalar::fingerprint_block(&block)
    );
    assert_eq!(
        fingerprint::extent(&block),
        paper::f22::fingerprint_extent(&block)
    );
}

#[test]
fn pinned_scalar_matches_default_for_generated_blocks() {
    for seed in 0..128_u64 {
        let block = deterministic_block(seed ^ 0x0A5A_5F22);
        assert_eq!(
            fingerprint::block(&block),
            fingerprint::kernels::scalar::block(&block),
            "seed={seed}"
        );
    }
}

#[test]
fn calibration_matches_f21_sidecar_h1_when_available() {
    let Some(path) = calibration_path_or_skip("F22 sidecar", f22_sidecar_path()) else {
        return;
    };

    let rows = read_sidecar_rows(&path, calibration_extent_limit(500));
    let mut total_diff = 0.0_f64;
    let mut max_diff = 0.0_f32;

    for row in &rows {
        let fp = fingerprint::kernels::scalar::extent(&row.payload);
        let diff = (fp.h1 - row.features.h1).abs();
        total_diff += f64::from(diff);
        max_diff = max_diff.max(diff);
        assert!(
            diff < F21_H1_MAX_ABS_DIFF,
            "extent {}: tokenfs-algos H1={} vs F21 H1={} diff={diff}",
            row.extent_id,
            fp.h1,
            row.features.h1
        );
    }

    eprintln!(
        "F22 sidecar H1 calibration: extents={}, avg_diff={:.6}, max_diff={:.6}, threshold={F21_H1_MAX_ABS_DIFF}",
        rows.len(),
        total_diff / rows.len().max(1) as f64,
        max_diff
    );
}

#[test]
fn calibration_matches_f21_feature_vector_when_available() {
    let Some(path) = calibration_path_or_skip("F22 sidecar", f22_sidecar_path()) else {
        return;
    };

    let rows = read_sidecar_rows(&path, calibration_extent_limit(128));
    let mut max = F21Features::default();
    let mut sum = F21Features::default();

    for row in &rows {
        let got = f21_compatible_features(&row.payload);
        let diff = got.abs_diff(row.features);
        max = max.max(diff);
        sum = sum + diff;

        assert!(
            diff.h1 <= F21_H1_MAX_ABS_DIFF,
            "{} H1 drift: got={} expected={} diff={}",
            row.extent_id,
            got.h1,
            row.features.h1,
            diff.h1
        );
        assert!(
            diff.h2 <= F21_H2_MAX_ABS_DIFF,
            "{} H2 drift: got={} expected={} diff={}",
            row.extent_id,
            got.h2,
            row.features.h2,
            diff.h2
        );
        assert!(
            diff.h3 <= F21_H3_MAX_ABS_DIFF,
            "{} H3 drift: got={} expected={} diff={}",
            row.extent_id,
            got.h3,
            row.features.h3,
            diff.h3
        );
        assert!(
            diff.top16_coverage <= F21_TOP16_MAX_ABS_DIFF,
            "{} top16 drift: got={} expected={} diff={}",
            row.extent_id,
            got.top16_coverage,
            row.features.top16_coverage,
            diff.top16_coverage
        );
        assert!(
            diff.rl_fraction <= F21_RL_MAX_ABS_DIFF,
            "{} run-length drift: got={} expected={} diff={}",
            row.extent_id,
            got.rl_fraction,
            row.features.rl_fraction,
            diff.rl_fraction
        );
        assert!(
            diff.byte_entropy_skew <= F21_SKEW_MAX_ABS_DIFF,
            "{} skew drift: got={} expected={} diff={}",
            row.extent_id,
            got.byte_entropy_skew,
            row.features.byte_entropy_skew,
            diff.byte_entropy_skew
        );
    }

    let avg = sum.scale(1.0 / rows.len().max(1) as f32);
    eprintln!(
        "F21 feature calibration: extents={}, avg={avg:?}, max={max:?}",
        rows.len()
    );
}

#[test]
fn calibration_selector_analysis_gate_when_available() {
    let Some(path) = calibration_path_or_skip("F21 analysis", f21_analysis_path()) else {
        return;
    };
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read `{}`: {error}", path.display()));
    let value: serde_json::Value = serde_json::from_str(&json)
        .unwrap_or_else(|error| panic!("failed to parse `{}`: {error}", path.display()));
    let selector = &value["selector"];
    let test_accuracy = selector["test_accuracy"]
        .as_f64()
        .expect("selector.test_accuracy must be present");
    let top2_accuracy = selector["test_top2_accuracy"]
        .as_f64()
        .expect("selector.test_top2_accuracy must be present");
    let n_extents_used = value["n_extents_used"]
        .as_u64()
        .expect("n_extents_used must be present");

    assert!(
        test_accuracy
            >= calibration_f64(
                "TOKENFS_ALGOS_F21_SELECTOR_MIN_ACCURACY",
                F21_ANALYSIS_MIN_SELECTOR_ACCURACY,
            ),
        "F21 selector accuracy below gate: test_accuracy={test_accuracy:.4}"
    );
    assert!(
        top2_accuracy
            >= calibration_f64(
                "TOKENFS_ALGOS_F21_SELECTOR_MIN_TOP2_ACCURACY",
                F21_ANALYSIS_MIN_TOP2_ACCURACY,
            ),
        "F21 selector top-2 accuracy below gate: top2_accuracy={top2_accuracy:.4}"
    );
    assert!(
        n_extents_used >= calibration_usize("TOKENFS_ALGOS_F21_MIN_EXTENTS", 5_000) as u64,
        "F21 analysis used too few extents: n_extents_used={n_extents_used}"
    );

    eprintln!(
        "F21 selector gate: test_accuracy={test_accuracy:.4}, top2_accuracy={top2_accuracy:.4}, extents={n_extents_used}"
    );
}

#[test]
fn f22_fingerprint_throughput_gate_when_available() {
    if env::var_os("TOKENFS_ALGOS_SKIP_THROUGHPUT_GATE").is_some() {
        eprintln!(
            "TOKENFS_ALGOS_SKIP_THROUGHPUT_GATE set; skipping (intended for \
             QEMU/emulated cross-test runs where wall-clock timings are not \
             representative of native execution)"
        );
        return;
    }
    let Some(path) = calibration_path_or_skip("F22 sidecar", f22_sidecar_path()) else {
        return;
    };
    let rows = read_sidecar_rows(&path, calibration_extent_limit(128));
    let mut bytes = Vec::new();
    for row in &rows {
        bytes.extend_from_slice(&row.payload);
        if bytes.len() >= calibration_usize("TOKENFS_ALGOS_F22_THROUGHPUT_BYTES", 16 * 1024 * 1024)
        {
            break;
        }
    }
    let block_len = bytes.len() / fingerprint::BLOCK_SIZE * fingerprint::BLOCK_SIZE;
    assert!(
        block_len >= fingerprint::BLOCK_SIZE,
        "F22 throughput gate needs at least one fingerprint block"
    );
    let block_bytes = &bytes[..block_len];
    let blocks = block_len / fingerprint::BLOCK_SIZE;
    let block_repeats = calibration_usize("TOKENFS_ALGOS_F22_BLOCK_REPEATS", 8);
    let extent_repeats = calibration_usize("TOKENFS_ALGOS_F22_EXTENT_REPEATS", 4);

    let start = Instant::now();
    let mut acc = 0_u64;
    for _ in 0..block_repeats {
        for chunk in block_bytes.chunks_exact(fingerprint::BLOCK_SIZE) {
            let block: &[u8; fingerprint::BLOCK_SIZE] = chunk
                .try_into()
                .expect("chunks_exact yields block-sized slices");
            let fp = std::hint::black_box(fingerprint::block(std::hint::black_box(block)));
            acc ^= u64::from(fp.h1_q4)
                ^ (u64::from(fp.h4_q4) << 8)
                ^ (u64::from(fp.rl_runs_ge4) << 16);
        }
    }
    std::hint::black_box(acc);
    let block_elapsed = start.elapsed();
    let ns_per_block =
        block_elapsed.as_secs_f64() * 1e9 / (blocks.saturating_mul(block_repeats)) as f64;

    let start = Instant::now();
    let mut extent_acc = 0_u64;
    for _ in 0..extent_repeats {
        let fp = std::hint::black_box(fingerprint::extent(std::hint::black_box(&bytes)));
        extent_acc ^= fp.h1.to_bits() as u64 ^ ((fp.h4.to_bits() as u64) << 1);
    }
    std::hint::black_box(extent_acc);
    let extent_elapsed = start.elapsed();
    let gib = (bytes.len().saturating_mul(extent_repeats)) as f64 / 1024.0_f64.powi(3);
    let extent_gib_s = gib / extent_elapsed.as_secs_f64().max(f64::MIN_POSITIVE);

    let block_threshold = if cfg!(debug_assertions) {
        calibration_f64("TOKENFS_ALGOS_F22_BLOCK_MAX_NS", F22_DEBUG_BLOCK_MAX_NS)
    } else {
        calibration_f64("TOKENFS_ALGOS_F22_BLOCK_MAX_NS", F22_RELEASE_BLOCK_MAX_NS)
    };
    let extent_threshold = if cfg!(debug_assertions) {
        calibration_f64(
            "TOKENFS_ALGOS_F22_EXTENT_MIN_GIB_S",
            F22_DEBUG_EXTENT_MIN_GIB_S,
        )
    } else {
        calibration_f64(
            "TOKENFS_ALGOS_F22_EXTENT_MIN_GIB_S",
            F22_RELEASE_EXTENT_MIN_GIB_S,
        )
    };

    assert!(
        ns_per_block <= block_threshold,
        "F22 block throughput below gate: ns_per_block={ns_per_block:.1}, threshold={block_threshold:.1}"
    );
    assert!(
        extent_gib_s >= extent_threshold,
        "F22 extent throughput below gate: GiB/s={extent_gib_s:.3}, threshold={extent_threshold:.3}"
    );

    eprintln!(
        "F22 throughput gate: bytes={}, blocks={}, ns_per_block={ns_per_block:.1}, extent_gib_s={extent_gib_s:.3}",
        bytes.len(),
        blocks
    );
}

fn f22_sidecar_path() -> Option<PathBuf> {
    env::var_os("TOKENFS_ALGOS_F22_DATA")
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| {
            let path = PathBuf::from("/nas4/data/tokenfs-ubuntu/bench/cow/f22-extent-bytes.bin");
            path.exists().then_some(path)
        })
}

fn f21_analysis_path() -> Option<PathBuf> {
    env::var_os("TOKENFS_ALGOS_F21_ANALYSIS")
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| {
            let path = PathBuf::from("/nas4/data/tokenfs-ubuntu/bench/cow/f21-analysis.json");
            path.exists().then_some(path)
        })
}

fn calibration_path_or_skip(label: &str, path: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(path) = path {
        return Some(path);
    }

    if cfg!(feature = "calibration") || env::var_os("TOKENFS_ALGOS_REQUIRE_CALIBRATION").is_some() {
        panic!("{label} missing; set TOKENFS_ALGOS_F22_DATA / TOKENFS_ALGOS_F21_ANALYSIS");
    }

    eprintln!(
        "{label} missing; enable the calibration feature and set paper data paths to hard-gate"
    );
    None
}

#[derive(Clone, Debug)]
struct SidecarRow {
    extent_id: String,
    _policy: String,
    features: F21Features,
    payload: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Default)]
struct F21Features {
    h1: f32,
    h2: f32,
    h3: f32,
    top16_coverage: f32,
    rl_fraction: f32,
    byte_entropy_skew: f32,
}

impl F21Features {
    fn abs_diff(self, other: Self) -> Self {
        Self {
            h1: (self.h1 - other.h1).abs(),
            h2: (self.h2 - other.h2).abs(),
            h3: (self.h3 - other.h3).abs(),
            top16_coverage: (self.top16_coverage - other.top16_coverage).abs(),
            rl_fraction: (self.rl_fraction - other.rl_fraction).abs(),
            byte_entropy_skew: (self.byte_entropy_skew - other.byte_entropy_skew).abs(),
        }
    }

    fn max(self, other: Self) -> Self {
        Self {
            h1: self.h1.max(other.h1),
            h2: self.h2.max(other.h2),
            h3: self.h3.max(other.h3),
            top16_coverage: self.top16_coverage.max(other.top16_coverage),
            rl_fraction: self.rl_fraction.max(other.rl_fraction),
            byte_entropy_skew: self.byte_entropy_skew.max(other.byte_entropy_skew),
        }
    }

    fn scale(self, factor: f32) -> Self {
        Self {
            h1: self.h1 * factor,
            h2: self.h2 * factor,
            h3: self.h3 * factor,
            top16_coverage: self.top16_coverage * factor,
            rl_fraction: self.rl_fraction * factor,
            byte_entropy_skew: self.byte_entropy_skew * factor,
        }
    }
}

impl std::ops::Add for F21Features {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            h1: self.h1 + rhs.h1,
            h2: self.h2 + rhs.h2,
            h3: self.h3 + rhs.h3,
            top16_coverage: self.top16_coverage + rhs.top16_coverage,
            rl_fraction: self.rl_fraction + rhs.rl_fraction,
            byte_entropy_skew: self.byte_entropy_skew + rhs.byte_entropy_skew,
        }
    }
}

fn read_sidecar_rows(path: &PathBuf, limit: usize) -> Vec<SidecarRow> {
    let file = File::open(path)
        .unwrap_or_else(|error| panic!("failed to open F22 sidecar `{}`: {error}", path.display()));
    let mut reader = BufReader::new(file);

    let mut magic = [0_u8; 8];
    reader
        .read_exact(&mut magic)
        .expect("failed to read F22 sidecar magic");
    assert_eq!(&magic, b"F22EXTV1");

    let n_extents = read_u32(&mut reader) as usize;
    let target = n_extents.min(limit);
    let mut rows = Vec::with_capacity(target);
    for _ in 0..target {
        let extent_id = nul_padded_string(read_n(&mut reader, 32));
        let policy = nul_padded_string(read_n(&mut reader, 16));
        let n_bytes = read_u32(&mut reader) as usize;
        let features = F21Features {
            h1: read_f32(&mut reader),
            h2: read_f32(&mut reader),
            h3: read_f32(&mut reader),
            top16_coverage: read_f32(&mut reader),
            rl_fraction: read_f32(&mut reader),
            byte_entropy_skew: read_f32(&mut reader),
        };
        let payload = read_n(&mut reader, n_bytes);
        rows.push(SidecarRow {
            extent_id,
            _policy: policy,
            features,
            payload,
        });
    }
    rows
}

fn f21_compatible_features(bytes: &[u8]) -> F21Features {
    if bytes.is_empty() {
        return F21Features::default();
    }

    let histogram = ByteHistogram::from_block(bytes);
    let h1 = entropy::shannon::h1(&histogram);
    let h2 = if bytes.len() >= 2 {
        entropy::joint::h2_pairs(bytes) * (bytes.len() - 1) as f32 / bytes.len() as f32
    } else {
        h1
    };
    let h3 = if bytes.len() >= 3 {
        entropy::ngram::h3(bytes) * (bytes.len() - 2) as f32 / bytes.len() as f32
    } else {
        h1
    };
    let top16_coverage = top16_4gram_coverage(bytes);
    let rl_fraction = run_length_fraction(bytes);

    F21Features {
        h1,
        h2,
        h3,
        top16_coverage,
        rl_fraction,
        byte_entropy_skew: h1 / 8.0,
    }
}

fn top16_4gram_coverage(bytes: &[u8]) -> f32 {
    if bytes.len() < 4 {
        return 0.0;
    }

    let mut counts = HashMap::<u32, u32>::new();
    for window in bytes.windows(4) {
        let word = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
        *counts.entry(word).or_default() += 1;
    }
    let mut counts = counts.into_values().collect::<Vec<_>>();
    counts.sort_unstable_by(|left, right| right.cmp(left));
    let top = counts.into_iter().take(16).map(u64::from).sum::<u64>();
    top as f32 * 4.0 / bytes.len() as f32
}

fn run_length_fraction(bytes: &[u8]) -> f32 {
    if bytes.len() < 4 {
        return 0.0;
    }

    let mut bytes_in_runs = 0_usize;
    let mut index = 0_usize;
    while index < bytes.len() {
        let byte = bytes[index];
        let start = index;
        index += 1;
        while index < bytes.len() && bytes[index] == byte {
            index += 1;
        }
        let len = index - start;
        if len >= 4 {
            bytes_in_runs += len;
        }
    }
    bytes_in_runs as f32 / bytes.len() as f32
}

fn nul_padded_string(mut bytes: Vec<u8>) -> String {
    let nul = bytes
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(bytes.len());
    bytes.truncate(nul);
    String::from_utf8(bytes).expect("sidecar strings must be UTF-8")
}

fn calibration_extent_limit(default: usize) -> usize {
    calibration_usize("TOKENFS_ALGOS_CALIBRATION_EXTENTS", default)
}

fn calibration_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn calibration_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn read_u32<R: Read>(reader: &mut R) -> u32 {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).expect("failed to read u32");
    u32::from_le_bytes(bytes)
}

fn read_f32<R: Read>(reader: &mut R) -> f32 {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).expect("failed to read f32");
    f32::from_le_bytes(bytes)
}

fn read_n<R: Read>(reader: &mut R, n: usize) -> Vec<u8> {
    let mut bytes = vec![0_u8; n];
    reader
        .read_exact(&mut bytes)
        .expect("failed to read byte payload");
    bytes
}

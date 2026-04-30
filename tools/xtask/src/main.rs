//! Workspace automation tasks.

use std::{
    cmp::Ordering,
    collections::HashMap,
    env,
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use serde_json::{Value, json};

type Result<T> = std::result::Result<T, String>;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("xtask: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let mut args = env::args_os();
    let _program = args.next();
    let task = args
        .next()
        .unwrap_or_else(|| OsString::from("help"))
        .to_string_lossy()
        .into_owned();
    let rest = args.collect::<Vec<_>>();

    match task.as_str() {
        "check" => check(),
        "test" => test(),
        "bench" => bench(&rest),
        "bench-adaptive" => bench_adaptive(&rest),
        "bench-kernels" => bench_kernels(&rest),
        "bench-real" => bench_real(&rest),
        "bench-adaptive-real" => bench_adaptive_real(&rest),
        "bench-kernels-real" => bench_kernels_real(&rest),
        "bench-adaptive-contexts-real" => bench_adaptive_contexts_real(&rest),
        "bench-workloads" => bench_workloads(&rest),
        "bench-workloads-real" => bench_workloads_real(&rest),
        "bench-workloads-adaptive" => bench_workloads_adaptive(&rest),
        "bench-workloads-adaptive-real" => bench_workloads_adaptive_real(&rest),
        "bench-calibrate" => bench_calibrate(&rest),
        "bench-compare" => bench_compare(&rest),
        "bench-log" => record_bench_history(None),
        "profile" => profile(&rest),
        "profile-real" => profile_real(&rest),
        "ci" => ci(),
        "help" | "-h" | "--help" => {
            help();
            Ok(())
        }
        unknown => Err(format!("unknown task `{unknown}`")),
    }
}

fn check() -> Result<()> {
    cargo(["fmt", "--all", "--check"])?;
    cargo([
        "clippy",
        "--workspace",
        "--all-targets",
        "--all-features",
        "--",
        "-D",
        "warnings",
    ])?;
    cargo(["doc", "--workspace", "--all-features", "--no-deps"])?;
    cargo([
        "check",
        "-p",
        "tokenfs-algos",
        "--no-default-features",
        "--features",
        "std",
    ])?;
    Ok(())
}

fn test() -> Result<()> {
    cargo(["test", "--workspace", "--all-targets"])?;
    cargo(["test", "-p", "tokenfs-algos", "--release", "--all-features"])?;
    Ok(())
}

fn bench(extra: &[OsString]) -> Result<()> {
    bench_with_env(extra, Vec::new())
}

fn bench_kernels(extra: &[OsString]) -> Result<()> {
    bench_with_env(
        extra,
        vec![("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into())],
    )
}

fn bench_adaptive(extra: &[OsString]) -> Result<()> {
    bench_with_env(
        extra,
        vec![("TOKENFS_ALGOS_ADAPTIVE_ONLY".into(), "1".into())],
    )
}

fn bench_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    bench_with_env(
        &extra,
        vec![("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string())],
    )
}

fn bench_adaptive_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    bench_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_ADAPTIVE_ONLY".into(), "1".into()),
        ],
    )
}

fn bench_kernels_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    bench_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
        ],
    )
}

fn bench_adaptive_contexts_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    bench_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_ADAPTIVE_ONLY".into(), "1".into()),
            ("TOKENFS_ALGOS_CONTEXT_SWEEP".into(), "1".into()),
        ],
    )
}

fn bench_workloads(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(extra, Vec::new())
}

fn bench_workloads_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    bench_workloads_with_env(
        &extra,
        vec![("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string())],
    )
}

fn bench_workloads_adaptive(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(
        extra,
        vec![("TOKENFS_ALGOS_ADAPTIVE_ONLY".into(), "1".into())],
    )
}

fn bench_workloads_adaptive_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    bench_workloads_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_ADAPTIVE_ONLY".into(), "1".into()),
        ],
    )
}

fn bench_calibrate(extra: &[OsString]) -> Result<()> {
    let default_args = [
        "--",
        "--sample-size",
        "10",
        "--warm-up-time",
        "0.03",
        "--measurement-time",
        "0.03",
        "workload_matrix/adaptive",
    ];
    let args = if extra.is_empty() {
        cargo_args(default_args)
    } else {
        extra.to_vec()
    };

    bench_workloads_with_env(
        &args,
        vec![
            ("TOKENFS_ALGOS_ADAPTIVE_ONLY".into(), "1".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "1".into()),
        ],
    )
}

fn bench_compare(args: &[OsString]) -> Result<()> {
    if args.len() != 2 {
        return Err("usage: cargo xtask bench-compare <old.jsonl> <new.jsonl>".into());
    }

    let old_path = PathBuf::from(args[0].clone());
    let new_path = PathBuf::from(args[1].clone());
    let old = read_bench_log_records(&old_path)?;
    let new = read_bench_log_records(&new_path)?;

    let mut deltas = Vec::new();
    for (key, old_record) in &old {
        let Some(new_record) = new.get(key) else {
            continue;
        };
        let change_pct = if old_record.gib_per_s == 0.0 {
            0.0
        } else {
            (new_record.gib_per_s - old_record.gib_per_s) / old_record.gib_per_s * 100.0
        };
        deltas.push(BenchDelta {
            label: old_record.label.clone(),
            old_gib_per_s: old_record.gib_per_s,
            new_gib_per_s: new_record.gib_per_s,
            old_mean_ns: old_record.mean_ns,
            new_mean_ns: new_record.mean_ns,
            change_pct,
        });
    }

    deltas.sort_by(|left, right| {
        right
            .change_pct
            .abs()
            .partial_cmp(&left.change_pct.abs())
            .unwrap_or(Ordering::Equal)
    });

    println!("# Benchmark Compare");
    println!();
    println!("- old: `{}`", old_path.display());
    println!("- new: `{}`", new_path.display());
    println!("- old_records: `{}`", old.len());
    println!("- new_records: `{}`", new.len());
    println!("- matched_records: `{}`", deltas.len());
    println!(
        "- unmatched_records: `{}`",
        old.len() + new.len() - (2 * deltas.len())
    );
    println!();

    if deltas.is_empty() {
        println!("No matching benchmark IDs found.");
        return Ok(());
    }

    println!("| Benchmark | Old GiB/s | New GiB/s | Change | Old Mean | New Mean |");
    println!("|---|---:|---:|---:|---:|---:|");
    for delta in deltas.iter().take(30) {
        println!(
            "| {} | {:.2} | {:.2} | {:+.1}% | {} | {} |",
            delta.label,
            delta.old_gib_per_s,
            delta.new_gib_per_s,
            delta.change_pct,
            format_duration_ns(delta.old_mean_ns),
            format_duration_ns(delta.new_mean_ns),
        );
    }

    Ok(())
}

fn bench_workloads_with_env(
    extra: &[OsString],
    mut env_vars: Vec<(OsString, OsString)>,
) -> Result<()> {
    let started = SystemTime::now()
        .checked_sub(Duration::from_secs(2))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let manifest_path = env::current_dir()
        .map_err(|error| format!("failed to resolve current directory: {error}"))?
        .join("target/tokenfs-algos/workload-manifest.jsonl");

    env_vars.push(("TOKENFS_ALGOS_WORKLOAD_MATRIX".into(), "1".into()));
    env_vars.push((
        "TOKENFS_ALGOS_WORKLOAD_MANIFEST".into(),
        manifest_path.into_os_string(),
    ));
    bench_with_env(extra, env_vars)?;
    record_bench_history_since(Some("workload_matrix"), Some(started))
}

fn bench_with_env(extra: &[OsString], env_vars: Vec<(OsString, OsString)>) -> Result<()> {
    let mut args = cargo_args([
        "bench",
        "-p",
        "tokenfs-algos",
        "--all-features",
        "--bench",
        "histogram",
    ]);
    args.extend(extra.iter().cloned());
    run_command_with_env("cargo", args, env_vars)
}

fn profile(extra: &[OsString]) -> Result<()> {
    profile_with_env(extra, Vec::new())
}

fn profile_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    profile_with_env(
        &extra,
        vec![("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string())],
    )
}

fn profile_with_env(extra: &[OsString], env_vars: Vec<(OsString, OsString)>) -> Result<()> {
    ensure_profile_dir()?;
    cargo([
        "bench",
        "-p",
        "tokenfs-algos",
        "--bench",
        "histogram",
        "--all-features",
        "--no-run",
    ])?;

    if command_exists("perf") {
        let mut args = vec![
            OsString::from("stat"),
            OsString::from("-d"),
            OsString::from("cargo"),
            OsString::from("bench"),
            OsString::from("-p"),
            OsString::from("tokenfs-algos"),
            OsString::from("--bench"),
            OsString::from("histogram"),
            OsString::from("--all-features"),
        ];
        args.extend(extra.iter().cloned());
        run_command_with_env("perf", args, env_vars)?;
    } else {
        eprintln!("xtask: `perf` not found; running Criterion benchmark only");
        bench_with_env(extra, env_vars)?;
    }

    if command_exists("cargo-flamegraph") {
        eprintln!(
            "xtask: cargo-flamegraph is installed; run targeted flamegraphs manually when needed"
        );
    }

    Ok(())
}

fn ci() -> Result<()> {
    check()?;
    test()?;
    cargo([
        "bench",
        "-p",
        "tokenfs-algos",
        "--all-features",
        "--bench",
        "histogram",
        "--no-run",
    ])?;
    Ok(())
}

#[derive(Debug)]
struct CriterionRecord {
    full_id: String,
    group: String,
    kernel: String,
    workload_id: String,
    metadata: HashMap<String, String>,
    throughput_bytes: u64,
    mean_ns: f64,
    gib_per_s: f64,
}

struct RunMetadata {
    timestamp: u64,
    commit: String,
    dirty: bool,
    rustc: String,
    host_cpu: String,
    processor: Value,
}

struct BenchLogRecord {
    label: String,
    gib_per_s: f64,
    mean_ns: f64,
}

struct BenchDelta {
    label: String,
    old_gib_per_s: f64,
    new_gib_per_s: f64,
    old_mean_ns: f64,
    new_mean_ns: f64,
    change_pct: f64,
}

fn record_bench_history(group_filter: Option<&str>) -> Result<()> {
    record_bench_history_since(group_filter, None)
}

fn record_bench_history_since(
    group_filter: Option<&str>,
    min_modified: Option<SystemTime>,
) -> Result<()> {
    let records = collect_criterion_records(group_filter, min_modified)?;
    if records.is_empty() {
        eprintln!("xtask: no Criterion records found to log");
        return Ok(());
    }

    let manifest = read_workload_manifest();
    let run = RunMetadata {
        timestamp: unix_timestamp(),
        commit: capture_command("git", &["rev-parse", "--short=12", "HEAD"])
            .unwrap_or_else(|_| "no-commit".into()),
        dirty: capture_command("git", &["status", "--porcelain"])
            .map(|status| !status.trim().is_empty())
            .unwrap_or(false),
        rustc: capture_command("rustc", &["-Vv"]).unwrap_or_else(|_| "unknown rustc".into()),
        host_cpu: host_cpu_model().unwrap_or_else(|| "unknown cpu".into()),
        processor: processor_profile_json(),
    };
    let log_dir = bench_history_dir();
    let run_dir = log_dir.join("runs");
    fs::create_dir_all(&run_dir).map_err(|error| {
        format!(
            "failed to create benchmark history directory `{}`: {error}",
            run_dir.display()
        )
    })?;

    let dirty_suffix = if run.dirty { "-dirty" } else { "" };
    let run_id = format!("{}-{}{}", run.timestamp, run.commit, dirty_suffix);
    let jsonl_path = run_dir.join(format!("{run_id}.jsonl"));
    let md_path = run_dir.join(format!("{run_id}.md"));

    write_bench_jsonl(&jsonl_path, &records, &manifest, &run)?;
    write_bench_markdown(&md_path, &records, &manifest, &run)?;
    fs::copy(&md_path, log_dir.join("latest.md")).map_err(|error| {
        format!(
            "failed to update latest benchmark summary from `{}`: {error}",
            md_path.display()
        )
    })?;
    append_bench_index(&log_dir, &run, records.len(), &jsonl_path, &md_path)?;

    eprintln!(
        "xtask: wrote benchmark history `{}` and `{}`",
        jsonl_path.display(),
        md_path.display()
    );

    Ok(())
}

fn collect_criterion_records(
    group_filter: Option<&str>,
    min_modified: Option<SystemTime>,
) -> Result<Vec<CriterionRecord>> {
    let mut benchmark_files = Vec::new();
    collect_named_files(
        Path::new("target/criterion"),
        "benchmark.json",
        &mut benchmark_files,
    )?;

    let mut records = Vec::new();
    for benchmark_path in benchmark_files {
        if benchmark_path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            != Some("new")
        {
            continue;
        }

        let benchmark = read_json_file(&benchmark_path)?;
        let Some(full_id) = benchmark.get("full_id").and_then(Value::as_str) else {
            continue;
        };

        if let Some(filter) = group_filter
            && !full_id.starts_with(filter)
        {
            continue;
        }

        let Some(parent) = benchmark_path.parent() else {
            continue;
        };
        let estimates_path = parent.join("estimates.json");
        if let Some(min_modified) = min_modified {
            let modified = fs::metadata(&estimates_path)
                .and_then(|metadata| metadata.modified())
                .map_err(|error| {
                    format!(
                        "failed to read metadata for `{}`: {error}",
                        estimates_path.display()
                    )
                })?;
            if modified < min_modified {
                continue;
            }
        }

        let estimates = read_json_file(&estimates_path)?;
        let mean_ns = estimates
            .get("mean")
            .and_then(|mean| mean.get("point_estimate"))
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "missing mean.point_estimate in `{}`",
                    parent.join("estimates.json").display()
                )
            })?;
        let throughput_bytes = benchmark
            .get("throughput")
            .and_then(|throughput| throughput.get("Bytes"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let gib_per_s = if throughput_bytes == 0 || mean_ns == 0.0 {
            0.0
        } else {
            throughput_bytes as f64 / (mean_ns / 1_000_000_000.0) / 1024_f64.powi(3)
        };

        let mut parts = full_id.split('/').collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let group = parts.remove(0).to_owned();
        let kernel = parts.remove(0).to_owned();
        let workload_id = parts.join("/");
        let metadata = parse_key_value_segments(&parts);

        records.push(CriterionRecord {
            full_id: full_id.to_owned(),
            group,
            kernel,
            workload_id,
            metadata,
            throughput_bytes,
            mean_ns,
            gib_per_s,
        });
    }

    records.sort_by(|left, right| left.full_id.cmp(&right.full_id));
    Ok(records)
}

fn collect_named_files(path: &Path, name: &str, out: &mut Vec<PathBuf>) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }

    let entries = fs::read_dir(path)
        .map_err(|error| format!("failed to read directory `{}`: {error}", path.display()))?;

    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "failed to read directory entry under `{}`: {error}",
                path.display()
            )
        })?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            collect_named_files(&entry_path, name, out)?;
        } else if entry_path
            .file_name()
            .and_then(|file_name| file_name.to_str())
            == Some(name)
        {
            out.push(entry_path);
        }
    }

    Ok(())
}

fn parse_key_value_segments(segments: &[&str]) -> HashMap<String, String> {
    let mut metadata = HashMap::new();
    for segment in segments {
        if let Some((key, value)) = segment.split_once('=') {
            metadata.insert(key.to_owned(), value.to_owned());
        }
    }
    metadata
}

fn read_workload_manifest() -> HashMap<String, Value> {
    let path = env::var_os("TOKENFS_ALGOS_WORKLOAD_MANIFEST")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/tokenfs-algos/workload-manifest.jsonl"));
    let Ok(file) = File::open(&path) else {
        return HashMap::new();
    };

    let mut manifest = HashMap::new();
    for line in BufReader::new(file)
        .lines()
        .map_while(std::result::Result::ok)
    {
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(workload_id) = value.get("workload_id").and_then(Value::as_str) else {
            continue;
        };
        manifest.insert(workload_id.to_owned(), value);
    }

    manifest
}

fn write_bench_jsonl(
    path: &Path,
    records: &[CriterionRecord],
    manifest: &HashMap<String, Value>,
    run: &RunMetadata,
) -> Result<()> {
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;

    for record in records {
        let line = json!({
            "timestamp_unix": run.timestamp,
            "git_commit": run.commit,
            "git_dirty": run.dirty,
            "rustc": run.rustc,
            "host_cpu": run.host_cpu,
            "processor": run.processor.clone(),
            "full_id": record.full_id,
            "group": record.group,
            "kernel": record.kernel,
            "workload_id": record.workload_id,
            "metadata": record.metadata,
            "manifest": manifest.get(&record.workload_id),
            "throughput_bytes": record.throughput_bytes,
            "mean_ns": record.mean_ns,
            "gib_per_s": record.gib_per_s,
        });
        serde_json::to_writer(&mut file, &line)
            .map_err(|error| format!("failed to serialize benchmark record: {error}"))?;
        file.write_all(b"\n")
            .map_err(|error| format!("failed to write `{}`: {error}", path.display()))?;
    }

    Ok(())
}

fn write_bench_markdown(
    path: &Path,
    records: &[CriterionRecord],
    manifest: &HashMap<String, Value>,
    run: &RunMetadata,
) -> Result<()> {
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;
    let dirty_label = if run.dirty { "yes" } else { "no" };

    writeln!(file, "# Benchmark Run").map_err(write_error(path))?;
    writeln!(file).map_err(write_error(path))?;
    writeln!(file, "- timestamp_unix: `{}`", run.timestamp).map_err(write_error(path))?;
    writeln!(file, "- git_commit: `{}`", run.commit).map_err(write_error(path))?;
    writeln!(file, "- git_dirty: `{dirty_label}`").map_err(write_error(path))?;
    writeln!(file, "- host_cpu: `{}`", run.host_cpu).map_err(write_error(path))?;
    writeln!(
        file,
        "- logical_cpus: `{}`",
        logical_cpus_string(&run.processor)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "- cpu_features: `{}`",
        cpu_features_string(&run.processor)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "- cache_summary: `{}`",
        cache_summary_string(&run.processor)
    )
    .map_err(write_error(path))?;
    writeln!(file, "- records: `{}`", records.len()).map_err(write_error(path))?;
    writeln!(file, "- rustc: `{}`", first_line(&run.rustc)).map_err(write_error(path))?;
    writeln!(file).map_err(write_error(path))?;

    write_best_by_workload(&mut file, path, records, manifest)?;
    write_all_results(&mut file, path, records, manifest)?;

    Ok(())
}

fn write_best_by_workload(
    file: &mut File,
    path: &Path,
    records: &[CriterionRecord],
    manifest: &HashMap<String, Value>,
) -> Result<()> {
    let mut best: HashMap<&str, &CriterionRecord> = HashMap::new();
    for record in records {
        best.entry(&record.workload_id)
            .and_modify(|current| {
                if record.gib_per_s > current.gib_per_s {
                    *current = record;
                }
            })
            .or_insert(record);
    }

    let mut rows = best.into_values().collect::<Vec<_>>();
    rows.sort_by_key(|record| workload_sort_key(record));

    writeln!(file, "## Best By Workload").map_err(write_error(path))?;
    writeln!(file).map_err(write_error(path))?;
    writeln!(
        file,
        "| Case | Content | Entropy | Scale | Access | Chunk | Threads | Planner | Best kernel | GiB/s | Mean | H1 |"
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|"
    )
    .map_err(write_error(path))?;

    for record in rows {
        writeln!(
            file,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.2} | {} | {} |",
            cell(record, manifest, "case"),
            cell(record, manifest, "content"),
            cell_any(record, manifest, &["entropy_class", "entropy"]),
            cell_any(record, manifest, &["entropy_scale", "scale"]),
            cell(record, manifest, "access"),
            cell_any(record, manifest, &["chunk", "chunk_size"]),
            cell(record, manifest, "threads"),
            cell(record, manifest, "planned_kernel"),
            record.kernel,
            record.gib_per_s,
            format_duration_ns(record.mean_ns),
            cell(record, manifest, "h1_bits_per_byte"),
        )
        .map_err(write_error(path))?;
    }

    writeln!(file).map_err(write_error(path))?;
    Ok(())
}

fn write_all_results(
    file: &mut File,
    path: &Path,
    records: &[CriterionRecord],
    manifest: &HashMap<String, Value>,
) -> Result<()> {
    let mut rows = records.iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        workload_sort_key(left)
            .cmp(&workload_sort_key(right))
            .then_with(|| left.kernel.cmp(&right.kernel))
    });

    writeln!(file, "## All Results").map_err(write_error(path))?;
    writeln!(file).map_err(write_error(path))?;
    writeln!(
        file,
        "| Case | Access | Chunk | Threads | Kernel | GiB/s | Mean | Bytes | Pattern |"
    )
    .map_err(write_error(path))?;
    writeln!(file, "|---|---|---:|---:|---|---:|---:|---:|---|").map_err(write_error(path))?;

    for record in rows {
        writeln!(
            file,
            "| {} | {} | {} | {} | {} | {:.2} | {} | {} | {} |",
            cell(record, manifest, "case"),
            cell(record, manifest, "access"),
            cell_any(record, manifest, &["chunk", "chunk_size"]),
            cell(record, manifest, "threads"),
            record.kernel,
            record.gib_per_s,
            format_duration_ns(record.mean_ns),
            record.throughput_bytes,
            cell(record, manifest, "pattern"),
        )
        .map_err(write_error(path))?;
    }

    Ok(())
}

fn append_bench_index(
    log_dir: &Path,
    run: &RunMetadata,
    records: usize,
    jsonl_path: &Path,
    md_path: &Path,
) -> Result<()> {
    let index_path = log_dir.join("index.tsv");
    let is_new = !index_path.exists();
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&index_path)
        .map_err(|error| format!("failed to open `{}`: {error}", index_path.display()))?;

    if is_new {
        writeln!(
            file,
            "timestamp_unix\tgit_commit\tgit_dirty\trecords\tjsonl\tmarkdown"
        )
        .map_err(write_error(&index_path))?;
    }

    writeln!(
        file,
        "{}\t{}\t{}\t{}\t{}\t{}",
        run.timestamp,
        run.commit,
        run.dirty,
        records,
        jsonl_path.display(),
        md_path.display()
    )
    .map_err(write_error(&index_path))?;

    Ok(())
}

fn meta(record: &CriterionRecord, manifest: &HashMap<String, Value>, key: &str) -> Option<String> {
    if let Some(value) = record.metadata.get(key) {
        return Some(value.clone());
    }

    manifest
        .get(&record.workload_id)
        .and_then(|workload| workload.get(key))
        .map(value_to_summary_string)
}

fn cell(record: &CriterionRecord, manifest: &HashMap<String, Value>, key: &str) -> String {
    meta(record, manifest, key).unwrap_or_default()
}

fn cell_any(record: &CriterionRecord, manifest: &HashMap<String, Value>, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| meta(record, manifest, key))
        .unwrap_or_default()
}

fn value_to_summary_string(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        text.to_owned()
    } else if let Some(number) = value.as_u64() {
        number.to_string()
    } else if let Some(number) = value.as_i64() {
        number.to_string()
    } else if let Some(number) = value.as_f64() {
        format!("{number:.2}")
    } else {
        value.to_string()
    }
}

fn workload_sort_key(record: &CriterionRecord) -> String {
    format!(
        "{}|{}|{}|{}|{}",
        record.metadata.get("case").map_or("", String::as_str),
        record.metadata.get("access").map_or("", String::as_str),
        record.metadata.get("chunk").map_or("", String::as_str),
        record.metadata.get("threads").map_or("", String::as_str),
        record.kernel,
    )
}

fn read_json_file(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|error| format!("failed to parse `{}`: {error}", path.display()))
}

fn read_bench_log_records(path: &Path) -> Result<HashMap<String, BenchLogRecord>> {
    let file = File::open(path)
        .map_err(|error| format!("failed to open `{}`: {error}", path.display()))?;
    let mut records = HashMap::new();

    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line_number = index + 1;
        let line = line.map_err(|error| {
            format!(
                "failed to read line {line_number} from `{}`: {error}",
                path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let value = serde_json::from_str::<Value>(&line).map_err(|error| {
            format!(
                "failed to parse line {line_number} from `{}`: {error}",
                path.display()
            )
        })?;
        let full_id = value
            .get("full_id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!(
                    "missing `full_id` on line {line_number} from `{}`",
                    path.display()
                )
            })?;
        let kernel = value.get("kernel").and_then(Value::as_str).unwrap_or("");
        let workload_id = value
            .get("workload_id")
            .and_then(Value::as_str)
            .unwrap_or(full_id);
        let gib_per_s = value
            .get("gib_per_s")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "missing `gib_per_s` on line {line_number} from `{}`",
                    path.display()
                )
            })?;
        let mean_ns = value
            .get("mean_ns")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "missing `mean_ns` on line {line_number} from `{}`",
                    path.display()
                )
            })?;
        let label = if kernel.is_empty() {
            workload_id.to_owned()
        } else {
            format!("{kernel}/{workload_id}")
        };

        records.insert(
            full_id.to_owned(),
            BenchLogRecord {
                label,
                gib_per_s,
                mean_ns,
            },
        );
    }

    Ok(records)
}

fn bench_history_dir() -> PathBuf {
    env::var_os("TOKENFS_ALGOS_BENCH_HISTORY")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/bench-history"))
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn capture_command(program: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run `{program}`: {error}"))?;

    if !output.status.success() {
        return Err(format!("`{program}` exited with {}", output.status));
    }

    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn host_cpu_model() -> Option<String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").ok()?;
    cpuinfo.lines().find_map(|line| {
        line.strip_prefix("model name")
            .and_then(|rest| rest.split_once(':'))
            .map(|(_, value)| value.trim().to_owned())
    })
}

fn processor_profile_json() -> Value {
    json!({
        "arch": env::consts::ARCH,
        "os": env::consts::OS,
        "family": env::consts::FAMILY,
        "logical_cpus": std::thread::available_parallelism().ok().map(core::num::NonZeroUsize::get),
        "cpu_features": detected_cpu_features(),
        "caches": cache_topology_json(),
    })
}

fn detected_cpu_features() -> Vec<&'static str> {
    let mut features = Vec::new();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            features.push("sse2");
        }
        if std::arch::is_x86_feature_detected!("ssse3") {
            features.push("ssse3");
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            features.push("sse4.2");
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            features.push("avx2");
        }
        if std::arch::is_x86_feature_detected!("bmi1") {
            features.push("bmi1");
        }
        if std::arch::is_x86_feature_detected!("bmi2") {
            features.push("bmi2");
        }
        if std::arch::is_x86_feature_detected!("popcnt") {
            features.push("popcnt");
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            features.push("avx512f");
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            features.push("neon");
        }
        if std::arch::is_aarch64_feature_detected!("sve") {
            features.push("sve");
        }
        if std::arch::is_aarch64_feature_detected!("sve2") {
            features.push("sve2");
        }
    }

    features
}

fn cache_topology_json() -> Vec<Value> {
    let root = Path::new("/sys/devices/system/cpu/cpu0/cache");
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };

    let mut caches = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if !name.starts_with("index") {
            continue;
        }

        let level = read_trimmed(path.join("level"));
        let kind = read_trimmed(path.join("type"));
        let size_raw = read_trimmed(path.join("size"));
        let line_size_raw = read_trimmed(path.join("coherency_line_size"));
        let shared_cpu_list = read_trimmed(path.join("shared_cpu_list"));

        caches.push(json!({
            "level": level.as_deref().and_then(|value| value.parse::<u8>().ok()),
            "type": kind,
            "size_raw": size_raw,
            "size_bytes": size_raw.as_deref().and_then(parse_cache_size_bytes),
            "line_size": line_size_raw.as_deref().and_then(|value| value.parse::<u64>().ok()),
            "shared_cpu_list": shared_cpu_list,
        }));
    }

    caches.sort_by_key(|cache| {
        (
            cache.get("level").and_then(Value::as_u64).unwrap_or(0),
            cache
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        )
    });
    caches
}

fn read_trimmed(path: PathBuf) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn parse_cache_size_bytes(value: &str) -> Option<u64> {
    let split = value
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, suffix) = value.split_at(split);
    let number = number.parse::<u64>().ok()?;
    let multiplier = match suffix.trim().to_ascii_uppercase().as_str() {
        "" => 1,
        "K" | "KB" | "KIB" => 1024,
        "M" | "MB" | "MIB" => 1024 * 1024,
        "G" | "GB" | "GIB" => 1024 * 1024 * 1024,
        _ => return None,
    };
    Some(number.saturating_mul(multiplier))
}

fn logical_cpus_string(processor: &Value) -> String {
    processor
        .get("logical_cpus")
        .and_then(Value::as_u64)
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn cpu_features_string(processor: &Value) -> String {
    processor
        .get("cpu_features")
        .and_then(Value::as_array)
        .map(|features| {
            features
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(",")
        })
        .filter(|features| !features.is_empty())
        .unwrap_or_else(|| "none-detected".into())
}

fn cache_summary_string(processor: &Value) -> String {
    let Some(caches) = processor.get("caches").and_then(Value::as_array) else {
        return "unknown".into();
    };

    let summary = caches
        .iter()
        .filter_map(|cache| {
            let level = cache.get("level").and_then(Value::as_u64)?;
            let kind = cache.get("type").and_then(Value::as_str).unwrap_or("?");
            let size = cache.get("size_raw").and_then(Value::as_str).unwrap_or("?");
            Some(format!("L{level} {kind} {size}"))
        })
        .collect::<Vec<_>>()
        .join("; ");

    if summary.is_empty() {
        "unknown".into()
    } else {
        summary
    }
}

fn first_line(value: &str) -> &str {
    value.lines().next().unwrap_or(value)
}

fn format_duration_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.1} us", ns / 1_000.0)
    } else {
        format!("{ns:.1} ns")
    }
}

fn write_error(path: &Path) -> impl FnOnce(std::io::Error) -> String + '_ {
    |error| format!("failed to write `{}`: {error}", path.display())
}

fn cargo<const N: usize>(args: [&str; N]) -> Result<()> {
    run_command("cargo", cargo_args(args))
}

fn cargo_args<const N: usize>(args: [&str; N]) -> Vec<OsString> {
    args.into_iter().map(OsString::from).collect()
}

fn run_command(program: &str, args: Vec<OsString>) -> Result<()> {
    run_command_with_env(program, args, Vec::new())
}

fn run_command_with_env(
    program: &str,
    args: Vec<OsString>,
    env_vars: Vec<(OsString, OsString)>,
) -> Result<()> {
    eprintln!(
        "xtask: running `{program} {}`",
        args.iter()
            .map(|arg| arg.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    );

    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    for (name, value) in env_vars {
        eprintln!(
            "xtask: setting `{}` to `{}`",
            name.to_string_lossy(),
            value.to_string_lossy()
        );
        command.env(name, value);
    }

    let status = command
        .status()
        .map_err(|error| format!("failed to run `{program}`: {error}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("`{program}` exited with {status}"))
    }
}

fn real_data_args(args: &[OsString]) -> Result<(PathBuf, Vec<OsString>)> {
    let (path, extra) = if let Some(first) = args.first() {
        if !first.to_string_lossy().starts_with('-') {
            (PathBuf::from(first), args[1..].to_vec())
        } else {
            (default_real_data_path()?, args.to_vec())
        }
    } else {
        (default_real_data_path()?, Vec::new())
    };

    if path.is_file() {
        Ok((path, extra))
    } else {
        Err(format!(
            "real benchmark data `{}` does not exist or is not a file",
            path.display()
        ))
    }
}

fn default_real_data_path() -> Result<PathBuf> {
    if let Some(path) = env::var_os("TOKENFS_ALGOS_REAL_DATA") {
        return Ok(PathBuf::from(path));
    }

    let Some(home) = env::var_os("HOME") else {
        return Err(
            "set TOKENFS_ALGOS_REAL_DATA or pass a path to `cargo xtask bench-real`".into(),
        );
    };

    Ok(PathBuf::from(home).join("ubuntu-26.04-desktop-amd64.iso"))
}

fn command_exists(program: &str) -> bool {
    let Some(path) = env::var_os("PATH") else {
        return false;
    };

    env::split_paths(&path).any(|dir| {
        let candidate = dir.join(program);
        candidate.is_file()
    })
}

fn ensure_profile_dir() -> Result<()> {
    let path = Path::new("target/profiles");
    std::fs::create_dir_all(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))
}

fn help() {
    eprintln!(
        "usage: cargo xtask <task>\n\n\
         tasks:\n\
           check    fmt, clippy, and docs\n\
           test     debug and release tests\n\
           bench    criterion benchmarks\n\
           bench-adaptive\n\
                    compare adaptive histogram kernels\n\
           bench-kernels\n\
                    compare experimental histogram kernels\n\
           bench-real [path]\n\
                    criterion benchmarks with TOKENFS_ALGOS_REAL_DATA\n\
           bench-adaptive-real [path]\n\
                    adaptive benchmarks with TOKENFS_ALGOS_REAL_DATA\n\
           bench-kernels-real [path]\n\
                    kernel comparison with TOKENFS_ALGOS_REAL_DATA\n\
           bench-adaptive-contexts-real [path]\n\
                    adaptive file/sequential context benchmarks\n\
           bench-workloads\n\
                    run the synthetic workload matrix and log results\n\
           bench-workloads-real [path]\n\
                    run synthetic + real workload matrix and log results\n\
           bench-workloads-adaptive\n\
                    compare adaptive kernels over the workload matrix\n\
           bench-workloads-adaptive-real [path]\n\
                    compare adaptive kernels over synthetic + real workloads\n\
           bench-calibrate\n\
                    run a short adaptive calibration matrix and log results\n\
           bench-compare <old.jsonl> <new.jsonl>\n\
                    compare two benchmark history JSONL runs\n\
           bench-log\n\
                    log the current target/criterion results to benchmark history\n\
           profile  benchmark under perf when available\n\
           profile-real [path]\n\
                    real-data benchmark under perf when available\n\
           ci       local CI gate"
    );
}

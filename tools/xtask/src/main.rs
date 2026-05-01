//! Workspace automation tasks.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
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
        "bench-smoke" => bench_smoke(&rest),
        "bench-synthetic-full" => bench_synthetic_full(&rest),
        "bench-real-iso" => bench_real_iso(&rest),
        "bench-real-f21" => bench_real_f21(&rest),
        "bench-size-sweep" => bench_size_sweep(&rest),
        "bench-alignment-sweep" => bench_alignment_sweep(&rest),
        "bench-thread-topology" => bench_thread_topology(&rest),
        "bench-planner-parity" => bench_planner_parity(&rest),
        "bench-planner-parity-real" => bench_planner_parity_real(&rest),
        "bench-cache-hot-cold" => bench_cache_hot_cold(&rest),
        "bench-real-magic-bpe" => bench_real_magic_bpe(&rest),
        "bench-profile" => bench_profile_suite(&rest),
        "bench-primitives" => bench_primitives(&rest),
        "bench-primitives-real" => bench_primitives_real(&rest),
        "bench-histogram-primitive" => bench_primitive_filter(&rest, "histogram"),
        "bench-histogram-primitive-real" => bench_primitive_filter_real(&rest, "histogram"),
        "bench-fingerprint" => bench_primitive_filter(&rest, "fingerprint"),
        "bench-sketch" => bench_primitive_filter(&rest, "sketch"),
        "bench-byteclass" => bench_primitive_filter(&rest, "byteclass"),
        "bench-byteclass-real" => bench_primitive_filter_real(&rest, "byteclass"),
        "bench-runlength" => bench_primitive_filter(&rest, "runlength"),
        "bench-entropy" => bench_primitive_filter(&rest, "entropy"),
        "bench-divergence" => bench_primitive_filter(&rest, "divergence"),
        "bench-distribution" => bench_primitive_filter(&rest, "distribution"),
        "bench-distribution-real" => bench_primitive_filter_real(&rest, "distribution"),
        "bench-ngram-sketch" => bench_primitive_filter(&rest, "ngram sketch-dense"),
        "bench-ngram-sketch-real" => bench_primitive_filter_real(&rest, "ngram sketch-dense"),
        "bench-selector" => bench_primitive_filter(&rest, "selector"),
        "bench-compare" => bench_compare(&rest),
        "bench-report" => bench_report(&rest),
        "bench-log" => record_bench_history(None),
        "calibrate-magic-bpe" => calibrate_magic_bpe(&rest),
        "profile" => profile(&rest),
        "profile-real" => profile_real(&rest),
        "profile-flamegraph" => profile_flamegraph(&rest),
        "profile-flamegraph-real" => profile_flamegraph_real(&rest),
        "profile-primitives" => profile_primitives(&rest),
        "profile-primitives-real" => profile_primitives_real(&rest),
        "profile-primitives-flamegraph" => profile_primitives_flamegraph(&rest),
        "profile-primitives-flamegraph-real" => profile_primitives_flamegraph_real(&rest),
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
            ("TOKENFS_ALGOS_INCLUDE_DIRECT".into(), "1".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
        ],
    )
}

fn bench_smoke(extra: &[OsString]) -> Result<()> {
    let args = default_or_extra(
        extra,
        [
            "--",
            "--sample-size",
            "10",
            "--warm-up-time",
            "0.01",
            "--measurement-time",
            "0.01",
            "workload_matrix",
        ],
    );
    bench_workloads_with_env(
        &args,
        vec![
            ("TOKENFS_ALGOS_ADAPTIVE_ONLY".into(), "1".into()),
            ("TOKENFS_ALGOS_INCLUDE_DIRECT".into(), "1".into()),
        ],
    )
}

fn bench_synthetic_full(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(
        extra,
        vec![
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_MATRIX_LEVEL".into(), "full".into()),
            (
                "TOKENFS_ALGOS_WORKLOAD_SUITE".into(),
                "synthetic-full".into(),
            ),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
        ],
    )
}

fn bench_real_iso(args: &[OsString]) -> Result<()> {
    let (path, extra) = path_or_default(args, "~/ubuntu-26.04-desktop-amd64.iso")?;
    bench_workloads_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_MATRIX_LEVEL".into(), "full".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
        ],
    )
}

fn bench_real_f21(args: &[OsString]) -> Result<()> {
    let (path, extra) = path_or_first_existing(
        args,
        &[
            "/nas4/data/tokenfs-ubuntu/bench/cow/f22-extent-bytes.bin",
            "../tokenfs-paper/data/tokenizers-corpus-matched/rootfs-65536.json",
            "../tokenfs-paper/data/tokenizers-corpus-matched/rootfs-4096.json",
        ],
    )?;
    bench_workloads_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_F21_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_MATRIX_LEVEL".into(), "full".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
        ],
    )
}

fn bench_size_sweep(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(
        extra,
        vec![
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_SIZE_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_ACCESS_SWEEP".into(), "1".into()),
        ],
    )
}

fn bench_alignment_sweep(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(
        extra,
        vec![
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_ALIGNMENT_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_ACCESS_SWEEP".into(), "1".into()),
        ],
    )
}

fn bench_thread_topology(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(
        extra,
        vec![
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
        ],
    )
}

fn bench_planner_parity(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(
        extra,
        vec![
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
            (
                "TOKENFS_ALGOS_WORKLOAD_SUITE".into(),
                "synthetic-full".into(),
            ),
        ],
    )
}

fn bench_planner_parity_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = path_or_first_existing(
        args,
        &[
            "/nas4/data/tokenfs-ubuntu/bench/cow/f22-extent-bytes.bin",
            "../tokenfs-paper/data/tokenizers-corpus-matched/rootfs-65536.json",
            "../tokenfs-paper/data/tokenizers-corpus-matched/rootfs-4096.json",
        ],
    )?;
    bench_workloads_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_F21_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            (
                "TOKENFS_ALGOS_WORKLOAD_SUITE".into(),
                "synthetic-full,paper".into(),
            ),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
        ],
    )
}

fn bench_real_magic_bpe(args: &[OsString]) -> Result<()> {
    let (path, extra) = path_or_default(args, "/nas4/data/training/magic-bpe/project/data")?;
    bench_workloads_with_env(
        &extra,
        vec![
            ("TOKENFS_ALGOS_MAGIC_BPE_DATA".into(), path.into_os_string()),
            ("TOKENFS_ALGOS_ONLY_MAGIC_BPE".into(), "1".into()),
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "quick".into()),
            ("TOKENFS_ALGOS_MATRIX_LEVEL".into(), "quick".into()),
        ],
    )
}

fn calibrate_magic_bpe(args: &[OsString]) -> Result<()> {
    let (root, _extra) = path_or_default(args, "/nas4/data/training/magic-bpe/project/data")?;
    let index = root.join("processed-index.jsonl");
    let processed = root.join("processed");
    let limit = env_usize("TOKENFS_ALGOS_MAGIC_BPE_LIMIT").unwrap_or(4096);
    let per_mime_limit = env_usize("TOKENFS_ALGOS_MAGIC_BPE_PER_MIME_LIMIT").unwrap_or(128);
    let seed = env_usize("TOKENFS_ALGOS_MAGIC_BPE_SEED").unwrap_or(0x51f1_5eed) as u64;
    let shuffle = env_bool("TOKENFS_ALGOS_MAGIC_BPE_SHUFFLE").unwrap_or(true);
    let output = env::var_os("TOKENFS_ALGOS_MAGIC_BPE_CALIBRATION")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/calibration/magic-bpe-byte-histograms.jsonl"));

    let mut candidates = magic_bpe_candidates(&index, seed, shuffle)?;
    candidates.sort_by_key(|candidate| candidate.key);

    let mut aggregates = BTreeMap::<String, MagicBpeAggregate>::new();
    let mut per_mime = BTreeMap::<String, usize>::new();
    let mut selected = 0_usize;
    for candidate in candidates {
        if limit != 0 && selected >= limit {
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

        let aggregate = aggregates
            .entry(candidate.mime.clone())
            .or_insert_with(|| MagicBpeAggregate::new(candidate.mime.clone()));
        aggregate.samples += 1;
        aggregate.bytes += bytes.len() as u64;
        for &byte in &bytes {
            aggregate.counts[byte as usize] += 1;
        }

        *count += 1;
        selected += 1;
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create `{}`: {error}", parent.display()))?;
    }
    let mut file = File::create(&output)
        .map_err(|error| format!("failed to create `{}`: {error}", output.display()))?;
    for aggregate in aggregates.values() {
        let line = json!({
            "mime_type": aggregate.mime,
            "samples": aggregate.samples,
            "bytes": aggregate.bytes,
            "h1_bits_per_byte": entropy_from_counts(&aggregate.counts),
            "counts": aggregate.counts.to_vec(),
        });
        serde_json::to_writer(&mut file, &line)
            .map_err(|error| format!("failed to write calibration JSON: {error}"))?;
        file.write_all(b"\n").map_err(write_error(&output))?;
    }

    eprintln!(
        "xtask: wrote {} MIME byte-histogram calibrations from {} samples to `{}`",
        aggregates.len(),
        selected,
        output.display()
    );
    Ok(())
}

fn bench_cache_hot_cold(extra: &[OsString]) -> Result<()> {
    bench_workloads_with_env(
        extra,
        vec![
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_CACHE_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_ACCESS_SWEEP".into(), "1".into()),
        ],
    )
}

fn bench_profile_suite(extra: &[OsString]) -> Result<()> {
    profile_with_env(
        extra,
        vec![
            ("TOKENFS_ALGOS_KERNEL_SWEEP".into(), "1".into()),
            ("TOKENFS_ALGOS_WORKLOAD_MATRIX".into(), "1".into()),
            ("TOKENFS_ALGOS_THREAD_SWEEP".into(), "full".into()),
        ],
    )
}

fn bench_primitives(extra: &[OsString]) -> Result<()> {
    let args = default_or_extra(
        extra,
        [
            "--",
            "--sample-size",
            "10",
            "--warm-up-time",
            "0.01",
            "--measurement-time",
            "0.02",
            "primitive_matrix",
        ],
    );
    bench_primitives_with_env(&args, Vec::new())
}

fn bench_primitives_real(extra: &[OsString]) -> Result<()> {
    let args = default_or_extra(
        extra,
        [
            "--",
            "--sample-size",
            "10",
            "--warm-up-time",
            "0.01",
            "--measurement-time",
            "0.02",
            "primitive_matrix",
        ],
    );
    bench_primitives_with_env(
        &args,
        vec![("TOKENFS_ALGOS_PRIMITIVE_REAL".into(), "1".into())],
    )
}

fn bench_primitive_filter(extra: &[OsString], filter: &str) -> Result<()> {
    let args = default_or_extra(
        extra,
        [
            "--",
            "--sample-size",
            "10",
            "--warm-up-time",
            "0.01",
            "--measurement-time",
            "0.02",
            "primitive_matrix",
        ],
    );
    bench_primitives_with_env(
        &args,
        vec![(
            "TOKENFS_ALGOS_PRIMITIVE_FILTER".into(),
            OsString::from(filter),
        )],
    )
}

fn bench_primitive_filter_real(extra: &[OsString], filter: &str) -> Result<()> {
    let args = default_or_extra(
        extra,
        [
            "--",
            "--sample-size",
            "10",
            "--warm-up-time",
            "0.01",
            "--measurement-time",
            "0.02",
            "primitive_matrix",
        ],
    );
    bench_primitives_with_env(
        &args,
        vec![
            (
                "TOKENFS_ALGOS_PRIMITIVE_FILTER".into(),
                OsString::from(filter),
            ),
            ("TOKENFS_ALGOS_PRIMITIVE_REAL".into(), "1".into()),
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

fn bench_report(args: &[OsString]) -> Result<()> {
    if args.len() > 1 {
        return Err("usage: cargo xtask bench-report [run.jsonl]".into());
    }

    let jsonl_path = if let Some(path) = args.first() {
        PathBuf::from(path)
    } else {
        latest_bench_jsonl()?
    };
    let mut records = read_bench_report_records(&jsonl_path)?;
    if records
        .iter()
        .any(|record| record.group == "workload_matrix")
    {
        records.retain(|record| record.group == "workload_matrix");
    }
    if records.is_empty() {
        return Err(format!(
            "benchmark run `{}` has no rows",
            jsonl_path.display()
        ));
    }

    let report_id = jsonl_path
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("bench-report");
    let report_dir = bench_history_dir().join("reports").join(report_id);
    fs::create_dir_all(&report_dir).map_err(|error| {
        format!(
            "failed to create benchmark report directory `{}`: {error}",
            report_dir.display()
        )
    })?;

    let timing_csv = report_dir.join("timing.csv");
    let planner_parity_csv = report_dir.join("planner-parity.csv");
    let heatmap_html = report_dir.join("heatmap.html");
    let histogram_svg = report_dir.join("throughput-histogram.svg");
    let summary_md = report_dir.join("summary.md");

    write_throughput_histogram_svg(&histogram_svg, &records)?;
    let mut visual_artifacts = vec![ReportArtifact {
        file_name: local_href(&histogram_svg),
        title: "Throughput Distribution".into(),
        caption: "Distribution of measured GiB/s values across every row in this report.".into(),
    }];
    visual_artifacts.extend(write_dimension_visuals(&report_dir, &records)?);
    write_timing_csv(&timing_csv, &records)?;
    let planner_parity_written = write_planner_parity_csv(&planner_parity_csv, &records)?;
    write_heatmap_html(
        &heatmap_html,
        &records,
        &jsonl_path,
        &timing_csv,
        planner_parity_written.then_some(planner_parity_csv.as_path()),
        &histogram_svg,
        &summary_md,
        &visual_artifacts,
    )?;
    write_bench_report_summary(
        &summary_md,
        &records,
        &jsonl_path,
        &timing_csv,
        planner_parity_written.then_some(planner_parity_csv.as_path()),
        &heatmap_html,
        &histogram_svg,
        &visual_artifacts,
    )?;

    eprintln!("xtask: wrote benchmark report `{}`", summary_md.display());
    eprintln!("xtask: wrote timing table `{}`", timing_csv.display());
    if planner_parity_written {
        eprintln!(
            "xtask: wrote planner parity `{}`",
            planner_parity_csv.display()
        );
    }
    eprintln!("xtask: wrote heatmap `{}`", heatmap_html.display());
    eprintln!(
        "xtask: wrote throughput histogram `{}`",
        histogram_svg.display()
    );

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

fn bench_primitives_with_env(
    extra: &[OsString],
    env_vars: Vec<(OsString, OsString)>,
) -> Result<()> {
    let started = SystemTime::now()
        .checked_sub(Duration::from_secs(2))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    bench_primitives_with_env_no_log(extra, env_vars)?;
    record_bench_history_since(Some("primitive_matrix"), Some(started))
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

fn bench_primitives_with_env_no_log(
    extra: &[OsString],
    env_vars: Vec<(OsString, OsString)>,
) -> Result<()> {
    let mut args = cargo_args([
        "bench",
        "-p",
        "tokenfs-algos",
        "--all-features",
        "--bench",
        "primitives",
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

fn profile_flamegraph(extra: &[OsString]) -> Result<()> {
    profile_flamegraph_with_env(extra, Vec::new())
}

fn profile_flamegraph_real(args: &[OsString]) -> Result<()> {
    let (path, extra) = real_data_args(args)?;
    profile_flamegraph_with_env(
        &extra,
        vec![("TOKENFS_ALGOS_REAL_DATA".into(), path.into_os_string())],
    )
}

fn profile_primitives(extra: &[OsString]) -> Result<()> {
    profile_primitives_with_env(extra, Vec::new())
}

fn profile_primitives_real(extra: &[OsString]) -> Result<()> {
    profile_primitives_with_env(
        extra,
        vec![("TOKENFS_ALGOS_PRIMITIVE_REAL".into(), "1".into())],
    )
}

fn profile_primitives_flamegraph(extra: &[OsString]) -> Result<()> {
    profile_primitives_flamegraph_with_env(extra, Vec::new())
}

fn profile_primitives_flamegraph_real(extra: &[OsString]) -> Result<()> {
    profile_primitives_flamegraph_with_env(
        extra,
        vec![("TOKENFS_ALGOS_PRIMITIVE_REAL".into(), "1".into())],
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
        let perf_path = profile_output_path("perf-stat", "txt");
        let mut args = vec![
            OsString::from("stat"),
            OsString::from("-d"),
            OsString::from("-o"),
            perf_path.clone().into_os_string(),
            OsString::from("cargo"),
            OsString::from("bench"),
            OsString::from("-p"),
            OsString::from("tokenfs-algos"),
            OsString::from("--bench"),
            OsString::from("histogram"),
            OsString::from("--all-features"),
        ];
        args.extend(extra.iter().cloned());
        match run_command_with_env("perf", args, env_vars.clone()) {
            Ok(()) => eprintln!("xtask: wrote perf stat output `{}`", perf_path.display()),
            Err(error) => {
                eprintln!("xtask: perf stat failed: {error}");
                eprintln!("xtask: running Criterion benchmark without perf counters");
                bench_with_env(extra, env_vars.clone())?;
            }
        }
    } else {
        eprintln!("xtask: `perf` not found; running Criterion benchmark only");
        bench_with_env(extra, env_vars)?;
    }

    if command_exists("cargo-flamegraph") {
        eprintln!(
            "xtask: cargo-flamegraph is installed; run `cargo xtask profile-flamegraph` for SVG output"
        );
    }

    Ok(())
}

fn profile_primitives_with_env(
    extra: &[OsString],
    env_vars: Vec<(OsString, OsString)>,
) -> Result<()> {
    ensure_profile_dir()?;
    cargo([
        "bench",
        "-p",
        "tokenfs-algos",
        "--bench",
        "primitives",
        "--all-features",
        "--no-run",
    ])?;

    if command_exists("perf") {
        let perf_path = profile_output_path("primitive-perf-stat", "txt");
        let mut args = vec![
            OsString::from("stat"),
            OsString::from("-d"),
            OsString::from("-o"),
            perf_path.clone().into_os_string(),
            OsString::from("cargo"),
            OsString::from("bench"),
            OsString::from("-p"),
            OsString::from("tokenfs-algos"),
            OsString::from("--bench"),
            OsString::from("primitives"),
            OsString::from("--all-features"),
        ];
        args.extend(extra.iter().cloned());
        match run_command_with_env("perf", args, env_vars.clone()) {
            Ok(()) => eprintln!("xtask: wrote perf stat output `{}`", perf_path.display()),
            Err(error) => {
                eprintln!("xtask: primitive perf stat failed: {error}");
                eprintln!("xtask: running primitive Criterion benchmark without perf counters");
                bench_primitives_with_env_no_log(extra, env_vars.clone())?;
            }
        }
    } else {
        eprintln!("xtask: `perf` not found; running primitive Criterion benchmark only");
        bench_primitives_with_env_no_log(extra, env_vars)?;
    }

    if command_exists("cargo-flamegraph") {
        eprintln!(
            "xtask: cargo-flamegraph is installed; run `cargo xtask profile-primitives-flamegraph` for SVG output"
        );
    }

    Ok(())
}

fn profile_flamegraph_with_env(
    extra: &[OsString],
    env_vars: Vec<(OsString, OsString)>,
) -> Result<()> {
    if !command_exists("cargo-flamegraph") {
        return Err(
            "`cargo-flamegraph` not found; install it with `cargo install flamegraph`".into(),
        );
    }

    ensure_profile_dir()?;
    let output = profile_output_path("flamegraph", "svg");
    let mut args = vec![
        OsString::from("flamegraph"),
        OsString::from("-o"),
        output.clone().into_os_string(),
        OsString::from("-p"),
        OsString::from("tokenfs-algos"),
        OsString::from("--features"),
        OsString::from("bench-internals"),
        OsString::from("--bench"),
        OsString::from("histogram"),
        OsString::from("--"),
    ];
    args.extend(criterion_args(extra));

    run_command_with_env("cargo", args, env_vars)?;
    eprintln!("xtask: wrote flamegraph `{}`", output.display());
    Ok(())
}

fn profile_primitives_flamegraph_with_env(
    extra: &[OsString],
    env_vars: Vec<(OsString, OsString)>,
) -> Result<()> {
    if !command_exists("cargo-flamegraph") {
        return Err(
            "`cargo-flamegraph` not found; install it with `cargo install flamegraph`".into(),
        );
    }

    ensure_profile_dir()?;
    let output = profile_output_path("primitive-flamegraph", "svg");
    let mut args = vec![
        OsString::from("flamegraph"),
        OsString::from("-o"),
        output.clone().into_os_string(),
        OsString::from("-p"),
        OsString::from("tokenfs-algos"),
        OsString::from("--features"),
        OsString::from("bench-internals"),
        OsString::from("--bench"),
        OsString::from("primitives"),
        OsString::from("--"),
    ];
    args.extend(criterion_args(extra));

    run_command_with_env("cargo", args, env_vars)?;
    eprintln!("xtask: wrote primitive flamegraph `{}`", output.display());
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

struct BenchReportRecord {
    full_id: String,
    group: String,
    kernel: String,
    workload_id: String,
    primitive: String,
    case: String,
    source: String,
    content: String,
    entropy: String,
    scale: String,
    access: String,
    chunk: String,
    threads: String,
    pattern: String,
    planned_kernel: String,
    planned_confidence_q8: String,
    planned_confidence_source: String,
    bytes: String,
    throughput_bytes: u64,
    mean_ns: f64,
    gib_per_s: f64,
}

#[derive(Clone)]
struct ReportArtifact {
    file_name: String,
    title: String,
    caption: String,
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

fn read_bench_report_records(path: &Path) -> Result<Vec<BenchReportRecord>> {
    let file = File::open(path)
        .map_err(|error| format!("failed to open `{}`: {error}", path.display()))?;
    let mut records = Vec::new();

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
        let full_id = json_string(&value, "full_id").ok_or_else(|| {
            format!(
                "missing `full_id` on line {line_number} from `{}`",
                path.display()
            )
        })?;
        let kernel = json_string(&value, "kernel").unwrap_or_default();
        let workload_id = json_string(&value, "workload_id").unwrap_or_else(|| full_id.clone());
        let throughput_bytes = value
            .get("throughput_bytes")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let mean_ns = value
            .get("mean_ns")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "missing `mean_ns` on line {line_number} from `{}`",
                    path.display()
                )
            })?;
        let gib_per_s = value
            .get("gib_per_s")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                format!(
                    "missing `gib_per_s` on line {line_number} from `{}`",
                    path.display()
                )
            })?;

        records.push(BenchReportRecord {
            full_id,
            group: json_string(&value, "group").unwrap_or_default(),
            kernel,
            workload_id,
            primitive: bench_field(&value, "primitive"),
            case: bench_field(&value, "case"),
            source: bench_field(&value, "source"),
            content: bench_field(&value, "content"),
            entropy: bench_field_any(&value, &["entropy_class", "entropy"]),
            scale: bench_field_any(&value, &["entropy_scale", "scale"]),
            access: bench_field(&value, "access"),
            chunk: bench_field_any(&value, &["chunk", "chunk_size"]),
            threads: bench_field(&value, "threads"),
            pattern: bench_field(&value, "pattern"),
            planned_kernel: bench_field(&value, "planned_kernel"),
            planned_confidence_q8: bench_field(&value, "planned_confidence_q8"),
            planned_confidence_source: bench_field(&value, "planned_confidence_source"),
            bytes: bench_field(&value, "bytes"),
            throughput_bytes,
            mean_ns,
            gib_per_s,
        });
    }

    records.sort_by(|left, right| {
        workload_label(left)
            .cmp(&workload_label(right))
            .then_with(|| left.kernel.cmp(&right.kernel))
    });
    Ok(records)
}

fn latest_bench_jsonl() -> Result<PathBuf> {
    let runs_dir = bench_history_dir().join("runs");
    let entries = fs::read_dir(&runs_dir)
        .map_err(|error| format!("failed to read `{}`: {error}", runs_dir.display()))?;
    let mut paths = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|error| format!("failed to read `{}` entry: {error}", runs_dir.display()))?
            .path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            paths.push(path);
        }
    }
    paths.sort();
    paths.pop().ok_or_else(|| {
        format!(
            "no benchmark JSONL runs found under `{}`",
            runs_dir.display()
        )
    })
}

fn write_timing_csv(path: &Path, records: &[BenchReportRecord]) -> Result<()> {
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;
    writeln!(
        file,
        "full_id,group,workload_id,primitive,case,source,content,entropy,scale,access,chunk,threads,pattern,kernel,planned_kernel,planned_confidence_q8,planned_confidence_source,planner_match,bytes,throughput_bytes,mean_ns,gib_per_s"
    )
    .map_err(write_error(path))?;

    for record in records {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6}",
            csv_cell(&record.full_id),
            csv_cell(&record.group),
            csv_cell(&record.workload_id),
            csv_cell(&record.primitive),
            csv_cell(&record.case),
            csv_cell(&record.source),
            csv_cell(&record.content),
            csv_cell(&record.entropy),
            csv_cell(&record.scale),
            csv_cell(&record.access),
            csv_cell(&record.chunk),
            csv_cell(&record.threads),
            csv_cell(&record.pattern),
            csv_cell(&record.kernel),
            csv_cell(&record.planned_kernel),
            csv_cell(&record.planned_confidence_q8),
            csv_cell(&record.planned_confidence_source),
            record.planned_kernel == record.kernel,
            csv_cell(&record.bytes),
            record.throughput_bytes,
            record.mean_ns,
            record.gib_per_s,
        )
        .map_err(write_error(path))?;
    }

    Ok(())
}

fn write_planner_parity_csv(path: &Path, records: &[BenchReportRecord]) -> Result<bool> {
    if !records
        .iter()
        .any(|record| !record.planned_kernel.trim().is_empty())
    {
        return Ok(false);
    }

    let mut workloads: BTreeMap<String, Vec<&BenchReportRecord>> = BTreeMap::new();
    for record in records {
        workloads
            .entry(workload_label(record))
            .or_default()
            .push(record);
    }

    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;
    writeln!(
        file,
        "workload,case,source,content,entropy,scale,access,chunk,threads,pattern,planned_kernel,planned_confidence_q8,planned_confidence_source,winner_kernel,winner_gib_per_s,planned_gib_per_s,gap_gib_per_s,gap_pct,kernel_results"
    )
    .map_err(write_error(path))?;

    for (workload, mut rows) in workloads {
        rows.sort_by(|left, right| {
            right
                .gib_per_s
                .partial_cmp(&left.gib_per_s)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.kernel.cmp(&right.kernel))
        });
        let Some(best) = rows.first().copied() else {
            continue;
        };
        let planned_kernel = rows
            .iter()
            .find(|row| !row.planned_kernel.is_empty())
            .map_or("", |row| row.planned_kernel.as_str());
        let planned_confidence_source = rows
            .iter()
            .find(|row| !row.planned_confidence_source.is_empty())
            .map_or("", |row| row.planned_confidence_source.as_str());
        let planned = rows.iter().find(|row| row.kernel == planned_kernel);
        let planned_gib = planned.map_or(0.0, |row| row.gib_per_s);
        let gap = best.gib_per_s - planned_gib;
        let gap_pct = if planned_gib > 0.0 {
            gap / planned_gib * 100.0
        } else {
            0.0
        };
        let kernel_results = rows
            .iter()
            .map(|row| format!("{}:{:.6}", row.kernel, row.gib_per_s))
            .collect::<Vec<_>>()
            .join(";");

        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.3},{}",
            csv_cell(&workload),
            csv_cell(&best.case),
            csv_cell(&best.source),
            csv_cell(&best.content),
            csv_cell(&best.entropy),
            csv_cell(&best.scale),
            csv_cell(&best.access),
            csv_cell(&best.chunk),
            csv_cell(&best.threads),
            csv_cell(&best.pattern),
            csv_cell(planned_kernel),
            csv_cell(&best.planned_confidence_q8),
            csv_cell(planned_confidence_source),
            csv_cell(&best.kernel),
            best.gib_per_s,
            planned_gib,
            gap,
            gap_pct,
            csv_cell(&kernel_results),
        )
        .map_err(write_error(path))?;
    }

    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn write_heatmap_html(
    path: &Path,
    records: &[BenchReportRecord],
    source: &Path,
    timing_csv: &Path,
    planner_parity_csv: Option<&Path>,
    histogram_svg: &Path,
    summary_md: &Path,
    visual_artifacts: &[ReportArtifact],
) -> Result<()> {
    let mut kernels = BTreeSet::new();
    let mut workloads: BTreeMap<String, Vec<&BenchReportRecord>> = BTreeMap::new();
    let mut thread_counts: BTreeMap<String, usize> = BTreeMap::new();
    for record in records {
        kernels.insert(record.kernel.clone());
        workloads
            .entry(workload_label(record))
            .or_default()
            .push(record);
        *thread_counts
            .entry(empty_as_unknown(&record.threads).to_owned())
            .or_default() += 1;
    }
    let kernels = kernels.into_iter().collect::<Vec<_>>();
    let thread_summary = thread_counts
        .iter()
        .map(|(threads, count)| format!("threads={threads}: {count} rows"))
        .collect::<Vec<_>>()
        .join(", ");
    let winner_summary = winner_counts(records)
        .into_iter()
        .map(|(kernel, count)| format!("{kernel}: {count} wins"))
        .collect::<Vec<_>>()
        .join(", ");

    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;
    writeln!(
        file,
        "<!doctype html><meta charset=\"utf-8\"><title>tokenfs-algos benchmark heatmap</title>"
    )
    .map_err(write_error(path))?;
    writeln!(file, "<style>{}</style>", heatmap_css()).map_err(write_error(path))?;
    writeln!(file, "<h1>Benchmark Report</h1>").map_err(write_error(path))?;
    writeln!(
        file,
        "<p>Source: <code>{}</code>. Cells show GiB/s and are colored relative to the best kernel for that workload row.</p>",
        html_escape(&source.display().to_string())
    )
    .map_err(write_error(path))?;
    write!(
        file,
        "<nav><a href=\"{}\">summary.md</a><a href=\"{}\">timing.csv</a>",
        html_escape(&local_href(summary_md)),
        html_escape(&local_href(timing_csv)),
    )
    .map_err(write_error(path))?;
    if let Some(planner_parity_csv) = planner_parity_csv {
        write!(
            file,
            "<a href=\"{}\">planner-parity.csv</a>",
            html_escape(&local_href(planner_parity_csv)),
        )
        .map_err(write_error(path))?;
    }
    writeln!(
        file,
        "<a href=\"{}\">throughput-histogram.svg</a></nav>",
        html_escape(&local_href(histogram_svg)),
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<section class=\"guide\"><h2>How to read this</h2><p>Each row is one workload shape: case, access pattern, data pattern, chunk size, and thread count. <b>Planner</b> is the planner recommendation from the workload manifest; timing CSV and planner parity CSV include confidence score and confidence source. <b>Best measured kernel</b> is the fastest measured kernel in this run. Cells marked <b>winner</b> are row winners. Green cells are closest to the row winner; red cells are furthest from it. The title tooltip on each cell includes mean time and full benchmark id.</p><p><b>Thread rows:</b> {}</p><p><b>Winner counts:</b> {}</p></section>",
        html_escape(&thread_summary),
        html_escape(&winner_summary)
    )
    .map_err(write_error(path))?;
    write_visual_gallery(path, &mut file, visual_artifacts)?;
    writeln!(file, "<h2>Throughput Heatmap</h2>").map_err(write_error(path))?;
    writeln!(
        file,
        "<table><thead><tr><th>Workload</th><th>Planner</th><th>Best measured kernel</th>"
    )
    .map_err(write_error(path))?;
    for kernel in &kernels {
        writeln!(file, "<th>{}</th>", html_escape(kernel)).map_err(write_error(path))?;
    }
    writeln!(file, "</tr></thead><tbody>").map_err(write_error(path))?;

    for (workload, rows) in workloads {
        let best = rows
            .iter()
            .fold(0.0_f64, |best, row| best.max(row.gib_per_s));
        let best_kernel = rows
            .iter()
            .max_by(|left, right| {
                left.gib_per_s
                    .partial_cmp(&right.gib_per_s)
                    .unwrap_or(Ordering::Equal)
            })
            .map_or("", |row| row.kernel.as_str());
        let planned_kernel = rows
            .iter()
            .find(|row| !row.planned_kernel.is_empty())
            .map_or("", |row| row.planned_kernel.as_str());

        writeln!(
            file,
            "<tr><th>{}</th><td>{}</td><td>{}</td>",
            html_escape(&workload),
            html_escape(planned_kernel),
            html_escape(best_kernel)
        )
        .map_err(write_error(path))?;

        for kernel in &kernels {
            let row = rows.iter().find(|row| &row.kernel == kernel);
            if let Some(row) = row {
                let ratio = if best == 0.0 {
                    0.0
                } else {
                    (row.gib_per_s / best).clamp(0.0, 1.0)
                };
                let winner = (row.gib_per_s - best).abs() <= f64::EPSILON;
                let class = if winner { "metric winner" } else { "metric" };
                let content = if winner {
                    format!(
                        "<span class=\"cell-value\">{:.2}</span><span class=\"winner-label\">winner</span>",
                        row.gib_per_s
                    )
                } else {
                    format!("{:.2}", row.gib_per_s)
                };
                writeln!(
                    file,
                    "<td class=\"{}\" style=\"{}\" title=\"mean {}; full_id {}\">{}</td>",
                    class,
                    heatmap_cell_style(ratio),
                    html_escape(&format_duration_ns(row.mean_ns)),
                    html_escape(&row.full_id),
                    content
                )
                .map_err(write_error(path))?;
            } else {
                writeln!(file, "<td class=\"missing\"></td>").map_err(write_error(path))?;
            }
        }
        writeln!(file, "</tr>").map_err(write_error(path))?;
    }

    writeln!(file, "</tbody></table>").map_err(write_error(path))?;
    Ok(())
}

fn write_throughput_histogram_svg(path: &Path, records: &[BenchReportRecord]) -> Result<()> {
    let max = records
        .iter()
        .fold(0.0_f64, |max, record| max.max(record.gib_per_s));
    let bucket_count = 24_usize;
    let mut buckets = vec![0_usize; bucket_count];
    if max > 0.0 {
        for record in records {
            let mut index = ((record.gib_per_s / max) * bucket_count as f64) as usize;
            if index >= bucket_count {
                index = bucket_count - 1;
            }
            buckets[index] += 1;
        }
    }

    let width = 960.0_f64;
    let height = 420.0_f64;
    let margin_left = 56.0_f64;
    let margin_bottom = 52.0_f64;
    let plot_width = width - margin_left - 24.0;
    let plot_height = height - 42.0 - margin_bottom;
    let max_count = buckets.iter().copied().max().unwrap_or(1).max(1) as f64;
    let bar_gap = 4.0_f64;
    let bar_width = (plot_width / bucket_count as f64) - bar_gap;

    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;
    writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"{margin_left}\" y=\"26\" font-family=\"sans-serif\" font-size=\"18\">Throughput Distribution</text>"
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<line x1=\"{margin_left}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\"/>",
        height - margin_bottom,
        width - 24.0,
        height - margin_bottom
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<line x1=\"{margin_left}\" y1=\"42\" x2=\"{margin_left}\" y2=\"{}\" stroke=\"#333\"/>",
        height - margin_bottom
    )
    .map_err(write_error(path))?;

    for (index, count) in buckets.iter().enumerate() {
        let x = margin_left + index as f64 * (bar_width + bar_gap);
        let bar_height = (*count as f64 / max_count) * plot_height;
        let y = height - margin_bottom - bar_height;
        let bucket_start = max * index as f64 / bucket_count as f64;
        let bucket_end = max * (index + 1) as f64 / bucket_count as f64;
        writeln!(
            file,
            "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{bar_width:.1}\" height=\"{bar_height:.1}\" fill=\"#2f7ed8\"><title>{bucket_start:.2}-{bucket_end:.2} GiB/s: {count}</title></rect>"
        )
        .map_err(write_error(path))?;
    }

    writeln!(
        file,
        "<text x=\"{}\" y=\"{}\" font-family=\"sans-serif\" font-size=\"12\">0 GiB/s</text>",
        margin_left,
        height - 18.0
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"{}\" y=\"{}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">{max:.2} GiB/s</text>",
        width - 24.0,
        height - 18.0
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"52\" font-family=\"sans-serif\" font-size=\"12\" transform=\"rotate(-90 18 52)\">count</text>"
    )
    .map_err(write_error(path))?;
    writeln!(file, "</svg>").map_err(write_error(path))?;
    Ok(())
}

fn write_visual_gallery(path: &Path, file: &mut File, artifacts: &[ReportArtifact]) -> Result<()> {
    writeln!(file, "<h2>Dimension Visuals</h2>").map_err(write_error(path))?;
    writeln!(file, "<section class=\"gallery\">").map_err(write_error(path))?;
    for artifact in artifacts {
        writeln!(
            file,
            "<figure><h3>{}</h3><a href=\"{}\"><img src=\"{}\" alt=\"{}\"></a><figcaption>{}</figcaption></figure>",
            html_escape(&artifact.title),
            html_escape(&artifact.file_name),
            html_escape(&artifact.file_name),
            html_escape(&artifact.title),
            html_escape(&artifact.caption)
        )
        .map_err(write_error(path))?;
    }
    writeln!(file, "</section>").map_err(write_error(path))?;
    Ok(())
}

struct DimensionSpec {
    id: &'static str,
    title: &'static str,
    value: fn(&BenchReportRecord) -> &str,
}

fn write_dimension_visuals(
    report_dir: &Path,
    records: &[BenchReportRecord],
) -> Result<Vec<ReportArtifact>> {
    let mut artifacts = Vec::new();
    let all_records = records.iter().collect::<Vec<_>>();
    let parallel_records = records
        .iter()
        .filter(|record| {
            record.access == "parallel-sequential" && parse_usize(&record.threads).is_some()
        })
        .collect::<Vec<_>>();

    let path = report_dir.join("planner-vs-best.svg");
    if records
        .iter()
        .any(|record| !record.planned_kernel.is_empty())
        && write_planner_confusion_svg(&path, records)?
    {
        artifacts.push(ReportArtifact {
            file_name: local_href(&path),
            title: "Planner vs Best Kernel".into(),
            caption: "Confusion matrix: static planner recommendation crossed with the measured winner for each workload row.".into(),
        });
    }

    let path = report_dir.join("planner-gap-top.svg");
    if write_planner_gap_svg(&path, records)? {
        artifacts.push(ReportArtifact {
            file_name: local_href(&path),
            title: "Largest Planner Gaps".into(),
            caption:
                "Worst measured throughput gaps between the planner-selected kernel and the row winner."
                    .into(),
        });
    }

    let path = report_dir.join("winner-counts.svg");
    if write_winner_counts_svg(&path, records)? {
        artifacts.push(ReportArtifact {
            file_name: local_href(&path),
            title: "Winner Counts".into(),
            caption: "How many workload rows each kernel won in this run.".into(),
        });
    }

    let path = report_dir.join("thread-scaling-by-kernel.svg");
    if write_thread_scaling_svg(&path, &parallel_records)? {
        artifacts.push(ReportArtifact {
            file_name: local_href(&path),
            title: "Thread Scaling by Kernel".into(),
            caption:
                "Median GiB/s by thread count for parallel-sequential rows, grouped by kernel."
                    .into(),
        });
    }

    for dimension in dimension_specs() {
        if dimension.id != "kernel" && dimension_has_values(&all_records, dimension.value) {
            let path = report_dir.join(format!("dimension-{}-by-kernel.svg", dimension.id));
            let title = format!("Median GiB/s: {} by Kernel", dimension.title);
            if write_dimension_heatmap_svg(
                &path,
                &title,
                dimension.title,
                "Kernel",
                &all_records,
                dimension.value,
                dim_kernel,
            )? {
                artifacts.push(ReportArtifact {
                    file_name: local_href(&path),
                    title,
                    caption: format!(
                        "Median throughput for each {} value crossed with measured kernel.",
                        dimension.title.to_ascii_lowercase()
                    ),
                });
            }
        }

        if dimension.id != "threads" && dimension_has_values(&parallel_records, dimension.value) {
            let path = report_dir.join(format!("dimension-{}-by-thread.svg", dimension.id));
            let title = format!("Median GiB/s: {} by Thread Count", dimension.title);
            if write_dimension_heatmap_svg(
                &path,
                &title,
                dimension.title,
                "Threads",
                &parallel_records,
                dimension.value,
                dim_threads,
            )? {
                artifacts.push(ReportArtifact {
                    file_name: local_href(&path),
                    title,
                    caption: format!(
                        "Parallel-sequential median throughput for each {} value as thread count changes.",
                        dimension.title.to_ascii_lowercase()
                    ),
                });
            }
        }
    }

    Ok(artifacts)
}

fn dimension_has_values(
    records: &[&BenchReportRecord],
    value: fn(&BenchReportRecord) -> &str,
) -> bool {
    records
        .iter()
        .any(|record| !value(record).trim().is_empty())
}

fn dimension_specs() -> Vec<DimensionSpec> {
    vec![
        DimensionSpec {
            id: "primitive",
            title: "Primitive",
            value: dim_primitive,
        },
        DimensionSpec {
            id: "case",
            title: "Case",
            value: dim_case,
        },
        DimensionSpec {
            id: "source",
            title: "Source",
            value: dim_source,
        },
        DimensionSpec {
            id: "content",
            title: "Content",
            value: dim_content,
        },
        DimensionSpec {
            id: "entropy",
            title: "Entropy",
            value: dim_entropy,
        },
        DimensionSpec {
            id: "scale",
            title: "Scale",
            value: dim_scale,
        },
        DimensionSpec {
            id: "access",
            title: "Access",
            value: dim_access,
        },
        DimensionSpec {
            id: "chunk",
            title: "Chunk",
            value: dim_chunk,
        },
        DimensionSpec {
            id: "threads",
            title: "Threads",
            value: dim_threads,
        },
        DimensionSpec {
            id: "confidence-source",
            title: "Planner Confidence Source",
            value: dim_confidence_source,
        },
        DimensionSpec {
            id: "pattern",
            title: "Pattern",
            value: dim_pattern,
        },
        DimensionSpec {
            id: "bytes",
            title: "Bytes",
            value: dim_bytes,
        },
        DimensionSpec {
            id: "kernel",
            title: "Kernel",
            value: dim_kernel,
        },
    ]
}

fn dim_primitive(record: &BenchReportRecord) -> &str {
    &record.primitive
}

fn dim_case(record: &BenchReportRecord) -> &str {
    &record.case
}

fn dim_source(record: &BenchReportRecord) -> &str {
    &record.source
}

fn dim_content(record: &BenchReportRecord) -> &str {
    &record.content
}

fn dim_entropy(record: &BenchReportRecord) -> &str {
    &record.entropy
}

fn dim_scale(record: &BenchReportRecord) -> &str {
    &record.scale
}

fn dim_access(record: &BenchReportRecord) -> &str {
    &record.access
}

fn dim_chunk(record: &BenchReportRecord) -> &str {
    &record.chunk
}

fn dim_threads(record: &BenchReportRecord) -> &str {
    &record.threads
}

fn dim_confidence_source(record: &BenchReportRecord) -> &str {
    &record.planned_confidence_source
}

fn dim_pattern(record: &BenchReportRecord) -> &str {
    &record.pattern
}

fn dim_bytes(record: &BenchReportRecord) -> &str {
    &record.bytes
}

fn dim_kernel(record: &BenchReportRecord) -> &str {
    &record.kernel
}

fn write_dimension_heatmap_svg(
    path: &Path,
    title: &str,
    row_axis: &str,
    col_axis: &str,
    records: &[&BenchReportRecord],
    row_value: fn(&BenchReportRecord) -> &str,
    col_value: fn(&BenchReportRecord) -> &str,
) -> Result<bool> {
    let mut cells: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    let mut rows = BTreeSet::new();
    let mut cols = BTreeSet::new();

    for record in records {
        let row = dimension_value(row_value(record));
        let col = dimension_value(col_value(record));
        rows.insert(row.clone());
        cols.insert(col.clone());
        cells.entry((row, col)).or_default().push(record.gib_per_s);
    }

    if rows.is_empty() || cols.is_empty() {
        return Ok(false);
    }

    let rows = sort_dimension_values(rows.into_iter().collect());
    let cols = sort_dimension_values(cols.into_iter().collect());
    let mut medians = BTreeMap::new();
    let mut max_value = 0.0_f64;
    for (key, values) in cells {
        let value = median(values);
        max_value = max_value.max(value);
        medians.insert(key, value);
    }

    let row_label_width = axis_label_width(&rows);
    let col_width = axis_label_width(&cols).clamp(112.0, 176.0);
    let row_height = 34.0_f64;
    let header_height = 82.0_f64;
    let width = row_label_width + col_width * cols.len() as f64 + 36.0;
    let height = header_height + row_height * rows.len() as f64 + 50.0;
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;

    writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" height=\"{height:.0}\" viewBox=\"0 0 {width:.0} {height:.0}\">"
    )
    .map_err(write_error(path))?;
    write_svg_background(&mut file, path)?;
    writeln!(
        file,
        "<text x=\"18\" y=\"28\" font-family=\"sans-serif\" font-size=\"18\" font-weight=\"700\">{}</text>",
        html_escape(title)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"50\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#57606a\">cell value = median GiB/s; color is relative to this chart's max</text>"
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"74\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"700\">{}</text>",
        html_escape(row_axis)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"{:.1}\" y=\"74\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"700\">{}</text>",
        row_label_width,
        html_escape(col_axis)
    )
    .map_err(write_error(path))?;

    for (index, col) in cols.iter().enumerate() {
        let x = row_label_width + index as f64 * col_width + col_width / 2.0;
        writeln!(
            file,
            "<text x=\"{x:.1}\" y=\"74\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">{}</text>",
            html_escape(&short_label(col, 22))
        )
        .map_err(write_error(path))?;
    }

    for (row_index, row) in rows.iter().enumerate() {
        let y = header_height + row_index as f64 * row_height;
        writeln!(
            file,
            "<text x=\"18\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"12\">{}</text>",
            y + 21.0,
            html_escape(&short_label(row, 30))
        )
        .map_err(write_error(path))?;
        for (col_index, col) in cols.iter().enumerate() {
            let x = row_label_width + col_index as f64 * col_width;
            let value = medians.get(&(row.clone(), col.clone())).copied();
            match value {
                Some(value) => {
                    let ratio = if max_value == 0.0 {
                        0.0
                    } else {
                        (value / max_value).clamp(0.0, 1.0)
                    };
                    let (fill, text) = heatmap_svg_colors(ratio);
                    writeln!(
                        file,
                        "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\"><title>{} / {}: {:.3} GiB/s median</title></rect>",
                        col_width - 3.0,
                        row_height - 3.0,
                        fill,
                        html_escape(row),
                        html_escape(col),
                        value
                    )
                    .map_err(write_error(path))?;
                    writeln!(
                        file,
                        "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"{}\">{:.2}</text>",
                        x + col_width / 2.0,
                        y + 21.0,
                        text,
                        value
                    )
                    .map_err(write_error(path))?;
                }
                None => {
                    writeln!(
                        file,
                        "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"#f1f3f4\"/>",
                        col_width - 3.0,
                        row_height - 3.0
                    )
                    .map_err(write_error(path))?;
                }
            }
        }
    }

    writeln!(file, "</svg>").map_err(write_error(path))?;
    Ok(true)
}

fn write_thread_scaling_svg(path: &Path, records: &[&BenchReportRecord]) -> Result<bool> {
    let mut by_kernel_thread: BTreeMap<String, BTreeMap<usize, Vec<f64>>> = BTreeMap::new();
    let mut thread_set = BTreeSet::new();
    for record in records {
        let Some(threads) = parse_usize(&record.threads) else {
            continue;
        };
        thread_set.insert(threads);
        by_kernel_thread
            .entry(record.kernel.clone())
            .or_default()
            .entry(threads)
            .or_default()
            .push(record.gib_per_s);
    }

    if by_kernel_thread.is_empty() || thread_set.len() < 2 {
        return Ok(false);
    }

    let threads = thread_set.into_iter().collect::<Vec<_>>();
    let width = 1120.0_f64;
    let height = 520.0_f64;
    let left = 72.0_f64;
    let right = 220.0_f64;
    let top = 58.0_f64;
    let bottom = 70.0_f64;
    let plot_width = width - left - right;
    let plot_height = height - top - bottom;
    let mut series = Vec::new();
    let mut max_value = 0.0_f64;
    for (kernel, by_thread) in by_kernel_thread {
        let mut points = Vec::new();
        for thread in &threads {
            if let Some(values) = by_thread.get(thread) {
                let value = median(values.clone());
                max_value = max_value.max(value);
                points.push((*thread, value));
            }
        }
        if points.len() >= 2 {
            series.push((kernel, points));
        }
    }

    if series.is_empty() || max_value == 0.0 {
        return Ok(false);
    }

    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;
    writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"
    )
    .map_err(write_error(path))?;
    write_svg_background(&mut file, path)?;
    writeln!(
        file,
        "<text x=\"18\" y=\"28\" font-family=\"sans-serif\" font-size=\"18\" font-weight=\"700\">Thread Scaling by Kernel</text>"
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"50\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#57606a\">parallel-sequential rows only; y = median GiB/s across matching workloads</text>"
    )
    .map_err(write_error(path))?;
    let area = ChartArea {
        left,
        top,
        width: plot_width,
        height: plot_height,
    };
    write_chart_axes(&mut file, path, area, "threads", "GiB/s")?;

    for (index, thread) in threads.iter().enumerate() {
        let x = categorical_x(index, threads.len(), left, plot_width);
        writeln!(
            file,
            "<text x=\"{x:.1}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">{thread}</text>",
            height - 28.0
        )
        .map_err(write_error(path))?;
    }

    for tick in 0..=4 {
        let value = max_value * tick as f64 / 4.0;
        let y = top + plot_height - (value / max_value) * plot_height;
        writeln!(
            file,
            "<text x=\"{}\" y=\"{:.1}\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\">{value:.1}</text>",
            left - 8.0,
            y + 4.0
        )
        .map_err(write_error(path))?;
    }

    for (series_index, (kernel, points)) in series.iter().enumerate() {
        let color = chart_color(series_index);
        let mut point_strings = Vec::new();
        for (thread, value) in points {
            let thread_index = threads
                .iter()
                .position(|candidate| candidate == thread)
                .unwrap_or(0);
            let x = categorical_x(thread_index, threads.len(), left, plot_width);
            let y = top + plot_height - (value / max_value) * plot_height;
            point_strings.push(format!("{x:.1},{y:.1}"));
        }
        writeln!(
            file,
            "<polyline fill=\"none\" stroke=\"{}\" stroke-width=\"2.5\" points=\"{}\"/>",
            color,
            point_strings.join(" ")
        )
        .map_err(write_error(path))?;
        for (thread, value) in points {
            let thread_index = threads
                .iter()
                .position(|candidate| candidate == thread)
                .unwrap_or(0);
            let x = categorical_x(thread_index, threads.len(), left, plot_width);
            let y = top + plot_height - (value / max_value) * plot_height;
            writeln!(
                file,
                "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"4\" fill=\"{}\"><title>{}: threads={} median {:.3} GiB/s</title></circle>",
                color,
                html_escape(kernel),
                thread,
                value
            )
            .map_err(write_error(path))?;
        }
        if let Some((last_thread, last_value)) = points.last() {
            let thread_index = threads
                .iter()
                .position(|candidate| candidate == last_thread)
                .unwrap_or(0);
            let x = categorical_x(thread_index, threads.len(), left, plot_width) + 8.0;
            let y = top + plot_height - (last_value / max_value) * plot_height + 4.0;
            writeln!(
                file,
                "<text x=\"{x:.1}\" y=\"{y:.1}\" font-family=\"sans-serif\" font-size=\"11\" fill=\"{}\">{}</text>",
                color,
                html_escape(&short_label(kernel, 28))
            )
            .map_err(write_error(path))?;
        }
    }

    writeln!(file, "</svg>").map_err(write_error(path))?;
    Ok(true)
}

fn write_winner_counts_svg(path: &Path, records: &[BenchReportRecord]) -> Result<bool> {
    let counts = winner_counts(records);
    if counts.is_empty() {
        return Ok(false);
    }
    let bars = counts
        .into_iter()
        .map(|(kernel, count)| (kernel, count as f64, format!("{count} wins")))
        .collect::<Vec<_>>();
    write_bar_chart_svg(
        path,
        "Winner Counts by Kernel",
        "count of workload rows where kernel had max GiB/s",
        &bars,
    )?;
    Ok(true)
}

fn write_planner_confusion_svg(path: &Path, records: &[BenchReportRecord]) -> Result<bool> {
    let mut workloads: BTreeMap<String, Vec<&BenchReportRecord>> = BTreeMap::new();
    for record in records {
        workloads
            .entry(workload_label(record))
            .or_default()
            .push(record);
    }

    let mut cells: BTreeMap<(String, String), f64> = BTreeMap::new();
    let mut planned_values = BTreeSet::new();
    let mut best_values = BTreeSet::new();
    for rows in workloads.values() {
        let planned = rows
            .iter()
            .find(|row| !row.planned_kernel.is_empty())
            .map_or("no-plan", |row| row.planned_kernel.as_str());
        let Some(best) = rows.iter().max_by(|left, right| {
            left.gib_per_s
                .partial_cmp(&right.gib_per_s)
                .unwrap_or(Ordering::Equal)
        }) else {
            continue;
        };
        let planned = dimension_value(planned);
        let best = dimension_value(&best.kernel);
        planned_values.insert(planned.clone());
        best_values.insert(best.clone());
        *cells.entry((planned, best)).or_default() += 1.0;
    }

    if planned_values.is_empty() || best_values.is_empty() {
        return Ok(false);
    }

    write_count_heatmap_svg(
        path,
        "Planner vs Best Kernel",
        "Planner",
        "Best measured kernel",
        planned_values.into_iter().collect(),
        best_values.into_iter().collect(),
        cells,
    )?;
    Ok(true)
}

fn write_planner_gap_svg(path: &Path, records: &[BenchReportRecord]) -> Result<bool> {
    let mut workloads: BTreeMap<String, Vec<&BenchReportRecord>> = BTreeMap::new();
    for record in records {
        workloads
            .entry(workload_label(record))
            .or_default()
            .push(record);
    }

    let mut bars = Vec::new();
    for (workload, rows) in workloads {
        let planned = rows
            .iter()
            .find(|row| !row.planned_kernel.is_empty())
            .map_or("", |row| row.planned_kernel.as_str());
        if planned.is_empty() {
            continue;
        }
        let Some(best) = rows.iter().max_by(|left, right| {
            left.gib_per_s
                .partial_cmp(&right.gib_per_s)
                .unwrap_or(Ordering::Equal)
        }) else {
            continue;
        };
        let Some(planned_row) = rows.iter().find(|row| row.kernel == planned) else {
            bars.push((
                format!(
                    "{workload} | planned {planned} missing, winner {}",
                    best.kernel
                ),
                best.gib_per_s,
                "planned kernel was not measured".to_owned(),
            ));
            continue;
        };
        if best.kernel == planned_row.kernel {
            continue;
        }
        let gap_pct = if planned_row.gib_per_s > 0.0 {
            (best.gib_per_s - planned_row.gib_per_s) / planned_row.gib_per_s * 100.0
        } else {
            0.0
        };
        bars.push((
            format!(
                "{workload} | planner {} -> winner {}",
                planned_row.kernel, best.kernel
            ),
            gap_pct.max(0.0),
            format!(
                "{gap_pct:.1}% gap; planned {:.2} GiB/s, winner {:.2} GiB/s",
                planned_row.gib_per_s, best.gib_per_s
            ),
        ));
    }

    if bars.is_empty() {
        return Ok(false);
    }

    bars.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    bars.truncate(30);
    write_bar_chart_svg(
        path,
        "Largest Planner Throughput Gaps",
        "top planner misses by percent gap versus measured winner",
        &bars,
    )?;
    Ok(true)
}

fn write_count_heatmap_svg(
    path: &Path,
    title: &str,
    row_axis: &str,
    col_axis: &str,
    rows: Vec<String>,
    cols: Vec<String>,
    cells: BTreeMap<(String, String), f64>,
) -> Result<()> {
    let rows = sort_dimension_values(rows);
    let cols = sort_dimension_values(cols);
    let max_value = cells.values().copied().fold(0.0_f64, f64::max);
    let row_label_width = axis_label_width(&rows);
    let col_width = axis_label_width(&cols).clamp(132.0, 190.0);
    let row_height = 34.0_f64;
    let header_height = 82.0_f64;
    let width = row_label_width + col_width * cols.len() as f64 + 36.0;
    let height = header_height + row_height * rows.len() as f64 + 50.0;
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;

    writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width:.0}\" height=\"{height:.0}\" viewBox=\"0 0 {width:.0} {height:.0}\">"
    )
    .map_err(write_error(path))?;
    write_svg_background(&mut file, path)?;
    writeln!(
        file,
        "<text x=\"18\" y=\"28\" font-family=\"sans-serif\" font-size=\"18\" font-weight=\"700\">{}</text>",
        html_escape(title)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"50\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#57606a\">cell value = number of workload rows</text>"
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"74\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"700\">{}</text>",
        html_escape(row_axis)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"{:.1}\" y=\"74\" font-family=\"sans-serif\" font-size=\"12\" font-weight=\"700\">{}</text>",
        row_label_width,
        html_escape(col_axis)
    )
    .map_err(write_error(path))?;

    for (index, col) in cols.iter().enumerate() {
        let x = row_label_width + index as f64 * col_width + col_width / 2.0;
        writeln!(
            file,
            "<text x=\"{x:.1}\" y=\"74\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">{}</text>",
            html_escape(&short_label(col, 22))
        )
        .map_err(write_error(path))?;
    }

    for (row_index, row) in rows.iter().enumerate() {
        let y = header_height + row_index as f64 * row_height;
        writeln!(
            file,
            "<text x=\"18\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"12\">{}</text>",
            y + 21.0,
            html_escape(&short_label(row, 30))
        )
        .map_err(write_error(path))?;
        for (col_index, col) in cols.iter().enumerate() {
            let x = row_label_width + col_index as f64 * col_width;
            let value = cells
                .get(&(row.clone(), col.clone()))
                .copied()
                .unwrap_or(0.0);
            let ratio = if max_value == 0.0 {
                0.0
            } else {
                (value / max_value).clamp(0.0, 1.0)
            };
            let (fill, text) = heatmap_svg_colors(ratio);
            writeln!(
                file,
                "<rect x=\"{x:.1}\" y=\"{y:.1}\" width=\"{:.1}\" height=\"{:.1}\" fill=\"{}\"><title>{} -> {}: {:.0}</title></rect>",
                col_width - 3.0,
                row_height - 3.0,
                fill,
                html_escape(row),
                html_escape(col),
                value
            )
            .map_err(write_error(path))?;
            writeln!(
                file,
                "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"{}\">{:.0}</text>",
                x + col_width / 2.0,
                y + 21.0,
                text,
                value
            )
            .map_err(write_error(path))?;
        }
    }

    writeln!(file, "</svg>").map_err(write_error(path))?;
    Ok(())
}

fn write_bar_chart_svg(
    path: &Path,
    title: &str,
    subtitle: &str,
    bars: &[(String, f64, String)],
) -> Result<()> {
    let width = 960.0_f64;
    let bar_height = 30.0_f64;
    let top = 62.0_f64;
    let left = 250.0_f64;
    let plot_width = 650.0_f64;
    let height = top + bars.len() as f64 * bar_height + 42.0;
    let max_value = bars
        .iter()
        .fold(0.0_f64, |max, (_, value, _)| max.max(*value));
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;

    writeln!(
        file,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height:.0}\" viewBox=\"0 0 {width} {height:.0}\">"
    )
    .map_err(write_error(path))?;
    write_svg_background(&mut file, path)?;
    writeln!(
        file,
        "<text x=\"18\" y=\"28\" font-family=\"sans-serif\" font-size=\"18\" font-weight=\"700\">{}</text>",
        html_escape(title)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"50\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#57606a\">{}</text>",
        html_escape(subtitle)
    )
    .map_err(write_error(path))?;

    for (index, (label, value, value_label)) in bars.iter().enumerate() {
        let y = top + index as f64 * bar_height;
        let ratio = if max_value == 0.0 {
            0.0
        } else {
            value / max_value
        };
        let width = plot_width * ratio;
        writeln!(
            file,
            "<text x=\"18\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"12\">{}</text>",
            y + 19.0,
            html_escape(&short_label(label, 34))
        )
        .map_err(write_error(path))?;
        writeln!(
            file,
            "<rect x=\"{left}\" y=\"{:.1}\" width=\"{width:.1}\" height=\"22\" fill=\"#2f7ed8\"><title>{}: {}</title></rect>",
            y + 4.0,
            html_escape(label),
            html_escape(value_label)
        )
        .map_err(write_error(path))?;
        writeln!(
            file,
            "<text x=\"{:.1}\" y=\"{:.1}\" font-family=\"sans-serif\" font-size=\"12\">{}</text>",
            left + width + 8.0,
            y + 20.0,
            html_escape(value_label)
        )
        .map_err(write_error(path))?;
    }

    writeln!(file, "</svg>").map_err(write_error(path))?;
    Ok(())
}

fn winner_counts(records: &[BenchReportRecord]) -> Vec<(String, usize)> {
    let mut workloads: BTreeMap<String, Vec<&BenchReportRecord>> = BTreeMap::new();
    for record in records {
        workloads
            .entry(workload_label(record))
            .or_default()
            .push(record);
    }

    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for rows in workloads.values() {
        if let Some(best) = rows.iter().max_by(|left, right| {
            left.gib_per_s
                .partial_cmp(&right.gib_per_s)
                .unwrap_or(Ordering::Equal)
        }) {
            *counts.entry(best.kernel.clone()).or_default() += 1;
        }
    }

    let mut counts = counts.into_iter().collect::<Vec<_>>();
    counts.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    counts
}

#[allow(clippy::too_many_arguments)]
fn write_bench_report_summary(
    path: &Path,
    records: &[BenchReportRecord],
    source: &Path,
    timing_csv: &Path,
    planner_parity_csv: Option<&Path>,
    heatmap_html: &Path,
    histogram_svg: &Path,
    visual_artifacts: &[ReportArtifact],
) -> Result<()> {
    let mut file = File::create(path)
        .map_err(|error| format!("failed to create `{}`: {error}", path.display()))?;
    let workload_count = records
        .iter()
        .map(workload_label)
        .collect::<BTreeSet<_>>()
        .len();
    let kernel_count = records
        .iter()
        .map(|record| record.kernel.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    let fastest = records.iter().max_by(|left, right| {
        left.gib_per_s
            .partial_cmp(&right.gib_per_s)
            .unwrap_or(Ordering::Equal)
    });

    writeln!(file, "# Benchmark Report").map_err(write_error(path))?;
    writeln!(file).map_err(write_error(path))?;
    writeln!(file, "- source: `{}`", source.display()).map_err(write_error(path))?;
    writeln!(file, "- records: `{}`", records.len()).map_err(write_error(path))?;
    writeln!(file, "- workloads: `{workload_count}`").map_err(write_error(path))?;
    writeln!(file, "- kernels: `{kernel_count}`").map_err(write_error(path))?;
    if let Some(fastest) = fastest {
        writeln!(
            file,
            "- fastest: `{}` at `{:.2} GiB/s`",
            fastest.full_id, fastest.gib_per_s
        )
        .map_err(write_error(path))?;
    }
    writeln!(file).map_err(write_error(path))?;
    writeln!(file, "## Artifacts").map_err(write_error(path))?;
    writeln!(file).map_err(write_error(path))?;
    writeln!(file, "- timing_csv: `{}`", timing_csv.display()).map_err(write_error(path))?;
    if let Some(planner_parity_csv) = planner_parity_csv {
        writeln!(
            file,
            "- planner_parity_csv: `{}`",
            planner_parity_csv.display()
        )
        .map_err(write_error(path))?;
    }
    writeln!(file, "- heatmap_html: `{}`", heatmap_html.display()).map_err(write_error(path))?;
    writeln!(
        file,
        "- throughput_histogram_svg: `{}`",
        histogram_svg.display()
    )
    .map_err(write_error(path))?;
    writeln!(file, "- visual_count: `{}`", visual_artifacts.len()).map_err(write_error(path))?;
    for artifact in visual_artifacts {
        writeln!(file, "  - `{}`: {}", artifact.file_name, artifact.title)
            .map_err(write_error(path))?;
    }
    writeln!(file).map_err(write_error(path))?;
    writeln!(file, "## Top Throughput").map_err(write_error(path))?;
    writeln!(file).map_err(write_error(path))?;
    writeln!(file, "| Kernel | Workload | GiB/s | Mean |").map_err(write_error(path))?;
    writeln!(file, "|---|---|---:|---:|").map_err(write_error(path))?;

    let mut rows = records.iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .gib_per_s
            .partial_cmp(&left.gib_per_s)
            .unwrap_or(Ordering::Equal)
    });
    for record in rows.iter().take(20) {
        writeln!(
            file,
            "| {} | {} | {:.2} | {} |",
            md_cell(&record.kernel),
            md_cell(&workload_label(record)),
            record.gib_per_s,
            format_duration_ns(record.mean_ns)
        )
        .map_err(write_error(path))?;
    }

    Ok(())
}

fn bench_field(value: &Value, key: &str) -> String {
    json_nested_string(value, "metadata", key)
        .or_else(|| json_nested_string(value, "manifest", key))
        .unwrap_or_default()
}

fn bench_field_any(value: &Value, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| {
            json_nested_string(value, "metadata", key)
                .or_else(|| json_nested_string(value, "manifest", key))
        })
        .unwrap_or_default()
}

fn json_string(value: &Value, key: &str) -> Option<String> {
    value.get(key).map(value_to_summary_string)
}

fn json_nested_string(value: &Value, object: &str, key: &str) -> Option<String> {
    value
        .get(object)
        .and_then(|nested| nested.get(key))
        .map(value_to_summary_string)
}

fn workload_label(record: &BenchReportRecord) -> String {
    if !record.primitive.is_empty() {
        return format!(
            "{} | {} | {} | bytes={}",
            empty_as_unknown(&record.primitive),
            empty_as_unknown(&record.case),
            empty_as_unknown(&record.pattern),
            empty_as_unknown(&record.bytes),
        );
    }

    format!(
        "{} | {} | {} | chunk={} | threads={}",
        empty_as_unknown(&record.case),
        empty_as_unknown(&record.access),
        empty_as_unknown(&record.pattern),
        empty_as_unknown(&record.chunk),
        empty_as_unknown(&record.threads)
    )
}

fn empty_as_unknown(value: &str) -> &str {
    if value.is_empty() { "?" } else { value }
}

fn csv_cell(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn html_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn local_href(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_owned()
}

fn dimension_value(value: &str) -> String {
    let value = value.trim();
    if value.is_empty() {
        "?".into()
    } else {
        value.into()
    }
}

fn parse_usize(value: &str) -> Option<usize> {
    value.trim().parse::<usize>().ok()
}

fn sort_dimension_values(mut values: Vec<String>) -> Vec<String> {
    let numeric = values.iter().all(|value| value.parse::<f64>().is_ok());
    if numeric {
        values.sort_by(|left, right| {
            left.parse::<f64>()
                .unwrap_or(0.0)
                .partial_cmp(&right.parse::<f64>().unwrap_or(0.0))
                .unwrap_or(Ordering::Equal)
        });
    } else {
        values.sort();
    }
    values
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn axis_label_width(values: &[String]) -> f64 {
    let max_chars = values
        .iter()
        .map(|value| value.chars().count())
        .max()
        .unwrap_or(8);
    ((max_chars as f64 * 7.0) + 32.0).clamp(120.0, 300.0)
}

fn short_label(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_owned();
    }
    let mut label = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    label.push_str("...");
    label
}

fn heatmap_svg_colors(ratio: f64) -> (String, &'static str) {
    let red = (230.0 * (1.0 - ratio) + 38.0 * ratio).round() as u8;
    let green = (75.0 * (1.0 - ratio) + 166.0 * ratio).round() as u8;
    let blue = (64.0 * (1.0 - ratio) + 91.0 * ratio).round() as u8;
    let text = if ratio < 0.45 { "#fff" } else { "#111" };
    (format!("rgb({red},{green},{blue})"), text)
}

fn write_svg_background(file: &mut File, path: &Path) -> Result<()> {
    writeln!(
        file,
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
    )
    .map_err(write_error(path))
}

#[derive(Clone, Copy)]
struct ChartArea {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
}

fn write_chart_axes(
    file: &mut File,
    path: &Path,
    area: ChartArea,
    x_label: &str,
    y_label: &str,
) -> Result<()> {
    writeln!(
        file,
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\"/>",
        area.left,
        area.top + area.height,
        area.left + area.width,
        area.top + area.height
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<line x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"#333\"/>",
        area.left,
        area.top,
        area.left,
        area.top + area.height
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"{}\" y=\"{}\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">{}</text>",
        area.left + area.width / 2.0,
        area.top + area.height + 52.0,
        html_escape(x_label)
    )
    .map_err(write_error(path))?;
    writeln!(
        file,
        "<text x=\"18\" y=\"{}\" font-family=\"sans-serif\" font-size=\"12\" transform=\"rotate(-90 18 {})\">{}</text>",
        area.top + 56.0,
        area.top + 56.0,
        html_escape(y_label)
    )
    .map_err(write_error(path))?;
    Ok(())
}

fn categorical_x(index: usize, count: usize, left: f64, width: f64) -> f64 {
    if count <= 1 {
        left + width / 2.0
    } else {
        left + width * index as f64 / (count - 1) as f64
    }
}

fn chart_color(index: usize) -> &'static str {
    const COLORS: [&str; 10] = [
        "#0969da", "#cf222e", "#1a7f37", "#8250df", "#bf8700", "#0a8080", "#d1248f", "#57606a",
        "#953800", "#116329",
    ];
    COLORS[index % COLORS.len()]
}

fn md_cell(value: &str) -> String {
    value.replace('|', "\\|")
}

fn heatmap_css() -> &'static str {
    "body{font-family:system-ui,sans-serif;margin:24px;color:#202124}\
     nav{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 18px}\
     nav a{color:#0969da;text-decoration:none}\
     nav a:hover{text-decoration:underline}\
     .guide{max-width:1100px;background:#f6f8fa;border:1px solid #d0d7de;padding:12px 14px;margin:14px 0}\
     .guide h2{font-size:16px;margin:0 0 8px}\
     .guide p{margin:6px 0;line-height:1.4}\
     figure{margin:18px 0 22px;max-width:980px}\
     figure img{display:block;max-width:100%;border:1px solid #d0d7de}\
     figure h3{font-size:14px;margin:0 0 8px}\
     figcaption{font-size:12px;color:#57606a;margin-top:6px}\
     .gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;align-items:start}\
     .gallery figure{margin:0;max-width:none}\
     h2{font-size:18px;margin-top:22px}\
     table{border-collapse:collapse;font-size:12px}\
     th,td{border:1px solid #d0d7de;padding:4px 6px;white-space:nowrap}\
     thead th{position:sticky;top:0;background:#f6f8fa;z-index:2}\
     tbody th{position:sticky;left:0;background:#f6f8fa;text-align:left;z-index:1}\
     td.metric{text-align:right;font-variant-numeric:tabular-nums}\
     td.winner{font-weight:700;outline:2px solid #1a7f37;outline-offset:-2px}\
     .cell-value{display:block}\
     .winner-label{display:block;text-transform:uppercase;font-size:9px;letter-spacing:.04em}\
     td.missing{background:#f1f3f4}"
}

fn heatmap_cell_style(ratio: f64) -> String {
    let red = (230.0 * (1.0 - ratio) + 38.0 * ratio).round() as u8;
    let green = (75.0 * (1.0 - ratio) + 166.0 * ratio).round() as u8;
    let blue = (64.0 * (1.0 - ratio) + 91.0 * ratio).round() as u8;
    let color = if ratio < 0.45 { "#fff" } else { "#111" };
    format!("background-color:rgb({red},{green},{blue});color:{color}")
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

fn default_or_extra<const N: usize>(extra: &[OsString], default_args: [&str; N]) -> Vec<OsString> {
    if extra.is_empty() {
        cargo_args(default_args)
    } else {
        extra.to_vec()
    }
}

fn path_or_default(args: &[OsString], default: &str) -> Result<(PathBuf, Vec<OsString>)> {
    let (path, extra) = if let Some(first) = args.first() {
        if !first.to_string_lossy().starts_with('-') {
            (PathBuf::from(first), args[1..].to_vec())
        } else {
            (expand_tilde(default), args.to_vec())
        }
    } else {
        (expand_tilde(default), Vec::new())
    };

    if path.exists() {
        Ok((path, extra))
    } else {
        Err(format!(
            "benchmark data `{}` does not exist",
            path.display()
        ))
    }
}

fn path_or_first_existing(
    args: &[OsString],
    defaults: &[&str],
) -> Result<(PathBuf, Vec<OsString>)> {
    if let Some(first) = args.first()
        && !first.to_string_lossy().starts_with('-')
    {
        let path = PathBuf::from(first);
        if path.exists() {
            return Ok((path, args[1..].to_vec()));
        }
        return Err(format!(
            "benchmark data `{}` does not exist",
            path.display()
        ));
    }

    for default in defaults {
        let path = expand_tilde(default);
        if path.exists() {
            return Ok((path, args.to_vec()));
        }
    }

    Err(format!(
        "none of the default benchmark data paths exist: {}",
        defaults.join(", ")
    ))
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = env::var_os("HOME")
    {
        return PathBuf::from(home).join(rest);
    }
    PathBuf::from(path)
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

struct MagicBpeCalibrationCandidate {
    key: u64,
    sample_relpath: String,
    mime: String,
}

struct MagicBpeAggregate {
    mime: String,
    samples: u64,
    bytes: u64,
    counts: [u64; 256],
}

impl MagicBpeAggregate {
    fn new(mime: String) -> Self {
        Self {
            mime,
            samples: 0,
            bytes: 0,
            counts: [0; 256],
        }
    }
}

fn magic_bpe_candidates(
    index: &Path,
    seed: u64,
    shuffle: bool,
) -> Result<Vec<MagicBpeCalibrationCandidate>> {
    let file = File::open(index)
        .map_err(|error| format!("failed to open `{}`: {error}", index.display()))?;
    let mut candidates = Vec::new();

    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| {
            format!(
                "failed to read line {} from `{}`: {error}",
                line_index + 1,
                index.display()
            )
        })?;
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
        let original_path = value.get("path").and_then(Value::as_str).unwrap_or("");
        let key = if shuffle {
            mix_u64(
                seed ^ line_index as u64 ^ stable_str_hash(mime) ^ stable_str_hash(original_path),
            )
        } else {
            line_index as u64
        };

        candidates.push(MagicBpeCalibrationCandidate {
            key,
            sample_relpath: sample_relpath.to_owned(),
            mime: mime.to_owned(),
        });
    }

    Ok(candidates)
}

fn entropy_from_counts(counts: &[u64; 256]) -> f64 {
    let total = counts.iter().copied().sum::<u64>();
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    counts
        .iter()
        .copied()
        .filter(|count| *count != 0)
        .map(|count| {
            let p = count as f64 / total_f;
            -p * p.log2()
        })
        .sum()
}

fn env_usize(name: &str) -> Option<usize> {
    env::var(name).ok().and_then(|value| value.parse().ok())
}

fn env_bool(name: &str) -> Option<bool> {
    env::var(name).ok().map(|value| {
        matches!(
            value.to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on" | "shuffle"
        )
    })
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

fn profile_output_path(kind: &str, extension: &str) -> PathBuf {
    Path::new("target/profiles").join(format!("{}-{kind}.{extension}", unix_timestamp()))
}

fn criterion_args(args: &[OsString]) -> Vec<OsString> {
    if args.first().is_some_and(|arg| arg == "--") {
        args[1..].to_vec()
    } else {
        args.to_vec()
    }
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
           bench-smoke\n\
                    fast workload-matrix sanity suite\n\
           bench-synthetic-full\n\
                    full synthetic matrix with all kernels and topology threads\n\
           bench-real-iso [path]\n\
                    Ubuntu ISO matrix with all kernels and topology threads\n\
           bench-real-f21 [path]\n\
                    F21/F22/rootfs paper-data calibration matrix\n\
           bench-size-sweep\n\
                    powers-of-two payload-size sweep\n\
           bench-alignment-sweep\n\
                    pointer-alignment sweep at +0/+1/+3/+7/+31 offsets\n\
           bench-thread-topology\n\
                    2/4/physical/logical/saturated thread sweep\n\
           bench-planner-parity\n\
                    planner-vs-all-kernels matrix with no hidden direct kernels\n\
           bench-planner-parity-real [path]\n\
                    planner-vs-all-kernels matrix including F21/F22/rootfs data\n\
           bench-real-magic-bpe [path]\n\
                    optional Magic-BPE processed-index matrix; limit/shuffle via TOKENFS_ALGOS_MAGIC_BPE_*\n\
           bench-cache-hot-cold\n\
                    hot-repeat, cold-sweep, and same-file-repeat access patterns\n\
           bench-profile\n\
                    perf profile over workload matrix when available\n\
           bench-primitives\n\
                    isolated primitive matrix with HTML/SVG-reportable metadata\n\
           bench-primitives-real\n\
                    primitive matrix including default ISO/F22/repo real-file slices\n\
           bench-histogram-primitive\n\
                    isolated histogram primitive benchmarks\n\
           bench-histogram-primitive-real\n\
                    isolated histogram primitive benchmarks with real-file slices\n\
           bench-fingerprint\n\
                    isolated fingerprint primitive benchmarks\n\
           bench-sketch\n\
                    isolated sketch primitive benchmarks\n\
           bench-byteclass\n\
                    isolated byte-class primitive benchmarks\n\
           bench-byteclass-real\n\
                    isolated byte-class primitive benchmarks with real-file slices\n\
           bench-runlength\n\
                    isolated run-length primitive benchmarks\n\
           bench-entropy\n\
                    isolated entropy primitive benchmarks\n\
           bench-divergence\n\
                    isolated byte-distribution divergence benchmarks\n\
           bench-distribution\n\
                    isolated calibrated byte-distribution nearest-reference benchmarks\n\
           bench-distribution-real\n\
                    distribution benchmarks with real-file slices\n\
           bench-ngram-sketch\n\
                    isolated 2-gram/4-gram hash-bin sketch benchmarks\n\
           bench-ngram-sketch-real\n\
                    n-gram sketch benchmarks with real-file slices\n\
           bench-selector\n\
                    isolated selector-signal benchmarks\n\
           bench-compare <old.jsonl> <new.jsonl>\n\
                    compare two benchmark history JSONL runs\n\
           bench-report [run.jsonl]\n\
                    generate heatmap, histogram, and timing-table artifacts\n\
           bench-log\n\
                    log the current target/criterion results to benchmark history\n\
           calibrate-magic-bpe [path]\n\
                    write MIME-grouped byte-histogram calibration JSONL from Magic-BPE samples\n\
           profile  benchmark under perf when available\n\
           profile-real [path]\n\
                    real-data benchmark under perf when available\n\
           profile-flamegraph\n\
                    generate a flamegraph SVG with cargo-flamegraph\n\
           profile-flamegraph-real [path]\n\
                    real-data flamegraph SVG with cargo-flamegraph\n\
           profile-primitives\n\
                    primitive benchmark under perf when available\n\
           profile-primitives-real\n\
                    primitive benchmark with real-file slices under perf when available\n\
           profile-primitives-flamegraph\n\
                    primitive flamegraph SVG with cargo-flamegraph\n\
           profile-primitives-flamegraph-real\n\
                    primitive flamegraph SVG with real-file slices\n\
           ci       local CI gate"
    );
}

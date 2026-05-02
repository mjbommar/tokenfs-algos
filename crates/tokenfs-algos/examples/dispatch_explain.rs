//! Print detected processor facts and histogram planner examples.

use tokenfs_algos::dispatch::{
    ApiContext, CacheState, ContentKind, EntropyClass, EntropyScale, ReadPattern, SourceHint,
    WorkloadShape, detected_processor_profile, histogram_kernel_catalog, plan_histogram,
};

fn main() {
    let profile = detected_processor_profile();

    println!("processor");
    println!("  backend: {:?}", profile.backend);
    println!("  logical_cpus: {}", format_option(profile.logical_cpus));
    println!(
        "  cache: line={} l1d={} l2={} l3={}",
        format_bytes(profile.cache.line_bytes),
        format_bytes(profile.cache.l1d_bytes),
        format_bytes(profile.cache.l2_bytes),
        format_bytes(profile.cache.l3_bytes),
    );
    println!(
        "  accelerators: amx_tile={} amx_int8={} amx_bf16={} sme={} sme2={}",
        profile.accelerators.amx_tile,
        profile.accelerators.amx_int8,
        profile.accelerators.amx_bf16,
        profile.accelerators.sme,
        profile.accelerators.sme2,
    );
    println!(
        "    summary: any_amx={} any_sme={}",
        profile.accelerators.has_any_amx(),
        profile.accelerators.has_any_sme(),
    );
    println!();

    println!("histogram kernel catalog");
    for info in histogram_kernel_catalog() {
        println!(
            "  {:>24} min={:>7} chunk={:>7} sample={:>5} {:?} {:?}",
            info.strategy.as_str(),
            format_bytes(Some(info.min_bytes)),
            format_bytes(nonzero(info.preferred_chunk_bytes)),
            format_bytes(nonzero(info.sample_bytes)),
            info.working_set,
            info.statefulness,
        );
    }
    println!();

    println!("planner examples");
    for workload in example_workloads() {
        let (plan, trace) =
            tokenfs_algos::dispatch::planner::plan_histogram_traced(&profile, &workload);
        println!(
            "  {:?} total={} chunk={} threads={} scale={:?} entropy={:?} => {} chunk={} sample={}",
            workload.context,
            format_bytes(Some(workload.total_bytes)),
            format_bytes(nonzero(workload.chunk_bytes)),
            workload.threads,
            workload.scale,
            workload.entropy,
            plan.strategy.as_str(),
            format_bytes(nonzero(plan.chunk_bytes)),
            format_bytes(nonzero(plan.sample_bytes)),
        );
        println!(
            "    rule={} confidence={}/255 source={} reason={}",
            trace.last().map(|d| d.name).unwrap_or("?"),
            plan.confidence_q8,
            plan.confidence_source.as_str(),
            plan.reason,
        );
        if trace.len() > 1 {
            println!(
                "    walked {} rules; misses: {}",
                trace.len(),
                trace[..trace.len() - 1]
                    .iter()
                    .map(|d| d.name)
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }
    }

    // Silence the unused-import warning on `plan_histogram` — we use the
    // traced variant in this example, but leave the simple import in
    // place because it's the public surface most users will touch.
    let _ = plan_histogram;
}

fn example_workloads() -> [WorkloadShape; 5] {
    [
        shape(
            ApiContext::Random,
            ContentKind::Binary,
            EntropyClass::Mixed,
            EntropyScale::Micro,
            16 * 1024,
            1,
            1,
        ),
        shape(
            ApiContext::Sequential,
            ContentKind::Text,
            EntropyClass::Medium,
            EntropyScale::Micro,
            1024 * 1024,
            4 * 1024,
            1,
        ),
        shape(
            ApiContext::Block,
            ContentKind::Binary,
            EntropyClass::High,
            EntropyScale::Flat,
            1024 * 1024,
            0,
            1,
        ),
        shape(
            ApiContext::File,
            ContentKind::Mixed,
            EntropyClass::Mixed,
            EntropyScale::Macro,
            16 * 1024 * 1024,
            64 * 1024,
            1,
        ),
        shape(
            ApiContext::Parallel,
            ContentKind::Binary,
            EntropyClass::High,
            EntropyScale::Flat,
            64 * 1024 * 1024,
            64 * 1024,
            4,
        ),
    ]
}

fn shape(
    context: ApiContext,
    content: ContentKind,
    entropy: EntropyClass,
    scale: EntropyScale,
    total_bytes: usize,
    chunk_bytes: usize,
    threads: usize,
) -> WorkloadShape {
    WorkloadShape {
        context,
        read_pattern: ReadPattern::from_context(context),
        content,
        entropy,
        scale,
        total_bytes,
        chunk_bytes,
        threads,
        alignment_offset: 0,
        cache_state: CacheState::Unknown,
        source_hint: SourceHint::Unknown,
    }
}

fn nonzero(value: usize) -> Option<usize> {
    if value == 0 { None } else { Some(value) }
}

fn format_option(value: Option<usize>) -> String {
    value.map_or_else(|| "unknown".into(), |value| value.to_string())
}

fn format_bytes(value: Option<usize>) -> String {
    let Some(value) = value else {
        return "-".into();
    };

    if value >= 1024 * 1024 && value % (1024 * 1024) == 0 {
        format!("{}MiB", value / (1024 * 1024))
    } else if value >= 1024 && value % 1024 == 0 {
        format!("{}KiB", value / 1024)
    } else {
        value.to_string()
    }
}

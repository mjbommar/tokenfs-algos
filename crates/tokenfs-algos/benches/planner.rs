//! Planner microbench. Confirms that the rule-table dispatch is not a
//! regression vs. the legacy if/else chain.
//!
//! Run: `cargo bench -p tokenfs-algos --bench planner`

#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tokenfs_algos::dispatch::{
    ApiContext, Backend, CacheState, ContentKind, EntropyClass, EntropyScale, ProcessorProfile,
    ReadPattern, SourceHint, WorkloadShape, plan_histogram,
};

fn workload_micro_random() -> (ProcessorProfile, WorkloadShape) {
    let profile = ProcessorProfile::portable();
    let mut workload = WorkloadShape::new(ApiContext::Random, 16 * 1024);
    workload.chunk_bytes = 1;
    (profile, workload)
}

fn workload_paper_extent() -> (ProcessorProfile, WorkloadShape) {
    let profile = ProcessorProfile::portable();
    let mut workload = WorkloadShape::new(ApiContext::Random, 1024 * 1024);
    workload.chunk_bytes = 4 * 1024;
    workload.content = ContentKind::Binary;
    workload.entropy = EntropyClass::Mixed;
    workload.scale = EntropyScale::Macro;
    workload.source_hint = SourceHint::PaperExtent;
    workload.read_pattern = ReadPattern::Random;
    (profile, workload)
}

fn workload_parallel_avx2_macro() -> (ProcessorProfile, WorkloadShape) {
    let profile = ProcessorProfile {
        backend: Backend::Avx2,
        logical_cpus: Some(8),
        ..ProcessorProfile::portable()
    };
    let mut workload = WorkloadShape::new(ApiContext::Parallel, 64 * 1024 * 1024);
    workload.read_pattern = ReadPattern::ParallelSequential;
    workload.content = ContentKind::Binary;
    workload.entropy = EntropyClass::Mixed;
    workload.scale = EntropyScale::Macro;
    workload.threads = 4;
    (profile, workload)
}

fn workload_fallback() -> (ProcessorProfile, WorkloadShape) {
    let profile = ProcessorProfile::portable();
    let workload = WorkloadShape::new(ApiContext::Block, 32 * 1024);
    (profile, workload)
}

fn workload_text_large() -> (ProcessorProfile, WorkloadShape) {
    let profile = ProcessorProfile::portable();
    let mut workload = WorkloadShape::new(ApiContext::Block, 32 * 1024 * 1024);
    workload.content = ContentKind::Text;
    workload.entropy = EntropyClass::Medium;
    workload.scale = EntropyScale::Macro;
    workload.cache_state = CacheState::Cold;
    (profile, workload)
}

fn bench_plan_histogram(c: &mut Criterion) {
    for (name, (profile, workload)) in [
        ("micro-random", workload_micro_random()),
        ("paper-extent-random-4k", workload_paper_extent()),
        ("parallel-avx2-macro-mixed", workload_parallel_avx2_macro()),
        ("fallback-32k-block", workload_fallback()),
        ("text-large", workload_text_large()),
    ] {
        c.bench_function(&format!("plan_histogram/{name}"), |b| {
            b.iter(|| plan_histogram(black_box(&profile), black_box(&workload)));
        });
    }
}

criterion_group!(benches, bench_plan_histogram);
criterion_main!(benches);

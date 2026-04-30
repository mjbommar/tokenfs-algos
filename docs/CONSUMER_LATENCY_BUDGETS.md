# Consumer Latency Budgets

Date: 2026-04-30.

The same primitive library serves several consumers with different latency
tolerance. Planner decisions should be judged against the consumer context, not
only by peak GiB/s on one benchmark.

## Consumers

| Consumer | Typical call shape | Initial budget | Planner implication |
|---|---|---:|---|
| FUSE read path | 4 KiB to 64 KiB sequential reads. | <= 5 us per 4 KiB read when on the hot path. | Avoid full-read classifiers; reuse stream/file state when available. |
| Kernel-adjacent module | Page-sized or extent-sized buffers. | <= 1 us fixed overhead before scanning. | No heap allocation, no sysfs lookup, no heavyweight dispatch in the hot call. |
| Batch image build | Many extents over a full image/rootfs. | <= 100 ms per large extent batch. | Per-region planning and richer fingerprints are acceptable if they improve classification. |
| Paper calibration | F21/F22/F23a extents and sidecars. | Reproducibility over latency. | Pinned kernels and stable feature extraction matter more than default ergonomics. |
| Python binding | One Python call over bytes/memoryview. | <= 1 ms call overhead target for small/medium buffers. | Batch work inside Rust; avoid exposing tiny per-block Python loops. |
| Benchmark harness | Synthetic, ISO, parquet/sidecar inputs. | Measurement quality over latency. | Record planner, kernel, processor, cache, commit, and rustc for every run. |

These numbers are starting anchors, not final SLA claims. They should be
tightened with measurements as FUSE/PyO3 consumers come online.

## ABI Direction

The hot Rust ABI should separate three surfaces:

| Surface | Purpose |
|---|---|
| Planned default API | Ergonomic calls such as `histogram::block(bytes)` that choose a strategy. |
| Pinned kernel API | Reproducible calls such as `histogram::kernels::stripe8_u32::block(bytes)`. |
| Stateful file/stream API | Future planner state for sequential reads and large extents. |

FUSE and kernel-adjacent users should use planned or stateful APIs with bounded
overhead. Paper and benchmark users should be able to pin kernels and record
exactly what ran.

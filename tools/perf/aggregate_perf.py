#!/usr/bin/env python3
"""Join per-runner perf TSVs into a single cross-architecture markdown report.

Input
-----
One or more TSV files produced by `cargo run --example bench_compare`.
Each file is expected to start with `# runner=...` header lines followed
by a column-name row and one data row per measurement:

    primitive\tbackend\tpayload_bytes\tmedian_ns\tthroughput_GBps

The runner label is read from the `# runner=...` header.

Output
------
A markdown report on stdout. One section per primitive; within each
section, one row per (backend, payload_bytes), one column per runner.
Each cell shows median ns + GB/s. A "speedup vs scalar" column is
appended for backends other than scalar/scalar-direct-u64.

Suggested invocation in CI:
    python3 tools/perf/aggregate_perf.py perf-*.tsv >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys
from collections import defaultdict
from typing import Iterable


@dataclasses.dataclass(frozen=True)
class Row:
    primitive: str
    backend: str
    payload_bytes: int
    median_ns: int
    throughput_gbps: float


def parse_file(path: pathlib.Path) -> tuple[str, list[Row]]:
    runner = path.stem.removeprefix("perf-")
    rows: list[Row] = []
    seen_header = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if line.startswith("# runner="):
                runner = line.split("=", 1)[1].strip()
            continue
        if not seen_header:
            # First non-comment line is the column header.
            seen_header = True
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            print(f"warn: skipping malformed line in {path}: {line!r}", file=sys.stderr)
            continue
        primitive, backend, payload_bytes, median_ns, throughput = parts
        rows.append(
            Row(
                primitive=primitive,
                backend=backend,
                payload_bytes=int(payload_bytes),
                median_ns=int(median_ns),
                throughput_gbps=float(throughput),
            )
        )
    return runner, rows


def fmt_payload(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n // (1024 * 1024)} MiB"
    if n >= 1024:
        return f"{n // 1024} KiB"
    return f"{n} B"


def fmt_cell(median_ns: int | None, gbps: float | None) -> str:
    if median_ns is None or gbps is None:
        return "—"
    return f"{median_ns:,} ns<br>{gbps:.2f} GB/s"


def is_scalar_baseline(backend: str) -> bool:
    # Anything starting with "scalar" is treated as a baseline candidate.
    return backend.startswith("scalar")


def render(by_runner: dict[str, list[Row]]) -> str:
    runners = sorted(by_runner.keys())

    # Collect every (primitive, backend, payload) cell, indexed by runner.
    # cells[(primitive, backend, payload)][runner] = Row
    cells: dict[tuple[str, str, int], dict[str, Row]] = defaultdict(dict)
    for runner, rows in by_runner.items():
        for row in rows:
            cells[(row.primitive, row.backend, row.payload_bytes)][runner] = row

    primitives = sorted({k[0] for k in cells})
    out: list[str] = []
    out.append("## tokenfs-algos perf comparison\n")
    out.append(
        f"Runners: {', '.join(f'`{r}`' for r in runners)}\n\n"
        "Cells show median ns / throughput. "
        "Speedup is vs. the fastest scalar backend on the same runner.\n"
    )

    for primitive in primitives:
        out.append(f"\n### {primitive}\n\n")
        # Group by payload size, then list backends within each.
        backend_payload = sorted(
            {(b, p) for (pname, b, p) in cells if pname == primitive},
            key=lambda bp: (bp[1], bp[0]),
        )

        header_cells = ["payload", "backend"] + runners + ["best speedup"]
        out.append("| " + " | ".join(header_cells) + " |\n")
        out.append("|" + "|".join(["---"] * len(header_cells)) + "|\n")

        # Compute scalar baseline per (payload, runner): smallest median_ns
        # across any backend whose name starts with "scalar".
        scalar_baseline: dict[tuple[int, str], int] = {}
        for (pname, b, p), per_runner in cells.items():
            if pname != primitive or not is_scalar_baseline(b):
                continue
            for runner, row in per_runner.items():
                key = (p, runner)
                cur = scalar_baseline.get(key)
                if cur is None or row.median_ns < cur:
                    scalar_baseline[key] = row.median_ns

        for backend, payload in backend_payload:
            cell_row = [fmt_payload(payload), f"`{backend}`"]
            best_speedup: float | None = None
            for runner in runners:
                row = cells[(primitive, backend, payload)].get(runner)
                if row is None:
                    cell_row.append("—")
                    continue
                cell_row.append(fmt_cell(row.median_ns, row.throughput_gbps))
                base = scalar_baseline.get((payload, runner))
                if base and not is_scalar_baseline(backend):
                    speedup = base / max(row.median_ns, 1)
                    if best_speedup is None or speedup > best_speedup:
                        best_speedup = speedup
            cell_row.append(f"{best_speedup:.2f}×" if best_speedup is not None else "—")
            out.append("| " + " | ".join(cell_row) + " |\n")

    return "".join(out)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tsvs",
        nargs="+",
        type=pathlib.Path,
        help="One or more perf TSV files (e.g. perf-linux-x86_64.tsv).",
    )
    args = parser.parse_args(argv)

    by_runner: dict[str, list[Row]] = {}
    for path in args.tsvs:
        if not path.exists():
            print(f"warn: missing file: {path}", file=sys.stderr)
            continue
        runner, rows = parse_file(path)
        by_runner.setdefault(runner, []).extend(rows)

    if not by_runner:
        print("error: no perf rows parsed", file=sys.stderr)
        return 1

    sys.stdout.write(render(by_runner))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

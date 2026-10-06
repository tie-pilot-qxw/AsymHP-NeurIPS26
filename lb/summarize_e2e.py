#!/usr/bin/env python3
"""Summarize end-to-end latency, the sparse-region fraction, and the AsymHP
critical-path breakdown (paper Table 2 and the appendix critical-path breakdown).

Usage: python lb/summarize_e2e.py <run-dir written by lb/run_e2e.sh>
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path


E2E_RE = re.compile(r"E2E wall-clock:\s*([\d.]+)\s*s")


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    # Two-sided Student-t critical values are important for the small number of
    # full-generation repetitions used here (five by default).
    t975 = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        21: 2.080,
        22: 2.074,
        23: 2.069,
        24: 2.064,
        25: 2.060,
        26: 2.056,
        27: 2.052,
        28: 2.048,
        29: 2.045,
        30: 2.042,
    }
    critical = t975.get(len(values) - 1, 1.96)
    return mean, critical * statistics.stdev(values) / math.sqrt(len(values))


def load_run(log_path: Path, dump_path: Path) -> dict:
    text = log_path.read_text()
    matches = E2E_RE.findall(text)
    if not matches:
        raise ValueError(f"no E2E wall-clock in {log_path}")
    e2e_s = float(matches[-1])
    dump = json.loads(dump_path.read_text())
    calls = dump["per_call_C_max"]
    flags = dump.get("sparse_flags", [True] * len(calls))
    if calls is None or len(flags) != len(calls):
        raise ValueError(f"invalid or unaligned timing dump: {dump_path}")
    sparse_calls = [v for v, sparse in zip(calls, flags) if sparse]
    sparse_total_ms = sum(sparse_calls)
    return {
        "pattern": dump.get("pattern", "unknown"),
        "e2e_s": e2e_s,
        "region_all_ms": statistics.mean(calls),
        "region_sparse_ms": statistics.mean(sparse_calls),
        "sparse_fraction": sparse_total_ms / (e2e_s * 1000.0),
        "sparse_calls": len(sparse_calls),
        "breakdown": dump.get("critical_sparse_breakdown_ms"),
        "schedule_exposed_ms": (
            dump.get("schedule_exposed_total_ms", 0.0) /
            max(dump.get("schedule_calls", 0), 1)
        ),
        "schedule_per_sparse_invocation_ms": (
            dump.get("schedule_exposed_total_ms", 0.0) /
            max(len(sparse_calls), 1)
        ),
        "density_exchange_ms": (
            dump.get("density_exchange_rank0_total_ms", 0.0) /
            max(len(sparse_calls), 1)
        ),
    }


def collect(root: Path, tag: str) -> dict[int, dict]:
    """Complete runs of one execution, keyed by repetition number."""
    runs = {}
    for log_path in root.glob(f"{tag}_r*.log"):
        match = re.fullmatch(rf"{tag}_r(\d+)\.log", log_path.name)
        dump_path = log_path.with_suffix(".json")
        if match and dump_path.exists():
            runs[int(match.group(1))] = load_run(log_path, dump_path)
    return runs


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "result/paper/e2e_wan13b_720p_120f_w4")
    by_rep = {tag: collect(root, tag) for tag in ("baseline", "asymhp")}
    # Pair baseline and AsymHP by repetition; drop reps missing either side.
    reps = sorted(set(by_rep["baseline"]) & set(by_rep["asymhp"]))
    if not reps:
        raise ValueError(f"no repetition with both baseline and asymhp runs under {root}")
    runs = {tag: [by_rep[tag][r] for r in reps] for tag in by_rep}
    pattern = runs["baseline"][0]["pattern"]

    print("# End-to-end summary")
    print()
    print(f"Wan2.1-1.3B, 720p, 50 denoising steps, local sparse method={pattern} "
          "from the first step/layer, W=4.")
    print()
    print("| execution | repetitions | E2E (s, mean ± 95% CI) | sparse-region (ms/call) | "
          "fraction of E2E in sparse attention |")
    print("|---|---:|---:|---:|---:|")
    for tag in ("baseline", "asymhp"):
        e2e, ci = mean_ci95([r["e2e_s"] for r in runs[tag]])
        region = statistics.mean(r["region_sparse_ms"] for r in runs[tag])
        fraction = statistics.mean(r["sparse_fraction"] for r in runs[tag])
        print(f"| {tag} | {len(runs[tag])} | {e2e:.3f} ± {ci:.3f} | "
              f"{region:.3f} | {100*fraction:.2f}% |")

    base_mean = statistics.mean(r["e2e_s"] for r in runs["baseline"])
    asym_mean = statistics.mean(r["e2e_s"] for r in runs["asymhp"])
    base_region = statistics.mean(r["region_sparse_ms"] for r in runs["baseline"])
    asym_region = statistics.mean(r["region_sparse_ms"] for r in runs["asymhp"])
    base_fraction = statistics.mean(r["sparse_fraction"] for r in runs["baseline"])
    region_speedup = base_region / asym_region
    amdahl_speedup = 1.0 / ((1.0 - base_fraction) + base_fraction / region_speedup)
    paired = [
        b["e2e_s"] / a["e2e_s"]
        for b, a in zip(runs["baseline"], runs["asymhp"])
    ]
    print()
    print(f"- E2E speedup (ratio of means): **{base_mean/asym_mean:.4f}×**")
    print(f"- Paired-run speedup: **{statistics.mean(paired):.4f}×** "
          f"(values: {', '.join(f'{v:.4f}×' for v in paired)})")
    print(f"- Sparse-region speedup: **{region_speedup:.4f}×**; "
          f"Amdahl prediction from the baseline affected fraction: **{amdahl_speedup:.4f}×** E2E.")

    breakdowns = [r["breakdown"] for r in runs["asymhp"] if r["breakdown"]]
    if breakdowns:
        keys = list(breakdowns[0])
        means = {key: statistics.mean(b[key] for b in breakdowns) for key in keys}
        total = sum(means.values())
        print()
        print("## AsymHP critical-path breakdown (sparse invocations)")
        print()
        print("| phase | ms/invocation | share |")
        print("|---|---:|---:|")
        for key in keys:
            share = means[key] / total if total else float("nan")
            print(f"| {key} | {means[key]:.4f} | {100*share:.2f}% |")
        sched = statistics.mean(r["schedule_exposed_ms"] for r in runs["asymhp"])
        sched_per_sparse = statistics.mean(
            r["schedule_per_sparse_invocation_ms"] for r in runs["asymhp"]
        )
        density = statistics.mean(r["density_exchange_ms"] for r in runs["asymhp"])
        print()
        print(f"- Exposed event-wait + CPU LPT scheduling: **{sched:.4f} ms/re-plan**.")
        print(f"- Density scatter/all-reduce/D2H on the overlapped side stream "
              f"(rank 0): **{density:.4f} ms/sparse invocation**.")
        print(f"- Scheduling plus density metadata work: "
              f"**{100*(sched_per_sparse+density)/asym_region:.2f}%** "
              "of one sparse-region invocation (density exchange is overlapped).")


if __name__ == "__main__":
    main()

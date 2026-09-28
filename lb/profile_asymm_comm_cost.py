#!/usr/bin/env python
"""Fit per-rank asymmetric A2A communication cost versus assigned head count.

The measured regions intentionally exclude symmetric-memory barriers:

* forward measures the three local Q/K/V pull kernels;
* reverse measures the local output push kernel.

All ranks launch concurrently, so the samples retain PCIe-fabric contention,
but an early rank's final barrier wait is not mis-attributed to its own copy
cost.  The resulting affine fits are consumed by the communication-aware
head-placement policy in ``bench_sp_all2all_attention.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist


sys.path.insert(0, str(Path(__file__).resolve().parent))

from symm_a2a import SymmAsymA2A


DEFAULT_COUNT_CASES = (
    "3,3,3,3;"
    "1,3,4,4;3,4,4,1;4,4,1,3;4,1,3,4;"
    "1,2,4,5;2,4,5,1;4,5,1,2;5,1,2,4"
)


def parse_count_cases(raw: str, world_size: int, total_heads: int) -> List[List[int]]:
    cases: List[List[int]] = []
    for case_raw in raw.split(";"):
        if not case_raw.strip():
            continue
        counts = [int(item.strip()) for item in case_raw.split(",")]
        if len(counts) != world_size:
            raise ValueError(
                f"count case {counts} has {len(counts)} ranks, expected {world_size}"
            )
        if any(count <= 0 for count in counts):
            raise ValueError(f"all ranks must own at least one head, got {counts}")
        if sum(counts) != total_heads:
            raise ValueError(
                f"count case {counts} sums to {sum(counts)}, expected {total_heads}"
            )
        if counts not in cases:
            cases.append(counts)
    if not cases:
        raise ValueError("no communication calibration count cases")
    return cases


def contiguous_assignment(counts: Sequence[int]) -> List[List[int]]:
    assigned: List[List[int]] = []
    start = 0
    for count in counts:
        assigned.append(list(range(start, start + count)))
        start += count
    return assigned


def fit_affine(rows: Sequence[Dict], key: str, rank: int) -> Dict:
    selected = [row for row in rows if int(row["rank"]) == rank]
    xs = [float(row["heads"]) for row in selected]
    ys = [float(row[key]) for row in selected]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = 0.0 if denom == 0.0 else sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / denom
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * x for x in xs]
    ss_res = sum((y - pred) ** 2 for y, pred in zip(ys, predictions))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return {
        "rank": rank,
        "intercept_ms": intercept,
        "slope_ms_per_head": slope,
        "r2": r2,
        "num_samples": len(selected),
    }


def elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event, iters: int) -> float:
    end.synchronize()
    return float(start.elapsed_time(end)) / iters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit barrier-free per-rank pull/push cost for asymmetric A2A."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--total-heads", type=int, default=12)
    parser.add_argument("--s-local", type=int, default=27900)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--count-cases", default=DEFAULT_COUNT_CASES)
    args = parser.parse_args()

    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("--warmup must be non-negative and --iters must be positive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    cases = parse_count_cases(args.count_cases, world_size, args.total_heads)
    a2a = SymmAsymA2A(
        dist.group.WORLD,
        buffer_shape=(
            args.batch,
            args.total_heads,
            args.s_local,
            args.head_dim,
        ),
        dtype=dtype,
        device=device,
        s_block=128,
        enable_reverse=True,
    )
    for name in ("q", "k", "v"):
        a2a._symm[name].zero_()
    dist.barrier()

    local_rows: List[Dict] = []
    for case_index, counts in enumerate(cases):
        assigned = contiguous_assignment(counts)
        local_heads = assigned[rank]
        h_idxs = torch.tensor(local_heads, device=device, dtype=torch.int32)
        h_local = len(local_heads)
        s_padded = a2a.s_local_padded
        pull_shape = (
            args.batch,
            h_local,
            world_size * s_padded,
            args.head_dim,
        )
        pull_out = {
            name: torch.empty(pull_shape, device=device, dtype=dtype)
            for name in ("q", "k", "v")
        }
        push_src = torch.zeros(pull_shape, device=device, dtype=dtype)

        for _ in range(args.warmup):
            for name in ("q", "k", "v"):
                a2a.pull_seq_to_heads(
                    name,
                    h_idxs,
                    out=pull_out[name],
                    pre_barrier=False,
                    post_barrier=False,
                )
            a2a.push_heads_to_seq(
                push_src,
                h_idxs,
                pre_barrier=False,
                post_barrier=False,
            )
        torch.cuda.synchronize(device)
        dist.barrier()

        pull_start = torch.cuda.Event(enable_timing=True)
        pull_end = torch.cuda.Event(enable_timing=True)
        pull_start.record()
        for _ in range(args.iters):
            for name in ("q", "k", "v"):
                a2a.pull_seq_to_heads(
                    name,
                    h_idxs,
                    out=pull_out[name],
                    pre_barrier=False,
                    post_barrier=False,
                )
        pull_end.record()
        pull_qkv_ms = elapsed_ms(pull_start, pull_end, args.iters)
        dist.barrier()

        push_start = torch.cuda.Event(enable_timing=True)
        push_end = torch.cuda.Event(enable_timing=True)
        push_start.record()
        for _ in range(args.iters):
            a2a.push_heads_to_seq(
                push_src,
                h_idxs,
                pre_barrier=False,
                post_barrier=False,
            )
        push_end.record()
        push_out_ms = elapsed_ms(push_start, push_end, args.iters)
        dist.barrier()

        local_rows.append(
            {
                "case": case_index,
                "counts": list(counts),
                "rank": rank,
                "heads": h_local,
                "pull_qkv_ms": pull_qkv_ms,
                "push_out_ms": push_out_ms,
            }
        )
        print(
            f"rank={rank} case={case_index} counts={counts} heads={h_local} "
            f"pull_qkv={pull_qkv_ms:.3f} ms push_out={push_out_ms:.3f} ms",
            flush=True,
        )

        del pull_out, push_src, h_idxs
        torch.cuda.empty_cache()

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_rows, gathered, dst=0)
    if rank == 0:
        rows = [row for rank_rows in gathered for row in rank_rows]
        model = {
            "schema_version": 1,
            "backend": a2a.backend,
            "pcie_config": (
                list(a2a.pcie_config)
                if a2a.pcie_config is not None
                else None
            ),
            "world_size": world_size,
            "shape": {
                "batch": args.batch,
                "total_heads": args.total_heads,
                "s_local": args.s_local,
                "seq_len": args.s_local * world_size,
                "head_dim": args.head_dim,
                "dtype": str(dtype),
            },
            "measurement": {
                "warmup": args.warmup,
                "iters": args.iters,
                "barriers_in_timed_region": False,
                "pull_kernels_per_iter": 3,
                "push_kernels_per_iter": 1,
            },
            "pull_qkv": {
                "rank_fits": [
                    fit_affine(rows, "pull_qkv_ms", fit_rank)
                    for fit_rank in range(world_size)
                ]
            },
            "push_out": {
                "rank_fits": [
                    fit_affine(rows, "push_out_ms", fit_rank)
                    for fit_rank in range(world_size)
                ]
            },
            "samples": sorted(rows, key=lambda row: (row["case"], row["rank"])),
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(model, indent=2))
        print(f"wrote communication cost model: {path}")
        for direction in ("pull_qkv", "push_out"):
            for fit in model[direction]["rank_fits"]:
                print(
                    f"{direction} rank={fit['rank']}: "
                    f"{fit['intercept_ms']:.4f} + "
                    f"{fit['slope_ms_per_head']:.4f} * heads "
                    f"(R2={fit['r2']:.4f})"
                )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

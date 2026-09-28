#!/usr/bin/env python3
"""Distributed SpargeAttn operator replay on one real Wan snapshot (paper
Figure 6(b)).

Every rank replays the Q/K/V tensors captured from a 720p Wan2.1-1.3B
generation (125 frames, so the per-GPU sequence length is 128-aligned); no
synthetic shapes are used. Launched by lb/run_sparge.sh.

The comparison is:
  * contiguous head placement + symmetric NCCL Ulysses exchange;
  * density-aware unequal placement + AsymHP's asymmetric TMA exchange.

Sparge's density is taken from the block-count metadata generated for this same
snapshot during fixed-workload calibration.  The AsymHP timing conservatively
includes a non-overlapped 12-float density all-reduce after the reverse
exchange; production overlaps this metadata exchange with subsequent compute.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from lb.bench_sp_all2all_attention import load_cost_model
from lb.wan_sp.context import SPContext
from lb.wan_sp.planning import build_one_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--cost-model-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--simthreshd1", type=float, default=0.3)
    parser.add_argument("--cdfthreshd", type=float, default=0.6)
    parser.add_argument("--pvthreshd", type=int, default=50)
    return parser.parse_args()


def calibrated_density(cost_path: str, num_heads: int) -> list[float]:
    payload = json.loads(Path(cost_path).read_text())
    density: list[float | None] = [None] * num_heads
    for row in payload.get("samples", []):
        ids = row.get("head_ids", [])
        if int(row.get("heads", 0)) == 1 and len(ids) == 1:
            density[int(ids[0])] = float(row["density_mean"])
    if any(value is None for value in density):
        raise ValueError(
            "cost-model JSON must contain one single-head calibration sample "
            f"for each of the {num_heads} heads"
        )
    return [float(value) for value in density]


def gather_max(values: list[float], world_size: int, device: torch.device) -> list[float]:
    local = torch.tensor(values, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    stacked = torch.stack(gathered)
    return stacked.max(dim=0).values.cpu().tolist()


def mean_ci95(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    # Paper protocol: 50 measured iterations after 20 warmups (df=49).
    t975 = {49: 2.010}
    critical = t975.get(len(values) - 1, 1.96)
    return mean, critical * statistics.stdev(values) / math.sqrt(len(values))


def main() -> None:
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    from spas_sage_attn import spas_sage2_attn_meansim_cuda

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    query, key, value = (payload["inputs"][name] for name in ("query", "key", "value"))
    cfg, num_heads, seq_len, head_dim = query.shape
    if cfg != 1 or seq_len % world_size:
        raise ValueError(
            f"expected cfg=1 and sequence divisible by W; got shape={tuple(query.shape)}, "
            f"W={world_size}"
        )
    if num_heads % world_size:
        raise ValueError(f"contiguous baseline requires heads={num_heads} divisible by W={world_size}")

    seq_local = seq_len // world_size
    seq_slice = slice(rank * seq_local, (rank + 1) * seq_local)
    q_seq = query[:, :, seq_slice, :].to(device).contiguous()
    k_seq = key[:, :, seq_slice, :].to(device).contiguous()
    v_seq = value[:, :, seq_slice, :].to(device).contiguous()
    del payload, query, key, value

    density = calibrated_density(args.cost_model_json, num_heads)
    cost_model = load_cost_model(args.cost_model_json)
    plan = build_one_plan(
        density,
        num_heads,
        world_size,
        "greedy_unequal",
        seq_len,
        cost_model,
        min_heads_per_rank=1,
    )
    assigned = plan.assigned

    symmetric = SPContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        group=dist.group.WORLD,
        a2a_backend="symm",
    )
    asymmetric = SPContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        group=dist.group.WORLD,
        a2a_backend="asymm",
    )
    asymmetric.setup_asymm(
        b=cfg,
        h_total=num_heads,
        s_local=seq_local,
        d=head_dim,
        dtype=q_seq.dtype,
    )
    head_ids = torch.as_tensor(assigned[rank], dtype=torch.int32, device=device)

    def sparge(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, density_out: bool):
        return spas_sage2_attn_meansim_cuda(
            q,
            k,
            v,
            is_causal=False,
            scale=None,
            smooth_k=True,
            simthreshd1=args.simthreshd1,
            cdfthreshd=args.cdfthreshd,
            pvthreshd=args.pvthreshd,
            tensor_layout="HND",
            output_dtype=q.dtype,
            return_head_density=density_out,
        )

    def run_baseline():
        qh = symmetric.a2a_seq_to_heads(q_seq)
        kh = symmetric.a2a_seq_to_heads(k_seq)
        vh = symmetric.a2a_seq_to_heads(v_seq)
        out_h = sparge(qh, kh, vh, density_out=False)
        return symmetric.a2a_heads_to_seq(out_h)

    def run_asymhp():
        qh, kh, vh = asymmetric.a2a_asymm_forward(q_seq, k_seq, v_seq, head_ids)
        out_h, local_density = sparge(qh, kh, vh, density_out=True)
        out_seq = asymmetric.a2a_asymm_reverse(out_h, head_ids)
        # Conservative timing: serialize the tiny global-density exchange here.
        # The E2E implementation launches it on a side stream one invocation
        # ahead, where it overlaps subsequent transformer compute.
        global_density = torch.zeros(num_heads, dtype=torch.float32, device=device)
        global_density.index_copy_(0, head_ids.long(), local_density.reshape(-1))
        dist.all_reduce(global_density, op=dist.ReduceOp.SUM)
        return out_seq, global_density

    def warm(fn) -> None:
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize(device)
        dist.barrier()

    warm(run_baseline)
    warm(run_asymhp)

    def timed(fn) -> float:
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        del result
        return float(start.elapsed_time(end))

    baseline_local: list[float] = []
    asymhp_local: list[float] = []
    for index in range(args.iters):
        # Alternate order to avoid assigning a fixed thermal/cache advantage.
        if index % 2 == 0:
            baseline_local.append(timed(run_baseline))
            asymhp_local.append(timed(run_asymhp))
        else:
            asymhp_local.append(timed(run_asymhp))
            baseline_local.append(timed(run_baseline))

    baseline = gather_max(baseline_local, world_size, device)
    asymhp = gather_max(asymhp_local, world_size, device)
    if rank == 0:
        paired = [base / asym for base, asym in zip(baseline, asymhp)]
        baseline_mean, baseline_ci = mean_ci95(baseline)
        asymhp_mean, asymhp_ci = mean_ci95(asymhp)
        paired_mean, paired_ci = mean_ci95(paired)
        result = {
            "scope": "distributed operator replay of one real paper-setting QKV snapshot",
            "input": args.input,
            "source_metadata": {
                key: (value.item() if isinstance(value, torch.Tensor) and value.numel() == 1 else value)
                for key, value in metadata.items()
                if not isinstance(value, torch.Tensor) or value.numel() == 1
            },
            "world_size": world_size,
            "sequence_length": seq_len,
            "num_heads": num_heads,
            "sparse_method": "SpargeAttn",
            "thresholds": {
                "simthreshd1": args.simthreshd1,
                "cdfthreshd": args.cdfthreshd,
                "pvthreshd": args.pvthreshd,
            },
            "density": density,
            "assignment": assigned,
            "heads_per_rank": [len(heads) for heads in assigned],
            "baseline_ms": baseline,
            "asymhp_ms": asymhp,
            "paired_speedup": paired,
            "summary": {
                "baseline_mean_ms": baseline_mean,
                "baseline_ci95_ms": baseline_ci,
                "baseline_median_ms": statistics.median(baseline),
                "asymhp_mean_ms": asymhp_mean,
                "asymhp_ci95_ms": asymhp_ci,
                "asymhp_median_ms": statistics.median(asymhp),
                "speedup_ratio_of_means": baseline_mean / asymhp_mean,
                "paired_speedup_mean": paired_mean,
                "paired_speedup_ci95": paired_ci,
                "paired_speedup_median": statistics.median(paired),
            },
            "asymhp_timing_includes_nonoverlapped_density_allreduce": True,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["summary"], indent=2), flush=True)
        print(f"heads_per_rank={result['heads_per_rank']} assignment={assigned}", flush=True)
        print(f"wrote {output}", flush=True)

    dist.barrier()
    # Tear symmetric-memory objects down before the NCCL process group.
    asymmetric.symm = None
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

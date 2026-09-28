#!/usr/bin/env python3
"""Calibrate AsymHP's per-head cost model for real SpargeAttn.

This profiler intentionally stays at one paper workload.  It replays subsets of
heads from one real Wan QKV snapshot and fits

    latency_ms = launch + a * num_heads + b * sum(per_head_density)

at the snapshot's fixed sequence length.  The fitted ``a`` and ``b`` are then
encoded in the cost-model schema consumed by AsymHP:

    per_head_cost = a + b * density
                  = mask_slope * N + attention_slope * density * N^2.

The timed operation matches the online SpargeAttn path: the real fused
SpargeAttn kernel plus a reduction of its already-built valid-block counts to
expose per-head densities (no second block-map pass).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def jsonable(value):
    """Convert snapshot metadata to plain JSON without changing fit inputs."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Real Wan attention-core QKV snapshot.")
    parser.add_argument("--output", required=True, help="Output AsymHP cost-model JSON.")
    parser.add_argument("--simthreshd1", type=float, default=0.3)
    parser.add_argument("--cdfthreshd", type=float, default=0.6)
    parser.add_argument("--pvthreshd", type=int, default=50)
    parser.add_argument("--max-subset-heads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    return parser.parse_args()


def timed_ms(fn) -> tuple[object, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, start.elapsed_time(end)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    from spas_sage_attn import spas_sage2_attn_meansim_cuda

    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    device = torch.device("cuda", torch.cuda.current_device())
    query = payload["inputs"]["query"].to(device).contiguous()
    key = payload["inputs"]["key"].to(device).contiguous()
    value = payload["inputs"]["value"].to(device).contiguous()
    cfg, num_heads, seq_len, head_dim = query.shape
    if cfg != 1:
        raise ValueError(f"expected cfg=1, got {cfg}")

    def prepare_subset(
        head_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = torch.tensor(head_ids, device=device)
        q = query.index_select(1, idx).contiguous()
        k = key.index_select(1, idx).contiguous()
        v = value.index_select(1, idx).contiguous()
        return q, k, v

    def run_prepared(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, density = spas_sage2_attn_meansim_cuda(
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
            return_head_density=True,
        )
        return out, density

    subsets: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for count in range(1, min(args.max_subset_heads, num_heads) + 1):
        for start in range(num_heads):
            subset = tuple(sorted((start + offset) % num_heads for offset in range(count)))
            if subset not in seen:
                seen.add(subset)
                subsets.append(subset)

    rows = []
    for index, subset in enumerate(subsets, 1):
        q, k, v = prepare_subset(subset)
        for _ in range(args.warmup):
            run_prepared(q, k, v)
        times = []
        density = None
        for _ in range(args.iters):
            (out, density), elapsed = timed_ms(lambda: run_prepared(q, k, v))
            times.append(elapsed)
            del out
        assert density is not None
        density_mean = float(density.mean().item())
        row = {
            "heads": len(subset),
            "head_ids": list(subset),
            "density_mean": density_mean,
            "density_sum": density_mean * len(subset),
            "latency_mean_ms": statistics.mean(times),
            "latency_samples_ms": times,
        }
        rows.append(row)
        print(
            f"[{index:02d}/{len(subsets):02d}] heads={len(subset)} "
            f"density={density_mean:.4f} latency={row['latency_mean_ms']:.4f} ms",
            flush=True,
        )
        del q, k, v

    # Fit [launch, per-head floor, density-proportional per-head work].
    x = torch.tensor(
        [[1.0, float(row["heads"]), float(row["density_sum"])] for row in rows],
        dtype=torch.float64,
    )
    y = torch.tensor([float(row["latency_mean_ms"]) for row in rows], dtype=torch.float64)
    coefficients = torch.linalg.lstsq(x, y).solution
    launch_ms, per_head_ms, per_density_head_ms = coefficients.tolist()

    # The runtime model is intentionally non-negative.  If batching noise makes
    # the unconstrained per-head floor slightly negative, set it to zero and
    # refit the remaining launch/density terms.
    if per_head_ms < 0:
        x_reduced = x[:, [0, 2]]
        reduced = torch.linalg.lstsq(x_reduced, y).solution
        launch_ms, per_density_head_ms = reduced.tolist()
        per_head_ms = 0.0
    if per_density_head_ms <= 0:
        raise ValueError(
            f"non-positive density coefficient {per_density_head_ms}; "
            "the calibration does not support density-aware placement"
        )

    fitted = launch_ms + per_head_ms * x[:, 1] + per_density_head_ms * x[:, 2]
    ss_res = float(((y - fitted) ** 2).sum().item())
    ss_tot = float(((y - y.mean()) ** 2).sum().item())
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0

    result = {
        "input": str(args.input),
        "metadata": {
            "source_metadata": jsonable(metadata),
            "sparse_method": "SpargeAttn",
            "simthreshd1": args.simthreshd1,
            "cdfthreshd": args.cdfthreshd,
            "pvthreshd": args.pvthreshd,
            "sequence_length": seq_len,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "fit_scope": "fixed paper workload; cyclic head subsets",
        },
        "formula": {
            "measured_ms": "launch_ms + per_head_ms * heads + per_density_head_ms * sum_density",
            "placement_cost": "mask_slope * N + attention_slope * density * N^2",
        },
        "fixed_workload_fit": {
            "launch_ms": launch_ms,
            "per_head_ms": per_head_ms,
            "per_density_head_ms": per_density_head_ms,
            "r2": r2,
            "num_samples": len(rows),
        },
        "mask_fit": {
            "intercept_ms": 0.0,
            "slope_ms_per_unit": per_head_ms / seq_len,
            "r2": r2,
            "x_key": "heads * seq_len",
            "y_key": "latency_mean_ms",
            "num_samples": len(rows),
        },
        "attention_fit": {
            "intercept_ms": launch_ms,
            "slope_ms_per_unit": per_density_head_ms / (seq_len * seq_len),
            "r2": r2,
            "x_key": "sum_density * seq_len^2",
            "y_key": "latency_mean_ms",
            "num_samples": len(rows),
        },
        "samples": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[fit] launch={launch_ms:.6f} ms, per_head={per_head_ms:.6f} ms, "
        f"per_density_head={per_density_head_ms:.6f} ms, R2={r2:.6f}"
    )
    print(f"[fit] wrote {output}")


if __name__ == "__main__":
    main()

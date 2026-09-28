"""Build fixed per-layer head-placement plans for SP Wan T2V.

Reuses the pure load-balancing helpers from ``bench_sp_all2all_attention``
(density parsing, cost model, greedy/LPT assignment, uniform padding, restore
index) so the E2E path and the microbenchmark share one planner.

The plan is fixed for the whole denoising run: per-head density is a
structural property of a head and is stable across denoising steps, so we
aggregate a density profile per layer (mean over recorded steps) and place
heads once. Fixing the plan also keeps each rank's kmeans centroid cache
valid across steps (a head never migrates ranks mid-run).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

# lb/ must be importable so bench's `from sap_state import ...` resolves.
_LB_DIR = Path(__file__).resolve().parents[1]
if str(_LB_DIR) not in sys.path:
    sys.path.insert(0, str(_LB_DIR))

from bench_sp_all2all_attention import (  # noqa: E402
    estimate_head_costs,
    greedy_head_assignment,
    greedy_lpt_assignment,
    load_cost_model,
    load_density_log,
    pad_rank_heads_uniform,
    restore_indices_from_head_order,
)

from .context import LayerPlan  # noqa: E402


def _contiguous_plan(num_heads: int, world_size: int) -> LayerPlan:
    if num_heads % world_size != 0:
        raise ValueError(
            f"contiguous plan requires num_heads ({num_heads}) divisible by "
            f"world_size ({world_size}); pick a compatible world size."
        )
    hpr = num_heads // world_size
    assigned = [list(range(r * hpr, (r + 1) * hpr)) for r in range(world_size)]
    head_order = list(range(num_heads))
    real = [hpr] * world_size
    restore = restore_indices_from_head_order(head_order, world_size, num_heads, real)
    return LayerPlan(head_order, real, hpr, restore, assigned=assigned, strategy="contiguous")


def _plan_from_assignment(
    assigned: List[List[int]], num_heads: int, world_size: int, strategy: str, info: Optional[dict]
) -> LayerPlan:
    head_order, real, max_hpr = pad_rank_heads_uniform(assigned)
    restore = restore_indices_from_head_order(head_order, world_size, num_heads, real)
    return LayerPlan(head_order, real, max_hpr, restore, assigned=assigned, strategy=strategy, info=info)


def _static_even_plan(num_heads: int, world_size: int) -> LayerPlan:
    """Deterministic near-even contiguous placement that ALSO works when
    num_heads is NOT divisible by world_size (the first ``rem`` ranks get one
    extra head). Used as the online step-0 bootstrap and the no-density fallback,
    so AsymHP runs the non-divisible configs the paper highlights (e.g. 12 heads
    on 5 GPUs) where the equal-count contiguous baseline cannot. For divisible
    configs it is identical to ``_contiguous_plan``."""
    base, rem = divmod(num_heads, world_size)
    assigned: List[List[int]] = []
    start = 0
    for r in range(world_size):
        cnt = base + (1 if r < rem else 0)
        assigned.append(list(range(start, start + cnt)))
        start += cnt
    return _plan_from_assignment(assigned, num_heads, world_size, "static_even", None)


def _mean_layer_density(rows) -> List[float]:
    """Mean per-head density across all recorded (timestep) rows for a layer."""
    n = len(rows[0][1])
    acc = [0.0] * n
    for _ts, vals in rows:
        for i, v in enumerate(vals):
            acc[i] += v
    return [a / len(rows) for a in acc]


def build_one_plan(
    density: List[float],
    num_heads: int,
    world_size: int,
    strategy: str,
    seq_len: int,
    cost_model,
    min_heads_per_rank: int = 1,
) -> LayerPlan:
    """Build a single layer's placement plan from ONE per-head density vector.

    Shared by the install-time planner (mean-over-steps density) and the online
    causal scheduler (previous denoising step's density), so both paths produce
    identical plan objects from the same LPT/greedy code.
    """
    if strategy == "contiguous":
        return _contiguous_plan(num_heads, world_size)
    costs = estimate_head_costs(density, seq_len, cost_model)
    if strategy == "greedy":
        assigned, _ = greedy_head_assignment(costs, world_size)
    else:
        assigned, _ = greedy_lpt_assignment(costs, world_size, min_heads_per_rank)
    return _plan_from_assignment(assigned, num_heads, world_size, strategy, info=None)


def build_layer_plans(
    num_layers: int,
    num_heads: int,
    world_size: int,
    strategy: str,
    seq_len: int,
    density_log_path: Optional[str] = None,
    cost_model_path: Optional[str] = None,
    min_heads_per_rank: int = 1,
) -> Dict[int, LayerPlan]:
    """Return {layer_idx: LayerPlan} for every layer.

    strategy:
      - "contiguous": identity placement (baseline; no density needed).
      - "greedy": equal heads/rank, balanced by density (symmetric a2a).
      - "greedy_unequal": LPT, variable heads/rank (needs asymmetric a2a to
        avoid padding away the benefit).

    Layers with no density in the log (e.g. FP-warmup layers) fall back to a
    contiguous plan.
    """
    if strategy == "contiguous":
        base = _contiguous_plan(num_heads, world_size)
        return {i: base for i in range(num_layers)}

    if strategy == "static":
        # near-even static placement; works for non-divisible head/GPU configs
        base = _static_even_plan(num_heads, world_size)
        return {i: base for i in range(num_layers)}

    if strategy not in ("greedy", "greedy_unequal"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if density_log_path is None:
        raise ValueError(f"strategy={strategy!r} requires a density log")

    log = load_density_log(density_log_path)
    cost_model = load_cost_model(cost_model_path)

    plans: Dict[int, LayerPlan] = {}
    fell_back: List[int] = []
    for layer in range(num_layers):
        rows = log.get(layer)
        if not rows:
            # near-even (not equal-count) so non-divisible configs don't raise
            plans[layer] = _static_even_plan(num_heads, world_size)
            fell_back.append(layer)
            continue
        density = _mean_layer_density(rows)
        if len(density) != num_heads:
            raise ValueError(
                f"layer {layer}: density length {len(density)} != num_heads {num_heads}"
            )
        costs = estimate_head_costs(density, seq_len, cost_model)
        if strategy == "greedy":
            assigned, loads = greedy_head_assignment(costs, world_size)
        else:
            assigned, loads = greedy_lpt_assignment(costs, world_size, min_heads_per_rank)
        info = {
            "layer": layer,
            "strategy": strategy,
            "heads_per_rank": [len(a) for a in assigned],
            "predicted_loads": [round(x, 4) for x in loads],
            "predicted_contig": [
                round(sum(costs[g * (num_heads // world_size):(g + 1) * (num_heads // world_size)]), 4)
                for g in range(world_size)
            ] if num_heads % world_size == 0 else None,
        }
        plans[layer] = _plan_from_assignment(assigned, num_heads, world_size, strategy, info)

    if fell_back:
        print(f"[wan-sp] {len(fell_back)} layer(s) fell back to contiguous (no density): {fell_back}")
    return plans


def summarize_plans(plans: Dict[int, LayerPlan], world_size: int) -> str:
    """Human-readable predicted-imbalance summary (rank-0 only)."""
    lines = []
    worst = None
    for layer, plan in sorted(plans.items()):
        if plan.info is None or plan.info.get("predicted_loads") is None:
            continue
        loads = plan.info["predicted_loads"]
        contig = plan.info.get("predicted_contig")
        mx, mean = max(loads), sum(loads) / len(loads)
        imb = mx / mean if mean > 0 else float("nan")
        if contig:
            cmx = max(contig)
            speedup = cmx / mx if mx > 0 else float("nan")
        else:
            speedup = float("nan")
        rec = (layer, imb, speedup, plan.info["heads_per_rank"])
        if worst is None or imb > worst[1]:
            worst = rec
        lines.append(
            f"  layer {layer:2d}: heads/rank={plan.info['heads_per_rank']} "
            f"max/mean={imb:.3f} predicted_speedup_vs_contig={speedup:.3f}x"
        )
    header = f"[wan-sp] per-layer plan ({len(lines)} balanced layers), W={world_size}"
    if worst:
        header += f"; worst imbalance layer {worst[0]} ({worst[1]:.3f} max/mean)"
    return header + "\n" + "\n".join(lines)

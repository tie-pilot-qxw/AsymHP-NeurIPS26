"""Split planner — planner-only, post-permutation Q-split.

Pure Python, no GPU. Takes prior-iteration per-head density + cost model and
produces a SplitPlan that combines greedy_unequal LPT placement with at most
one post-permutation Q-row split of the bottleneck rank's heaviest
attention-dominated head.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple


MIN_SPLIT_ATTN_MS = 1.0
MIN_ATTN_FRACTION = 0.5
SHARE_EPS_MS = 1e-6   # below this, treat a water-fill share as zero


@dataclass
class ComputeUnit:
    global_head: int
    q_lo: int
    q_hi: int
    role: str          # "full" | "split_owner" | "helper"
    cost_ms: float
    owner_rank: int
    split_id: int = -1


@dataclass
class SplitPlan:
    units_per_rank: List[List[ComputeUnit]]
    predicted_max_ms: float
    baseline_max_ms: float
    speedup: float
    diagnostics: Dict


def head_costs(
    densities: Sequence[float], cost_model: Dict, seq_len: int
) -> Tuple[List[float], List[float], List[float]]:
    """Returns (cost_mask_per_head, cost_attn_per_head, cost_total_per_head).

    Per-head SLOPE terms only:
        C^mask_h = alpha^mask * N          (a per-head floor -- constant across
                                            heads, but present per head; this is
                                            what makes fitted-cost placement
                                            differ from density-only placement)
        C^attn_h = alpha^attn * rho_h * N^2
    The fitted mask/attn INTERCEPTS (beta) are per-KERNEL-LAUNCH: the profiler
    (profile_maskgen_aware_cost.py) fits `beta + alpha*heads*work`, i.e. one
    mask-gen call + one attention call per rank batched over that rank's heads,
    so beta is a per-RANK fixed cost, NOT per head. Folding beta into the
    per-head cost would multiply the fitted per-call intercept by the head count.
    Add it once per active rank at makespan time via `rank_intercept()`; it does
    not change LPT placement (a uniform per-rank baseline), only the makespan
    level."""
    mask_slope = float(cost_model["mask_fit"]["slope_ms_per_unit"])
    attn_slope = float(cost_model["attention_fit"]["slope_ms_per_unit"])
    cost_mask = [mask_slope * seq_len for _ in densities]
    cost_attn = [attn_slope * float(d) * seq_len * seq_len for d in densities]
    cost_total = [m + a for m, a in zip(cost_mask, cost_attn)]
    return cost_mask, cost_attn, cost_total


def rank_intercept(cost_model: Dict) -> float:
    """Per-rank fixed cost = fitted mask-gen intercept + attention intercept
    (one mask-gen + one attention kernel launch per rank; the profiler fits beta
    as a per-call constant). Added once to every active rank's load at makespan
    time: makespan = max_r [rank_intercept + sum_h slope_cost_h]. Does not affect
    LPT placement (uniform across ranks); omitting it inflates makespan *ratios*
    (a common addend to numerator and denominator moves the ratio toward 1)."""
    return float(cost_model["mask_fit"]["intercept_ms"]) + float(
        cost_model["attention_fit"]["intercept_ms"]
    )


def greedy_lpt_assignment(
    costs: Sequence[float], world_size: int, min_heads_per_rank: int = 1
) -> Tuple[List[List[int]], List[float]]:
    """Same algorithm as bench_sp_all2all_attention.greedy_lpt_assignment."""
    n = len(costs)
    if min_heads_per_rank * world_size > n:
        raise ValueError(
            f"min_heads_per_rank={min_heads_per_rank} * world_size={world_size} > n_heads={n}"
        )
    loads = [0.0] * world_size
    assigned: List[List[int]] = [[] for _ in range(world_size)]
    order = sorted(range(n), key=lambda i: -costs[i])

    idx = 0
    for _ in range(min_heads_per_rank):
        for r in range(world_size):
            head = order[idx]
            assigned[r].append(head)
            loads[r] += costs[head]
            idx += 1

    for head in order[idx:]:
        best = min(range(world_size), key=lambda r: loads[r])
        assigned[best].append(head)
        loads[best] += costs[head]

    return [sorted(h) for h in assigned], loads


def water_fill(floors: Sequence[float], pool: float) -> Tuple[float, List[float]]:
    """Water-fill `pool` over participants with given `floors`.

    Returns (T, shares) where shares[i] = max(0, T - floors[i]) and
    Σ shares = pool. Drops the participant with the highest floor when
    T_candidate falls below it (that participant is "above the waterline"
    so it can't absorb any pool).
    """
    if pool < 0:
        raise ValueError(f"pool must be non-negative, got {pool}")
    n = len(floors)
    if n == 0:
        raise ValueError("water_fill needs at least 1 participant")

    # Sort indices by floor descending so we drop from index 0.
    order_desc = sorted(range(n), key=lambda i: -floors[i])
    active = list(order_desc)

    while active:
        floor_sum = sum(floors[i] for i in active)
        T = (pool + floor_sum) / len(active)
        max_floor = max(floors[i] for i in active)
        if T >= max_floor - 1e-12:
            shares = [0.0] * n
            for i in active:
                shares[i] = max(0.0, T - floors[i])
            return T, shares
        active.pop(0)  # drop highest-floor participant

    raise RuntimeError("water_fill: all participants dropped (should not happen)")


def derive_q_boundaries(
    shares_in_order: Sequence[float],
    pool: float,
    seq_len: int,
    q_granularity: int,
) -> Optional[List[int]]:
    """Cumulative boundaries with double-clamp + pre-feasibility check.

    Returns [b0=0, b1, b2, ..., bN=seq_len] of length N+1 where N = len(shares),
    or None if infeasible.
    """
    n = len(shares_in_order)
    if n < 1:
        return None
    if n * q_granularity > seq_len:
        return None
    if pool <= 0:
        return None

    boundaries = [0]
    cum = 0.0
    for i in range(n - 1):
        cum += shares_in_order[i]
        raw = cum / pool * seq_len
        b = round(raw / q_granularity) * q_granularity
        # Lower clamp: at least q_granularity past previous boundary.
        lower = boundaries[-1] + q_granularity
        # Upper clamp: leave q_granularity for each remaining participant.
        remaining_after_this = (n - 1) - i  # this is participants AFTER index i
        upper = seq_len - remaining_after_this * q_granularity
        if lower > upper:
            return None
        b = max(lower, min(upper, b))
        boundaries.append(b)
    boundaries.append(seq_len)

    # Sanity: monotone strictly increasing.
    for j in range(len(boundaries) - 1):
        if boundaries[j + 1] <= boundaries[j]:
            return None
    return boundaries


def _floor_for(rank: int, owner: int, current_loads: Sequence[float],
               cost_total_h: float, cost_mask_h: float) -> float:
    if rank == owner:
        return current_loads[rank] - cost_total_h + cost_mask_h
    return current_loads[rank]


def plan_with_splits(
    densities: Sequence[float],
    cost_model: Dict,
    seq_len: int,
    world_size: int,
    *,
    delta: float = 0.10,
    max_helpers_per_split: int = 2,
    max_splits_per_plan: int = 1,
    min_heads_per_rank: int = 1,
    q_granularity: int = 1,
    min_improvement_ms: float = 0.5,
) -> SplitPlan:
    """Whole-head LPT placement plus optional Q-row splits of bottleneck heads.

    Loads, makespans, and speedups here are slope-only: they exclude the fixed
    per-rank intercept (``rank_intercept``), as in the paper's whole-head
    granularity simulation. Add the intercept to every active rank to compare
    absolute times.
    """
    if max_splits_per_plan < 0:
        raise ValueError(f"max_splits_per_plan must be >= 0, got {max_splits_per_plan}")
    # Multi-split policy (naive): each rank is at most one of {owner, helper,
    # non-participant} per plan; an owner only splits one of its heads (so a
    # rank already chosen as owner is excluded from later iterations as both
    # owner and helper). A helper can be reused across splits. Already-split
    # heads are auto-excluded by the role=="full" filter below.
    if q_granularity < 1:
        raise ValueError(f"q_granularity must be >= 1, got {q_granularity}")
    if seq_len % q_granularity != 0:
        raise ValueError(
            f"seq_len={seq_len} must be divisible by q_granularity={q_granularity}; "
            f"otherwise the final boundary (= seq_len) violates the snap-to-grid invariant"
        )

    cost_mask, cost_attn, cost_total = head_costs(densities, cost_model, seq_len)

    assigned, baseline_loads = greedy_lpt_assignment(
        cost_total, world_size, min_heads_per_rank
    )
    baseline_max = max(baseline_loads)
    mean_load = sum(baseline_loads) / world_size

    units_per_rank: List[List[ComputeUnit]] = [
        [
            ComputeUnit(
                global_head=h, q_lo=0, q_hi=seq_len, role="full",
                cost_ms=cost_total[h], owner_rank=r, split_id=-1,
            )
            for h in heads
        ]
        for r, heads in enumerate(assigned)
    ]

    diagnostics: Dict = {
        "world_size": world_size,
        "seq_len": seq_len,
        "num_heads": len(densities),
        "delta_used": delta,
        "baseline_per_rank_ms": list(baseline_loads),
        "baseline_assigned_heads": [list(h) for h in assigned],
        "baseline_max_ms": baseline_max,
        "baseline_mean_ms": mean_load,
        "splits": [],
    }

    # Step: residual-tail check.
    if baseline_max <= (1.0 + delta) * mean_load:
        diagnostics["planned_per_rank_ms"] = list(baseline_loads)
        diagnostics["one_split_upper_bound"] = 1.0
        return SplitPlan(
            units_per_rank=units_per_rank,
            predicted_max_ms=baseline_max,
            baseline_max_ms=baseline_max,
            speedup=1.0,
            diagnostics=diagnostics,
        )

    current_loads = list(baseline_loads)
    next_split_id = 0
    owner_ranks: set = set()
    helper_ranks: set = set()   # locked: cannot become owner (helper-or-owner exclusive)

    for _split_iter in range(max_splits_per_plan):
        # Bottleneck candidate: not already locked as owner OR helper.
        cand = [r for r in range(world_size)
                if r not in owner_ranks and r not in helper_ranks]
        if not cand:
            break
        r_star = max(cand, key=lambda r: current_loads[r])

        # Iter 2+ residual-tail check (iter 0 was checked before the loop).
        if _split_iter > 0:
            mean_now = sum(current_loads) / world_size
            if current_loads[r_star] <= (1.0 + delta) * mean_now:
                break

        full_heads = [u.global_head for u in units_per_rank[r_star] if u.role == "full"]
        qualifying = []
        for h in full_heads:
            if cost_attn[h] < MIN_SPLIT_ATTN_MS:
                continue
            if cost_total[h] <= 0:
                continue
            if cost_attn[h] / cost_total[h] < MIN_ATTN_FRACTION:
                continue
            qualifying.append(h)
        if not qualifying:
            break
        h_split = max(qualifying, key=lambda gh: cost_attn[gh])

        # Helper candidates: not r_star, and not any already-locked owner.
        other_ranks = [
            r for r in range(world_size)
            if r != r_star and r not in owner_ranks
        ]
        if not other_ranks:
            break
        candidates: List[Dict] = []
        seen_active: set = set()

        for k in range(1, max_helpers_per_split + 1):
            for H in combinations(other_ranks, k):
                participants = [r_star] + list(H)
                floors = [
                    _floor_for(r, r_star, current_loads, cost_total[h_split], cost_mask[h_split])
                    for r in participants
                ]
                pool = cost_attn[h_split]
                T_H, shares_desired = water_fill(floors, pool)

                if shares_desired[0] <= SHARE_EPS_MS:
                    continue  # owner-zero rejection (under the desired water-fill)
                helper_active = [(r, s) for r, s in zip(H, shares_desired[1:]) if s > SHARE_EPS_MS]
                if not helper_active:
                    continue
                H_active = tuple(sorted(p[0] for p in helper_active))
                if H_active in seen_active:
                    continue
                seen_active.add(H_active)

                # Re-water-fill on the active set if any helper was dropped.
                if len(helper_active) != len(H):
                    participants = [r_star] + list(H_active)
                    floors = [
                        _floor_for(r, r_star, current_loads, cost_total[h_split], cost_mask[h_split])
                        for r in participants
                    ]
                    T_H, shares_desired = water_fill(floors, pool)
                    if shares_desired[0] <= SHARE_EPS_MS:
                        continue
                    if any(s <= SHARE_EPS_MS for s in shares_desired[1:]):
                        continue

                boundaries = derive_q_boundaries(
                    shares_in_order=shares_desired, pool=pool, seq_len=seq_len,
                    q_granularity=q_granularity,
                )
                if boundaries is None:
                    continue

                # snap-aware: actual share per participant follows the rounded
                # row count, not the water-fill ideal. With q_granularity > 1
                # these can differ significantly. All downstream cost/load
                # accounting uses actual_shares; shares_desired is kept only
                # for diagnostics.
                shares_actual = [
                    (boundaries[j + 1] - boundaries[j]) / seq_len * pool
                    for j in range(len(participants))
                ]

                # owner / helper zero re-check on actual shares (granularity
                # could have rounded a small share to zero rows).
                if shares_actual[0] <= SHARE_EPS_MS:
                    continue
                if any(s <= SHARE_EPS_MS for s in shares_actual[1:]):
                    continue

                candidate_loads = list(current_loads)
                for j, r in enumerate(participants):
                    candidate_loads[r] = floors[j] + shares_actual[j]
                planned_max = max(candidate_loads)
                total_offload = sum(shares_actual[1:])

                candidates.append({
                    "planned_max": planned_max,
                    "n_helpers": len(participants) - 1,
                    "total_offload": total_offload,
                    "T_H": T_H,
                    "participants": list(participants),
                    "shares_desired": list(shares_desired),
                    "shares_actual": list(shares_actual),
                    "boundaries": boundaries,
                    "candidate_loads": candidate_loads,
                })

        if not candidates:
            break
        # Tiebreak philosophy: when planned_max ties (very common — capped by
        # an untouched rank's load), pick the candidate that pushes r_star
        # lowest (smallest T_H), then most fresh helpers and most offloaded
        # work — this leaves a flatter post-split state, giving subsequent
        # iters better helper floors.
        candidates.sort(key=lambda c: (c["planned_max"], c["T_H"], -c["n_helpers"], -c["total_offload"]))
        best = candidates[0]
        # Stop only if splitting r_star fails to meaningfully reduce r_star's
        # own load. Comparing against global max would falsely reject a useful
        # split whenever a different rank's untouched load is the new max.
        r_star_drop = current_loads[r_star] - best["candidate_loads"][r_star]
        if r_star_drop <= min_improvement_ms:
            break

        # Snapshot pre-split loads — bound calculations below need them, and
        # the global bound also re-runs water_fill against this snapshot.
        pre_split_loads = list(current_loads)

        sid = next_split_id
        next_split_id += 1
        boundaries = best["boundaries"]
        participants = best["participants"]
        shares_actual = best["shares_actual"]
        shares_desired = best["shares_desired"]

        # Remove the split head's "full" unit from owner.
        units_per_rank[r_star] = [
            u for u in units_per_rank[r_star] if u.global_head != h_split
        ]
        # Emit split_owner. Cost = mask + actual snapped attn share.
        units_per_rank[r_star].append(ComputeUnit(
            global_head=h_split, q_lo=boundaries[0], q_hi=boundaries[1],
            role="split_owner", cost_ms=cost_mask[h_split] + shares_actual[0],
            owner_rank=r_star, split_id=sid,
        ))
        # Emit helpers. Cost = actual snapped attn share.
        for j in range(1, len(participants)):
            r = participants[j]
            units_per_rank[r].append(ComputeUnit(
                global_head=h_split, q_lo=boundaries[j], q_hi=boundaries[j + 1],
                role="helper", cost_ms=shares_actual[j],
                owner_rank=r_star, split_id=sid,
            ))
        for r, load in enumerate(best["candidate_loads"]):
            current_loads[r] = load
        owner_ranks.add(r_star)
        # Lock helpers from becoming an owner in later iterations (a helper
        # can still be helper for additional splits — that's allowed).
        for j in range(1, len(participants)):
            helper_ranks.add(participants[j])

        non_participants = [r for r in range(world_size) if r not in participants]
        max_non_part = (
            max(pre_split_loads[r] for r in non_participants) if non_participants else 0.0
        )

        # K-cap bound: we already exhausted candidates, so the best one IS
        # the K-cap optimum. No formula needed.
        bound_under_K = best["planned_max"]
        speedup_bound_under_K = baseline_max / bound_under_K if bound_under_K > 0 else float("inf")

        # Global one-split ceiling: water-fill cost_attn(h) over ALL ranks
        # treated as participants (no max_helpers cap, no comm cost). Other
        # ranks' floors stay fixed because we're only redistributing this
        # head's attention; their other heads are not split.
        all_participants_global = [r_star] + [r for r in range(world_size) if r != r_star]
        floors_all = [
            _floor_for(r, r_star, pre_split_loads, cost_total[h_split], cost_mask[h_split])
            for r in all_participants_global
        ]
        _T_global, shares_global = water_fill(floors_all, cost_attn[h_split])
        global_loads = list(pre_split_loads)
        for j, r in enumerate(all_participants_global):
            global_loads[r] = floors_all[j] + shares_global[j]
        global_bound = max(global_loads)
        speedup_bound_global = (
            baseline_max / global_bound if global_bound > 0 else float("inf")
        )

        diagnostics["splits"].append({
            "head": h_split,
            "owner": r_star,
            "helpers": list(participants[1:]),
            "max_helpers_per_split": max_helpers_per_split,
            "T_target_ms": best["T_H"],
            "shares_actual": {r: shares_actual[j] for j, r in enumerate(participants)},
            "shares_desired": {r: shares_desired[j] for j, r in enumerate(participants)},
            "q_ranges": {r: (boundaries[j], boundaries[j + 1]) for j, r in enumerate(participants)},
            "head_cost": {
                "mask": cost_mask[h_split],
                "attn": cost_attn[h_split],
                "total": cost_total[h_split],
            },
            "candidates_top5": [
                {
                    "helpers": list(c["participants"][1:]),
                    "planned_max_ms": c["planned_max"],
                    "T_H_ms": c["T_H"],
                }
                for c in candidates[:5]
            ],
            "max_nonparticipant_load_ms": max_non_part,
            "best_planned_max_ms": bound_under_K,
            "best_planned_max_speedup": speedup_bound_under_K,
            "global_one_split_ceiling_ms": global_bound,
            "global_one_split_ceiling_speedup": speedup_bound_global,
        })

    diagnostics["planned_per_rank_ms"] = list(current_loads)
    predicted_max = max(current_loads)
    return SplitPlan(
        units_per_rank=units_per_rank,
        predicted_max_ms=predicted_max,
        baseline_max_ms=baseline_max,
        speedup=baseline_max / predicted_max if predicted_max > 0 else 1.0,
        diagnostics=diagnostics,
    )


# ---------------- CLI helpers ----------------

def _flatten_density(nested) -> List[float]:
    out: List[float] = []
    if isinstance(nested, (list, tuple)):
        for x in nested:
            out.extend(_flatten_density(x))
    else:
        out.append(float(nested))
    return out


def load_density_log(path: str) -> Dict[int, List[Tuple[int, List[float]]]]:
    by_layer: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            layer = int(row["layer"])
            ts = int(row["timestep"])
            vals = _flatten_density(row["density"])
            by_layer[layer].append((ts, vals))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda r: -r[0])
    return by_layer


def pick_prev_density(
    log: Dict[int, List[Tuple[int, List[float]]]], layer: int, timestep: int
) -> Optional[Tuple[int, List[float]]]:
    """Returns (prev_ts, densities) — the latest entry with ts > timestep,
    or None if no such entry exists. The ts is included so callers can
    print/log it without having to re-scan the log."""
    rows = log.get(layer, [])
    prev = None
    for ts, vals in rows:
        if ts > timestep:
            prev = (ts, vals)
        else:
            break
    return prev


def _format_unit(u: ComputeUnit) -> str:
    if u.role == "full":
        return f"h{u.global_head}"
    if u.role == "split_owner":
        return f"h{u.global_head}q[{u.q_lo}:{u.q_hi})so"
    return f"h{u.global_head}q[{u.q_lo}:{u.q_hi})hp(o={u.owner_rank})"


def _per_rank_breakdown(units: List[ComputeUnit], cost_mask: List[float]) -> Tuple[float, float, float]:
    """Returns (mask_sum, attn_sum, total) for one rank's units."""
    mask_sum = 0.0
    attn_sum = 0.0
    for u in units:
        if u.role in ("full", "split_owner"):
            mask_sum += cost_mask[u.global_head]
        # u.cost_ms = (mask if owner) + attn share
        if u.role == "full":
            attn_sum += u.cost_ms - cost_mask[u.global_head]
        elif u.role == "split_owner":
            attn_sum += u.cost_ms - cost_mask[u.global_head]
        else:  # helper
            attn_sum += u.cost_ms
    return mask_sum, attn_sum, mask_sum + attn_sum


def format_baseline_table(plan: SplitPlan, cost_mask: List[float]) -> str:
    out = []
    out.append("[predicted baseline: greedy_unequal]")
    out.append("rank  heads                                    cost_mask  cost_attn   total")
    out.append("-" * 78)
    baseline_loads = plan.diagnostics["baseline_per_rank_ms"]
    baseline_assigned = plan.diagnostics["baseline_assigned_heads"]
    for r, heads in enumerate(baseline_assigned):
        m = sum(cost_mask[h] for h in heads)
        t = baseline_loads[r]
        a = t - m
        head_str = "[" + ",".join(str(h) for h in heads) + "]"
        out.append(f"  {r}   {head_str:<40} {m:>9.2f} {a:>10.2f}  {t:>6.2f}")
    out.append(f"predicted baseline max = {plan.baseline_max_ms:.2f} ms")
    return "\n".join(out)


def format_planned_table(plan: SplitPlan, cost_mask: List[float]) -> str:
    out = []
    n_splits = len(plan.diagnostics["splits"])
    if n_splits == 0:
        out.append("[predicted: greedy_unequal (no split — load already balanced or no qualifying head)]")
    else:
        out.append(f"[predicted: greedy_unequal + {n_splits} split]")
    out.append("rank  units                                              cost_mask  cost_attn   total")
    out.append("-" * 86)
    world = len(plan.units_per_rank)
    planned_loads = plan.diagnostics["planned_per_rank_ms"]
    for r in range(world):
        units = plan.units_per_rank[r]
        m, a, _ = _per_rank_breakdown(units, cost_mask)
        t = planned_loads[r]
        unit_strs = " ".join(_format_unit(u) for u in units)
        out.append(f"  {r}   {unit_strs[:50]:<50} {m:>9.2f} {a:>10.2f}  {t:>6.2f}")
    if n_splits > 0:
        sp = plan.diagnostics["splits"][-1]
        bottleneck_rank = max(range(world), key=lambda r: planned_loads[r])
        out.append(
            f"predicted max = {plan.predicted_max_ms:.2f} ms  "
            f"(limited by rank {bottleneck_rank}; speedup vs baseline: {plan.speedup:.3f}x)"
        )
        out.append(
            f"best chosen split predicted max (max_helpers_per_split="
            f"{sp['max_helpers_per_split']}): {sp['best_planned_max_ms']:.2f} ms  "
            f"({sp['best_planned_max_speedup']:.3f}x)"
        )
        out.append(
            f"global one-split ceiling (water-fill over all ranks) = "
            f"{sp['global_one_split_ceiling_ms']:.2f} ms  "
            f"({sp['global_one_split_ceiling_speedup']:.3f}x)"
        )
    else:
        out.append(f"predicted max = {plan.predicted_max_ms:.2f} ms (no split applied)")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(
        description="Split planner: predict the speedup of a post-permutation Q-split.",
    )
    p.add_argument("--density-log", required=True)
    p.add_argument("--cost-model-json", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--timestep", type=int, required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--world-size", type=int, required=True)
    p.add_argument("--delta", type=float, default=0.10)
    p.add_argument("--max-helpers", type=int, default=2)
    p.add_argument("--max-splits", type=int, default=1)
    p.add_argument("--min-heads-per-rank", type=int, default=1)
    p.add_argument("--q-granularity", type=int, default=1)
    p.add_argument("--min-improvement-ms", type=float, default=0.5)
    p.add_argument("--output-json", default=None,
                   help="If set, dump the full diagnostics + plan as JSON.")
    args = p.parse_args()

    log = load_density_log(args.density_log)
    prev = pick_prev_density(log, args.layer, args.timestep)
    if prev is None:
        raise SystemExit(
            f"No prior density for layer={args.layer} timestep={args.timestep} "
            f"in {args.density_log}"
        )
    prev_ts, densities = prev

    with open(args.cost_model_json) as f:
        cost_model = json.load(f)

    plan = plan_with_splits(
        densities=densities,
        cost_model=cost_model,
        seq_len=args.seq_len,
        world_size=args.world_size,
        delta=args.delta,
        max_helpers_per_split=args.max_helpers,
        max_splits_per_plan=args.max_splits,
        min_heads_per_rank=args.min_heads_per_rank,
        q_granularity=args.q_granularity,
        min_improvement_ms=args.min_improvement_ms,
    )

    cost_mask, _, _ = head_costs(densities, cost_model, args.seq_len)
    print(f"densities (layer={args.layer}, ts={args.timestep}, prev_ts={prev_ts}):")
    print("  " + ", ".join(f"h{i}={d:.4f}" for i, d in enumerate(densities)))
    print()
    print(format_baseline_table(plan, cost_mask))
    print()
    print(format_planned_table(plan, cost_mask))

    if args.output_json:
        # Dump plan + diagnostics. ComputeUnits → dicts.
        dumpable = {
            "predicted_max_ms": plan.predicted_max_ms,
            "baseline_max_ms": plan.baseline_max_ms,
            "speedup": plan.speedup,
            "units_per_rank": [
                [asdict(u) for u in rank_units]
                for rank_units in plan.units_per_rank
            ],
            "diagnostics": plan.diagnostics,
        }
        with open(args.output_json, "w") as f:
            json.dump(dumpable, f, indent=2)
        print(f"\nwrote diagnostics JSON: {args.output_json}")


if __name__ == "__main__":
    main()

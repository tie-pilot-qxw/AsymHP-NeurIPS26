"""Predict-only placement ablation — find (layer, timestep) entries that best
showcase the gradual gain ladder:

  1. contiguous            : plan = [r*hpr:(r+1)*hpr]
  2. equal-shuffle (prev)  : plan = equal-LPT(prev_density, cost = density)
  3. density-only  (prev)  : plan = LPT(prev_density, cost = density)         (variable heads)
  4. AsymHP        (prev)  : plan = LPT(prev_density, cost = cost_model)
  5. oracle        (cur)   : plan = LPT(cur_density,  cost = cost_model)

All four are *evaluated* with the same "ground truth": cost_model applied to
the CURRENT-step density. So makespan reflects what the real run would see.

Usage:
  python lb/predict_ablation.py \\
      --density-log <jsonl> --cost-model-json <json> \\
      --seq-len 32760 --world-size 4 --top-k 10
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent))
import json
from split_planner import (
    head_costs,
    greedy_lpt_assignment as lpt_assignment,
    load_density_log,
)


def contig_plan(num_heads: int, world_size: int) -> List[List[int]]:
    if num_heads % world_size != 0:
        raise SystemExit(f"contig requires num_heads ({num_heads}) % world_size ({world_size}) == 0")
    hpr = num_heads // world_size
    return [list(range(r * hpr, (r + 1) * hpr)) for r in range(world_size)]


def equal_lpt_plan(costs: Sequence[float], world_size: int) -> List[List[int]]:
    n = len(costs)
    if n % world_size != 0:
        raise SystemExit(f"equal-lpt requires {n} % {world_size} == 0")
    cap = n // world_size
    loads = [0.0] * world_size
    out: List[List[int]] = [[] for _ in range(world_size)]
    for h in sorted(range(n), key=lambda i: -costs[i]):
        cand = [r for r in range(world_size) if len(out[r]) < cap]
        best = min(cand, key=lambda r: loads[r])
        out[best].append(h)
        loads[best] += costs[h]
    return [sorted(h) for h in out]


def loads_for(plan: List[List[int]], truth_costs: Sequence[float]) -> List[float]:
    return [sum(truth_costs[h] for h in heads) for heads in plan]


def pick_density_at(log, layer: int, ts: int) -> Optional[List[float]]:
    """Exact-match lookup."""
    for t, vals in log.get(layer, []):
        if t == ts:
            return vals
    return None


def pick_prev_density(log, layer: int, ts: int) -> Optional[List[float]]:
    """Strictly-larger ts (prev step in denoising order)."""
    rows = log.get(layer, [])
    prev = None
    for t, vals in rows:  # iterating desc
        if t > ts:
            prev = vals
        else:
            break
    return prev


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--density-log", required=True)
    p.add_argument("--cost-model-json", required=True)
    p.add_argument("--seq-len", type=int, required=True,
                   help="Seq length the cost model is evaluated at. Use the .pt's metadata seq_len.")
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--min-heads-per-rank", type=int, default=1)
    p.add_argument("--top-k", type=int, default=10,
                   help="How many top candidates to list per ranking criterion.")
    p.add_argument("--csv", default=None, help="Optional dump of per-entry ablation rows.")
    p.add_argument("--max-skew",
                   type=float,
                   default=None,
                   help="Optional filter: skip entries where max(d)/mean(d) >= this. "
                        "A common cause of low cost-model gain is a single dominant head; "
                        "e.g. 2.5 keeps moderately-skewed entries.")
    args = p.parse_args()

    with open(args.cost_model_json) as f:
        cost_model = json.load(f)

    log = load_density_log(args.density_log)

    # Flatten and compute, for each (layer, ts) that has a prev neighbor.
    rows: List[Dict] = []
    for layer in sorted(log.keys()):
        per_layer = log[layer]  # sorted desc by ts
        for i, (ts, cur) in enumerate(per_layer):
            prev = None
            if i > 0:
                prev = per_layer[i - 1][1]   # the entry with strictly larger ts (denoising prev)
            if prev is None:
                continue
            n = len(cur)
            if n != len(prev):
                continue
            if args.min_heads_per_rank * args.world_size > n:
                continue

            # Skew filter (on current density).
            mean_d = sum(cur) / n
            mx_d = max(cur)
            skew = mx_d / mean_d if mean_d > 0 else float("inf")
            if args.max_skew is not None and skew >= args.max_skew:
                continue

            # Cost vectors.
            _, _, truth_total = head_costs(cur, cost_model, args.seq_len)         # ground truth for eval
            _, _, prev_costmodel = head_costs(prev, cost_model, args.seq_len)     # planning cost for sys
            cur_costmodel = truth_total                                            # planning cost for oracle

            # Plans.
            plan_contig       = contig_plan(n, args.world_size)
            plan_eqshuffle    = equal_lpt_plan(prev, args.world_size)                              # equal heads, density cost
            plan_density_prev = lpt_assignment(prev, args.world_size, args.min_heads_per_rank)[0]
            plan_sys_prev     = lpt_assignment(prev_costmodel, args.world_size, args.min_heads_per_rank)[0]
            plan_oracle_cur   = lpt_assignment(cur_costmodel, args.world_size, args.min_heads_per_rank)[0]

            # Evaluate makespan under truth (cost_model(current)).
            mk_contig    = max(loads_for(plan_contig, truth_total))
            mk_eqshuffle = max(loads_for(plan_eqshuffle, truth_total))
            mk_density   = max(loads_for(plan_density_prev, truth_total))
            mk_sys       = max(loads_for(plan_sys_prev, truth_total))
            mk_oracle    = max(loads_for(plan_oracle_cur, truth_total))

            rows.append({
                "layer": layer, "timestep": ts,
                "skew": skew, "n_heads": n, "max_density": mx_d, "mean_density": mean_d,
                "mk_contig": mk_contig,
                "mk_eqshuffle": mk_eqshuffle,
                "mk_density": mk_density,
                "mk_sys": mk_sys,
                "mk_oracle": mk_oracle,
                "spd_eqshuffle": mk_contig / mk_eqshuffle,
                "spd_density":   mk_contig / mk_density,
                "spd_sys":       mk_contig / mk_sys,
                "spd_oracle":    mk_contig / mk_oracle,
                "gain_costmodel_pct": 100.0 * (mk_density - mk_sys) / mk_density,  # >0 means cost model helped
                "gain_oracle_pct":    100.0 * (mk_sys - mk_oracle) / mk_sys,        # >0 means current density helped
            })

    if not rows:
        raise SystemExit("no entries survived (check --density-log and --max-skew).")

    print(f"# scanned {len(rows)} (layer, timestep) entries with prev-step neighbor "
          f"(world_size={args.world_size}, seq_len={args.seq_len})")
    print(f"# planning costs: density-only=density, sys=cost_model(prev), oracle=cost_model(current)")
    print(f"# eval (truth): cost_model(current)")
    print()

    def _print(rows_subset: List[Dict], title: str):
        print(f"=== {title} (top {args.top_k}) ===")
        hdr = (f"{'layer':>5} {'ts':>5} {'skew':>5} "
               f"{'contig':>7} {'eqshuf':>7} {'density':>8} {'sys':>6} {'oracle':>7}  "
               f"{'spd_eq':>7} {'spd_den':>8} {'spd_sys':>8} {'spd_or':>7}  "
               f"{'cm%':>6} {'orc%':>6}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows_subset:
            print(f"{r['layer']:>5} {r['timestep']:>5} {r['skew']:>5.2f} "
                  f"{r['mk_contig']:>7.2f} {r['mk_eqshuffle']:>7.2f} {r['mk_density']:>8.2f} {r['mk_sys']:>6.2f} {r['mk_oracle']:>7.2f}  "
                  f"{r['spd_eqshuffle']:>6.2f}x {r['spd_density']:>7.2f}x {r['spd_sys']:>7.2f}x {r['spd_oracle']:>6.2f}x  "
                  f"{r['gain_costmodel_pct']:>5.1f}% {r['gain_oracle_pct']:>5.1f}%")
        print()

    # Three rankings: best to showcase cost-model, best to showcase oracle, combined.
    by_cm = sorted(rows, key=lambda r: -r["gain_costmodel_pct"])
    by_or = sorted(rows, key=lambda r: -r["gain_oracle_pct"])
    by_combined = sorted(
        rows,
        key=lambda r: -(r["gain_costmodel_pct"] + r["gain_oracle_pct"]),
    )
    by_total_speedup = sorted(rows, key=lambda r: -r["spd_oracle"])

    # Strict 5-row monotone ladder: contig > eqshuffle >= density >= sys >= oracle.
    # (>= rather than > on the inner steps: it's OK if a later step plateaus,
    # but no row may go *backwards*.) Ranked by total span = (contig - oracle).
    monotone = [r for r in rows
                if r["mk_contig"]    >= r["mk_eqshuffle"]
                and r["mk_eqshuffle"] >= r["mk_density"]
                and r["mk_density"]   >= r["mk_sys"]
                and r["mk_sys"]       >= r["mk_oracle"]]
    by_monotone = sorted(monotone, key=lambda r: -(r["mk_contig"] - r["mk_oracle"]))

    _print(by_monotone[:args.top_k],   "Strict-monotone 5-row ladder (contig >= eqshuf >= density >= sys >= oracle), by absolute span")
    _print(by_cm[:args.top_k],         "Largest cost-model gain  (sys vs density-only, on current truth)")
    _print(by_or[:args.top_k],         "Largest oracle gain      (oracle vs sys, on current truth)")
    _print(by_combined[:args.top_k],   "Largest combined gain    (cost-model + oracle, both >0 ideal)")
    _print(by_total_speedup[:args.top_k], "Largest absolute speedup vs contiguous (oracle row)")
    print(f"# {len(monotone)}/{len(rows)} entries satisfy strict-monotone ladder")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(by_combined)
        print(f"[csv] wrote {len(rows)} rows (sorted by combined gain) to {out}")


if __name__ == "__main__":
    main()

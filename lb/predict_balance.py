"""Predict-only LB comparison — no GPU, no torchrun.

Reads a density log (jsonl) + cost model (json) and reports, for each
(layer, timestep) entry, what each balance strategy would assign and how
balanced the result would be. Aggregates across all entries with --summary.

Strategies covered:
  - contiguous       : heads [r*hpr : (r+1)*hpr] per rank (no density needed)
  - greedy_equal     : LPT with cap = num_heads / world_size
  - greedy_unequal   : LPT, variable heads per rank (>= --min-heads-per-rank)
  - split            : greedy_unequal + post-permutation Q-split (planner)

Cost = mask + attention, using the cost model slopes (matches the bench
script). Without --cost-model-json, falls back to using density directly
as a relative cost — split is then disabled.

Caveats: cost only models mask + attention. Real per-rank totals also
include a2a_in / a2a_out / inv_perm; those are NOT predicted here.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from split_planner import (
    head_costs,
    greedy_lpt_assignment as lpt_assignment,
    plan_with_splits,
    load_density_log,
)

METHODS_DEFAULT = ["contiguous", "greedy_equal", "greedy_unequal", "split"]


def contiguous_assignment(num_heads: int, world_size: int) -> Optional[List[List[int]]]:
    if num_heads % world_size != 0:
        return None
    hpr = num_heads // world_size
    return [list(range(r * hpr, (r + 1) * hpr)) for r in range(world_size)]


def greedy_equal_assignment(
    costs: Sequence[float], world_size: int
) -> Optional[Tuple[List[List[int]], List[float]]]:
    n = len(costs)
    if n % world_size != 0:
        return None
    cap = n // world_size
    loads = [0.0] * world_size
    assigned: List[List[int]] = [[] for _ in range(world_size)]
    order = sorted(range(n), key=lambda i: -costs[i])
    for h in order:
        cand = [r for r in range(world_size) if len(assigned[r]) < cap]
        best = min(cand, key=lambda r: loads[r])
        assigned[best].append(h)
        loads[best] += costs[h]
    return [sorted(h) for h in assigned], loads


def loads_for(assigned: List[List[int]], costs: Sequence[float]) -> List[float]:
    return [sum(costs[h] for h in heads) for heads in assigned]


def imbalance(loads: Sequence[float]) -> Tuple[float, float, float]:
    mx, mn = max(loads), min(loads)
    return mx, mn, (mx / mn if mn > 0 else float("inf"))


def costs_from(densities: Sequence[float], cost_model: Optional[Dict], seq_len: int):
    if cost_model is None:
        d = [float(x) for x in densities]
        return d, d, [0.0] * len(d)  # cost_total, cost_attn, cost_mask (mask=0 fallback)
    cm, ca, ct = head_costs(densities, cost_model, seq_len)
    return ct, ca, cm


def predict_one(
    densities: Sequence[float],
    cost_model: Optional[Dict],
    seq_len: int,
    world_size: int,
    methods: Sequence[str],
    *,
    max_helpers: int = 2,
    max_splits: int = 1,
    min_heads_per_rank: int = 1,
    q_granularity: int = 1,
    min_improvement_ms: float = 0.5,
    delta: float = 0.10,
) -> Dict[str, Optional[Dict]]:
    cost_total, _, _ = costs_from(densities, cost_model, seq_len)
    n = len(densities)
    out: Dict[str, Optional[Dict]] = {}

    if "contiguous" in methods:
        a = contiguous_assignment(n, world_size)
        out["contiguous"] = (
            None if a is None else {"loads": loads_for(a, cost_total), "n_assigned": [len(h) for h in a]}
        )

    if "greedy_equal" in methods:
        r = greedy_equal_assignment(cost_total, world_size)
        out["greedy_equal"] = (
            None if r is None else {"loads": r[1], "n_assigned": [len(h) for h in r[0]]}
        )

    if "greedy_unequal" in methods:
        if min_heads_per_rank * world_size > n:
            out["greedy_unequal"] = None
        else:
            a, ld = lpt_assignment(cost_total, world_size, min_heads_per_rank)
            out["greedy_unequal"] = {"loads": ld, "n_assigned": [len(h) for h in a]}

    if "split" in methods:
        if cost_model is None or min_heads_per_rank * world_size > n:
            out["split"] = None
        else:
            plan = plan_with_splits(
                densities=densities, cost_model=cost_model, seq_len=seq_len,
                world_size=world_size, delta=delta,
                max_helpers_per_split=max_helpers, max_splits_per_plan=max_splits,
                min_heads_per_rank=min_heads_per_rank,
                q_granularity=q_granularity, min_improvement_ms=min_improvement_ms,
            )
            ld = plan.diagnostics["planned_per_rank_ms"]
            out["split"] = {
                "loads": ld,
                "n_splits": len(plan.diagnostics["splits"]),
                "splits": [
                    {
                        "head": s["head"], "owner": s["owner"], "helpers": list(s["helpers"]),
                    }
                    for s in plan.diagnostics["splits"]
                ],
            }
    return out


def fmt_loads(loads: Sequence[float], width: int = 6) -> str:
    return "[" + ", ".join(f"{x:{width}.2f}" for x in loads) + "]"


def print_entry(
    layer: int, ts: int, results: Dict[str, Optional[Dict]],
    densities: Sequence[float], cost_model: Optional[Dict], seq_len: int,
    world_size: int, methods: Sequence[str], baseline: str,
):
    cost_total, _, _ = costs_from(densities, cost_model, seq_len)
    unit = "ms" if cost_model is not None else "den"
    total = sum(cost_total)
    ideal = total / world_size
    print(
        f"\n=== layer={layer} timestep={ts}  world={world_size}  "
        f"total={total:.2f}{unit}  ideal={ideal:.2f}{unit}/rank  "
        f"num_heads={len(densities)} ==="
    )
    base = results.get(baseline)
    base_max = max(base["loads"]) if base else None

    header = f"{'method':<16}  {'max':>8}  {'min':>8}  {'max/min':>7}  {'vs '+baseline:>9}  loads"
    print(header)
    print("-" * (len(header) + max(0, world_size * 8)))
    for m in methods:
        r = results.get(m)
        if r is None:
            print(f"{m:<16}  N/A")
            continue
        ld = r["loads"]
        mx, mn, ratio = imbalance(ld)
        spd = (base_max / mx) if (base_max is not None and mx > 0) else float("nan")
        suffix = ""
        if m == "split" and "n_splits" in r:
            if r["n_splits"] == 0:
                suffix = "  (no split applied)"
            else:
                bits = []
                for s in r["splits"]:
                    bits.append(f"h{s['head']}@r{s['owner']}->{s['helpers']}")
                suffix = f"  ({r['n_splits']} split: " + "; ".join(bits) + ")"
        print(f"{m:<16}  {mx:>8.2f}  {mn:>8.2f}  {ratio:>7.3f}  {spd:>8.3f}x  {fmt_loads(ld)}{suffix}")


def aggregate(
    rows: List[Dict], methods: Sequence[str], baseline: str,
) -> List[Dict]:
    """Per-method stats across entries: max load p50/p90, imbalance p50/p90,
    speedup mean / p50."""
    # rows: each row = {method: {"max": ..., "ratio": ..., "speedup": ...}, "layer", "ts"}
    out = []
    for m in methods:
        maxes = [row[m]["max"] for row in rows if row.get(m)]
        ratios = [row[m]["ratio"] for row in rows if row.get(m)]
        spds = [row[m]["speedup"] for row in rows if row.get(m) and row[m]["speedup"] == row[m]["speedup"]]  # filter NaN
        n = len(maxes)
        if n == 0:
            out.append({"method": m, "n": 0})
            continue

        def pct(xs, p):
            xs = sorted(xs)
            k = (len(xs) - 1) * p
            lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
            return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

        out.append({
            "method": m,
            "n": n,
            "max_mean": statistics.mean(maxes),
            "max_p50": pct(maxes, 0.5),
            "max_p90": pct(maxes, 0.9),
            "ratio_mean": statistics.mean(ratios),
            "ratio_p50": pct(ratios, 0.5),
            "ratio_p90": pct(ratios, 0.9),
            "speedup_mean": statistics.mean(spds) if spds else float("nan"),
            "speedup_p50": pct(spds, 0.5) if spds else float("nan"),
        })
    return out


def print_summary(stats_by_world: Dict[int, List[Dict]], baseline: str, unit: str):
    print(f"\n=== summary across all entries (baseline={baseline}) ===")
    header = (
        f"{'world':>5}  {'method':<16}  {'n':>4}  "
        f"{'max_mean':>9}  {'max_p50':>9}  {'max_p90':>9}  "
        f"{'ratio_mean':>10}  {'ratio_p50':>9}  {'ratio_p90':>9}  "
        f"{'spdup_mean':>10}  {'spdup_p50':>9}"
    )
    print(header)
    print("-" * len(header))
    for ws in sorted(stats_by_world.keys()):
        for s in stats_by_world[ws]:
            if s["n"] == 0:
                print(f"{ws:>5}  {s['method']:<16}  {0:>4}  N/A")
                continue
            print(
                f"{ws:>5}  {s['method']:<16}  {s['n']:>4}  "
                f"{s['max_mean']:>9.2f}  {s['max_p50']:>9.2f}  {s['max_p90']:>9.2f}  "
                f"{s['ratio_mean']:>10.3f}  {s['ratio_p50']:>9.3f}  {s['ratio_p90']:>9.3f}  "
                f"{s['speedup_mean']:>10.3f}  {s['speedup_p50']:>9.3f}"
            )
        if ws != max(stats_by_world.keys()):
            print()
    print(f"  (max/ratio in {unit}; spdup is per-entry max_baseline / max_method)")


def parse_world_sizes(spec: str) -> List[int]:
    """Accepts: '6' | '2,4,6,8' | '2-8' | '2-8:2'. Returns sorted unique list."""
    out: set = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            range_part, _, step_part = tok.partition(":")
            lo_s, _, hi_s = range_part.partition("-")
            lo, hi = int(lo_s), int(hi_s)
            step = int(step_part) if step_part else 1
            if lo <= 0 or hi < lo or step <= 0:
                raise ValueError(f"bad world-size range '{tok}'")
            out.update(range(lo, hi + 1, step))
        else:
            v = int(tok)
            if v <= 0:
                raise ValueError(f"world-size must be positive, got {v}")
            out.add(v)
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--density-log", required=True)
    p.add_argument("--cost-model-json", default=None,
                   help="Optional. Without it, density is used as cost and split is disabled.")
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--world-size", required=True,
                   help="Single value, comma list, or range. Examples: '6', '2,4,6,8', '2-8', '2-8:2'.")
    p.add_argument("--methods", default=",".join(METHODS_DEFAULT),
                   help=f"Comma-separated subset of {METHODS_DEFAULT}.")
    p.add_argument("--baseline", default="contiguous",
                   help="Method to compute speedup against (default: contiguous).")
    p.add_argument("--layer", type=int, default=None,
                   help="Filter to this layer only.")
    p.add_argument("--timestep", type=int, default=None,
                   help="Filter to this timestep only (used together with --layer).")
    p.add_argument("--max-entries", type=int, default=0,
                   help="Cap entries printed (0 = all). Aggregation still uses all.")
    p.add_argument("--summary", action="store_true",
                   help="Print aggregate stats across all entries; suppress per-entry detail unless --verbose.")
    p.add_argument("--verbose", action="store_true",
                   help="With --summary, also print per-entry tables.")
    p.add_argument("--max-helpers", type=int, default=2)
    p.add_argument("--max-splits", type=int, default=1)
    p.add_argument("--min-heads-per-rank", type=int, default=1)
    p.add_argument("--split-q-granularity", type=int, default=1)
    p.add_argument("--split-min-improvement-ms", type=float, default=0.5)
    p.add_argument("--split-delta", type=float, default=0.10)
    p.add_argument("--csv", default=None,
                   help="If set, dump per-entry per-method rows to this CSV.")
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS_DEFAULT:
            raise SystemExit(f"unknown method '{m}'; pick from {METHODS_DEFAULT}")
    if args.baseline not in methods:
        raise SystemExit(f"--baseline '{args.baseline}' must be one of --methods {methods}")

    try:
        world_sizes = parse_world_sizes(args.world_size)
    except ValueError as e:
        raise SystemExit(f"bad --world-size: {e}")
    if not world_sizes:
        raise SystemExit("--world-size produced no values")

    cost_model = None
    if args.cost_model_json:
        with open(args.cost_model_json) as f:
            cost_model = json.load(f)
    if "split" in methods and cost_model is None:
        print("[warn] --cost-model-json not given; 'split' will be skipped (N/A).", file=sys.stderr)

    log = load_density_log(args.density_log)
    # Flatten (layer, ts, densities) and sort by layer, ts ascending.
    entries: List[Tuple[int, int, List[float]]] = []
    for layer, rows in log.items():
        for ts, vals in rows:
            if args.layer is not None and layer != args.layer:
                continue
            if args.timestep is not None and ts != args.timestep:
                continue
            entries.append((layer, ts, vals))
    entries.sort(key=lambda e: (e[0], e[1]))
    if not entries:
        raise SystemExit("no entries match --layer/--timestep filter")

    unit = "ms" if cost_model is not None else "den"
    csv_rows: List[Dict] = []
    stats_by_world: Dict[int, List[Dict]] = {}

    print_per_entry = not args.summary or args.verbose
    n_to_print = args.max_entries if args.max_entries > 0 else len(entries)

    for ws in world_sizes:
        if len(world_sizes) > 1 and print_per_entry:
            print(f"\n############### world_size = {ws} ###############")
        agg_rows: List[Dict] = []
        for i, (layer, ts, dens) in enumerate(entries):
            results = predict_one(
                dens, cost_model, args.seq_len, ws, methods,
                max_helpers=args.max_helpers, max_splits=args.max_splits,
                min_heads_per_rank=args.min_heads_per_rank,
                q_granularity=args.split_q_granularity,
                min_improvement_ms=args.split_min_improvement_ms,
                delta=args.split_delta,
            )
            if print_per_entry and i < n_to_print:
                print_entry(layer, ts, results, dens, cost_model,
                            args.seq_len, ws, methods, args.baseline)

            base = results.get(args.baseline)
            base_max = max(base["loads"]) if base else None
            per: Dict = {}
            for m in methods:
                r = results.get(m)
                if r is None:
                    per[m] = None
                    continue
                mx, mn, ratio = imbalance(r["loads"])
                spd = (base_max / mx) if (base_max and mx > 0) else float("nan")
                per[m] = {"max": mx, "min": mn, "ratio": ratio, "speedup": spd}
                if args.csv:
                    csv_rows.append({
                        "world_size": ws,
                        "layer": layer, "timestep": ts, "method": m,
                        "max": mx, "min": mn, "ratio": ratio,
                        "speedup_vs_baseline": spd,
                        "loads": ";".join(f"{x:.4f}" for x in r["loads"]),
                        "n_splits": r.get("n_splits", 0) if m == "split" else 0,
                    })
            per["layer"] = layer
            per["ts"] = ts
            agg_rows.append(per)
        stats_by_world[ws] = aggregate(agg_rows, methods, args.baseline)

    if args.summary:
        print_summary(stats_by_world, args.baseline, unit)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n[csv] wrote {len(csv_rows)} rows to {path}")


if __name__ == "__main__":
    main()

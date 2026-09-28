"""Does the placement need to be updated over time? (paper Appendix D.3)

Trace-driven analysis, in the same spirit as the paper's whole-head granularity
study: no new GPU run, we replay the recorded per-head density trace through the
planner and score the resulting placements under the fitted cost model.

Three quantities, for each pair of adjacent denoising steps of each layer:

  1. how often the cost-optimal placement actually CHANGES between adjacent steps
     (if it never changed, a static placement would suffice);
  2. what AsymHP's one-step-lag predictor costs: score the placement built from
     step t-1's density against step t's realized cost, relative to the placement
     built from step t's own density (the oracle of the paper's Table 1);
  3. what NOT re-planning costs: keep the deterministic near-even initial
     placement (paper Section 3.2) in force for the whole
     trajectory and score it the same way.

Defaults use the 720p/125-frame Wan2.1-1.3B trace (seq_len 115200, W=4).
"""
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from split_planner import (  # noqa: E402
    greedy_lpt_assignment,
    head_costs,
    load_density_log,
    rank_intercept,
)

ROOT = Path(__file__).resolve().parents[1]

COST_JSON = ROOT / "result/wan/t2v/sap_1.3b/maskgen_aware_cost.json"
DENSITY_LOG = ROOT / "data/traces/wan2.1-1.3b_720p_125f.jsonl"
SEQ_LEN = 115200
WORLD = 4


def main():
    cost = json.load(open(COST_JSON))
    icpt = rank_intercept(cost)
    log = load_density_log(DENSITY_LOG)

    def makespan(plan, c):
        return max(icpt + sum(c[h] for h in rank) for rank in plan)

    def shape(plan):
        return tuple(tuple(sorted(r)) for r in plan)

    # the deterministic near-even initial placement (paper Section 3.2)
    base_n, rem = divmod(len(next(iter(log.values()))[0][1]), WORLD)
    static, _s0 = [], 0
    for r in range(WORLD):
        n = base_n + (1 if r < rem else 0)
        static.append(list(range(_s0, _s0 + n))); _s0 += n

    changed = same = 0
    lag_penalty, frozen_penalty = [], []
    # Aggregate (whole-trajectory) makespan sums. This is the headline number:
    # a mean of per-invocation RATIOS is not the ratio of the totals, and what a
    # deployment actually pays is the summed critical path over the trajectory.
    sum_best = sum_lag = sum_frozen = 0.0

    for _layer, series in log.items():
        # denoising order: timestep decreases as the trajectory progresses
        series = sorted(series, key=lambda x: -x[0])
        plans, costs = [], []
        for _ts, dens in series:
            if len(dens) % WORLD:
                continue
            c = head_costs(dens, cost, SEQ_LEN)[2]
            plan, _ = greedy_lpt_assignment(c, WORLD, 1)
            plans.append(plan)
            costs.append(c)
        for i in range(1, len(plans)):
            if shape(plans[i]) == shape(plans[i - 1]):
                same += 1
            else:
                changed += 1
            # step t scored under: previous step's plan, and the first step's plan
            best = makespan(plans[i], costs[i])
            lag = makespan(plans[i - 1], costs[i])
            frozen = makespan(static, costs[i])
            lag_penalty.append(lag / best)
            frozen_penalty.append(frozen / best)
            sum_best += best
            sum_lag += lag
            sum_frozen += frozen

    total = changed + same
    lag_penalty.sort()
    frozen_penalty.sort()
    p99 = lambda v: v[int(0.99 * len(v))]  # noqa: E731

    print(f"720p trace, seq_len={SEQ_LEN}, W={WORLD}, "
          f"{total} adjacent step pairs over {len(log)} layers\n")
    print(f"  optimal placement differs between adjacent steps: "
          f"{changed / total * 100:.1f}% of invocations\n")
    print("  aggregate makespan summed over the whole trajectory:")
    print(f"    current-step density (oracle) : {sum_best:9.1f} ms   1.0000x")
    print(f"    previous-step density (AsymHP): {sum_lag:9.1f} ms   "
          f"{sum_lag / sum_best:.4f}x")
    print(f"    static initial placement      : {sum_frozen:9.1f} ms   "
          f"{sum_frozen / sum_best:.4f}x\n")
    print("  per-invocation ratios (for reference):")
    print(f"    previous-step density (AsymHP): mean {st.mean(lag_penalty):.4f}x  "
          f"p99 {p99(lag_penalty):.4f}x  max {lag_penalty[-1]:.4f}x")
    print(f"    static initial placement      : mean {st.mean(frozen_penalty):.4f}x  "
          f"p99 {p99(frozen_penalty):.4f}x  max {frozen_penalty[-1]:.4f}x")


if __name__ == "__main__":
    main()

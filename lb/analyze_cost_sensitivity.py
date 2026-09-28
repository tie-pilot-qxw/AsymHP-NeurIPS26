"""Cost-model sensitivity (paper Appendix Table 9): how much does placement
degrade when the fitted coefficients are wrong?

Mechanism this probes
---------------------
Per-head cost is  c_h = A + B * rho_h  with
    A = alpha_mask * N          (a fixed floor paid by EVERY head)
    B = alpha_attn * N^2        (scales with that head's density)
Greedy LPT placement is invariant to a global scaling of c, so the absolute
coefficients are irrelevant; only the SHAPE matters, captured by the single ratio

    r = alpha_mask / alpha_attn        (units: tokens)

r sets how strongly a rank is penalized for holding MANY heads vs holding DENSE
heads. r is a property of the sparse kernels + head geometry, NOT of the model --
which is why one calibration transfers across models that share them (see
analyze_cost_portability.py).

This script deliberately mis-scales r by a factor f and reports the realized
makespan regret vs the correctly-calibrated placement, scored under the TRUE
cost. That answers "how sensitive is placement quality to coefficient error"
directly, instead of relying on three real fits that happen to be similar.
"""
import copy
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
DENSITY_LOG = ROOT / "data/traces/wan2.1-1.3b_480p_253f.jsonl"
SEQ_LEN = 99840
WORLD = 4
FACTORS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

COST_SOURCES = {
    "Wan-1.3B": ROOT / "result/wan/t2v/sap_1.3b/maskgen_aware_cost.json",
    "Wan-14B": ROOT / "result/wan/t2v/sap_14b/maskgen_aware_cost.json",
    "HunyuanVideo": ROOT / "result/hyvideo/t2v/sap/maskgen_aware_cost.json",
}


def report_shape_ratios():
    """Show that the three real stacks differ mostly by a SCALE factor (which
    placement ignores) and only mildly in the shape ratio r (which it sees)."""
    print("### Fitted coefficients across stacks (all head_dim=128, same kernels)")
    print(f"{'stack':14s} {'mask slope':>13s} {'attn slope':>13s} {'r = mask/attn':>14s}")
    ratios = {}
    for name, path in COST_SOURCES.items():
        d = json.load(open(path))
        ms = float(d["mask_fit"]["slope_ms_per_unit"])
        at = float(d["attention_fit"]["slope_ms_per_unit"])
        ratios[name] = ms / at
        print(f"{name:14s} {ms:13.4e} {at:13.4e} {ms / at:14.0f}")
    names = list(ratios)
    print("\npairwise relative distance in r (this is what placement actually sees):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = ratios[names[i]], ratios[names[j]]
            print(f"  {names[i]:12s} vs {names[j]:12s}: {abs(a - b) / min(a, b) * 100:5.1f}%")


def report_sensitivity():
    cost = json.load(open(COST_JSON))
    icpt = rank_intercept(cost)
    log = load_density_log(DENSITY_LOG)
    entries = [(l, ts, d) for l, rows in log.items() for ts, d in rows]

    def makespan(plan, true_cost):
        return max(icpt + sum(true_cost[h] for h in rank) for rank in plan)

    print(f"\n### Deliberate coefficient error, Wan-1.3B W={WORLD}, "
          f"{len(entries)} (layer,step) points")
    print(f"{'r error':>9s} {'mean regret':>12s} {'p99 regret':>11s} {'max regret':>11s}")
    for f in FACTORS:
        perturbed = copy.deepcopy(cost)
        perturbed["mask_fit"]["slope_ms_per_unit"] *= f
        regrets = []
        for _l, _ts, dens in entries:
            if len(dens) % WORLD:
                continue
            true_cost = head_costs(dens, cost, SEQ_LEN)[2]
            native, _ = greedy_lpt_assignment(true_cost, WORLD, 1)
            wrong_cost = head_costs(dens, perturbed, SEQ_LEN)[2]
            wrong, _ = greedy_lpt_assignment(wrong_cost, WORLD, 1)
            regrets.append(makespan(wrong, true_cost) / makespan(native, true_cost))
        regrets.sort()
        print(f"{f:>9.3f} {st.mean(regrets):>12.4f}x "
              f"{regrets[int(0.99 * len(regrets))]:>10.4f}x {regrets[-1]:>10.4f}x")


if __name__ == "__main__":
    report_shape_ratios()
    report_sensitivity()

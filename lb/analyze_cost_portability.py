"""Cost-model portability (paper appendix, cost-model calibration and mismatch): how much does placement degrade
when a workload is planned with another model's fitted coefficients, e.g. the
Wan2.1-1.3B fit on HunyuanVideo?

Method (pure CPU, no GPU): hold the workload fixed (its own density trace +
seq_len), and vary ONLY which cost-model json supplies the (mask_slope,
attn_slope). Build the greedy-LPT placement with each candidate cost model,
then score EVERY placement under the workload's OWN (native) cost model -- the
best available estimate of the true per-head time. Placement "regret" =
makespan(plan built with mismatched coeffs) / makespan(plan built with native
coeffs). Regret ~ 1.00 => placement is insensitive to which model's
coefficients are used => the cost model is portable.

We also report the whole-trace makespan speedup vs the contiguous baseline for
each cost source, so a reader sees the LB gain barely moves under a wrong fit.
"""
import sys, json, statistics
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from split_planner import head_costs, greedy_lpt_assignment, load_density_log, rank_intercept

ROOT = Path(__file__).resolve().parents[1]

COST = {
    "wan1.3b": ROOT / "result/wan/t2v/sap_1.3b/maskgen_aware_cost.json",
    "wan14b":  ROOT / "result/wan/t2v/sap_14b/maskgen_aware_cost.json",
    "hyvideo": ROOT / "result/hyvideo/t2v/sap/maskgen_aware_cost.json",
}
# (label, density-log, native cost key, seq_len, world_size, min_heads)
WORKLOADS = [
    ("Wan14B-720p (W=8)", ROOT / "data/traces/wan2.1-14b_720p_125f.jsonl", "wan14b", 115200, 8, 1),
    ("Wan14B-720p (W=4)", ROOT / "data/traces/wan2.1-14b_720p_125f.jsonl", "wan14b", 115200, 4, 1),
    ("Wan1.3B-253f (W=4)", ROOT / "data/traces/wan2.1-1.3b_480p_253f.jsonl", "wan1.3b", 99840, 4, 1),
    # HunyuanVideo as the workload, planned with the Wan fits. 24 heads, seq 108256.
    ("HunyuanVideo-120f (W=4)", ROOT / "data/traces/hunyuanvideo_720p_120f.jsonl", "hyvideo", 108256, 4, 1),
    ("HunyuanVideo-120f (W=8)", ROOT / "data/traces/hunyuanvideo_720p_120f.jsonl", "hyvideo", 108256, 8, 1),
]

def contig(nheads, W):
    hpr = nheads // W
    return [list(range(r*hpr, (r+1)*hpr)) for r in range(W)]

def makespan(plan, true_cost, icpt=0.0):
    # per-rank kernel-launch intercept added once per active rank
    return max(icpt + sum(true_cost[h] for h in rank) for rank in plan)

def run():
    costs = {k: json.load(open(v)) for k, v in COST.items()}
    for label, dlog, native, seq_len, W, mh in WORKLOADS:
        log = load_density_log(dlog)
        entries = [(l, ts, d) for l, rows in log.items() for ts, d in rows]
        nheads = len(entries[0][2])
        cands = [native] + [k for k in COST if k != native]
        # accumulate whole-trace makespan for each plan source + contiguous
        acc = {c: 0.0 for c in cands}
        acc_contig = 0.0
        regrets = {c: [] for c in cands}
        for (l, ts, dens) in entries:
            if nheads % W != 0:
                continue
            true_cost = head_costs(dens, costs[native], seq_len)[2]
            icpt = rank_intercept(costs[native])  # makespan scored under native cost
            m_contig = makespan(contig(nheads, W), true_cost, icpt)
            acc_contig += m_contig
            m_native = None
            for c in cands:
                pcost = head_costs(dens, costs[c], seq_len)[2]
                plan, _ = greedy_lpt_assignment(pcost, W, mh)
                m = makespan(plan, true_cost, icpt)
                acc[c] += m
                if c == native:
                    m_native = m
                regrets[c].append(m / m_native if m_native else 1.0)
        print(f"\n### {label}   heads={nheads} seq_len={seq_len}  native_cost={native}")
        print(f"{'cost source':14s} {'whole-trace makespan':>20s} {'speedup vs contig':>18s} {'mean regret vs native':>22s} {'p99 regret':>12s}")
        print(f"{'contiguous':14s} {acc_contig:18.1f}   {'1.000x':>16s}   {'--':>20s}   {'--':>10s}")
        for c in cands:
            sp = acc_contig / acc[c]
            mr = statistics.mean(regrets[c])
            p99 = sorted(regrets[c])[int(0.99*len(regrets[c]))-1]
            tag = " (native)" if c == native else " (MISMATCHED)"
            print(f"{c+tag:14s} {acc[c]:18.1f}   {sp:15.3f}x   {mr:20.4f}x   {p99:10.4f}x")

if __name__ == "__main__":
    run()

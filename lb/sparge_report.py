"""Print one line of summary for a SpargeAttn per-threshold replan result."""
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
s = d["summary"]
g = d["density"]
m = st.mean(g)
print(
    f"  placement={d['heads_per_rank']}  density={m:.3f}  skew={max(g) / m:.2f}  "
    f"baseline={s['baseline_mean_ms']:.2f}  asymhp={s['asymhp_mean_ms']:.2f}  "
    f"speedup={s['speedup_ratio_of_means']:.3f}x"
)

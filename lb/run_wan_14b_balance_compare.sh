#!/usr/bin/env bash
# Wan 2.1 T2V 14B balance comparison: baseline / symm / asym+cost.
#
# Usage:
#   bash lb/run_wan_14b_balance_compare.sh                      # default 81f W=8 720p
#   WORLD_SIZE=4 NUM_FRAMES=81 bash lb/run_wan_14b_balance_compare.sh
#   SIM_WORLD=8 bash lb/run_wan_14b_balance_compare.sh          # 2-GPU sim of 8

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NUM_FRAMES="${NUM_FRAMES:-81}"
LAYER="${LAYER:-21}"
STEP="${STEP:-20}"
RESOLUTION="${RESOLUTION:-720p}"
WORLD_SIZE="${WORLD_SIZE:-8}"
SIM_WORLD="${SIM_WORLD:-0}"
PROMPT_ID="${PROMPT_ID:-1}"

NUM_INFERENCE_STEPS=50
QC=300
KC=1000
TOP_P=0.9
MIN_KC_RATIO=0.10
KM_INIT=50
KM_STEP=2
FIRST_TIMES_FP=0.2
FIRST_LAYERS_FP=0.03

OUT_DIR="result/wan/t2v/sap_14b"
LOG_DIR="${OUT_DIR}/Step_${NUM_INFERENCE_STEPS}-Res_${RESOLUTION}/TFP_${FIRST_TIMES_FP}-LFP_${FIRST_LAYERS_FP}/QC_${QC}-KC_${KC}-TopP_${TOP_P}/Init_${KM_INIT}-Step_${KM_STEP}-MinR_${MIN_KC_RATIO}_${NUM_FRAMES}frames"
DUMP_DIR="${LOG_DIR}/attn_dumps"
INPUT="${INPUT:-${DUMP_DIR}/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt}"
DENSITY="${DENSITY:-${LOG_DIR}/${PROMPT_ID}-0.jsonl}"
COST_JSON="${COST_JSON:-${OUT_DIR}/maskgen_aware_cost.json}"

SCRIPT="$REPO_ROOT/lb/bench_sp_all2all_attention.py"

if [[ ! -f "$INPUT" ]]; then
  echo "[ERR] dump not found: $INPUT"
  echo "      Run: NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION bash lb/dump_wan_14b_attn.sh"
  exit 2
fi

if [[ ! -f "$DENSITY" ]]; then
  echo "[ERR] density log not found: $DENSITY"
  exit 2
fi

if [[ ! -f "$COST_JSON" ]]; then
  echo "[ERR] cost model JSON not found: $COST_JSON"
  echo "      Generate once with:"
  echo "        MODEL=14b NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/profile_cost.sh"
  exit 2
fi

NUM_HEADS_IN_DUMP="$(
python - "$INPUT" <<'PY'
import sys
import torch

p = sys.argv[1]
d = torch.load(p, map_location="cpu", weights_only=False)
print(int(d["inputs"]["query"].shape[1]))
PY
)"

if [[ "$SIM_WORLD" -gt 0 ]]; then
  REAL_W=2
  EFFECTIVE_W="$SIM_WORLD"
  SIM_FLAGS=(--asymm-a2a pull_qkv --sim-world "$SIM_WORLD" --sim-passive-rank 1)
  TAG="wan_14b_sim${SIM_WORLD}_${NUM_FRAMES}f_${RESOLUTION}_p${PROMPT_ID}_step${STEP}_layer${LAYER}"
  CONFIGS_TO_RUN="baseline,asym_cost"
  echo "[run] SIM mode: 2 real GPUs simulating $SIM_WORLD effective ranks"
else
  REAL_W="$WORLD_SIZE"
  EFFECTIVE_W="$WORLD_SIZE"
  SIM_FLAGS=()
  TAG="wan_14b_w${WORLD_SIZE}_${NUM_FRAMES}f_${RESOLUTION}_p${PROMPT_ID}_step${STEP}_layer${LAYER}"
  CONFIGS_TO_RUN="baseline,symm,asym_cost"
fi

# Contiguous and equal-count greedy require num_heads % effective_world == 0.
# Unequal greedy + asymm-a2a can handle non-divisible head counts, e.g. 40 heads / 3 GPUs.
if (( NUM_HEADS_IN_DUMP % EFFECTIVE_W != 0 )); then
  echo "[run] heads=$NUM_HEADS_IN_DUMP not divisible by effective_W=$EFFECTIVE_W"
  echo "[run] skip baseline/symm; only asym_cost is valid with greedy_unequal + asymm-a2a"
  CONFIGS_TO_RUN="asym_cost"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((REAL_W - 1)))}"

COMMON=(
  --input "$INPUT"
  --density-log "$DENSITY"
  --iters 50
  --warmup 20
)

echo "[run] model=wan-14b real_W=$REAL_W effective_W=$EFFECTIVE_W heads=$NUM_HEADS_IN_DUMP frames=$NUM_FRAMES res=$RESOLUTION"
echo "[run] input  -> $INPUT"
echo "[run] dens   -> $DENSITY"
echo "[run] cost   -> $COST_JSON  (used by asym+cost only)"
echo "[run] configs -> $CONFIGS_TO_RUN"

run_cfg() {
  local name="$1"
  local port="$2"
  shift 2

  if [[ ",$CONFIGS_TO_RUN," != *",$name,"* ]]; then
    echo
    echo "=== skip $name (not applicable in current mode) ==="
    return
  fi

  echo
  echo "=== $name ==="
  torchrun --nproc-per-node="$REAL_W" --master-port="$port" "$SCRIPT" \
    "${COMMON[@]}" "${SIM_FLAGS[@]}" "$@" \
    --rank-csv "/tmp/${TAG}_${name}.csv" 2>&1
}

run_cfg baseline  29810 --balance contiguous
run_cfg symm      29811 --balance greedy

if [[ "$SIM_WORLD" -gt 0 ]]; then
  run_cfg asym_cost 29812 \
    --balance greedy_unequal \
    --cost-model-json "$COST_JSON"
else
  run_cfg asym_cost 29812 \
    --balance greedy_unequal \
    --cost-model-json "$COST_JSON" \
    --asymm-a2a pull_qkv
fi

echo
echo "=== summary (makespan = max per-rank total_ms_mean) ==="
python - "$TAG" "$CONFIGS_TO_RUN" <<'PY'
import csv
import os
import sys

tag = sys.argv[1]
configs = [c for c in sys.argv[2].split(",") if c]
labels = {
    "baseline": "baseline",
    "symm": "symm",
    "asym_cost": "asym+cost",
}

baseline = None
baseline_label = None

for cfg in configs:
    path = f"/tmp/{tag}_{cfg}.csv"
    label = labels.get(cfg, cfg)

    if not os.path.exists(path):
        print(f"{label:14s} <missing csv>")
        continue

    rows = list(csv.DictReader(open(path)))
    if not rows:
        print(f"{label:14s} <empty csv>")
        continue

    totals = [float(r["total_ms_mean"]) for r in rows]
    makespan = max(totals)

    if baseline is None:
        baseline = makespan
        baseline_label = label

    speedup = baseline / makespan if makespan > 0 else float("nan")
    per_rank = [round(t, 2) for t in totals]

    suffix = (
        "speedup_vs_baseline"
        if baseline_label == "baseline"
        else f"speedup_vs_{baseline_label}"
    )
    print(
        f"{label:14s} per-rank={per_rank}  "
        f"makespan={makespan:6.2f} ms  {suffix}={speedup:.3f}x"
    )
PY
#!/usr/bin/env bash
# HunyuanVideo T2V balance comparison: baseline / symm / asym+cost.
#
# Hunyuan uses a split-replicated layout: video tokens go through SP all2all,
# prompt/context tokens are present on every rank and are head-sliced locally.
# The asym+cost row uses asymm-a2a for video Q/K/V and head-gathers prompt
# output after attention.
#
#   baseline   — contiguous heads, symm a2a, no LB
#   symm       — greedy (density-only), symm a2a (Option C), no cost model
#   asym+cost  — greedy_unequal, asymm-a2a pull_qkv, cost model
#
# Usage:
#   bash lb/run_hunyuan_t2v_balance_compare.sh                  # default 129f W=8 480p
#   WORLD_SIZE=4 NUM_FRAMES=65 bash lb/run_hunyuan_t2v_balance_compare.sh
#
# Knobs (env vars):
#   NUM_FRAMES   — must match dump. Default 129 (the paper uses 120 at 720p).
#   LAYER, STEP  — must match dump. Default 21, 20.
#   RESOLUTION   — 480p / 720p. Default 480p.
#   WORLD_SIZE   — real GPU count, ignored when SIM_WORLD>0. Default 8.
#   SIM_WORLD    — if >0: 2-real-GPU sweep playing N effective sim ranks via
#                  asymm-a2a pull. Skips the "symm" config. Default 0 = off.
#   PROMPT_ID    — Default 7.
#   COST_JSON    — maskgen-aware cost model JSON.
#   CUDA_VISIBLE_DEVICES

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NUM_FRAMES="${NUM_FRAMES:-129}"
LAYER="${LAYER:-21}"
STEP="${STEP:-20}"
RESOLUTION="${RESOLUTION:-480p}"
WORLD_SIZE="${WORLD_SIZE:-8}"
SIM_WORLD="${SIM_WORLD:-0}"
PROMPT_ID="${PROMPT_ID:-7}"

NUM_INFERENCE_STEPS=50
QC=200; KC=1000; TOP_P=0.9; MIN_KC_RATIO=0.10
KM_INIT=50; KM_STEP=2
FIRST_TIMES_FP=0.04; FIRST_LAYERS_FP=0.0

OUT_DIR="result/hyvideo/t2v/sap"
LOG_DIR="${OUT_DIR}/Step_${NUM_INFERENCE_STEPS}-Res_${RESOLUTION}/TFP_${FIRST_TIMES_FP}-LFP_${FIRST_LAYERS_FP}/QC_${QC}-KC_${KC}-TopP_${TOP_P}/Init_${KM_INIT}-Step_${KM_STEP}-MinR_${MIN_KC_RATIO}_${NUM_FRAMES}frames"
CSV_DIR="${CSV_DIR:-${LOG_DIR}/bench}"  # per-rank CSVs
mkdir -p "$CSV_DIR"
export CSV_DIR
DUMP_DIR="${LOG_DIR}/attn_dumps"
INPUT="${INPUT:-${DUMP_DIR}/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt}"
DENSITY="${DENSITY:-${LOG_DIR}/${PROMPT_ID}-0.jsonl}"
COST_JSON="${COST_JSON:-${OUT_DIR}/maskgen_aware_cost.json}"

if [[ ! -f "$INPUT" ]]; then
  echo "[ERR] dump not found: $INPUT"
  echo "      Run: NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION bash lb/dump_hunyuan_t2v_attn.sh"; exit 2
fi
if [[ ! -f "$DENSITY" ]]; then
  echo "[ERR] density log not found: $DENSITY"; exit 2
fi
if [[ ! -f "$COST_JSON" ]]; then
  echo "[ERR] cost model JSON not found: $COST_JSON"
  echo "      Generate once with:"
  echo "        MODEL=hunyuan NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/profile_cost.sh"
  exit 2
fi

SCRIPT="$REPO_ROOT/lb/bench_sp_all2all_attention.py"
if [[ "$SIM_WORLD" -gt 0 ]]; then
  REAL_W=2
  SIM_FLAGS=(--asymm-a2a pull_qkv --sim-world "$SIM_WORLD" --sim-passive-rank 1)
  TAG="hunyuan_t2v_sim${SIM_WORLD}_${NUM_FRAMES}f_${RESOLUTION}_p${PROMPT_ID}_step${STEP}_layer${LAYER}"
  EFFECTIVE_W="$SIM_WORLD"
  CONFIGS_TO_RUN="baseline,asym_cost"
  echo "[run] SIM mode: 2 real GPUs simulating $SIM_WORLD effective ranks"
else
  REAL_W="$WORLD_SIZE"
  SIM_FLAGS=()
  TAG="hunyuan_t2v_w${WORLD_SIZE}_${NUM_FRAMES}f_${RESOLUTION}_p${PROMPT_ID}_step${STEP}_layer${LAYER}"
  EFFECTIVE_W="$WORLD_SIZE"
  CONFIGS_TO_RUN="baseline,symm,asym_cost"
fi

# Contiguous and equal-count placement need num_heads % W == 0; AsymHP
# (greedy_unequal + asymmetric exchange) also runs non-divisible configs.
NUM_HEADS="${NUM_HEADS:-24}"
if (( NUM_HEADS % EFFECTIVE_W != 0 )); then
  echo "[run] heads=$NUM_HEADS not divisible by effective_W=$EFFECTIVE_W; running asym_cost only"
  CONFIGS_TO_RUN="asym_cost"
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((REAL_W - 1)))"
fi

COMMON=(
  --input "$INPUT"
  --density-log "$DENSITY"
  --iters 50
  --warmup 20
)

echo "[run] model=hunyuan-t2v real_W=$REAL_W effective_W=$EFFECTIVE_W frames=$NUM_FRAMES res=$RESOLUTION"
echo "[run] input  -> $INPUT"
echo "[run] dens   -> $DENSITY"
echo "[run] cost   -> $COST_JSON  (used by asym+cost only)"

run_cfg() {
  local name="$1"; shift
  if [[ ",$CONFIGS_TO_RUN," != *",$name,"* ]]; then
    echo
    echo "=== skip $name (not applicable in current mode) ==="
    return
  fi
  echo
  echo "=== $name ==="
  local csv="${CSV_DIR}/${TAG}_${name}.csv"
  rm -f "$csv"
  # The bench may abort in the CUDASymmetricMemory destructor after writing its
  # CSV, so only a missing CSV counts as a failure; later configs still run.
  torchrun --nproc-per-node="$REAL_W" --master-port="$2" "$SCRIPT" \
    "${COMMON[@]}" "${SIM_FLAGS[@]}" "${@:3}" \
    --rank-csv "$csv" 2>&1 || true
  [[ -s "$csv" ]] || echo "[ERR] $name produced no CSV"
}

run_cfg baseline      _ 29820 --balance contiguous
run_cfg symm          _ 29821 --balance greedy
if [[ "$SIM_WORLD" -gt 0 ]]; then
  run_cfg asym_cost   _ 29822 --balance greedy_unequal --cost-model-json "$COST_JSON"
else
  run_cfg asym_cost   _ 29822 --balance greedy_unequal --cost-model-json "$COST_JSON" --asymm-a2a pull_qkv
fi

echo
echo "=== summary (makespan = max per-rank total_ms_mean) ==="
python - "$TAG" "$CONFIGS_TO_RUN" <<'PY'
import csv, os, sys
tag = sys.argv[1]
configs = sys.argv[2].split(",")
labels = {"baseline": "baseline", "symm": "symm", "asym_cost": "asym+cost"}
baseline = None
for cfg in configs:
    label = labels.get(cfg, cfg)
    path = f"{os.environ['CSV_DIR']}/{tag}_{cfg}.csv"
    if not os.path.exists(path):
        print(f"{label:14s} <missing csv>"); continue
    rows = list(csv.DictReader(open(path)))
    totals = [float(r["total_ms_mean"]) for r in rows]
    mk = max(totals)
    if baseline is None: baseline = mk
    speedup = baseline / mk if mk > 0 else float("nan")
    per_rank = [round(t, 2) for t in totals]
    print(f"{label:14s} per-rank={per_rank}  makespan={mk:6.2f} ms  speedup_vs_baseline={speedup:.3f}x")
PY

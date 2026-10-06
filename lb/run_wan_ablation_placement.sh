#!/usr/bin/env bash
# Placement-policy ablation (paper Table 1).
# All rows reuse the same SVG2 sparse-attention kernels and the same asymmetric
# exchange (--asymm-a2a pull_qkv) -- only head placement changes.
#
#   1. Contiguous equal-head : --balance contiguous
#   2. Equal-head shuffle    : --balance greedy            (prev density, density-as-cost)
#   3. Density-only variant  : --balance greedy_unequal    (prev density, density-as-cost)
#   4. AsymHP                : --balance greedy_unequal    + --cost-model-json (prev density)
#   5. Oracle AsymHP         : --balance greedy_unequal    + --cost-model-json + --oracle-density (current density)
#
# Output: per-config CSVs under $LOG_DIR/bench/ and a final makespan + speedup table.
#
# Usage:
#   bash lb/run_wan_ablation_placement.sh                    # uses GPUs 0..5
#   CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash lb/run_wan_ablation_placement.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NUM_FRAMES="${NUM_FRAMES:-480}"
LAYER="${LAYER:-21}"
STEP="${STEP:-20}"
RESOLUTION="${RESOLUTION:-480p}"
PROMPT_ID="${PROMPT_ID:-1}"
NPROC="${NPROC:-6}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC - 1)))}"

NUM_INFERENCE_STEPS=50
QC=300; KC=1000; TOP_P=0.9; MIN_KC_RATIO=0.10
KM_INIT=50; KM_STEP=2
FIRST_TIMES_FP=0.2; FIRST_LAYERS_FP=0.03

OUT_DIR="$REPO_ROOT/result/wan/t2v/sap_1.3b"
LOG_DIR="${OUT_DIR}/Step_${NUM_INFERENCE_STEPS}-Res_${RESOLUTION}/TFP_${FIRST_TIMES_FP}-LFP_${FIRST_LAYERS_FP}/QC_${QC}-KC_${KC}-TopP_${TOP_P}/Init_${KM_INIT}-Step_${KM_STEP}-MinR_${MIN_KC_RATIO}_${NUM_FRAMES}frames"
CSV_DIR="${CSV_DIR:-${LOG_DIR}/bench}"  # per-rank CSVs
mkdir -p "$CSV_DIR"
export CSV_DIR
DUMP_DIR="${LOG_DIR}/attn_dumps"
INPUT="${INPUT:-${DUMP_DIR}/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt}"
DENSITY="${DENSITY:-${LOG_DIR}/${PROMPT_ID}-0.jsonl}"
COST="${COST:-${OUT_DIR}/maskgen_aware_cost.json}"

ITERS="${ITERS:-50}"
WARMUP="${WARMUP:-20}"
TAG="wan_abl_${NUM_FRAMES}f_${RESOLUTION}_p${PROMPT_ID}_step${STEP}_layer${LAYER}_w${NPROC}"

COMMON=(
  --input "$INPUT"
  --density-log "$DENSITY"
  --asymm-a2a pull_qkv
  --iters "$ITERS"
  --warmup "$WARMUP"
)

SCRIPT="$REPO_ROOT/lb/bench_sp_all2all_attention.py"

if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERR] benchmark script not found: $SCRIPT"
  exit 2
fi
if [[ ! -f "$INPUT" ]]; then
  echo "[ERR] attention dump not found: $INPUT"
  echo "      Generate it with:"
  echo "        NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/dump_wan_1.3b_attn.sh"
  exit 2
fi
if [[ ! -f "$DENSITY" ]]; then
  echo "[ERR] density log not found: $DENSITY"
  echo "      Generate it with:"
  echo "        NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/dump_wan_1.3b_attn.sh"
  exit 2
fi
if [[ ! -f "$COST" ]]; then
  echo "[ERR] cost model JSON not found: $COST"
  echo "      Generate it with:"
  echo "        MODEL=1.3b NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/profile_cost.sh"
  exit 2
fi

echo "[run] frames=$NUM_FRAMES layer=$LAYER step=$STEP res=$RESOLUTION prompt=$PROMPT_ID nproc=$NPROC"
echo "[run] input -> $INPUT"
echo "[run] dens  -> $DENSITY"
echo "[run] cost  -> $COST"
echo

run_row() {   # $1 csv name  $2 port  $3.. bench args
  local name="$1" port="$2"; shift 2
  local csv="${CSV_DIR}/${TAG}_${name}.csv"
  rm -f "$csv"
  # The bench may abort in the CUDASymmetricMemory destructor after writing its
  # CSV, so only a missing CSV counts as a failure; later configs still run.
  torchrun --nproc-per-node="$NPROC" --master-port="$port" "$SCRIPT" \
    "${COMMON[@]}" "$@" --rank-csv "$csv" 2>&1 || true
  [[ -s "$csv" ]] || echo "[ERR] $name produced no CSV"
}

echo "=== 1/5 contiguous equal-head (baseline) ==="
run_row 1_contiguous 29810 --balance contiguous

echo
echo "=== 2/5 greedy + density-as-cost (Equal-head shuffle) ==="
run_row 2_equal_shuffle 29811 --balance greedy

echo
echo "=== 3/5 greedy_unequal + density-as-cost (Density-only variant) ==="
run_row 3_density_only 29812 --balance greedy_unequal

echo
echo "=== 4/5 greedy_unequal + cost model (AsymHP, prev density) ==="
run_row 4_sys 29813 --balance greedy_unequal --cost-model-json "$COST"

echo
echo "=== 5/5 greedy_unequal + cost model + oracle (Oracle AsymHP, current density) ==="
run_row 5_oracle 29814 --balance greedy_unequal --cost-model-json "$COST" --oracle-density

echo
echo "=== summary (makespan = max per-rank total_ms_mean; speedup vs baseline = contiguous) ==="
python - "$TAG" <<'PY'
import csv, os
import sys
tag = sys.argv[1]
configs = [
    ("1. Contiguous equal-head",          f"{os.environ['CSV_DIR']}/{tag}_1_contiguous.csv"),
    ("2. Equal-head shuffle",              f"{os.environ['CSV_DIR']}/{tag}_2_equal_shuffle.csv"),
    ("3. Density-only variant",            f"{os.environ['CSV_DIR']}/{tag}_3_density_only.csv"),
    ("4. AsymHP",                           f"{os.environ['CSV_DIR']}/{tag}_4_sys.csv"),
    ("5. Oracle AsymHP",                    f"{os.environ['CSV_DIR']}/{tag}_5_oracle.csv"),
]
baseline = None
print(f"{'config':35s}  {'makespan_ms':>12s}  {'speedup_vs_baseline':>20s}  per-rank")
print("-" * 110)
for tag, path in configs:
    if not os.path.exists(path):
        print(f"{tag:35s} <missing csv at {path}>")
        continue
    rows = list(csv.DictReader(open(path)))
    totals = [float(r["total_ms_mean"]) for r in rows]
    mk = max(totals)
    if baseline is None:
        baseline = mk
    speedup = baseline / mk if mk > 0 else float("nan")
    per_rank = [round(t, 2) for t in totals]
    print(f"{tag:35s}  {mk:>12.2f}  {speedup:>19.3f}x  {per_rank}")
PY

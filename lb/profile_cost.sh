#!/usr/bin/env bash
# One-shot maskgen-aware cost profile per model/GPU. The profile reads one
# captured dump for shapes, but the default output is shared by that model.
#
# Usage:
#   MODEL=1.3b   bash lb/profile_cost.sh                      # default 480 frames 480p
#   MODEL=14b    NUM_FRAMES=81  bash lb/profile_cost.sh
#   MODEL=hunyuan NUM_FRAMES=129 RESOLUTION=480p bash lb/profile_cost.sh
#
# Knobs:
#   MODEL        — required: "1.3b" | "14b" | "hunyuan".
#   NUM_FRAMES, LAYER, STEP, RESOLUTION, PROMPT_ID
#                — must match the dump file produced by lb/dump_*.sh.
#   INPUT        — explicit dump path (overrides MODEL+frame derivation).
#   OUTPUT       — explicit cost JSON path (overrides model-shared default).
#   SEQ_LENS     — pass-through to profiler (e.g. "1024,4096,16384"). Default "auto".
#   HEAD_COUNTS  — pass-through. Default "auto".
#   ITERS        — pass-through (default 20).
#   WARMUP       — pass-through (default 3).
#   CUDA_VISIBLE_DEVICES — single GPU is fine.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:?MODEL=1.3b|14b|hunyuan required}"
NUM_FRAMES="${NUM_FRAMES:-}"
LAYER="${LAYER:-21}"
STEP="${STEP:-20}"
RESOLUTION="${RESOLUTION:-480p}"
PROMPT_ID_DEFAULT_HY=7
PROMPT_ID_DEFAULT_WAN=1

case "$MODEL" in
  1.3b)
    OUT_DIR="result/wan/t2v/sap_1.3b"
    : "${NUM_FRAMES:=480}"
    : "${PROMPT_ID:=$PROMPT_ID_DEFAULT_WAN}"
    QC=300; KC=1000; TOP_P=0.9; MIN_KC_RATIO=0.10
    KM_INIT=50; KM_STEP=2
    FIRST_TIMES_FP=0.2; FIRST_LAYERS_FP=0.03
    ;;
  14b)
    OUT_DIR="result/wan/t2v/sap_14b"
    : "${NUM_FRAMES:=81}"
    : "${PROMPT_ID:=$PROMPT_ID_DEFAULT_WAN}"
    QC=300; KC=1000; TOP_P=0.9; MIN_KC_RATIO=0.10
    KM_INIT=50; KM_STEP=2
    FIRST_TIMES_FP=0.2; FIRST_LAYERS_FP=0.03
    ;;
  hunyuan)
    OUT_DIR="result/hyvideo/t2v/sap"
    : "${NUM_FRAMES:=129}"
    : "${PROMPT_ID:=$PROMPT_ID_DEFAULT_HY}"
    QC=200; KC=1000; TOP_P=0.9; MIN_KC_RATIO=0.10
    KM_INIT=50; KM_STEP=2
    FIRST_TIMES_FP=0.04; FIRST_LAYERS_FP=0.0
    ;;
  *)
    echo "[ERR] unknown MODEL=$MODEL (use 1.3b | 14b | hunyuan)"; exit 2 ;;
esac

NUM_INFERENCE_STEPS=50
LOG_DIR="${OUT_DIR}/Step_${NUM_INFERENCE_STEPS}-Res_${RESOLUTION}/TFP_${FIRST_TIMES_FP}-LFP_${FIRST_LAYERS_FP}/QC_${QC}-KC_${KC}-TopP_${TOP_P}/Init_${KM_INIT}-Step_${KM_STEP}-MinR_${MIN_KC_RATIO}_${NUM_FRAMES}frames"
DUMP_DIR="${LOG_DIR}/attn_dumps"
INPUT="${INPUT:-${DUMP_DIR}/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt}"
OUTPUT="${OUTPUT:-${OUT_DIR}/maskgen_aware_cost.json}"

if [[ ! -f "$INPUT" ]]; then
  echo "[ERR] dump not found: $INPUT"
  echo "      Run the matching dump script first, e.g.:"
  case "$MODEL" in
    1.3b) echo "        NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/dump_wan_1.3b_attn.sh" ;;
    14b) echo "        NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/dump_wan_14b_attn.sh" ;;
    hunyuan) echo "        NUM_FRAMES=$NUM_FRAMES LAYER=$LAYER STEP=$STEP RESOLUTION=$RESOLUTION PROMPT_ID=$PROMPT_ID bash lb/dump_hunyuan_t2v_attn.sh" ;;
  esac
  exit 2
fi

PROFILE_ARGS=(
  --input "$INPUT"
  --model-json "$OUTPUT"
  --seq-lens "${SEQ_LENS:-auto}"
  --head-counts "${HEAD_COUNTS:-auto}"
  --iters "${ITERS:-20}"
  --warmup "${WARMUP:-3}"
)

mkdir -p "$(dirname "$OUTPUT")"

echo "[profile-cost] MODEL=$MODEL frames=$NUM_FRAMES layer=$LAYER step=$STEP res=$RESOLUTION"
echo "[profile-cost] input  -> $INPUT"
echo "[profile-cost] output -> $OUTPUT"
echo

python lb/profile_maskgen_aware_cost.py "${PROFILE_ARGS[@]}"

echo
echo "[profile-cost] done. Compare scripts will pick this up automatically:"
case "$MODEL" in
  1.3b|14b) echo "  bash lb/run_wan_${MODEL}_balance_compare.sh" ;;
  hunyuan) echo "  bash lb/run_hunyuan_t2v_balance_compare.sh" ;;
esac

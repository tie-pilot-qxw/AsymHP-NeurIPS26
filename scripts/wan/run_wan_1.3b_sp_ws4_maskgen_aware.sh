#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

NUM_FRAMES="${NUM_FRAMES:-480}"
LAYER="${LAYER:-5}"
STEP="${STEP:-30}"
RESOLUTION="${RESOLUTION:-480p}"
PROMPT_ID="${PROMPT_ID:-1}"
WORLD_SIZE="${WORLD_SIZE:-4}"

NUM_INFERENCE_STEPS=50
QC=300; KC=1000; TOP_P=0.9; MIN_KC_RATIO=0.10
KM_INIT=50; KM_STEP=2
FIRST_TIMES_FP=0.2; FIRST_LAYERS_FP=0.03

OUT_DIR="result/wan/t2v/sap_1.3b"
LOG_DIR="${OUT_DIR}/Step_${NUM_INFERENCE_STEPS}-Res_${RESOLUTION}/TFP_${FIRST_TIMES_FP}-LFP_${FIRST_LAYERS_FP}/QC_${QC}-KC_${KC}-TopP_${TOP_P}/Init_${KM_INIT}-Step_${KM_STEP}-MinR_${MIN_KC_RATIO}_${NUM_FRAMES}frames"
DUMP_DIR="${LOG_DIR}/attn_dumps"
INPUT="${INPUT:-${DUMP_DIR}/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt}"
DENSITY="${DENSITY:-${LOG_DIR}/${PROMPT_ID}-0.jsonl}"
COST_JSON="${COST_JSON:-${OUT_DIR}/maskgen_aware_cost.json}"
CSV_DIR="${LOG_DIR}/bench"
mkdir -p "$CSV_DIR"

FLASHINFER_WORKSPACE_BASE=/tmp \
TRITON_CACHE_DIR=/tmp/triton-cache \
torchrun --nproc-per-node="$WORLD_SIZE" lb/bench_sp_all2all_attention.py \
  --input "$INPUT" \
  --warmup 1 \
  --iters 3 \
  --balance greedy \
  --density-log "$DENSITY" \
  --cost-model-json "$COST_JSON" \
  --rank-csv "${CSV_DIR}/sp_rank_times_p${PROMPT_ID}_step${STEP}_layer${LAYER}_ws${WORLD_SIZE}_maskgen_aware.csv" \
  --density-csv "${CSV_DIR}/sp_head_density_p${PROMPT_ID}_step${STEP}_layer${LAYER}_ws${WORLD_SIZE}_maskgen_aware.csv" \
  --q-chunk-density-csv "${CSV_DIR}/sp_q_chunk_density_p${PROMPT_ID}_step${STEP}_layer${LAYER}_ws${WORLD_SIZE}_maskgen_aware.csv" \
  --q-density-chunks 8

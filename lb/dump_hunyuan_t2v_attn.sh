#!/usr/bin/env bash
# Generate the (QKV dump, density JSONL) pair for HunyuanVideo T2V (SAP),
# consumed by lb/bench_sp_all2all_attention.py + lb/profile_maskgen_aware_cost.py.
#
# Outputs:
#   result/hyvideo/t2v/sap/Step_${STEPS}-Res_${RESOLUTION}/.../${NUM_FRAMES}frames/${PROMPT_ID}-0.jsonl
#   result/hyvideo/t2v/sap/Step_${STEPS}-Res_${RESOLUTION}/.../${NUM_FRAMES}frames/attn_dumps/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt
#
# Usage:
#     bash lb/dump_hunyuan_t2v_attn.sh                     # default 129 frames, 480p, layer=21, step=20
#     NUM_FRAMES=65 bash lb/dump_hunyuan_t2v_attn.sh
#     RESOLUTION=720p NUM_FRAMES=129 bash lb/dump_hunyuan_t2v_attn.sh
#
# Knobs (env vars, all optional):
#     NUM_FRAMES   — frames in generated video. Default 129. Hunyuan expects 4n+1
#                    (e.g. 65, 97, 129); pick accordingly.
#     LAYER        — attention layer to dump. Default 21.
#     STEP         — denoising step (0-indexed) to dump. Default 20.
#                    With first_times_fp=0.04 + --zero_step_kmeans_init, SAP
#                    is active from step ~2 onwards, so most STEP values work.
#     RESOLUTION   — "480p" (480x720) or "720p" (720x1280). Default 480p.
#     PROMPT_ID    — examples/<id>/prompt.txt to feed. Default 7 (from the
#                    existing scripts/hyvideo/hyvideo_t2v_480p_sap.sh).
#     CUDA_VISIBLE_DEVICES — pick GPUs (HunyuanVideo needs a beefy GPU; 1x H100).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NUM_FRAMES="${NUM_FRAMES:-129}"
LAYER="${LAYER:-30}"
STEP="${STEP:-20}"
RESOLUTION="${RESOLUTION:-480p}"
PROMPT_ID="${PROMPT_ID:-7}"

# SAP hyperparams — match scripts/hyvideo/hyvideo_t2v_480p_sap.sh
# (Hunyuan uses qc=200, lower TFP/LFP than wan, and zero-step kmeans init).
NUM_INFERENCE_STEPS=50
QC=200
KC=1000
TOP_P=0.9
MIN_KC_RATIO=0.10
KM_INIT=50
KM_STEP=2
FIRST_TIMES_FP=0.04
FIRST_LAYERS_FP=0.0

case "$RESOLUTION" in
  480p) HEIGHT=480; WIDTH=720 ;;
  720p) HEIGHT=720; WIDTH=1280 ;;
  *) echo "Unknown RESOLUTION=$RESOLUTION (use 480p or 720p)"; exit 2 ;;
esac

OUT_DIR="result/hyvideo/t2v/sap"
LOG_DIR="${OUT_DIR}/Step_${NUM_INFERENCE_STEPS}-Res_${RESOLUTION}/TFP_${FIRST_TIMES_FP}-LFP_${FIRST_LAYERS_FP}/QC_${QC}-KC_${KC}-TopP_${TOP_P}/Init_${KM_INIT}-Step_${KM_STEP}-MinR_${MIN_KC_RATIO}_${NUM_FRAMES}frames"
DUMP_DIR="${LOG_DIR}/attn_dumps"
DUMP_PATH="${DUMP_DIR}/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt"
LOG_PATH="${LOG_DIR}/${PROMPT_ID}-0.jsonl"
VIDEO_PATH="${LOG_DIR}/${PROMPT_ID}-0.mp4"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$DUMP_DIR"

PROMPT="$(cat examples/${PROMPT_ID}/prompt.txt)"

echo "[dump-hunyuan] frames=$NUM_FRAMES layer=$LAYER step=$STEP res=$RESOLUTION prompt=$PROMPT_ID"
echo "[dump-hunyuan] dump  -> $DUMP_PATH"
echo "[dump-hunyuan] log   -> $LOG_PATH"

SVG_WAN_ATTN_EXPORT_PATH="$DUMP_PATH" \
SVG_WAN_ATTN_EXPORT_MAX=1 \
SVG_WAN_ATTN_EXPORT_LAYER="$LAYER" \
SVG_WAN_ATTN_EXPORT_STEP="$STEP" \
SVG_WAN_ATTN_EXPORT_REQUIRE_CACHE=1 \
python hyvideo_t2v_inference.py \
  --model_id "tencent/HunyuanVideo" \
  --seed 0 \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --resolution "$RESOLUTION" \
  --prompt "$PROMPT" \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --pattern SAP \
  --num_q_centroids "$QC" \
  --num_k_centroids "$KC" \
  --top_p_kmeans "$TOP_P" \
  --min_kc_ratio "$MIN_KC_RATIO" \
  --kmeans_iter_init "$KM_INIT" \
  --kmeans_iter_step "$KM_STEP" \
  --zero_step_kmeans_init \
  --first_times_fp "$FIRST_TIMES_FP" \
  --first_layers_fp "$FIRST_LAYERS_FP" \
  --output_file "$VIDEO_PATH" \
  --logging_file "$LOG_PATH"

echo
echo "[dump-hunyuan] done. Bench inputs:"
echo "  --input        $DUMP_PATH"
echo "  --density-log  $LOG_PATH"

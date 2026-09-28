#!/usr/bin/env bash
# Generate the (QKV dump, density JSONL) pair for Wan 2.1 T2V 1.3B,
# consumed by lb/bench_sp_all2all_attention.py + lb/profile_maskgen_aware_cost.py.
#
# Outputs:
#   result/wan/t2v/sap_1.3b/Step_${STEPS}-Res_${RESOLUTION}/.../${NUM_FRAMES}frames/${PROMPT_ID}-0.jsonl
#   result/wan/t2v/sap_1.3b/Step_${STEPS}-Res_${RESOLUTION}/.../${NUM_FRAMES}frames/attn_dumps/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt
#
# Usage:
#     bash lb/dump_wan_1.3b_attn.sh                       # default 480 frames, 480p, layer=21, step=20
#     NUM_FRAMES=240 bash lb/dump_wan_1.3b_attn.sh
#     NUM_FRAMES=480 LAYER=21 STEP=20 bash lb/dump_wan_1.3b_attn.sh
#     RESOLUTION=720p NUM_FRAMES=81 bash lb/dump_wan_1.3b_attn.sh
#
#   Paper snapshots (layer 21, step 20):
#     RESOLUTION=720p NUM_FRAMES=120 bash lb/dump_wan_1.3b_attn.sh      # operator agreement, PCIe
#     RESOLUTION=720p NUM_FRAMES=125 FIRST_TIMES_FP=0 FIRST_LAYERS_FP=0 \
#       bash lb/dump_wan_1.3b_attn.sh                                  # SpargeAttn, top-p sweep
#
# Knobs (env vars, all optional):
#     NUM_FRAMES   — frames in generated video. Default 480.
#     LAYER        — attention layer to dump. Default 21.
#     STEP         — denoising step (0-indexed) to dump. Default 20.
#                    Must be >= ceil(first_times_fp * num_inference_steps) = 10
#                    so that SAP (not full-attention warmup) runs at that step.
#     RESOLUTION   — "480p" (480x832) or "720p" (720x1280). Default 480p.
#     PROMPT_ID    — examples/<id>/prompt.txt to feed. Default 1.
#     FIRST_TIMES_FP / FIRST_LAYERS_FP — dense-attention prefix (fraction of
#                    steps / layers). Default 0.2 / 0.03 (SVG2 default).
#     CUDA_VISIBLE_DEVICES — pick GPUs (Wan 1.3B is small; 1x A100 is plenty).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NUM_FRAMES="${NUM_FRAMES:-480}"
LAYER="${LAYER:-21}"
STEP="${STEP:-20}"
RESOLUTION="${RESOLUTION:-480p}"
PROMPT_ID="${PROMPT_ID:-1}"
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"

# SAP hyperparams.
NUM_INFERENCE_STEPS=50
QC=300
KC=1000
TOP_P=0.9
MIN_KC_RATIO=0.10
KM_INIT=50
KM_STEP=2
FIRST_TIMES_FP="${FIRST_TIMES_FP:-0.2}"
FIRST_LAYERS_FP="${FIRST_LAYERS_FP:-0.03}"

case "$RESOLUTION" in
  480p) HEIGHT=480; WIDTH=832 ;;
  720p) HEIGHT=720; WIDTH=1280 ;;
  *) echo "Unknown RESOLUTION=$RESOLUTION (use 480p or 720p)"; exit 2 ;;
esac

OUT_DIR="result/wan/t2v/sap_1.3b"
LOG_DIR="${OUT_DIR}/Step_${NUM_INFERENCE_STEPS}-Res_${RESOLUTION}/TFP_${FIRST_TIMES_FP}-LFP_${FIRST_LAYERS_FP}/QC_${QC}-KC_${KC}-TopP_${TOP_P}/Init_${KM_INIT}-Step_${KM_STEP}-MinR_${MIN_KC_RATIO}_${NUM_FRAMES}frames"
DUMP_DIR="${LOG_DIR}/attn_dumps"
DUMP_PATH="${DUMP_DIR}/${PROMPT_ID}-0_step${STEP}_layer${LAYER}.pt"
LOG_PATH="${LOG_DIR}/${PROMPT_ID}-0.jsonl"
VIDEO_PATH="${LOG_DIR}/${PROMPT_ID}-0.mp4"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$DUMP_DIR"

PROMPT="$(cat examples/${PROMPT_ID}/prompt.txt)"

echo "[dump-wan-1.3b] frames=$NUM_FRAMES layer=$LAYER step=$STEP res=$RESOLUTION prompt=$PROMPT_ID"
echo "[dump-wan-1.3b] dump  -> $DUMP_PATH"
echo "[dump-wan-1.3b] log   -> $LOG_PATH"

SVG_WAN_ATTN_EXPORT_PATH="$DUMP_PATH" \
SVG_WAN_ATTN_EXPORT_MAX=1 \
SVG_WAN_ATTN_EXPORT_LAYER="$LAYER" \
SVG_WAN_ATTN_EXPORT_STEP="$STEP" \
SVG_WAN_ATTN_EXPORT_REQUIRE_CACHE=1 \
python wan_t2v_inference.py \
  --model_id "$MODEL_ID" \
  --prompt "$PROMPT" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num_frames "$NUM_FRAMES" \
  --seed 0 \
  --num_inference_steps "$NUM_INFERENCE_STEPS" \
  --pattern SAP \
  --num_q_centroids "$QC" \
  --num_k_centroids "$KC" \
  --top_p_kmeans "$TOP_P" \
  --min_kc_ratio "$MIN_KC_RATIO" \
  --kmeans_iter_init "$KM_INIT" \
  --kmeans_iter_step "$KM_STEP" \
  --first_times_fp "$FIRST_TIMES_FP" \
  --first_layers_fp "$FIRST_LAYERS_FP" \
  --output_file "$VIDEO_PATH" \
  --logging_file "$LOG_PATH"

echo
echo "[dump-wan-1.3b] done. Bench inputs:"
echo "  --input        $DUMP_PATH"
echo "  --density-log  $LOG_PATH"

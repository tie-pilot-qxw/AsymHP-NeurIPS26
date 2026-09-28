#!/usr/bin/env bash
# Per-setting density traces for the SVG2 sparsity sweep (paper Appendix D.1).
#
# AsymHP plans each top-p setting from that setting's own previous-step
# density, so each top-p needs its own trace. Single GPU per setting, 720p,
# 125 frames, 50 steps, sparse from the first step/layer. The top-p=0.9 trace
# and the replayed snapshot come from:
#   RESOLUTION=720p NUM_FRAMES=125 FIRST_TIMES_FP=0 FIRST_LAYERS_FP=0 \
#     bash lb/dump_wan_1.3b_attn.sh
#
#   GPUS=0,1,2 bash lb/run_topp_dumps.sh        # top-p 0.7, 0.8, 0.95
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:-0,1,2}"
TOPPS="${TOPPS:-0.7 0.8 0.95}"
PROMPT="$(cat examples/1/prompt.txt)"
i=0
pids=()
for TP in $TOPPS; do
  G=$(echo "$GPUS" | cut -d, -f$((i + 1))); i=$((i + 1))
  DIR="result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0-LFP_0/QC_300-KC_1000-TopP_${TP}/Init_50-Step_2-MinR_0.10_125frames"
  mkdir -p "$DIR"
  echo "[dump] top_p=$TP on GPU $G -> $DIR/1-0.jsonl"
  CUDA_VISIBLE_DEVICES="$G" python wan_t2v_inference.py \
    --model_id "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" --prompt "$PROMPT" \
    --height 720 --width 1280 --num_frames 125 --seed 0 --num_inference_steps 50 \
    --pattern SAP --num_q_centroids 300 --num_k_centroids 1000 \
    --top_p_kmeans "$TP" --min_kc_ratio 0.10 \
    --kmeans_iter_init 50 --kmeans_iter_step 2 \
    --first_times_fp 0 --first_layers_fp 0 \
    --output_file "$DIR/1-0.mp4" --logging_file "$DIR/1-0.jsonl" \
    >"$DIR/dump.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
for TP in $TOPPS; do
  DIR="result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0-LFP_0/QC_300-KC_1000-TopP_${TP}/Init_50-Step_2-MinR_0.10_125frames"
  echo "  top_p=$TP -> $(wc -l <"$DIR/1-0.jsonl" 2>/dev/null || echo 0) density rows"
done
echo TOPP_DUMPS_DONE

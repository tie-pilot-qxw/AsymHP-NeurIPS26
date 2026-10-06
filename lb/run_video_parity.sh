#!/usr/bin/env bash
# Decoded-video agreement with a baseline repeatability control (paper
# appendix, decoded-video table).
#
# For each prompt, generate the equal-head baseline twice (the second run is the
# run-to-run control) and AsymHP once, then report PSNR against the first
# baseline. Wan2.1-1.3B, 720p, 125 frames, 50 steps, seed 0, sparse attention
# from the first step/layer, four GPUs. AsymHP keeps at least two heads per GPU.
#
# Each saved video array is about 1.4 GB (nine arrays for three prompts).
#
#   GPUS=0,1,2,3 bash lb/run_video_parity.sh
#   GPUS=0,1,2,3 PROMPTS="1" bash lb/run_video_parity.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:-0,1,2,3}"
PROMPTS="${PROMPTS:-1 2 3}"
OUT="${OUT:-result/paper/video_parity}"
COST="${COST:-result/wan/t2v/sap_1.3b/maskgen_aware_cost.json}"
SAP="--pattern SAP --num_q_centroids 300 --num_k_centroids 1000 --top_p_kmeans 0.9 --min_kc_ratio 0.10 --kmeans_iter_init 50 --kmeans_iter_step 2"
BASE="--model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers --height 720 --width 1280 --num_frames 125 --num_inference_steps 50 --warmup_steps 2 --seed 0 --first_times_fp 0 --first_layers_fp 0 $SAP"
mkdir -p "$OUT"

run() {   # $1 output prefix  $2 prompt  $3.. extra args
  local pref="$1" prompt="$2"; shift 2
  rm -f "${pref}.npy"
  CUDA_VISIBLE_DEVICES="$GPUS" torchrun --nproc-per-node=4 \
    --master-port=$((44000 + RANDOM % 500)) wan_t2v_sp_inference.py \
    $BASE --prompt "$prompt" "$@" \
    --save_latents "${pref}.npy" --output_file "${pref}.mp4" >"${pref}.log" 2>&1
  [[ -s "${pref}.npy" ]] || { echo "[ERR] no video array: ${pref}.npy (see ${pref}.log)" >&2; return 1; }
}

for pid in $PROMPTS; do
  P="$(cat examples/${pid}/prompt.txt)"
  echo "===== prompt $pid ====="
  run "$OUT/p${pid}_baseline" "$P" --balance contiguous --a2a symm
  run "$OUT/p${pid}_baseline2" "$P" --balance contiguous --a2a symm
  run "$OUT/p${pid}_asymhp" "$P" --balance greedy_unequal --a2a asymm \
      --online_schedule --cost_model_json "$COST" --min_heads_per_rank 2
done

echo
echo "| prompt | AsymHP vs baseline (dB) | baseline vs baseline (dB) |"
echo "|---|---:|---:|"
for pid in $PROMPTS; do
  a=$(python lb/video_psnr.py "$OUT/p${pid}_baseline.npy" "$OUT/p${pid}_asymhp.npy")
  c=$(python lb/video_psnr.py "$OUT/p${pid}_baseline.npy" "$OUT/p${pid}_baseline2.npy")
  echo "| $pid | $a | $c |"
done

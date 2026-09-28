#!/usr/bin/env bash
# End-to-end generation latency and critical-path breakdown (paper Table 2,
# Appendix Table 3).
#
# Workload: Wan2.1-1.3B, 720p, 120 requested frames (Wan rounds to 121 actual
# frames, N=111600), 50 denoising steps, SVG2 (SAP) top-p=0.9, four GPUs.
# Sparse attention is active from the first denoising step and first layer
# (FIRST_TIMES_FP=0, FIRST_LAYERS_FP=0) on both compared paths. The only
# compared variable is the distributed execution:
#   contiguous/symmetric baseline vs. AsymHP (online one-step-lag placement +
#   asymmetric exchange).
#
# Usage (four free H100s):
#   GPUS=0,1,2,3 bash lb/run_e2e.sh            # 5 paired repetitions
#   GPUS=0,1,2,3 REPS=1 NUM_FRAMES=125 bash lb/run_e2e.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:?Set GPUS to four confirmed-free H100 indices, e.g. 1,2,3,4}"
REPS="${REPS:-5}"
NUM_FRAMES="${NUM_FRAMES:-120}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
N=4
MODEL_ID="${MODEL_ID:-Wan-AI/Wan2.1-T2V-1.3B-Diffusers}"
OUT="${OUT:-result/paper/e2e_wan13b_720p_${NUM_FRAMES}f_w4}"
COST="${COST:-result/wan/t2v/sap_1.3b/maskgen_aware_cost.json}"
PROMPT="${PROMPT:-$(cat examples/1/prompt.txt)}"
FIRST_TIMES_FP="${FIRST_TIMES_FP:-0}"
FIRST_LAYERS_FP="${FIRST_LAYERS_FP:-0}"
mkdir -p "$OUT"

[[ -f "$COST" ]] || { echo "[ERR] missing cost model: $COST" >&2; exit 2; }
[[ "$(awk -F, '{print NF}' <<<"$GPUS")" -eq "$N" ]] || {
  echo "[ERR] GPUS must contain exactly $N comma-separated indices: $GPUS" >&2
  exit 2
}

SAP=(
  --pattern SAP
  --num_q_centroids 300
  --num_k_centroids 1000
  --top_p_kmeans 0.9
  --min_kc_ratio 0.10
  --kmeans_iter_init 50
  --kmeans_iter_step 2
)
BASE=(
  --model_id "$MODEL_ID"
  --height 720
  --width 1280
  --num_frames "$NUM_FRAMES"
  --num_inference_steps 50
  --warmup_steps "$WARMUP_STEPS"
  --seed 0
  --first_times_fp "$FIRST_TIMES_FP"
  --first_layers_fp "$FIRST_LAYERS_FP"
  --prompt "$PROMPT"
  "${SAP[@]}"
)

gpu_guard() {
  local busy=0
  IFS=, read -ra ids <<<"$GPUS"
  for id in "${ids[@]}"; do
    read -r mem util < <(
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$id" |
        awk -F', ' '{print $1, $2}'
    )
    if (( mem >= 1000 || util >= 10 )); then
      echo "[ERR] GPU $id is not free (memory=${mem} MiB, util=${util}%)." >&2
      busy=1
    fi
  done
  (( busy == 0 ))
}

run_one() {
  local tag="$1" rep="$2"
  local log="$OUT/${tag}_r${rep}.log"
  local dump="$OUT/${tag}_r${rep}.json"
  local video="$OUT/${tag}_r${rep}.mp4"
  local port=$((31000 + RANDOM % 1000))
  local args=()

  case "$tag" in
    baseline)
      args=(--balance contiguous --a2a symm)
      ;;
    asymhp)
      args=(
        --balance greedy_unequal
        --a2a asymm
        --online_schedule
        --cost_model_json "$COST"
        --min_heads_per_rank 1
      )
      ;;
    *)
      echo "[ERR] unknown tag: $tag" >&2
      return 2
      ;;
  esac

  gpu_guard
  echo "[$(date -Is)] START $tag rep=$rep GPUs=$GPUS warmup_steps=$WARMUP_STEPS" |
    tee -a "$OUT/driver.log"
  SVG_SP_ATTN_DUMP="$dump" CUDA_VISIBLE_DEVICES="$GPUS" \
    torchrun --nproc-per-node="$N" --master-port="$port" \
      wan_t2v_sp_inference.py "${BASE[@]}" "${args[@]}" --timing \
      --output_file "$video" >"$log" 2>&1
  grep -E "SPARSE-ATTN-REGION|SPARSE critical-path breakdown|ONLINE-SCHEDULE|E2E wall-clock|Traceback|Error" \
    "$log" | tail -n 12 | tee -a "$OUT/driver.log"
  [[ -s "$dump" ]] || { echo "[ERR] missing timing dump: $dump" >&2; return 3; }
  echo "[$(date -Is)] DONE $tag rep=$rep" | tee -a "$OUT/driver.log"
}

for rep in $(seq 1 "$REPS"); do
  # Pair baseline/AsymHP within each repetition to reduce temporal drift.
  run_one baseline "$rep"
  run_one asymhp "$rep"
done

python lb/summarize_e2e.py "$OUT" | tee "$OUT/summary.md"

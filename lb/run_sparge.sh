#!/usr/bin/env bash
# Second sparse method: SpargeAttn operator replay (paper Figure 6(b)).
#
# For each CDF threshold, calibrate the per-head cost model on SpargeAttn's own
# valid-block counts, plan from that density, and replay the distributed
# operator (contiguous/symmetric baseline vs AsymHP) with 20 warmups and 50
# measured iterations. Requires the patched SpargeAttn build (see README).
#
# Input snapshot (720p, 125 frames, layer 21, step 20):
#   RESOLUTION=720p NUM_FRAMES=125 FIRST_TIMES_FP=0 FIRST_LAYERS_FP=0 \
#     bash lb/dump_wan_1.3b_attn.sh
#
# The asymmetric path is sensitive to GPU co-tenancy; run on GPUs you own
# outright.
#
#   GPUS=0,1,2,3 bash lb/run_sparge.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:-0,1,2,3}"
G0="${GPUS%%,*}"
SNAP="${SNAP:-result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0-LFP_0/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_125frames/attn_dumps/1-0_step20_layer21.pt}"
OUT="${OUT:-result/paper/sparge}"
mkdir -p "$OUT"

for CDF in 0.6 0.7 0.8 0.9; do
  echo "===== CDF=$CDF (calibrated and planned on its own density) ====="
  rm -f "$OUT/cost_cdf${CDF}.json" "$OUT/op_cdf${CDF}.json"
  CUDA_VISIBLE_DEVICES="$G0" python lb/profile_sparge_cost.py \
    --input "$SNAP" --output "$OUT/cost_cdf${CDF}.json" \
    --simthreshd1 0.3 --cdfthreshd "$CDF" --pvthreshd 50 \
    >"$OUT/prof_${CDF}.log" 2>&1
  [[ -s "$OUT/cost_cdf${CDF}.json" ]] || { echo "  calibration FAILED (see $OUT/prof_${CDF}.log)"; continue; }
  CUDA_VISIBLE_DEVICES="$GPUS" timeout 900 torchrun --nproc-per-node=4 \
    --master-port=$((42000 + RANDOM % 400)) \
    lb/bench_sparge_operator.py \
    --input "$SNAP" --cost-model-json "$OUT/cost_cdf${CDF}.json" \
    --output "$OUT/op_cdf${CDF}.json" \
    --simthreshd1 0.3 --cdfthreshd "$CDF" --pvthreshd 50 \
    --warmup 20 --iters 50 >"$OUT/bench_${CDF}.log" 2>&1
  [[ -s "$OUT/op_cdf${CDF}.json" ]] || { echo "  replay FAILED (see $OUT/bench_${CDF}.log)"; continue; }
  python3 lb/sparge_report.py "$OUT/op_cdf${CDF}.json"
done
echo SPARGE_DONE

#!/usr/bin/env bash
# SVG2 sparsity sweep with per-setting planning (paper Appendix D.1, Table 7).
#
# Replays the same 720p/125-frame snapshot (layer 21, step 20) while varying
# only top-p; each setting is planned from its own density trace (produced by
# lb/run_topp_dumps.sh). 20 warmups and 50 measured iterations per point.
# Timing-sensitive: run on GPUs you own outright.
#
#   GPUS=0,1,2,3 bash lb/run_topp_sweep.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS="${GPUS:-0,1,2,3}"
OUT="${OUT:-result/paper/topp_sweep}"
COST="${COST:-result/wan/t2v/sap_1.3b/maskgen_aware_cost.json}"
SNAP="result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0-LFP_0/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_125frames/attn_dumps/1-0_step20_layer21.pt"
mkdir -p "$OUT"

dens_for() {   # density trace recorded at this top_p
  echo "result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0-LFP_0/QC_300-KC_1000-TopP_${1}/Init_50-Step_2-MinR_0.10_125frames/1-0.jsonl"
}

for TP in 0.7 0.8 0.9 0.95; do
  D=$(dens_for "$TP")
  if [ ! -s "$D" ]; then echo "  top_p=$TP: missing density trace $D"; continue; fi
  echo "===== top_p=$TP (planned from its own density trace) ====="
  for cfg in "contiguous off" "greedy_unequal pull_qkv"; do
    set -- $cfg
    CUDA_VISIBLE_DEVICES="$GPUS" timeout 900 torchrun --nproc-per-node=4 \
      --master-port=$((43000 + RANDOM % 400)) lb/bench_sp_all2all_attention.py \
      --input "$SNAP" --density-log "$D" --balance "$1" --asymm-a2a "$2" \
      --top_p "$TP" --warmup 20 --iters 50 \
      --cost-model-json "$COST" --min-heads-per-rank 1 \
      >"$OUT/${1}_tp${TP}.log" 2>&1
    tot=$(grep -A6 "^rank heads" "$OUT/${1}_tp${TP}.log" | awk '/^ *[0-9]/{print $NF}' | sort -rn | head -1)
    echo "  $1: max-rank total = ${tot:-n/a} ms"
  done
done
echo TOPP_SWEEP_DONE

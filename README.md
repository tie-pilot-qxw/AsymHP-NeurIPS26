# AsymHP: Load-Balanced Sparse Attention for Video Diffusion Transformers

Code for the NeurIPS 2026 paper
[*AsymHP: Load-Balanced Sparse Attention for Video Diffusion Transformers*](https://openreview.net/forum?id=XyeAckrYOC).

Dynamic sparse attention (top-p or threshold rules) retains very different
numbers of blocks per attention head, so equal-head parallel execution leaves
GPUs waiting for the ones that received dense heads. AsymHP predicts each
head's cost from the previous denoising step's per-head density, assigns a
non-uniform number of heads to each GPU, and moves only the required head
shards with an asymmetric pull–push exchange. The sparse masks and local
kernels are unchanged.

This repository builds on
[Sparse VideoGen / Sparse VideoGen2](https://github.com/svg-project/Sparse-VideoGen)
(Apache-2.0), which provides the SVG2 sparse-attention backend under `svg/`.
The AsymHP contribution lives in `lb/`.

## Repository layout

| Path | Contents |
|---|---|
| `lb/` | AsymHP: planner (`split_planner.py`, `predict_balance.py`), asymmetric exchange (`symm_a2a.py`, `asymm_pull_kernel.py`), operator benchmark (`bench_sp_all2all_attention.py`), cost-model profiling, experiment and analysis scripts, tests. |
| `lb/wan_sp/` | Sequence-parallel AsymHP integration into the Wan2.1 denoising loop, used by `wan_t2v_sp_inference.py` for end-to-end runs. |
| `wan_t2v_sp_inference.py` | Multi-GPU end-to-end Wan2.1 generation with equal-head or AsymHP execution. |
| `svg/`, `*_inference.py`, `scripts/`, `examples/` | Upstream SVG2 backend, model adapters, single-GPU drivers, and prompts (small SVG2 changes: Wan attention export, placement-invariant k-means seeding). |
| `result/*/maskgen_aware_cost.json` | Fitted H100 cost models for Wan2.1-1.3B, Wan2.1-14B, and HunyuanVideo. |
| `data/traces/` | Recorded per-head density traces for the GPU-free analyses. |
| `artifacts/l40s_w4_wan21_13b_720p_121f/` | Measured evidence for the PCIe-only L40S study. |

## Setup

We ran all experiments on H100 GPUs with CUDA 12.9, PyTorch 2.9.1, NCCL 2.27.5,
FlashInfer 0.5.3, diffusers 0.34.0, and transformers 4.57.1. The asymmetric
exchange uses PyTorch symmetric memory and TMA and requires Hopper (sm_90) for
the default path; pre-Hopper GPUs automatically use a PCIe copy backend.

The simplest environment is an NGC PyTorch container, e.g.
`nvcr.io/nvidia/pytorch:25.06-py3`:

```bash
git clone --recursive https://github.com/tie-pilot-qxw/AsymHP-NeurIPS26.git asymhp && cd asymhp
python -m venv --system-site-packages .venv && source .venv/bin/activate
pip install -e . --no-deps
pip install diffusers==0.34.0 transformers==4.57.1 accelerate \
            flashinfer-python==0.5.3 cuvs-cu12 einops av loguru termcolor
( cd svg/kernels && bash setup.sh )   # SVG2 CUDA/Triton kernels
```

Outside NGC, install `torch==2.9.1+cu129` and `flash-attn` first.
`lb/requirements-freeze.txt` lists the full package set of our container.

**SpargeAttn (only for the second-method experiment).** Build
[SpargeAttn](https://github.com/thu-ml/SpargeAttn) from source with our small
patch, which exposes per-head valid-block counts (`return_head_density=True`):

```bash
git clone https://github.com/thu-ml/SpargeAttn.git && cd SpargeAttn
git apply <asymhp>/lb/patches/sparge_per_head_density.patch
TORCH_CUDA_ARCH_LIST=9.0 pip install -e . --no-build-isolation --no-deps
```

**Notes.**

- Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` once models are cached, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the scripts do this).
- The first denoising steps JIT-compile FlashInfer; timed runs use warmup.
- The operator benchmark may abort in the `CUDASymmetricMemory` destructor at
  teardown; results are written before that point.
- The unpadded operator replay needs a per-GPU sequence length divisible by
  128, which is why some replays use 125 frames; `wan_t2v_sp_inference.py`
  pads internally.
- Timing runs are sensitive to other jobs on the same GPUs.

## Reproducing the paper

All commands run from the repository root. Outputs go to `result/`
(git-ignored). The shipped cost models in `result/*/maskgen_aware_cost.json`
are used by default; `lb/profile_cost.sh` refits them.

### GPU-free analyses

```bash
python lb/analyze_cost_sensitivity.py   # appendix: coefficient-error sensitivity
python lb/analyze_cost_portability.py   # appendix: HunyuanVideo planned with Wan fits
python lb/analyze_plan_stability.py     # appendix: per-step re-planning vs fixed placement
```

`lb/predict_balance.py` plans a recorded trace with every placement policy,
including the split-head simulation used for the whole-head granularity study
(whole-head granularity analysis in the appendix).

### Sparse-attention region (Figures 5 and 6(a), Table 1)

Each experiment replays one captured (layer, step) Q/K/V snapshot. First dump
the snapshot and its density trace on one GPU, then run the comparison:

```bash
RESOLUTION=720p NUM_FRAMES=120 bash lb/dump_wan_1.3b_attn.sh

# Figure 5: WORLD_SIZE in {2,3,4,5,6,8}; same pattern for the other models
RESOLUTION=720p NUM_FRAMES=120 WORLD_SIZE=4 bash lb/run_wan_1.3b_balance_compare.sh
RESOLUTION=720p NUM_FRAMES=120 WORLD_SIZE=8 bash lb/run_wan_14b_balance_compare.sh
RESOLUTION=720p NUM_FRAMES=120 WORLD_SIZE=8 bash lb/run_hunyuan_t2v_balance_compare.sh

# Figure 6(a): NUM_FRAMES in {15,30,60,120,240}, four GPUs
RESOLUTION=720p NUM_FRAMES=240 WORLD_SIZE=4 bash lb/run_wan_1.3b_balance_compare.sh

# Table 1 (placement ablation; the paper used 20 iterations after 5 warmups)
RESOLUTION=720p NUM_FRAMES=240 NPROC=4 ITERS=20 WARMUP=5 bash lb/run_wan_ablation_placement.sh
```

The Wan2.1-14B and HunyuanVideo dumps use `lb/dump_wan_14b_attn.sh` and
`lb/dump_hunyuan_t2v_attn.sh` with the same variables.

### End-to-end generation (Table 2 and the appendix critical-path breakdown)

```bash
GPUS=0,1,2,3 bash lb/run_e2e.sh
```

Runs five paired generations (equal-head baseline and AsymHP; Wan2.1-1.3B,
720p, 120 requested frames, 50 steps, sparse from the first step and layer) and
writes a summary with the E2E speedup, sparse-region fraction, Amdahl
estimate, and AsymHP critical-path breakdown.

### Second sparse method, PCIe, and sparsity sweep

```bash
# 125-frame snapshot (layer 21, step 20), sparse from the first step/layer
RESOLUTION=720p NUM_FRAMES=125 FIRST_TIMES_FP=0 FIRST_LAYERS_FP=0 bash lb/dump_wan_1.3b_attn.sh

GPUS=0,1,2,3 bash lb/run_sparge.sh        # Figure 6(b): SpargeAttn, CDF 0.6-0.9
GPUS=0,1,2 bash lb/run_topp_dumps.sh      # density traces for top-p 0.7/0.8/0.95
GPUS=0,1,2,3 bash lb/run_topp_sweep.sh    # appendix: measured sparsity sweep
```

The PCIe study (Table 3) runs `lb/bench_sp_all2all_attention.py` on
the 120-frame snapshot on four L40S GPUs; the measured per-rank breakdowns are
in `artifacts/l40s_w4_wan21_13b_720p_121f/`.

### Numerical agreement (appendix)

```bash
# operator level, on the 120-frame snapshot
SNAP=result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_120frames
W=2 python lb/op_parity.py $SNAP/attn_dumps/1-0_step20_layer21.pt $SNAP/1-0.jsonl
W=4 MIN_HEADS=2 python lb/op_parity.py $SNAP/attn_dumps/1-0_step20_layer21.pt $SNAP/1-0.jsonl

# decoded videos, three prompts with baseline repeatability controls
GPUS=0,1,2,3 bash lb/run_video_parity.sh
```

## Tests

```bash
python lb/test_split_planner.py                         # CPU
python lb/test_head_index_staging.py                    # CPU
torchrun --nproc_per_node=2 lb/test_asymm_pull.py       # asymmetric exchange, H100
torchrun --nproc_per_node=2 lb/test_symmetric_memory_basic.py
```

## Citation

```bibtex
@inproceedings{qiang2026asymhp,
  title     = {Asym{HP}: Load-Balanced Sparse Attention for Video Diffusion Transformers},
  author    = {Qiang, Xinwei and Guan, Yue and Zhu, Ruihan and Jagtap, Mihir and
               Pan, Zaifeng and Yu, Zhongkai and Chen, Chang and Hu, Zhengding and
               Ding, Yufei and Aziz, Adnan},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=XyeAckrYOC}
}
```

Please also cite Sparse VideoGen and Sparse VideoGen2 if you use the SVG2
backend.

## License

Apache License 2.0; see `LICENSE.txt`. Third-party components keep their
licenses; see `THIRD_PARTY.md`.

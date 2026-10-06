"""Sequence-parallel + load-balanced Wan 2.1 T2V inference (end-to-end).

Launch with torchrun; each rank loads the full model (weights replicated),
shards the token sequence, and runs head-parallel load-balanced sparse
attention via lb/wan_sp. Reports E2E wall-clock and the per-rank self-attention
breakdown so the load-balancing effect is visible end-to-end.

Example (Wan 1.3B, world=6, contiguous baseline):
  torchrun --nproc-per-node=6 wan_t2v_sp_inference.py \
      --model_id Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
      --height 480 --width 832 --num_frames 480 --num_inference_steps 50 \
      --balance contiguous --timing --output_file result/sp/contig.mp4

Load-balanced (greedy) needs a density log from a single-card SAP dump:
  ... --balance greedy --density_log <path.jsonl>
"""

import argparse
import json
import math
import os
import time
from copy import deepcopy

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from termcolor import colored

from dataloader import load_prompt_or_image
from svg.logger import logger
from svg.utils.seed import seed_everything

# lb/ package
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lb"))
from wan_sp import SPContext, install_wan_sp, get_sp_context  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="SP + load-balanced Wan T2V inference")
    p.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--negative_prompt", type=str, default=None)
    p.add_argument("--prompt_source", type=str, default="prompt", choices=["prompt", "T2V_Wan_VBench"])
    p.add_argument("--prompt_idx", type=int, default=0)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=480)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--flow_shift", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_file", type=str, default="result/sp/output.mp4")
    p.add_argument("--save_latents", type=str, default=None, help="Optional .npy path for the decoded video frames (float32 in [0,1]; see lb/video_psnr.py).")

    # SP / load balancing
    p.add_argument("--balance", type=str, default="contiguous",
                   choices=["contiguous", "greedy", "greedy_unequal"])
    p.add_argument("--a2a", type=str, default="symm", choices=["symm", "asymm"])
    p.add_argument("--density_log", type=str, default=None, help="JSONL density log for greedy/greedy_unequal.")
    p.add_argument("--cost_model_json", type=str, default=None)
    p.add_argument("--min_heads_per_rank", type=int, default=1)
    p.add_argument("--timing", action="store_true", help="Record per-rank attention breakdown.")

    # warmup / SAP hyperparams (match lb/dump_wan_1.3b_attn.sh defaults)
    p.add_argument("--first_layers_fp", type=float, default=0.03)
    p.add_argument("--first_times_fp", type=float, default=0.2)
    p.add_argument("--num_q_centroids", "--qc", type=int, default=300)
    p.add_argument("--num_k_centroids", "--kc", type=int, default=1000)
    p.add_argument("--top_p_kmeans", type=float, default=0.9)
    p.add_argument("--min_kc_ratio", type=float, default=0.10)
    # Local sparse method: SAP (kmeans+permute) or the real SpargeAttn.
    p.add_argument("--pattern", type=str, default="SAP", choices=["SAP", "SpargeAttn"])
    p.add_argument("--simthreshd1", type=float, default=0.6, help="SpargeAttn mean-similarity threshold.")
    p.add_argument("--cdfthreshd", type=float, default=0.98, help="SpargeAttn CDF/top-p threshold.")
    p.add_argument("--pvthreshd", type=int, default=50, help="SpargeAttn PV threshold.")
    p.add_argument("--online_schedule", "--online", action="store_true",
                   help="Causal one-step-lag scheduler: re-plan each layer from the previous denoising "
                        "step's realized density (paper-faithful). Default off = fixed install-time plan.")
    p.add_argument("--kmeans_iter_init", type=int, default=50)
    p.add_argument("--kmeans_iter_step", type=int, default=2)
    p.add_argument("--warmup_steps", type=int, default=2,
                   help="Denoising steps to run (untimed) before the measured run, to "
                        "compile all Triton/flashinfer kernels so JIT is NOT in the E2E timer.")
    p.add_argument("--profile_nvtx", action="store_true",
                   help="Emit phase NVTX ranges and one outer capture range spanning complete "
                        "transformer passes. Intended for Nsight Systems capture-range=nvtx.")
    p.add_argument("--profile_cuda_api", action="store_true",
                   help="Have rank 0 delimit the same complete-pass window with "
                        "cudaProfilerStart/Stop for Nsight Systems.")
    p.add_argument("--profile_pass_start", type=int, default=32,
                   help="Zero-based measured transformer pass at which the NVTX capture begins.")
    p.add_argument("--profile_pass_count", type=int, default=2,
                   help="Number of complete transformer passes inside the NVTX capture range.")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- distributed init ----
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        # Pass device_id so NCCL binds each rank to its exact GPU instead of
        # "guessing device ID based on global rank" (which can allocate comm
        # buffers on the wrong/full GPU and OOM).
        dist.init_process_group(backend="nccl", init_method="env://", device_id=device)

    is_main = rank == 0
    seed_everything(args.seed)
    torch.backends.cuda.preferred_linalg_library(backend="magma")

    # ---- load model (replicated on every rank) ----
    model_id = args.model_id
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=args.flow_shift
    )
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to(device)
    config = pipe.transformer.config

    # ---- translate warmup percentages to absolute layer/timestep (same as single-card) ----
    ref_scheduler = deepcopy(pipe.scheduler)
    ref_scheduler.set_timesteps(args.num_inference_steps)
    num_fp_timesteps = math.floor(args.first_times_fp * args.num_inference_steps)
    num_fp_layers = math.floor(args.first_layers_fp * config.num_layers)
    first_times_fp = (ref_scheduler.timesteps[num_fp_timesteps - 1] - 1) if num_fp_timesteps > 0 else 1001
    first_layers_fp = num_fp_layers
    if is_main:
        logger.info(f"Warmup timesteps: {num_fp_timesteps}/{args.num_inference_steps} (t<={first_times_fp} use FP)")
        logger.info(f"Warmup layers: {num_fp_layers}/{config.num_layers} use FP")

    # ---- prompt ----
    args.prompt, _ = load_prompt_or_image(args.prompt_source, args.prompt_idx, args.prompt, None)
    if args.prompt is None:
        args.prompt = "A cat walks on the grass, realistic"
    if args.negative_prompt is None:
        args.negative_prompt = (
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
            "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
            "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
            "fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
        )
    if is_main:
        print("=" * 20 + " Prompt " + "=" * 20)
        print(args.prompt)

    # Pre-encode the prompt and free the (~11 GB) text encoder before the run.
    # This shrinks each rank's footprint to roughly the transformer and VAE.
    import gc
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
            prompt=args.prompt, negative_prompt=args.negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1, device=device,
        )
    pipe.text_encoder = None
    if hasattr(pipe, "text_encoder_2"):
        pipe.text_encoder_2 = None
    gc.collect()
    torch.cuda.empty_cache()

    # ---- install SP + load balancing ----
    ctx = SPContext(rank=rank, world_size=world_size, local_rank=local_rank, device=device, a2a_backend=args.a2a)
    ctx.timing.enabled = args.timing
    ctx.profile_nvtx = args.profile_nvtx
    ctx.profile_cuda_api = args.profile_cuda_api
    ctx.profile_pass_start = args.profile_pass_start
    ctx.profile_pass_count = args.profile_pass_count
    # Time install separately: setup_asymm allocates + RENDEZVOUS the symmetric-
    # memory buffers here (eagerly), so this one-time cost is measured and NOT in
    # the E2E timer below.
    if world_size > 1:
        dist.barrier()
    torch.cuda.synchronize(device)
    _t_setup = time.perf_counter()
    install_wan_sp(
        pipe, ctx, args.height, args.width, args.num_frames,
        strategy=args.balance,
        density_log_path=args.density_log,
        cost_model_path=args.cost_model_json,
        min_heads_per_rank=args.min_heads_per_rank,
        first_layers_fp=first_layers_fp,
        first_times_fp=first_times_fp,
        num_q_centroids=args.num_q_centroids,
        num_k_centroids=args.num_k_centroids,
        top_p_kmeans=args.top_p_kmeans,
        min_kc_ratio=args.min_kc_ratio,
        kmeans_iter_init=args.kmeans_iter_init,
        kmeans_iter_step=args.kmeans_iter_step,
        pattern=args.pattern,
        simthreshd1=args.simthreshd1,
        cdfthreshd=args.cdfthreshd,
        pvthreshd=args.pvthreshd,
        online=args.online_schedule,
    )
    # Snapshot the install-time (bootstrap) plans so we can restore them after the
    # warm-up run mutates ctx.plans under the online scheduler (online_replan
    # REPLACES entries, so a shallow copy preserves the originals).
    _bootstrap_plans = dict(ctx.plans)

    torch.cuda.synchronize(device)
    setup_s = time.perf_counter() - _t_setup  # install incl. symm alloc/rendezvous

    # ---- warmup (UNtimed): compile all Triton/flashinfer kernels so JIT and any
    # lazy first-call init do not land in the measured E2E. Events recorded during
    # warmup are cleared afterwards so the per-call dump is steady-state only.
    warm_s = 0.0
    if args.warmup_steps > 0:
        was = ctx.timing.enabled
        ctx.timing.enabled = False
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize(device)
        _tw = time.perf_counter()
        _ = pipe(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
            height=args.height, width=args.width, num_frames=args.num_frames,
            guidance_scale=5.0, num_inference_steps=args.warmup_steps,
        )
        torch.cuda.synchronize(device)
        warm_s = time.perf_counter() - _tw
        ctx.timing.enabled = was
        for _k in list(ctx.timing.events):           # drop warmup events
            ctx.timing.events[_k] = []
        ctx.timing.sparse_flags = []
        ctx.timing.n_calls = 0
        ctx.timing._pending_block = None
        ctx.timing.sched_ms = 0.0                     # drop warmup scheduling time
        ctx.timing.sched_calls = 0

        # Reset ALL run state the warm-up mutated, so the timed generation starts
        # from a clean, deterministic point (identical to a no-warmup run):
        #  - RNG: warm-up's SAP k-means torch.randint advanced each rank's RNG by
        #    its local head count, which DIFFERS under greedy_unequal -> ranks
        #    would otherwise desync and generate different initial latents.
        #  - k-means centroid caches + online centroid store + pending density +
        #    the (online-mutated) plans -> back to the install bootstrap.
        seed_everything(args.seed)
        if world_size > 1:
            dist.barrier()
        torch.cuda.synchronize(device)
        ctx._pending_dens = {}
        ctx.plans = dict(_bootstrap_plans)
        for _blk in pipe.transformer.blocks:
            _p = getattr(_blk.attn1, "processor", None)
            if _p is None:
                continue
            _p.centroids_init = False
            _p.q_centroids = None
            _p.k_centroids = None
            _p._centroid_store = None
            _p._pending_centroids = None  # drop any async-published migration from warmup
            _p._last_density = None

    # ---- generate (timed, fully warm) ----
    if world_size > 1:
        dist.barrier()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    result = pipe(
        prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        height=args.height, width=args.width, num_frames=args.num_frames,
        guidance_scale=5.0, num_inference_steps=args.num_inference_steps,
    )
    torch.cuda.synchronize(device)
    if world_size > 1:
        dist.barrier()
    e2e = time.perf_counter() - t0

    if is_main:
        print(colored(f"[wan-sp] setup (install incl symm alloc/rendezvous) = {setup_s:.2f} s;  "
                      f"warmup ({args.warmup_steps} steps, JIT) = {warm_s:.2f} s  "
                      f"[both EXCLUDED from E2E below]", "magenta"))
    report_timing(ctx, e2e, args)

    # ---- export on rank 0 ----
    if is_main:
        video = result.frames[0]  # list/array of [H, W, C] frames in [0,1] (np, default)
        out_dir = os.path.dirname(args.output_file)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        export_to_video(video, args.output_file, fps=16)
        print(colored(f"[wan-sp] saved video -> {args.output_file}", "green"))
        if args.save_latents:
            import numpy as np
            np.save(args.save_latents, np.asarray(video))

    # Barrier so non-main ranks WAIT for rank 0 to finish export/save before the
    # process tears down (else a rank races ahead and torchrun kills rank 0
    # mid-save). Then hard-exit BEFORE freeing symm — the CUDASymmetricMemory
    # destructor aborts (SIGABRT) on a stale CUDA handle at teardown; all real
    # work is done, so os._exit(0) skips it for a clean exit.
    sys.stdout.flush()
    sys.stderr.flush()
    if world_size > 1:
        dist.barrier()
    os._exit(0)


def report_timing(ctx: SPContext, e2e: float, args):
    is_main = ctx.rank == 0
    if not ctx.timing.enabled:
        if is_main:
            print(colored(f"\n[wan-sp] E2E wall-clock: {e2e:.2f} s "
                          f"(W={ctx.world_size} balance={args.balance} a2a={ctx.a2a_backend})", "cyan"))
        return

    stats = ctx.timing.reduce()  # this rank's per-region ms sums
    per_rank = {
        "attn": stats["attn_ms"],
        "a2a_in": stats["a2a_in_ms"],
        "a2a_out": stats["a2a_out_ms"],
        "reorder": stats["reorder_ms"],
        "compute_total": stats["attn_ms"] + stats["a2a_in_ms"] + stats["a2a_out_ms"] + stats["reorder_ms"],
    }
    attn_calls = ctx.timing.attn_per_call()  # per-(layer,call) attn ms, execution order
    payload = {"per_rank": per_rank, "attn_calls": attn_calls,
               "sparse_flags": list(ctx.timing.sparse_flags),
               # symm-path per-call regions for the whole-self-attn-layer metric (metric C)
               "reorder_calls": ctx.timing.per_call("reorder"),
               "a2a_in_calls": ctx.timing.per_call("a2a_in"),
               "a2a_out_calls": ctx.timing.per_call("a2a_out"),
               # asymm-path per-call regions (AsymHP: layout exchange=pull, redistribution=push,
               # plus barriers=sync); metric C for asymm = copy+pre_bar+pull_k+attn+push_k+push_bar
               # (there is no separate post-pull barrier, so pull_barrier stays empty)
               "copy_calls": ctx.timing.per_call("copy"),
               "pre_barrier_calls": ctx.timing.per_call("pre_barrier"),
               "pull_kernel_calls": ctx.timing.per_call("pull_kernel"),
               "pull_barrier_calls": ctx.timing.per_call("pull_barrier"),
               "push_kernel_calls": ctx.timing.per_call("push_kernel"),
               "push_barrier_calls": ctx.timing.per_call("push_barrier"),
               # local sparse-attention phases; these arrays include sparse
               # invocations only and are aligned through ``sparse_flags``.
               "local_mask_calls": ctx.timing.per_call("local_mask"),
               "local_sparse_kernel_calls": ctx.timing.per_call("local_sparse_kernel"),
               "local_inverse_permute_calls": ctx.timing.per_call("local_inverse_permute"),
               "local_density_calls": ctx.timing.per_call("local_density"),
               "density_exchange_calls": ctx.timing.per_call("density_exchange"),
               # whole transformer block per call (metric D)
               "block_calls": ctx.timing.per_call("block"),
               # online causal scheduler overhead (host wall-time: density all-reduce + CPU re-plan)
               "schedule_ms": stats["schedule_ms"], "schedule_calls": stats["schedule_calls"]}
    if ctx.world_size > 1:
        obj_list = [None] * ctx.world_size
        dist.all_gather_object(obj_list, payload)
    else:
        obj_list = [payload]

    if is_main:
        gathered = [o["per_rank"] for o in obj_list]
        # TRUE attention makespan: per call, the slowest rank bounds that layer
        # (all2all barrier), so sum over calls of max-over-ranks. Per-rank SUMS
        # average out because different layers are hot on different head-groups.
        calls = [o["attn_calls"] for o in obj_list]
        L = min(len(c) for c in calls)
        true_makespan = sum(max(c[i] for c in calls) for i in range(L))
        mean_over_ranks = sum(sum(c[:L]) for c in calls) / len(calls)
        per_rank_sums = [g["attn"] for g in gathered]

        # ---- per-call metrics (ALWAYS computed; call index i -> layer i % num_layers) ----
        def region(name):
            arr = [o.get(name, []) for o in obj_list]
            return arr if all(len(a) >= L for a in arr) else None
        # metric B: attention compute only (kmeans+mask+block-sparse+invperm).
        per_call_max = [max(c[i] for c in calls) for i in range(L)]
        per_call_mean = [sum(c[i] for c in calls) / len(calls) for i in range(L)]
        # metric C: the FULL sparse-attention region (paper's metric = layout exchange +
        # mask construction + sparse kernel + output redistribution + synchronization).
        rr, ai, ao = region("reorder_calls"), region("a2a_in_calls"), region("a2a_out_calls")
        per_call_C_max = per_call_C_mean = None
        per_rank_C = None
        if rr and ai and ao and any(sum(a) > 0 for a in ai):
            # symm metric C = reorder + a2a_in + attn + a2a_out
            tot = [[rr[r][i] + ai[r][i] + c[i] + ao[r][i] for i in range(L)] for r, c in enumerate(calls)]
            per_rank_C = tot
            per_call_C_max = [max(t[i] for t in tot) for i in range(L)]
            per_call_C_mean = [sum(t[i] for t in tot) / len(tot) for i in range(L)]
        else:
            # asymm metric C = copy + pre_barrier + pull_kernel + attn + push_kernel
            #             + push_barrier  (AsymHP sparse-attn region; the 2-barrier
            #             protocol has NO post-pull barrier -- see context.py)
            cp, pb = region("copy_calls"), region("pre_barrier_calls")
            pk = region("pull_kernel_calls")
            uk, ubar = region("push_kernel_calls"), region("push_barrier_calls")
            if all(x is not None for x in (cp, pb, pk, uk, ubar)):
                tot = [[cp[r][i] + pb[r][i] + pk[r][i] + c[i] + uk[r][i] + ubar[r][i]
                        for i in range(L)] for r, c in enumerate(calls)]
                per_rank_C = tot
                per_call_C_max = [max(t[i] for t in tot) for i in range(L)]
                per_call_C_mean = [sum(t[i] for t in tot) / len(tot) for i in range(L)]
        # metric D: whole transformer block (self-attn + cross-attn + FFN + norms).
        blk = region("block_calls")
        per_call_D_max = per_call_D_mean = None
        if blk:
            Lb = min(len(b) for b in blk)
            Ld = min(L, Lb)
            per_call_D_max = [max(b[i] for b in blk) for i in range(Ld)]
            per_call_D_mean = [sum(b[i] for b in blk) / len(blk) for i in range(Ld)]

        # ---- PAPER METRIC: sparse-attention-region latency averaged over invocations ----
        # Each invocation's latency = max-over-ranks (the region's actual wall time, bounded
        # by the slowest rank); then mean over the L (layer,step) invocations in this run.
        if per_call_C_max is not None:
            paper_C = sum(per_call_C_max) / len(per_call_C_max)
            print(colored(
                f"[wan-sp] SPARSE-ATTN-REGION latency (metric C) avg/invocation = {paper_C:.3f} ms  "
                f"over {L} invocations  [balance={args.balance} a2a={ctx.a2a_backend} W={ctx.world_size}]",
                "green"))

        # Detailed breakdown along the actual per-invocation critical rank.
        # This preserves the paper's max-over-ranks latency semantics while
        # separating mask construction, the sparse kernel, communication, and
        # synchronization. Only sparse invocations are included.
        critical_breakdown = None
        flags = obj_list[0].get("sparse_flags", [])
        if per_rank_C is not None and len(flags) >= L:
            sparse_ids = [i for i in range(L) if flags[i]]

            def _aligned_sparse(rank_obj, key):
                vals = iter(rank_obj.get(key, []))
                out = []
                for sparse in rank_obj.get("sparse_flags", [])[:L]:
                    out.append(next(vals, 0.0) if sparse else 0.0)
                return out

            if sparse_ids:
                phase_keys = {
                    "mask": "local_mask_calls",
                    "kernel": "local_sparse_kernel_calls",
                    "inverse": "local_inverse_permute_calls",
                    "density": "local_density_calls",
                }
                aligned = {
                    name: [_aligned_sparse(o, key) for o in obj_list]
                    for name, key in phase_keys.items()
                }
                sums = {
                    "mask": 0.0, "kernel": 0.0, "inverse": 0.0,
                    "density": 0.0, "local_other": 0.0,
                    "communication": 0.0, "synchronization": 0.0, "copy": 0.0,
                }
                asymm_components = {
                    "copy": region("copy_calls"),
                    "pre": region("pre_barrier_calls"),
                    "pull": region("pull_kernel_calls"),
                    "push": region("push_kernel_calls"),
                    "post": region("push_barrier_calls"),
                }
                for i in sparse_ids:
                    r = max(range(len(obj_list)), key=lambda x: per_rank_C[x][i])
                    known_local = 0.0
                    for name in ("mask", "kernel", "inverse", "density"):
                        v = aligned[name][r][i]
                        sums[name] += v
                        known_local += v
                    sums["local_other"] += max(0.0, calls[r][i] - known_local)
                    if all(v is not None for v in asymm_components.values()):
                        sums["copy"] += asymm_components["copy"][r][i]
                        sums["communication"] += (
                            asymm_components["pull"][r][i] + asymm_components["push"][r][i]
                        )
                        sums["synchronization"] += (
                            asymm_components["pre"][r][i] + asymm_components["post"][r][i]
                        )
                critical_breakdown = {k: v / len(sparse_ids) for k, v in sums.items()}
                if ctx.a2a_backend == "asymm":
                    summary = "  ".join(f"{k}={v:.3f}" for k, v in critical_breakdown.items())
                    print(colored(
                        f"[wan-sp] SPARSE critical-path breakdown avg/invocation (ms): {summary}",
                        "magenta"))
        # Online causal-scheduler overhead (host: density all-reduce + CPU re-plan),
        # max over ranks. NOTE: this is the EXPOSED (on-critical-path) scheduling
        # cost only -- the event-sync wait + CPU LPT plan build consumed in
        # plan_for_layer. The producer side (host alloc/scatter, the NCCL density
        # all-reduce, and the D2H copy) runs on a side stream one step ahead and is
        # deliberately NOT counted here because it overlaps compute (that is the
        # point of the one-step lag). 0 when --online is off.
        sched_tot = max(o.get("schedule_ms", 0.0) for o in obj_list)
        sched_calls = max(o.get("schedule_calls", 0) for o in obj_list)
        if sched_calls:
            print(colored(
                f"[wan-sp] ONLINE-SCHEDULE exposed cost (event-sync + plan-build, on critical path) "
                f"= {sched_tot:.2f} ms total / {sched_tot/max(sched_calls,1):.4f} ms per (layer,step) "
                f"over {sched_calls} re-plans; density all-reduce + D2H overlap on a side stream "
                f"[online causal one-step-lag]", "yellow"))

        dump_path = os.environ.get("SVG_SP_ATTN_DUMP")
        if dump_path:
            with open(dump_path, "w") as f:
                json.dump({"num_layers": len(ctx.plans), "balance": args.balance,
                           "world_size": ctx.world_size, "a2a": ctx.a2a_backend,
                           "pattern": args.pattern,
                           "online_schedule": ctx.online_schedule,
                           "sparse_flags": flags,
                           "per_call_max": per_call_max, "per_call_mean": per_call_mean,
                           "per_call_C_max": per_call_C_max, "per_call_C_mean": per_call_C_mean,
                           "per_call_D_max": per_call_D_max, "per_call_D_mean": per_call_D_mean,
                           "critical_sparse_breakdown_ms": critical_breakdown,
                           "schedule_exposed_total_ms": sched_tot,
                           "schedule_calls": sched_calls,
                           "density_exchange_rank0_total_ms": stats.get("density_exchange_ms", 0.0)}, f)
            print(f"[wan-sp] per-call dump (metric B attn / C sparse-attn region / D whole block) -> {dump_path}")

        print(colored("\n" + "=" * 64, "cyan"))
        print(colored(f"[wan-sp] E2E wall-clock: {e2e:.2f} s  "
                      f"(W={ctx.world_size} balance={args.balance} a2a={ctx.a2a_backend} "
                      f"calls={stats['n_calls']})", "cyan"))
        print(f"[wan-sp] per rank (ms, summed over layers*steps):")
        for r, g in enumerate(gathered):
            print(f"    rank {r}: attn={g['attn']:8.1f}  a2a_in={g['a2a_in']:7.1f}  "
                  f"a2a_out={g['a2a_out']:7.1f}  reorder={g['reorder']:6.1f}")
        print(f"[wan-sp] per-rank attn sums={[round(x) for x in per_rank_sums]}  "
              f"(these AVERAGE OUT the per-layer imbalance)")
        print(colored(
            f"[wan-sp] TRUE attn makespan (sum of per-call max-over-ranks) = {true_makespan:.1f} ms;  "
            f"balanced-ideal (mean) = {mean_over_ranks:.1f} ms;  "
            f"per-layer imbalance = {true_makespan/mean_over_ranks:.3f}x", "yellow"))
        # asymm-path breakdown (rank-0 sums) if present
        if stats.get("copy_ms") or stats.get("pull_kernel_ms"):
            print(colored(
                f"[wan-sp] asymm breakdown (rank0, ms): copy={stats.get('copy_ms',0):.1f}  "
                f"pre_bar={stats.get('pre_barrier_ms',0):.1f}  "
                f"pull_k={stats.get('pull_kernel_ms',0):.1f}  pull_bar={stats.get('pull_barrier_ms',0):.1f}  "
                f"push_k={stats.get('push_kernel_ms',0):.1f}  push_bar={stats.get('push_barrier_ms',0):.1f}  "
                f"attn={stats.get('attn_ms',0):.1f}", "magenta"))
        sparse_n = sum(ctx.timing.sparse_flags)
        if sparse_n:
            phase_totals = {
                "mask": stats.get("local_mask_ms", 0.0),
                "kernel": stats.get("local_sparse_kernel_ms", 0.0),
                "inverse": stats.get("local_inverse_permute_ms", 0.0),
                "density": stats.get("local_density_ms", 0.0),
            }
            phase_known = sum(phase_totals.values())
            sparse_attn_total = sum(
                v for v, sparse in zip(attn_calls, ctx.timing.sparse_flags) if sparse
            )
            phase_other = max(0.0, sparse_attn_total - phase_known)
            print(colored(
                f"[wan-sp] local sparse breakdown (rank0, ms over {sparse_n} sparse calls): "
                f"mask={phase_totals['mask']:.1f}  kernel={phase_totals['kernel']:.1f}  "
                f"inverse={phase_totals['inverse']:.1f}  density={phase_totals['density']:.1f}  "
                f"other={phase_other:.1f}",
                "magenta"))
        print(colored("=" * 64, "cyan"))


if __name__ == "__main__":
    main()

"""Install the sequence-parallel + load-balanced attention stack onto a
diffusers ``WanPipeline`` (already loaded, weights replicated on each rank).

Order matters:
  1. ``replace_wan_attention(pattern="SAP")`` sets the SAP class config, builds
     masks, installs SAP processors on attn1, and monkeypatches the DiT/block
     forwards (``replace_sparse_forward``).
  2. Swap each attn1 processor to the SP subclass (inherits SAP config).
  3. Build the fixed per-layer head plans from the density profile.
  4. Monkeypatch the DiT forward with the sequence-parallel version.
"""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist

from svg.models.wan.attention import (
    WanAttn_SAPAttn_Processor,
    WanAttn_SpargeAttn_Processor,
)
from svg.models.wan.inference import replace_wan_attention

from .attention import WanAttn_SP_Processor, WanAttn_SP_Sparge_Processor
from .context import SPContext, set_sp_context
from .forward import wan_sp_dit_forward
from .planning import build_layer_plans, summarize_plans


def install_wan_sp(
    pipe,
    ctx: SPContext,
    height: int,
    width: int,
    num_frames: int,
    *,
    strategy: str = "contiguous",
    density_log_path: Optional[str] = None,
    cost_model_path: Optional[str] = None,
    min_heads_per_rank: int = 1,
    first_layers_fp: int = 0,
    first_times_fp: float = 1001,
    num_q_centroids: int = 300,
    num_k_centroids: int = 1000,
    top_p_kmeans: float = 0.9,
    min_kc_ratio: float = 0.10,
    kmeans_iter_init: int = 50,
    kmeans_iter_step: int = 2,
    pattern: str = "SAP",
    simthreshd1: float = 0.6,
    cdfthreshd: float = 0.98,
    pvthreshd: int = 50,
    online: bool = False,
) -> SPContext:
    # 1. Install the single-card sparse stack (config, masks, processors, forwards).
    #    `pattern` selects the LOCAL per-rank sparse method; AsymHP's placement +
    #    asymmetric a2a are identical either way (they only see per-head density).
    if pattern == "SpargeAttn":
        replace_wan_attention(
            pipe, height, width, num_frames,
            first_layers_fp=first_layers_fp, first_times_fp=first_times_fp,
            pattern="SpargeAttn", simthreshd1=simthreshd1, cdfthreshd=cdfthreshd,
            pvthreshd=pvthreshd, logging_file=None,
        )
        base_cls = WanAttn_SpargeAttn_Processor
        sp_cls = WanAttn_SP_Sparge_Processor
    else:
        replace_wan_attention(
            pipe, height, width, num_frames,
            first_layers_fp=first_layers_fp, first_times_fp=first_times_fp,
            pattern="SAP",
            num_q_centroids=num_q_centroids, num_k_centroids=num_k_centroids,
            top_p_kmeans=top_p_kmeans, min_kc_ratio=min_kc_ratio, logging_file=None,
            kmeans_iter_init=kmeans_iter_init, kmeans_iter_step=kmeans_iter_step,
        )
        base_cls = WanAttn_SAPAttn_Processor
        sp_cls = WanAttn_SP_Processor

    num_layers = len(pipe.transformer.blocks)
    num_heads = pipe.transformer.config.num_attention_heads
    # num_frame / frame_size were computed and stored on the sparse class.
    seq_len = int(base_cls.num_frame) * int(base_cls.frame_size)

    # 2. Swap attn1 processors to the SP variant (inherits the sparse class config).
    for layer_idx, block in enumerate(pipe.transformer.blocks):
        block.attn1.set_processor(sp_cls(layer_idx=layer_idx))

    ctx.online_schedule = bool(online) and strategy in ("greedy", "greedy_unequal")

    # 3. Build the per-layer head plans.
    if ctx.online_schedule:
        # Paper-faithful bootstrap (design.tex): step 0 uses a deterministic
        # STATIC placement, NOT an offline density profile. This avoids requiring
        # an offline trace and avoids leaking future-timestep density into the
        # plan; from step 1 on, the causal one-step-lag scheduler re-plans from
        # each step's realized density. Uses a near-even ("static") placement --
        # not equal-count "contiguous" -- so the non-divisible head/GPU configs
        # the paper highlights (e.g. 12 heads on 5 GPUs) bootstrap without raising.
        ctx.plans = build_layer_plans(num_layers, num_heads, ctx.world_size, "static", seq_len)
    else:
        ctx.plans = build_layer_plans(
            num_layers=num_layers,
            num_heads=num_heads,
            world_size=ctx.world_size,
            strategy=strategy,
            seq_len=seq_len,
            density_log_path=density_log_path,
            cost_model_path=cost_model_path,
            min_heads_per_rank=min_heads_per_rank,
        )

    # Materialize the bootstrap plan's head indices during setup, outside the
    # measured generation. Online replacement plans are prepared in
    # SPContext.plan_for_layer through the same pinned-staging path.
    for plan in ctx.plans.values():
        plan.prepare_h_idxs(ctx.rank, ctx.device)

    # 3a. Online causal scheduler (paper-faithful one-step-lag): each layer
    # re-plans from the density it realized this step, for the next step. The
    # tiny per-head density all-reduce + the kmeans-centroid migration both use
    # NCCL (GPU); the CPU LPT rebuild syncs once per re-plan.
    if ctx.online_schedule:
        from bench_sp_all2all_attention import load_cost_model
        ctx.sched = {
            "num_heads": num_heads,
            "seq_len": seq_len,
            "strategy": strategy,
            "cost_model": load_cost_model(cost_model_path),
            "min_heads": min_heads_per_rank,
        }
        if ctx.world_size > 1:
            # dedicated NCCL group so the side-stream density all-reduce never
            # interleaves with the a2a collective on the default group.
            ctx.sched_group = dist.new_group()

    # 3b. Set up symmetric-memory buffers for the asymmetric a2a path.
    if ctx.a2a_backend == "asymm" and ctx.world_size > 1:
        from .context import sp_asymm_s_local
        head_dim = pipe.transformer.config.attention_head_dim
        # padded s_local (multiple of the TMA block); forward.py pads the seq to match.
        s_local = sp_asymm_s_local(seq_len, ctx.world_size, align=128)
        ctx.setup_asymm(b=1, h_total=num_heads, s_local=s_local, d=head_dim, dtype=pipe.dtype)

    # 4. Monkeypatch the DiT forward with the SP version.
    type(pipe.transformer).forward = wan_sp_dit_forward

    # 4b. Block-level timing hooks (metric D: whole transformer block =
    # self-attn + cross-attn + FFN + norms). Only active when timing enabled.
    def _pre_hook(layer_idx):
        def hook(_module, _args, _kwargs):
            ctx.profile_block_start(layer_idx)
            ctx.timing.block_start()
        return hook

    def _post_hook(layer_idx):
        def hook(_module, _args, _kwargs, _output):
            ctx.timing.block_end()
            ctx.profile_block_end(layer_idx)
        return hook

    for layer_idx, block in enumerate(pipe.transformer.blocks):
        block.register_forward_pre_hook(_pre_hook(layer_idx), with_kwargs=True)
        block.register_forward_hook(_post_hook(layer_idx), with_kwargs=True)

    set_sp_context(ctx)

    if ctx.rank == 0:
        print(
            f"[wan-sp] installed: W={ctx.world_size} strategy={strategy} "
            f"num_layers={num_layers} num_heads={num_heads} seq_len={seq_len} "
            f"a2a={ctx.a2a_backend}"
        )
        if strategy != "contiguous":
            print(summarize_plans(ctx.plans, ctx.world_size))
    return ctx

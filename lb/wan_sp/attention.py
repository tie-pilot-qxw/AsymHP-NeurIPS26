"""Sequence-parallel self-attention processor for Wan T2V.

Wraps the existing single-card SAP attention (``WanAttn_SAPAttn_Processor``)
in a load-balanced all2all so the model runs head-parallel across ranks:

    q,k,v : [1, H, S_local, D]          # this rank's seq shard, all heads (RoPE applied)
      -> all2all seq->heads             # [1, H_local, S_full, D]
      -> super().attention_core_logic   # SAP / full attention on full seq, local heads
      -> all2all heads->seq             # [1, H, S_local, D]

Two all2all backends:
  * symm  — NCCL all_to_all_single; needs uniform head count per rank, so heads
            are reordered to the plan's rank-slot layout (padded) and restored
            after. Best paired with the "greedy" (equal-count) plan.
  * asymm — symm-mem + Triton TMA pull/push (sm_90); moves only each rank's
            assigned heads (no padding), so it fits "greedy_unequal". The pull
            selects heads by index and the push writes global-head rows, so no
            reorder/restore is needed.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from svg.models.wan.attention import (
    WanAttn_SAPAttn_Processor,
    WanAttn_SpargeAttn_Processor,
)

from .context import get_sp_context


class WanAttn_SP_Processor(WanAttn_SAPAttn_Processor):
    """SP variant: only ``attention_core_logic`` changes vs. the SAP parent."""

    # Only SAP keeps a k-means centroid cache across steps, so only it needs the
    # centroid migration when a head moves ranks. SpargeAttn is stateless
    # per step (they recompute the block map from Q/K) -> no migration, and
    # touching their (nonexistent) q_centroids would crash. Subclasses override.
    _migrate_centroids = True

    def _profile_local_phase(self, name, fn):
        """Record non-synchronizing CUDA events for the critical-path breakdown."""
        return self._region(get_sp_context(), f"local_{name}", fn)

    def _record_sparse_call(self, is_sparse):
        ctx = get_sp_context()
        if ctx.timing.enabled:
            ctx.timing.sparse_flags.append(bool(is_sparse))

    def get_transpose_qkv(self, attn, query, key, value):
        """Asymm path: transpose q/k/v DIRECTLY into the symm buffers, replacing
        the parent's `.contiguous()` alloc. RoPE (next, in-place) then runs on
        the symm views, so `a2a_asymm_forward` needs no separate copy (saves one
        ~HBM-peak copy per call = ~1/5 of the asymm comm)."""
        ctx = get_sp_context()
        symm = getattr(ctx, "symm", None)
        if ctx.a2a_backend == "asymm" and ctx.world_size > 1 and symm is not None:
            s = ctx._asymm_s_local
            H = attn.heads
            symm.q_symm[:, :, :s, :].copy_(query.unflatten(2, (H, -1)).transpose(1, 2))
            symm.k_symm[:, :, :s, :].copy_(key.unflatten(2, (H, -1)).transpose(1, 2))
            symm.v_symm[:, :, :s, :].copy_(value.unflatten(2, (H, -1)).transpose(1, 2))
            return (symm.q_symm[:, :, :s, :], symm.k_symm[:, :, :s, :], symm.v_symm[:, :, :s, :])
        return super().get_transpose_qkv(attn, query, key, value)

    def attention_core_logic(self, query, key, value, timestep):
        ctx = get_sp_context()
        if ctx.world_size == 1:
            return super().attention_core_logic(query, key, value, timestep)
        if ctx.a2a_backend == "asymm":
            return self._asymm_core(ctx, query, key, value, timestep)
        return self._symm_core(ctx, query, key, value, timestep)

    def _region(self, ctx, name, fn):
        timing = ctx.timing
        nvtx = ctx.profile_nvtx and ctx._profile_capture_active
        if nvtx:
            n_heads = len(getattr(self, "_my_heads", ()))
            torch.cuda.nvtx.range_push(
                f"L{self.layer_idx:02d}/{name}/rank_heads={n_heads}"
            )
        if not timing.enabled:
            try:
                return fn()
            finally:
                if nvtx:
                    torch.cuda.nvtx.range_pop()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        try:
            start.record()
            out = fn()
            end.record()
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()
        timing.record(name, start, end)
        return out

    def _nvtx_region(self, ctx, name, fn):
        """Annotate CPU waits/enqueues without adding CUDA timing events."""
        nvtx = ctx.profile_nvtx and ctx._profile_capture_active
        if nvtx:
            torch.cuda.nvtx.range_push(f"L{self.layer_idx:02d}/{name}")
        try:
            return fn()
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()

    # ---- online kmeans-centroid migration (keeps the warm-start cache valid
    #      when the causal scheduler moves a head to a different rank) ----
    def _is_warmup(self, timestep) -> bool:
        return (self.layer_idx < self.first_layers_fp) or bool(timestep[0] > self.first_times_fp)

    def _online_prep_centroids(self, head_idx):
        """Inject THIS rank's current heads' previous-step centroids (sliced from
        the per-layer global store) so SAP's kmeans_step warm-starts correctly
        even if this head just migrated here. No store yet (step 0) -> leave the
        processor uninitialized so SAP does a fresh kmeans_init.

        The store is published asynchronously on a side stream at this layer's
        previous invocation (see _online_publish_centroids); consume it here by
        waiting on that event. The wait is off the sparse-attention critical path:
        the migration was launched ~a full transformer pass ago and overlapped
        compute, so the event is already signalled and .synchronize() returns
        immediately."""
        pend = getattr(self, "_pending_centroids", None)
        if pend is not None:
            qg, kg, ev = pend
            self._nvtx_region(
                get_sp_context(), "centroid_wait", lambda: ev.synchronize()
            )
            self._centroid_store = {"q": qg, "k": kg}
            self._pending_centroids = None
        store = getattr(self, "_centroid_store", None)
        if store is None:
            return
        self.q_centroids = store["q"].index_select(0, head_idx).contiguous()
        self.k_centroids = store["k"].index_select(0, head_idx).contiguous()
        self.centroids_init = True

    def _online_publish_centroids(self, ctx, head_idx):
        """Scatter this rank's freshly-updated centroids into a [num_heads,...]
        buffer at their global-head rows and all_reduce(SUM) (each head owned by
        exactly one rank) so every rank holds every head's centroids for the next
        step's warm-start.

        This is SAP-specific kmeans state (the paper's scheduler itself only
        communicates one density scalar per head). To keep it OFF the sparse-
        attention critical path -- the region the paper measures -- the scatter +
        NCCL all_reduce run on the scheduler SIDE stream and are consumed at the
        layer's NEXT invocation, overlapping a full transformer pass. Uses
        ctx.sched_group (the dedicated scheduler comm) on ctx._sched_stream, the
        same stream as the density all-reduce, so the two never issue concurrent
        collectives on the same communicator."""
        # Slice to the real head count: on the symmetric path the centroids have
        # max_hpr (padded) rows while my_heads holds only the real assigned heads.
        n_real = head_idx.numel()
        qc, kc = self.q_centroids[:n_real].detach(), self.k_centroids[:n_real].detach()
        H = ctx.sched["num_heads"]
        device = qc.device
        if ctx._sched_stream is None:
            ctx._sched_stream = torch.cuda.Stream(device)
        strm = ctx._sched_stream
        # centroids were just produced on the compute stream -> side stream waits
        strm.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(strm):
            qg = qc.new_zeros(H, qc.shape[-2], qc.shape[-1]); qg.index_copy_(0, head_idx, qc)
            kg = kc.new_zeros(H, kc.shape[-2], kc.shape[-1]); kg.index_copy_(0, head_idx, kc)
            if ctx.world_size > 1:
                dist.all_reduce(qg, group=ctx.sched_group)
                dist.all_reduce(kg, group=ctx.sched_group)
            ev = torch.cuda.Event()
            ev.record(strm)
        # keep the source centroids alive for the side-stream reads (allocator
        # must not reuse their memory until the migration kernels finish).
        qc.record_stream(strm)
        kc.record_stream(strm)
        self._pending_centroids = (qg, kg, ev)

    def _local_attn(self, q_h, k_h, v_h, timestep):
        """Run the local sparse attention on the REAL sequence only. The asymm
        path pads the sequence to a 128-aligned s_local; those trailing zero
        tokens must NOT enter attention (zero keys still shift the softmax
        denominator, and they perturb SAP's k-means / block selection). So trim
        q/k/v to the real video length, attend, then zero-pad the output back to
        the padded length for the reverse a2a."""
        real_len = int(self.context_length) + int(self.num_frame) * int(self.frame_size)
        s = q_h.shape[2]
        _super = super(WanAttn_SP_Processor, self)
        if real_len >= s:
            return _super.attention_core_logic(q_h, k_h, v_h, timestep)
        o = _super.attention_core_logic(
            q_h[:, :, :real_len, :].contiguous(),
            k_h[:, :, :real_len, :].contiguous(),
            v_h[:, :, :real_len, :].contiguous(),
            timestep,
        )
        out = o.new_zeros(o.shape[0], o.shape[1], s, o.shape[3])
        out[:, :, :real_len, :] = o
        return out

    # ---- symmetric NCCL path (reorder heads -> a2a -> attn -> a2a -> restore) ----
    def _symm_core(self, ctx, query, key, value, timestep):
        device = query.device
        plan = self._nvtx_region(
            ctx, "plan_wait_build", lambda: ctx.plan_for_layer(self.layer_idx)
        )
        head_idx = plan.h_idxs_tensor(ctx.rank, device, dtype=torch.long)
        head_order = plan.head_order_tensor(device)
        restore = plan.restore_tensor(device)

        def reorder():
            return (
                query.index_select(1, head_order).contiguous(),
                key.index_select(1, head_order).contiguous(),
                value.index_select(1, head_order).contiguous(),
            )

        my_heads = plan.assigned[ctx.rank]
        # expose global head ids so SAP's kmeans_init seeds per-head (placement-
        # invariant, deterministic output regardless of which rank owns a head).
        self._my_heads = my_heads
        self._num_heads_global = query.shape[1]
        online_c = ctx.online_schedule and not self._is_warmup(timestep)
        self._want_density = online_c  # tell the local method (Sparge) to expose density
        q_r, k_r, v_r = self._region(ctx, "reorder", reorder)
        q_h, k_h, v_h = self._region(ctx, "a2a_in", lambda: (
            ctx.a2a_seq_to_heads(q_r), ctx.a2a_seq_to_heads(k_r), ctx.a2a_seq_to_heads(v_r)))
        if online_c and self._migrate_centroids:
            self._online_prep_centroids(head_idx)
        # The symmetric a2a delivers max_hpr (uniform) head rows: the first
        # n_real are this rank's real heads, the rest are DUPLICATE padding
        # (pad_rank_heads_uniform). Run attention on the real heads ONLY, then
        # zero-pad the head dim back to max_hpr for the reverse a2a. This (a)
        # makes num_heads==len(_my_heads) so the global-head kmeans seeding and
        # the online centroid cache line up (no stale-cache re-init, no seed
        # IndexError on ranks with n_real<max_hpr under greedy_unequal), and (b)
        # avoids wasting compute on the duplicated rows (their output is discarded
        # by `restore` anyway). With `greedy` (equal counts) n_real==max_hpr and
        # this is a no-op.
        n_real = len(my_heads)
        max_hpr = q_h.shape[1]
        def _attn_real():
            if n_real < max_hpr:
                o = self._local_attn(
                    q_h[:, :n_real].contiguous(),
                    k_h[:, :n_real].contiguous(),
                    v_h[:, :n_real].contiguous(),
                    timestep)
                out = o.new_zeros(o.shape[0], max_hpr, o.shape[2], o.shape[3])
                out[:, :n_real] = o
                return out
            return self._local_attn(q_h, k_h, v_h, timestep)
        out_h = self._region(ctx, "attn", _attn_real)
        if online_c and self._migrate_centroids:
            self._region(ctx, "sched_comm", lambda: self._online_publish_centroids(ctx, head_idx))
        out_r = self._region(ctx, "a2a_out", lambda: ctx.a2a_heads_to_seq(out_h))
        out = out_r.index_select(1, restore).contiguous()
        if online_c:
            # causal one-INVOCATION lag: re-plan this layer from the density it
            # just realized, consumed the next time this layer runs. NB the
            # transformer is called once per CFG branch (twice per denoising
            # step), so consecutive consumers alternate cond/uncond -- valid
            # (per-head density is ~identical across branches/steps, r~0.999; the
            # scheduler simply uses the most recent available density). Gated on
            # online_c (NOT online_schedule) so the dense warm-up steps -- which
            # have no per-head density signal -- do NOT rebuild the static
            # bootstrap plan into an interleaved ownership map before the first
            # sparse invocation.
            self._nvtx_region(
                ctx,
                "density_publish",
                lambda: ctx.online_replan(
                    self.layer_idx,
                    getattr(self, "_last_density", None),
                    head_idx,
                ),
            )
        ctx.timing.n_calls += 1
        return out

    # ---- asymmetric TMA path (pull assigned heads -> attn -> push back) ----
    def _asymm_core(self, ctx, query, key, value, timestep):
        device = query.device
        plan = self._nvtx_region(
            ctx, "plan_wait_build", lambda: ctx.plan_for_layer(self.layer_idx)
        )
        h_idxs_r = plan.h_idxs_tensor(ctx.rank, device)
        head_idx = plan.h_idxs_tensor(ctx.rank, device, dtype=torch.long)
        my_heads = plan.assigned[ctx.rank]
        self._my_heads = my_heads
        self._num_heads_global = query.shape[1]
        online_c = ctx.online_schedule and not self._is_warmup(timestep)
        self._want_density = online_c  # tell the local method (Sparge) to expose density

        q_h, k_h, v_h = self._region(
            ctx, "a2a_in", lambda: ctx.a2a_asymm_forward(query, key, value, h_idxs_r))
        if online_c and self._migrate_centroids:
            self._online_prep_centroids(head_idx)
        out_h = self._region(
            ctx, "attn",
            lambda: self._local_attn(q_h, k_h, v_h, timestep))
        if online_c and self._migrate_centroids:
            self._region(ctx, "sched_comm", lambda: self._online_publish_centroids(ctx, head_idx))
        out = self._region(ctx, "a2a_out", lambda: ctx.a2a_asymm_reverse(out_h, h_idxs_r))
        if online_c:
            # causal one-INVOCATION lag: re-plan this layer from the density it
            # just realized, consumed the next time this layer runs. NB the
            # transformer is called once per CFG branch (twice per denoising
            # step), so consecutive consumers alternate cond/uncond -- valid
            # (per-head density is ~identical across branches/steps, r~0.999; the
            # scheduler simply uses the most recent available density). Gated on
            # online_c (NOT online_schedule) so the dense warm-up steps -- which
            # have no per-head density signal -- do NOT rebuild the static
            # bootstrap plan into an interleaved ownership map before the first
            # sparse invocation.
            self._nvtx_region(
                ctx,
                "density_publish",
                lambda: ctx.online_replan(
                    self.layer_idx,
                    getattr(self, "_last_density", None),
                    head_idx,
                ),
            )
        ctx.timing.n_calls += 1
        return out

class WanAttn_SP_Sparge_Processor(WanAttn_SP_Processor, WanAttn_SpargeAttn_Processor):
    """SP variant whose LOCAL sparse attention is the REAL SpargeAttn kernel.
    Same MRO trick: the SP a2a/placement plumbing is inherited
    verbatim and only the local method changes. Requires ``spas_sage_attn``."""

    _migrate_centroids = False  # SpargeAttn is stateless per step: no centroid cache

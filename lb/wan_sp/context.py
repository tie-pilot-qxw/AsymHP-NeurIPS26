"""Global sequence-parallel (SP) runtime context for Wan T2V.

Holds the torchrun/NCCL state, the per-layer head-placement plans, the
all2all backend (symmetric NCCL or asymmetric symmetric-memory pull/push), and
timing accumulators used to report the load-balancing effect end-to-end.

This is the E2E sibling of ``lb/bench_sp_all2all_attention.py``: the bench
replays a single (layer, step) dump; here the same primitives run inside the
real Wan denoising loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


@dataclass
class LayerPlan:
    """Fixed head->rank placement for one transformer layer.

    ``head_order`` is a length ``max_hpr * world_size`` list laid out by rank
    slot: slots ``[r*max_hpr : (r+1)*max_hpr]`` are the (padded) global heads
    that rank ``r`` computes after the seq->heads all2all. Padding slots
    duplicate a rank's first real head so the symmetric all2all (uniform head
    count) works; ``real_heads_per_rank[r]`` marks how many leading slots are
    real. ``restore_index[gh]`` maps original global head ``gh`` back to its
    slot in the heads->seq all2all output.
    """

    head_order: List[int]
    real_heads_per_rank: List[int]
    max_hpr: int
    restore_index: List[int]
    # per-rank real (unpadded) global head lists; assigned[r] = rank r's heads.
    # Used by the asymmetric a2a path (pull selects these heads directly).
    assigned: Optional[List[List[int]]] = None
    strategy: str = "contiguous"
    # rank-0 diagnostics (predicted per-rank cost etc.), optional
    info: Optional[dict] = None

    # Cached plan tensors. Head indices are staged through pinned host memory:
    # constructing a CUDA tensor directly from a temporary Python list performs
    # a pageable H2D copy followed by cudaStreamSynchronize. This is especially
    # harmful on the scheduler side stream after it has waited on attention.
    _head_order_t: Optional[torch.Tensor] = None
    _restore_t: Optional[torch.Tensor] = None
    _head_order_host: Optional[torch.Tensor] = None
    _restore_host: Optional[torch.Tensor] = None
    _h_idxs_t: dict = field(default_factory=dict)  # rank -> int32 head-idx tensor
    _h_idxs_long_t: dict = field(default_factory=dict)
    _h_idxs_host: dict = field(default_factory=dict)

    def prepare_h_idxs(self, rank: int, device) -> None:
        key = (rank, str(device))
        if key in self._h_idxs_t:
            return
        values = self.assigned[rank]
        device = torch.device(device)
        if device.type == "cuda":
            # Keep the pinned source alive with the plan until the non-blocking
            # H2D completes. Build int64 on device from the int32 copy so both
            # consumers share one host transfer.
            host = torch.tensor(values, dtype=torch.int32, pin_memory=True)
            idx = torch.empty(len(values), dtype=torch.int32, device=device)
            idx.copy_(host, non_blocking=True)
            self._h_idxs_host[key] = host
        else:
            idx = torch.tensor(values, dtype=torch.int32, device=device)
        self._h_idxs_t[key] = idx
        self._h_idxs_long_t[key] = idx.to(dtype=torch.long)

    def h_idxs_tensor(self, rank: int, device, *, dtype=torch.int32) -> torch.Tensor:
        key = (rank, str(device))
        self.prepare_h_idxs(rank, device)
        if dtype == torch.int32:
            return self._h_idxs_t[key]
        if dtype == torch.long:
            return self._h_idxs_long_t[key]
        raise ValueError(f"unsupported head-index dtype: {dtype}")

    def head_order_tensor(self, device) -> torch.Tensor:
        device = torch.device(device)
        if self._head_order_t is None or self._head_order_t.device != device:
            if device.type == "cuda":
                host = torch.tensor(self.head_order, dtype=torch.long, pin_memory=True)
                value = torch.empty(len(self.head_order), dtype=torch.long, device=device)
                value.copy_(host, non_blocking=True)
                self._head_order_host = host
                self._head_order_t = value
            else:
                self._head_order_t = torch.tensor(
                    self.head_order, dtype=torch.long, device=device
                )
        return self._head_order_t

    def restore_tensor(self, device) -> torch.Tensor:
        device = torch.device(device)
        if self._restore_t is None or self._restore_t.device != device:
            if device.type == "cuda":
                host = torch.tensor(self.restore_index, dtype=torch.long, pin_memory=True)
                value = torch.empty(len(self.restore_index), dtype=torch.long, device=device)
                value.copy_(host, non_blocking=True)
                self._restore_host = host
                self._restore_t = value
            else:
                self._restore_t = torch.tensor(
                    self.restore_index, dtype=torch.long, device=device
                )
        return self._restore_t


@dataclass
class SPTiming:
    """Per-rank timing over the whole run.

    To avoid distorting the E2E wall-clock, region timings are recorded as CUDA
    event pairs during the run and reduced once (after a single sync) via
    ``reduce()``. Enable with ``enabled=True``.
    """

    enabled: bool = False
    events: Dict[str, list] = field(default_factory=lambda: {
        "reorder": [], "a2a_in": [], "attn": [], "a2a_out": [], "block": [],
        "sched_comm": [],  # online centroid-migration all_reduce (GPU)
    })
    # One flag per local-attention invocation, aligned with ``attn`` events.
    # Phase-specific event arrays contain entries only for sparse invocations.
    sparse_flags: List[bool] = field(default_factory=list)
    n_calls: int = 0
    _pending_block: object = None
    # Host wall-time of the online causal scheduler (density gather + re-plan),
    # measured with perf_counter since it is host/collective work, not a GPU
    # stream op. This is the true online scheduling overhead.
    sched_ms: float = 0.0
    sched_calls: int = 0

    def record_sched(self, seconds: float):
        if self.enabled:
            self.sched_ms += seconds * 1e3
            self.sched_calls += 1

    def record(self, region: str, start, end):
        if self.enabled:
            self.events.setdefault(region, []).append((start, end))

    def block_start(self):
        """Mark the start of one transformer block (metric D). Blocks run
        sequentially so a simple pending-start pairs with the next block_end."""
        if not self.enabled:
            return
        import torch as _t
        s = _t.cuda.Event(enable_timing=True); s.record()
        self._pending_block = s

    def block_end(self):
        if not self.enabled or self._pending_block is None:
            return
        import torch as _t
        e = _t.cuda.Event(enable_timing=True); e.record()
        self.events["block"].append((self._pending_block, e))
        self._pending_block = None

    def reduce(self) -> Dict[str, float]:
        """Sum elapsed ms per region. Call after torch.cuda.synchronize()."""
        out = {}
        for region, pairs in self.events.items():
            out[f"{region}_ms"] = float(sum(s.elapsed_time(e) for s, e in pairs))
        out["n_calls"] = self.n_calls
        out["schedule_ms"] = self.sched_ms
        out["schedule_calls"] = self.sched_calls
        return out

    def attn_per_call(self) -> list:
        """Per-attention-call elapsed ms in execution order (one entry per
        layer per forward call). Identical length/ordering across ranks, so
        the elementwise max across ranks summed = the TRUE attention makespan
        (the per-layer critical path), which per-rank *sums* wrongly average
        out. Call after torch.cuda.synchronize()."""
        return [s.elapsed_time(e) for s, e in self.events["attn"]]

    def per_call(self, region: str) -> list:
        """Per-call elapsed ms for any region ('reorder','a2a_in','attn',
        'a2a_out'), execution order. Call after torch.cuda.synchronize()."""
        return [s.elapsed_time(e) for s, e in self.events.get(region, [])]


@dataclass
class SPContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    group: Optional[object] = None  # dist ProcessGroup; None == default WORLD
    a2a_backend: str = "symm"  # "symm" | "asymm"
    plans: Dict[int, LayerPlan] = field(default_factory=dict)
    timing: SPTiming = field(default_factory=SPTiming)
    # asymm-a2a state (set up by setup_asymm)
    symm: Optional[object] = None
    num_sms: Optional[int] = None
    # online causal scheduler (set up by install_wan_sp when online=True)
    online_schedule: bool = False
    sched: Optional[dict] = None              # {num_heads, seq_len, strategy, cost_model, min_heads}
    sched_group: Optional[object] = None      # dedicated NCCL group for the density all-reduce
    _sched_stream: Optional[object] = None    # side CUDA stream (density gather off the compute path)
    _pinned_pool: Dict[int, object] = field(default_factory=dict)   # layer -> pinned host buffer
    _pending_dens: Dict[int, object] = field(default_factory=dict)  # layer -> (pinned, cuda event)
    # Opt-in Nsight Systems capture state. A "pass" is one complete transformer
    # invocation across every layer, not one attention layer.
    profile_nvtx: bool = False
    profile_cuda_api: bool = False
    profile_pass_start: int = 0
    profile_pass_count: int = 1
    _profile_pass_idx: int = 0
    _profile_capture_active: bool = False

    def profile_block_start(self, layer_idx: int):
        """Mark a complete transformer pass and every block for Nsight Systems."""
        if not self.profile_nvtx or not self.timing.enabled:
            return
        if layer_idx == 0:
            idx = self._profile_pass_idx
            stop = self.profile_pass_start + self.profile_pass_count
            if idx == self.profile_pass_start:
                self._profile_capture_active = True
                if self.profile_cuda_api and self.rank == 0:
                    torch.cuda.cudart().cudaProfilerStart()
                torch.cuda.nvtx.range_push("asymhp_full_transformer_pass")
            elif not self.profile_pass_start < idx < stop:
                self._profile_capture_active = False
        if self._profile_capture_active:
            torch.cuda.nvtx.range_push(f"transformer_block/L{layer_idx:02d}")

    def profile_block_end(self, layer_idx: int):
        if not self.profile_nvtx or not self.timing.enabled:
            return
        if self._profile_capture_active:
            torch.cuda.nvtx.range_pop()  # transformer_block/Lxx
        if layer_idx == len(self.plans) - 1:
            stop = self.profile_pass_start + self.profile_pass_count
            if self._profile_capture_active and self._profile_pass_idx + 1 >= stop:
                torch.cuda.nvtx.range_pop()  # asymhp_full_transformer_pass
                if self.profile_cuda_api and self.rank == 0:
                    torch.cuda.cudart().cudaProfilerStop()
                self._profile_capture_active = False
            self._profile_pass_idx += 1

    def plan_for_layer(self, layer_idx: int) -> LayerPlan:
        # Consume a plan queued by the online scheduler at this layer's PREVIOUS
        # invocation (one transformer call ago -- i.e. the other CFG branch, or
        # the previous denoising step). The density all-reduce + async D2H copy
        # were launched ~num_layers layers ago on a side stream, so by now the
        # event is signalled and the pinned host buffer already holds the data
        # -> ev.synchronize() returns immediately
        # (the schedule was computed off the critical path; only the tiny CPU LPT
        # build is on-path here).
        pend = self._pending_dens.pop(layer_idx, None)
        if pend is not None:
            import time as _time
            from .planning import build_one_plan
            pinned, ev = pend
            t0 = _time.perf_counter()
            ev.synchronize()
            s = self.sched
            self.plans[layer_idx] = build_one_plan(
                pinned.tolist(), s["num_heads"], self.world_size, s["strategy"],
                s["seq_len"], s["cost_model"], s["min_heads"])
            self.timing.record_sched(_time.perf_counter() - t0)
        plan = self.plans[layer_idx]
        # Online re-planning replaces LayerPlan every invocation. Materialize
        # its indices once, on the compute stream, through pinned memory. All
        # pull/push and scheduler-state paths reuse these device tensors.
        plan.prepare_h_idxs(self.rank, self.device)
        return plan

    def online_replan(self, layer_idx: int, local_density, h_idxs: torch.Tensor):
        """Causal one-step-lag scheduler: given THIS step's realized per-head
        density for the heads this rank just computed (``local_density`` over the
        global head ids ``h_idxs``), gather the full per-head density on the host
        and rebuild ``plans[layer_idx]`` for the NEXT time this layer runs.

        Tiny (num_heads floats): each rank scatters its owned heads into a fixed
        [num_heads] device vector on a side stream, then a dedicated-group NCCL
        all_reduce(SUM) merges them (each head owned by exactly one rank -> no
        double count) and the result is async-copied to a pinned host buffer,
        consumed lazily at this layer's next invocation in ``plan_for_layer``
        (off the critical path).
        """
        if not self.online_schedule or self.sched is None:
            return
        nh = self.sched["num_heads"]
        if self._sched_stream is None:
            self._sched_stream = torch.cuda.Stream(self.device)
        strm = self._sched_stream
        # The side stream must see the density that was produced on the compute
        # stream (during the local attention), so make it wait for that work.
        strm.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(strm):
            phase_start = torch.cuda.Event(enable_timing=True) if self.timing.enabled else None
            phase_end = torch.cuda.Event(enable_timing=True) if self.timing.enabled else None
            if phase_start is not None:
                phase_start.record(strm)
            dens = torch.zeros(nh, dtype=torch.float32, device=self.device)
            if h_idxs.device != self.device or h_idxs.dtype != torch.long:
                raise ValueError(
                    "online_replan requires the plan-owned int64 device head indices"
                )
            idx = h_idxs
            if local_density is not None:
                # Defensive: keep exactly one density row per assigned head.
                ld = local_density.detach().to(torch.float32).reshape(-1)[: idx.numel()]
                dens.index_copy_(0, idx, ld)
            else:
                dens.index_fill_(0, idx, 1.0)  # (only reachable off the critical path; kept defensive)
            if self.world_size > 1:
                # dedicated NCCL group so this tiny all-reduce never conflicts with
                # the a2a collective on ctx.group.
                dist.all_reduce(dens, op=dist.ReduceOp.SUM, group=self.sched_group)
            pinned = self._pinned_pool.get(layer_idx)
            if pinned is None:
                pinned = torch.empty(nh, dtype=torch.float32, pin_memory=True)
                self._pinned_pool[layer_idx] = pinned
            pinned.copy_(dens, non_blocking=True)   # async D2H to pinned host buffer
            ev = torch.cuda.Event()
            ev.record(strm)
            if phase_end is not None:
                phase_end.record(strm)
                self.timing.record("density_exchange", phase_start, phase_end)
        # Queue it; consumed lazily in plan_for_layer at this layer's next
        # invocation (off critical path).
        # The compute stream was NEVER synchronized here.
        self._pending_dens[layer_idx] = (pinned, ev)

    # ---- symmetric all2all (NCCL all_to_all_single) ----
    def a2a_seq_to_heads(self, x_seq: torch.Tensor) -> torch.Tensor:
        """[B, H_padded, S_local, D] -> [B, H_local, W*S_local, D].

        H_padded == max_hpr * world_size; H_local == max_hpr.
        """
        W = self.world_size
        if W == 1:
            return x_seq.contiguous()
        b, h, s_local, d = x_seq.shape
        assert h % W == 0, f"heads {h} not divisible by world {W}"
        hpr = h // W
        send = x_seq.view(b, W, hpr, s_local, d).permute(1, 0, 2, 3, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        return recv.permute(1, 2, 0, 3, 4).reshape(b, hpr, W * s_local, d).contiguous()

    def a2a_heads_to_seq(self, x_head: torch.Tensor) -> torch.Tensor:
        """[B, H_local, W*S_local, D] -> [B, H_padded, S_local, D]."""
        W = self.world_size
        if W == 1:
            return x_head.contiguous()
        b, hpr, seq, d = x_head.shape
        assert seq % W == 0, f"seq {seq} not divisible by world {W}"
        s_local = seq // W
        send = x_head.view(b, hpr, W, s_local, d).permute(2, 0, 1, 3, 4).contiguous()
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        return recv.permute(1, 0, 2, 3, 4).reshape(b, W * hpr, s_local, d).contiguous()

    # ---- asymmetric all2all (symm-mem + Triton TMA pull/push; sm_90) ----
    def setup_asymm(self, b: int, h_total: int, s_local: int, d: int, dtype: torch.dtype):
        """Allocate symmetric-memory buffers and the pull/push handles once.

        Requires ``s_local`` be a multiple of the TMA block (pick_s_block); we
        assert this so the pull/push moves no meaningless padding into the
        attention (keeps the E2E path exact without per-peer trimming).
        """
        import sys as _sys
        from pathlib import Path as _Path
        lb_dir = _Path(__file__).resolve().parents[1]
        if str(lb_dir) not in _sys.path:
            _sys.path.insert(0, str(lb_dir))
        from bench_sp_all2all_attention import pick_s_block  # noqa
        from symm_a2a import SymmAsymA2A  # noqa

        # BW-sweep-tuned: 256 tile when divisible (else 128), 2x SM over-subscribe.
        s_block = 256 if s_local % 256 == 0 else pick_s_block(s_local)
        if s_local % s_block != 0:
            raise ValueError(
                f"asymm a2a needs s_local ({s_local}) divisible by TMA block "
                f"({s_block}); choose num_frames so seq_len/world is 128-aligned "
                f"(e.g. num_frames=125 -> s_local=8320 at W=6)."
            )
        self.symm = SymmAsymA2A(
            self.group if self.group is not None else dist.group.WORLD,
            buffer_shape=(b, h_total, s_local, d),
            dtype=dtype, device=self.device, s_block=s_block, enable_reverse=True,
        )
        self.num_sms = 2 * torch.cuda.get_device_properties(self.device).multi_processor_count
        self._asymm_s_local = s_local

    def _atimed(self, region, fn):
        """Fine-grained asymm-path timing (copy/abarrier/pull/push breakdown)."""
        t = self.timing
        nvtx = self.profile_nvtx and self._profile_capture_active
        if nvtx:
            torch.cuda.nvtx.range_push(region)
        if not t.enabled:
            try:
                return fn()
            finally:
                if nvtx:
                    torch.cuda.nvtx.range_pop()
        import torch as _t
        s = _t.cuda.Event(enable_timing=True); e = _t.cuda.Event(enable_timing=True)
        try:
            s.record(); out = fn(); e.record()
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()
        t.events.setdefault(region, []).append((s, e))
        return out

    def a2a_asymm_forward(self, q_seq, k_seq, v_seq, h_idxs_r):
        """Populate this rank's symm buffers and pull its assigned heads.

        q/k/v_seq: [B, H_total, S_local, D]. Returns three
        [B, H_local_r, W*S_local, D] tensors (H_local_r == h_idxs_r.numel()).
        """
        symm = self.symm
        s = self._asymm_s_local

        # Skip the copy if q/k/v are ALREADY the symm buffers (the SP processor's
        # get_transpose_qkv wrote the transpose straight into symm). Detect via
        # storage pointer.
        already_in_symm = (q_seq.data_ptr() == symm.q_symm.data_ptr())

        def _copy():
            if already_in_symm:
                return
            symm.q_symm[:, :, :s, :].copy_(q_seq)
            symm.k_symm[:, :, :s, :].copy_(k_seq)
            symm.v_symm[:, :, :s, :].copy_(v_seq)
        self._atimed("copy", _copy)
        self._atimed("pre_barrier", lambda: symm.barrier())

        def _pull():  # 3 pull kernels ONLY (no post-pull barrier -- see below)
            q_h = symm.pull_seq_to_heads("q", h_idxs_r, self.num_sms, pre_barrier=False, post_barrier=False)
            k_h = symm.pull_seq_to_heads("k", h_idxs_r, self.num_sms, pre_barrier=False, post_barrier=False)
            v_h = symm.pull_seq_to_heads("v", h_idxs_r, self.num_sms, pre_barrier=False, post_barrier=False)
            return q_h, k_h, v_h
        out = self._atimed("pull_kernel", _pull)
        # NO post-pull barrier: the paper's protocol uses exactly two syncs
        # (before pull, after push). A post-pull barrier would be redundant --
        # pulls read the Q/K/V symm buffers, which are only reused (overwritten)
        # at the NEXT sparse layer's write. That write cannot happen until this
        # layer's post-push barrier, which every rank reaches only after its own
        # pull->attn->push; the collective therefore guarantees all pulls are
        # complete before any rank overwrites Q/K/V. Local pull->attn ordering is
        # handled by the compute stream. The push output uses a SEPARATE recv_symm
        # buffer, so pull and push never alias.
        return out

    def a2a_asymm_reverse(self, out_h, h_idxs_r):
        """Push this rank's attn output back to seq layout.

        out_h: [B, H_local_r, W*S_local, D] -> returns [B, H_total, S_local, D]
        already in original global-head order (push writes global rows).
        """
        src = out_h.contiguous()
        recv = self._atimed("push_kernel", lambda: self.symm.push_heads_to_seq(
            src, h_idxs_r, self.num_sms, pre_barrier=False, post_barrier=False))
        self._atimed("push_barrier", lambda: self.symm.barrier())
        return recv[:, :, :self._asymm_s_local, :]


def sp_asymm_s_local(seq_len: int, world: int, align: int = 128) -> int:
    """Padded per-rank sequence length for the asymm TMA path.

    The pull/push kernels tile the S dim by a power-of-2 block (128 or 256), so
    s_local must be a multiple of ``align``. When seq_len/world isn't, we pad
    the sequence upstream (forward.py); the pad is trimmed before local
    attention and dropped after the gather, so the result is preserved.
    """
    # Round UP seq_len over (world*align): s_local must satisfy
    # s_local % align == 0 AND s_local*world >= seq_len. Doing `seq_len // world`
    # first truncates the remainder and can give s_local*world < seq_len (a
    # negative pad_len) when the quotient is already align-aligned but seq_len is
    # not divisible by world (e.g. seq_len=14850, world=4).
    return ((seq_len + world * align - 1) // (world * align)) * align


_CTX: Optional[SPContext] = None


def set_sp_context(ctx: SPContext):
    global _CTX
    _CTX = ctx


def get_sp_context() -> SPContext:
    if _CTX is None:
        raise RuntimeError("SP context not initialized; call install_wan_sp() first")
    return _CTX


def sp_enabled() -> bool:
    return _CTX is not None and _CTX.world_size > 1

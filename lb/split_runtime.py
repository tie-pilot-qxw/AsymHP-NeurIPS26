"""Runtime for executing a SplitPlan on real GPUs (NCCL p2p).

Two-step compute on each GPU:

  Step 1 (local, overlaps with comm)
    - Every rank: mask + attn for its FULL heads (existing path)
    - split_owner: mask for the split head -> q_perm/k_perm/v_perm/dyn_map
                   launch async broadcast of post-perm tensors to helpers
                   compute own Q-range attn share
    - helper: launch async recv, in parallel finish its full-head work

  Step 2 (remote)
    - helper: wait for recv, run sparse attn on received slice, async-send
              attn_out_slice back to owner
    - split_owner: recv helper attn_out_slice(s), concat into full attn_out

  Step 3 (existing path, unchanged)
    - inverse_permute + push_heads_to_seq

Helpers do NOT push the split head directly — they ship the result back to
the owner, who concatenates and walks the existing full-head push path. No
push-kernel changes are required.

Sub-group broadcast handles owner -> helpers fan-out for K/V/Q/dyn_map.
Helper -> owner attn_out_slice goes via dist.isend/irecv.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from split_planner import ComputeUnit, SplitPlan


@dataclass
class SplitInfo:
    """Per-split runtime metadata derived from a SplitPlan."""
    split_id: int
    global_head: int           # which logical head is split
    owner: int                 # rank running mask + holding K/V/Q
    helpers: List[int]         # ascending; rank ids that compute Q-sub-ranges
    q_ranges: Dict[int, Tuple[int, int]]  # rank (incl. owner) -> (q_lo, q_hi)

    def participants(self) -> List[int]:
        return sorted([self.owner] + self.helpers)


def extract_splits(plan: SplitPlan) -> List[SplitInfo]:
    """Walk plan.units_per_rank, group helper/split_owner units by split_id."""
    by_id: Dict[int, Dict] = {}
    for rank, units in enumerate(plan.units_per_rank):
        for u in units:
            if u.split_id < 0:
                continue
            sid = u.split_id
            if sid not in by_id:
                by_id[sid] = {
                    "global_head": u.global_head,
                    "owner": u.owner_rank,
                    "q_ranges": {},
                }
            assert by_id[sid]["global_head"] == u.global_head, "split_id collision"
            by_id[sid]["q_ranges"][rank] = (u.q_lo, u.q_hi)
    out: List[SplitInfo] = []
    for sid in sorted(by_id):
        info = by_id[sid]
        helpers = sorted(r for r in info["q_ranges"] if r != info["owner"])
        out.append(SplitInfo(
            split_id=sid,
            global_head=info["global_head"],
            owner=info["owner"],
            helpers=helpers,
            q_ranges=info["q_ranges"],
        ))
    return out


def make_split_groups(
    splits: List[SplitInfo], world_size: int
) -> Dict[int, "dist.ProcessGroup"]:
    """Create a sub-group per split. dist.new_group is collective — every
    rank in the world (member or not) must call it identically.

    Returns {split_id: group}. When the split's participants cover the
    entire world we reuse `dist.group.WORLD` instead of creating a new
    sub-communicator (avoids NCCL multi-communicator interference for
    small worlds where sub == world).
    """
    groups: Dict[int, "dist.ProcessGroup"] = {}
    for s in splits:
        ranks = s.participants()
        if len(ranks) == world_size:
            groups[s.split_id] = dist.group.WORLD
        else:
            # Even non-members must call new_group; group is None for them.
            groups[s.split_id] = dist.new_group(ranks=ranks)
    return groups


@dataclass
class _Step1Buffers:
    """Per-split buffers populated during Step 1 broadcast.

    Both owner and helper(s) inflate these to the same shape; on owner the
    tensors are the real post-perm outputs, on helper they're empty buffers
    that get filled by the broadcast.

    `cluster_boundaries` is an int32 tensor of shape [N+1] where N is the
    number of split participants ordered owner-first then helpers-by-rank.
    Element i is the cluster index in qc_sz where participant i's range
    starts; element N == qc_num. Snap is computed by the owner from the
    runtime qc_sz (the planner's q_lo/q_hi are in row space and may not
    align with cluster boundaries — flashinfer requires sum(qc_sz)==S so
    we must slice on cluster boundaries).
    """
    k_perm: torch.Tensor          # [B, 1, S, D]
    v_perm: torch.Tensor          # [B, 1, S, D]
    q_perm: torch.Tensor          # [B, 1, S, D]
    dyn_map: torch.Tensor         # [B, 1, num_q_centroids, num_k_centroids]
    qc_sz: torch.Tensor           # [B, 1, num_q_centroids]
    kc_sz: torch.Tensor           # [B, 1, num_k_centroids]
    cluster_boundaries: torch.Tensor   # [N+1] int32 — snapped cluster indices


def snap_to_cluster_boundaries(
    qc_sz_one_head: torch.Tensor, planned_boundaries: List[int]
) -> Tuple[List[int], List[int]]:
    """Snap planner row-space boundaries to cluster boundaries.

    `qc_sz_one_head`: 1-D int tensor [qc_num] of cluster sizes (sum=S).
    `planned_boundaries`: [b_0=0, b_1, ..., b_N=S] from the planner.

    Returns (snapped_q_rows, snapped_cluster_indices), both length N+1.
    snapped_cluster_indices[0]=0, [-1]=qc_num. Each interior c_i picks the
    cluster boundary closest to b_i, monotone increasing, leaving room for
    at least one cluster per remaining segment.
    """
    qc_num = int(qc_sz_one_head.numel())
    N = len(planned_boundaries) - 1   # number of segments == participants
    if N < 1:
        raise ValueError("need at least one segment")
    if N > qc_num:
        raise ValueError(
            f"can't split {qc_num} clusters across {N} participants — "
            "increase q_granularity or reduce helpers"
        )
    cumsum = [0]
    sizes = qc_sz_one_head.detach().cpu().tolist()
    for s in sizes:
        cumsum.append(cumsum[-1] + int(s))
    # cumsum[c] gives the row index where cluster c starts
    snapped_clusters = [0]
    for i in range(1, N):
        b = int(planned_boundaries[i])
        c_lo = snapped_clusters[-1] + 1
        # Leave at least one cluster for each segment after this boundary.
        c_hi = qc_num - (N - i)
        # Find c in [c_lo, c_hi] minimizing |cumsum[c] - b|.
        best_c = c_lo
        best_diff = abs(cumsum[c_lo] - b)
        for c in range(c_lo + 1, c_hi + 1):
            d = abs(cumsum[c] - b)
            if d < best_diff:
                best_c = c
                best_diff = d
        snapped_clusters.append(best_c)
    snapped_clusters.append(qc_num)
    snapped_q_rows = [cumsum[c] for c in snapped_clusters]
    return snapped_q_rows, snapped_clusters


def slice_split_head_inputs(
    q_perm: torch.Tensor, dyn_map: torch.Tensor,
    qc_sz: torch.Tensor, kc_sz: torch.Tensor,
    q_lo: int, q_hi: int, c_lo: int, c_hi: int,
):
    """Slice the (single-head) split head's kernel inputs to a Q sub-range.

    All inputs have head dim already == 1 (caller has selected the split
    head). Returns kernel inputs (q_sub, dyn_map_sub, qc_sub, kc_sz).
    """
    q_sub = q_perm[:, :, q_lo:q_hi, :].contiguous()
    dyn_map_sub = dyn_map[:, :, c_lo:c_hi, :].contiguous()
    qc_sub = qc_sz[:, :, c_lo:c_hi].contiguous()
    return q_sub, dyn_map_sub, qc_sub, kc_sz


def variable_qkv_block_sparse_fwd_flashinfer(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    block_mask_map: torch.Tensor, block_row_sz: torch.Tensor,
    block_col_sz: torch.Tensor,
):
    """Variant of `dynamic_block_sparse_fwd_flashinfer` that supports
    q_len != kv_len (needed for split-head Q-row sub-ranges over full kv).

    Same mask/cluster-size semantics as the original; only the reshape uses
    each tensor's own S dim instead of inheriting q's.
    """
    import flashinfer
    from svg.flashinfer_patch import flashinfer_patch_enabled

    B, H, S_q, D = q.shape
    Bk, Hk, S_kv, Dk = k.shape
    assert (Bk, Hk, Dk) == (B, H, D), (
        f"q[{B},{H},*,{D}] vs k[{Bk},{Hk},*,{Dk}] head/dim mismatch"
    )
    qc_num = block_row_sz.shape[-1]
    kc_num = block_col_sz.shape[-1]
    assert block_mask_map.shape == (B, H, qc_num, kc_num)

    float_workspace_buffer = torch.empty(128 * 1024 * 1024, device=q.device)
    vector_sparse_indices_buffer = torch.empty(1024 * 1024 * 1024, device=q.device)
    wrapper = flashinfer.sparse.VariableBlockSparseAttentionWrapper(
        float_workspace_buffer, backend="auto",
    )
    wrapper.reset_workspace_buffer(
        float_workspace_buffer=wrapper._float_workspace_buffer,
        int_workspace_buffer=wrapper._int_workspace_buffer,
        vector_sparse_indices_buffer=vector_sparse_indices_buffer,
        vector_sparse_indptr_buffer=wrapper._vector_sparse_indptr_buffer,
    )

    q_r = q.reshape(B * H, S_q, D)
    k_r = k.reshape(B * H, S_kv, D)
    v_r = v.reshape(B * H, S_kv, D)
    block_mask_map = block_mask_map.reshape(B * H, qc_num, kc_num)
    block_row_sz = block_row_sz.reshape(B * H, qc_num)
    block_col_sz = block_col_sz.reshape(B * H, kc_num)

    with flashinfer_patch_enabled():
        wrapper.plan(
            block_mask_map=block_mask_map,
            block_row_sz=block_row_sz,
            block_col_sz=block_col_sz,
            num_qo_heads=B * H,
            num_kv_heads=B * H,
            head_dim=D,
            q_data_type=q_r.dtype,
            kv_data_type=k_r.dtype,
        )
    o = wrapper.run(q_r, k_r, v_r)
    return o.reshape(B, H, S_q, D)


@dataclass
class SplitIterCtx:
    """Per-iteration runtime state for ONE split.

    Owner workflow:
      1. owner_post_mask_publish(...) — broadcast post-perm tensors async
      2. owner runs own Q-range attn during the broadcast
      3. owner_recv_helper_outputs(...) — irecv helper attn slices, concat
         into the full attn_out_permuted for this head

    Helper workflow:
      1. helper_post_mask_recv_async(...) — launch broadcast recv (async)
      2. helper runs its own full-head work
      3. helper_run_remote_attn(...) — wait on recv, run sparse attn
      4. helper_send_back(...) — isend the attn_out slice to owner

    Non-participant ranks: still must call broadcast (it's a collective on
    the sub-group). We use the sub-group, so non-members of the group skip
    naturally — they hold a None group and just don't call.
    """
    info: SplitInfo
    group: "dist.ProcessGroup"
    rank: int
    is_owner: bool
    is_helper: bool
    bufs: Optional[_Step1Buffers] = None
    bcast_works: List = field(default_factory=list)
    p2p_works: List = field(default_factory=list)
    # Filled after wait_step1 from cluster_boundaries; ordered participants.
    snapped_q_rows: Optional[List[int]] = None
    snapped_clusters: Optional[List[int]] = None

    def participants_ordered(self) -> List[int]:
        """Participants in the same order the broadcast/boundaries use."""
        return [self.info.owner] + list(self.info.helpers)


def owner_post_mask_publish(
    ctx: SplitIterCtx,
    *,
    k_perm: torch.Tensor,    # [B, 1, S, D] — owner's local mask output for the split head
    v_perm: torch.Tensor,    # [B, 1, S, D]
    q_perm: torch.Tensor,    # [B, 1, S, D]
    dyn_map: torch.Tensor,   # [B, 1, num_q_centroids, num_k_centroids]
    qc_sz: torch.Tensor,     # [B, 1, num_q_centroids]
    kc_sz: torch.Tensor,     # [B, 1, num_k_centroids]
) -> None:
    """OWNER side. Snap planner boundaries to qc_sz cluster boundaries and
    launch async broadcast of post-perm tensors + snapped boundaries.

    Caller must keep these tensors alive until the broadcast completes
    (until ctx.bcast_works are wait()'d). They remain valid in ctx.bufs.
    """
    assert ctx.is_owner
    # 1) Compute snapped boundaries from THIS rank's qc_sz (single head).
    # Planner stored q_ranges as (q_lo, q_hi) per participant; rebuild flat
    # boundary list [0, b1, ..., bN=S] in participants_ordered order.
    participants = ctx.participants_ordered()
    planned = [0] + [int(ctx.info.q_ranges[r][1]) for r in participants]
    snapped_rows, snapped_clusters = snap_to_cluster_boundaries(
        qc_sz[0, 0], planned
    )
    ctx.snapped_q_rows = snapped_rows
    ctx.snapped_clusters = snapped_clusters
    cb = torch.tensor(snapped_clusters, dtype=torch.int32, device=qc_sz.device)

    bufs = _Step1Buffers(
        k_perm=k_perm.contiguous(),
        v_perm=v_perm.contiguous(),
        q_perm=q_perm.contiguous(),
        dyn_map=dyn_map.contiguous(),
        qc_sz=qc_sz.contiguous(),
        kc_sz=kc_sz.contiguous(),
        cluster_boundaries=cb,
    )
    ctx.bufs = bufs
    src = ctx.info.owner
    for t in (bufs.k_perm, bufs.v_perm, bufs.q_perm, bufs.dyn_map,
              bufs.qc_sz, bufs.kc_sz, bufs.cluster_boundaries):
        w = dist.broadcast(t, src=src, group=ctx.group, async_op=True)
        ctx.bcast_works.append(w)


def helper_post_mask_recv_async(
    ctx: SplitIterCtx,
    *,
    cfg: int,
    seq_len: int,
    head_dim: int,
    num_q_centroids: int,
    num_k_centroids: int,
    dtype: torch.dtype,
    device: torch.device,
    dyn_map_dtype: Optional[torch.dtype] = None,
    cluster_size_dtype: Optional[torch.dtype] = None,
) -> None:
    """HELPER side. Allocate buffers + launch async broadcast recv."""
    assert ctx.is_helper
    n_participants = 1 + len(ctx.info.helpers)
    bufs = _Step1Buffers(
        k_perm=torch.empty(cfg, 1, seq_len, head_dim, dtype=dtype, device=device),
        v_perm=torch.empty(cfg, 1, seq_len, head_dim, dtype=dtype, device=device),
        q_perm=torch.empty(cfg, 1, seq_len, head_dim, dtype=dtype, device=device),
        dyn_map=torch.empty(
            cfg, 1, num_q_centroids, num_k_centroids,
            dtype=dyn_map_dtype if dyn_map_dtype is not None else torch.bool,
            device=device,
        ),
        qc_sz=torch.empty(
            cfg, 1, num_q_centroids,
            dtype=cluster_size_dtype if cluster_size_dtype is not None else torch.int32,
            device=device,
        ),
        kc_sz=torch.empty(
            cfg, 1, num_k_centroids,
            dtype=cluster_size_dtype if cluster_size_dtype is not None else torch.int32,
            device=device,
        ),
        cluster_boundaries=torch.empty(
            n_participants + 1, dtype=torch.int32, device=device,
        ),
    )
    ctx.bufs = bufs
    src = ctx.info.owner
    for t in (bufs.k_perm, bufs.v_perm, bufs.q_perm, bufs.dyn_map,
              bufs.qc_sz, bufs.kc_sz, bufs.cluster_boundaries):
        w = dist.broadcast(t, src=src, group=ctx.group, async_op=True)
        ctx.bcast_works.append(w)


def wait_step1(ctx: SplitIterCtx) -> None:
    """Block until all post-perm tensors are ready (broadcast done).

    On helpers, also unpack cluster_boundaries → ctx.snapped_clusters /
    snapped_q_rows. On owner these are already populated by publish().
    """
    for w in ctx.bcast_works:
        w.wait()
    ctx.bcast_works.clear()
    if ctx.is_helper and ctx.snapped_clusters is None:
        bufs = ctx.bufs
        assert bufs is not None
        clusters = bufs.cluster_boundaries.detach().cpu().tolist()
        ctx.snapped_clusters = [int(c) for c in clusters]
        # Recover snapped row positions from received qc_sz cumsum.
        sizes = bufs.qc_sz[0, 0].detach().cpu().tolist()
        cumsum = [0]
        for s in sizes:
            cumsum.append(cumsum[-1] + int(s))
        ctx.snapped_q_rows = [cumsum[c] for c in ctx.snapped_clusters]


def _participant_index(ctx: SplitIterCtx, rank: int) -> int:
    """Index of `rank` in participants_ordered (owner-first)."""
    parts = ctx.participants_ordered()
    return parts.index(rank)


def snapped_q_range_for(ctx: SplitIterCtx, rank: int) -> Tuple[int, int, int, int]:
    """(q_lo_rows, q_hi_rows, c_lo, c_hi) for `rank` after snap."""
    assert ctx.snapped_clusters is not None and ctx.snapped_q_rows is not None
    i = _participant_index(ctx, rank)
    return (
        ctx.snapped_q_rows[i], ctx.snapped_q_rows[i + 1],
        ctx.snapped_clusters[i], ctx.snapped_clusters[i + 1],
    )


def helper_remote_attn(
    ctx: SplitIterCtx,
    attention_fn,
) -> torch.Tensor:
    """HELPER side. Wait for broadcast, slice the helper's Q range against
    the snapped cluster boundaries, run `attention_fn` (signature: q, k, v,
    dyn_map, qc_sz, kc_sz) and return attn_out_slice [B, 1, q_hi-q_lo, D].
    """
    assert ctx.is_helper
    wait_step1(ctx)
    bufs = ctx.bufs
    assert bufs is not None
    q_lo, q_hi, c_lo, c_hi = snapped_q_range_for(ctx, ctx.rank)
    q_sub, dm_sub, qc_sub, kc_sz = slice_split_head_inputs(
        bufs.q_perm, bufs.dyn_map, bufs.qc_sz, bufs.kc_sz,
        q_lo, q_hi, c_lo, c_hi,
    )
    out = attention_fn(q_sub, bufs.k_perm, bufs.v_perm, dm_sub, qc_sub, kc_sz)
    return out


def owner_local_split_attn(
    ctx: SplitIterCtx,
    attention_fn,
    *,
    q_perm: torch.Tensor,    # [B, 1, S, D] — owner's split-head perm Q
    k_perm: torch.Tensor,    # [B, 1, S, D]
    v_perm: torch.Tensor,    # [B, 1, S, D]
    dyn_map: torch.Tensor,   # [B, 1, qc_num, kc_num]
    qc_sz: torch.Tensor,     # [B, 1, qc_num]
    kc_sz: torch.Tensor,     # [B, 1, kc_num]
) -> torch.Tensor:
    """OWNER side. Compute owner's Q-sub of the split head. Must be called
    only after owner_post_mask_publish (which fills ctx.snapped_*)."""
    assert ctx.is_owner
    assert ctx.snapped_clusters is not None
    q_lo, q_hi, c_lo, c_hi = snapped_q_range_for(ctx, ctx.info.owner)
    q_sub, dm_sub, qc_sub, kc_sz_ = slice_split_head_inputs(
        q_perm, dyn_map, qc_sz, kc_sz, q_lo, q_hi, c_lo, c_hi,
    )
    return attention_fn(q_sub, k_perm, v_perm, dm_sub, qc_sub, kc_sz_)


def helper_send_back(
    ctx: SplitIterCtx,
    attn_out_slice: torch.Tensor,
    main_group: "dist.ProcessGroup",
    tag_base: int = 100000,
) -> None:
    """HELPER side. Async-send attn_out_slice to the owner via p2p on the
    main group. tag = tag_base + split_id*16 + helper_index_in_split.
    """
    assert ctx.is_helper
    helper_idx = ctx.info.helpers.index(ctx.rank)
    tag = tag_base + ctx.info.split_id * 16 + helper_idx
    w = dist.isend(
        attn_out_slice.contiguous(), dst=ctx.info.owner,
        group=main_group, tag=tag,
    )
    ctx.p2p_works.append(w)


def owner_recv_helper_outputs(
    ctx: SplitIterCtx,
    *,
    cfg: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    main_group: "dist.ProcessGroup",
    tag_base: int = 100000,
) -> Dict[int, torch.Tensor]:
    """OWNER side. Issue irecv from each helper using snapped row counts,
    wait, return {helper_rank: attn_out_slice [B, 1, q_hi-q_lo, D]}.
    """
    assert ctx.is_owner
    assert ctx.snapped_q_rows is not None
    recv_buffers: Dict[int, torch.Tensor] = {}
    works = []
    for helper_idx, helper_rank in enumerate(ctx.info.helpers):
        q_lo, q_hi, _, _ = snapped_q_range_for(ctx, helper_rank)
        buf = torch.empty(cfg, 1, q_hi - q_lo, head_dim, dtype=dtype, device=device)
        tag = tag_base + ctx.info.split_id * 16 + helper_idx
        w = dist.irecv(buf, src=helper_rank, group=main_group, tag=tag)
        recv_buffers[helper_rank] = buf
        works.append(w)
    for w in works:
        w.wait()
    return recv_buffers


def owner_concat_full_attn_out(
    ctx: SplitIterCtx,
    own_attn_out_slice: torch.Tensor,
    helper_outputs: Dict[int, torch.Tensor],
    seq_len: int,
) -> torch.Tensor:
    """OWNER side. Concatenate owner's Q-range output + helper outputs into
    the full [B, 1, S, D] attn_out_permuted for the split head.

    Uses snapped row boundaries (from snap_to_cluster_boundaries) which may
    differ from the planner's q_ranges when q boundaries didn't fall on
    cluster boundaries.
    """
    assert ctx.is_owner
    assert ctx.snapped_q_rows is not None
    B = own_attn_out_slice.shape[0]
    D = own_attn_out_slice.shape[-1]
    full = torch.empty(B, 1, seq_len, D,
                       dtype=own_attn_out_slice.dtype,
                       device=own_attn_out_slice.device)
    # Owner's slice
    own_lo, own_hi, _, _ = snapped_q_range_for(ctx, ctx.info.owner)
    assert own_attn_out_slice.shape[2] == own_hi - own_lo
    full[:, :, own_lo:own_hi, :].copy_(own_attn_out_slice)
    # Each helper's slice
    for helper_rank, hbuf in helper_outputs.items():
        hlo, hhi, _, _ = snapped_q_range_for(ctx, helper_rank)
        assert hbuf.shape[2] == hhi - hlo, (
            f"helper {helper_rank} sent shape {tuple(hbuf.shape)} "
            f"but snapped q_range is [{hlo},{hhi})"
        )
        full[:, :, hlo:hhi, :].copy_(hbuf)
    return full


def wait_p2p(ctx: SplitIterCtx) -> None:
    """Wait for any pending p2p sends launched by this ctx."""
    for w in ctx.p2p_works:
        w.wait()
    ctx.p2p_works.clear()

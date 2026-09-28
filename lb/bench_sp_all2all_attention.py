#!/usr/bin/env python
import argparse
import csv
import gc
import json
import os
import statistics
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp")
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton-cache")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Sibling imports (lb/)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg.kernels.triton.permute import apply_inverse_permutation_triton
from svg.kmeans_utils import density_calculation, dynamic_block_sparse_fwd_flashinfer

from sap_state import make_sap_state


def pick_s_block(s_local: int, max_block: int = 128) -> int:
    """Largest power-of-2 <= min(max_block, s_local). TMA requires pow-2 block
    shapes; we pad the S dim upstream so s_local divisibility isn't required."""
    cap = min(max_block, max(1, s_local))
    b = 1
    while b * 2 <= cap:
        b *= 2
    return b


CudaEventPair = Optional[Tuple[torch.cuda.Event, torch.cuda.Event]]


def record_cuda_region(fn, label: str = "timed_region", stream: Optional[torch.cuda.Stream] = None):
    if stream is None:
        stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    with torch.profiler.record_function(label):
        out = fn()
    end.record(stream)
    return out, (start, end)


def cuda_elapsed_ms(events: CudaEventPair) -> float:
    if events is None:
        return 0.0
    return events[0].elapsed_time(events[1])


def q_sequence_chunk_density(
    dynamic_map: torch.Tensor,
    qlabels: torch.Tensor,
    k_cluster_sizes: torch.Tensor,
    seq_len: int,
    chunks: int,
) -> torch.Tensor:
    """Density per head for contiguous chunks along the original q sequence dimension.

    For a q token assigned to q-cluster c, the kept key-token count is the sum of
    key cluster sizes selected by dynamic_map[c, :].  Chunk density is the
    average kept-key fraction over tokens in that q chunk.
    """
    cfg, heads, qc_num, _ = dynamic_map.shape
    qlabels = qlabels.view(cfg, heads, seq_len).long()
    kept_keys_per_qcluster = (dynamic_map.to(k_cluster_sizes.dtype) * k_cluster_sizes[:, :, None, :]).sum(dim=-1)
    total_keys = k_cluster_sizes.sum(dim=-1).clamp(min=1).to(kept_keys_per_qcluster.dtype)

    out = []
    for chunk_idx in range(chunks):
        start = seq_len * chunk_idx // chunks
        end = seq_len * (chunk_idx + 1) // chunks
        if end <= start:
            chunk_density = torch.zeros((cfg, heads), device=dynamic_map.device, dtype=kept_keys_per_qcluster.dtype)
        else:
            labels = qlabels[:, :, start:end]
            kept = torch.gather(kept_keys_per_qcluster, dim=-1, index=labels)
            chunk_density = kept.sum(dim=-1) / ((end - start) * total_keys)
        out.append(chunk_density)
    return torch.stack(out, dim=-1)


def split_video_prompt_and_pad(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: Dict,
    world_size: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],
    int,
]:
    """Split a SAP-attention QKV dump into the SP-shardable video region and
    the head-replicated prompt region, padding video to W-divisible.

    Layout: only the visual (video) tokens go through the seq all-to-all;
    the text (prompt) tokens are replicated per-rank and head-sliced after
    the a2a (the same split used by Ulysses-style sequence-parallel attention).

    Returns
    -------
    (video_q, video_k, video_v, prompt_q, prompt_k, prompt_v, video_length_padded)

    For ``model_type="wan"`` (or missing): prompt_* are None and
    ``video_length_padded == seq_len``. The legacy "slice the whole seq
    uniformly" path is kept by giving the whole tensor as the video region.

    For ``model_type="hunyuan"``: video is ``[:, :, :video_length, :]`` zero-padded
    on the seq axis to the next multiple of ``world_size``; prompt is
    ``[:, :, video_length:video_length+context_length, :]`` returned in full.
    """
    model_type = str(metadata.get("model_type", "wan"))
    cfg, num_heads, seq_len, dim = query.shape

    if model_type != "hunyuan":
        if seq_len % world_size != 0:
            raise ValueError(
                f"seq_len={seq_len} must be divisible by world_size={world_size} "
                f"for model_type={model_type!r} (no prompt split available)"
            )
        return query, key, value, None, None, None, seq_len

    context_length = int(metadata.get("context_length", 0))
    video_length = int(metadata.get("video_length", seq_len - context_length))
    if video_length + context_length != seq_len:
        raise ValueError(
            f"hunyuan dump shape mismatch: seq_len={seq_len} != "
            f"video_length({video_length}) + context_length({context_length})"
        )

    pad_len = (-video_length) % world_size  # 0 if already divisible
    video_padded = video_length + pad_len

    def _split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v = x[:, :, :video_length, :]
        if pad_len > 0:
            z = torch.zeros((cfg, num_heads, pad_len, dim), dtype=x.dtype, device=x.device)
            v = torch.cat([v, z], dim=2)
        p = x[:, :, video_length : video_length + context_length, :]
        return v.contiguous(), p.contiguous()

    video_q, prompt_q = _split(query)
    video_k, prompt_k = _split(key)
    video_v, prompt_v = _split(value)
    return video_q, video_k, video_v, prompt_q, prompt_k, prompt_v, video_padded


def all_gather_heads(x_head_local: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, CudaEventPair]:
    """All-gather head-local tensor along the head dim. Used to recover the
    full-head prompt slice on every rank after the head all-gather."""
    if world_size == 1:
        return x_head_local.contiguous(), None

    cfg, h_local, s, d = x_head_local.shape
    # Gather along dim=1 (heads). all_gather_into_tensor expects the gathered
    # ranks to land contiguously at the leading dim, so reshape afterwards.
    flat_in = x_head_local.contiguous()
    flat_out = torch.empty(world_size, *flat_in.shape, dtype=flat_in.dtype, device=flat_in.device)
    _, events = record_cuda_region(
        lambda: dist.all_gather_into_tensor(flat_out, flat_in), "all_gather_prompt_heads"
    )
    # flat_out is (world_size, cfg, h_local, s, d). Reshape to (cfg, world*h_local, s, d).
    out = flat_out.permute(1, 0, 2, 3, 4).reshape(cfg, world_size * h_local, s, d).contiguous()
    return out, events


def all2all_sequence_to_heads(x_seq: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, CudaEventPair]:
    if world_size == 1:
        return x_seq.contiguous(), None

    batch, heads, seq_local, dim = x_seq.shape
    assert heads % world_size == 0
    heads_per_rank = heads // world_size
    send = x_seq.view(batch, world_size, heads_per_rank, seq_local, dim).permute(1, 0, 2, 3, 4).contiguous()
    recv = torch.empty_like(send)

    _, events = record_cuda_region(lambda: dist.all_to_all_single(recv, send), "all2all_sequence_to_heads")
    x_head = recv.permute(1, 2, 0, 3, 4).reshape(batch, heads_per_rank, world_size * seq_local, dim).contiguous()
    return x_head, events


def all2all_heads_to_sequence(x_head: torch.Tensor, world_size: int) -> Tuple[torch.Tensor, CudaEventPair]:
    if world_size == 1:
        return x_head.contiguous(), None

    batch, heads_per_rank, seq_len, dim = x_head.shape
    assert seq_len % world_size == 0
    seq_local = seq_len // world_size
    send = x_head.view(batch, heads_per_rank, world_size, seq_local, dim).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)

    _, events = record_cuda_region(lambda: dist.all_to_all_single(recv, send), "all2all_heads_to_sequence")
    x_seq = recv.permute(1, 0, 2, 3, 4).reshape(batch, world_size * heads_per_rank, seq_local, dim).contiguous()
    return x_seq, events


def start_qkv_allgather(local: Dict, world_size: int, device: torch.device):
    if world_size == 1:
        return None

    current_stream = torch.cuda.current_stream(device)
    comm_stream = torch.cuda.Stream(device=device)
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    gathered = {}
    works = []

    with torch.cuda.stream(comm_stream), torch.profiler.record_function("overlap_qkv_allgather_start"):
        comm_stream.wait_stream(current_stream)
        comm_start.record(comm_stream)
        for name in ("query", "key", "value"):
            inp = local[f"{name}_seq"].contiguous().view(-1)
            out = torch.empty(inp.numel() * world_size, device=device, dtype=inp.dtype)
            work = dist.all_gather_into_tensor(out, inp, async_op=True)
            gathered[name] = out
            works.append(work)
        comm_end.record(comm_stream)

    return {
        "stream": comm_stream,
        "start": comm_start,
        "end": comm_end,
        "works": works,
        "gathered": gathered,
    }


def wait_qkv_allgather(handle: Optional[Dict], device: torch.device) -> CudaEventPair:
    if handle is None:
        return None

    current_stream = torch.cuda.current_stream(device)
    wait_start = torch.cuda.Event(enable_timing=True)
    wait_end = torch.cuda.Event(enable_timing=True)
    with torch.profiler.record_function("overlap_qkv_allgather_wait"):
        wait_start.record(current_stream)
        current_stream.wait_stream(handle["stream"])
        wait_end.record(current_stream)
    return wait_start, wait_end


def finish_qkv_allgather(handle: Optional[Dict]):
    if handle is None:
        return
    for work in handle["works"]:
        work.wait()
    del handle["gathered"]


def _flatten_density(nested) -> List[float]:
    out: List[float] = []
    if isinstance(nested, (list, tuple)):
        for x in nested:
            out.extend(_flatten_density(x))
    else:
        out.append(float(nested))
    return out


def load_density_log(path: str) -> Dict[int, List[Tuple[int, List[float]]]]:
    """Parse a density JSONL into {layer_idx: [(timestep, [density_per_head]), ...]} in denoising order (timestep desc)."""
    by_layer: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            layer = int(row["layer"])
            ts = int(row["timestep"])
            vals = _flatten_density(row["density"])
            by_layer[layer].append((ts, vals))
    for layer in by_layer:
        by_layer[layer].sort(key=lambda r: -r[0])
    return by_layer


def pick_prev_density(
    log: Dict[int, List[Tuple[int, List[float]]]], layer: int, timestep: int
) -> Optional[List[float]]:
    """Return the density vector from the timestep immediately *before* `timestep` for this layer in denoising order.

    Denoising order is descending timesteps. The "previous" timestep is the smallest
    recorded timestep that is strictly larger than the current one.
    """
    rows = log.get(layer)
    if not rows:
        return None
    prev = None
    for ts, vals in rows:  # iterating desc
        if ts > timestep:
            prev = (ts, vals)
        else:
            break
    if prev is None:
        return None
    return prev[1]


def pick_current_density(
    log: Dict[int, List[Tuple[int, List[float]]]], layer: int, timestep: int
) -> Optional[List[float]]:
    """Oracle lookup: return the density vector recorded at the current timestep."""
    rows = log.get(layer)
    if not rows:
        return None
    for ts, vals in rows:
        if ts == timestep:
            return vals
    return None


def greedy_head_assignment(
    densities: Sequence[float], world_size: int
) -> Tuple[List[List[int]], List[float]]:
    """Assign heads to ranks greedily with equal heads-per-rank.

    Sort heads by density desc; place each onto the least-loaded rank that still
    has capacity (= n_heads / world_size). Returns (heads_per_rank, predicted_load).
    Within each rank the returned head-index list is sorted ascending.
    """
    n = len(densities)
    if n % world_size != 0:
        raise ValueError(f"n_heads={n} not divisible by world_size={world_size}")
    cap = n // world_size
    loads = [0.0] * world_size
    assigned: List[List[int]] = [[] for _ in range(world_size)]
    order = sorted(range(n), key=lambda i: -densities[i])
    for head in order:
        best = -1
        for r in range(world_size):
            if len(assigned[r]) >= cap:
                continue
            if best == -1 or loads[r] < loads[best]:
                best = r
        assigned[best].append(head)
        loads[best] += densities[head]
    return [sorted(h) for h in assigned], loads


def load_cost_model(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def load_comm_cost_model(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


def validate_comm_cost_model_shape(
    comm_cost_model: Dict,
    *,
    world_size: int,
    batch: int,
    total_heads: int,
    s_local: int,
    seq_len: int,
    head_dim: int,
    dtype,
) -> None:
    """Reject a communication fit captured for a different tensor shape."""
    model_world = int(comm_cost_model["world_size"])
    if model_world != world_size:
        raise ValueError(
            f"communication model world_size={model_world} != "
            f"effective_world={world_size}"
        )

    model_shape = comm_cost_model.get("shape")
    if not isinstance(model_shape, dict):
        raise ValueError("communication model is missing shape metadata")

    expected = {
        "batch": int(batch),
        "total_heads": int(total_heads),
        "s_local": int(s_local),
        "seq_len": int(seq_len),
        "head_dim": int(head_dim),
        "dtype": str(dtype),
    }
    for name, expected_value in expected.items():
        if name not in model_shape:
            raise ValueError(
                f"communication model shape is missing {name!r}"
            )
        model_value = model_shape[name]
        if name != "dtype":
            model_value = int(model_value)
        else:
            model_value = str(model_value)
        if model_value != expected_value:
            raise ValueError(
                f"communication model {name}={model_value} != "
                f"capture {name}={expected_value}"
            )


def estimate_head_costs(densities: Sequence[float], seq_len: int, cost_model: Optional[Dict]) -> List[float]:
    if cost_model is None:
        return [float(d) for d in densities]
    # Per-head SLOPE cost (placement input). The mask-slope term alpha*N is a
    # per-head floor that already distinguishes fitted-cost placement from a
    # density-only placement. The fitted beta intercepts are per-KERNEL-LAUNCH
    # (per rank), NOT per head -- see split_planner.head_costs/rank_intercept --
    # so they are added once per active rank at makespan time, not here (and they
    # do not change LPT placement).
    mask_slope = float(cost_model["mask_fit"]["slope_ms_per_unit"])
    attention_slope = float(cost_model["attention_fit"]["slope_ms_per_unit"])
    return [
        mask_slope * seq_len + attention_slope * float(density) * seq_len * seq_len
        for density in densities
    ]


def greedy_lpt_assignment(
    densities: Sequence[float], world_size: int, min_heads_per_rank: int = 1
) -> Tuple[List[List[int]], List[float]]:
    """Longest-Processing-Time-first greedy, no equal-heads constraint.

    Sort heads by density desc; place each onto the least-loaded rank (no cap).
    `min_heads_per_rank` guarantees every rank gets at least that many heads
    (seeded with the largest densities in round-robin) so none stays idle.
    Returns (heads_per_rank, predicted_load); per-rank lists sorted ascending.
    """
    n = len(densities)
    if min_heads_per_rank * world_size > n:
        raise ValueError(
            f"min_heads_per_rank={min_heads_per_rank} * world_size={world_size} > n_heads={n}"
        )
    loads = [0.0] * world_size
    assigned: List[List[int]] = [[] for _ in range(world_size)]
    order = sorted(range(n), key=lambda i: -densities[i])

    # Seed: give each rank `min_heads_per_rank` heads round-robin from the top.
    idx = 0
    for seed in range(min_heads_per_rank):
        for r in range(world_size):
            head = order[idx]
            assigned[r].append(head)
            loads[r] += densities[head]
            idx += 1

    # Remaining heads: pure LPT — least-loaded rank wins.
    for head in order[idx:]:
        best = min(range(world_size), key=lambda r: loads[r])
        assigned[best].append(head)
        loads[best] += densities[head]

    return [sorted(h) for h in assigned], loads


def estimate_rank_comm_cost(
    comm_cost_model: Dict,
    rank: int,
    head_count: int,
) -> float:
    """Predict barrier-free local pull+push duration for one rank.

    The communication calibration stores affine fits for three Q/K/V pulls and
    one output push.  Clamp each direction at zero so a noisy fitted intercept
    cannot create a negative cost for small head counts.
    """
    if head_count <= 0:
        return 0.0
    world_size = int(comm_cost_model["world_size"])
    if not (0 <= rank < world_size):
        raise ValueError(f"rank={rank} outside communication model world_size={world_size}")

    total = 0.0
    for direction in ("pull_qkv", "push_out"):
        fits = comm_cost_model[direction]["rank_fits"]
        fit = next((item for item in fits if int(item["rank"]) == rank), None)
        if fit is None:
            raise ValueError(f"communication model {direction} has no fit for rank {rank}")
        predicted = (
            float(fit["intercept_ms"])
            + float(fit["slope_ms_per_head"]) * head_count
        )
        total += max(0.0, predicted)
    return total


def greedy_lpt_assignment_comm_aware(
    head_costs: Sequence[float],
    world_size: int,
    comm_cost_model: Dict,
    min_heads_per_rank: int = 1,
    refine: bool = False,
) -> Tuple[List[List[int]], List[float]]:
    """LPT placement over compute plus rank-local pull/push communication.

    All ranks begin their barrier-free pulls together and synchronize only
    after the reverse push.  The modeled objective is therefore

        max_r(sum(compute_cost[head] for head in rank_r)
              + pull_qkv_r(num_heads_r)
              + push_out_r(num_heads_r)).

    ``refine=True`` enables a move/swap local search for offline analysis.
    Online placement leaves it disabled because the measured real workload
    reaches the same assignment with plain LPT at roughly one third the CPU
    planning cost.
    """
    n = len(head_costs)
    if int(comm_cost_model["world_size"]) != world_size:
        raise ValueError(
            f"communication model world_size={comm_cost_model['world_size']} "
            f"!= requested world_size={world_size}"
        )
    if min_heads_per_rank * world_size > n:
        raise ValueError(
            f"min_heads_per_rank={min_heads_per_rank} * world_size={world_size} > n_heads={n}"
        )

    assigned: List[List[int]] = [[] for _ in range(world_size)]
    compute_loads = [0.0] * world_size
    order = sorted(range(n), key=lambda head: (-float(head_costs[head]), head))
    idx = 0
    cache = comm_cost_model.setdefault("_rank_count_cost_cache", {})
    cache_key = f"{world_size}:{n}"
    rank_count_comm = cache.get(cache_key)
    if rank_count_comm is None:
        rank_count_comm = [
            [
                estimate_rank_comm_cost(comm_cost_model, rank, head_count)
                for head_count in range(n + 1)
            ]
            for rank in range(world_size)
        ]
        cache[cache_key] = rank_count_comm

    # Seed every rank while allowing topology-specific fits to choose which
    # physical rank receives each of the largest heads.
    for seed_count in range(min_heads_per_rank):
        available = set(range(world_size))
        for _ in range(world_size):
            head = order[idx]
            rank = min(
                available,
                key=lambda candidate: (
                    compute_loads[candidate]
                    + float(head_costs[head])
                    + rank_count_comm[candidate][len(assigned[candidate]) + 1],
                    candidate,
                ),
            )
            assigned[rank].append(head)
            compute_loads[rank] += float(head_costs[head])
            available.remove(rank)
            idx += 1

    for head in order[idx:]:
        best_rank = -1
        best_score = None
        for candidate in range(world_size):
            candidate_totals = [
                compute_loads[rank]
                + rank_count_comm[rank][len(assigned[rank])]
                for rank in range(world_size)
            ]
            candidate_totals[candidate] = (
                compute_loads[candidate]
                + float(head_costs[head])
                + rank_count_comm[candidate][len(assigned[candidate]) + 1]
            )
            score = (
                max(candidate_totals),
                candidate_totals[candidate],
                len(assigned[candidate]),
                candidate,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_rank = candidate
        assigned[best_rank].append(head)
        compute_loads[best_rank] += float(head_costs[head])

    def objective(candidate: Sequence[Sequence[int]]) -> Tuple[float, List[float]]:
        loads = [
            sum(float(head_costs[head]) for head in rank_heads)
            + rank_count_comm[rank][len(rank_heads)]
            for rank, rank_heads in enumerate(candidate)
        ]
        return max(loads), loads

    if not refine:
        _, loads = objective(assigned)
        return [sorted(rank_heads) for rank_heads in assigned], loads

    # Improve the greedy solution with single-head moves and pairwise swaps.
    # Only strict objective reductions are accepted, so termination is finite.
    current_obj, current_loads = objective(assigned)
    while True:
        best_obj = current_obj
        best_candidate = None

        for src in range(world_size):
            if len(assigned[src]) <= min_heads_per_rank:
                continue
            for head in list(assigned[src]):
                for dst in range(world_size):
                    if dst == src:
                        continue
                    candidate = [list(rank_heads) for rank_heads in assigned]
                    candidate[src].remove(head)
                    candidate[dst].append(head)
                    candidate_obj, _ = objective(candidate)
                    if candidate_obj < best_obj - 1e-9:
                        best_obj = candidate_obj
                        best_candidate = candidate

        for left in range(world_size):
            for right in range(left + 1, world_size):
                for left_head in assigned[left]:
                    for right_head in assigned[right]:
                        candidate = [list(rank_heads) for rank_heads in assigned]
                        candidate[left].remove(left_head)
                        candidate[right].remove(right_head)
                        candidate[left].append(right_head)
                        candidate[right].append(left_head)
                        candidate_obj, _ = objective(candidate)
                        if candidate_obj < best_obj - 1e-9:
                            best_obj = candidate_obj
                            best_candidate = candidate

        if best_candidate is None:
            break
        assigned = best_candidate
        current_obj, current_loads = objective(assigned)

    return [sorted(rank_heads) for rank_heads in assigned], current_loads


def pad_rank_heads_uniform(
    assigned: List[List[int]],
) -> Tuple[List[int], List[int], int]:
    """Pad per-rank head lists to a uniform length so the symmetric all2all works.

    Each rank's slot is padded by duplicating its first real head. Returns:
      - head_order_padded: flat list of length max_hpr * world_size (may contain duplicates)
      - real_heads_per_rank: how many heads at the start of each rank's slot are real
      - max_hpr: slot size per rank
    """
    world_size = len(assigned)
    max_hpr = max(len(a) for a in assigned)
    head_order: List[int] = []
    real_counts: List[int] = []
    for rank_heads in assigned:
        if not rank_heads:
            raise ValueError("cannot pad an empty rank slot; use min_heads_per_rank>=1")
        real_counts.append(len(rank_heads))
        head_order.extend(rank_heads)
        head_order.extend([rank_heads[0]] * (max_hpr - len(rank_heads)))
    return head_order, real_counts, max_hpr


def restore_indices_from_head_order(
    head_order: Sequence[int],
    world_size: int,
    num_heads: int,
    real_heads_per_rank: Optional[Sequence[int]] = None,
) -> List[int]:
    """Map original global-head order to slots in the reverse symmetric a2a output.

    `head_order` is laid out by rank slots. When `real_heads_per_rank` is set,
    only the first `real_heads_per_rank[r]` slots of each rank are real; later
    slots are duplicate padding and must not be used for restoration.
    """
    if len(head_order) % world_size != 0:
        raise ValueError(
            f"head_order length {len(head_order)} not divisible by world_size {world_size}"
        )
    heads_per_rank = len(head_order) // world_size
    if real_heads_per_rank is not None and len(real_heads_per_rank) != world_size:
        raise ValueError(
            f"real_heads_per_rank length {len(real_heads_per_rank)} != world_size {world_size}"
        )

    restore = [-1] * num_heads
    for r in range(world_size):
        real_count = (
            int(real_heads_per_rank[r])
            if real_heads_per_rank is not None
            else heads_per_rank
        )
        if real_count < 0 or real_count > heads_per_rank:
            raise ValueError(
                f"rank {r} real head count {real_count} outside [0, {heads_per_rank}]"
            )
        slot_base = r * heads_per_rank
        for local_slot in range(real_count):
            slot = slot_base + local_slot
            gh = int(head_order[slot])
            if not (0 <= gh < num_heads):
                raise ValueError(f"head_order contains out-of-range head index {gh}")
            if restore[gh] >= 0:
                raise ValueError(
                    f"global head {gh} appears in multiple real head slots "
                    f"({restore[gh]} and {slot})"
                )
            restore[gh] = slot

    missing = [gh for gh, slot in enumerate(restore) if slot < 0]
    if missing:
        raise ValueError(f"head_order does not cover real global heads {missing}")
    return restore


def compute_rank_heads(
    log_path: Optional[str],
    metadata: Dict,
    num_heads: int,
    world_size: int,
    strategy: str,
    rank: int = 0,
    min_heads_per_rank: int = 1,
    cost_model: Optional[Dict] = None,
    comm_cost_model: Optional[Dict] = None,
    oracle: bool = False,
) -> Tuple[List[List[int]], Optional[Dict]]:
    """Return per-rank head lists for the requested strategy.

    strategy:
      - "contiguous": [r*hpr : (r+1)*hpr] for each r (requires num_heads % world_size == 0)
      - "greedy":     equal heads per rank, greedy by prev-step density
      - "greedy_unequal": LPT, variable heads per rank (>= min_heads_per_rank)

    `oracle=True` looks up the *current*-step density (cheating upper bound)
    instead of the previous-step density.

    Missing or unusable density is an error. Callers that want contiguous must
    request `strategy="contiguous"` explicitly.
    """
    heads_per_rank = num_heads // world_size

    def contiguous_split() -> List[List[int]]:
        if num_heads % world_size != 0:
            raise ValueError(
                f"contiguous split requires num_heads ({num_heads}) divisible by "
                f"world_size ({world_size})"
            )
        return [list(range(r * heads_per_rank, (r + 1) * heads_per_rank)) for r in range(world_size)]

    if strategy == "contiguous":
        return contiguous_split(), None
    if strategy not in ("greedy", "greedy_unequal"):
        raise ValueError(f"unknown strategy: {strategy}")

    if log_path is None:
        raise ValueError(f"strategy={strategy!r} requires a density log")

    log = load_density_log(log_path)
    if metadata.get("layer_idx") is None:
        raise ValueError(f"strategy={strategy!r} requires metadata['layer_idx']")
    layer = int(metadata.get("layer_idx"))
    timestep = metadata.get("timestep")
    if timestep is None:
        timestep = metadata.get("linear_step")
    if timestep is None:
        raise ValueError(
            f"strategy={strategy!r} requires metadata['timestep'] or metadata['linear_step']"
        )

    if oracle:
        density_vec = pick_current_density(log, layer, int(timestep))
        density_label = "current"
    else:
        density_vec = pick_prev_density(log, layer, int(timestep))
        density_label = "prev"
    if density_vec is None:
        raise ValueError(
            f"strategy={strategy!r}: no {density_label} density for "
            f"layer={layer} timestep={timestep} in {log_path}"
        )
    if len(density_vec) != num_heads:
        raise ValueError(
            f"strategy={strategy!r}: {density_label} density length "
            f"{len(density_vec)} != num_heads {num_heads} in {log_path}"
        )

    seq_len = int(metadata.get("seq_len"))
    costs = estimate_head_costs(density_vec, seq_len, cost_model)

    if comm_cost_model is not None and strategy != "greedy_unequal":
        raise ValueError(
            "communication-aware placement currently requires strategy='greedy_unequal'"
        )

    if strategy == "greedy":
        assigned, predicted_loads = greedy_head_assignment(costs, world_size)
    elif comm_cost_model is not None:
        assigned, predicted_loads = greedy_lpt_assignment_comm_aware(
            costs,
            world_size,
            comm_cost_model,
            min_heads_per_rank,
        )
    else:  # greedy_unequal
        assigned, predicted_loads = greedy_lpt_assignment(costs, world_size, min_heads_per_rank)

    if num_heads % world_size == 0:
        predicted_contig = [
            sum(costs[g * heads_per_rank : (g + 1) * heads_per_rank])
            for g in range(world_size)
        ]
        if comm_cost_model is not None:
            predicted_contig = [
                load + estimate_rank_comm_cost(comm_cost_model, g, heads_per_rank)
                for g, load in enumerate(predicted_contig)
            ]
    else:
        predicted_contig = None

    info = {
        "strategy": strategy,
        "layer": layer,
        "timestep": int(timestep),
        "density_source": density_label,
        "prev_density": density_vec,
        "head_costs": costs,
        "cost_model": (
            "maskgen_aware+comm"
            if cost_model is not None and comm_cost_model is not None
            else "density+comm"
            if comm_cost_model is not None
            else "maskgen_aware"
            if cost_model is not None
            else "density"
        ),
        "comm_cost_model": comm_cost_model,
        "assigned": assigned,
        "predicted_loads": predicted_loads,
        "predicted_contig": predicted_contig,
        "heads_per_rank": [len(a) for a in assigned],
    }
    return assigned, info


def compute_head_order(
    log_path: Optional[str],
    metadata: Dict,
    num_heads: int,
    world_size: int,
    rank: int,
    cost_model: Optional[Dict] = None,
) -> Tuple[List[int], Optional[Dict]]:
    """Decide the global head ordering used before the seq->head all2all.

    Returns (head_order, info). `head_order` is a permutation of range(num_heads)
    where slots [r*cap:(r+1)*cap] are the global heads assigned to rank r.
    `info` (rank-0-friendly) carries diagnostics. Missing density is an error.
    """
    heads_per_rank = num_heads // world_size
    if log_path is None:
        raise ValueError("compute_head_order requires a density log")

    log = load_density_log(log_path)
    if metadata.get("layer_idx") is None:
        raise ValueError("compute_head_order requires metadata['layer_idx']")
    layer = int(metadata.get("layer_idx"))
    timestep = metadata.get("timestep")
    if timestep is None:
        timestep = metadata.get("linear_step")
    if timestep is None:
        raise ValueError("compute_head_order requires metadata['timestep'] or metadata['linear_step']")

    prev = pick_prev_density(log, layer, int(timestep))
    if prev is None:
        raise ValueError(
            f"compute_head_order: no prior density in log for layer={layer} "
            f"before timestep={timestep} in {log_path}"
        )
    if len(prev) != num_heads:
        raise ValueError(
            f"compute_head_order: prior density length {len(prev)} != "
            f"num_heads {num_heads} in {log_path}"
        )

    seq_len = int(metadata.get("seq_len"))
    costs = estimate_head_costs(prev, seq_len, cost_model)
    assigned, predicted_loads = greedy_head_assignment(costs, world_size)
    head_order = [h for rank_heads in assigned for h in rank_heads]

    predicted_contig = [
        sum(costs[g * heads_per_rank : (g + 1) * heads_per_rank]) for g in range(world_size)
    ]

    info = {
        "layer": layer,
        "timestep": int(timestep),
        "prev_density": prev,
        "head_costs": costs,
        "cost_model": "maskgen_aware" if cost_model is not None else "density",
        "assigned": assigned,
        "predicted_loads": predicted_loads,
        "predicted_contig": predicted_contig,
    }
    return head_order, info


def load_local_shards(
    path: str,
    rank: int,
    world_size: int,
    device: torch.device,
    head_order: Optional[List[int]] = None,
    real_heads_per_rank: Optional[List[int]] = None,
    symm_a2a=None,
    h_idxs_r: Optional[List[int]] = None,
    max_heads_per_rank: Optional[int] = None,
    all_rank_heads: Optional[List[List[int]]] = None,
):
    """Load the local (seq-sharded) QKV + centroid caches for this rank.

    `head_order`: optional flat list of global head indices. Its length must be a
    multiple of `world_size`; rank r gets slots `[r*hpr : (r+1)*hpr]` where
    `hpr = len(head_order) // world_size`. For `greedy_unequal`, the list may
    contain duplicates (pad slots filled by duplicating a real head) so that
    the symmetric all2all still works.

    `real_heads_per_rank`: when padding is used, how many of each rank's slots
    are real heads. Padded slots are ignored downstream (density / metrics).
    """
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    query = data["inputs"]["query"]
    key = data["inputs"]["key"]
    value = data["inputs"]["value"]
    q_cache = data["centroid_cache"]["q_centroids"]
    k_cache = data["centroid_cache"]["k_centroids"]

    cfg, num_heads_original, seq_len, dim = query.shape
    if cfg != 1:
        raise ValueError(f"Only cfg=1 is supported by this benchmark, got cfg={cfg}")

    # Split into the SP-shardable video region (zero-padded to W-divisible)
    # and the head-replicated prompt region. For wan dumps with no prompt,
    # prompt_* are None and video_q/k/v == query/key/value verbatim.
    is_hunyuan = str(metadata.get("model_type", "wan")) == "hunyuan"
    (
        video_q, video_k, video_v,
        prompt_q, prompt_k, prompt_v,
        video_length_padded,
    ) = split_video_prompt_and_pad(query, key, value, metadata, world_size)
    if video_length_padded % world_size != 0:
        raise AssertionError(
            f"split_video_prompt_and_pad failed to produce a divisible video length "
            f"({video_length_padded} % {world_size} != 0)"
        )

    seq_local = video_length_padded // world_size
    s0, s1 = rank * seq_local, (rank + 1) * seq_local

    if symm_a2a is not None:
        # Asymmetric pull path: populate symm buffers with this rank's seq shard
        # for ALL heads (no permute — the kernel gathers by global index).
        if h_idxs_r is None:
            raise ValueError("symm_a2a path requires h_idxs_r")
        q_shard = video_q[:, :, s0:s1, :].to(device).contiguous()
        k_shard = video_k[:, :, s0:s1, :].to(device).contiguous()
        v_shard = video_v[:, :, s0:s1, :].to(device).contiguous()
        # Only the first `s_local` seq slots carry real data; the remaining
        # [s_local:s_local_padded] are garbage (TMA block-shape padding) and
        # will be pulled along with real data, then trimmed on the recv side.
        real_s = symm_a2a.s_local
        symm_a2a.q_symm[:, :, :real_s, :].copy_(q_shard)
        symm_a2a.k_symm[:, :, :real_s, :].copy_(k_shard)
        symm_a2a.v_symm[:, :, :real_s, :].copy_(v_shard)

        # Simulate post-all-gather steady state: every rank holds the full
        # centroid cache on GPU, then index_selects its assigned heads. In
        # production the all_gather happens after each step's kmeans_step;
        # here we just preload the captured (full) cache to GPU once. The
        # GPU-side select is what would actually run per step.
        q_cache_full_dev = q_cache.to(device).contiguous()
        k_cache_full_dev = k_cache.to(device).contiguous()
        h_idxs_cpu = torch.tensor(list(h_idxs_r), dtype=torch.long)
        h_idxs_long = h_idxs_cpu.to(device)
        q_cache_local = q_cache_full_dev.index_select(0, h_idxs_long).contiguous()
        k_cache_local = k_cache_full_dev.index_select(0, h_idxs_long).contiguous()
        del q_cache_full_dev, k_cache_full_dev

        max_hpr = int(max_heads_per_rank) if max_heads_per_rank else len(h_idxs_r)
        # For hunyuan: replace `video_length` so SAPState's hunyuan branch sees
        # the padded video length carried by the asymmetric video buffers.
        runtime_metadata = dict(metadata)
        if is_hunyuan:
            runtime_metadata["video_length"] = video_length_padded
        local = {
            "metadata": runtime_metadata,
            "symm_a2a": symm_a2a,
            "h_idxs_r": torch.tensor(list(h_idxs_r), dtype=torch.int32, device=device),
            "h_idxs_r_py": list(h_idxs_r),
            "q_cache": q_cache_local,
            "k_cache": k_cache_local,
            "head_start": int(h_idxs_r[0]) if len(h_idxs_r) else 0,
            "head_end": int(h_idxs_r[0]) + len(h_idxs_r) if len(h_idxs_r) else 0,
            # Prompt output uses NCCL all-gather, which needs a uniform head
            # width across ranks when greedy_unequal assigns variable heads.
            "heads_per_rank_padded": max_hpr,
            "real_heads_count": len(h_idxs_r),
            "assigned_heads": list(h_idxs_r),
            "seq_start": s0,
            "seq_end": s1,
            "seq_total": video_length_padded,
            "video_length_padded": video_length_padded,
        }
        if prompt_q is not None:
            local["query_prompt_local"] = prompt_q.index_select(1, h_idxs_cpu).contiguous().to(device)
            local["key_prompt_local"] = prompt_k.index_select(1, h_idxs_cpu).contiguous().to(device)
            local["value_prompt_local"] = prompt_v.index_select(1, h_idxs_cpu).contiguous().to(device)
            if all_rank_heads is not None:
                padded_order, real_counts, _ = pad_rank_heads_uniform(all_rank_heads)
                local["restore_head_indices"] = torch.tensor(
                    restore_indices_from_head_order(
                        padded_order,
                        world_size,
                        num_heads_original,
                        real_heads_per_rank=real_counts,
                    ),
                    dtype=torch.long,
                    device=device,
                )
        del data, query, key, value, q_cache, k_cache
        del video_q, video_k, video_v, prompt_q, prompt_k, prompt_v
        gc.collect()
        return local

    # Simulate post-all-gather steady state for centroid caches: full cache to
    # GPU first, then permute / slice on device.
    q_cache_dev = q_cache.to(device).contiguous()
    k_cache_dev = k_cache.to(device).contiguous()

    restore_head_indices: Optional[torch.Tensor] = None
    if head_order is not None:
        if len(head_order) % world_size != 0:
            raise ValueError(
                f"head_order length {len(head_order)} not divisible by world_size {world_size}"
            )
        heads_per_rank = len(head_order) // world_size
        h0, h1 = rank * heads_per_rank, (rank + 1) * heads_per_rank
        if not all(0 <= h < num_heads_original for h in head_order):
            raise ValueError("head_order contains out-of-range head indices")
        perm = torch.tensor(head_order, dtype=torch.long)
        # Apply the head permutation to both video and prompt (when present)
        # so the rank-local slice [h0:h1] on the head dim gives a consistent
        # set of heads on both branches.
        video_q = video_q.index_select(1, perm)
        video_k = video_k.index_select(1, perm)
        video_v = video_v.index_select(1, perm)
        if prompt_q is not None:
            prompt_q = prompt_q.index_select(1, perm)
            prompt_k = prompt_k.index_select(1, perm)
            prompt_v = prompt_v.index_select(1, perm)
        perm_dev = perm.to(device)
        q_cache_dev = q_cache_dev.index_select(0, perm_dev)
        k_cache_dev = k_cache_dev.index_select(0, perm_dev)
        restore_head_indices = torch.tensor(
            restore_indices_from_head_order(
                head_order,
                world_size,
                num_heads_original,
                real_heads_per_rank=real_heads_per_rank,
            ),
            dtype=torch.long,
            device=device,
        )

        if real_heads_per_rank is not None:
            real_count = int(real_heads_per_rank[rank])
            assigned_heads = list(head_order[h0 : h0 + real_count])
        else:
            real_count = heads_per_rank
            assigned_heads = list(head_order[h0:h1])
    else:
        if num_heads_original % world_size != 0:
            raise ValueError(
                f"num_heads={num_heads_original} must be divisible by world_size={world_size}"
            )
        heads_per_rank = num_heads_original // world_size
        h0, h1 = rank * heads_per_rank, (rank + 1) * heads_per_rank
        real_count = heads_per_rank
        assigned_heads = list(range(h0, h1))

    runtime_metadata = dict(metadata)
    if is_hunyuan:
        runtime_metadata["video_length"] = video_length_padded
    local = {
        "metadata": runtime_metadata,
        "query_seq": video_q[:, :, s0:s1, :].contiguous().to(device),
        "key_seq": video_k[:, :, s0:s1, :].contiguous().to(device),
        "value_seq": video_v[:, :, s0:s1, :].contiguous().to(device),
        "q_cache": q_cache_dev[h0:h1].contiguous(),
        "k_cache": k_cache_dev[h0:h1].contiguous(),
        "head_start": h0,
        "head_end": h1,
        "heads_per_rank_padded": heads_per_rank,
        "real_heads_count": real_count,
        "assigned_heads": assigned_heads,
        "seq_start": s0,
        "seq_end": s1,
        "video_length_padded": video_length_padded,
    }
    if restore_head_indices is not None:
        local["restore_head_indices"] = restore_head_indices
    if prompt_q is not None:
        # Hunyuan: prompt is replicated on every rank — pre-slice to this
        # rank's head range so we don't carry the unused heads in memory
        # (matches the post-a2a head-slice step of Ulysses-style SP).
        local["query_prompt_local"] = prompt_q[:, h0:h1, :, :].contiguous().to(device)
        local["key_prompt_local"] = prompt_k[:, h0:h1, :, :].contiguous().to(device)
        local["value_prompt_local"] = prompt_v[:, h0:h1, :, :].contiguous().to(device)
    del q_cache_dev, k_cache_dev

    del data, query, key, value, q_cache, k_cache
    del video_q, video_k, video_v, prompt_q, prompt_k, prompt_v
    gc.collect()
    return local


def load_full_data(path: str, device: torch.device, world_size: int) -> Dict:
    """Load full Q/K/V + centroid caches onto `device`. Used by sweep mode so
    each sim_rank can pick its own seq shard without reloading from disk."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    query = data["inputs"]["query"].to(device).contiguous()
    key = data["inputs"]["key"].to(device).contiguous()
    value = data["inputs"]["value"].to(device).contiguous()
    (
        video_q, video_k, video_v,
        prompt_q, prompt_k, prompt_v,
        video_length_padded,
    ) = split_video_prompt_and_pad(query, key, value, metadata, world_size)
    full = {
        "metadata": metadata,
        "num_heads": int(query.shape[1]),
        "video_q": video_q,
        "video_k": video_k,
        "video_v": video_v,
        "prompt_q": prompt_q,
        "prompt_k": prompt_k,
        "prompt_v": prompt_v,
        "video_length_padded": video_length_padded,
        "q_cache": data["centroid_cache"]["q_centroids"].to(device).contiguous(),
        "k_cache": data["centroid_cache"]["k_centroids"].to(device).contiguous(),
    }
    del data
    gc.collect()
    return full


def populate_for_sim_rank(
    symm_a2a, full_data: Dict, sim_rank: int,
    assigned: List[List[int]], effective_world: int, device: torch.device,
) -> Dict:
    """Re-fill q/k/v_symm with `sim_rank`'s seq shard and return a `local`
    dict in the same layout `run_iteration` consumes from the symm path."""
    metadata = full_data["metadata"]
    is_hunyuan = str(metadata.get("model_type", "wan")) == "hunyuan"
    seq_total = int(full_data["video_length_padded"])
    seq_local = seq_total // effective_world
    s0 = sim_rank * seq_local
    s1 = (sim_rank + 1) * seq_local
    real_s = symm_a2a.s_local
    symm_a2a.q_symm[:, :, :real_s, :].copy_(full_data["video_q"][:, :, s0:s1, :])
    symm_a2a.k_symm[:, :, :real_s, :].copy_(full_data["video_k"][:, :, s0:s1, :])
    symm_a2a.v_symm[:, :, :real_s, :].copy_(full_data["video_v"][:, :, s0:s1, :])

    h_idxs_r_list = list(assigned[sim_rank])
    h_idxs_cpu = torch.tensor(h_idxs_r_list, dtype=torch.long)
    h_idxs_long = torch.tensor(h_idxs_r_list, dtype=torch.long, device=device)
    q_cache_local = full_data["q_cache"].index_select(0, h_idxs_long).contiguous()
    k_cache_local = full_data["k_cache"].index_select(0, h_idxs_long).contiguous()
    runtime_metadata = dict(metadata)
    if is_hunyuan:
        runtime_metadata["video_length"] = seq_total

    local = {
        "metadata": runtime_metadata,
        "symm_a2a": symm_a2a,
        "h_idxs_r": torch.tensor(h_idxs_r_list, dtype=torch.int32, device=device),
        "h_idxs_r_py": h_idxs_r_list,
        "q_cache": q_cache_local,
        "k_cache": k_cache_local,
        "head_start": int(h_idxs_r_list[0]) if h_idxs_r_list else 0,
        "head_end": int(h_idxs_r_list[0]) + len(h_idxs_r_list) if h_idxs_r_list else 0,
        "heads_per_rank_padded": max(len(rank_heads) for rank_heads in assigned),
        "real_heads_count": len(h_idxs_r_list),
        "assigned_heads": list(h_idxs_r_list),
        "seq_start": s0,
        "seq_end": s1,
        "seq_total": seq_total,
    }
    if is_hunyuan:
        local["video_length_padded"] = seq_total
        local["query_prompt_local"] = full_data["prompt_q"].index_select(1, h_idxs_cpu.to(device)).contiguous()
        local["key_prompt_local"] = full_data["prompt_k"].index_select(1, h_idxs_cpu.to(device)).contiguous()
        local["value_prompt_local"] = full_data["prompt_v"].index_select(1, h_idxs_cpu.to(device)).contiguous()
        padded_order, real_counts, _ = pad_rank_heads_uniform(assigned)
        local["restore_head_indices"] = torch.tensor(
            restore_indices_from_head_order(
                padded_order,
                effective_world,
                int(full_data["num_heads"]),
                real_heads_per_rank=real_counts,
            ),
            dtype=torch.long,
            device=device,
        )
    return local


def run_iteration(local: Dict, args: argparse.Namespace, rank: int, world_size: int, device: torch.device):
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()

    symm_a2a = local.get("symm_a2a")
    if symm_a2a is not None:
        # Asymmetric pull: no permute needed, kernel gathers by global head idx.
        # Low-sync path: Q/K/V are write-once (populated in load_local_shards
        # and guarded by a one-time setup barrier in main). Per-iter pulls run
        # barrier-free; visibility is already established.
        h_idxs_r = local["h_idxs_r"]
        num_sms = args.num_sms if args.num_sms > 0 else None

        def _pull(name: str):
            return symm_a2a.pull_seq_to_heads(
                name, h_idxs_r, num_sms=num_sms,
                pre_barrier=False, post_barrier=False,
            )

        q_head_full, q_a2a_events = record_cuda_region(
            lambda: _pull("q"), "all2all_sequence_to_heads",
        )
        k_head_full, k_a2a_events = record_cuda_region(
            lambda: _pull("k"), "all2all_sequence_to_heads",
        )
        v_head_full, v_a2a_events = record_cuda_region(
            lambda: _pull("v"), "all2all_sequence_to_heads",
        )
        # Pull output is [B, H_local, world * S_local_padded, D]. Trim per peer
        # back to the real S_local (when S_block had to pad for divisibility).
        seq_total = local["seq_total"]
        if q_head_full.shape[2] != seq_total:
            s_padded = q_head_full.shape[2] // world_size
            s_real = seq_total // world_size
            def _trim(x):
                x = x.view(x.shape[0], x.shape[1], world_size, s_padded, x.shape[3])
                x = x[:, :, :, :s_real, :].contiguous()
                return x.view(x.shape[0], x.shape[1], world_size * s_real, x.shape[4])
            q_head = _trim(q_head_full)
            k_head = _trim(k_head_full)
            v_head = _trim(v_head_full)
        else:
            q_head, k_head, v_head = q_head_full, k_head_full, v_head_full
        real_count = q_head.shape[1]
        # For the reverse (still padded symmetric all2all), pad attn_out to
        # a uniform width across ranks.
        heads_padded = int(local.get("heads_per_rank_padded", real_count))
    else:
        q_head, q_a2a_events = all2all_sequence_to_heads(local["query_seq"], world_size)
        k_head, k_a2a_events = all2all_sequence_to_heads(local["key_seq"], world_size)
        v_head, v_a2a_events = all2all_sequence_to_heads(local["value_seq"], world_size)

        # When padding was used to keep the all2all symmetric, slice off the pad
        # slots here so mask/kmeans/attention only see this rank's real heads.
        heads_padded = q_head.shape[1]
        real_count = int(local.get("real_heads_count", heads_padded))
        if real_count < heads_padded:
            q_head = q_head[:, :real_count, :, :].contiguous()
            k_head = k_head[:, :real_count, :, :].contiguous()
            v_head = v_head[:, :real_count, :, :].contiguous()

    # Hunyuan split-replicated layout (Option C): the a2a above carried only
    # the video region. Concat each rank's pre-loaded local-head slice of the
    # prompt onto the seq dim so the attention kernel sees the full
    # [video_padded, prompt] per-rank seq.
    has_replicated_prompt = "query_prompt_local" in local
    video_seq_padded = q_head.shape[2]  # captured BEFORE the concat
    if has_replicated_prompt:
        # Slice the head-local prompt to `real_count` to match q_head's heads
        # (greedy_unequal padding may have made heads_padded > real_count).
        pq = local["query_prompt_local"][:, :real_count, :, :]
        pk = local["key_prompt_local"][:, :real_count, :, :]
        pv = local["value_prompt_local"][:, :real_count, :, :]
        q_head = torch.cat([q_head, pq], dim=2).contiguous()
        k_head = torch.cat([k_head, pk], dim=2).contiguous()
        v_head = torch.cat([v_head, pv], dim=2).contiguous()

    metadata = local["metadata"]
    state = make_sap_state(
        metadata,
        q_centroids=local["q_cache"][:real_count].clone(),
        k_centroids=local["k_cache"][:real_count].clone(),
        top_p_override=args.top_p,
        min_kc_ratio_override=args.min_kc_ratio,
    )

    overlap_handle = None
    if args.overlap_qkv_allgather_during_mask:
        overlap_handle = start_qkv_allgather(local, world_size, device)

    mask_outputs, mask_events = record_cuda_region(
        lambda: state.semantic_aware_permutation(q_head, k_head, v_head), "mask_semantic_aware_permutation"
    )
    qkv_allgather_wait_events = wait_qkv_allgather(overlap_handle, device)
    q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, q_sorted_indices, qlabels, klabels = mask_outputs

    split_runtime_meta = local.get("split_runtime")
    split_events = None   # (start, end) cuda events; evaluated after sync
    head_restore_events: List[CudaEventPair] = []

    if split_runtime_meta is None or (
        split_runtime_meta["my_owner_split"] is None
        and not split_runtime_meta["my_helper_splits"]
    ):
        # Non-participant in any split (or balance != "split"): main path.
        attn_out_permuted, attention_events = record_cuda_region(
            lambda: dynamic_block_sparse_fwd_flashinfer(
                q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, is_cpu=False
            ),
            "dynamic_block_sparse_attention",
        )
    else:
        from split_runtime import (
            SplitIterCtx,
            owner_post_mask_publish,
            owner_local_split_attn,
            owner_recv_helper_outputs,
            owner_concat_full_attn_out,
            helper_post_mask_recv_async,
            helper_remote_attn,
            helper_send_back,
            wait_p2p,
            variable_qkv_block_sparse_fwd_flashinfer,
        )
        # Split-head sub-range attention: q_len != kv_len (helper sees a
        # Q sub-range against the full kv). Use the variable-qkv variant.
        split_attn_fn = variable_qkv_block_sparse_fwd_flashinfer
        my_owner_split = split_runtime_meta["my_owner_split"]
        my_helper_splits = split_runtime_meta["my_helper_splits"]
        groups = split_runtime_meta["groups"]
        is_owner = my_owner_split is not None
        is_helper = bool(my_helper_splits)
        cfg = q_perm.shape[0]
        S = q_perm.shape[2]
        D = q_perm.shape[3]
        qc_num = state.num_q_centroids
        kc_num = state.num_k_centroids

        _dbg = os.environ.get("SPLIT_DEBUG", "0") == "1"
        role = "owner" if is_owner else "helper"
        def _dlog(msg):
            if _dbg:
                print(f"[split-dbg rank{rank} role={role}] {msg}", flush=True)

        if is_helper:
            # Helper of >= 1 splits. Owner-XOR-helper rule guarantees this
            # rank is NOT also an owner.
            attn_out_permuted, attention_events = record_cuda_region(
                lambda: dynamic_block_sparse_fwd_flashinfer(
                    q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz, is_cpu=False
                ),
                "dynamic_block_sparse_attention",
            )
            torch.cuda.synchronize(device)
            # flashinfer wrapper.run output aliases the wrapper's workspace;
            # later flashinfer calls reuse that storage and clobber the data.
            attn_out_permuted = attn_out_permuted.clone()
            split_start = torch.cuda.Event(enable_timing=True)
            split_end = torch.cuda.Event(enable_timing=True)
            split_start.record()
            # Build one ctx per split this rank helps. Launch async recv for
            # each before doing any compute, so all owners can proceed.
            ctxs: List = []
            for s in my_helper_splits:
                ctx_s = SplitIterCtx(
                    info=s, group=groups[s.split_id], rank=rank,
                    is_owner=False, is_helper=True,
                )
                ctxs.append(ctx_s)
                helper_post_mask_recv_async(
                    ctx_s, cfg=cfg, seq_len=S, head_dim=D,
                    num_q_centroids=qc_num, num_k_centroids=kc_num,
                    dtype=q_perm.dtype, device=device,
                    dyn_map_dtype=dyn_map.dtype,
                    cluster_size_dtype=qc_sz.dtype,
                )
            _dlog(f"recv_async started for {len(ctxs)} split(s)")
            # Compute + send-back per split (serial — no concurrency wins
            # required, just correctness).
            for ctx_s in ctxs:
                helper_slice = helper_remote_attn(ctx_s, split_attn_fn).clone()
                helper_send_back(ctx_s, helper_slice, main_group=dist.group.WORLD)
            for ctx_s in ctxs:
                wait_p2p(ctx_s)
            _dlog(f"helper done for all {len(ctxs)} split(s)")
            split_end.record()
            split_events = (split_start, split_end)
        else:
            # Owner of exactly 1 split (rule: owner XOR helper, owner of at
            # most one split).
            ctx = SplitIterCtx(
                info=my_owner_split, group=groups[my_owner_split.split_id],
                rank=rank, is_owner=True, is_helper=False,
            )
            local_h_idx = split_runtime_meta["my_owner_local_h_idx"]
            sl = slice(local_h_idx, local_h_idx + 1)
            k_split = k_perm[:, sl].contiguous()
            v_split = v_perm[:, sl].contiguous()
            q_split = q_perm[:, sl].contiguous()
            dyn_map_split = dyn_map[:, sl].contiguous()
            qc_sz_split = qc_sz[:, sl].contiguous()
            kc_sz_split = kc_sz[:, sl].contiguous()
            split_start = torch.cuda.Event(enable_timing=True)
            split_end = torch.cuda.Event(enable_timing=True)
            split_start.record()
            H_owner = q_perm.shape[1]
            ns_idx = [i for i in range(H_owner) if i != local_h_idx]
            if ns_idx:
                ns_t = torch.tensor(ns_idx, dtype=torch.long, device=device)
                q_ns = q_perm.index_select(1, ns_t).contiguous()
                k_ns = k_perm.index_select(1, ns_t).contiguous()
                v_ns = v_perm.index_select(1, ns_t).contiguous()
                dm_ns = dyn_map.index_select(1, ns_t).contiguous()
                qc_ns = qc_sz.index_select(1, ns_t).contiguous()
                kc_ns = kc_sz.index_select(1, ns_t).contiguous()
                out_ns, attention_events = record_cuda_region(
                    lambda: dynamic_block_sparse_fwd_flashinfer(
                        q_ns, k_ns, v_ns, dm_ns, qc_ns, kc_ns, is_cpu=False
                    ),
                    "dynamic_block_sparse_attention",
                )
                out_ns = out_ns.clone()
            else:
                out_ns = None
                attention_events = None
            torch.cuda.synchronize(device)
            _dlog(f"owner non-split done; publishing split_id={my_owner_split.split_id}")
            owner_post_mask_publish(
                ctx,
                k_perm=k_split, v_perm=v_split, q_perm=q_split,
                dyn_map=dyn_map_split, qc_sz=qc_sz_split, kc_sz=kc_sz_split,
            )
            owner_split_slice = owner_local_split_attn(
                ctx, split_attn_fn,
                q_perm=q_split, k_perm=k_split, v_perm=v_split,
                dyn_map=dyn_map_split, qc_sz=qc_sz_split, kc_sz=kc_sz_split,
            ).clone()
            helper_outputs = owner_recv_helper_outputs(
                ctx, cfg=cfg, head_dim=D,
                dtype=q_perm.dtype, device=device,
                main_group=dist.group.WORLD,
            )
            split_full = owner_concat_full_attn_out(
                ctx, owner_split_slice, helper_outputs, seq_len=S,
            )
            attn_out_permuted = torch.empty(
                cfg, H_owner, S, D, dtype=q_perm.dtype, device=device,
            )
            attn_out_permuted[:, local_h_idx:local_h_idx + 1].copy_(split_full)
            if out_ns is not None:
                ns_t = torch.tensor(ns_idx, dtype=torch.long, device=device)
                attn_out_permuted.index_copy_(1, ns_t, out_ns)
            split_end.record()
            split_events = (split_start, split_end)

    attn_out, inverse_permute_events = record_cuda_region(
        lambda: apply_inverse_permutation_triton(attn_out_permuted, q_sorted_indices, dim=2), "inverse_permute"
    )

    prompt_out_events = None
    if symm_a2a is not None:
        if has_replicated_prompt:
            attn_out_video = attn_out[:, :, :video_seq_padded, :].contiguous()
            attn_out_prompt = attn_out[:, :, video_seq_padded:, :].contiguous()
        else:
            attn_out_video = attn_out
            attn_out_prompt = None

        # Reverse asymm push: stage video attn_out into a local padded buffer
        # ([B, H_local, world * s_padded, D]) and let the push kernel
        # deposit each global head directly on the rank that owns the seq
        # shard. Push needs only ONE barrier (after the kernel) — no
        # mid-iter sync between local-attn-finish and peer reads.
        B_, H_loc_, S_full_, D_ = attn_out_video.shape
        assert S_full_ == world_size * (S_full_ // world_size)
        s_real = S_full_ // world_size
        s_padded = symm_a2a.s_local_padded
        if s_padded != s_real:
            attn_src_padded = torch.empty(
                B_, H_loc_, world_size * s_padded, D_,
                dtype=attn_out_video.dtype, device=attn_out_video.device,
            )
            attn_src_view = attn_src_padded.view(B_, H_loc_, world_size, s_padded, D_)
            attn_out_view = attn_out_video.reshape(B_, H_loc_, world_size, s_real, D_)
            attn_src_view[:, :, :, :s_real, :].copy_(attn_out_view)
        else:
            attn_src_padded = attn_out_video.contiguous()

        num_sms = args.num_sms if args.num_sms > 0 else None
        h_idxs_r = local["h_idxs_r"]

        out_seq_full, all2all_out_events = record_cuda_region(
            lambda: symm_a2a.push_heads_to_seq(
                attn_src_padded, h_idxs_r, num_sms=num_sms,
                pre_barrier=False, post_barrier=True,
            ),
            "all2all_heads_to_sequence",
        )
        # [B, H_total, s_padded, D] → trim padding to [B, H_total, s_real, D]
        if s_padded != s_real:
            out_video = out_seq_full[:, :, :s_real, :].contiguous()
        else:
            out_video = out_seq_full

        if attn_out_prompt is not None:
            if real_count < heads_padded:
                pad_p = torch.zeros(
                    attn_out_prompt.shape[0],
                    heads_padded - real_count,
                    attn_out_prompt.shape[2],
                    attn_out_prompt.shape[3],
                    device=attn_out_prompt.device,
                    dtype=attn_out_prompt.dtype,
                )
                attn_prompt_for_ag = torch.cat([attn_out_prompt, pad_p], dim=1).contiguous()
            else:
                attn_prompt_for_ag = attn_out_prompt
            prompt_full, prompt_out_events = all_gather_heads(attn_prompt_for_ag, world_size)
            restore_head_indices = local.get("restore_head_indices")
            if restore_head_indices is not None:
                prompt_full, restore_events = record_cuda_region(
                    lambda: prompt_full.index_select(1, restore_head_indices).contiguous(),
                    "restore_head_order_prompt",
                )
                head_restore_events.append(restore_events)
            out_seq = torch.cat([out_video, prompt_full], dim=2).contiguous()
        else:
            out_seq = out_video
    else:
        # Hunyuan split-replicated layout: split attn_out into video and prompt
        # parts before the reverse comm. Prompt portion is gathered along the
        # head dim (recovers full-head slice on every rank); video portion goes
        # through the standard reverse a2a back to seq-sharded.
        if has_replicated_prompt:
            attn_out_video = attn_out[:, :, :video_seq_padded, :].contiguous()
            attn_out_prompt = attn_out[:, :, video_seq_padded:, :].contiguous()
        else:
            attn_out_video = attn_out
            attn_out_prompt = None

        # Pad attn_out_video back to the symmetric head count before the reverse all2all.
        # The pad slots' outputs are junk; downstream is expected to ignore them.
        if real_count < heads_padded:
            pad = torch.zeros(
                attn_out_video.shape[0],
                heads_padded - real_count,
                attn_out_video.shape[2],
                attn_out_video.shape[3],
                device=attn_out_video.device,
                dtype=attn_out_video.dtype,
            )
            attn_out_for_a2a = torch.cat([attn_out_video, pad], dim=1).contiguous()
        else:
            attn_out_for_a2a = attn_out_video

        out_video, all2all_out_events = all2all_heads_to_sequence(attn_out_for_a2a, world_size)
        restore_head_indices = local.get("restore_head_indices")
        if restore_head_indices is not None:
            out_video, restore_events = record_cuda_region(
                lambda: out_video.index_select(1, restore_head_indices).contiguous(),
                "restore_head_order_video",
            )
            head_restore_events.append(restore_events)

        if attn_out_prompt is not None:
            # All-gather along the head dim to recover the full-head prompt
            # slice on every rank. Pad to heads_padded first if greedy_unequal
            # produced an asymmetric head count, so all ranks contribute the
            # same shape (pad slots discarded after gather).
            if real_count < heads_padded:
                pad_p = torch.zeros(
                    attn_out_prompt.shape[0],
                    heads_padded - real_count,
                    attn_out_prompt.shape[2],
                    attn_out_prompt.shape[3],
                    device=attn_out_prompt.device,
                    dtype=attn_out_prompt.dtype,
                )
                attn_prompt_for_ag = torch.cat([attn_out_prompt, pad_p], dim=1).contiguous()
            else:
                attn_prompt_for_ag = attn_out_prompt
            prompt_full, prompt_out_events = all_gather_heads(attn_prompt_for_ag, world_size)
            if restore_head_indices is not None:
                prompt_full, restore_events = record_cuda_region(
                    lambda: prompt_full.index_select(1, restore_head_indices).contiguous(),
                    "restore_head_order_prompt",
                )
                head_restore_events.append(restore_events)
            # Concat the video shard (post a2a, length video_seq_padded/W) with
            # the recovered full prompt onto the seq dim, matching the natural
            # per-rank "[video_local, prompt_full]" layout consumed downstream
            # of the SP attention block.
            out_seq = torch.cat([out_video, prompt_full], dim=2).contiguous()
        else:
            out_seq = out_video

    total_end.record()

    density_gpu = density_calculation(dyn_map, qc_sz, kc_sz).reshape(-1).detach().float()
    # qlabels covers only the kmeans'd region: full seq for wan, video-only
    # (length video_seq_padded) for hunyuan. Match the seq_len passed to the
    # chunker to qlabels' actual length so view(cfg, heads, seq_len) succeeds.
    qlabels_seq_len = video_seq_padded if has_replicated_prompt else q_head.shape[2]
    q_chunk_density_gpu = q_sequence_chunk_density(
        dyn_map,
        qlabels,
        kc_sz,
        seq_len=qlabels_seq_len,
        chunks=args.q_density_chunks,
    ).reshape(q_head.shape[1], args.q_density_chunks).detach().float()

    torch.cuda.synchronize(device)
    finish_qkv_allgather(overlap_handle)

    all2all_in_ms = cuda_elapsed_ms(q_a2a_events) + cuda_elapsed_ms(k_a2a_events) + cuda_elapsed_ms(v_a2a_events)
    mask_ms = cuda_elapsed_ms(mask_events)
    qkv_allgather_comm_ms = (
        cuda_elapsed_ms((overlap_handle["start"], overlap_handle["end"])) if overlap_handle is not None else 0.0
    )
    qkv_allgather_wait_ms = cuda_elapsed_ms(qkv_allgather_wait_events)
    attention_ms = cuda_elapsed_ms(attention_events)
    split_extra_ms = cuda_elapsed_ms(split_events) if split_events is not None else 0.0
    inverse_permute_ms = cuda_elapsed_ms(inverse_permute_events)
    all2all_out_ms = cuda_elapsed_ms(all2all_out_events) + cuda_elapsed_ms(prompt_out_events)
    head_restore_ms = sum(cuda_elapsed_ms(events) for events in head_restore_events)
    total_ms = cuda_elapsed_ms((total_start, total_end))
    density = density_gpu.cpu()
    q_chunk_density = q_chunk_density_gpu.cpu()

    metrics = {
        "rank": rank,
        "head_start": local["head_start"],
        "head_end": local["head_end"],
        "seq_start": local["seq_start"],
        "seq_end": local["seq_end"],
        "all2all_in_ms": all2all_in_ms,
        "mask_ms": mask_ms,
        "qkv_allgather_comm_ms": qkv_allgather_comm_ms,
        "qkv_allgather_wait_ms": qkv_allgather_wait_ms,
        "attention_ms": attention_ms,
        "split_extra_ms": split_extra_ms,
        "inverse_permute_ms": inverse_permute_ms,
        "all2all_out_ms": all2all_out_ms,
        "head_restore_ms": head_restore_ms,
        "total_ms": total_ms,
    }
    if local.get("dump_out_seq_path"):
        torch.save(
            out_seq.detach().cpu(),
            local["dump_out_seq_path"],
        )

    assigned_heads = local.get("assigned_heads") or list(range(local["head_start"], local["head_end"]))
    real_count = int(local.get("real_heads_count", len(assigned_heads)))
    head_density = [
        {"global_head": int(assigned_heads[i]), "rank": rank, "density": float(value)}
        for i, value in enumerate(density.tolist()[:real_count])
    ]

    q_chunk_density_rows = []
    for local_head_idx, values in enumerate(q_chunk_density.tolist()[:real_count]):
        global_head = int(assigned_heads[local_head_idx])
        for chunk_idx, value in enumerate(values):
            start = q_head.shape[2] * chunk_idx // args.q_density_chunks
            end = q_head.shape[2] * (chunk_idx + 1) // args.q_density_chunks
            q_chunk_density_rows.append(
                {
                    "global_head": global_head,
                    "rank": rank,
                    "q_chunk": chunk_idx,
                    "q_start": start,
                    "q_end": end,
                    "q_len": end - start,
                    "density": float(value),
                }
            )

    del q_head, k_head, v_head, mask_outputs, q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz
    del q_sorted_indices, qlabels, klabels, attn_out_permuted, attn_out, out_seq, density_gpu, q_chunk_density_gpu, state
    return metrics, head_density, q_chunk_density_rows


def summarize_rank_metrics(rows: List[Dict]) -> List[Dict]:
    by_rank: Dict[int, List[Dict]] = {}
    for row in rows:
        by_rank.setdefault(int(row["rank"]), []).append(row)

    summary = []
    for rank, rank_rows in sorted(by_rank.items()):
        out = {
            "rank": rank,
            "head_start": rank_rows[0]["head_start"],
            "head_end": rank_rows[0]["head_end"],
            "seq_start": rank_rows[0]["seq_start"],
            "seq_end": rank_rows[0]["seq_end"],
            "iters": len(rank_rows),
        }
        for key in [
            "all2all_in_ms",
            "mask_ms",
            "qkv_allgather_comm_ms",
            "qkv_allgather_wait_ms",
            "attention_ms",
            "split_extra_ms",
            "inverse_permute_ms",
            "all2all_out_ms",
            "head_restore_ms",
            "total_ms",
        ]:
            vals = [float(row[key]) for row in rank_rows]
            out[f"{key}_mean"] = statistics.mean(vals)
            out[f"{key}_median"] = statistics.median(vals)
        summary.append(out)
    return summary


def summarize_density(rows: List[Dict]) -> List[Dict]:
    by_head: Dict[int, List[Dict]] = {}
    for row in rows:
        by_head.setdefault(int(row["global_head"]), []).append(row)

    summary = []
    for head, head_rows in sorted(by_head.items()):
        vals = [float(row["density"]) for row in head_rows]
        summary.append(
            {
                "global_head": head,
                "rank": int(head_rows[0]["rank"]),
                "density_mean": statistics.mean(vals),
                "density_min": min(vals),
                "density_max": max(vals),
            }
        )
    return summary


def summarize_q_chunk_density(rows: List[Dict]) -> List[Dict]:
    by_key: Dict[Tuple[int, int], List[Dict]] = {}
    for row in rows:
        key = (int(row["global_head"]), int(row["q_chunk"]))
        by_key.setdefault(key, []).append(row)

    summary = []
    for (head, chunk), chunk_rows in sorted(by_key.items()):
        vals = [float(row["density"]) for row in chunk_rows]
        first = chunk_rows[0]
        summary.append(
            {
                "global_head": head,
                "rank": int(first["rank"]),
                "q_chunk": chunk,
                "q_start": int(first["q_start"]),
                "q_end": int(first["q_end"]),
                "q_len": int(first["q_len"]),
                "density_mean": statistics.mean(vals),
                "density_min": min(vals),
                "density_max": max(vals),
            }
        )
    return summary


def write_csv(path: Path, rows: List[Dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rank_summary(rows: List[Dict]):
    header = (
        "rank heads      seq           a2a_in  mask     qkv_ag  ag_wait  attention inv_perm a2a_out restore  total"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['rank']:>4} "
            f"{row['head_start']:>2}-{row['head_end'] - 1:<5} "
            f"{row['seq_start']:>6}-{row['seq_end'] - 1:<6} "
            f"{row['all2all_in_ms_mean']:>7.2f} "
            f"{row['mask_ms_mean']:>8.2f} "
            f"{row['qkv_allgather_comm_ms_mean']:>7.2f} "
            f"{row['qkv_allgather_wait_ms_mean']:>8.2f} "
            f"{row['attention_ms_mean']:>9.2f} "
            f"{row['inverse_permute_ms_mean']:>8.2f} "
            f"{row['all2all_out_ms_mean']:>7.2f} "
            f"{row['head_restore_ms_mean']:>7.2f} "
            f"{row['total_ms_mean']:>7.2f}"
        )


def print_density_summary(rows: List[Dict]):
    print("\nper-head density:")
    for row in rows:
        print(f"head {row['global_head']:>2} rank {row['rank']:>2}: {row['density_mean']:.6f}")


def print_q_chunk_density_summary(rows: List[Dict]):
    print("\nper-head q-chunk density:")
    for row in rows:
        print(
            f"head {row['global_head']:>2} rank {row['rank']:>2} "
            f"chunk {row['q_chunk']:>2} q[{row['q_start']},{row['q_end']}): "
            f"{row['density_mean']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Reproduce Wan SVG2 sequence-parallel all2all + per-rank head attention.")
    parser.add_argument("--input", required=True, help="Path to SVG_WAN_ATTN_EXPORT_PATH .pt file.")
    parser.add_argument("--top_p", "--top-p", type=float, default=None,
                        help="Override the dump's SAP top_p_kmeans sparsity knob (replay the same "
                             "Q/K/V at a different sparsity level; for the sparsity-sensitivity sweep).")
    parser.add_argument("--min-kc-ratio", type=float, default=None,
                        help="Override the dump's SAP min_kc_ratio (optional sparsity-floor knob).")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rank-csv", default=None)
    parser.add_argument("--density-csv", default=None)
    parser.add_argument("--q-chunk-density-csv", default=None)
    parser.add_argument(
        "--q-density-chunks",
        type=int,
        default=8,
        help="Number of contiguous chunks along q sequence for per-head chunk density reporting.",
    )
    parser.add_argument(
        "--overlap-qkv-allgather-during-mask",
        action="store_true",
        help="Start async all-gather of sequence-sharded Q/K/V on a separate CUDA stream while mask/permutation runs.",
    )
    parser.add_argument(
        "--density-log",
        default=None,
        help="Path to a density JSONL (layer/timestep/density). When set, heads are "
        "redistributed across ranks with greedy balancing using the density from "
        "the previous timestep of this layer (the only predictor available at runtime).",
    )
    parser.add_argument(
        "--balance",
        choices=["contiguous", "greedy", "greedy_unequal", "split"],
        default="contiguous",
        help="How to assign heads to ranks. 'greedy'/'greedy_unequal'/'split' require "
        "--density-log. 'greedy_unequal' drops the equal-heads constraint; pads shorter "
        "ranks by duplicating a real head so the symmetric all2all still works. 'split' "
        "uses greedy_unequal placement plus one post-permutation Q-row split of the "
        "bottleneck rank's heaviest head over a helper set (NCCL p2p, C2 path); requires "
        "--asymm-a2a pull_qkv and --cost-model-json.",
    )
    parser.add_argument(
        "--split-max-helpers", type=int, default=2,
        help="For --balance split: max helpers per split (planner-side cap).",
    )
    parser.add_argument(
        "--split-max-splits", type=int, default=1,
        help="For --balance split: max number of splits in a single plan. "
        "Each rank can be owner of at most one split (per the planner's "
        "owner-or-helper exclusion rule); a rank may be helper to several owners.",
    )
    parser.add_argument(
        "--split-delta", type=float, default=0.10,
        help="For --balance split: residual-tail threshold (planner-side).",
    )
    parser.add_argument(
        "--split-q-granularity", type=int, default=1,
        help="For --balance split: planner q-row alignment (rows).",
    )
    parser.add_argument(
        "--split-min-improvement-ms", type=float, default=0.5,
        help="For --balance split: skip split if improvement below this.",
    )
    parser.add_argument(
        "--dump-out-seq", default=None,
        help="If set, each rank writes its post-iter `out_seq` (last iteration) "
        "to '<path>.rank<R>.pt'. Used by test_split_e2e.py for correctness checks.",
    )
    parser.add_argument(
        "--min-heads-per-rank",
        type=int,
        default=1,
        help="For --balance greedy_unequal: guarantee each rank gets at least this many heads.",
    )
    parser.add_argument(
        "--cost-model-json",
        default=None,
        help="Optional model from profile_maskgen_aware_cost.py. With --balance greedy, "
        "predicts load as mask_slope*seq_len + attn_slope*density*seq_len^2 instead of raw density.",
    )
    parser.add_argument(
        "--comm-cost-model-json",
        default=None,
        help="Optional barrier-free per-rank pull/push model from "
        "profile_asymm_comm_cost.py. Requires --balance greedy_unequal and "
        "--asymm-a2a pull_qkv; placement minimizes compute + local communication.",
    )
    parser.add_argument(
        "--oracle-density",
        action="store_true",
        help="Ablation upper bound: feed the CURRENT timestep's density to the "
        "placement policy instead of the previous step's. Cheats the online "
        "constraint; useful for measuring how much head room better prediction has.",
    )
    parser.add_argument(
        "--asymm-a2a",
        choices=["off", "pull_qkv"],
        default="off",
        help="Replace both forward (seq→heads for Q/K/V) and reverse "
        "(heads→seq for attn_out) all2alls with pull-based kernels over "
        "torch symmetric memory (Triton + TMA). Reverse scatters by global "
        "head index on the fly, so no pad+permute needed.",
    )
    parser.add_argument(
        "--num-sms",
        type=int,
        default=0,
        help="Persistent kernel grid size. 0 = device SM count.",
    )
    parser.add_argument(
        "--sim-world",
        type=int,
        default=0,
        help="2-GPU N-rank simulation. When >0, real PG must have world_size=2 "
        "and --asymm-a2a=pull_qkv. Each real GPU plays one sim rank from a "
        "world of this size; cross-rank traffic funnels onto the single real "
        "NVLink pair so cross-rank throughput is roughly (sim_world-1)x "
        "slower than the real configuration (latency / SMs / sync surface "
        "area stay accurate). 0 = off.",
    )
    parser.add_argument(
        "--sim-ranks",
        default=None,
        help="Comma-separated `r0,r1` picking which sim ranks the two real "
        "GPUs play. Default: `0,sim_world-1`. Ignored without --sim-world or "
        "when --sim-passive-rank is set.",
    )
    parser.add_argument(
        "--sim-passive-rank",
        type=int,
        default=-1,
        help="2-GPU sweep mode: real rank R becomes a passive buffer/barrier "
        "host while the other real rank (active) sequentially plays every "
        "sim_rank in [0, sim_world). Output table has one row per sim_rank, "
        "indistinguishable from a real N-card run. v1 only supports R=1 "
        "(rank 0 must stay active so it can print). Requires --sim-world and "
        "--asymm-a2a pull_qkv. -1 = off.",
    )
    parser.add_argument(
        "--s-block",
        type=int,
        default=0,
        help="Kernel S_BLOCK. 0 = auto (largest divisor of S_local that is <=128).",
    )
    parser.add_argument("--profile-dir", default=None, help="Directory for per-rank torch profiler Chrome traces.")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--profile-shapes", action="store_true")
    parser.add_argument("--profile-stack", action="store_true")
    args = parser.parse_args()
    if args.q_density_chunks <= 0:
        raise SystemExit("--q-density-chunks must be positive")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 2-GPU N-rank simulation: real PG stays at world_size=2, but everything
    # downstream of head/seq sharding (and the symm-mem kernels) sees
    # `effective_world` ranks. Real `world_size` / `rank` are still used for
    # dist primitives (init/barrier/gather) and for the device assignment.
    sim_sweep = args.sim_passive_rank >= 0
    is_passive = False
    if sim_sweep:
        if args.sim_world <= 0:
            raise SystemExit("--sim-passive-rank requires --sim-world > 0")
        if args.asymm_a2a != "pull_qkv":
            raise SystemExit("--sim-passive-rank requires --asymm-a2a pull_qkv")
        if args.sim_passive_rank not in (0, 1):
            raise SystemExit("--sim-passive-rank must be 0 or 1")
        if args.sim_passive_rank == 0:
            raise SystemExit(
                "v1: --sim-passive-rank must be 1; rank 0 stays active so it "
                "can print/aggregate the unified table"
            )
        if world_size != 2:
            raise SystemExit(
                f"--sim-passive-rank requires real world_size=2, got {world_size}"
            )
        is_passive = (rank == args.sim_passive_rank)
        effective_world = args.sim_world
        # Active starts at sim_rank=0 (will sweep); passive parks at 0 too
        # (its peer_ptrs are never used because passive never calls pull/push).
        effective_rank = 0
        if rank == 0:
            print(
                f"[sim_world={effective_world}, sweep] real rank 0 sweeps "
                f"sim_rank 0..{effective_world - 1}; real rank 1 passive "
                f"(host buffer + barrier only); cross-rank bandwidth "
                f"pessimistic by ≈{effective_world - 1}x"
            )
    elif args.sim_world > 0:
        if args.asymm_a2a != "pull_qkv":
            raise SystemExit("--sim-world requires --asymm-a2a pull_qkv")
        if world_size != 2:
            raise SystemExit(
                f"--sim-world requires real world_size=2, got {world_size}"
            )
        if args.sim_ranks:
            sim_ranks = [int(x) for x in args.sim_ranks.split(",")]
            if len(sim_ranks) != 2:
                raise SystemExit(f"--sim-ranks must list 2 ints, got {sim_ranks}")
        else:
            sim_ranks = [0, args.sim_world - 1]
        for sr in sim_ranks:
            if not (0 <= sr < args.sim_world):
                raise SystemExit(
                    f"--sim-ranks {sim_ranks} out of [0, {args.sim_world})"
                )
        if sim_ranks[0] == sim_ranks[1]:
            raise SystemExit(f"--sim-ranks must be distinct, got {sim_ranks}")
        effective_world = args.sim_world
        effective_rank = sim_ranks[rank]
        if rank == 0:
            print(
                f"[sim_world={effective_world}] real rank 0 ↔ sim_rank "
                f"{sim_ranks[0]}, real rank 1 ↔ sim_rank {sim_ranks[1]}; "
                f"cross-rank bandwidth pessimistic by ≈{effective_world - 1}x"
            )
    else:
        effective_world = world_size
        effective_rank = rank

    if args.balance in ("greedy", "greedy_unequal", "split") and args.density_log is None:
        raise SystemExit(f"--balance {args.balance} requires --density-log")
    if args.balance == "split":
        if args.asymm_a2a != "pull_qkv":
            raise SystemExit("--balance split requires --asymm-a2a pull_qkv")
        if args.cost_model_json is None:
            raise SystemExit("--balance split requires --cost-model-json")
        if args.sim_world > 0:
            raise SystemExit("--balance split is incompatible with --sim-world (yet)")
    if args.comm_cost_model_json is not None:
        if args.balance != "greedy_unequal":
            raise SystemExit(
                "--comm-cost-model-json requires --balance greedy_unequal"
            )
        if args.asymm_a2a != "pull_qkv":
            raise SystemExit(
                "--comm-cost-model-json requires --asymm-a2a pull_qkv"
            )
        if args.sim_world > 0:
            raise SystemExit(
                "--comm-cost-model-json is incompatible with --sim-world"
            )
    cost_model = load_cost_model(args.cost_model_json)
    comm_cost_model = load_comm_cost_model(args.comm_cost_model_json)

    # Peek at metadata + head count to compute the greedy assignment (if enabled).
    head_order: Optional[List[int]] = None
    real_heads_per_rank: Optional[List[int]] = None
    greedy_info: Optional[Dict] = None
    assigned: Optional[List[List[int]]] = None
    split_plan = None  # set when args.balance == "split"
    if args.balance in ("greedy", "greedy_unequal", "split"):
        peek = torch.load(args.input, map_location="cpu", weights_only=False)
        peek_metadata = peek["metadata"]
        peek_query = peek["inputs"]["query"]
        cfg, num_heads, query_seq_len, head_dim = peek_query.shape
        capture_dtype = peek_query.dtype
        if comm_cost_model is not None:
            if str(peek_metadata.get("model_type", "wan")) == "hunyuan":
                context_length = int(peek_metadata.get("context_length", 0))
                video_length = int(
                    peek_metadata.get(
                        "video_length",
                        query_seq_len - context_length,
                    )
                )
                comm_seq_len = video_length + (
                    (-video_length) % effective_world
                )
            else:
                comm_seq_len = query_seq_len
            if comm_seq_len % effective_world != 0:
                raise SystemExit(
                    f"capture communication seq_len={comm_seq_len} is not "
                    f"divisible by effective_world={effective_world}"
                )
            try:
                validate_comm_cost_model_shape(
                    comm_cost_model,
                    world_size=effective_world,
                    batch=cfg,
                    total_heads=num_heads,
                    s_local=comm_seq_len // effective_world,
                    seq_len=comm_seq_len,
                    head_dim=head_dim,
                    dtype=capture_dtype,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        del peek_query, peek
        gc.collect()
        # Split path uses greedy_unequal placement as its baseline. Planner
        # runs over the same densities; assignments match by construction.
        assignment_strategy = "greedy_unequal" if args.balance == "split" else args.balance
        assigned, greedy_info = compute_rank_heads(
            args.density_log,
            peek_metadata,
            num_heads,
            effective_world,
            strategy=assignment_strategy,
            rank=effective_rank,
            min_heads_per_rank=args.min_heads_per_rank,
            cost_model=cost_model,
            comm_cost_model=comm_cost_model,
            oracle=args.oracle_density,
        )
        if args.balance == "greedy_unequal":
            head_order, real_heads_per_rank, max_hpr = pad_rank_heads_uniform(assigned)
            if rank == 0 and greedy_info is not None:
                print(
                    f"[greedy_unequal] heads/rank={real_heads_per_rank} "
                    f"padded to {max_hpr} each (pads duplicate rank's first real head)"
                )
        else:
            # equal-split greedy or split: assigned has variable size; for
            # split the asymm path is mandatory so head_order isn't used.
            head_order = [h for rank_heads in assigned for h in rank_heads]

        if args.balance == "split":
            import split_planner
            from split_planner import plan_with_splits, pick_prev_density, load_density_log as load_dl_split
            # Lower the planner's qualifying-head filter for tests on small
            # captures (real configs leave the default 1.0 ms).
            min_attn_env = os.environ.get("WAN_SPLIT_MIN_ATTN_MS")
            if min_attn_env is not None:
                split_planner.MIN_SPLIT_ATTN_MS = float(min_attn_env)
                if rank == 0:
                    print(f"[split] WAN_SPLIT_MIN_ATTN_MS={min_attn_env}")
            split_log = load_dl_split(args.density_log)
            ts = peek_metadata.get("timestep") or peek_metadata.get("linear_step")
            if args.oracle_density:
                cur = pick_current_density(split_log, int(peek_metadata["layer_idx"]), int(ts))
                if cur is None:
                    raise SystemExit(
                        f"--balance split --oracle-density: no current-step density for "
                        f"layer={peek_metadata['layer_idx']} ts={ts} in {args.density_log}"
                    )
                _prev_ts, prev_densities = int(ts), cur
            else:
                prev = pick_prev_density(split_log, int(peek_metadata["layer_idx"]), int(ts))
                if prev is None:
                    raise SystemExit(
                        f"--balance split: no prior density for layer="
                        f"{peek_metadata['layer_idx']} ts={ts} in {args.density_log}"
                    )
                _prev_ts, prev_densities = prev
            split_plan = plan_with_splits(
                densities=prev_densities,
                cost_model=cost_model,
                seq_len=int(peek_metadata["seq_len"]),
                world_size=effective_world,
                delta=args.split_delta,
                max_helpers_per_split=args.split_max_helpers,
                max_splits_per_plan=args.split_max_splits,
                min_heads_per_rank=args.min_heads_per_rank,
                q_granularity=args.split_q_granularity,
                min_improvement_ms=args.split_min_improvement_ms,
            )
            if rank == 0:
                n_splits = len(split_plan.diagnostics["splits"])
                print(
                    f"[split] {n_splits} split(s); predicted_max="
                    f"{split_plan.predicted_max_ms:.2f} ms vs baseline_max="
                    f"{split_plan.baseline_max_ms:.2f} ms; speedup="
                    f"{split_plan.speedup:.3f}x"
                )

    # Asymmetric all2all setup: allocate Q/K/V directly in symmetric memory and
    # skip the forward padded permute path entirely.
    symm_a2a = None
    h_idxs_r_asymm: Optional[List[int]] = None
    max_hpr_asymm: Optional[int] = None
    if args.asymm_a2a == "pull_qkv":
        from symm_a2a import SymmAsymA2A

        if assigned is None:
            # contiguous balance: generate trivial assignment
            peek = torch.load(args.input, map_location="cpu", weights_only=False)
            peek_metadata = peek["metadata"]
            num_heads = int(peek["inputs"]["query"].shape[1])
            cfg, _, seq_len, dim = peek["inputs"]["query"].shape
            dtype = peek["inputs"]["query"].dtype
            del peek
            gc.collect()
            if num_heads % effective_world != 0:
                raise SystemExit(
                    f"contiguous balance + asymm requires num_heads ({num_heads}) "
                    f"divisible by effective_world ({effective_world})"
                )
            hpr = num_heads // effective_world
            assigned = [
                list(range(r * hpr, (r + 1) * hpr)) for r in range(effective_world)
            ]
        else:
            peek = torch.load(args.input, map_location="cpu", weights_only=False)
            peek_metadata = peek["metadata"]
            cfg, num_heads, seq_len, dim = peek["inputs"]["query"].shape
            dtype = peek["inputs"]["query"].dtype
            del peek
            gc.collect()

        h_idxs_r_asymm = assigned[effective_rank]
        max_hpr_asymm = max(len(h) for h in assigned)
        if str(peek_metadata.get("model_type", "wan")) == "hunyuan":
            context_length = int(peek_metadata.get("context_length", 0))
            video_length = int(peek_metadata.get("video_length", seq_len - context_length))
            seq_total_for_a2a = video_length + ((-video_length) % effective_world)
        else:
            seq_total_for_a2a = seq_len
        if seq_total_for_a2a % effective_world != 0:
            raise SystemExit(
                f"asymm-a2a seq length {seq_total_for_a2a} is not divisible by "
                f"effective_world {effective_world}"
            )
        s_local_real = seq_total_for_a2a // effective_world
        s_block = args.s_block if args.s_block > 0 else pick_s_block(s_local_real)
        if s_block <= 0 or (s_block & (s_block - 1)) != 0:
            raise SystemExit(
                f"--s-block must be a power of 2, got {s_block}"
            )

        sim_kw = {}
        if args.sim_world > 0:
            sim_kw = {"sim_world": effective_world, "sim_rank": effective_rank}

        symm_a2a = SymmAsymA2A(
            dist.group.WORLD,
            buffer_shape=(cfg, num_heads, s_local_real, dim),
            dtype=dtype,
            device=device,
            s_block=s_block,
            enable_reverse=True,
            **sim_kw,
        )
        if comm_cost_model is not None:
            model_backend = str(comm_cost_model.get("backend", symm_a2a.backend))
            if model_backend != symm_a2a.backend:
                raise SystemExit(
                    f"communication model backend={model_backend!r} != "
                    f"active asymmetric backend={symm_a2a.backend!r}"
                )
            model_config = comm_cost_model.get("pcie_config")
            active_config = (
                list(symm_a2a.pcie_config)
                if symm_a2a.pcie_config is not None
                else None
            )
            if model_config is not None and list(model_config) != active_config:
                raise SystemExit(
                    f"communication model pcie_config={model_config} != "
                    f"active pcie_config={active_config}"
                )
        if rank == 0:
            print(
                f"[asymm_a2a=pull_qkv] symm buffers [{cfg},{num_heads},{s_local_real},{dim}] "
                f"dtype={dtype} s_block={s_block} num_sms={args.num_sms or 'auto'}; "
                f"backend={symm_a2a.backend} pcie_config={symm_a2a.pcie_config}; "
                f"per-rank heads={[len(h) for h in assigned]} max_hpr={max_hpr_asymm} "
                f"effective_world={effective_world}"
            )
        # Asymm path drives its own Q/K/V — don't also pad+permute in load_local_shards.
        head_order = None
        real_heads_per_rank = None

    full_data: Optional[Dict] = None
    local: Optional[Dict] = None
    if sim_sweep and not is_passive:
        # Active rank in sweep: load full Q/K/V once. Per-sim_rank populate +
        # `local` rebuild happen inside the sweep loop, so that warmup and
        # main iters of each sim_rank run on the right shard / heads.
        full_data = load_full_data(args.input, device, effective_world)
    elif sim_sweep and is_passive:
        # Passive rank: seed q/k/v_symm with sim_rank=0's shard so the active
        # rank's non-self pulls return realistic data (not zeros) regardless
        # of which sim_rank it is currently playing.
        local = load_local_shards(
            args.input, 0, effective_world, device,
            symm_a2a=symm_a2a,
            h_idxs_r=assigned[0],
            max_heads_per_rank=max_hpr_asymm,
            all_rank_heads=assigned,
        )
    else:
        local = load_local_shards(
            args.input, effective_rank, effective_world, device,
            head_order=head_order, real_heads_per_rank=real_heads_per_rank,
            symm_a2a=symm_a2a,
            h_idxs_r=h_idxs_r_asymm,
            max_heads_per_rank=max_hpr_asymm,
            all_rank_heads=assigned,
        )
    # In non-sweep mode the symm Q/K/V are write-once: a single barrier here
    # makes peers' setup writes visible for the rest of the run. In sweep
    # mode the active rank rewrites Q/K/V per sim_rank, so the barrier moves
    # inside the sweep loop.
    if symm_a2a is not None and not sim_sweep:
        symm_a2a.barrier()
    dist.barrier()

    if args.dump_out_seq is not None and local is not None:
        local["dump_out_seq_path"] = f"{args.dump_out_seq}.rank{rank}.pt"

    # --- split runtime setup: build SplitInfo + sub-process-groups once ---
    # dist.new_group is collective: every rank in WORLD must call it the
    # same number of times in the same order, even non-members.
    #
    # Multi-split (max_splits_per_plan > 1) under the "owner XOR helper"
    # rule from the planner: a rank is at most one of {owner, helper,
    # non-participant} across the whole plan. So `my_owner_split` is
    # at most 1; `my_helper_splits` may have several entries (helper
    # serving multiple owners).
    if args.balance == "split":
        from split_runtime import extract_splits, make_split_groups
        split_infos = extract_splits(split_plan)
        split_infos = [s for s in split_infos if s.helpers]
        split_groups = make_split_groups(split_infos, effective_world)
        my_owner_split = None
        my_owner_local_h_idx = -1
        my_helper_splits: List = []
        for s in split_infos:
            if effective_rank == s.owner:
                assert my_owner_split is None, (
                    "planner produced two splits with the same owner — violates "
                    "owner-or-helper rule"
                )
                my_owner_split = s
                my_owner_local_h_idx = h_idxs_r_asymm.index(s.global_head)
            elif effective_rank in s.helpers:
                my_helper_splits.append(s)
        if my_owner_split is not None and my_helper_splits:
            raise RuntimeError(
                f"rank {effective_rank} is owner of split {my_owner_split.split_id} "
                f"AND helper of {len(my_helper_splits)} other splits — violates "
                "owner-or-helper exclusion rule"
            )
        local["split_runtime"] = {
            "splits": split_infos,
            "groups": split_groups,
            "my_owner_split": my_owner_split,
            "my_owner_local_h_idx": my_owner_local_h_idx,
            "my_helper_splits": my_helper_splits,
        }
        if rank == 0:
            for s in split_infos:
                print(
                    f"[split-runtime] split_id={s.split_id} head={s.global_head} "
                    f"owner={s.owner} helpers={s.helpers} planned_q_ranges={s.q_ranges}"
                )

    if sim_sweep:
        sweep_ranks = list(range(effective_world))
    else:
        sweep_ranks = [effective_rank]

    profiler_ctx = nullcontext()
    if args.profile_dir:
        profiler_ctx = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=args.profile_shapes,
            profile_memory=args.profile_memory,
            with_stack=args.profile_stack,
        )

    rank_rows = []
    density_rows = []
    q_chunk_density_rows = []
    with profiler_ctx as prof:
        for sr in sweep_ranks:
            if sim_sweep:
                if not is_passive:
                    symm_a2a.set_sim_rank(sr)
                    local = populate_for_sim_rank(
                        symm_a2a, full_data, sr,
                        assigned, effective_world, device,
                    )
                # Per-sim_rank publish barrier (matches both ranks; q_handle is
                # fine here because we're matching `symm_a2a.barrier()` on both
                # sides — same handle, same channel.).
                symm_a2a.barrier()
                dist.barrier()

            for _ in range(args.warmup):
                if is_passive:
                    # Match active rank's push_heads_to_seq post_barrier, which
                    # uses recv_handle. Different handles' barriers do NOT pair.
                    symm_a2a.recv_handle.barrier(channel=0)
                else:
                    run_iteration(local, args, sr, effective_world, device)
                dist.barrier()
                if not is_passive:
                    torch.cuda.empty_cache()

            for _ in range(args.iters):
                if is_passive:
                    symm_a2a.recv_handle.barrier(channel=0)
                else:
                    metrics, densities, q_chunk_densities = run_iteration(
                        local, args, sr, effective_world, device,
                    )
                    rank_rows.append(metrics)
                    density_rows.extend(densities)
                    q_chunk_density_rows.extend(q_chunk_densities)
                dist.barrier()
                if args.profile_dir and not is_passive:
                    prof.step()

    if args.profile_dir:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        trace_path = profile_dir / f"rank{rank}.json"
        prof.export_chrome_trace(str(trace_path))

    gathered_rank_rows = [None for _ in range(world_size)] if rank == 0 else None
    gathered_density_rows = [None for _ in range(world_size)] if rank == 0 else None
    gathered_q_chunk_density_rows = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(rank_rows, gathered_rank_rows, dst=0)
    dist.gather_object(density_rows, gathered_density_rows, dst=0)
    dist.gather_object(q_chunk_density_rows, gathered_q_chunk_density_rows, dst=0)

    if rank == 0:
        flat_rank_rows = [row for rank_part in gathered_rank_rows for row in rank_part]
        flat_density_rows = [row for rank_part in gathered_density_rows for row in rank_part]
        flat_q_chunk_density_rows = [row for rank_part in gathered_q_chunk_density_rows for row in rank_part]
        rank_summary = summarize_rank_metrics(flat_rank_rows)
        density_summary = summarize_density(flat_density_rows)
        q_chunk_density_summary = summarize_q_chunk_density(flat_q_chunk_density_rows)

        metadata = local["metadata"]
        print(
            f"capture: layer={metadata.get('layer_idx')} linear_step={metadata.get('linear_step')} "
            f"timestep={metadata.get('timestep')} heads={metadata.get('num_heads')} "
            f"seq_len={metadata.get('seq_len')} world_size={world_size}"
        )
        if args.overlap_qkv_allgather_during_mask:
            print("overlap experiment: async Q/K/V all-gather launched during mask/permutation")
        density_src = "current (ORACLE)" if args.oracle_density else "previous"
        print(f"balance strategy: {args.balance}  density_source: {density_src}")
        if greedy_info is not None:
            pl = greedy_info["predicted_loads"]
            pc = greedy_info["predicted_contig"]
            def _ratio(loads):
                lo = min(loads)
                return float("inf") if lo == 0 else max(loads) / lo
            if pc is None:
                contig_msg = "vs contiguous=N/A (num_heads not divisible by effective_world)"
            else:
                contig_msg = (
                    f"vs contiguous={[round(x, 4) for x in pc]} "
                    f"max/min={_ratio(pc):.3f}"
                )

            print(
                f"greedy prediction ({greedy_info['cost_model']}, "
                f"{greedy_info['density_source']} layer={greedy_info['layer']} "
                f"timestep={greedy_info['timestep']}): "
                f"loads={[round(x, 4) for x in pl]} max/min={_ratio(pl):.3f} "
                f"{contig_msg}"
            )
            # actual per-rank density sum observed this run
            actual = defaultdict(float)
            for row in flat_density_rows:
                actual[int(row["rank"])] += float(row["density"])
            # each rank logs densities per iteration, so average across iters
            iters = max(1, args.iters)
            actual_loads = [actual[r] / iters for r in range(effective_world)]
            print(
                f"actual per-rank density (mean across {iters} iters): "
                f"{[round(x, 4) for x in actual_loads]} max/min={_ratio(actual_loads):.3f}"
            )
            for r, heads in enumerate(greedy_info["assigned"]):
                comm_model = greedy_info.get("comm_cost_model")
                if comm_model is None:
                    print(f"  rank {r} heads: {heads}")
                else:
                    comm_ms = estimate_rank_comm_cost(comm_model, r, len(heads))
                    compute_ms = float(greedy_info["predicted_loads"][r]) - comm_ms
                    print(
                        f"  rank {r} heads: {heads}  predicted compute="
                        f"{compute_ms:.4f} ms comm={comm_ms:.4f} ms total="
                        f"{greedy_info['predicted_loads'][r]:.4f} ms"
                    )
        print_rank_summary(rank_summary)
        print_density_summary(density_summary)
        if args.q_chunk_density_csv:
            print_q_chunk_density_summary(q_chunk_density_summary)

        if args.rank_csv:
            write_csv(Path(args.rank_csv), rank_summary)
            print(f"\nwrote rank CSV: {args.rank_csv}")
        if args.density_csv:
            write_csv(Path(args.density_csv), density_summary)
            print(f"wrote density CSV: {args.density_csv}")
        if args.q_chunk_density_csv:
            write_csv(Path(args.q_chunk_density_csv), q_chunk_density_summary)
            print(f"wrote q-chunk density CSV: {args.q_chunk_density_csv}")
        if args.profile_dir:
            print(f"wrote profiler traces under: {args.profile_dir}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

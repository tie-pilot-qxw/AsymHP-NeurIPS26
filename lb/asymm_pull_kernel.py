"""Triton kernels for asymmetric all2all in sequence-parallel attention (pull forward, push reverse).

Two persistent, peer-swizzled, TMA-descriptor-per-peer kernels:

  `asymm_pull_seq_to_heads`:  forward.  seq-sharded Q/K/V → head-sharded.
      src (peer, symm_mem):  [B, H_total, S_local, D]
      dst (local):           [B, H_local, WORLD * S_local, D]

  `asymm_push_heads_to_seq`:  reverse.  head-sharded attn_out → seq-sharded.
      src (local):           [B, H_local, WORLD * S_local, D]
      dst (peer, symm_mem):  [B, H_total, S_local, D]   (write at global-head row)

The reverse direction is push, not pull. A push lets the producer write
its local result straight into peers' recv buffers, so a single barrier
after the launch is enough to publish the data — no mid-iter sync between
"finish local attn" and "peer reads my output". Cross-rank head ownership
is disjoint, so different ranks write disjoint rows of each peer's
recv_symm — no row-level race.

Sync is the caller's job; see `symm_a2a.SymmAsymA2A` for where the
barriers go.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


_ALLOCATOR_REGISTERED = False


def _tma_allocator(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def ensure_tma_allocator() -> None:
    global _ALLOCATOR_REGISTERED
    if _ALLOCATOR_REGISTERED:
        return
    # Older Triton (e.g. 3.2) has no public allocator hook; its
    # descriptor fallback owns the scratch allocation internally.
    if hasattr(triton, "set_allocator"):
        triton.set_allocator(_tma_allocator)
    _ALLOCATOR_REGISTERED = True


_VALID_BACKENDS = {"auto", "descriptor", "pcie"}


def resolve_asymm_backend(device: torch.device | int | str) -> str:
    """Resolve the asymmetric-copy backend.

    Tensor descriptors map to hardware TMA on Hopper+, but Triton's pre-Hopper
    correctness fallback expands a 2-D descriptor tile into a very large
    scalar-load/store kernel. Use the explicit streaming PCIe kernel there.
    """
    requested = os.environ.get("ASYM_A2A_BACKEND", "auto").strip().lower()
    if requested not in _VALID_BACKENDS:
        raise ValueError(
            f"ASYM_A2A_BACKEND={requested!r}; expected one of "
            f"{sorted(_VALID_BACKENDS)}"
        )
    if requested != "auto":
        return requested
    major, _ = torch.cuda.get_device_capability(device)
    return "descriptor" if major >= 9 else "pcie"


def pcie_launch_config() -> tuple[int, int]:
    block_words = int(os.environ.get("ASYM_A2A_PCIE_BLOCK_WORDS", "256"))
    num_warps = int(os.environ.get("ASYM_A2A_PCIE_NUM_WARPS", "4"))
    if block_words <= 0 or (block_words & (block_words - 1)) != 0:
        raise ValueError(
            "ASYM_A2A_PCIE_BLOCK_WORDS must be a positive power of two, "
            f"got {block_words}"
        )
    if num_warps not in (1, 2, 4, 8):
        raise ValueError(
            "ASYM_A2A_PCIE_NUM_WARPS must be one of 1,2,4,8, "
            f"got {num_warps}"
        )
    return block_words, num_warps


def pcie_peer_mode(world_size: int) -> str:
    requested = os.environ.get("ASYM_A2A_PCIE_PEER_MODE", "auto").strip().lower()
    if requested not in {"auto", "grid", "serial"}:
        raise ValueError(
            "ASYM_A2A_PCIE_PEER_MODE must be one of auto,grid,serial, "
            f"got {requested!r}"
        )
    if requested != "auto":
        return requested
    # On a two-rank link, local and remote traffic coexist without routing
    # ambiguity. With more peers, phase the swizzled routes to avoid making
    # every GPU contend for every PCIe/NUMA path at the same instant.
    return "grid" if world_size <= 2 else "serial"


def _words_per_head(s_local: int, d: int, dtype: torch.dtype) -> int:
    bytes_per_head = s_local * d * torch.empty((), dtype=dtype).element_size()
    if bytes_per_head % 8 != 0:
        raise ValueError(
            "PCIe asymmetric copy requires an 8-byte-aligned per-head slab, "
            f"got {bytes_per_head} bytes"
        )
    return bytes_per_head // 8


@triton.jit
def asymm_pull_seq_to_heads_pcie_kernel(
    peer_ptrs,      # *uint64, [WORLD]
    h_idxs_r,       # *int32, [H_LOCAL]
    recv_ptr,       # destination bits, [B, H_LOCAL, WORLD, WORDS_PER_HEAD]
    B: tl.constexpr,
    H_TOTAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    PEER_OFFSET,
    WORDS_PER_HEAD: tl.constexpr,
    TILES_PER_HEAD: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    """Pre-Hopper PCIe pull: explicit coalesced 64-bit streaming copies.

    Peer is a grid dimension rather than a statically unrolled loop. This
    keeps register lifetime independent of WORLD and lets the scheduler overlap
    traffic to different peers when the topology permits.
    """
    # Tiles go on grid axis 0 (up to 2^31-1 blocks); axis 1 is capped at 65535,
    # which long sequences exceed, so it only carries the (small) peer index.
    linear_tile = tl.program_id(0)
    peer_axis = tl.program_id(1)

    peer = (peer_axis + RANK + PEER_OFFSET) % WORLD
    tiles_per_batch = H_LOCAL * TILES_PER_HEAD
    b = linear_tile // tiles_per_batch
    rem = linear_tile % tiles_per_batch
    lh = rem // TILES_PER_HEAD
    head_tile = rem % TILES_PER_HEAD

    gh = tl.load(h_idxs_r + lh).to(tl.int64)
    peer_base_u64 = tl.load(peer_ptrs + peer)
    src_words = peer_base_u64.to(tl.pointer_type(tl.uint64))
    dst_words = recv_ptr.to(tl.pointer_type(tl.uint64))

    offsets = head_tile * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    mask = offsets < WORDS_PER_HEAD
    src_base = (b.to(tl.int64) * H_TOTAL + gh) * WORDS_PER_HEAD
    dst_base = (
        (b.to(tl.int64) * H_LOCAL + lh.to(tl.int64)) * WORLD
        + peer.to(tl.int64)
    ) * WORDS_PER_HEAD

    data = tl.load(
        src_words + src_base + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        dst_words + dst_base + offsets,
        data,
        mask=mask,
        cache_modifier=".wb",
    )


def _asymm_pull_seq_to_heads_pcie(
    peer_ptrs: torch.Tensor,
    h_idxs_r: torch.Tensor,
    recv: torch.Tensor,
    *,
    b: int,
    h_total: int,
    s_local: int,
    d: int,
    world_size: int,
    rank: int,
) -> None:
    h_local = h_idxs_r.numel()
    words_per_head = _words_per_head(s_local, d, recv.dtype)
    block_words, num_warps = pcie_launch_config()
    tiles_per_head = triton.cdiv(words_per_head, block_words)
    peer_mode = pcie_peer_mode(world_size)
    peer_offsets = range(1, world_size + 1) if peer_mode == "serial" else (1,)
    peer_grid = 1 if peer_mode == "serial" else world_size
    grid = (b * h_local * tiles_per_head, peer_grid)
    for peer_offset in peer_offsets:
        asymm_pull_seq_to_heads_pcie_kernel[grid](
            peer_ptrs,
            h_idxs_r,
            recv,
            B=b,
            H_TOTAL=h_total,
            H_LOCAL=h_local,
            WORLD=world_size,
            RANK=rank,
            PEER_OFFSET=peer_offset,
            WORDS_PER_HEAD=words_per_head,
            TILES_PER_HEAD=tiles_per_head,
            BLOCK_WORDS=block_words,
            num_warps=num_warps,
            num_stages=1,
        )


@triton.jit
def asymm_pull_seq_to_heads_kernel(
    peer_ptrs,      # *uint64, [WORLD]
    h_idxs_r,       # *int32, [H_LOCAL]
    recv_ptr,       # fp16/bf16 base pointer of [B, H_LOCAL, WORLD*S_LOCAL, D]
    B: tl.constexpr,
    H_TOTAL: tl.constexpr,
    S_LOCAL: tl.constexpr,
    D: tl.constexpr,
    H_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    S_BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DTYPE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    S_TILES: tl.constexpr = S_LOCAL // S_BLOCK
    T_PER_PEER: tl.constexpr = B * H_LOCAL * S_TILES

    recv_desc = tl.make_tensor_descriptor(
        recv_ptr,
        shape=[B, H_LOCAL, WORLD * S_LOCAL, D],
        strides=[H_LOCAL * WORLD * S_LOCAL * D, WORLD * S_LOCAL * D, D, 1],
        block_shape=[1, 1, S_BLOCK, D],
    )

    # Outer peer loop fully unrolled: descriptor built once per peer, not per tile.
    for peer_axis in tl.static_range(WORLD):
        peer = (peer_axis + RANK + 1) % WORLD  # swizzle: rank r starts at r+1
        src_ptr_u64 = tl.load(peer_ptrs + peer)
        src_ptr = src_ptr_u64.to(tl.pointer_type(DTYPE))
        src_desc = tl.make_tensor_descriptor(
            src_ptr,
            shape=[B, H_TOTAL, S_LOCAL, D],
            strides=[H_TOTAL * S_LOCAL * D, S_LOCAL * D, D, 1],
            block_shape=[1, 1, S_BLOCK, D],
        )

        # Inner: CTA strides through tiles within this peer. tl.range with
        # num_stages software-pipelines the TMA load->store (prefetch next tile).
        for tile in tl.range(pid, T_PER_PEER, NUM_SMS, num_stages=NUM_STAGES):
            b = tile // (H_LOCAL * S_TILES)
            rem = tile % (H_LOCAL * S_TILES)
            lh = rem // S_TILES
            s_blk_in_peer = rem % S_TILES

            gh = tl.load(h_idxs_r + lh).to(tl.int32)
            data = src_desc.load([b, gh, s_blk_in_peer * S_BLOCK, 0])

            s_global = peer * S_LOCAL + s_blk_in_peer * S_BLOCK
            recv_desc.store([b, lh, s_global, 0], data)


_TRITON_DTYPE = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def asymm_pull_seq_to_heads(
    peer_ptrs: torch.Tensor,    # [WORLD] uint64 on device
    h_idxs_r: torch.Tensor,     # [H_LOCAL] int32 on device
    recv: torch.Tensor,         # [B, H_LOCAL, WORLD*S_LOCAL, D]
    b: int,
    h_total: int,
    s_local: int,
    d: int,
    world_size: int,
    rank: int,
    s_block: int = 128,
    num_sms: int | None = None,
    num_stages: int = 2,
    num_warps: int = 4,
) -> None:
    """Launch the pull kernel. Caller must barrier before and after.

    Tuned defaults (from a BW sweep on H100): num_sms = 2x SM count
    (over-subscribe to hide TMA latency), num_stages=2 (software-pipeline
    load->store), s_block up to 256 when divisible. ~+13% over 1-CTA/SM,
    s_block=128, no-pipeline.
    """
    if recv.dtype not in _TRITON_DTYPE:
        raise ValueError(f"Unsupported dtype {recv.dtype}")
    if resolve_asymm_backend(recv.device) == "pcie":
        _asymm_pull_seq_to_heads_pcie(
            peer_ptrs,
            h_idxs_r,
            recv,
            b=b,
            h_total=h_total,
            s_local=s_local,
            d=d,
            world_size=world_size,
            rank=rank,
        )
        return

    ensure_tma_allocator()
    dtype = _TRITON_DTYPE[recv.dtype]

    h_local = h_idxs_r.numel()
    if s_local % s_block != 0:
        raise ValueError(f"s_local={s_local} not divisible by s_block={s_block}")

    if num_sms is None:
        num_sms = 2 * torch.cuda.get_device_properties(recv.device).multi_processor_count

    grid = (num_sms,)
    asymm_pull_seq_to_heads_kernel[grid](
        peer_ptrs,
        h_idxs_r,
        recv,
        B=b,
        H_TOTAL=h_total,
        S_LOCAL=s_local,
        D=d,
        H_LOCAL=h_local,
        WORLD=world_size,
        RANK=rank,
        S_BLOCK=s_block,
        NUM_SMS=num_sms,
        DTYPE=dtype,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )


@triton.jit
def asymm_push_heads_to_seq_pcie_kernel(
    peer_ptrs,      # *uint64, [WORLD]
    h_idxs_r,       # *int32, [H_LOCAL]
    src_ptr,        # source bits, [B, H_LOCAL, WORLD, WORDS_PER_HEAD]
    B: tl.constexpr,
    H_TOTAL: tl.constexpr,
    H_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    PEER_OFFSET,
    WORDS_PER_HEAD: tl.constexpr,
    TILES_PER_HEAD: tl.constexpr,
    BLOCK_WORDS: tl.constexpr,
):
    """Pre-Hopper PCIe push: explicit coalesced 64-bit streaming copies."""
    # Tiles go on grid axis 0 (up to 2^31-1 blocks); axis 1 is capped at 65535,
    # which long sequences exceed, so it only carries the (small) peer index.
    linear_tile = tl.program_id(0)
    peer_axis = tl.program_id(1)

    peer = (peer_axis + RANK + PEER_OFFSET) % WORLD
    tiles_per_batch = H_LOCAL * TILES_PER_HEAD
    b = linear_tile // tiles_per_batch
    rem = linear_tile % tiles_per_batch
    lh = rem // TILES_PER_HEAD
    head_tile = rem % TILES_PER_HEAD

    gh = tl.load(h_idxs_r + lh).to(tl.int64)
    dst_base_u64 = tl.load(peer_ptrs + peer)
    dst_words = dst_base_u64.to(tl.pointer_type(tl.uint64))
    src_words = src_ptr.to(tl.pointer_type(tl.uint64))

    offsets = head_tile * BLOCK_WORDS + tl.arange(0, BLOCK_WORDS)
    mask = offsets < WORDS_PER_HEAD
    src_base = (
        (b.to(tl.int64) * H_LOCAL + lh.to(tl.int64)) * WORLD
        + peer.to(tl.int64)
    ) * WORDS_PER_HEAD
    dst_base = (b.to(tl.int64) * H_TOTAL + gh) * WORDS_PER_HEAD

    data = tl.load(
        src_words + src_base + offsets,
        mask=mask,
        other=0,
    )
    tl.store(
        dst_words + dst_base + offsets,
        data,
        mask=mask,
        cache_modifier=".wb",
    )


def _asymm_push_heads_to_seq_pcie(
    peer_ptrs: torch.Tensor,
    h_idxs_r: torch.Tensor,
    src: torch.Tensor,
    *,
    b: int,
    h_total: int,
    s_local: int,
    d: int,
    world_size: int,
    rank: int,
) -> None:
    h_local = h_idxs_r.numel()
    words_per_head = _words_per_head(s_local, d, src.dtype)
    block_words, num_warps = pcie_launch_config()
    tiles_per_head = triton.cdiv(words_per_head, block_words)
    peer_mode = pcie_peer_mode(world_size)
    peer_offsets = range(1, world_size + 1) if peer_mode == "serial" else (1,)
    peer_grid = 1 if peer_mode == "serial" else world_size
    grid = (b * h_local * tiles_per_head, peer_grid)
    for peer_offset in peer_offsets:
        asymm_push_heads_to_seq_pcie_kernel[grid](
            peer_ptrs,
            h_idxs_r,
            src,
            B=b,
            H_TOTAL=h_total,
            H_LOCAL=h_local,
            WORLD=world_size,
            RANK=rank,
            PEER_OFFSET=peer_offset,
            WORDS_PER_HEAD=words_per_head,
            TILES_PER_HEAD=tiles_per_head,
            BLOCK_WORDS=block_words,
            num_warps=num_warps,
            num_stages=1,
        )


@triton.jit
def asymm_push_heads_to_seq_kernel(
    peer_ptrs,      # *uint64, [WORLD] — peers' recv_symm base addresses
    h_idxs_r,       # *int32, [H_LOCAL] — this rank's global head indices
    src_ptr,        # local src [B, H_LOCAL, WORLD*S_LOCAL, D]
    B: tl.constexpr,
    H_TOTAL: tl.constexpr,
    S_LOCAL: tl.constexpr,
    D: tl.constexpr,
    H_LOCAL: tl.constexpr,
    WORLD: tl.constexpr,
    RANK: tl.constexpr,
    S_BLOCK: tl.constexpr,
    NUM_SMS: tl.constexpr,
    DTYPE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    S_TILES: tl.constexpr = S_LOCAL // S_BLOCK
    T_PER_PEER: tl.constexpr = B * H_LOCAL * S_TILES

    src_desc = tl.make_tensor_descriptor(
        src_ptr,
        shape=[B, H_LOCAL, WORLD * S_LOCAL, D],
        strides=[H_LOCAL * WORLD * S_LOCAL * D, WORLD * S_LOCAL * D, D, 1],
        block_shape=[1, 1, S_BLOCK, D],
    )

    # Mirror image of the forward pull: same swizzle, descriptor-per-peer.
    # Each rank writes its H_LOCAL global-head rows into every peer's recv_symm,
    # at peer-local S range [0, S_LOCAL). Cross-rank heads are disjoint, so
    # different ranks' writes land on disjoint rows of the same recv_symm.
    for peer_axis in tl.static_range(WORLD):
        peer = (peer_axis + RANK + 1) % WORLD
        dst_ptr_u64 = tl.load(peer_ptrs + peer)
        dst_ptr = dst_ptr_u64.to(tl.pointer_type(DTYPE))
        dst_desc = tl.make_tensor_descriptor(
            dst_ptr,
            shape=[B, H_TOTAL, S_LOCAL, D],
            strides=[H_TOTAL * S_LOCAL * D, S_LOCAL * D, D, 1],
            block_shape=[1, 1, S_BLOCK, D],
        )

        for tile in tl.range(pid, T_PER_PEER, NUM_SMS, num_stages=NUM_STAGES):
            b = tile // (H_LOCAL * S_TILES)
            rem = tile % (H_LOCAL * S_TILES)
            lh = rem // S_TILES
            s_blk_in_peer = rem % S_TILES

            gh = tl.load(h_idxs_r + lh).to(tl.int32)
            s_in_src = peer * S_LOCAL + s_blk_in_peer * S_BLOCK
            data = src_desc.load([b, lh, s_in_src, 0])
            dst_desc.store([b, gh, s_blk_in_peer * S_BLOCK, 0], data)


def asymm_push_heads_to_seq(
    peer_ptrs: torch.Tensor,    # [WORLD] uint64 on device
    h_idxs_r: torch.Tensor,     # [H_LOCAL] int32 on device
    src: torch.Tensor,          # [B, H_LOCAL, WORLD*S_LOCAL, D] (local)
    b: int,
    h_total: int,
    s_local: int,
    d: int,
    world_size: int,
    rank: int,
    s_block: int = 128,
    num_sms: int | None = None,
    num_stages: int = 2,
    num_warps: int = 4,
) -> None:
    """Launch the push kernel. Caller must barrier before and after.
    Tuned defaults: num_sms=2x SM (over-subscribe), num_stages=2 (pipeline)."""
    if src.dtype not in _TRITON_DTYPE:
        raise ValueError(f"Unsupported dtype {src.dtype}")
    if resolve_asymm_backend(src.device) == "pcie":
        _asymm_push_heads_to_seq_pcie(
            peer_ptrs,
            h_idxs_r,
            src,
            b=b,
            h_total=h_total,
            s_local=s_local,
            d=d,
            world_size=world_size,
            rank=rank,
        )
        return

    ensure_tma_allocator()
    dtype = _TRITON_DTYPE[src.dtype]

    h_local = h_idxs_r.numel()
    if s_local % s_block != 0:
        raise ValueError(f"s_local={s_local} not divisible by s_block={s_block}")

    if num_sms is None:
        num_sms = 2 * torch.cuda.get_device_properties(src.device).multi_processor_count

    grid = (num_sms,)
    asymm_push_heads_to_seq_kernel[grid](
        peer_ptrs,
        h_idxs_r,
        src,
        B=b,
        H_TOTAL=h_total,
        S_LOCAL=s_local,
        D=d,
        H_LOCAL=h_local,
        WORLD=world_size,
        RANK=rank,
        S_BLOCK=s_block,
        NUM_SMS=num_sms,
        DTYPE=dtype,
        NUM_STAGES=num_stages,
        num_warps=num_warps,
    )

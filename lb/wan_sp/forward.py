"""Sequence-parallel DiT forward for Wan T2V.

Mirrors ``WanTransformer3DModel_Sparse.forward`` but shards the token sequence
across ranks right after patch embedding and all-gathers it back just before
the output norm. Between those points every block (self-attn via the SP
all2all processor, plus the per-token cross-attn / FFN / norms) runs on this
rank's ``S_local = S_full / W`` tokens, so the whole DiT is sequence-parallel
and only self-attention communicates (inside the processor).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed as dist
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers

from svg.logger import logger
from svg.models.wan.attention import ENABLE_FAST_KERNEL

from .context import get_sp_context


def _shard_seq(x: torch.Tensor, rank: int, world: int, dim: int) -> torch.Tensor:
    s = x.shape[dim]
    assert s % world == 0, (
        f"sequence length {s} not divisible by world_size {world}; "
        f"pick a world size dividing the token count (frames*frame_patches)."
    )
    return x.chunk(world, dim=dim)[rank].contiguous()


def _gather_seq(x: torch.Tensor, world: int, dim: int, group) -> torch.Tensor:
    parts = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(parts, x.contiguous(), group=group)
    return torch.cat(parts, dim=dim)


def wan_sp_dit_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    ctx = get_sp_context()
    rank, world, group = ctx.rank, ctx.world_size, ctx.group

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    elif attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
        logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    s_full = post_patch_num_frames * post_patch_height * post_patch_width
    # asymm TMA needs s_local a multiple of 128; pad the sequence to match
    # (the pad is trimmed before local attention and dropped after the gather).
    from .context import sp_asymm_s_local
    s_pad = s_full
    if world > 1 and ctx.a2a_backend == "asymm":
        s_pad = sp_asymm_s_local(s_full, world, align=128) * world
    pad_len = s_pad - s_full

    def _pad_seq(x, dim):
        if pad_len == 0:
            return x
        shape = list(x.shape); shape[dim] = pad_len
        return torch.cat([x, x.new_zeros(shape)], dim=dim)

    rotary_emb = self.rope(hidden_states)

    if ENABLE_FAST_KERNEL:
        rot_real = rotary_emb.real.squeeze(0).squeeze(0).contiguous().to(torch.float32)
        rot_imag = rotary_emb.imag.squeeze(0).squeeze(0).contiguous().to(torch.float32)
        # rot_{real,imag}: [S_full, D_head/2] -> pad then shard the sequence (dim 0).
        if world > 1:
            rot_real = _shard_seq(_pad_seq(rot_real, 0), rank, world, dim=0)
            rot_imag = _shard_seq(_pad_seq(rot_imag, 0), rank, world, dim=0)
        rotary_emb = (rot_real, rot_imag)
    elif world > 1:
        # complex freqs [1, 1, S_full, ...] -> pad then shard on the sequence dim (2).
        rotary_emb = _shard_seq(_pad_seq(rotary_emb, 2), rank, world, dim=2)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()  # [B, S_full, hidden]

    # ---- SP: pad (asymm alignment) then shard the token sequence across ranks ----
    if world > 1:
        hidden_states = _shard_seq(_pad_seq(hidden_states, 1), rank, world, dim=1)  # [B, S_local, hidden]

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    # 4. Transformer blocks (on the local shard).
    for block in self.blocks:
        hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, timestep=timestep)

    # ---- SP: all-gather the sequence back to full length, then trim any pad ----
    if world > 1:
        hidden_states = _gather_seq(hidden_states, world, dim=1, group=group)  # [B, S_pad, hidden]
        if pad_len:
            hidden_states = hidden_states[:, :s_full, :].contiguous()  # drop padding tokens

    # 5. Output norm, projection & unpatchify.
    shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)

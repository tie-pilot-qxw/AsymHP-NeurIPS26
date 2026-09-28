"""Shared semantic-aware-permutation (SAP) state for load-balancing tools.

Both `bench_sp_all2all_attention.py` and `profile_maskgen_aware_cost.py` need
to run the same kmeans + identify_dynamic_map + permute pipeline that the
WAN / Hunyuan SAP attention processors run. This module is the single source
of truth for that pipeline so the two tools cannot drift.

Two paths, selected by `model_type`:

- "wan": kmeans over the full Q/K/V, identify_dynamic_map, permute. Same
  behavior as `WanAttn_SAPAttn_Processor.semantic_aware_permutation`
  (svg/models/wan/attention.py).

- "hunyuan": only the video portion `[:, :, :video_length, :]` goes into
  kmeans. Then `dynamic_map_post_processing` extends the result back over
  the full sequence by appending two synthetic clusters (prompt,
  unprompt) plus the connectivity rules from
  `Hunyuan_SAPAttn_Processor2_0.dynamic_map_post_processing`
  (svg/models/hyvideo/attention.py:659-703). The returned q/k/v cover
  the full seq_len with the video region permuted in place; dyn_map and
  cluster sizes have +2 entries on each side (prompt, unprompt).

Return contract (9-tuple, identical across model_type):
    (q_perm, k_perm, v_perm, dyn_map, qc_sz, kc_sz,
     q_sorted_indices, qlabels, klabels)

For hunyuan, qlabels/klabels cover only the video region (length =
video_length), while q_perm/k_perm/v_perm/q_sorted_indices cover the
full sequence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from svg.kernels.triton.permute import permute_tensor_by_labels_triton
from svg.kmeans_utils import batch_kmeans_Euclid, identify_dynamic_map


@dataclass
class SAPState:
    model_type: str  # "wan" | "hunyuan"
    num_q_centroids: int
    num_k_centroids: int
    top_p_kmeans: float
    min_kc_ratio: float
    kmeans_iter_step: int
    q_centroids: torch.Tensor
    k_centroids: torch.Tensor
    # Hunyuan-only; required when model_type == "hunyuan".
    video_length: Optional[int] = None     # num_frame * frame_size
    context_length: Optional[int] = None   # full text region (prompt + unprompt)
    prompt_length: Optional[int] = None    # real prompt subset of context

    def __post_init__(self):
        if self.model_type not in ("wan", "hunyuan"):
            raise ValueError(f"unknown model_type: {self.model_type!r}")
        if self.model_type == "hunyuan":
            for name in ("video_length", "context_length", "prompt_length"):
                if getattr(self, name) is None:
                    raise ValueError(f"hunyuan SAPState requires {name}")

    def semantic_aware_permutation(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        if self.model_type == "wan":
            return self._sap_wan(query, key, value)
        return self._sap_hunyuan(query, key, value)

    def _sap_wan(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        cfg, num_heads, seq_len, dim = query.size()
        qlabels, qcentroids, qcluster_sizes, _ = batch_kmeans_Euclid(
            query.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, _ = batch_kmeans_Euclid(
            key.view(cfg * num_heads, seq_len, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )

        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)
        dynamic_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        q_permuted, q_sorted_indices = permute_tensor_by_labels_triton(query, qlabels, dim=2)
        k_permuted, k_sorted_indices = permute_tensor_by_labels_triton(key, klabels, dim=2)
        v_permuted, _ = permute_tensor_by_labels_triton(value, klabels, dim=2, sorted_indices=k_sorted_indices)

        return (
            q_permuted,
            k_permuted,
            v_permuted,
            dynamic_map,
            q_cluster_sizes,
            k_cluster_sizes,
            q_sorted_indices,
            qlabels,
            klabels,
        )

    def _sap_hunyuan(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        cfg, num_heads, seq_len, dim = query.size()
        video_length = int(self.video_length)
        context_length = int(self.context_length)
        prompt_length = int(self.prompt_length)
        unprompt_length = context_length - prompt_length
        if seq_len != video_length + context_length:
            raise ValueError(
                f"hunyuan SAP expects seq_len == video_length + context_length "
                f"(got seq_len={seq_len}, video={video_length}, context={context_length})"
            )

        # 1. Kmeans + permute over the video portion only.
        q_video = query[:, :, :video_length, :].contiguous()
        k_video = key[:, :, :video_length, :].contiguous()
        v_video = value[:, :, :video_length, :].contiguous()

        qlabels, qcentroids, qcluster_sizes, _ = batch_kmeans_Euclid(
            q_video.view(cfg * num_heads, video_length, dim),
            n_clusters=self.num_q_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.q_centroids,
        )
        klabels, kcentroids, kcluster_sizes, _ = batch_kmeans_Euclid(
            k_video.view(cfg * num_heads, video_length, dim),
            n_clusters=self.num_k_centroids,
            max_iters=self.kmeans_iter_step,
            init_centroids=self.k_centroids,
        )

        q_cluster_sizes = qcluster_sizes.view(cfg, num_heads, self.num_q_centroids)
        k_cluster_sizes = kcluster_sizes.view(cfg, num_heads, self.num_k_centroids)
        dyn_map = identify_dynamic_map(
            qcentroids.view(cfg, num_heads, self.num_q_centroids, dim),
            kcentroids.view(cfg, num_heads, self.num_k_centroids, dim),
            q_cluster_sizes,
            k_cluster_sizes,
            self.top_p_kmeans,
            self.min_kc_ratio,
        )

        q_perm_video, q_sorted_indices = permute_tensor_by_labels_triton(q_video, qlabels, dim=2)
        k_perm_video, k_sorted_indices = permute_tensor_by_labels_triton(k_video, klabels, dim=2)
        v_perm_video, _ = permute_tensor_by_labels_triton(
            v_video, klabels, dim=2, sorted_indices=k_sorted_indices
        )

        # 2. Post-processing: write video permutation back into a full-seq
        # tensor and extend dyn_map / cluster sizes by 2 synthetic clusters
        # (prompt, unprompt) per the Hunyuan rules:
        #   - prompt cluster (-2) attends everything except unprompt
        #   - everything attends prompt cluster
        #   - unprompt cluster (-1) only attends itself
        # This is a faithful port of
        # Hunyuan_SAPAttn_Processor2_0.dynamic_map_post_processing
        # (svg/models/hyvideo/attention.py:659-703).
        q_full = query.clone()
        k_full = key.clone()
        v_full = value.clone()
        q_full[:, :, :video_length, :] = q_perm_video
        k_full[:, :, :video_length, :] = k_perm_video
        v_full[:, :, :video_length, :] = v_perm_video

        dyn_map = F.pad(dyn_map, (0, 2, 0, 2), value=0)
        dyn_map[:, :, -2, :-1] = True
        dyn_map[:, :, :-1, -2] = True
        dyn_map[:, :, -1, -1] = True

        q_cluster_sizes = F.pad(q_cluster_sizes, (0, 2), value=0)
        q_cluster_sizes[:, :, -2] = prompt_length
        q_cluster_sizes[:, :, -1] = unprompt_length
        k_cluster_sizes = F.pad(k_cluster_sizes, (0, 2), value=0)
        k_cluster_sizes[:, :, -2] = prompt_length
        k_cluster_sizes[:, :, -1] = unprompt_length

        q_sorted_indices = F.pad(q_sorted_indices, (0, context_length), value=0)
        q_sorted_indices[..., video_length:] = torch.arange(
            video_length, video_length + context_length, device=q_sorted_indices.device
        )

        return (
            q_full,
            k_full,
            v_full,
            dyn_map,
            q_cluster_sizes,
            k_cluster_sizes,
            q_sorted_indices,
            qlabels,
            klabels,
        )


def make_sap_state(
    metadata: Dict,
    q_centroids: torch.Tensor,
    k_centroids: torch.Tensor,
    top_p_override: Optional[float] = None,
    min_kc_ratio_override: Optional[float] = None,
) -> SAPState:
    """Construct a SAPState from a dump's metadata dict.

    `model_type` defaults to "wan" if missing (legacy WAN-only dumps).
    For hunyuan dumps, `video_length` is derived from
    `num_frame * frame_size` if not present directly.

    `top_p_override` / `min_kc_ratio_override`: when set, replace the dump's
    sparsity knob so the SAME dumped Q/K/V can be replayed at a different
    sparsity level (used for the measured sparsity-sensitivity sweep -- only the
    sparsity knob changes, the input is fixed).
    """
    model_type = str(metadata.get("model_type", "wan"))
    kwargs = dict(
        model_type=model_type,
        num_q_centroids=int(metadata["num_q_centroids"]),
        num_k_centroids=int(metadata["num_k_centroids"]),
        top_p_kmeans=float(top_p_override if top_p_override is not None else metadata["top_p_kmeans"]),
        min_kc_ratio=float(min_kc_ratio_override if min_kc_ratio_override is not None else metadata["min_kc_ratio"]),
        kmeans_iter_step=int(metadata["kmeans_iter_step"]),
        q_centroids=q_centroids,
        k_centroids=k_centroids,
    )
    if model_type == "hunyuan":
        if "video_length" in metadata:
            video_length = int(metadata["video_length"])
        else:
            video_length = int(metadata["num_frame"]) * int(metadata["frame_size"])
        kwargs.update(
            video_length=video_length,
            context_length=int(metadata["context_length"]),
            prompt_length=int(metadata["prompt_length"]),
        )
    return SAPState(**kwargs)

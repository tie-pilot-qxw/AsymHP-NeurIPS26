import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch


_EXPORT_COUNT = 0
_ATTN_EXPORT_COUNT = 0
_LAST_WAN_TIMESTEP = None
_WAN_LINEAR_STEP_INDEX = -1


def _get_numbered_export_path(raw_path: Optional[str], index: int, default_filename: str) -> Optional[Path]:
    if not raw_path:
        return None

    path = Path(raw_path)
    if path.suffix:
        if index == 0:
            return path
        return path.with_name(f"{path.stem}_{index:04d}{path.suffix}")

    return path / default_filename.format(index=index)


def _get_export_path(index: int) -> Optional[Path]:
    return _get_numbered_export_path(
        os.getenv("SVG_WAN_SAP_EXPORT_PATH"),
        index,
        "wan_semantic_aware_permutation_input_{index:04d}.pt",
    )


def _get_attention_export_path(index: int) -> Optional[Path]:
    return _get_numbered_export_path(
        os.getenv("SVG_WAN_ATTN_EXPORT_PATH"),
        index,
        "wan_attention_core_input_{index:04d}.pt",
    )


def _get_max_exports() -> int:
    try:
        return max(0, int(os.getenv("SVG_WAN_SAP_EXPORT_MAX", "1")))
    except ValueError:
        return 1


def _get_attention_max_exports() -> int:
    try:
        return max(0, int(os.getenv("SVG_WAN_ATTN_EXPORT_MAX", "1")))
    except ValueError:
        return 1


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _tensor_scalar_to_int(value: Optional[torch.Tensor]) -> Optional[int]:
    if value is None:
        return None

    if torch.is_tensor(value):
        if value.numel() == 0:
            return None
        return int(value.detach().flatten()[0].item())

    return int(value)


def _matches_optional_int_filter(env_name: str, actual: Optional[int]) -> bool:
    raw_value = os.getenv(env_name)
    if raw_value is None or raw_value == "":
        return True
    if actual is None:
        return False

    return actual == int(raw_value)


def get_wan_sap_linear_step(timestep: Optional[torch.Tensor]) -> Optional[int]:
    global _LAST_WAN_TIMESTEP, _WAN_LINEAR_STEP_INDEX

    timestep_value = _tensor_scalar_to_int(timestep)
    if timestep_value is None:
        return None

    if timestep_value != _LAST_WAN_TIMESTEP:
        _WAN_LINEAR_STEP_INDEX += 1
        _LAST_WAN_TIMESTEP = timestep_value

    return _WAN_LINEAR_STEP_INDEX + 1


def _tensor_for_save(tensor: Optional[torch.Tensor], cuda_env_name: str = "SVG_WAN_SAP_EXPORT_CUDA_TENSORS") -> Optional[torch.Tensor]:
    if tensor is None:
        return None

    detached = tensor.detach()
    if _env_flag(cuda_env_name):
        return detached.clone()

    return detached.cpu()


def _cache_state(processor: Any, cuda_env_name: str = "SVG_WAN_SAP_EXPORT_CUDA_TENSORS") -> Dict[str, Any]:
    # Wan stores q/k_centroids as a single tensor on the per-layer processor
    # instance. Hunyuan stores them as a class-level dict keyed by layer_idx
    # (one entry per layer). `centroids_init` follows the same convention.
    # Pull this layer's slice so downstream consumers see a tensor either way.
    layer_idx = getattr(processor, "layer_idx", None)

    def _per_layer(attr_name: str):
        v = getattr(processor, attr_name, None)
        if isinstance(v, dict):
            return v.get(layer_idx)
        return v

    init = _per_layer("centroids_init")
    return {
        "centroids_init": bool(init) if init is not None else False,
        "q_centroids": _tensor_for_save(_per_layer("q_centroids"), cuda_env_name),
        "k_centroids": _tensor_for_save(_per_layer("k_centroids"), cuda_env_name),
    }


def _optional_tensor_payload(value: Any, cuda_env_name: str) -> Any:
    if torch.is_tensor(value):
        return _tensor_for_save(value, cuda_env_name)
    if isinstance(value, (list, tuple)):
        converted = [_optional_tensor_payload(item, cuda_env_name) for item in value]
        return type(value)(converted)
    return None


def _processor_tensor_state(processor: Any, cuda_env_name: str) -> Dict[str, Any]:
    return {
        "attention_masks": _optional_tensor_payload(getattr(processor, "attention_masks", None), cuda_env_name),
    }


def maybe_export_wan_semantic_aware_permutation_inputs(
    processor: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    timestep: Optional[torch.Tensor] = None,
    linear_step: Optional[int] = None,
) -> Optional[Path]:
    global _EXPORT_COUNT

    path = _get_export_path(_EXPORT_COUNT)
    if path is None or _EXPORT_COUNT >= _get_max_exports():
        return None

    timestep_value = _tensor_scalar_to_int(timestep)
    if not _matches_optional_int_filter("SVG_WAN_SAP_EXPORT_LAYER", int(processor.layer_idx)):
        return None
    if not _matches_optional_int_filter("SVG_WAN_SAP_EXPORT_STEP", linear_step):
        return None

    if _env_flag("SVG_WAN_SAP_EXPORT_REQUIRE_CACHE", "0") and not getattr(processor, "centroids_init", False):
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    cfg, num_heads, seq_len, head_dim = query.shape

    payload = {
        "metadata": {
            "target": "WanAttn_SAPAttn_Processor.semantic_aware_permutation",
            "layer_idx": processor.layer_idx,
            "linear_step": linear_step,
            "timestep": timestep_value,
            "cfg": cfg,
            "num_heads": num_heads,
            "seq_len": seq_len,
            "head_dim": head_dim,
            "num_q_centroids": processor.num_q_centroids,
            "num_k_centroids": processor.num_k_centroids,
            "top_p_kmeans": processor.top_p_kmeans,
            "min_kc_ratio": processor.min_kc_ratio,
            "kmeans_iter_init": processor.kmeans_iter_init,
            "kmeans_iter_step": processor.kmeans_iter_step,
            "export_index": _EXPORT_COUNT,
            "note": "Inputs captured at function entry before kmeans_clustering, dynamic_map generation, and permute.",
        },
        "inputs": {
            "query": _tensor_for_save(query),
            "key": _tensor_for_save(key),
            "value": _tensor_for_save(value),
        },
        "centroid_cache": _cache_state(processor),
    }

    torch.save(payload, path)
    _EXPORT_COUNT += 1
    print(f"[SVG Wan SAP export] saved semantic_aware_permutation input to {path}")
    return path


_ATTN_EXPORT_TARGETS = {
    "wan": "WanAttn_SAPAttn_Processor.attention_core_logic",
    "hunyuan": "Hunyuan_SAPAttn_Processor2_0.attention_core_logic",
}


def maybe_export_attention_core_inputs(
    processor: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    timestep: Optional[torch.Tensor],
    linear_step: Optional[int],
    full_attention_flag: bool,
    model_type: str = "wan",
) -> Optional[Path]:
    """Export QKV + centroid cache + metadata at SAP attention-core entry.

    Used by both WAN and Hunyuan SAP processors. `model_type` selects the
    metadata fields written and the target string. Hunyuan dumps additionally
    carry `prompt_length` (the real-prompt length within the text context).

    Env vars (shared, historically wan-prefixed):
        SVG_WAN_ATTN_EXPORT_PATH       — output file or directory
        SVG_WAN_ATTN_EXPORT_MAX        — max number of exports per process
        SVG_WAN_ATTN_EXPORT_LAYER/STEP/TIMESTEP — optional filters
        SVG_WAN_ATTN_EXPORT_REQUIRE_CACHE — only export after kmeans init
        SVG_WAN_ATTN_EXPORT_CUDA_TENSORS — keep tensors on GPU instead of CPU
    """
    global _ATTN_EXPORT_COUNT

    if model_type not in _ATTN_EXPORT_TARGETS:
        raise ValueError(f"unknown model_type for attention-core export: {model_type!r}")

    path = _get_attention_export_path(_ATTN_EXPORT_COUNT)
    if path is None or _ATTN_EXPORT_COUNT >= _get_attention_max_exports():
        return None

    timestep_value = _tensor_scalar_to_int(timestep)
    if not _matches_optional_int_filter("SVG_WAN_ATTN_EXPORT_LAYER", int(processor.layer_idx)):
        return None
    if not _matches_optional_int_filter("SVG_WAN_ATTN_EXPORT_STEP", linear_step):
        return None
    if not _matches_optional_int_filter("SVG_WAN_ATTN_EXPORT_TIMESTEP", timestep_value):
        return None

    if _env_flag("SVG_WAN_ATTN_EXPORT_REQUIRE_CACHE", "0"):
        # See _cache_state: hunyuan keeps centroids_init as a per-layer dict.
        ci = getattr(processor, "centroids_init", False)
        if isinstance(ci, dict):
            ci = ci.get(getattr(processor, "layer_idx", None), False)
        if not ci:
            return None

    path.parent.mkdir(parents=True, exist_ok=True)
    cfg, num_heads, seq_len, head_dim = query.shape
    cuda_env_name = "SVG_WAN_ATTN_EXPORT_CUDA_TENSORS"

    metadata = {
        "target": _ATTN_EXPORT_TARGETS[model_type],
        "model_type": model_type,
        "layer_idx": processor.layer_idx,
        "linear_step": linear_step,
        "timestep": timestep_value,
        "cfg": cfg,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "full_attention_flag": bool(full_attention_flag),
        "context_length": processor.context_length,
        "num_frame": processor.num_frame,
        "frame_size": processor.frame_size,
        "first_layers_fp": processor.first_layers_fp,
        "first_times_fp": processor.first_times_fp,
        "zero_step_kmeans_init": processor.zero_step_kmeans_init,
        "num_q_centroids": processor.num_q_centroids,
        "num_k_centroids": processor.num_k_centroids,
        "top_p_kmeans": processor.top_p_kmeans,
        "min_kc_ratio": processor.min_kc_ratio,
        "kmeans_iter_init": processor.kmeans_iter_init,
        "kmeans_iter_step": processor.kmeans_iter_step,
        "export_index": _ATTN_EXPORT_COUNT,
        "note": "Inputs captured at attention_core_logic entry after full_attention_flag is computed.",
    }
    if model_type == "hunyuan":
        # Required by lb/sap_state.py SAPState hunyuan path.
        metadata["prompt_length"] = int(processor.prompt_length)

    payload = {
        "metadata": metadata,
        "inputs": {
            "query": _tensor_for_save(query, cuda_env_name),
            "key": _tensor_for_save(key, cuda_env_name),
            "value": _tensor_for_save(value, cuda_env_name),
            "timestep": _tensor_for_save(timestep, cuda_env_name),
        },
        "centroid_cache": _cache_state(processor, cuda_env_name),
        "processor_tensor_state": _processor_tensor_state(processor, cuda_env_name),
    }

    torch.save(payload, path)
    _ATTN_EXPORT_COUNT += 1
    print(f"[SVG attention export/{model_type}] saved attention_core_logic input to {path}")
    return path


# Back-compat alias: the original wan-specific name still resolves.
maybe_export_wan_attention_core_inputs = maybe_export_attention_core_inputs

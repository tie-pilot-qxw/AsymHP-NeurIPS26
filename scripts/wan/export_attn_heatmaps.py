#!/usr/bin/env python
"""Export per-head attention-score and selected-mask heatmaps for a WAN SAP capture.

For each requested head, writes two PNGs under --out-dir:
  head{H:02d}_score_{order}.png  : block-pooled softmax(QK^T/sqrt(D)) heatmap
  head{H:02d}_mask_{order}.png   : block-pooled selected-mask heatmap (from dyn_map)

`--order permuted` (default) plots in the order the SAP attention kernel sees
(rows sorted by qlabels, cols sorted by klabels). The mask becomes block-aligned.
`--order original` keeps the input token order.

Both heatmaps use the same row/col blocking, so they are directly comparable.
Mask values are in [0, 1] (= fraction of (q,k) pairs in that cell that the mask
keeps). Score values are average attention probability per cell (sum of softmax
mass divided by rows-in-block).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TRITON_CACHE_DIR", f"/tmp/triton-cache-{os.getuid()}")

from svg.kmeans_utils import batch_kmeans_Euclid, identify_dynamic_map


def parse_heads(raw: Optional[str], num_heads: int) -> List[int]:
    if raw is None or raw == "all":
        return list(range(num_heads))
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    bad = [h for h in out if h < 0 or h >= num_heads]
    if bad:
        raise ValueError(f"head indices {bad} out of [0, {num_heads})")
    return sorted(set(out))


def run_kmeans(query: torch.Tensor, key: torch.Tensor, metadata: dict,
               q_cache: torch.Tensor, k_cache: torch.Tensor):
    """Mirror semantic_aware_permutation but only return labels + dyn_map.

    query/key: [1, H, S, D] on device. Returns:
      qlabels [1, H, S] int, klabels [1, H, S] int,
      dyn_map [1, H, qc, kc] bool, qc_sz, kc_sz.
    """
    cfg, num_heads, seq_len, dim = query.shape
    num_q = int(metadata["num_q_centroids"])
    num_k = int(metadata["num_k_centroids"])
    iters = int(metadata["kmeans_iter_step"])
    top_p = float(metadata["top_p_kmeans"])
    min_kc = float(metadata["min_kc_ratio"])

    qlabels, qcent, qcsz, _ = batch_kmeans_Euclid(
        query.view(cfg * num_heads, seq_len, dim),
        n_clusters=num_q, max_iters=iters, init_centroids=q_cache,
    )
    klabels, kcent, kcsz, _ = batch_kmeans_Euclid(
        key.view(cfg * num_heads, seq_len, dim),
        n_clusters=num_k, max_iters=iters, init_centroids=k_cache,
    )
    qcsz = qcsz.view(cfg, num_heads, num_q)
    kcsz = kcsz.view(cfg, num_heads, num_k)
    dyn_map = identify_dynamic_map(
        qcent.view(cfg, num_heads, num_q, dim),
        kcent.view(cfg, num_heads, num_k, dim),
        qcsz, kcsz, top_p, min_kc,
    )
    qlabels = qlabels.view(cfg, num_heads, seq_len)
    klabels = klabels.view(cfg, num_heads, seq_len)
    return qlabels, klabels, dyn_map, qcsz, kcsz


def block_pool_attention(Q: torch.Tensor, K: torch.Tensor, out_h: int, out_w: int,
                         row_chunk: int) -> torch.Tensor:
    """Return [out_h, out_w] mean attention-probability per cell.

    cell[i, j] = (1 / |row_block_i|) * sum_{q in row_block_i} sum_{k in col_block_j} P[q, k]
    where P = softmax(Q @ K^T / sqrt(D), dim=-1).

    Rows are streamed in chunks of `row_chunk` to bound memory. Column reduction
    is fused via scatter_add into [chunk, out_w].
    """
    S, D = Q.shape
    device = Q.device
    scale = 1.0 / math.sqrt(D)

    col_idx = (torch.arange(S, device=device, dtype=torch.long) * out_w) // S  # [S]
    row_idx = (torch.arange(S, device=device, dtype=torch.long) * out_h) // S  # [S]
    rows_per_block = torch.bincount(row_idx, minlength=out_h).clamp(min=1).float()

    out = torch.zeros(out_h, out_w, device=device, dtype=torch.float32)
    K_t = K.t().contiguous()  # [D, S]

    for c0 in range(0, S, row_chunk):
        c1 = min(c0 + row_chunk, S)
        scores = (Q[c0:c1].float() @ K_t.float()) * scale  # [chunk, S]
        P = torch.softmax(scores, dim=-1)  # [chunk, S]
        # column-pool to [chunk, out_w]
        P_col = torch.zeros(c1 - c0, out_w, device=device, dtype=torch.float32)
        P_col.scatter_add_(1, col_idx.unsqueeze(0).expand(c1 - c0, -1), P)
        # row-pool: for each row r in [c0, c1), add P_col[r-c0] to out[row_idx[r]]
        out.scatter_add_(0, row_idx[c0:c1].unsqueeze(1).expand(-1, out_w), P_col)
        del scores, P, P_col
    out /= rows_per_block.unsqueeze(1)
    return out


def block_pool_mask(qlabels_h: torch.Tensor, klabels_h: torch.Tensor,
                    dyn_map_h: torch.Tensor, num_q: int, num_k: int,
                    out_h: int, out_w: int) -> torch.Tensor:
    """Return [out_h, out_w] selected-mask density per cell ∈ [0, 1].

    cell[i, j] = #{(q, k) in cell (i, j) : dyn_map[qlabels[q], klabels[k]]}
                 / (rows in block i * cols in block j).
    Closed-form via bincounts: qcount[i, c] = #{q in row block i : qlabels[q] = c}.
    cell = (qcount @ dyn_map.float() @ kcount.T) / (rows_i * cols_j).
    """
    device = qlabels_h.device
    S = qlabels_h.shape[0]
    row_idx = (torch.arange(S, device=device, dtype=torch.long) * out_h) // S
    col_idx = (torch.arange(S, device=device, dtype=torch.long) * out_w) // S

    qflat = row_idx * num_q + qlabels_h.long()
    kflat = col_idx * num_k + klabels_h.long()
    qcount = torch.bincount(qflat, minlength=out_h * num_q).reshape(out_h, num_q).float()
    kcount = torch.bincount(kflat, minlength=out_w * num_k).reshape(out_w, num_k).float()
    rows_per_block = qcount.sum(1).clamp(min=1)
    cols_per_block = kcount.sum(1).clamp(min=1)
    sel = qcount @ dyn_map_h.float() @ kcount.t()  # [out_h, out_w]
    sel /= rows_per_block.unsqueeze(1) * cols_per_block.unsqueeze(0)
    return sel


def make_heatmap_fig(arr: torch.Tensor, title: str, vmax: Optional[float] = None,
                     cmap: str = "viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr_np = arr.detach().cpu().float().numpy()
    if vmax is None:
        vmax = float(np.percentile(arr_np, 99.9)) or float(arr_np.max() or 1.0)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    im = ax.imshow(arr_np, cmap=cmap, vmin=0.0, vmax=vmax, aspect="equal",
                   interpolation="nearest", origin="upper")
    ax.set_xlabel("k blocks")
    ax.set_ylabel("q blocks")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def save_heatmap(arr: torch.Tensor, path: Path, title: str, vmax: Optional[float] = None,
                 cmap: str = "viridis"):
    import matplotlib.pyplot as plt
    fig = make_heatmap_fig(arr, title, vmax, cmap)
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="WAN attention .pt capture path")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--heads", default="all", help="e.g. '0,3,5-7'; default all")
    p.add_argument("--downsample", type=int, default=512,
                   help="output heatmap resolution (square); default 512")
    p.add_argument("--order", choices=["permuted", "original"], default="permuted")
    p.add_argument("--row-chunk", type=int, default=512,
                   help="rows of Q processed per chunk in score path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cmap-score", default="magma")
    p.add_argument("--cmap-mask", default="viridis")
    p.add_argument("--score-vmax-percentile", type=float, default=99.5,
                   help="clip score colormap at this percentile per head")
    p.add_argument("--format", choices=["png", "pdf"], default="png",
                   help="output image format (per-head files)")
    p.add_argument("--combined-pdf", default=None,
                   help="if set, also write all heatmaps into a single multi-page PDF at this path")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print(f"[load] {args.input}")
    data = torch.load(args.input, map_location="cpu", weights_only=False)
    metadata = data["metadata"]
    cfg, num_heads, seq_len, head_dim = data["inputs"]["query"].shape
    if cfg != 1:
        raise ValueError(f"only cfg=1 supported, got {cfg}")
    heads = parse_heads(args.heads, num_heads)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  layer={metadata.get('layer_idx')} timestep={metadata.get('timestep')} "
          f"H={num_heads} S={seq_len} D={head_dim} "
          f"qc={metadata['num_q_centroids']} kc={metadata['num_k_centroids']}")
    print(f"  heads={heads}  downsample={args.downsample}  order={args.order}")

    query = data["inputs"]["query"].to(device).contiguous()
    key = data["inputs"]["key"].to(device).contiguous()
    q_cache = data["centroid_cache"]["q_centroids"].to(device).contiguous()
    k_cache = data["centroid_cache"]["k_centroids"].to(device).contiguous()
    del data

    print("[kmeans] running batch kmeans + dyn_map ...")
    qlabels, klabels, dyn_map, qcsz, kcsz = run_kmeans(query, key, metadata, q_cache, k_cache)
    num_q = int(metadata["num_q_centroids"])
    num_k = int(metadata["num_k_centroids"])

    out_h = out_w = int(args.downsample)
    score_summary = []
    pdf_pages = None
    if args.combined_pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        Path(args.combined_pdf).parent.mkdir(parents=True, exist_ok=True)
        pdf_pages = PdfPages(args.combined_pdf)
    for h in heads:
        Q_h = query[0, h]   # [S, D]
        K_h = key[0, h]
        ql = qlabels[0, h]  # [S]
        kl = klabels[0, h]
        dm = dyn_map[0, h]  # [qc, kc] bool

        if args.order == "permuted":
            q_perm = torch.argsort(ql, stable=True)
            k_perm = torch.argsort(kl, stable=True)
            Q_view = Q_h.index_select(0, q_perm).contiguous()
            K_view = K_h.index_select(0, k_perm).contiguous()
            ql_view = ql.index_select(0, q_perm).contiguous()
            kl_view = kl.index_select(0, k_perm).contiguous()
        else:
            Q_view, K_view = Q_h, K_h
            ql_view, kl_view = ql, kl

        # Mask heatmap (cheap): closed-form
        mask_block = block_pool_mask(ql_view, kl_view, dm, num_q, num_k, out_h, out_w)

        # Score heatmap (chunked)
        torch.cuda.synchronize(device)
        score_block = block_pool_attention(Q_view, K_view, out_h, out_w, args.row_chunk)
        torch.cuda.synchronize(device)

        # Reporting metrics
        kept_frac = float(mask_block.mean().item())
        # Block-level recall approximation: per row block i, kept prob per q ≈
        # sum_j (mean P in cell (i,j)) * (kept fraction in cell (i,j)). Average
        # over i. This under-counts because intra-cell correlation between high-P
        # entries and mask is lost (mask is block-aligned in permuted view, so
        # bias is small there).
        recall_approx = float((score_block * mask_block).sum(dim=1).mean().item())
        score_summary.append({"head": h, "mask_frac": kept_frac, "recall_approx": recall_approx})
        print(f"  head {h:2d}: mask_kept_frac={kept_frac:.4f}  recall_approx={recall_approx:.4f}")

        ext = args.format
        score_vmax = float(np.percentile(score_block.detach().cpu().float().numpy(),
                                         args.score_vmax_percentile))
        score_title = f"head {h} attn score ({args.order}, {out_h}x{out_w})"
        mask_title = f"head {h} selected mask ({args.order}, {out_h}x{out_w})"
        save_heatmap(score_block, out_dir / f"head{h:02d}_score_{args.order}.{ext}",
                     title=score_title, vmax=score_vmax, cmap=args.cmap_score)
        save_heatmap(mask_block, out_dir / f"head{h:02d}_mask_{args.order}.{ext}",
                     title=mask_title, vmax=1.0, cmap=args.cmap_mask)
        if pdf_pages is not None:
            import matplotlib.pyplot as plt
            fig_s = make_heatmap_fig(score_block, score_title, score_vmax, args.cmap_score)
            pdf_pages.savefig(fig_s); plt.close(fig_s)
            fig_m = make_heatmap_fig(mask_block, mask_title, 1.0, args.cmap_mask)
            pdf_pages.savefig(fig_m); plt.close(fig_m)

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"input={args.input}\n")
        f.write(f"layer={metadata.get('layer_idx')} timestep={metadata.get('timestep')} "
                f"H={num_heads} S={seq_len} D={head_dim} order={args.order} "
                f"downsample={args.downsample}\n\n")
        f.write("head  mask_kept_frac  recall_approx\n")
        for r in score_summary:
            f.write(f"{r['head']:>4}  {r['mask_frac']:>14.4f}  {r['recall_approx']:>13.4f}\n")
    if pdf_pages is not None:
        pdf_pages.close()
        print(f"[done] also wrote combined PDF: {args.combined_pdf}")
    print(f"[done] wrote {2 * len(heads)} {args.format.upper()}s + summary to {out_dir}")


if __name__ == "__main__":
    main()

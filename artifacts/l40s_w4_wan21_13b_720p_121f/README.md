# PCIe-only replay on four L40S GPUs (paper Table 3)

Measured evidence from a four-GPU L40S host whose GPUs are connected only
through PCIe Gen4 x16 (no NVLink). The replayed workload is Wan2.1-1.3B T2V,
720p, 120 requested / 121 actual frames, layer 21, step 20: batch 1, 12 heads,
sequence length 111600, head dimension 128, bfloat16.

## Contents

- `inputs/1-0.jsonl`: the per-head density trace of the source generation.
- `calibration/`: the L40S mask-generation-aware cost model
  (`maskgen_aware_cost_720p_120req_121actual.json`, mask/attention R^2 =
  0.982/0.997) and the samples it was fitted on.
- `replay/`: per-rank (`*_rank*.csv`), per-head (`*_density.csv`), and
  per-query-chunk (`*_qchunk.csv`) breakdowns for each configuration. The
  `head_start`/`head_end` columns of the per-rank files describe contiguous
  placements only; for the unequal placements, `*_density.csv` lists each
  rank's heads.
- `SHA256SUMS`: checksums of all files above.

## Mapping to Table 3

The region latency is the maximum over ranks of `total_ms_mean` (mean over 50
measured iterations) in the `*_rank*.csv` files.

| Table 3 row | File | Region (ms) |
|---|---|---:|
| Contiguous equal-head + NCCL | `replay/contiguous_nccl_rank.csv` | 63.42 |
| Cost-aware unequal + padded NCCL | `replay/cost_unequal_padded_nccl_rank.csv` | 56.84 |
| Cost-aware unequal + asymmetric | `replay/cost_unequal_pcie_grid_rank.csv` | 53.86 |

`cost_unequal_pcie_grid_rank_rep2.csv` and `..._rep3.csv` are two additional
repetitions of the asymmetric configuration (55.89 and 55.47 ms). On pre-Hopper
GPUs the asymmetric exchange uses the PCIe copy backend in
`lb/asymm_pull_kernel.py` (selected automatically; these runs used
`ASYM_A2A_PCIE_PEER_MODE=grid`).

The raw Q/K/V snapshot (about 1 GB) is not included. It can be regenerated with
`RESOLUTION=720p NUM_FRAMES=120 bash lb/dump_wan_1.3b_attn.sh`.

"""Operator-level output agreement between AsymHP and the symmetric Ulysses
baseline (paper Appendix Table 5).

The baseline is the real symmetric path (contiguous placement + NCCL
all_to_all, `--asymm-a2a off`); the test is AsymHP (greedy_unequal + the
asymmetric TMA pull/push, `--asymm-a2a pull_qkv`). Greedy equal-count placement
is included as a middle point.

Replays one real (layer, step) attention snapshot and reports max|delta|,
relative L2 error, and PSNR over every rank's reassembled sequence-sharded
output. Two checks:
  1. contiguous/symmetric baseline vs unequal/asymmetric AsymHP (placement and
     transport both change);
  2. unequal/symmetric vs unequal/asymmetric (placement fixed; isolates the
     transport).

Usage (paper setting: 720p, 120 requested frames, layer 21, step 20):
  RESOLUTION=720p NUM_FRAMES=120 LAYER=21 STEP=20 bash lb/dump_wan_1.3b_attn.sh
  W=2 python lb/op_parity.py <dump.pt> <density.jsonl>
  W=4 MIN_HEADS=1 python lb/op_parity.py <dump.pt> <density.jsonl>   # 1/3/4/4
  W=4 MIN_HEADS=2 python lb/op_parity.py <dump.pt> <density.jsonl>   # 2/3/3/4
"""
import os, sys, subprocess, tempfile, math
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
BENCH = HERE / "bench_sp_all2all_attention.py"
INPUT = sys.argv[1] if len(sys.argv) > 1 else "result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_120frames/attn_dumps/1-0_step20_layer21.pt"
DENS = sys.argv[2] if len(sys.argv) > 2 else "result/wan/t2v/sap_1.3b/Step_50-Res_720p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9/Init_50-Step_2-MinR_0.10_120frames/1-0.jsonl"
COST = "result/wan/t2v/sap_1.3b/maskgen_aware_cost.json"
W = int(os.environ.get("W", "2"))
PORT = os.environ.get("PORT", "29677")
MIN_HEADS = os.environ.get("MIN_HEADS", "1")

def run(balance, a2a, pref, port):
    # NOTE: the bench crashes in the CUDASymmetricMemory destructor at teardown
    # (harmless); the out_seq dump is written BEFORE teardown, so we tolerate a
    # nonzero exit and just verify the dump exists.
    cmd = ["torchrun", f"--nproc-per-node={W}", f"--master-port={port}", str(BENCH),
           "--input", INPUT, "--density-log", DENS, "--cost-model-json", COST,
           "--balance", balance, "--asymm-a2a", a2a, "--iters", "1", "--warmup", "0",
           "--min-heads-per-rank", MIN_HEADS, "--dump-out-seq", str(pref)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not Path(f"{pref}.rank0.pt").exists():
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:])
        raise SystemExit(f"bench {balance}/{a2a} produced no dump")

def load_all_ranks(pref):
    return [
        torch.load(f"{pref}.rank{rank}.pt", map_location="cpu").float()
        for rank in range(W)
    ]


def parity_metrics(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate have different rank counts")

    max_abs = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    peak = 0.0
    numel = 0
    for rank, (base, value) in enumerate(zip(reference, candidate)):
        if base.shape != value.shape:
            raise ValueError(
                f"rank {rank} shape mismatch: {tuple(base.shape)} vs {tuple(value.shape)}"
            )
        diff = base - value
        max_abs = max(max_abs, diff.abs().max().item())
        squared_error += diff.square().sum().item()
        squared_reference += base.square().sum().item()
        peak = max(peak, base.abs().max().item())
        numel += base.numel()

    relative_l2 = (
        0.0
        if squared_error == 0.0
        else math.sqrt(squared_error / squared_reference)
    )
    mse = squared_error / numel
    psnr = (
        float("inf")
        if mse == 0.0
        else 20 * math.log10(peak + 1e-12) - 10 * math.log10(mse)
    )
    return max_abs, relative_l2, psnr

def main():
    d = tempfile.mkdtemp(prefix="parity_")
    # (label, balance, a2a)
    configs = [
        ("contiguous + SYMMETRIC (Ulysses baseline)", "contiguous", "off"),
        ("greedy + symmetric (equal-LB)",             "greedy",     "off"),
        ("greedy_unequal + SYMMETRIC (padding control)",
                                                        "greedy_unequal", "off"),
        ("greedy_unequal + ASYMMETRIC (AsymHP)",      "greedy_unequal", "pull_qkv"),
    ]
    outs = {}
    for i, (label, bal, a2a) in enumerate(configs):
        p = Path(d) / f"c{i}"
        run(bal, a2a, p, str(int(PORT) + i))
        outs[label] = load_all_ranks(p)
        print(f"[ran] {label}")
    base_label = configs[0][0]
    base = outs[base_label]
    print(
        f"\n# baseline = {base_label}   ranks={len(base)}  "
        f"per-rank shape={tuple(base[0].shape)}"
    )
    print(
        f"{'config':52s} {'max|Δ|':>12s} "
        f"{'relative L2':>14s} {'PSNR(dB)':>10s}"
    )
    for label in [c[0] for c in configs]:
        max_abs, relative_l2, psnr = parity_metrics(base, outs[label])
        print(f"{label:52s} {max_abs:12.6e} {relative_l2:14.6e} {psnr:10.2f}")

    symmetric_control = configs[2][0]
    asymmetric_test = configs[3][0]
    max_abs, relative_l2, psnr = parity_metrics(
        outs[symmetric_control], outs[asymmetric_test]
    )
    print(
        "\n# fixed-placement transport check: "
        f"{symmetric_control} vs {asymmetric_test}"
    )
    print(
        f"max|Δ|={max_abs:.6e}  relative_L2={relative_l2:.6e}  "
        f"PSNR(dB)={psnr:.2f}"
    )

if __name__ == "__main__":
    main()

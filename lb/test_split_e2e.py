#!/usr/bin/env python
"""End-to-end correctness test for `--balance split`.

Runs the bench script twice via torchrun (greedy_unequal baseline + split)
and compares per-rank `out_seq` tensors. The split path must reproduce the
baseline's output to within `--threshold` max-abs-diff (defaults to 1e-3,
the spec's pass criterion).

Usage:
    python lb/test_split_e2e.py [--world 6] [--threshold 1e-3]

Defaults to the 6-rank layer21/480frames sample referenced in the spec.
The host needs at least `--world` GPUs; we don't gate on this here so
torchrun's own error message fires when GPUs are missing.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BENCH = HERE / "bench_sp_all2all_attention.py"
RESULT_DIR = Path(os.environ.get("LB_RESULT_DIR", REPO_ROOT / "result/wan/t2v/sap_1.3b"))

DEFAULT_DENSITY_LOG = (
    RESULT_DIR
    / "Step_50-Res_480p/TFP_0.2-LFP_0.03/QC_300-KC_1000-TopP_0.9"
    / "Init_50-Step_2-MinR_0.10_480frames/1-0.jsonl"
)
DEFAULT_INPUT = DEFAULT_DENSITY_LOG.parent / "attn_dumps/1-0_step20_layer21.pt"
DEFAULT_COST_MODEL = RESULT_DIR / "maskgen_aware_cost.json"


def run_bench(world_size: int, balance: str, dump_prefix: Path, args) -> str:
    cmd = [
        "torchrun",
        f"--nproc-per-node={world_size}",
        f"--master-port={args.port}",
        str(BENCH),
        "--input", str(args.input),
        "--density-log", str(args.density_log),
        "--cost-model-json", str(args.cost_model_json),
        "--balance", balance,
        "--asymm-a2a", "pull_qkv",
        "--iters", "1",
        "--warmup", "0",
        "--dump-out-seq", str(dump_prefix),
    ]
    if balance == "split":
        cmd += [
            "--split-max-helpers", str(args.split_max_helpers),
            "--split-max-splits", str(args.split_max_splits),
            "--split-delta", str(args.split_delta),
            "--split-min-improvement-ms", str(args.split_min_improvement_ms),
        ]
    print("[run]", " ".join(cmd))
    out = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(args.env or {})})
    sys.stdout.write(out.stdout)
    sys.stderr.write(out.stderr)
    if out.returncode != 0:
        raise SystemExit(f"bench {balance} failed (exit {out.returncode})")
    return out.stdout


def compare(baseline_pref: Path, split_pref: Path, world_size: int, threshold: float) -> bool:
    overall = 0.0
    ok = True
    for r in range(world_size):
        a_path = Path(f"{baseline_pref}.rank{r}.pt")
        b_path = Path(f"{split_pref}.rank{r}.pt")
        if not a_path.exists() or not b_path.exists():
            print(f"[fail] missing dump for rank {r}: {a_path.exists()=} {b_path.exists()=}")
            return False
        a = torch.load(a_path, map_location="cpu")
        b = torch.load(b_path, map_location="cpu")
        if a.shape != b.shape:
            print(f"[fail] rank {r} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
            return False
        diff = (a.float() - b.float()).abs().max().item()
        overall = max(overall, diff)
        status = "ok" if diff < threshold else "FAIL"
        print(f"  rank {r}: max abs diff = {diff:.6e}  [{status}]")
        if diff >= threshold:
            ok = False
    print(f"[summary] overall max abs diff = {overall:.6e}  threshold = {threshold}")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world", type=int, default=6)
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--density-log", default=str(DEFAULT_DENSITY_LOG))
    p.add_argument("--cost-model-json", default=str(DEFAULT_COST_MODEL))
    p.add_argument("--threshold", type=float, default=1e-3)
    p.add_argument("--port", type=int, default=29501)
    p.add_argument("--split-max-helpers", type=int, default=2)
    p.add_argument("--split-max-splits", type=int, default=1,
                   help="Forwarded to bench/planner. 1 = single-split (default); >1 exercises the multi-split path.")
    p.add_argument("--split-delta", type=float, default=0.0,
                   help="Lower delta → more aggressive splitting; 0.0 forces a split for any imbalance.")
    p.add_argument("--split-min-improvement-ms", type=float, default=0.001,
                   help="Lower this in tests so the planner doesn't reject tiny-but-real splits.")
    p.add_argument("--require-split", action="store_true", default=True,
                   help="Fail if the bench reports 0 splits (E2E correctness only meaningfully tested when a split fires).")
    p.add_argument("--no-require-split", dest="require_split", action="store_false")
    p.add_argument("--keep-tmp", action="store_true")
    args = p.parse_args()
    args.env = None

    for path_attr in ("input", "density_log", "cost_model_json"):
        if not Path(getattr(args, path_attr)).exists():
            sys.exit(f"missing {path_attr}: {getattr(args, path_attr)}")

    workdir = Path(tempfile.mkdtemp(prefix="split_e2e_"))
    print(f"[workdir] {workdir}")
    try:
        baseline_pref = workdir / "baseline"
        split_pref = workdir / "split"
        run_bench(args.world, "greedy_unequal", baseline_pref, args)
        split_stdout = run_bench(args.world, "split", split_pref, args)

        # Sanity: assert the planner actually applied a split (pattern from
        # bench's "[split] N split(s); ..." line). Without this, the test
        # only verifies that 0-split-fallthrough matches greedy_unequal,
        # which is trivially true.
        n_splits = None
        for line in split_stdout.splitlines():
            line = line.strip()
            if line.startswith("[split]") and "split(s);" in line:
                try:
                    n_splits = int(line.split("[split]", 1)[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
                break
        if n_splits is None:
            print("[warn] could not parse '[split] N split(s)' line from bench stdout")
        else:
            print(f"[planner] applied {n_splits} split(s)")
            if args.require_split and n_splits == 0:
                print(f"FAIL: planner applied 0 splits — test is trivially passing. "
                      f"Lower --split-delta or pick a more imbalanced config.")
                sys.exit(1)

        ok = compare(baseline_pref, split_pref, args.world, args.threshold)
        print("PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    finally:
        if not args.keep_tmp:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

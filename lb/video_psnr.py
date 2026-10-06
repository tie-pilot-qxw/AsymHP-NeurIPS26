#!/usr/bin/env python3
"""PSNR between two decoded videos (paper appendix, decoded-video table).

Inputs are the arrays written by ``wan_t2v_sp_inference.py --save_latents``:
post-VAE frames of shape [frames, height, width, 3], float32 in [0, 1].
PSNR uses peak 1.0 and the MSE over all frames and pixels.

Usage:
  python lb/video_psnr.py reference.npy other.npy
"""
import math
import sys

import numpy as np


def video_psnr(reference_path: str, other_path: str, chunk: int = 8) -> float:
    reference = np.load(reference_path, mmap_mode="r")
    other = np.load(other_path, mmap_mode="r")
    if reference.shape != other.shape:
        raise SystemExit(f"shape mismatch: {reference.shape} vs {other.shape}")
    squared_error = 0.0
    for start in range(0, reference.shape[0], chunk):
        diff = reference[start:start + chunk].astype(np.float64) - other[start:start + chunk]
        squared_error += float(np.square(diff).sum())
    mse = squared_error / reference.size
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(f"{video_psnr(sys.argv[1], sys.argv[2]):.2f}")


if __name__ == "__main__":
    main()

"""Minimal NCCL correctness test.

Example:

    torchrun --standalone --nproc_per_node=2 lb/test_nccl_basic.py

This script only uses PyTorch distributed collectives.  It does not import or
exercise the project's Triton, TMA, or symmetric-memory kernels.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def _timed_collective(name: str, operation, device: torch.device) -> float:
    dist.barrier()
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    if dist.get_rank() == 0:
        print(f"PASS {name:<20} {elapsed_ms:8.3f} ms", flush=True)
    return elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elements-per-peer",
        type=int,
        default=1024,
        help="Number of float32 elements sent from each rank to each peer.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="NCCL process-group operation timeout.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not dist.is_nccl_available():
        raise RuntimeError("PyTorch was built without NCCL support")
    if args.elements_per_peer <= 0:
        raise ValueError("--elements-per-peer must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(seconds=args.timeout_seconds),
        device_id=device,
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        print(
            f"rank={rank}/{world_size} local_rank={local_rank} "
            f"device={torch.cuda.get_device_name(device)}",
            flush=True,
        )

        _timed_collective("barrier", dist.barrier, device)

        reduced = torch.tensor(float(rank), device=device)
        _timed_collective(
            "all_reduce",
            lambda: dist.all_reduce(reduced, op=dist.ReduceOp.SUM),
            device,
        )
        expected_sum = world_size * (world_size - 1) / 2
        torch.testing.assert_close(
            reduced,
            torch.tensor(expected_sum, device=device),
            rtol=0,
            atol=0,
        )

        # The chunk sent by source rank S to destination rank D contains the
        # value S * world_size + D.  After all-to-all, rank D can therefore
        # verify every received source chunk independently.
        elements = args.elements_per_peer
        send_chunks = [
            torch.full(
                (elements,),
                rank * world_size + destination,
                dtype=torch.float32,
                device=device,
            )
            for destination in range(world_size)
        ]
        send = torch.cat(send_chunks)
        received = torch.empty_like(send)
        _timed_collective(
            "all_to_all_single",
            lambda: dist.all_to_all_single(received, send),
            device,
        )

        expected_chunks = [
            torch.full(
                (elements,),
                source * world_size + rank,
                dtype=torch.float32,
                device=device,
            )
            for source in range(world_size)
        ]
        torch.testing.assert_close(
            received,
            torch.cat(expected_chunks),
            rtol=0,
            atol=0,
        )

        dist.barrier()
        if rank == 0:
            bytes_per_rank = world_size * elements * send.element_size()
            print(
                "PASS NCCL basic correctness: "
                f"{world_size} ranks, {bytes_per_rank} bytes/rank in all-to-all",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

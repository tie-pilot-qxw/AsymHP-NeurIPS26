"""Minimal PyTorch CUDA symmetric-memory correctness test.

Example:

    torchrun --standalone --nproc_per_node=2 \
        lb/test_symmetric_memory_basic.py

The test exercises symmetric allocation, rendezvous, the symmetric-memory
barrier, and direct reads from every peer buffer.  It does not launch the
project's Triton/TMA kernels.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.elements <= 0:
        raise ValueError("--elements must be positive")

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
        tensor = symm.empty(args.elements, dtype=torch.float32, device=device)
        tensor.fill_(rank + 1)
        torch.cuda.synchronize(device)

        start = time.perf_counter()
        handle = symm.rendezvous(tensor, dist.group.WORLD.group_name)
        rendezvous_ms = (time.perf_counter() - start) * 1_000
        print(
            f"rank={rank}/{world_size} device={torch.cuda.get_device_name(device)} "
            f"rendezvous={rendezvous_ms:.3f} ms "
            f"buffer_ptrs={[hex(ptr) for ptr in handle.buffer_ptrs]}",
            flush=True,
        )

        start = time.perf_counter()
        handle.barrier(channel=0, timeout_ms=args.timeout_seconds * 1_000)
        torch.cuda.synchronize(device)
        barrier_ms = (time.perf_counter() - start) * 1_000

        # get_buffer(peer, ...) returns a tensor view backed by the peer's
        # symmetric allocation.  Cloning it launches a local-device read from
        # that peer mapping, which validates more than address rendezvous.
        peer_copies = []
        for peer in range(world_size):
            peer_view = handle.get_buffer(
                peer,
                (args.elements,),
                torch.float32,
            )
            peer_copies.append(peer_view.clone())
        torch.cuda.synchronize(device)

        for peer, peer_copy in enumerate(peer_copies):
            expected = torch.full_like(peer_copy, peer + 1)
            torch.testing.assert_close(peer_copy, expected, rtol=0, atol=0)

        handle.barrier(channel=1, timeout_ms=args.timeout_seconds * 1_000)
        if rank == 0:
            print(
                "PASS symmetric memory: "
                f"{world_size} ranks, rendezvous + barrier ({barrier_ms:.3f} ms) "
                "+ direct peer reads",
                flush=True,
            )
    except BaseException:
        dist.destroy_process_group()
        raise
    # Freeing symmetric memory can abort in CUDASymmetricMemory's destructor
    # (torch 2.9); skip interpreter teardown once the test has passed.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

"""torchrun DDP helpers. Single API for 1-GPU and N-GPU setup."""
from __future__ import annotations
import os
import random
import socket
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


def init_distributed() -> dict:
    """Initialise process group from env vars set by torchrun.

    Returns a dict with: rank, world_size, local_rank, device, is_dist, is_main.
    """
    world_size = _env_int("WORLD_SIZE", 1)
    rank       = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    is_dist = world_size > 1

    if is_dist:
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo",
                init_method="env://",
            )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": device,
        "is_dist": is_dist,
        "is_main": rank == 0,
        "hostname": socket.gethostname(),
    }


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Mean-reduce across all DDP ranks. No-op on a single GPU."""
    if not dist.is_initialized():
        return t
    out = t.clone().detach()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / dist.get_world_size()


def all_gather_object(obj):
    """Gather Python objects from all ranks; returns a list of length world_size."""
    if not dist.is_initialized():
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def seed_everything(seed: int, rank: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Each rank gets a different seed so DataLoader shuffles differ across ranks.
    """
    s = seed + rank * 10_000
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


@contextmanager
def main_process_first(is_main: bool):
    """Run the body on the main process first, then on others (for dataset prep)."""
    if not dist.is_initialized():
        yield
        return
    if is_main:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield

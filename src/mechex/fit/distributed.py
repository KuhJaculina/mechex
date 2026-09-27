"""Distributed wrapping.

The manuscript does not state a distributed setting, so the default launch is a single
process. The wrappers are supplied so the same entry point can run under ``torchrun``
without a code change.
"""

from __future__ import annotations

import os

import torch
from torch import nn

from mechex.utils.logging import get_logger

_log = get_logger("fit.distributed")


def distributed_available() -> bool:
    """Whether a process group is already initialised."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def world_size() -> int:
    """Process count of the active group, or one when running alone."""
    if not distributed_available():
        return 1
    return int(torch.distributed.get_world_size())


def rank() -> int:
    """Rank of this process, or zero when running alone."""
    if not distributed_available():
        return 0
    return int(torch.distributed.get_rank())


def is_main_process() -> bool:
    """Whether this process owns writing and logging."""
    return rank() == 0


def initialise(backend: str | None = None) -> bool:
    """Initialise a process group when the launcher provides the environment."""
    if distributed_available():
        return True
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False
    resolved = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    torch.distributed.init_process_group(backend=resolved)
    _log.info("process group initialised with backend %s", resolved)
    return True


def wrap_distributed(
    module: nn.Module,
    device: torch.device,
    local_rank: int = 0,
) -> nn.Module:
    """Wrap a module in DDP when a group is active.

    Without a group the module is returned unchanged, so the single-process path is the
    same code as the distributed one.
    """
    if not distributed_available():
        return module
    if device.type == "cuda":
        return nn.parallel.DistributedDataParallel(module, device_ids=[local_rank])
    return nn.parallel.DistributedDataParallel(module)


def barrier() -> None:
    """Synchronise the group when one is active."""
    if distributed_available():
        torch.distributed.barrier()


def destroy() -> None:
    """Tear down the group when one is active."""
    if distributed_available():
        torch.distributed.destroy_process_group()

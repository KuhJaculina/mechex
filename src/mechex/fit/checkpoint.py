"""Atomic checkpoints carrying the seed and the calibration state.

Ref: the release contract's resume requirement; the manuscript fixes five independent
repeats with different random initialisations, so a repeat has to be resumable exactly.
"""

from __future__ import annotations

import hashlib
import io
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from mechex.settings import EstimatorConfig, config_to_dict
from mechex.utils.logging import get_logger

_log = get_logger("fit.checkpoint")

CHECKPOINT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every random stream the release draws from."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a snapshot written by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def build_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    seed: int,
    config: EstimatorConfig,
    metrics: dict[str, float],
    calibrator: dict[str, object] | None = None,
    history: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Assemble a checkpoint payload."""
    return {
        "version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "seed": int(seed),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "config": config_to_dict(config),
        "metrics": dict(metrics),
        "calibrator": calibrator,
        "history": list(history or []),
        "rng": capture_rng_state(),
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    """Write a checkpoint through a temporary file and an atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(payload, stream)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    _log.debug("checkpoint written")
    return path


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    """Read a checkpoint and verify its version."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("the checkpoint is not a MechEx payload")
    if int(payload.get("version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')}")
    return payload


def restore_into(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    resume_rng: bool = True,
) -> dict[str, Any]:
    """Load a checkpoint into a model and optionally its optimiser.

    The seed is reapplied before the optimiser state is loaded so a resumed run continues
    the same stream rather than a fresh one.
    """
    payload = load_checkpoint(path)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if resume_rng and payload.get("rng") is not None:
        restore_rng_state(payload["rng"])
    return payload


def tensor_payload_digest(path: Path) -> str:
    """SHA-256 over the tensor contents of a checkpoint.

    Hashing the file's bytes would report a different value on every write because
    ``torch.save`` embeds container metadata, so the digest is taken over the payloads.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name in sorted(payload.get("model", {})):
        tensor = payload["model"][name]
        digest.update(name.encode("utf-8"))
        if isinstance(tensor, torch.Tensor):
            buffer = io.BytesIO()
            torch.save(tensor.detach().cpu().contiguous(), buffer)
            digest.update(buffer.getvalue())
    return digest.hexdigest()


def config_from_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Config mapping stored in a checkpoint."""
    stored = payload.get("config")
    return dict(stored) if isinstance(stored, dict) else {}


def metrics_from_checkpoint(payload: dict[str, Any]) -> dict[str, float]:
    """Metrics stored in a checkpoint."""
    stored = payload.get("metrics")
    return (
        {str(key): float(value) for key, value in stored.items()}
        if isinstance(stored, dict)
        else {}
    )


def seed_from_checkpoint(payload: dict[str, Any]) -> int:
    """Seed stored in a checkpoint."""
    return int(payload.get("seed", 0))


def dataclass_payload(config: EstimatorConfig) -> dict[str, Any]:
    """Plain mapping of a config, kept for callers that only need the raw dict."""
    return asdict(config)

"""Filesystem helpers with the atomicity the release needs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from mechex.utils.logging import get_logger
from mechex.version import RELEASE_DIR

_log = get_logger("io")

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".ipynb_checkpoints",
}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def atomic_write_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    """Write ``payload`` to ``path`` through a temporary file in the same directory.

    ``tempfile.mkstemp`` creates the file 0600; chmod before the replace so the release
    artefacts are readable by whoever reads the checkout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def atomic_write_text(path: Path, text: str, mode: int = 0o644) -> None:
    """Atomic text write."""
    atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def atomic_write_json(path: Path, payload: Any, mode: int = 0o644) -> None:
    """Atomic JSON write with a stable key order and a trailing newline."""
    atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n", mode=mode
    )


def read_json(path: Path) -> Any:
    """Read a JSON document."""
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(root: Path) -> list[Path]:
    """Every regular file under ``root`` in a deterministic order, skipping caches."""
    collected: list[Path] = []
    for current, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            candidate = Path(current) / name
            if candidate.suffix in SKIP_SUFFIXES:
                continue
            collected.append(candidate)
    return sorted(collected, key=lambda p: str(p.relative_to(root)))


def relative_posix(path: Path, root: Path) -> str:
    """Posix-style path of ``path`` relative to ``root``."""
    return Path(os.path.relpath(path, root)).as_posix()


def display_root(root: Path) -> str:
    """Name a release records for itself.

    Reports carry the release slug rather than the directory they were built in, so a
    distributed artefact embeds neither the host filesystem layout nor the local checkout
    name; the same release therefore produces identical bytes wherever it is unpacked.
    """
    del root
    return RELEASE_DIR


def ensure_dir(path: Path) -> Path:
    """Create ``path`` if absent and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_tree(path: Path) -> None:
    """Delete a directory tree if it exists."""
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()
    path.rmdir()
    _log.debug("removed tree")

"""Integrity manifest over the release tree.

The manifest digests every shipped file except itself, so a reader can verify that the tree
they received is the tree the release was verified against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mechex.provenance.claims import claim_index
from mechex.utils.io import atomic_write_json, display_root, iter_files, relative_posix, sha256_file
from mechex.version import __release__, __version__

MANIFEST_NAME = "integrity_manifest.json"
EXCLUDED_NAMES = {MANIFEST_NAME}


@dataclass(frozen=True)
class ManifestEntry:
    """One digested file."""

    path: str
    sha256: str
    bytes: int

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


def collect_entries(root: Path, excluded: set[str] | None = None) -> list[ManifestEntry]:
    """Digest every file under ``root`` except the manifest itself."""
    skip = EXCLUDED_NAMES | (excluded or set())
    entries: list[ManifestEntry] = []
    for path in iter_files(root):
        relative = relative_posix(path, root)
        if relative in skip:
            continue
        entries.append(
            ManifestEntry(path=relative, sha256=sha256_file(path), bytes=path.stat().st_size)
        )
    return entries


def manifest_digest(entries: list[ManifestEntry]) -> str:
    """SHA-256 over the ordered ``path:digest`` lines of the entries."""
    import hashlib

    payload = "\n".join(f"{entry.path}:{entry.sha256}" for entry in entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(
    root: Path, verified: bool, notes: dict[str, object] | None = None
) -> dict[str, object]:
    """Assemble the manifest payload."""
    entries = collect_entries(root)
    return {
        "release": __release__,
        "version": __version__,
        "root": display_root(root),
        "file_count": len(entries),
        "total_bytes": sum(entry.bytes for entry in entries),
        "manifest_digest": manifest_digest(entries),
        "verified": verified,
        "claims": claim_index(),
        "notes": dict(notes or {}),
        "files": [entry.as_dict() for entry in entries],
    }


def write_manifest(root: Path, verified: bool, notes: dict[str, object] | None = None) -> Path:
    """Write the manifest last, so it digests the finished tree."""
    payload = build_manifest(root, verified, notes)
    path = root / MANIFEST_NAME
    atomic_write_json(path, payload)
    return path


def verify_manifest(root: Path) -> dict[str, object]:
    """Re-hash the live tree and diff it against the shipped manifest."""
    from mechex.utils.io import read_json

    path = root / MANIFEST_NAME
    if not path.is_file():
        return {"present": False, "matches": False, "reason": "the manifest is absent"}
    payload = read_json(path)
    recorded = {str(entry["path"]): str(entry["sha256"]) for entry in payload.get("files", [])}
    live = {entry.path: entry.sha256 for entry in collect_entries(root)}
    missing = sorted(set(recorded) - set(live))
    added = sorted(set(live) - set(recorded))
    changed = sorted(
        pathname for pathname in set(recorded) & set(live) if recorded[pathname] != live[pathname]
    )
    digest_matches = manifest_digest(collect_entries(root)) == payload.get("manifest_digest")
    return {
        "present": True,
        "matches": not missing and not added and not changed and digest_matches,
        "missing": missing,
        "added": added,
        "changed": changed,
        "digest_matches": digest_matches,
        "recorded_count": len(recorded),
        "live_count": len(live),
    }

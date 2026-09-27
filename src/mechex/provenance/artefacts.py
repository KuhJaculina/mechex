"""Writers for the three root artefacts and the plain-text summary.

Every writer takes an explicit ``root`` so a test can point it at a temporary directory;
without that, a test that calls a writer would overwrite the shipped artefacts.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from mechex.provenance.claims import (
    CLAIMS,
    DEVIATIONS,
    all_anchors,
    all_code_references,
    claim_index,
)
from mechex.provenance.manifest import write_manifest
from mechex.utils.io import atomic_write_json, atomic_write_text, display_root
from mechex.version import MODEL_NAME, __release__, __version__

CLAIM_ARTEFACT = "claim_to_code.json"
REPORT_ARTEFACT = "verification_report.json"
MANIFEST_ARTEFACT = "integrity_manifest.json"
SUMMARY_ARTEFACT = "verification_summary.txt"

VERIFICATION_VOCABULARY = ("PASS", "FAIL", "NOT_RUN", "BLOCKED")

INTERPRETATION = (
    "PASS means the check ran on this host and held. FAIL means it ran and did not hold. "
    "NOT_RUN means the input it needs is unavailable, which for a private clinical cohort is "
    "the expected outcome and is never an invitation to substitute a number. BLOCKED means an "
    "external resource is required and is not present in this checkout. overall_status covers "
    "every check; code_status covers the same set without the manuscript family, so a "
    "disagreement inside the manuscript's own tables is not confused with a defect in this "
    "release."
)


def default_root() -> Path:
    """Repository root, resolved from this module's location."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class AnchorResolution:
    """Whether a ``module:qualname`` reference in the claim registry resolves."""

    reference: str
    resolved: bool
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {"reference": self.reference, "resolved": self.resolved, "detail": self.detail}


def resolve_reference(reference: str) -> AnchorResolution:
    """Import the module and look up the attribute chain of a claim's code reference."""
    module_name, _, qualname = reference.partition(":")
    if not module_name or not qualname:
        return AnchorResolution(
            reference, False, "the reference is not of the form module:qualname"
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return AnchorResolution(reference, False, f"module import failed: {error}")
    target: object = module
    for part in qualname.split("."):
        if not hasattr(target, part):
            return AnchorResolution(
                reference, False, f"attribute {part} is absent from {module_name}"
            )
        target = getattr(target, part)
    return AnchorResolution(reference, True, f"resolved on {module_name}")


def resolve_all_references() -> list[AnchorResolution]:
    """Resolve every code reference in the claim registry."""
    return [resolve_reference(reference) for reference in all_code_references()]


def build_claim_artefact(root: Path) -> dict[str, object]:
    """Assemble ``claim_to_code.json``."""
    resolutions = resolve_all_references()
    unresolved = [item.reference for item in resolutions if not item.resolved]
    return {
        "artifact": CLAIM_ARTEFACT,
        "release": __release__,
        "version": __version__,
        "model": MODEL_NAME,
        "root": display_root(root),
        "anchor_convention": "anchors are the manuscript's own section, equation and table numbers",
        "verification_vocabulary": list(VERIFICATION_VOCABULARY),
        "registry": claim_index(),
        "anchors": all_anchors(),
        "claims": [item.as_dict() for item in CLAIMS],
        "claim_count": len(CLAIMS),
        "resolved_references": sum(1 for item in resolutions if item.resolved),
        "unresolved_references": unresolved,
        "reference_resolutions": [item.as_dict() for item in resolutions],
        "deviations": [item.as_dict() for item in DEVIATIONS],
        "deviation_count": len(DEVIATIONS),
    }


def write_claim_artefact(root: Path) -> Path:
    """Write ``claim_to_code.json``."""
    path = root / CLAIM_ARTEFACT
    atomic_write_json(path, build_claim_artefact(root))
    return path


def summarise_checks(checks: list[dict[str, object]]) -> dict[str, object]:
    """Counts by status and by family, and the two headline statuses."""
    statuses: dict[str, int] = dict.fromkeys(VERIFICATION_VOCABULARY, 0)
    families: dict[str, dict[str, int]] = {}
    for check in checks:
        status = str(check.get("status", "NOT_RUN"))
        statuses[status] = statuses.get(status, 0) + 1
        family = str(check.get("family", "code"))
        families.setdefault(family, {})
        families[family][status] = families[family].get(status, 0) + 1
    # Any family whose name begins with "manuscript" is a check on the paper's own tables, not
    # on this release, and is excluded from the code status.
    code_checks = [
        check for check in checks if not str(check.get("family", "code")).startswith("manuscript")
    ]
    code_statuses: dict[str, int] = dict.fromkeys(VERIFICATION_VOCABULARY, 0)
    for check in code_checks:
        status = str(check.get("status", "NOT_RUN"))
        code_statuses[status] = code_statuses.get(status, 0) + 1
    return {
        "total": len(checks),
        "by_status": statuses,
        "by_family": families,
        "code_total": len(code_checks),
        "code_by_status": code_statuses,
        "overall_status": _headline(statuses, len(checks)),
        "code_status": _headline(code_statuses, len(code_checks)),
    }


def _headline(statuses: dict[str, int], total: int) -> str:
    """Headline status of one family of checks, read from what actually ran."""
    if total == 0:
        return "UNVERIFIED"
    if statuses.get("FAIL", 0) > 0:
        return "PARTIALLY_VERIFIED"
    if statuses.get("BLOCKED", 0) > 0 or statuses.get("NOT_RUN", 0) > 0:
        return "PARTIALLY_VERIFIED"
    if statuses.get("PASS", 0) == total:
        return "VERIFIED"
    return "UNVERIFIED"


def build_report(
    root: Path,
    checks: list[dict[str, object]],
    configured_run: dict[str, object],
    execution_profile: dict[str, object],
    manuscript_checks: list[dict[str, object]],
) -> dict[str, object]:
    """Assemble ``verification_report.json``."""
    all_checks = list(checks) + list(manuscript_checks)
    summary = summarise_checks(all_checks)
    return {
        "release": __release__,
        "version": __version__,
        "model": MODEL_NAME,
        "root": display_root(root),
        "summary": summary,
        "claim_mapping": {
            "claims": len(CLAIMS),
            "anchors": len(all_anchors()),
            "code_references": len(all_code_references()),
            "deviations": len(DEVIATIONS),
            "unresolved_references": [
                item.reference for item in resolve_all_references() if not item.resolved
            ],
        },
        "configured_run": configured_run,
        "execution_profile": execution_profile,
        "checks": checks,
        "manuscript_checks": manuscript_checks,
        "interpretation": INTERPRETATION,
    }


def render_summary(report: dict[str, object]) -> str:
    """Plain-text rendering of a report.

    The summary goes to ``.txt`` rather than markdown: the release ships no markdown beyond
    its README, and a CLI that emitted one would break that on its first run.
    """
    summary = report.get("summary", {})
    assert isinstance(summary, dict)
    lines: list[str] = []
    lines.append(f"{report.get('release')} verification summary")
    lines.append(f"root: {report.get('root')}")
    lines.append(f"overall status: {summary.get('overall_status')}")
    lines.append(f"code status   : {summary.get('code_status')}")
    lines.append(
        f"checks: {summary.get('total')} total ({_format_counts(summary.get('by_status', {}))})"
    )
    lines.append(
        f"code checks: {summary.get('code_total')} total "
        f"({_format_counts(summary.get('code_by_status', {}))})"
    )
    lines.append("")

    claim_mapping = report.get("claim_mapping", {})
    assert isinstance(claim_mapping, dict)
    lines.append(
        "claim mapping: "
        f"{claim_mapping.get('claims')} claims over {claim_mapping.get('anchors')} anchors, "
        f"{claim_mapping.get('code_references')} code references, "
        f"{claim_mapping.get('deviations')} deviations"
    )
    unresolved = claim_mapping.get("unresolved_references", [])
    lines.append(f"unresolved code references: {len(unresolved)}")
    lines.append("")

    lines.append("configured run")
    for key, value in sorted(_as_mapping(report.get("configured_run", {})).items()):
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.append("execution profile")
    for key, value in sorted(_as_mapping(report.get("execution_profile", {})).items()):
        lines.append(f"  {key}: {value}")
    lines.append("")

    lines.extend(_render_check_section("code checks", report.get("checks", [])))
    lines.extend(_render_check_section("manuscript checks", report.get("manuscript_checks", [])))
    lines.append("interpretation")
    lines.append(f"  {report.get('interpretation')}")
    return "\n".join(lines) + "\n"


def _format_counts(counts: object) -> str:
    mapping = _as_mapping(counts)
    ordered = [f"{status}={mapping.get(status, 0)}" for status in VERIFICATION_VOCABULARY]
    return ", ".join(ordered)


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _render_check_section(title: str, checks: object) -> list[str]:
    lines = [title]
    if not isinstance(checks, list):
        lines.append("  none")
        return [*lines, ""]
    for check in checks:
        assert isinstance(check, dict)
        status = str(check.get("status", "NOT_RUN"))
        name = str(check.get("name", "unnamed"))
        lines.append(f"  [{status}] {name}")
        detail = str(check.get("detail", ""))
        if detail:
            lines.append(f"      {detail}")
    lines.append("")
    return lines


def write_report(root: Path, report: dict[str, object]) -> Path:
    """Write ``verification_report.json``."""
    path = root / REPORT_ARTEFACT
    atomic_write_json(path, report)
    return path


def write_summary(root: Path, text: str) -> Path:
    """Write ``verification_summary.txt``."""
    path = root / SUMMARY_ARTEFACT
    atomic_write_text(path, text)
    return path


def write_integrity_manifest(root: Path, verified: bool, notes: dict[str, object]) -> Path:
    """Write ``integrity_manifest.json`` last."""
    return write_manifest(root, verified=verified, notes=notes)

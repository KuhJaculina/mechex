"""The shipped verification artefacts must be the complete ones.

The driver's writers default to the repository root, so a test that called them without a
root would replace the canonical report with a partial runner's report. These tests keep
that from happening and assert the shipped report is the full run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechex.console.audit import CHECK_REGISTRY
from mechex.provenance.arithmetic import manuscript_checks
from mechex.provenance.artefacts import (
    CLAIM_ARTEFACT,
    MANIFEST_ARTEFACT,
    REPORT_ARTEFACT,
    SUMMARY_ARTEFACT,
    build_report,
    render_summary,
    write_claim_artefact,
    write_report,
    write_summary,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOTAL = len(CHECK_REGISTRY) + len(manuscript_checks())


def test_registry_has_no_duplicate_names() -> None:
    names = [name for name, _ in CHECK_REGISTRY]
    assert len(names) == len(set(names))


def test_writers_only_touch_the_root_they_are_given(tmp_path: Path) -> None:
    """A writer called with a temporary root must not touch the release tree."""
    shipped = ROOT / REPORT_ARTEFACT
    before = shipped.read_bytes() if shipped.is_file() else None
    report = build_report(
        tmp_path,
        checks=[{"name": "probe", "family": "code", "status": "PASS", "detail": "ok"}],
        configured_run={},
        execution_profile={},
        manuscript_checks=[],
    )
    write_claim_artefact(tmp_path)
    write_report(tmp_path, report)
    write_summary(tmp_path, render_summary(report))
    assert (tmp_path / CLAIM_ARTEFACT).is_file()
    after = shipped.read_bytes() if shipped.is_file() else None
    assert after == before


def test_build_report_names_the_release_not_the_checkout(tmp_path: Path) -> None:
    """The recorded root is the release slug, so a clone under any directory agrees."""
    from mechex.version import RELEASE_DIR

    report = build_report(
        tmp_path, [], configured_run={}, execution_profile={}, manuscript_checks=[]
    )
    assert report["root"] == RELEASE_DIR
    assert tmp_path.resolve().name not in json.dumps(report)
    assert str(tmp_path.resolve()) not in json.dumps(report)


@pytest.mark.skipif(not (ROOT / REPORT_ARTEFACT).is_file(), reason="no shipped report yet")
def test_shipped_report_is_the_complete_run() -> None:
    report = json.loads((ROOT / REPORT_ARTEFACT).read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["total"] == EXPECTED_TOTAL, (
        "the shipped report does not cover every registered check; a partial runner's report "
        "has replaced the canonical one"
    )
    assert len(report["checks"]) == len(CHECK_REGISTRY)
    assert len(report["manuscript_checks"]) == len(manuscript_checks())


@pytest.mark.skipif(not (ROOT / REPORT_ARTEFACT).is_file(), reason="no shipped report yet")
def test_shipped_report_statuses_are_consistent() -> None:
    report = json.loads((ROOT / REPORT_ARTEFACT).read_text(encoding="utf-8"))
    summary = report["summary"]
    counted = sum(summary["by_status"].values())
    assert counted == summary["total"]
    assert summary["overall_status"] in {"VERIFIED", "PARTIALLY_VERIFIED", "UNVERIFIED"}
    for check in report["checks"]:
        assert check["status"] in {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
        assert check["name"] and check["detail"]


@pytest.mark.skipif(not (ROOT / SUMMARY_ARTEFACT).is_file(), reason="no shipped summary yet")
def test_shipped_summary_is_plain_text() -> None:
    text = (ROOT / SUMMARY_ARTEFACT).read_text(encoding="utf-8")
    assert text.startswith("mechex-")
    assert "#" not in text.splitlines()[0]
    assert "overall status" in text


@pytest.mark.skipif(not (ROOT / MANIFEST_ARTEFACT).is_file(), reason="no shipped manifest yet")
def test_shipped_manifest_covers_the_tree() -> None:
    from mechex.provenance.manifest import verify_manifest

    payload = json.loads((ROOT / MANIFEST_ARTEFACT).read_text(encoding="utf-8"))
    report = json.loads((ROOT / REPORT_ARTEFACT).read_text(encoding="utf-8"))
    assert payload["verified"] is (report["summary"]["overall_status"] == "VERIFIED")
    result = verify_manifest(ROOT)
    assert result["present"]
    assert result["digest_matches"]

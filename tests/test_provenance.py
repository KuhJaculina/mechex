"""Claim registry, manuscript arithmetic and the artefact writers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mechex.provenance.arithmetic import (
    FAMILY_ARITHMETIC,
    FAMILY_COMPLETENESS,
    manuscript_checks,
    manuscript_summary,
)
from mechex.provenance.artefacts import (
    CLAIM_ARTEFACT,
    MANIFEST_ARTEFACT,
    REPORT_ARTEFACT,
    SUMMARY_ARTEFACT,
    build_claim_artefact,
    build_report,
    default_root,
    render_summary,
    resolve_all_references,
    resolve_reference,
    summarise_checks,
    write_claim_artefact,
    write_integrity_manifest,
    write_report,
    write_summary,
)
from mechex.provenance.claims import (
    CLAIMS,
    DEVIATIONS,
    all_anchors,
    all_code_references,
    claim,
    claim_index,
)
from mechex.provenance.manifest import (
    build_manifest,
    collect_entries,
    manifest_digest,
    verify_manifest,
)
from mechex.provenance.tables import (
    TABLE_2_ROWS,
    TABLE_3_ROWS,
    TABLE_4_ROWS,
    baseline_identifiers,
    table_1_row,
)


def test_claim_registry_is_well_formed() -> None:
    identifiers = [item.identifier for item in CLAIMS]
    assert len(identifiers) == len(set(identifiers))
    assert all(item.statement and item.anchor and item.location for item in CLAIMS)
    assert all(item.code for item in CLAIMS)
    assert len(all_anchors()) > 20


def test_claim_index_reports_counts() -> None:
    index = claim_index()
    assert index["claim_count"] == len(CLAIMS)
    assert index["anchor_count"] == len(all_anchors())
    assert index["code_reference_count"] == len(all_code_references())
    assert index["deviation_count"] == len(DEVIATIONS)


def test_claim_lookup() -> None:
    assert claim("c13").code == ("mechex.conservation.law:systemic_availability",)
    with pytest.raises(KeyError):
        claim("not_a_claim")


def test_every_code_reference_resolves() -> None:
    resolutions = resolve_all_references()
    unresolved = [item.reference for item in resolutions if not item.resolved]
    assert not unresolved, unresolved


def test_reference_resolution_reports_bad_shapes() -> None:
    assert not resolve_reference("mechex.conservation.law").resolved
    assert not resolve_reference("mechex.not_a_module:thing").resolved
    assert not resolve_reference("mechex.conservation.law:not_there").resolved


def test_deviations_are_complete() -> None:
    for deviation in DEVIATIONS:
        assert deviation.paper_location
        assert deviation.deviation
        assert deviation.justification
    assert len({item.identifier for item in DEVIATIONS}) == len(DEVIATIONS)


def test_reported_tables_are_internally_ordered() -> None:
    assert len(TABLE_2_ROWS) == 14
    assert len(TABLE_3_ROWS) == 10
    assert len(TABLE_4_ROWS) == 5
    assert len(baseline_identifiers()) == 12
    assert table_1_row("B9")["ccc"] == pytest.approx(0.431)


def test_manuscript_checks_report_the_expected_failures() -> None:
    checks = manuscript_checks()
    statuses = {check.name: check.status for check in checks}
    assert statuses["ccc_margin_over_criterion"] == "PASS"
    assert statuses["auroc_margin_over_criterion"] == "PASS"
    assert statuses["interaction_synergy_excess"] == "FAIL"
    assert statuses["external_ccc_value_disagreement"] == "FAIL"
    assert statuses["fold_error_zone_placement"] == "FAIL"
    assert statuses["supplementary_collision_sc_1"] == "FAIL"
    assert statuses["table_4_degradation_arithmetic"] == "PASS"
    assert statuses["table_4_banding_consistency"] == "PASS"
    assert statuses["ablation_branch_removal_matches_single_branch"] == "PASS"


def test_manuscript_summary_counts_by_status() -> None:
    summary = manuscript_summary()
    assert summary["total"] == len(manuscript_checks())
    assert summary["by_status"]["FAIL"] >= 4
    assert "interaction_synergy_excess" in summary["failures"]


def test_manuscript_check_families() -> None:
    families = {check.family for check in manuscript_checks()}
    assert families <= {FAMILY_ARITHMETIC, FAMILY_COMPLETENESS}


def test_claim_artefact_shape() -> None:
    artefact = build_claim_artefact(Path("/tmp/probe-root"))
    assert artefact["claim_count"] == len(CLAIMS)
    assert artefact["deviation_count"] == len(DEVIATIONS)
    assert not artefact["unresolved_references"]
    assert artefact["resolved_references"] == len(all_code_references())


def test_write_claim_artefact_uses_the_requested_root(tmp_path: Path) -> None:
    written = write_claim_artefact(tmp_path)
    assert written == tmp_path / CLAIM_ARTEFACT
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["claim_count"] == len(CLAIMS)


def test_report_and_summary_writers_honour_the_root(tmp_path: Path) -> None:
    report = build_report(
        tmp_path,
        checks=[{"name": "probe", "family": "code", "status": "PASS", "detail": "ok"}],
        configured_run={"tau": 0.30},
        execution_profile={"python": "3.11"},
        manuscript_checks=[],
    )
    write_report(tmp_path, report)
    write_summary(tmp_path, render_summary(report))
    assert (tmp_path / REPORT_ARTEFACT).is_file()
    assert (tmp_path / SUMMARY_ARTEFACT).is_file()
    text = (tmp_path / SUMMARY_ARTEFACT).read_text(encoding="utf-8")
    assert "overall status" in text
    assert "PASS" in text
    assert not (tmp_path / SUMMARY_ARTEFACT).name.endswith(".md")


def test_summarise_checks_separates_code_from_manuscript() -> None:
    summary = summarise_checks(
        [
            {"name": "a", "family": "code", "status": "PASS"},
            {"name": "b", "family": "manuscript", "status": "FAIL"},
        ]
    )
    assert summary["overall_status"] == "PARTIALLY_VERIFIED"
    assert summary["code_status"] == "VERIFIED"


def test_summarise_checks_reports_not_run_as_partial() -> None:
    summary = summarise_checks([{"name": "a", "family": "code", "status": "NOT_RUN"}])
    assert summary["overall_status"] == "PARTIALLY_VERIFIED"
    assert summary["code_status"] == "PARTIALLY_VERIFIED"
    assert summarise_checks([])["overall_status"] == "UNVERIFIED"


def test_manifest_digest_is_order_sensitive() -> None:
    entries = collect_entries(default_root())
    assert entries
    assert manifest_digest(entries) == manifest_digest(entries)
    assert "integrity_manifest.json" not in {entry.path for entry in entries}


def test_manifest_round_trip_in_a_temporary_tree(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    payload = build_manifest(tmp_path, verified=True)
    assert payload["file_count"] == 2  # the manifest excludes itself, and does not exist yet
    (tmp_path / MANIFEST_ARTEFACT).write_text(json.dumps(payload), encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert result["present"]
    assert result["digest_matches"]


def test_manifest_detects_a_changed_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    payload = build_manifest(tmp_path, verified=True)
    (tmp_path / MANIFEST_ARTEFACT).write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    result = verify_manifest(tmp_path)
    assert not result["matches"]
    assert "a.txt" in result["changed"]


def test_integrity_manifest_writer_targets_the_root(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    path = write_integrity_manifest(
        tmp_path, verified=False, notes={"overall_status": "PARTIALLY_VERIFIED"}
    )
    assert path == tmp_path / MANIFEST_ARTEFACT
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["verified"] is False
    assert payload["notes"]["overall_status"] == "PARTIALLY_VERIFIED"

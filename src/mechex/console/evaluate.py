"""Evaluation entry point.

Ref: Sec. 3.9, external-site evaluation with the stratified, shift and abstention analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mechex.assess.abstain import retention_at
from mechex.assess.external import evaluate_external
from mechex.assess.shift import generalisation_table, shift_masks
from mechex.assess.strata import consistency_summary, derive_axes, evaluate_strata, stratum_sizes
from mechex.cohort.construction import StagedCohort, build_cohort
from mechex.estimator import build_model
from mechex.fit.checkpoint import restore_into
from mechex.fit.engine import evaluate_partition
from mechex.provenance.artefacts import default_root
from mechex.seeding import resolve_device
from mechex.settings import EstimatorConfig, load_config
from mechex.utils.io import atomic_write_json, ensure_dir
from mechex.utils.logging import configure_logging, get_logger

_log = get_logger("console.evaluate")


def build_parser() -> argparse.ArgumentParser:
    """Command-line surface of the evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate a trained MechEx checkpoint.")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--overlay", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint to restore")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reports-dir", type=Path, default=None)
    return parser


def _partition_predictions(
    model: torch.nn.Module,
    staged: StagedCohort,
    config: EstimatorConfig,
    partition: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from mechex.fit.engine import _dataloader

    loader = _dataloader(staged, config, partition, shuffle=False, generator=None)
    prediction_parts: list[np.ndarray] = []
    reference_parts: list[np.ndarray] = []
    deviation_parts: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            moved = batch.to(device)
            output = model(moved)
            prediction_parts.append(output.exposure.detach().cpu().numpy())
            deviation_parts.append(output.exposure_std.detach().cpu().numpy())
            if moved.exposure_reference is not None:
                reference_parts.append(moved.exposure_reference.detach().cpu().numpy())
    predictions = np.concatenate(prediction_parts) if prediction_parts else np.zeros(0)
    references = np.concatenate(reference_parts) if reference_parts else np.zeros(0)
    deviations = np.concatenate(deviation_parts) if deviation_parts else np.zeros(0)
    return predictions, references, deviations


def main(argv: list[str] | None = None) -> int:
    """Evaluate a checkpoint across the reported analyses."""
    configure_logging()
    args = build_parser().parse_args(argv)
    root = default_root()
    config_dir = args.config_dir or (root / "configs")
    reports_dir = args.reports_dir or (root / "reports")
    ensure_dir(reports_dir)
    config = load_config(config_dir, overlay=args.overlay)
    staged = build_cohort(config)
    device = resolve_device(args.device)
    model = build_model(config)
    if args.checkpoint is not None:
        restore_into(args.checkpoint, model)
    model = model.to(device)

    external = staged.split.external
    predictions, references, deviations = _partition_predictions(
        model, staged, config, "external", device
    )
    patients = np.asarray(staged.cohort.patient_id, dtype=object)[external]
    metrics = evaluate_external(
        "PROPOSED",
        "MechEx",
        predictions,
        references,
        patients,
        tau=config.decision.tau,
    )
    axes = derive_axes(staged.cohort, external)
    overall = float(evaluate_partition(model, staged, config, "external")["ccc"])
    strata = evaluate_strata(
        predictions, references, axes, overall, config.decision.tau, config.protocol.fdr
    )
    masks = shift_masks(staged.cohort, staged.split.mask("external", len(staged.cohort)))
    shifts = generalisation_table(predictions, references, masks, config.decision.tau)
    labels = (references >= 1.0 + config.decision.tau).astype(np.int64)
    abstention = retention_at(predictions, labels, deviations, config.decision.abstention_quantile)

    report = {
        "external": metrics.as_dict(),
        "strata": [row.as_dict() for row in strata],
        "consistency": consistency_summary(strata),
        "stratum_sizes": stratum_sizes(staged.cohort, external),
        "shifts": shifts,
        "abstention": abstention,
    }
    atomic_write_json(reports_dir / "evaluation_report.json", report)
    _log.info("external concordance %.4f discrimination %.4f", metrics.ccc, metrics.auroc)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

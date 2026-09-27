"""Training entry point.

Ref: Sec. 3.8, the reported training configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mechex.cohort.construction import build_cohort, stage
from mechex.estimator import build_model, count_parameters_millions
from mechex.fit.engine import Trainer
from mechex.fit.repeats import config_for_repeat, repeat_seeds, run_repeats
from mechex.optimise.schedule import describe_reference_schedule
from mechex.provenance.artefacts import default_root
from mechex.seeding import resolve_device
from mechex.settings import load_config
from mechex.utils.io import atomic_write_json, ensure_dir
from mechex.utils.logging import configure_logging, get_logger

_log = get_logger("console.train")


def build_parser() -> argparse.ArgumentParser:
    """Command-line surface of the training entry point."""
    parser = argparse.ArgumentParser(description="Train MechEx on the primary cohort.")
    parser.add_argument(
        "--config-dir", type=Path, default=None, help="directory holding the config groups"
    )
    parser.add_argument(
        "--overlay", type=str, default=None, help="optional overlay YAML at the config root"
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=None, help="where checkpoints and logs are written"
    )
    parser.add_argument("--seed", type=int, default=None, help="override the base seed")
    parser.add_argument("--repeats", type=int, default=None, help="override the number of repeats")
    parser.add_argument("--device", type=str, default="auto", help="cpu, cuda, mps or auto")
    parser.add_argument(
        "--reports-dir", type=Path, default=None, help="where the run report is written"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the configured number of repeats and write a run report."""
    configure_logging()
    args = build_parser().parse_args(argv)
    root = default_root()
    config_dir = args.config_dir or (root / "configs")
    config = load_config(config_dir, overlay=args.overlay)
    runs_dir = args.runs_dir or (root / "runs")
    reports_dir = args.reports_dir or (root / "runs")
    ensure_dir(runs_dir)
    ensure_dir(reports_dir)

    staged = build_cohort(config)
    stage(staged, root / "data")
    device = resolve_device(args.device)
    base_seed = config.optimiser.seed if args.seed is None else args.seed
    n_repeats = config.optimiser.repeats if args.repeats is None else args.repeats

    schedule = describe_reference_schedule(config.optimiser, staged.split.development.size)
    _log.info("schedule: %s", schedule)

    def fit_for_seed(seed: int) -> dict[str, float]:
        repeat_config = config_for_repeat(config, seed)
        model = build_model(repeat_config, seed)
        trainer = Trainer(
            model,
            repeat_config,
            staged,
            device=device,
            checkpoint_dir=runs_dir / f"seed_{seed}",
        )
        result = trainer.fit(seed)
        metrics = {
            "best_validation_ccc": result.best_validation_ccc,
            "n_epochs": float(len(result.history)),
            "final_train_total": result.history[-1].train_total if result.history else float("nan"),
        }
        if result.checkpoint is not None:
            _log.info("checkpoint written for seed %d", seed)
        return metrics

    summary = run_repeats(fit_for_seed, repeat_seeds(base_seed, n_repeats))
    report = {
        "release": "mechex",
        "schedule": schedule,
        "repeats": summary.as_dict(),
        "device": device.type,
        "parameters_millions": count_parameters_millions(build_model(config, base_seed)),
        "cohort": {
            "n_records": len(staged.cohort),
            "split": staged.split.as_dict(),
            "leakage_clean": staged.leakage.clean,
        },
    }
    atomic_write_json(reports_dir / "training_report.json", report)
    _log.info("training complete: %s", summary.mean)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())

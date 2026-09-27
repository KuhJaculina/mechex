"""Console entry points: training, evaluation and verification."""

from mechex.console.audit import main as audit_main
from mechex.console.evaluate import main as evaluate_main
from mechex.console.train import main as train_main

__all__ = ["audit_main", "evaluate_main", "train_main"]

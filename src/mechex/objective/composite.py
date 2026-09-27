"""Composite training objective.

Ref: Sec. 3.8, the model is trained on a concordance-oriented regression loss on the
exposure index, a binary cross-entropy term on the adequacy label and the conservation
penalty of Eq. (1), plus the evidential term that trains the uncertainty head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mechex.conservation.penalty import conservation_penalty, penalty_scale
from mechex.estimator import LOG_FLOOR, MechExBatch, MechExOutput
from mechex.objective.calibration import adequacy_labels, adequacy_loss, evidential_loss
from mechex.objective.concordance import log_concordance_loss
from mechex.settings import EstimatorConfig
from mechex.uncertainty.evidential import EvidentialOutput


@dataclass
class ObjectiveTerms:
    """Per-term values of the composite objective."""

    concordance: torch.Tensor
    decision: torch.Tensor
    conservation: torch.Tensor
    evidential: torch.Tensor
    total: torch.Tensor
    residual_exceedance: torch.Tensor

    def scalar_dict(self) -> dict[str, float]:
        return {
            "concordance": float(self.concordance.detach()),
            "decision": float(self.decision.detach()),
            "conservation": float(self.conservation.detach()),
            "evidential": float(self.evidential.detach()),
            "total": float(self.total.detach()),
            "residual_exceedance": float(self.residual_exceedance.detach()),
        }


class CompositeObjective(nn.Module):
    """Weighted sum of the four reported terms."""

    def __init__(self, config: EstimatorConfig, conservation_weight: float | None = None) -> None:
        super().__init__()
        self.config = config
        self.conservation_weight = (
            config.objective.conservation_weight
            if conservation_weight is None
            else conservation_weight
        )

    def forward(self, output: MechExOutput, batch: MechExBatch) -> ObjectiveTerms:
        if batch.exposure_reference is None:
            raise ValueError("the objective needs a reference exposure")
        log_reference = torch.log(batch.exposure_reference.clamp_min(LOG_FLOOR))
        concordance = self.config.objective.concordance_weight * log_concordance_loss(
            output.log_exposure, log_reference
        )
        if batch.adequacy_label is None:
            labels = adequacy_labels(
                batch.exposure_reference,
                self.config.decision.tau,
                self.config.decision.adequacy_relative,
            )
        else:
            labels = batch.adequacy_label.to(output.adequacy_logit.dtype)
        decision = self.config.objective.decision_weight * adequacy_loss(
            output.adequacy_logit, labels
        )
        conservation = conservation_penalty(
            output.residual,
            tolerance=self.config.conservation.residual_tolerance,
            margin=self.config.conservation.penalty_margin,
            weight=self.conservation_weight,
        )
        evidential = self.config.objective.evidential_weight * evidential_loss(
            log_reference, _as_evidential(output)
        )
        total = concordance + decision + conservation + evidential
        return ObjectiveTerms(
            concordance=concordance,
            decision=decision,
            conservation=conservation,
            evidential=evidential,
            total=total,
            residual_exceedance=penalty_scale(
                output.residual, self.config.conservation.residual_tolerance
            ),
        )


def _as_evidential(output: MechExOutput) -> EvidentialOutput:
    return EvidentialOutput(
        mean=output.log_exposure,
        nu=torch.ones_like(output.log_exposure),
        alpha=torch.ones_like(output.log_exposure),
        beta=output.exposure_variance.clamp_min(1e-8),
        variance=output.exposure_variance,
    )


def objective_without_evidential(config: EstimatorConfig) -> CompositeObjective:
    """Objective with the evidential term switched off, for the no-head ablation."""
    return CompositeObjective(config, conservation_weight=config.objective.conservation_weight)

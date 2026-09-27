"""MechEx: the six-component estimator.

Ref: Sec. 3.6, molecular derivation, vascular line, exposure-coupling module, conservation
head, uncertainty head and decision layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mechex.basis.projection import MolecularBranch
from mechex.conservation.law import implied_delivered_fraction
from mechex.conservation.residual import ConservationHead
from mechex.coupling.additive import build_coupling
from mechex.seeding import set_seed
from mechex.settings import EstimatorConfig
from mechex.uncertainty.heads import UncertaintyHead
from mechex.vascular.projection import (
    MAXIMUM_DELIVERED_FRACTION,
    MINIMUM_DELIVERED_FRACTION,
    VascularBranch,
)

LOG_FLOOR = 1e-6
ADAPTATION_NAMES = ("frozen_head", "full_finetune", "frozen_no_head")


@dataclass
class MechExBatch:
    """Everything the estimator reads for one batch."""

    atoms: torch.Tensor
    coords: torch.Tensor
    atom_mask: torch.Tensor
    bond_index: torch.Tensor
    descriptors: torch.Tensor
    volume: torch.Tensor
    availability_reference: torch.Tensor
    exposure_reference: torch.Tensor | None = None
    adequacy_label: torch.Tensor | None = None

    def to(self, device: torch.device) -> MechExBatch:
        return MechExBatch(
            atoms=self.atoms.to(device),
            coords=self.coords.to(device),
            atom_mask=self.atom_mask.to(device),
            bond_index=self.bond_index.to(device),
            descriptors=self.descriptors.to(device),
            volume=self.volume.to(device),
            availability_reference=self.availability_reference.to(device),
            exposure_reference=None
            if self.exposure_reference is None
            else self.exposure_reference.to(device),
            adequacy_label=None if self.adequacy_label is None else self.adequacy_label.to(device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.atoms.shape[0])


@dataclass
class MechExOutput:
    """All quantities the estimator emits for one batch."""

    partition: torch.Tensor
    delivered_fraction: torch.Tensor
    interaction: torch.Tensor
    availability_reconstruction: torch.Tensor
    availability_reference: torch.Tensor
    residual: torch.Tensor
    implied_fraction: torch.Tensor
    log_exposure: torch.Tensor
    exposure: torch.Tensor
    exposure_variance: torch.Tensor
    exposure_std: torch.Tensor
    adequacy_logit: torch.Tensor
    molecule_embedding: torch.Tensor
    image_embedding: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.exposure.shape[0])


class MechEx(nn.Module):
    """Couples a molecular partition term with a CTA delivery term under a conservation law."""

    def __init__(self, config: EstimatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or EstimatorConfig()
        self.molecular = MolecularBranch(self.config.basis)
        self.vascular = VascularBranch(self.config.vascular)
        self.coupling = build_coupling(self.config.coupling)
        self.conservation = ConservationHead(self.config.conservation)
        width = self.config.coupling.width
        self.uncertainty = UncertaintyHead(self.config.uncertainty, width=width + 1)
        self.adequacy = nn.Linear(width, 1)
        # The branch-removal variants of the ablation still have to produce a product, so a
        # removed branch contributes an optimised constant rather than a silently reused
        # encoder output.
        self.constant_log_partition = nn.Parameter(torch.zeros(1))
        self.constant_delivered = nn.Parameter(torch.tensor(0.60))
        apply_ablation(self)

    def forward(self, batch: MechExBatch) -> MechExOutput:
        if self.config.ablation.molecular_branch:
            partition, molecule_embedding = self.molecular(
                batch.atoms, batch.coords, batch.atom_mask, batch.bond_index, batch.descriptors
            )
        else:
            partition = (torch.nn.functional.softplus(self.constant_log_partition) + 1e-4).expand(
                batch.batch_size
            )
            molecule_embedding = torch.zeros(
                batch.batch_size, self.config.basis.latent_dim, device=partition.device
            )
        if self.config.ablation.imaging_branch:
            delivered, image_embedding = self.vascular(batch.volume)
        else:
            delivered = self.constant_delivered.clamp(
                MINIMUM_DELIVERED_FRACTION, MAXIMUM_DELIVERED_FRACTION
            ).expand(batch.batch_size)
            image_embedding = torch.zeros(
                batch.batch_size, self.config.vascular.latent_dim, device=partition.device
            )
        coupling_output = self.coupling(molecule_embedding, image_embedding)
        interaction = coupling_output.epsilon
        record = self.conservation(partition, delivered, interaction, batch.availability_reference)
        physics = torch.log((record.availability_reconstruction * delivered).clamp_min(LOG_FLOOR))
        head_input = torch.cat([coupling_output.embedding, physics.unsqueeze(-1)], dim=-1)
        evidential = self.uncertainty(head_input)
        log_exposure = evidential.mean
        exposure = torch.exp(log_exposure.clamp(-20.0, 20.0))
        if self.config.ablation.decision_layer:
            adequacy_logit: torch.Tensor = self.adequacy(coupling_output.embedding).squeeze(-1)
        else:
            adequacy_logit = torch.zeros(batch.batch_size, device=exposure.device)
        return MechExOutput(
            partition=partition,
            delivered_fraction=delivered,
            interaction=interaction,
            availability_reconstruction=record.availability_reconstruction,
            availability_reference=record.availability_reference,
            residual=record.residual,
            implied_fraction=record.implied_fraction,
            log_exposure=log_exposure,
            exposure=exposure,
            exposure_variance=evidential.variance,
            exposure_std=evidential.std,
            adequacy_logit=adequacy_logit,
            molecule_embedding=molecule_embedding,
            image_embedding=image_embedding,
        )

    def calibrate(self, batch: MechExBatch) -> None:
        """Fit the isotonic recalibration on a held-out calibration batch."""
        if batch.exposure_reference is None:
            raise ValueError("the calibration batch needs a reference exposure")
        partition, molecule_embedding = self.molecular(
            batch.atoms, batch.coords, batch.atom_mask, batch.bond_index, batch.descriptors
        )
        delivered, image_embedding = self.vascular(batch.volume)
        coupling_output = self.coupling(molecule_embedding, image_embedding)
        record = self.conservation(
            partition, delivered, coupling_output.epsilon, batch.availability_reference
        )
        physics = torch.log((record.availability_reconstruction * delivered).clamp_min(LOG_FLOOR))
        head_input = torch.cat([coupling_output.embedding, physics.unsqueeze(-1)], dim=-1)
        target = torch.log(batch.exposure_reference.clamp_min(LOG_FLOOR))
        self.uncertainty.recalibrate(head_input, target)

    def implied_delivered_fraction(self, output: MechExOutput) -> torch.Tensor:
        """Delivered fraction implied by the reference availability under the fit."""
        return implied_delivered_fraction(
            output.availability_reference, output.partition, output.interaction
        )

    def frozen_encoder_parameters(self) -> int:
        """Parameter count held frozen, reported for the budget statement."""
        return sum(
            parameter.numel()
            for module in (self.molecular.encoder, self.vascular.encoder)
            for parameter in module.parameters()
        )


def apply_ablation(model: MechEx) -> None:
    """Apply the adaptation and initialisation switches of the architectural-substitution tier.

    Ref: Table 2, Tier 2. Full fine-tuning unfreezes both encoders, the frozen-without-
    adaptation variant freezes the projection heads as well, and the random-initialisation
    variant redraws the encoder weights from the seeded generator.
    """
    ablation = model.config.ablation
    if ablation.adaptation not in ADAPTATION_NAMES:
        raise ValueError(f"unknown adaptation setting: {ablation.adaptation}")
    if ablation.adaptation == "full_finetune":
        for encoder in (model.molecular.encoder, model.vascular.encoder):
            for parameter in encoder.parameters():
                parameter.requires_grad_(True)
    elif ablation.adaptation == "frozen_no_head":
        for module in (model.molecular, model.vascular):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if ablation.initialisation == "random":
        for encoder in (model.molecular.encoder, model.vascular.encoder):
            _reinitialise(encoder)


def _reinitialise(module: nn.Module) -> None:
    """Redraw every parameter that has a reset routine, using the process generator."""
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters, in the units the manuscript's parameter column needs."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def count_parameters_millions(module: nn.Module) -> float:
    """Parameter count in millions."""
    return round(count_parameters(module) / 1e6, 1)


def build_model(config: EstimatorConfig, seed: int | None = None) -> MechEx:
    """Seed the process and construct the estimator."""
    set_seed(seed if seed is not None else config.optimiser.seed)
    return MechEx(config)


def frozen_module_names(model: MechEx) -> list[str]:
    """Names of the submodules whose parameters carry no gradient."""
    names = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return sorted({name.split(".")[0] + "." + name.split(".")[1] for name in names})

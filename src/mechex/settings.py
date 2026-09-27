"""Typed configuration for every component.

Each dataclass fixtures one YAML group under ``configs/``. Fields that the manuscript
reports carry its value; the remaining fields are engineering defaults and are labelled
as such in ``configs/`` and in the release's deviation register.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from omegaconf import OmegaConf

T = TypeVar("T")


@dataclass(frozen=True)
class MolecularBasisConfig:
    """Structure branch emitting the partition term.

    Ref: Sec. 3.6, molecular derivation. The feature widths are the lengths of the atom and
    bond feature vocabularies in the graph schema; a unit test asserts they stay in step
    rather than importing the schema here, which would close an import cycle.
    """

    latent_dim: int = 128
    layers: int = 4
    heads: int = 4
    atom_feature_dim: int = 19
    bond_feature_dim: int = 6
    dropout: float = 0.1
    frozen: bool = True


@dataclass(frozen=True)
class VascularEncoderConfig:
    """Vascular branch emitting the delivered fraction.

    Ref: Sec. 3.6, vascular line; Sec. 3.8 input preparation.
    """

    latent_dim: int = 128
    stem_channels: tuple[int, ...] = (16, 32, 64, 96)
    kernel_size: int = 3
    depth: int = 6
    dropout: float = 0.1
    frozen: bool = True
    volume_shape: tuple[int, int, int] = (64, 64, 48)
    isotropic_spacing_mm: float = 1.0
    arterial_window_hu: tuple[float, float] = (100.0, 700.0)


@dataclass(frozen=True)
class CouplingConfig:
    """Bidirectional cross-attention producing the interaction term.

    Ref: Sec. 3.6, exposure-coupling module.
    """

    mode: str = "cross_attention"
    width: int = 128
    heads: int = 4
    dropout: float = 0.1
    interaction_offset: float = -0.5


@dataclass(frozen=True)
class ConservationConfig:
    """Mass-conservation head.

    Ref: Sec. 3.6, Eq. (1).
    """

    residual_tolerance: float = 0.20
    penalty_weight: float = 1.0
    penalty_margin: float = 0.05


@dataclass(frozen=True)
class UncertaintyConfig:
    """Evidential head with post-hoc isotonic recalibration.

    Ref: Sec. 3.6, uncertainty head.
    """

    mode: str = "evidential"
    hidden_dim: int = 64
    isotonic_bins: int = 32
    recalibrate: bool = True


@dataclass(frozen=True)
class DecisionConfig:
    """Adequacy boundary and abstention policy.

    Ref: Sec. 3.5 threshold; Sec. 3.6 decision layer; Sec. 3.10 harm tiering.
    """

    tau: float = 0.30
    adequacy_relative: float = 1.0
    abstention_quantile: float = 0.10
    false_positive_budget: float = 0.05


@dataclass(frozen=True)
class ObjectiveConfig:
    """Composite training objective.

    Ref: Sec. 3.8.
    """

    concordance_weight: float = 1.0
    decision_weight: float = 1.0
    conservation_weight: float = 1.0
    evidential_weight: float = 1.0
    concordance_kind: str = "ccc"


@dataclass(frozen=True)
class OptimiserConfig:
    """AdamW with cosine decay, early stopping on the validation partition.

    Ref: Sec. 3.8.
    """

    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.999)
    epochs: int = 60
    warmup_fraction: float = 0.05
    min_lr_factor: float = 0.01
    grad_clip_norm: float = 1.0
    batch_size: int = 64
    grad_accum: int = 1
    patience: int = 12
    precision: str = "fp32"
    world_size: int = 1
    seed: int = 20260927
    repeats: int = 5


@dataclass(frozen=True)
class ProtocolConfig:
    """Cohort, split and reference-standard protocol.

    Ref: Sec. 3.1-3.5.
    """

    sites_development: tuple[str, ...] = ("A", "B")
    site_external: str = "C"
    development_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    calibration_fraction: float = 0.10
    tier1_assay_records: int = 418
    reference_kappa: float = 0.86
    ccc_threshold: float = 0.75
    auroc_threshold: float = 0.85
    bootstrap_resamples: int = 2000
    ece_bins: int = 15
    fdr: float = 0.05


@dataclass(frozen=True)
class CohortConfig:
    """Primary-cohort shape and the schema-compatible cohort fixture.

    Ref: Sec. 2.1, Sec. 3.2-3.3.
    """

    n_records: int = 8333
    n_molecules: int = 27
    fixture_seed: int = 20260927
    concentration_time_points: int = 9
    arterial_phase: str = "arterial"
    delayed_phase: str = "delayed"


@dataclass(frozen=True)
class AblationConfig:
    """Component switches that reproduce the reported ablation tiers.

    Ref: Table 2, component removal (Tier 1) and architectural substitution (Tier 2). These
    are the removal switches the table names, not general-purpose knobs.
    """

    molecular_branch: bool = True
    imaging_branch: bool = True
    decision_layer: bool = True
    adaptation: str = "frozen_head"
    initialisation: str = "pretrained"


@dataclass(frozen=True)
class StudyConfig:
    """A single reported comparison or ablation."""

    name: str = "main"
    variant: str = "full"
    axis: str = "external"
    molecule_subset: str = "all"
    reported: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EstimatorConfig:
    """Full estimator assembled from the component groups."""

    basis: MolecularBasisConfig = field(default_factory=MolecularBasisConfig)
    vascular: VascularEncoderConfig = field(default_factory=VascularEncoderConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    conservation: ConservationConfig = field(default_factory=ConservationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    optimiser: OptimiserConfig = field(default_factory=OptimiserConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


_GROUPS: dict[str, type[Any]] = {
    "basis": MolecularBasisConfig,
    "vascular": VascularEncoderConfig,
    "coupling": CouplingConfig,
    "conservation": ConservationConfig,
    "uncertainty": UncertaintyConfig,
    "decision": DecisionConfig,
    "objective": ObjectiveConfig,
    "optimiser": OptimiserConfig,
    "protocol": ProtocolConfig,
    "cohort": CohortConfig,
    "ablation": AblationConfig,
}


def _build(group: str, payload: dict[str, Any] | None) -> Any:
    cls = _GROUPS[group]
    if not payload:
        return cls()
    known = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in payload.items() if k in known}
    for key, value in list(filtered.items()):
        if isinstance(value, list):
            filtered[key] = tuple(value)
    return cls(**filtered)


SMOKE_OVERLAY_NAME = "smoke.yaml"


def load_config(config_dir: Path, overlay: str | None = None) -> EstimatorConfig:
    """Read every YAML group under ``config_dir`` into an :class:`EstimatorConfig`.

    ``overlay`` names a YAML file at the root of ``config_dir`` whose group blocks replace
    the defaults. The smoke overlay is the only one shipped, and it exists so the unit
    tests can run a forward and a two-step fit without a GPU.
    """
    payload: dict[str, Any] = {}
    for group in _GROUPS:
        candidate = config_dir / group
        if candidate.is_dir():
            merged: dict[str, Any] = {}
            for yaml_file in sorted(candidate.glob("*.yaml")):
                if yaml_file.name.startswith("_"):
                    continue
                block = OmegaConf.to_container(OmegaConf.load(yaml_file), resolve=True)
                if isinstance(block, dict):
                    for key, value in block.items():
                        merged[str(key)] = value
            payload[group] = merged
        elif (config_dir / f"{group}.yaml").is_file():
            payload[group] = OmegaConf.to_container(
                OmegaConf.load(config_dir / f"{group}.yaml"), resolve=True
            )
    if overlay is not None:
        overlay_path = config_dir / overlay
        if overlay_path.is_file():
            block = OmegaConf.to_container(OmegaConf.load(overlay_path), resolve=True)
            if isinstance(block, dict):
                for raw_group, values in block.items():
                    group_name = str(raw_group)
                    if group_name in _GROUPS and isinstance(values, dict):
                        current = payload.setdefault(group_name, {})
                        if isinstance(current, dict):
                            current.update({str(key): value for key, value in values.items()})
    return config_from_dict(payload)


def load_smoke_config(config_dir: Path) -> EstimatorConfig:
    """The configuration the unit tests and the smoke fit run on."""
    return load_config(config_dir, overlay=SMOKE_OVERLAY_NAME)


def config_group_names() -> tuple[str, ...]:
    """The config group names the loader reads."""
    return tuple(_GROUPS)


def config_from_dict(payload: dict[str, Any]) -> EstimatorConfig:
    """Build an :class:`EstimatorConfig` from a nested mapping."""
    groups = {name: _build(name, payload.get(name)) for name in _GROUPS}
    return EstimatorConfig(**groups)


def config_to_dict(config: EstimatorConfig) -> dict[str, Any]:
    """Serialise a config back to plain python types."""
    return {
        key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
        for key, value in asdict(config).items()
    }


def override(config: EstimatorConfig, dotted: str, value: Any) -> EstimatorConfig:
    """Return a copy of ``config`` with one dotted field replaced."""
    head, _, tail = dotted.partition(".")
    current = config_to_dict(config)
    if not tail:
        current[head] = value
    else:
        node = current
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return config_from_dict(current)

"""Registry of the reconstructed comparators.

Ref: Table 1, one entry per reconstructed baseline with the reported row it stands for.
"""

from __future__ import annotations

from dataclasses import dataclass

from mechex.bench.features import CLINICAL_BLOCK, IMAGING_BLOCK, MOLECULAR_BLOCK
from mechex.provenance.tables import table_1_row

FAMILY_TABULAR = "tabular"
FAMILY_FUSION = "fusion"
FAMILY_MECHANISTIC = "mechanistic"


@dataclass(frozen=True)
class BaselineSpec:
    """One reconstructed comparator."""

    identifier: str
    family: str
    estimator: str
    blocks: tuple[str, ...]
    note: str

    @property
    def name(self) -> str:
        return str(table_1_row(self.identifier)["method"])

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "method": self.name,
            "family": self.family,
            "estimator": self.estimator,
            "blocks": list(self.blocks),
            "note": self.note,
            "reported": {
                key: value
                for key, value in table_1_row(self.identifier).items()
                if key not in {"id", "method"}
            },
        }


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec(
        "B1",
        FAMILY_FUSION,
        "single_branch_mlp",
        (MOLECULAR_BLOCK,),
        "structure branch alone, so the delivery contribution is absent",
    ),
    BaselineSpec(
        "B2",
        FAMILY_TABULAR,
        "descriptor_ridge",
        (MOLECULAR_BLOCK,),
        "ridge regression on the descriptor block",
    ),
    BaselineSpec(
        "B3",
        FAMILY_TABULAR,
        "graph_propagation_ridge",
        (MOLECULAR_BLOCK,),
        "ridge on chemistry-similarity graph propagated descriptors",
    ),
    BaselineSpec(
        "B4",
        FAMILY_MECHANISTIC,
        "pbk_population",
        (MOLECULAR_BLOCK,),
        "Rodgers-Rowland partition with the population perfusion prior",
    ),
    BaselineSpec(
        "B5",
        FAMILY_MECHANISTIC,
        "pbk_high_throughput",
        (MOLECULAR_BLOCK,),
        "the same composition model with population-average binding",
    ),
    BaselineSpec(
        "B6",
        FAMILY_FUSION,
        "single_branch_mlp",
        (IMAGING_BLOCK,),
        "delivery branch alone, so the partition contribution is absent",
    ),
    BaselineSpec(
        "B7",
        FAMILY_TABULAR,
        "radiomics_gradient_boosting",
        (IMAGING_BLOCK,),
        "hand-designed image description with boosted trees",
    ),
    BaselineSpec(
        "B8",
        FAMILY_TABULAR,
        "clinical_logistic",
        (CLINICAL_BLOCK,),
        "clinical variables only, no molecule and no imaging",
    ),
    BaselineSpec(
        "B9",
        FAMILY_MECHANISTIC,
        "mechanism_only_score",
        (MOLECULAR_BLOCK,),
        "the current computational default: mechanism without delivery",
    ),
    BaselineSpec(
        "B10",
        FAMILY_FUSION,
        "early_concatenation",
        (MOLECULAR_BLOCK, IMAGING_BLOCK),
        "the two blocks meet as stacked coordinates only",
    ),
    BaselineSpec(
        "B11",
        FAMILY_FUSION,
        "cross_attention_unconstrained",
        (MOLECULAR_BLOCK, IMAGING_BLOCK),
        "the coupling architecture without the constraint or the uncertainty head",
    ),
    BaselineSpec(
        "B12",
        FAMILY_FUSION,
        "contrastive_alignment",
        (MOLECULAR_BLOCK, IMAGING_BLOCK),
        "representation-level alignment in place of coupling",
    ),
)

BY_IDENTIFIER = {spec.identifier: spec for spec in BASELINES}


def baseline_spec(identifier: str) -> BaselineSpec:
    """One comparator by identifier."""
    if identifier not in BY_IDENTIFIER:
        raise KeyError(identifier)
    return BY_IDENTIFIER[identifier]


def comparators() -> list[dict[str, object]]:
    """Every comparator as a plain mapping, in Table 1 order."""
    return [spec.as_dict() for spec in BASELINES]


def strongest_unconstrained() -> BaselineSpec:
    """The strongest comparator without the conservation constraint."""
    return baseline_spec("B11")


def strongest_single_branch() -> BaselineSpec:
    """The strongest single-branch comparator.

    Ref: Sec. 2.2, the volumetric CTA encoder was the most effective single-branch model.
    """
    return baseline_spec("B6")


def by_family(family: str) -> list[BaselineSpec]:
    """Comparators of one family."""
    return [spec for spec in BASELINES if spec.family == family]

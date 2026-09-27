"""The mechanistic in vitro efflux arm.

Ref: Sec. 3.11, cultured human vascular cells treated with four representative compounds
covering the expected efflux tendency range at three concentrations, with and without
pharmacological efflux inhibition, at least three biological repetitions, read by flow
cytometry and by an independent antibody-based method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_REPLICATES = 3
COMPOUND_COUNT = 4
CONCENTRATION_COUNT = 3
READOUT_METHODS = ("flow_cytometry", "immunofluorescence")


@dataclass(frozen=True)
class AssayCondition:
    """One compound, concentration, inhibitor state and readout."""

    compound: int
    concentration: float
    inhibited: bool
    method: str

    def as_dict(self) -> dict[str, object]:
        return {
            "compound": self.compound,
            "concentration": self.concentration,
            "inhibited": self.inhibited,
            "method": self.method,
        }


@dataclass(frozen=True)
class AssayDesign:
    """The full factorial design of the in vitro arm."""

    concentrations: tuple[float, ...] = (0.5, 2.0, 8.0)

    def conditions(self) -> list[AssayCondition]:
        return [
            AssayCondition(compound, concentration, inhibited, method)
            for compound in range(COMPOUND_COUNT)
            for concentration in self.concentrations
            for inhibited in (False, True)
            for method in READOUT_METHODS
        ]

    @property
    def n_conditions(self) -> int:
        return COMPOUND_COUNT * len(self.concentrations) * 2 * len(READOUT_METHODS)


def accumulation_ratio(
    intracellular_with_inhibition: np.ndarray,
    intracellular_without_inhibition: np.ndarray,
) -> np.ndarray:
    """Ratio of intracellular concentration with inhibition to without.

    A ratio above one means the inhibitor raised intracellular accumulation, which is the
    direction an efflux substrate has to show for the transporter to be limiting.
    """
    with_inhibitor = np.asarray(intracellular_with_inhibition, dtype=np.float64)
    without = np.asarray(intracellular_without_inhibition, dtype=np.float64)
    if with_inhibitor.shape != without.shape:
        raise ValueError("both readouts must share a shape")
    return with_inhibitor / np.clip(without, 1e-9, None)


def assay_fixture(
    efflux_propensity: np.ndarray,
    reference_ratio_low: float = 1.08,
    reference_ratio_high: float = 3.42,
    replicates: int = MIN_REPLICATES,
    seed: int = 20260927,
    design: AssayDesign | None = None,
) -> dict[str, np.ndarray]:
    """Assay readouts consistent with a supplied efflux propensity per compound.

    The generator makes the accumulation ratio increase with the compound's efflux tendency,
    which is the relationship the manuscript reports having measured; the release can then
    check that its own rank-correlation machinery recovers it.
    """
    propensity = np.asarray(efflux_propensity, dtype=np.float64)
    if propensity.size != COMPOUND_COUNT:
        raise ValueError(f"the design covers {COMPOUND_COUNT} compounds")
    resolved = design or AssayDesign()
    generator = np.random.default_rng(seed)
    span = max(float(propensity.max() - propensity.min()), 1e-9)
    normalised = (propensity - propensity.min()) / span
    ratio_target = reference_ratio_low + (reference_ratio_high - reference_ratio_low) * normalised

    rows: list[dict[str, object]] = []
    with_inhibitor: list[float] = []
    without_inhibitor: list[float] = []
    for condition in resolved.conditions():
        base = 1.0 + 0.4 * ratio_target[condition.compound]
        if condition.inhibited:
            base *= ratio_target[condition.compound]
        for replicate in range(replicates):
            value = base * (1.0 + generator.normal(0.0, 0.05))
            rows.append(
                {
                    "compound": condition.compound,
                    "concentration": condition.concentration,
                    "inhibited": condition.inhibited,
                    "method": condition.method,
                    "replicate": replicate,
                    "value": float(value),
                }
            )
            if condition.inhibited:
                with_inhibitor.append(float(value))
            else:
                without_inhibitor.append(float(value))
    return {
        "compound": np.array([row["compound"] for row in rows], dtype=np.int64),
        "concentration": np.array([row["concentration"] for row in rows], dtype=np.float64),
        "inhibited": np.array([row["inhibited"] for row in rows], dtype=bool),
        "method": np.array([row["method"] for row in rows], dtype=object),
        "replicate": np.array([row["replicate"] for row in rows], dtype=np.int64),
        "value": np.array([row["value"] for row in rows], dtype=np.float64),
    }


def per_compound_ratio(table: dict[str, np.ndarray]) -> np.ndarray:
    """Mean accumulation ratio per compound over the whole design."""
    compounds = np.asarray(table["compound"], dtype=np.int64)
    inhibited = np.asarray(table["inhibited"], dtype=bool)
    values = np.asarray(table["value"], dtype=np.float64)
    ratios = np.zeros(COMPOUND_COUNT, dtype=np.float64)
    for compound in range(COMPOUND_COUNT):
        selected = compounds == compound
        with_inhibitor = values[selected & inhibited]
        without = values[selected & ~inhibited]
        ratios[compound] = float(with_inhibitor.mean() / max(without.mean(), 1e-9))
    return ratios


def method_agreement(table: dict[str, np.ndarray]) -> dict[str, float]:
    """Agreement between the two readout methods.

    Ref: Sec. 3.11, the main comparison was validated with an independent antibody-based
    method so the conclusion does not rest on one assay type.
    """
    methods = np.asarray(table["method"], dtype=object)
    values = np.asarray(table["value"], dtype=np.float64)
    first = values[methods == READOUT_METHODS[0]]
    second = values[methods == READOUT_METHODS[1]]
    size = min(first.size, second.size)
    if size == 0:
        return {"n": 0.0, "correlation": float("nan"), "mean_absolute_difference": float("nan")}
    correlation = float(np.corrcoef(first[:size], second[:size])[0, 1])
    return {
        "n": float(size),
        "correlation": correlation,
        "mean_absolute_difference": float(np.mean(np.abs(first[:size] - second[:size]))),
    }


def validate_efflux_hypothesis(
    predicted_propensity: np.ndarray,
    measured_ratio: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 20260927,
) -> dict[str, float]:
    """Rank correlation between predicted efflux propensity and measured accumulation.

    Ref: Sec. 3.11 and Sec. 4, the reported rho of 0.87 with P of 0.002.
    """
    from mechex.quantify.correlation import permutation_p_value, spearman_rho

    rho = spearman_rho(predicted_propensity, measured_ratio)
    p_value = permutation_p_value(
        predicted_propensity, measured_ratio, n_permutations=n_permutations, seed=seed
    )
    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "n": float(np.asarray(measured_ratio).size),
    }


def required_sample_size(
    rho: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Sample size for detecting a correlation of ``rho`` at the given error rates.

    Fisher's transform gives the standard construction; the value is reported so a reader
    can see whether the four-compound design is powered for the reported effect.
    """
    from scipy import stats

    if not -1.0 < rho < 1.0 or rho == 0.0:
        raise ValueError("rho must lie strictly inside the unit interval and be non-zero")
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    return float(((z_alpha + z_power) / np.arctanh(rho)) ** 2 + 3.0)


def design_summary() -> dict[str, float | list[str]]:
    """Design quantities of the in vitro arm, for the report."""
    design = AssayDesign()
    return {
        "compounds": float(COMPOUND_COUNT),
        "concentrations": float(CONCENTRATION_COUNT),
        "min_replicates": float(MIN_REPLICATES),
        "readout_methods": list(READOUT_METHODS),
        "conditions": float(design.n_conditions),
    }

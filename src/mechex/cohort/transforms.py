"""The five explicit transforms from source fields to the exposure label.

Ref: Sec. 3.3, assay concentrations are unit-normalised, concentration records are
time-aligned to the imaging episode, the perfused-fraction estimate is converted into a
dimensionless delivery term, the exposure index is the concentration-time integral scaled
by that term, and the result is dichotomised against the frozen threshold.
"""

from __future__ import annotations

import numpy as np

from mechex.decision.threshold import concentration_time_integral, is_adequate
from mechex.quantify.concordance import TAU

ASSAY_UNIT_SCALE = 1000.0


def unit_normalise_concentrations(
    concentrations: np.ndarray, unit_scale: float = ASSAY_UNIT_SCALE
) -> np.ndarray:
    """Transform 1: bring assay concentrations onto one unit scale."""
    values = np.asarray(concentrations, dtype=np.float64)
    if unit_scale <= 0:
        raise ValueError("the unit scale must be positive")
    return values / float(unit_scale)


def time_align(
    times: np.ndarray,
    concentrations: np.ndarray,
    episode_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform 2: shift a concentration record onto the imaging episode clock."""
    t = np.asarray(times, dtype=np.float64).ravel()
    c = np.asarray(concentrations, dtype=np.float64).ravel()
    if t.shape != c.shape:
        raise ValueError("times and concentrations must share a shape")
    order = np.argsort(t)
    shifted = t[order] - float(episode_time)
    return shifted, c[order]


def perfused_fraction_to_delivery(
    fraction: np.ndarray, low: float = 0.05, high: float = 0.95
) -> np.ndarray:
    """Transform 3: convert the perfused fraction into a dimensionless delivery term."""
    values = np.asarray(fraction, dtype=np.float64)
    return np.clip(values, low, high)


def exposure_index(
    integral: np.ndarray,
    delivery: np.ndarray,
    reference_integral: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Transform 4: exposure index as the integral scaled by the delivery term.

    Ref: Sec. 3.3. Dividing by the molecule's reference integral makes the index
    dimensionless, which is what allows one adequacy threshold to serve every molecule.
    """
    values = np.asarray(integral, dtype=np.float64)
    term = np.asarray(delivery, dtype=np.float64)
    reference = np.asarray(reference_integral, dtype=np.float64)
    if np.any(reference <= 0):
        raise ValueError("the reference integral must be positive")
    return values / np.maximum(reference, TAU) * term


def dichotomise(exposure: np.ndarray, tau: float) -> np.ndarray:
    """Transform 5: dichotomise against the frozen threshold.

    Ref: Sec. 3.5, the threshold is the minimum clinically important difference fixed on
    the development split.
    """
    return is_adequate(np.asarray(exposure, dtype=np.float64), tau)


def recompute_integrals(times: np.ndarray, concentrations: np.ndarray) -> np.ndarray:
    """Trapezoidal integral for every row of a concentration table."""
    t = np.asarray(times, dtype=np.float64)
    c = np.asarray(concentrations, dtype=np.float64)
    if t.ndim == 1:
        return np.array([concentration_time_integral(t, c[index]) for index in range(c.shape[0])])
    return np.array(
        [concentration_time_integral(t[index], c[index]) for index in range(c.shape[0])]
    )


def molecule_reference(
    raw_exposure: np.ndarray,
    molecule_index: np.ndarray,
    n_molecules: int,
    development_mask: np.ndarray,
) -> np.ndarray:
    """Per-molecule reference exposure measured on the development split only.

    The normalisation statistic is estimated on development records and then applied
    everywhere; estimating it on the full cohort would leak test information into the
    label.
    """
    values = np.asarray(raw_exposure, dtype=np.float64)
    molecules = np.asarray(molecule_index, dtype=np.int64)
    allowed = np.asarray(development_mask, dtype=bool)
    reference = np.ones(n_molecules, dtype=np.float64)
    for index in range(n_molecules):
        selected = (molecules == index) & allowed
        if selected.any():
            reference[index] = float(np.median(values[selected]))
        elif (molecules == index).any():
            reference[index] = float(np.median(values[molecules == index]))
        else:
            reference[index] = 1.0
    return reference


def relative_exposure(
    raw_exposure: np.ndarray,
    molecule_index: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Apply the per-molecule normalisation."""
    values = np.asarray(raw_exposure, dtype=np.float64)
    molecules = np.asarray(molecule_index, dtype=np.int64)
    return values / np.maximum(np.asarray(reference, dtype=np.float64)[molecules], TAU)


def transform_chain(
    times: np.ndarray,
    concentrations: np.ndarray,
    delivery: np.ndarray,
    episode_time: float,
    reference_integral: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run the five transforms in order and return each intermediate.

    Returning the intermediates rather than only the label is what lets the construction
    record trace a derived variable back to its source field. Alignment is applied per
    record: a shared time grid says nothing about each record's own episode clock, so a
    table-wide shift would silently mis-date every later record.
    """
    resolved_times = np.atleast_2d(np.asarray(times, dtype=np.float64))
    resolved_concentrations = np.atleast_2d(np.asarray(concentrations, dtype=np.float64))
    n_records = resolved_concentrations.shape[0]
    if resolved_times.shape[0] == 1 and n_records > 1:
        resolved_times = np.repeat(resolved_times, n_records, axis=0)
    if resolved_times.shape != resolved_concentrations.shape:
        raise ValueError("times and concentrations must describe the same records and grid")

    aligned_times = np.empty_like(resolved_times)
    aligned_concentrations = np.empty_like(resolved_concentrations)
    for index in range(n_records):
        shifted, ordered = time_align(
            resolved_times[index], resolved_concentrations[index], episode_time
        )
        aligned_times[index] = shifted
        aligned_concentrations[index] = ordered

    normalised = unit_normalise_concentrations(aligned_concentrations)
    integral = np.array(
        [
            concentration_time_integral(aligned_times[index], normalised[index])
            for index in range(n_records)
        ]
    )
    term = perfused_fraction_to_delivery(np.asarray(delivery, dtype=np.float64))
    index_values = exposure_index(integral, term, reference_integral)
    return {
        "aligned_times": aligned_times,
        "normalised_concentrations": normalised,
        "integral": integral,
        "delivery": term,
        "exposure": index_values,
    }

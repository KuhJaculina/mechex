"""Concordance correlation coefficient.

Ref: Sec. 3.9, Lin's concordance coefficient for agreement between predicted and observed
local exposure.
"""

from __future__ import annotations

import numpy as np

TAU = 1e-12


def lin_ccc(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Lin's concordance correlation coefficient.

    Population moments are used, matching the estimator's definition rather than the
    sample-corrected form, so the value is the one the manuscript's interval is built on.
    """
    x = np.asarray(predicted, dtype=np.float64).ravel()
    y = np.asarray(reference, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("predicted and reference must have the same length")
    if x.size < 2:
        return float("nan")
    mean_x = float(x.mean())
    mean_y = float(y.mean())
    var_x = float(((x - mean_x) ** 2).mean())
    var_y = float(((y - mean_y) ** 2).mean())
    covariance = float(((x - mean_x) * (y - mean_y)).mean())
    denominator = var_x + var_y + (mean_x - mean_y) ** 2
    if denominator < TAU:
        return float("nan")
    return float(2.0 * covariance / denominator)


def ccc_components(predicted: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Decomposition into precision, accuracy and the coefficient itself.

    Exposed because a fall in concordance can come from scatter or from a systematic
    offset, and the two need different remedies.
    """
    x = np.asarray(predicted, dtype=np.float64).ravel()
    y = np.asarray(reference, dtype=np.float64).ravel()
    mean_x = float(x.mean())
    mean_y = float(y.mean())
    var_x = float(((x - mean_x) ** 2).mean())
    var_y = float(((y - mean_y) ** 2).mean())
    covariance = float(((x - mean_x) * (y - mean_y)).mean())
    precision = covariance / np.sqrt(max(var_x * var_y, TAU))
    accuracy = (
        2.0 * np.sqrt(max(var_x * var_y, TAU)) / max(var_x + var_y + (mean_x - mean_y) ** 2, TAU)
    )
    return {
        "ccc": lin_ccc(x, y),
        "pearson": precision,
        "bias_correction": accuracy,
        "mean_predicted": mean_x,
        "mean_reference": mean_y,
    }


def lin_ccc_ci(x: np.ndarray, y: np.ndarray, confidence: float = 0.95) -> dict[str, float]:
    """Confidence interval for the concordance coefficient.

    The interval comes from the Fisher transform of the correlation component, which is
    the standard normal-approximation construction; the release therefore labels it an
    approximation and reports the bootstrap interval alongside it where the manuscript
    asks for one.
    """
    from scipy import stats

    components = ccc_components(x, y)
    rho = min(max(components["pearson"], -1.0 + 1e-9), 1.0 - 1e-9)
    n = int(np.asarray(x).size)
    if n < 4:
        return {"estimate": components["ccc"], "lower": float("nan"), "upper": float("nan")}
    transformed = np.arctanh(rho)
    error = 1.0 / np.sqrt(n - 3)
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    low_rho = float(np.tanh(transformed - z * error))
    high_rho = float(np.tanh(transformed + z * error))
    scale = components["bias_correction"]
    return {
        "estimate": components["ccc"],
        "lower": float(min(low_rho * scale, components["ccc"])),
        "upper": float(max(high_rho * scale, components["ccc"])),
    }


def ccc_decrement(full: float, reduced: float) -> float:
    """Sign-positive concordance gain of the coupled model over a comparator."""
    return float(full - reduced)

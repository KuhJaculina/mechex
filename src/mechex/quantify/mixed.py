"""Mixed-effects logistic contrast with a random intercept per reader.

Ref: Sec. 2.6, the team's gain over the best isolated arm is tested with a mixed-effects
logistic model carrying a random intercept per reader.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

N_QUADRATURE = 32
# The random-effect scale is bounded because with one observation per group it is
# unidentified and the marginal likelihood is monotone in it, which would let the fit run
# away to an arbitrarily large scale instead of reporting a flat direction.
LOG_SIGMA_LOWER = -8.0
LOG_SIGMA_UPPER = 2.5


@dataclass(frozen=True)
class MixedEffectResult:
    """Fixed-effect contrast, its standard error, and the random-effect scale."""

    coefficient: float
    standard_error: float
    z_statistic: float
    p_value: float
    intercept: float
    random_effect_sd: float
    n_groups: int
    n_observations: int
    log_likelihood: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "coefficient": self.coefficient,
            "standard_error": self.standard_error,
            "z_statistic": self.z_statistic,
            "p_value": self.p_value,
            "intercept": self.intercept,
            "random_effect_sd": self.random_effect_sd,
            "n_groups": self.n_groups,
            "n_observations": self.n_observations,
            "log_likelihood": self.log_likelihood,
        }


def _quadrature(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(n_points)
    return nodes, weights / np.sqrt(np.pi)


def _group_log_likelihood(
    design: np.ndarray,
    outcome: np.ndarray,
    beta: np.ndarray,
    sigma: float,
    nodes: np.ndarray,
    weights: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
) -> float:
    """Marginal log-likelihood of every group, evaluated on the quadrature grid.

    The grid is built as one ``[nodes, observations]`` block and reduced per group with a
    weighted bincount, so the cost is linear in the number of records rather than a python
    loop over groups. The reduction is shifted by each group's own maximum before the
    exponential, which is the log-sum-exp form and keeps a group with a very negative
    likelihood from underflowing to zero.
    """
    linear = design @ beta
    eta = linear[None, :] + (sigma * np.sqrt(2.0) * nodes)[:, None]
    log_terms = outcome[None, :] * eta - np.logaddexp(0.0, eta)
    group_sums = np.empty((nodes.size, n_groups), dtype=np.float64)
    for node_index in range(nodes.size):
        group_sums[node_index] = np.bincount(
            group_index, weights=log_terms[node_index], minlength=n_groups
        )
    maxima = group_sums.max(axis=0)
    shifted = group_sums - maxima
    combined = np.log(np.sum(weights[:, None] * np.exp(shifted), axis=0)) + maxima
    return float(np.sum(combined))


def _negative_log_likelihood(
    parameters: np.ndarray,
    design: np.ndarray,
    outcome: np.ndarray,
    group_index: np.ndarray,
    n_groups: int,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    beta = parameters[:-1]
    sigma = float(np.exp(np.clip(parameters[-1], LOG_SIGMA_LOWER, LOG_SIGMA_UPPER)))
    return -_group_log_likelihood(
        design, outcome, beta, sigma, nodes, weights, group_index, n_groups
    )


def _numerical_hessian(
    function: Callable[[np.ndarray], float],
    point: np.ndarray,
    step: float = 1e-3,
) -> np.ndarray:
    size = point.size
    hessian = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(i, size):
            offset_i = np.zeros(size)
            offset_j = np.zeros(size)
            offset_i[i] = step
            offset_j[j] = step
            value = (
                function(point + offset_i + offset_j)
                - function(point + offset_i - offset_j)
                - function(point - offset_i + offset_j)
                + function(point - offset_i - offset_j)
            ) / (4.0 * step * step)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def mixed_effects_logistic(
    predictor: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    n_quadrature: int = N_QUADRATURE,
) -> MixedEffectResult:
    """Fit a random-intercept logistic model and return the predictor's contrast.

    The marginal likelihood is integrated with Gauss-Hermite quadrature, so the reported
    standard error accounts for the within-reader correlation rather than treating every
    reading as an independent case.
    """
    x = np.asarray(predictor, dtype=np.float64).ravel()
    y = np.asarray(outcome, dtype=np.float64).ravel()
    g = np.asarray(groups).ravel()
    if not (x.size == y.size == g.size):
        raise ValueError("predictor, outcome and groups must share a length")
    labels, group_index = np.unique(g, return_inverse=True)
    design = np.column_stack([np.ones_like(x), x])
    nodes, weights = _quadrature(n_quadrature)

    def objective(parameters: np.ndarray) -> float:
        return _negative_log_likelihood(
            parameters, design, y, group_index, labels.size, nodes, weights
        )

    start = np.zeros(design.shape[1] + 1)
    start[-1] = np.log(0.5)
    result = optimize.minimize(
        objective, start, method="BFGS", options={"gtol": 1e-7, "maxiter": 2000}
    )
    optimum = np.asarray(result.x, dtype=np.float64)
    hessian = _numerical_hessian(objective, optimum)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    standard_errors = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    coefficient = float(optimum[1])
    standard_error = float(standard_errors[1])
    z = coefficient / standard_error if standard_error > 0 else float("nan")
    p_value = float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return MixedEffectResult(
        coefficient=coefficient,
        standard_error=standard_error,
        z_statistic=float(z),
        p_value=p_value,
        intercept=float(optimum[0]),
        random_effect_sd=float(np.exp(optimum[-1])),
        n_groups=int(labels.size),
        n_observations=int(x.size),
        log_likelihood=float(-result.fun),
    )


def skill_gradient(
    correctness: np.ndarray,
    assisted: np.ndarray,
    skill: np.ndarray,
) -> dict[str, float]:
    """Gain from assistance in the least- and most-skilled reader thirds.

    Ref: Sec. 2.6, the highest gain for the least skilled group and the lowest for the
    most skilled group.
    """
    values = np.asarray(correctness, dtype=np.float64)
    flag = np.asarray(assisted, dtype=bool)
    level = np.asarray(skill, dtype=np.float64)
    if not (values.size == flag.size == level.size):
        raise ValueError("correctness, assisted and skill must share a length")
    order = np.argsort(level)
    third = max(1, values.size // 3)
    low = order[:third]
    high = order[-third:]
    low_gain = float(values[low][flag[low]].mean() - values[low][~flag[low]].mean())
    high_gain = float(values[high][flag[high]].mean() - values[high][~flag[high]].mean())
    return {
        "lowest_third_gain": low_gain,
        "highest_third_gain": high_gain,
        "overall_gain": float(values[flag].mean() - values[~flag].mean()),
    }

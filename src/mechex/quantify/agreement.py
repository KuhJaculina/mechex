"""Inter-rater agreement and paired classification contrast.

Ref: Sec. 2.1, double adjudication of the reference standard; Sec. 2.6, inter-reader
agreement and the McNemar comparison used for classification agreement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class KappaResult:
    """Cohen's kappa with its observed and expected agreement."""

    kappa: float
    observed: float
    expected: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "kappa": self.kappa,
            "observed_agreement": self.observed,
            "expected_agreement": self.expected,
            "n": self.n,
        }


def cohen_kappa(first: np.ndarray, second: np.ndarray) -> KappaResult:
    """Cohen's kappa for two raters over a shared categorical label set."""
    a = np.asarray(first, dtype=np.int64).ravel()
    b = np.asarray(second, dtype=np.int64).ravel()
    if a.size != b.size:
        raise ValueError("both ratings must have the same length")
    if a.size == 0:
        return KappaResult(float("nan"), float("nan"), float("nan"), 0)
    labels = np.unique(np.concatenate([a, b]))
    observed = float(np.mean(a == b))
    expected = 0.0
    for label in labels:
        expected += float(np.mean(a == label)) * float(np.mean(b == label))
    if abs(1.0 - expected) < 1e-12:
        return KappaResult(float("nan"), observed, expected, int(a.size))
    return KappaResult(
        kappa=float((observed - expected) / (1.0 - expected)),
        observed=observed,
        expected=float(expected),
        n=int(a.size),
    )


def kappa_with_ci(
    first: np.ndarray, second: np.ndarray, confidence: float = 0.95
) -> dict[str, float]:
    """Kappa with a normal-approximation interval."""
    result = cohen_kappa(first, second)
    n = max(result.n, 1)
    if not np.isfinite(result.kappa):
        return {**result.as_dict(), "lower": float("nan"), "upper": float("nan")}
    standard_error = float(
        np.sqrt(
            max(result.observed * (1.0 - result.observed), 0.0)
            / (n * max(1.0 - result.expected, 1e-6) ** 2)
        )
    )
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    return {
        **result.as_dict(),
        "lower": float(result.kappa - z * standard_error),
        "upper": float(result.kappa + z * standard_error),
    }


@dataclass(frozen=True)
class McNemarResult:
    """McNemar contrast between two classifiers on the same cases."""

    discordant_a: int
    discordant_b: int
    statistic: float
    p_value: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "a_correct_b_wrong": self.discordant_a,
            "b_correct_a_wrong": self.discordant_b,
            "statistic": self.statistic,
            "p_value": self.p_value,
        }


def mcnemar_test(
    first_correct: np.ndarray,
    second_correct: np.ndarray,
    exact: bool = True,
) -> McNemarResult:
    """McNemar test with the exact binomial form for small discordant counts."""
    a = np.asarray(first_correct, dtype=bool).ravel()
    b = np.asarray(second_correct, dtype=bool).ravel()
    if a.size != b.size:
        raise ValueError("both correctness vectors must have the same length")
    n_a = int(np.sum(a & ~b))
    n_b = int(np.sum(b & ~a))
    discordant = n_a + n_b
    if discordant == 0:
        return McNemarResult(n_a, n_b, 0.0, 1.0)
    if exact:
        p_value = float(min(1.0, 2.0 * stats.binom.cdf(min(n_a, n_b), discordant, 0.5)))
        statistic = float(abs(n_a - n_b))
    else:
        statistic = float((abs(n_a - n_b) - 1.0) ** 2 / discordant)
        p_value = float(stats.chi2.sf(statistic, df=1))
    return McNemarResult(n_a, n_b, statistic, p_value)


def correctness(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean correctness of a thresholded classifier."""
    predicted = (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
    correct: np.ndarray = predicted == np.asarray(labels, dtype=np.int64)
    return correct

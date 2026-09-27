"""Silent-mode implementation study endpoints.

Ref: Sec. 2.9 and Sec. 3.12, integration of the workflow, time effects and adoption rate
were the pre-specified endpoints, with recommendations recorded but not acted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PRESPECIFIED_ENDPOINTS = ("workflow_integration", "time_effect", "adoption_rate")


@dataclass
class DeploymentLog:
    """Per-assessment records from the silent-mode period."""

    site: np.ndarray
    week: np.ndarray
    assessment_minutes_before: np.ndarray
    assessment_minutes_after: np.ndarray
    recommendation_consulted: np.ndarray
    recommendation_approved: np.ndarray
    relevance_rated: np.ndarray
    relevance_positive: np.ndarray

    def __len__(self) -> int:
        return int(self.site.shape[0])


@dataclass
class SilentModeReport:
    """Every reported silent-mode quantity in one mapping."""

    n_assessments: int
    n_sites: int
    n_weeks: int
    consulted_percent: float
    approved_percent: float
    adoption_of_consulted_percent: float
    adoption_of_all_percent: float
    median_minutes_before: float
    median_minutes_after: float
    time_reduction_percent: float
    assessments_per_site: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "n_assessments": self.n_assessments,
            "n_sites": self.n_sites,
            "n_weeks": self.n_weeks,
            "consulted_percent": self.consulted_percent,
            "approved_percent": self.approved_percent,
            "adoption_of_consulted_percent": self.adoption_of_consulted_percent,
            "adoption_of_all_percent": self.adoption_of_all_percent,
            "median_minutes_before": self.median_minutes_before,
            "median_minutes_after": self.median_minutes_after,
            "time_reduction_percent": self.time_reduction_percent,
            "assessments_per_site": dict(self.assessments_per_site),
        }


def deployment_fixture(
    n_assessments: int = 1344,
    n_sites: int = 3,
    n_weeks: int = 14,
    seed: int = 20260927,
    consulted_rate: float = 0.763,
    approved_rate: float = 0.689,
    minutes_before: float = 14.2,
    minutes_after: float = 6.8,
) -> DeploymentLog:
    """Deployment log with the reported adoption and timing levels.

    The assessment count is a fixtureed input rather than a transcribed manuscript value,
    because the released PDF renders that count illegibly.
    """
    generator = np.random.default_rng(seed)
    sites = np.array([f"site_{index + 1}" for index in range(n_sites)], dtype=object)
    site_column = sites[generator.integers(0, n_sites, size=n_assessments)]
    week_column = generator.integers(1, n_weeks + 1, size=n_assessments)
    consulted = generator.random(n_assessments) < consulted_rate
    approved = consulted & (generator.random(n_assessments) < approved_rate)
    before = np.clip(generator.normal(minutes_before, 2.4, size=n_assessments), 2.0, None)
    after = np.clip(generator.normal(minutes_after, 1.3, size=n_assessments), 1.0, None)
    relevance = generator.random(n_assessments) < 0.714
    return DeploymentLog(
        site=site_column,
        week=week_column,
        assessment_minutes_before=before,
        assessment_minutes_after=after,
        recommendation_consulted=consulted,
        recommendation_approved=approved,
        relevance_rated=np.ones(n_assessments, dtype=bool),
        relevance_positive=relevance,
    )


def summarise_deployment(log: DeploymentLog) -> SilentModeReport:
    """Aggregate a deployment log into the reported quantities."""
    consulted = np.asarray(log.recommendation_consulted, dtype=bool)
    approved = np.asarray(log.recommendation_approved, dtype=bool)
    before = float(np.median(log.assessment_minutes_before))
    after = float(np.median(log.assessment_minutes_after))
    sites, counts = np.unique(log.site, return_counts=True)
    return SilentModeReport(
        n_assessments=len(log),
        n_sites=int(sites.size),
        n_weeks=int(np.unique(log.week).size),
        consulted_percent=float(100.0 * consulted.mean()),
        approved_percent=float(100.0 * approved.mean()),
        adoption_of_consulted_percent=float(100.0 * approved.sum() / max(consulted.sum(), 1)),
        adoption_of_all_percent=float(100.0 * approved.sum() / max(len(log), 1)),
        median_minutes_before=before,
        median_minutes_after=after,
        time_reduction_percent=float(100.0 * (before - after) / max(before, 1e-9)),
        assessments_per_site={
            str(site): int(count) for site, count in zip(sites, counts, strict=True)
        },
    )


def weekly_adoption(log: DeploymentLog) -> list[dict[str, float]]:
    """Adoption rate per week, the reported cadence.

    Ref: Sec. 3.12, adoption is reported weekly and for each site separately.
    """
    weeks = np.asarray(log.week, dtype=np.int64)
    rows: list[dict[str, float]] = []
    for week in np.unique(weeks):
        selected = weeks == week
        consulted = np.asarray(log.recommendation_consulted, dtype=bool)[selected]
        approved = np.asarray(log.recommendation_approved, dtype=bool)[selected]
        rows.append(
            {
                "week": float(week),
                "n": float(selected.sum()),
                "consulted_percent": float(100.0 * consulted.mean()),
                "approved_percent": float(100.0 * approved.mean()),
            }
        )
    return rows


def per_site_adoption(log: DeploymentLog) -> list[dict[str, float]]:
    """Adoption rate per site."""
    sites = np.asarray(log.site, dtype=object)
    consulted = np.asarray(log.recommendation_consulted, dtype=bool)
    approved = np.asarray(log.recommendation_approved, dtype=bool)
    rows: list[dict[str, float]] = []
    for site in np.unique(sites):
        selected = sites == site
        rows.append(
            {
                "site_index": float(np.flatnonzero(np.unique(sites) == site)[0]),
                "n": float(selected.sum()),
                "consulted_percent": float(100.0 * consulted[selected].mean()),
                "approved_percent": float(100.0 * approved[selected].mean()),
            }
        )
    return rows


def shortlist_agreement(
    model_shortlist: set[int],
    mechanism_shortlist: set[int],
    expert_shortlist: set[int],
) -> dict[str, float]:
    """Agreement of each shortlist with the independent expert panel.

    Ref: Sec. 2.9, the model-assisted shortlist agreed with the expert panel at 84.1% and
    the mechanism-based shortlist at 61.2%.
    """
    return {
        "model_agreement": float(
            len(model_shortlist & expert_shortlist) / max(len(expert_shortlist), 1)
        ),
        "mechanism_agreement": float(
            len(mechanism_shortlist & expert_shortlist) / max(len(expert_shortlist), 1)
        ),
        "expert_shortlist_size": float(len(expert_shortlist)),
    }


def missed_candidates(
    expert_shortlist: set[int],
    shortlist: set[int],
    adequate: set[int],
) -> int:
    """Adequate candidates the expert panel selected but the shortlist omitted."""
    return len((expert_shortlist & adequate) - shortlist)


def endpoint_coverage(summary: SilentModeReport) -> dict[str, float]:
    """Whether each pre-specified endpoint produced a value."""
    return {
        "workflow_integration": float(summary.n_sites),
        "time_effect": float(summary.time_reduction_percent),
        "adoption_rate": float(summary.adoption_of_all_percent),
    }

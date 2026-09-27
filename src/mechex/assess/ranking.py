"""Exposure-guided repurposing ranking.

Ref: Sec. 2.8, fourteen molecules moved at least five rank positions, the rank correlation
between the exposure ranking and the mechanism-only ranking was 0.31, and limiting the
shortlist to molecules predicted to give adequate exposure removed nine candidates that the
mechanism score had placed in the top ten while promoting six from outside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.quantify.correlation import moved_at_least, rank_movement, rank_positions, spearman_rho

SHORTLIST_DEFAULT = 10


@dataclass(frozen=True)
class RankingComparison:
    """The two rankings and the movement between them."""

    molecule_index: np.ndarray
    exposure_score: np.ndarray
    mechanism_score: np.ndarray
    exposure_rank: np.ndarray
    mechanism_rank: np.ndarray
    movement: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "n_molecules": int(self.molecule_index.size),
            "spearman_rho": float(spearman_rho(self.exposure_score, self.mechanism_score)),
            "moved_at_least_five": int(np.sum(np.abs(self.movement) >= 5)),
            "largest_upward_move": int(np.max(-self.movement)) if self.movement.size else 0,
            "largest_downward_move": int(np.max(self.movement)) if self.movement.size else 0,
        }


def per_molecule_exposure(
    exposure: np.ndarray,
    molecule_index: np.ndarray,
    n_molecules: int,
) -> np.ndarray:
    """Mean predicted exposure per molecule."""
    values = np.asarray(exposure, dtype=np.float64)
    molecules = np.asarray(molecule_index, dtype=np.int64)
    means = np.full(n_molecules, np.nan, dtype=np.float64)
    for index in range(n_molecules):
        selected = molecules == index
        if selected.any():
            means[index] = float(values[selected].mean())
    return means


def compare_rankings(
    exposure_score: np.ndarray,
    mechanism_score: np.ndarray,
    molecule_index: np.ndarray | None = None,
) -> RankingComparison:
    """Rank the panel by exposure and by mechanism and report the movement."""
    exposure_values = np.asarray(exposure_score, dtype=np.float64)
    mechanism_values = np.asarray(mechanism_score, dtype=np.float64)
    if exposure_values.shape != mechanism_values.shape:
        raise ValueError("both rankings must cover the same panel")
    indices = (
        np.arange(exposure_values.size)
        if molecule_index is None
        else np.asarray(molecule_index, dtype=np.int64)
    )
    exposure_rank = rank_positions(exposure_values)
    mechanism_rank = rank_positions(mechanism_values)
    return RankingComparison(
        molecule_index=indices,
        exposure_score=exposure_values,
        mechanism_score=mechanism_values,
        exposure_rank=exposure_rank,
        mechanism_rank=mechanism_rank,
        movement=rank_movement(mechanism_values, exposure_values),
    )


def shortlist_effect(
    exposure_score: np.ndarray,
    mechanism_score: np.ndarray,
    shortlist_size: int = SHORTLIST_DEFAULT,
    adequacy_threshold: float = 1.0,
) -> dict[str, object]:
    """How an exposure-guided shortlist differs from a mechanism-only shortlist."""
    exposure_values = np.asarray(exposure_score, dtype=np.float64)
    mechanism_values = np.asarray(mechanism_score, dtype=np.float64)
    mechanism_order = np.argsort(-mechanism_values, kind="mergesort")
    exposure_order = np.argsort(-exposure_values, kind="mergesort")
    mechanism_top = set(mechanism_order[:shortlist_size].tolist())
    exposure_top = set(exposure_order[:shortlist_size].tolist())
    adequate = set(np.flatnonzero(exposure_values >= adequacy_threshold).tolist())
    shortlist = sorted(exposure_top & adequate) if adequate else sorted(exposure_top)
    return {
        "shortlist_size": shortlist_size,
        "excluded_from_mechanism_top": len(mechanism_top - set(shortlist)),
        "promoted_from_outside_mechanism_top": len(
            [item for item in shortlist if item not in mechanism_top]
        ),
        "n_adequate": len(adequate),
        "shortlist": [int(item) for item in shortlist],
    }


def movement_table(comparison: RankingComparison, top: int = 10) -> list[dict[str, object]]:
    """Per-molecule ranks and movement, ordered by the exposure ranking."""
    rows: list[dict[str, object]] = []
    order = np.argsort(comparison.exposure_rank, kind="mergesort")
    for position in order[:top]:
        rows.append(
            {
                "molecule": int(comparison.molecule_index[position]),
                "exposure_rank": int(comparison.exposure_rank[position]),
                "mechanism_rank": int(comparison.mechanism_rank[position]),
                "movement": int(comparison.movement[position]),
                "exposure_score": float(comparison.exposure_score[position]),
                "mechanism_score": float(comparison.mechanism_score[position]),
            }
        )
    return rows


def ranking_summary(
    comparison: RankingComparison,
    movement_threshold: int = 5,
    shortlist_size: int = SHORTLIST_DEFAULT,
) -> dict[str, object]:
    """Everything the ranking paragraph needs."""
    moved = moved_at_least(
        comparison.mechanism_score, comparison.exposure_score, movement_threshold
    )
    effect = shortlist_effect(comparison.exposure_score, comparison.mechanism_score, shortlist_size)
    return {
        "n_molecules": int(comparison.molecule_index.size),
        "spearman_rho": float(spearman_rho(comparison.exposure_score, comparison.mechanism_score)),
        "moved_at_least": moved,
        "movement_threshold": movement_threshold,
        **effect,
    }


def direction_of_movement(comparison: RankingComparison) -> dict[str, list[int]]:
    """Molecules that rose and molecules that fell, in rank order."""
    movement = np.asarray(comparison.movement, dtype=np.int64)
    indices = np.asarray(comparison.molecule_index, dtype=np.int64)
    rose = indices[np.argsort(-movement, kind="mergesort")][:5]
    fell = indices[np.argsort(movement, kind="mergesort")][:5]
    return {
        "largest_upward": [int(item) for item in rose],
        "largest_downward": [int(item) for item in fell],
    }

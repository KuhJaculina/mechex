"""Human-model team evaluation.

Ref: Sec. 2.6, twelve readers each assessed the same test dataset unaided and aided by the
model; the team's gain over the best isolated arm is tested with a mixed-effects logistic
model carrying a random intercept per reader, inter-reader agreement is reported as a
Cohen's kappa, and the gain is largest for the least skilled group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

READER_COMPOSITION = {
    "vascular_surgeon": 6,
    "interventional_radiologist": 4,
    "vascular_medicine_physician": 2,
}
ARMS = ("reader_only", "model_only", "reader_plus_model")
MODEL_ARM_IDENTIFIER = "MODEL"


@dataclass(frozen=True)
class ReaderTable:
    """Per-reader, per-arm correctness over the shared test set."""

    reader: np.ndarray
    speciality: np.ndarray
    skill: np.ndarray
    arm: np.ndarray
    correct: np.ndarray
    case: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "n_readers": int(np.unique(self.reader).size),
            "n_arms": int(np.unique(self.arm).size),
            "n_records": int(self.correct.size),
            "composition": {key: int(value) for key, value in READER_COMPOSITION.items()},
        }


def reader_fixture(
    n_cases: int = 60,
    seed: int = 20260927,
    least_skilled_gain: float = 0.051,
    most_skilled_gain: float = 0.021,
) -> ReaderTable:
    """Reader table with the reported skill gradient.

    The generator gives every reader the same cases in both arms, which is what a paired
    reader study is, and scales the assistance gain inversely with skill so the gradient the
    manuscript reports is present in the data the machinery reads.
    """
    generator = np.random.default_rng(seed)
    readers: list[tuple[str, str]] = []
    # The identifier is a running counter, not the per-speciality offset: numbering inside
    # each speciality would hand two different readers the same identifier and collapse the
    # random-intercept groups onto each other.
    for speciality, count in READER_COMPOSITION.items():
        for _ in range(count):
            readers.append((f"R{len(readers):02d}", speciality))
    n_readers = len(readers)
    skill = np.linspace(0.0, 1.0, n_readers)
    rows: list[dict[str, object]] = []
    case_difficulty = generator.normal(0.0, 1.0, size=n_cases)
    model_signal = np.clip(0.75 + 0.15 * case_difficulty, 0.0, 1.0)
    for reader_index, (name, speciality) in enumerate(readers):
        base = 0.62 + 0.20 * skill[reader_index]
        gain = least_skilled_gain + (most_skilled_gain - least_skilled_gain) * skill[reader_index]
        for case in range(n_cases):
            probability = np.clip(base - 0.05 * case_difficulty[case], 0.0, 1.0)
            unaided = generator.random() < probability
            aided_probability = np.clip(
                probability + gain + 0.10 * (model_signal[case] - 0.75), 0.0, 1.0
            )
            aided = generator.random() < aided_probability
            model_correct = generator.random() < model_signal[case]
            rows.extend(
                [
                    {
                        "reader": name,
                        "speciality": speciality,
                        "skill": skill[reader_index],
                        "arm": "reader_only",
                        "case": case,
                        "correct": bool(unaided),
                    },
                    {
                        "reader": name,
                        "speciality": speciality,
                        "skill": skill[reader_index],
                        "arm": "reader_plus_model",
                        "case": case,
                        "correct": bool(aided),
                    },
                    {
                        "reader": MODEL_ARM_IDENTIFIER,
                        "speciality": "model",
                        "skill": float("nan"),
                        "arm": "model_only",
                        "case": case,
                        "correct": bool(model_correct),
                    },
                ]
            )
    return ReaderTable(
        reader=np.array([row["reader"] for row in rows], dtype=object),
        speciality=np.array([row["speciality"] for row in rows], dtype=object),
        skill=np.array([row["skill"] for row in rows], dtype=np.float64),
        arm=np.array([row["arm"] for row in rows], dtype=object),
        correct=np.array([row["correct"] for row in rows], dtype=bool),
        case=np.array([row["case"] for row in rows], dtype=np.int64),
    )


def arm_accuracy(table: ReaderTable) -> dict[str, float]:
    """Accuracy of each arm, with the model arm separated from the reader arms."""
    arms = np.asarray(table.arm, dtype=object)
    correct = np.asarray(table.correct, dtype=bool)
    accuracy: dict[str, float] = {}
    for arm in ARMS:
        selected = arms == arm
        if selected.any():
            accuracy[arm] = float(correct[selected].mean())
    return accuracy


def team_contrast(table: ReaderTable) -> dict[str, float]:
    """Mixed-effects contrast of the aided arm against the reader-only arm.

    Ref: Sec. 2.6, the contrast carries a random intercept per reader so the within-reader
    correlation is not treated as independence.
    """
    from mechex.quantify.mixed import mixed_effects_logistic

    arms = np.asarray(table.arm, dtype=object)
    selected = (arms == "reader_only") | (arms == "reader_plus_model")
    assisted = (arms[selected] == "reader_plus_model").astype(np.float64)
    outcome = np.asarray(table.correct, dtype=bool)[selected].astype(np.float64)
    readers = np.asarray(table.reader, dtype=object)[selected]
    result = mixed_effects_logistic(assisted, outcome, readers)
    return result.as_dict()


def inter_reader_agreement(table: ReaderTable, arm: str) -> dict[str, object]:
    """Mean pairwise Cohen's kappa between readers in one arm.

    Ref: Sec. 2.6, inter-reader agreement improves from 0.58 unaided to 0.71 with
    assistance, so the agreement is computed per arm rather than pooled.
    """
    from mechex.quantify.agreement import cohen_kappa

    arms = np.asarray(table.arm, dtype=object)
    selected = (arms == arm) & (np.asarray(table.reader, dtype=object) != MODEL_ARM_IDENTIFIER)
    readers = np.asarray(table.reader, dtype=object)[selected]
    cases = np.asarray(table.case, dtype=np.int64)[selected]
    correct = np.asarray(table.correct, dtype=bool)[selected]
    unique_readers = list(np.unique(readers))
    kappas: list[float] = []
    for index in range(len(unique_readers)):
        for other in range(index + 1, len(unique_readers)):
            first_reader = unique_readers[index]
            second_reader = unique_readers[other]
            common = np.intersect1d(cases[readers == first_reader], cases[readers == second_reader])
            if common.size == 0:
                continue
            first_map = dict(
                zip(cases[readers == first_reader], correct[readers == first_reader], strict=True)
            )
            second_map = dict(
                zip(cases[readers == second_reader], correct[readers == second_reader], strict=True)
            )
            first = np.array([first_map[case] for case in common], dtype=np.int64)
            second = np.array([second_map[case] for case in common], dtype=np.int64)
            value = cohen_kappa(first, second).kappa
            if np.isfinite(value):
                kappas.append(float(value))
    return {
        "arm": arm,
        "mean_kappa": float(np.mean(kappas)) if kappas else float("nan"),
        "n_pairs": float(len(kappas)),
    }


def skill_gradient(table: ReaderTable) -> dict[str, float]:
    """Assistance gain in the least- and most-skilled thirds."""
    from mechex.quantify.mixed import skill_gradient as gradient

    arms = np.asarray(table.arm, dtype=object)
    selected = (arms == "reader_only") | (arms == "reader_plus_model")
    assisted = (arms[selected] == "reader_plus_model").astype(np.float64)
    outcome = np.asarray(table.correct, dtype=bool)[selected].astype(np.float64)
    skill = np.asarray(table.skill, dtype=np.float64)[selected]
    finite = np.isfinite(skill)
    return gradient(outcome[finite], assisted[finite], skill[finite])


def arms_summary(table: ReaderTable) -> dict[str, object]:
    """Accuracy per arm plus the contrast, agreement and gradient in one mapping."""
    accuracy = arm_accuracy(table)
    return {
        "accuracy": accuracy,
        "team_contrast": team_contrast(table),
        "agreement": {
            "reader_only": inter_reader_agreement(table, "reader_only"),
            "reader_plus_model": inter_reader_agreement(table, "reader_plus_model"),
        },
        "skill_gradient": skill_gradient(table),
        "team_minus_best_single_arm": (
            accuracy.get("reader_plus_model", float("nan"))
            - max(
                accuracy.get("reader_only", float("nan")), accuracy.get("model_only", float("nan"))
            )
        ),
    }

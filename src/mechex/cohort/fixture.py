"""Schema-compatible fixture of the primary cohort.

Ref: Sec. 2.1 and Sec. 3.2-3.3, the primary layer is the de-identified multi-site cohort.

The fixture reproduces the schema, the strata sizes and the causal structure the
manuscript describes, and carries no real record. Its internal truth is the generator's
own construction: the partition term depends on the molecular descriptors with a negative
loading on efflux-substrate propensity, the perfused fraction is set by a per-record
occlusion, and systemic availability is their product times an interaction factor, which
is exactly what the conservation relation asserts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.basis.chemistry import descriptor_matrix, molecule_fixture
from mechex.basis.graph import MoleculeGraph
from mechex.cohort.records import (
    ACCRUAL_ERAS,
    DRUG_CLASSES,
    SCANNER_VENDORS,
    SEVERITY_LEVELS,
    TIER_COMPOSITE,
    TIER_DIRECT_ASSAY,
    TIER_MICRODIALYSIS,
    CohortFixture,
)
from mechex.decision.threshold import concentration_time_integral
from mechex.settings import CohortConfig

MOLECULE_SEED_BASE = 910000
INTEGRAL_SCALE = 1000.0

# Class membership by molecule index; the split between the named classes and the
# remainder is what makes the drug-class strata in Table 3 non-exhaustive.
CLASS_BY_INDEX: tuple[str, ...] = (
    ("lipid_lowering",) * 7
    + ("antiplatelet",) * 8
    + ("prostaglandin_analogue",) * 5
    + ("other",) * 7
)

# Ground-truth loadings of the partition term on the standardised descriptors. The
# negative loading on efflux-substrate propensity is the generator's counterpart of the
# in vitro finding reported in Sec. 3.11.
TRUE_DESCRIPTOR_LOADINGS = np.array(
    [-0.95, 0.48, -0.22, 0.05, 0.02, -0.08, 0.10, 0.04], dtype=np.float64
)

SAMPLE_TIMES = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 24.0], dtype=np.float64)

SITE_SIZES = {"A": 4041, "B": 2694, "C": 1598}

# External-site marginals transcribed from Table 3 so the fixture's composition can be
# checked against the reported stratum sizes. These are the only cohort counts the
# manuscript states; every performance value in that table stays out of reach.
EXTERNAL_SEVERITY_COUNTS = {
    "claudication": 612,
    "rest_pain": 508,
    "chronic_limb_threatening_ischaemia": 478,
}
EXTERNAL_CLASS_COUNTS = {
    "lipid_lowering": 396,
    "antiplatelet": 352,
    "prostaglandin_analogue": 214,
    "other": 636,
}
EXTERNAL_FEMALE = 566
EXTERNAL_AGE_75 = 431
EXTERNAL_DIABETES = 624
EXTERNAL_SEVERE_CKD = 188


@dataclass(frozen=True)
class MoleculePanel:
    """The twenty-seven molecules with their descriptors and their true partition terms."""

    graphs: list[MoleculeGraph]
    descriptor_matrix: np.ndarray
    drug_class: tuple[str, ...]
    partition_truth: np.ndarray
    reference_exposure_scale: np.ndarray


def _standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = np.where(matrix.std(axis=0) > 1e-9, matrix.std(axis=0), 1.0)
    return (matrix - mean) / scale, mean, scale


def build_molecule_panel(n_molecules: int = 27, seed: int = 20260927) -> MoleculePanel:
    """Construct the molecule panel used by the fixture."""
    if n_molecules != len(CLASS_BY_INDEX):
        raise ValueError(f"the panel is defined for {len(CLASS_BY_INDEX)} molecules")
    graphs = [
        molecule_fixture(
            n_atoms=int(18 + 2 * (index % 9)),
            seed=MOLECULE_SEED_BASE + index,
            identifier=f"mol-{index:03d}",
        )
        for index in range(n_molecules)
    ]
    matrix = descriptor_matrix(graphs)
    standardised, _, _ = _standardise(matrix)
    log_partition = 0.35 + standardised @ TRUE_DESCRIPTOR_LOADINGS
    partition_truth = np.exp(log_partition)
    reference_scale = float(partition_truth.mean())
    return MoleculePanel(
        graphs=graphs,
        descriptor_matrix=matrix,
        drug_class=CLASS_BY_INDEX,
        partition_truth=partition_truth,
        reference_exposure_scale=np.full(n_molecules, reference_scale, dtype=np.float64),
    )


def _assign_marginals(
    counts: dict[str, int],
    labels: tuple[str, ...],
    total: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """Assign a categorical column with exact per-level counts and a shuffled order.

    Shuffling matters: a cumulative assignment would correlate the two marginals of the
    external site, which would fabricate a severity-by-drug-class pattern.
    """
    values = np.empty(total, dtype=object)
    cursor = 0
    for label in labels:
        count = int(counts.get(label, 0))
        values[cursor : cursor + count] = label
        cursor += count
    if cursor != total:
        raise ValueError("the level counts must sum to the stratum total")
    order = generator.permutation(total)
    return np.asarray(values, dtype=object)[order]


def _boolean_marginal(count: int, total: int, generator: np.random.Generator) -> np.ndarray:
    """Boolean column with an exact number of positives."""
    column = np.zeros(total, dtype=bool)
    if count > 0:
        column[generator.permutation(total)[:count]] = True
    return column


def _concentration_profile(
    target_integral: float,
    generator: np.random.Generator,
    times: np.ndarray,
    noise: float,
) -> np.ndarray:
    """Concentration-time record whose trapezoidal integral matches the target.

    A one-compartment profile is scaled so the recomputed integral reproduces the drawn
    availability, which keeps the record traceable back to the generative quantity.
    """
    elimination = float(generator.uniform(0.06, 0.18))
    absorption = float(generator.uniform(0.4, 1.2))
    shape = np.exp(-elimination * times) - np.exp(-absorption * times)
    shape = np.clip(shape, 1e-6, None)
    raw_integral = concentration_time_integral(times, shape)
    scaled = shape * (target_integral / max(raw_integral, 1e-9))
    perturbed = scaled * (1.0 + generator.normal(0.0, noise, size=times.shape))
    clipped: np.ndarray = np.clip(perturbed, 1e-6, None)
    return clipped


def _tier_for(generator: np.random.Generator) -> int:
    """Assign the reference tier in the reported proportions.

    Ref: Sec. 2.1, 418 records carry direct assays; the remainder are anchored by
    microdialysis or by the composite index.
    """
    draw = generator.random()
    if draw < 0.05:
        return TIER_DIRECT_ASSAY
    if draw < 0.16:
        return TIER_MICRODIALYSIS
    return TIER_COMPOSITE


TIER_NOISE = {TIER_DIRECT_ASSAY: 0.03, TIER_MICRODIALYSIS: 0.08, TIER_COMPOSITE: 0.13}


def build_cohort_fixture(config: CohortConfig | None = None) -> CohortFixture:
    """Generate the full three-site fixture cohort."""
    settings = config or CohortConfig()
    generator = np.random.default_rng(settings.fixture_seed)
    panel = build_molecule_panel(settings.n_molecules, settings.fixture_seed)
    total = sum(SITE_SIZES.values())
    site = np.concatenate(
        [np.full(SITE_SIZES[name], name, dtype=object) for name in ("A", "B", "C")]
    )
    record_id = np.arange(total)
    molecule_index = np.empty(total, dtype=np.int64)
    severity = np.empty(total, dtype=object)
    drug_class = np.empty(total, dtype=object)
    sex_female = np.zeros(total, dtype=bool)
    age_75 = np.zeros(total, dtype=bool)
    diabetes = np.zeros(total, dtype=bool)
    severe_ckd = np.zeros(total, dtype=bool)

    development_mask = site != "C"
    development_index = np.flatnonzero(development_mask)
    external_index = np.flatnonzero(~development_mask)

    molecule_index[development_index] = generator.integers(
        0, settings.n_molecules, size=development_index.size
    )
    severity[development_index] = np.asarray(
        generator.choice(
            np.asarray(SEVERITY_LEVELS, dtype=object),
            size=development_index.size,
            p=[0.40, 0.32, 0.28],
        ),
        dtype=object,
    )
    drug_class[development_index] = np.asarray(
        [panel.drug_class[index] for index in molecule_index[development_index]], dtype=object
    )

    external_severity = _assign_marginals(
        EXTERNAL_SEVERITY_COUNTS, SEVERITY_LEVELS, external_index.size, generator
    )
    external_class = _assign_marginals(
        EXTERNAL_CLASS_COUNTS, DRUG_CLASSES, external_index.size, generator
    )
    severity[external_index] = external_severity
    drug_class[external_index] = external_class
    class_to_indices: dict[str, list[int]] = {}
    for index, label in enumerate(panel.drug_class):
        class_to_indices.setdefault(label, []).append(index)
    for record, label in zip(external_index, external_class, strict=True):
        molecule_index[record] = int(generator.choice(class_to_indices[str(label)]))

    sex_female[external_index] = _boolean_marginal(EXTERNAL_FEMALE, external_index.size, generator)
    age_75[external_index] = _boolean_marginal(EXTERNAL_AGE_75, external_index.size, generator)
    diabetes[external_index] = _boolean_marginal(EXTERNAL_DIABETES, external_index.size, generator)
    severe_ckd[external_index] = _boolean_marginal(
        EXTERNAL_SEVERE_CKD, external_index.size, generator
    )
    sex_female[development_index] = generator.random(development_index.size) < 0.46
    age_75[development_index] = generator.random(development_index.size) < 0.34
    diabetes[development_index] = generator.random(development_index.size) < 0.44
    severe_ckd[development_index] = generator.random(development_index.size) < 0.12

    occlusion = np.clip(generator.beta(2.2, 3.4, size=total), 0.0, 0.92)
    delivered = np.clip(1.0 - occlusion, 0.05, 0.95)
    interaction = np.clip(0.18 + generator.normal(0.0, 0.05, size=total), 0.02, 0.6)
    partition_truth = panel.partition_truth[molecule_index]
    availability_truth = partition_truth * delivered * (1.0 + interaction)

    reference_tier = np.array([_tier_for(generator) for _ in range(total)], dtype=np.int64)
    tiers_available = np.unique(reference_tier)
    concentrations = np.zeros((total, SAMPLE_TIMES.size), dtype=np.float64)
    for index in range(total):
        noise = TIER_NOISE[int(reference_tier[index])]
        target_integral = availability_truth[index] * INTEGRAL_SCALE
        concentrations[index] = _concentration_profile(
            target_integral, generator, SAMPLE_TIMES, noise
        )

    times = np.tile(SAMPLE_TIMES, (total, 1))
    integrals = np.array(
        [concentration_time_integral(SAMPLE_TIMES, concentrations[index]) for index in range(total)]
    )
    availability_reference = integrals / INTEGRAL_SCALE
    exposure_raw = integrals * delivered

    patient_id = _assign_patients(total, site, generator)
    scanner_vendor = np.asarray(
        generator.choice(np.asarray(SCANNER_VENDORS, dtype=object), size=total, p=[0.5, 0.3, 0.2]),
        dtype=object,
    )
    accrual_era = np.asarray(
        generator.choice(np.asarray(ACCRUAL_ERAS, dtype=object), size=total, p=[0.55, 0.45]),
        dtype=object,
    )
    volume_seed = np.arange(total, dtype=np.int64) + settings.fixture_seed

    cohort = CohortFixture(
        record_id=record_id,
        patient_id=patient_id,
        site=site,
        molecule_index=molecule_index,
        descriptors=panel.descriptor_matrix[molecule_index],
        times=times,
        concentrations=concentrations,
        occlusion=occlusion,
        availability_reference=availability_reference,
        exposure_reference=exposure_raw,
        severity=severity,
        drug_class=drug_class,
        scanner_vendor=scanner_vendor,
        accrual_era=accrual_era,
        sex=np.where(sex_female, "female", "male").astype(object),
        age=np.where(age_75, 76, 64).astype(np.int64),
        diabetes=diabetes,
        ckd_stage=np.where(severe_ckd, 4, 2).astype(np.int64),
        reference_tier=reference_tier,
        volume_seed=volume_seed,
        metadata={
            "fixture": True,
            "n_molecules": settings.n_molecules,
            "tiers_present": [int(value) for value in tiers_available],
            "integral_scale": INTEGRAL_SCALE,
            "sample_times": SAMPLE_TIMES.tolist(),
        },
    )
    return cohort


def _assign_patients(
    total: int,
    site: np.ndarray,
    generator: np.random.Generator,
) -> np.ndarray:
    """Assign patient identifiers so that a minority of patients contribute two records.

    Patient-level resampling and the leakage checks both need repeated records to be
    possible; a purely one-to-one mapping would make those checks vacuous.
    """
    patient_id = np.empty(total, dtype=object)
    counter = 0
    for index in range(total):
        if (
            index > 0
            and generator.random() < 0.18
            and patient_id[index - 1] is not None
            and str(site[index]) == str(site[index - 1])
        ):
            patient_id[index] = patient_id[index - 1]
            continue
        patient_id[index] = f"P{counter:06d}"
        counter += 1
    return patient_id


def external_site_mask(cohort: CohortFixture, site: str = "C") -> np.ndarray:
    """Boolean mask of one site."""
    return np.asarray(cohort.site == site, dtype=bool)


def strata_counts(cohort: CohortFixture, column: str, site: str = "C") -> dict[str, int]:
    """Record counts per level of a categorical column at one site."""
    subset = cohort.subset(external_site_mask(cohort, site))
    return subset.group_counts(column)

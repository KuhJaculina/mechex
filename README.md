# MechEx

Local tissue exposure estimation for repurposing candidates in peripheral arterial disease.
A molecule's mechanism of action says nothing about whether it reaches the ischaemic limb,
and a plasma concentration says nothing about how much of it arrives at the arterial wall.
MechEx estimates the missing quantity per patient and per molecule from two inputs the
clinical workflow already produces: a lower-extremity CT angiogram in the arterial and
delayed phases, and the structure of an approved molecule.

The release implements the estimator described in *Mechanistic Drug Structure and CT
Angiography Models for Local Tissue Exposure Prediction when Repurposing Medicines for
Peripheral Arterial Disease*: a structure branch that emits a partitioning term, a delivery
branch that emits a perfused fraction, a bidirectional cross-attention coupling that emits
the interaction term, a head that enforces the exposure-conservation relation over the
three, a calibrated uncertainty head, and a decision layer that dichotomises the exposure
index against a frozen adequacy boundary and abstains when it is not confident. The
repository also carries the cohort-construction layer, the reference standard, the
statistics the manuscript reports, the safety and robustness machinery, the reconstructed
comparison baselines, and an audit that produces the three root artefacts.

## What the code is, and what it is not

Everything here is the authors' own implementation of the described method. It is not a
reimplementation of a third-party system, and no published number is reused for any
reported quantity: the manuscript's baseline note states that every comparator was trained
and evaluated in-house on the identical split with identical preprocessing and an identical
tuning budget, and the release follows that.

The primary cohort is de-identified clinical material held by the corresponding author and
is not distributed. The shipped pipeline therefore reads a **schema-compatible cohort
fixture** written by the release: the same table schema, the same three-site and two-region
structure, the same strata sizes, and the same causal structure the manuscript describes
(availability is the product of a partition term, a delivered fraction and an interaction
factor). Every cohort-level quantity in the manuscript stays out of reach of this checkout
and is reported as `NOT_RUN` rather than substituted. The fixture's composition is the only
place the release deliberately aligns itself with the manuscript, and it aligns on the
stratum *sizes* the manuscript prints, never on a performance value.

## Repository layout

    configs/basis           the frozen 3D molecular basis transformer and its adapted head
    configs/vascular        the frozen volumetric CT encoder, the crop grid and the window
    configs/coupling        the bidirectional cross-attention coupling
    configs/conservation    the mass-conservation head, its tolerance and its margin cost
    configs/uncertainty     the evidential head and the isotonic recalibration
    configs/decision        the adequacy boundary, the abstention quantile and the FPR budget
    configs/objective       the four terms of the composite training objective
    configs/optimiser       AdamW, the cosine schedule, the patience and the repeat count
    configs/protocol        split fractions, reference tiers, criteria and statistics
    configs/cohort          the primary-cohort shape and the fixture seed
    configs/ablation        the component switches of the reported ablation tiers
    configs/study           one overlay per reported comparison, ablation, stratum and shift
    src/mechex/basis        structure branch: graphs, descriptors, encoder, partition head
    src/mechex/vascular     delivery branch: preparation, encoder, perfusion, head
    src/mechex/coupling     the coupling module and its additive alternatives
    src/mechex/conservation the conservation law, its residual, its penalty and its rate
    src/mechex/uncertainty  evidential head, isotonic recalibration, deep ensemble
    src/mechex/decision     exposure index, adequacy boundary, abstention and harm tiering
    src/mechex/cohort       fixture generator, linkage, transforms, reference, splitting
    src/mechex/objective    concordance, adequacy, calibration and conservation terms
    src/mechex/optimise     optimiser, schedule, early stopping
    src/mechex/fit          training engine, checkpoints, repeats, distributed wrapping
    src/mechex/assess       external evaluation, strata, shifts, abstention, readers, ranking
    src/mechex/quantify     the statistics: CCC, AUROC, DeLong, bootstrap, ECE, mixed effects
    src/mechex/guard        failure taxonomy, operating points, perturbation, drift
    src/mechex/bench        the twelve reconstructed comparators
    src/mechex/compress     post-training int8 quantisation with per-channel scales
    src/mechex/assay        the in vitro efflux arm and its accumulation ratios
    src/mechex/provenance   the claim registry, the manuscript arithmetic and the artefacts
    src/mechex/console      train, evaluate and audit entry points
    scripts                 cohort staging, training, evaluation, audit and gate launches
    tests                   the unit suite the audit gate runs

## Installation

pip:

    python -m venv .venv && . .venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

conda:

    conda env create -f environment.yml
    conda activate mechex

Docker (CPU):

    docker build -t mechex:1.0.0 .
    docker run --rm mechex:1.0.0 mechex.console.audit --skip-gates

The image installs the CPU build of PyTorch by default. For a CUDA build, replace the wheel
index in the Dockerfile with the CUDA channel matching the host driver.

## Data

The primary cohort of 8,333 records over 27 molecules is not distributed; see
`dataset_urls.txt` for its access conditions and for the auxiliary public resources the
manuscript names. Stage the fixture and the derived tables with:

    bash scripts/stage_cohort.sh

This writes `data/cohort_fixture.npz` (the tabular fixture) and
`data/cohort_fixture_summary.json` (its composition, the split sizes and the leakage report)
under `data/`, which is ignored by git. CTA volumes are not stored: each record carries a
generation seed and the dataset constructs its arterial and delayed pair on demand at the
configured grid, so the fixture stays small enough to ship while the volumetric encoder still
receives a real two-channel tensor.

Volumes are resampled to isotropic 1.0 mm spacing (reported), windowed to the arterial
contrast range and intensity-normalised per scan (reported); the 100 to 700 Hounsfield
bounds, the 64x64x48 grid and the fixture seed are engineering defaults and are exposed in
`configs/vascular/` and `configs/cohort/`.

## Training

One command per reported configuration; the study file supplies the overlay.

    python -m mechex.console.train --overlay study/main.yaml

Reported training settings, taken from the manuscript:

| setting | value | source |
| --- | --- | --- |
| optimiser | AdamW | Sec. 3.8 |
| schedule | cosine decay | Sec. 3.8 |
| batch size | 64 patients | Sec. 3.8 |
| early stopping | validation partition, patience 12 | Sec. 3.8 |
| repeats | 5 independent initialisations | Sec. 3.8 |
| spacing | isotropic 1.0 mm | Sec. 3.8 |
| encoders | frozen, adapted projection heads | Sec. 3.6 |
| quantisation | int8 post-training, per-channel scales | Sec. 3.8 |

The learning rate, weight decay, epoch horizon, warmup fraction, schedule floor, gradient
clipping norm, precision, process count, encoder widths and volume grid are not reported by
the manuscript. They are engineering defaults, they live in `configs/`, and each one is
listed in the `deviations` array of `claim_to_code.json`.

Ablations and substitutions, each an overlay:

    python -m mechex.console.train --overlay study/no_molecular_branch.yaml
    python -m mechex.console.train --overlay study/no_imaging_branch.yaml
    python -m mechex.console.train --overlay study/no_conservation_constraint.yaml
    python -m mechex.console.train --overlay study/constraint_weight_quarter.yaml
    python -m mechex.console.train --overlay study/constraint_weight_four.yaml
    python -m mechex.console.train --overlay study/concatenation_instead_of_cross_attention.yaml
    python -m mechex.console.train --overlay study/single_branch_then_stack.yaml
    python -m mechex.console.train --overlay study/randomly_initialised_encoders.yaml
    python -m mechex.console.train --overlay study/full_finetuning.yaml

Distributed launch uses the same entry point; the model is wrapped only when a process group
is active:

    torchrun --nproc_per_node=4 scripts/launch_distributed.sh main

## Evaluation

Evaluate a checkpoint on the external site, the stratified axes, the four shift conditions
and the abstention policy:

    bash scripts/launch_eval.sh runs/seed_20260927/mechex.pt

The pre-specified criteria at the fully held-out external site are a concordance correlation
coefficient of at least 0.75 and an AUROC of at least 0.85 for the adequate-to-inadequate
classification. The abstention analysis reports coverage and the retained AUROC after the
highest-deviation tenth of cases is referred for direct measurement.

**Every value in the manuscript's reported tables is `NOT_RUN` in this release.** The inputs
those values need are the de-identified cohort, the twelve reader assessments, the archived
assay material and the silent-mode deployment logs, none of which is distributed. What the
release does provide is the executable machinery for each of those analyses, the fixture on
which every mechanism runs end to end, and an audit that reports what actually ran.

## Compute budget

The manuscript does not report its compute target, so this section states what the release
was verified on and what the configuration implies rather than a reported figure.

- Verified on: a single CPU host, Python 3.14, PyTorch 2.11, no accelerator present.
- The audit's closed-loop check trains two epochs on 64 development records at the smoke
  grid and finishes in seconds; the full unit suite runs in well under a minute on the same
  host.
- At the reported settings, one repeat is 60 epochs over about 4,200 development records at
  batch 64, which is roughly 3,900 optimiser steps; the CTA branch dominates the cost and
  the frozen encoders mean only the projection heads, the coupling, the conservation head,
  the uncertainty head and the decision layer carry optimiser state.
- Five repeats are run per configuration, and the ablation table has thirteen variants, so a
  full reported sweep is 90 training runs. The release does not claim a wall-clock figure for
  that sweep, because the manuscript does not state the hardware it ran on.
- Disk: the fixture is a few megabytes; checkpoints and run logs land under `runs/` and are
  ignored by git.

## Reproducing the audit

    bash scripts/run_audit.sh

Run it **twice**. The audit gates on a manifest-freshness check that compares the shipped
`integrity_manifest.json` against the live tree; on the first run after any edit that check
fails because the manifest it reads predates the edit, and the run rewrites the manifest at
the end. On the second run the tree and the manifest agree, and the three artefacts are a
byte-identical fixed point from then on. The audit writes, in order: `claim_to_code.json`,
`verification_report.json`, `verification_summary.txt` and finally `integrity_manifest.json`,
so the manifest digests the finished tree and excludes only itself.

## Licence

MIT; see `LICENSE`. Third-party dependencies and the external resources the manuscript names
are documented in `NOTICE` and `dataset_urls.txt`.

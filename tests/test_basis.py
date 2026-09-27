"""Structure branch: graphs, descriptors, the frozen encoder and the partition head."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mechex.basis.chemistry import (
    DESCRIPTOR_NAMES,
    descriptor_matrix,
    descriptors,
    efflux_substrate_propensity,
    molecule_fixture,
)
from mechex.basis.graph import BOND_FEATURE_DIM, MoleculeGraph
from mechex.basis.projection import MolecularBranch, PartitionHead
from mechex.basis.transformer import (
    MolecularBasisEncoder,
    MolecularBatch,
    collate_graphs,
)
from mechex.settings import MolecularBasisConfig


def test_configured_feature_widths_match_the_graph_schema() -> None:
    """The configured widths must equal the graph schema's own vocabulary lengths."""
    from mechex.basis.graph import ATOM_FEATURE_DIM, BOND_FEATURE_DIM

    config = MolecularBasisConfig()
    assert config.atom_feature_dim == ATOM_FEATURE_DIM
    assert config.bond_feature_dim == BOND_FEATURE_DIM


def test_molecule_fixture_is_deterministic() -> None:
    first = molecule_fixture(14, seed=7, identifier="probe")
    second = molecule_fixture(14, seed=7, identifier="probe")
    assert first.n_atoms == 14
    assert np.array_equal(first.atoms, second.atoms)
    assert np.array_equal(first.coords, second.coords)
    assert np.array_equal(first.bonds, second.bonds)


def test_atom_features_are_one_hot_and_bounded() -> None:
    graph = molecule_fixture(12, seed=3)
    assert graph.atoms.shape == (12, len(_atom_names()))
    assert np.all(graph.atoms >= 0.0)
    assert np.all(graph.atoms <= 1.0 + 1e-9)
    assert graph.bond_features.shape[1] == BOND_FEATURE_DIM


def _atom_names() -> tuple[str, ...]:
    from mechex.basis.graph import ATOM_FEATURE_NAMES

    return ATOM_FEATURE_NAMES


def test_collation_pads_masks_and_symmetric_bonds() -> None:
    batch = collate_graphs([molecule_fixture(5, 1), molecule_fixture(9, 2)])
    assert batch.atoms.shape[0] == 2
    assert batch.atoms.shape[1] == 9
    assert int(batch.mask[0].sum()) == 5
    assert int(batch.mask[1].sum()) == 9
    assert torch.equal(batch.bond_index, batch.bond_index.transpose(1, 2))


def test_descriptor_vector_names_and_order() -> None:
    graph = molecule_fixture(16, seed=11)
    vector = descriptors(graph)
    assert vector.names == DESCRIPTOR_NAMES
    assert vector.values.shape == (len(DESCRIPTOR_NAMES),)
    assert set(vector.as_dict()) == set(DESCRIPTOR_NAMES)


def test_efflux_propensity_increases_with_polarity() -> None:
    polar = molecule_fixture(24, seed=101)
    apolar = molecule_fixture(24, seed=101)
    apolar.atoms[:24, :9] = apolar.atoms[:24, :9] * 0.0
    apolar.atoms[:24, 0] = 1.0
    assert efflux_substrate_propensity(polar) != pytest.approx(0.0)
    assert 0.0 <= efflux_substrate_propensity(polar) <= 1.0
    assert 0.0 <= efflux_substrate_propensity(apolar) <= 1.0


def test_descriptor_matrix_shape() -> None:
    graphs = [molecule_fixture(10 + index, 500 + index) for index in range(5)]
    matrix = descriptor_matrix(graphs)
    assert matrix.shape == (5, len(DESCRIPTOR_NAMES))


def test_encoder_is_frozen_and_emits_latent_width() -> None:
    config = MolecularBasisConfig(latent_dim=24, layers=2, heads=2)
    encoder = MolecularBasisEncoder(config)
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    batch = collate_graphs([molecule_fixture(6, 4), molecule_fixture(11, 5)])
    embedding = encoder(batch)
    assert embedding.shape == (2, 24)
    assert torch.isfinite(embedding).all()


def test_padded_batch_stays_finite() -> None:
    """Padding must not leak NaN into the pooled embedding."""
    encoder = MolecularBasisEncoder(MolecularBasisConfig(latent_dim=16, layers=2, heads=2))
    batch = collate_graphs([molecule_fixture(3, 61), molecule_fixture(21, 62)])
    embedding = encoder(batch)
    assert torch.isfinite(embedding).all()


def test_encoder_train_flag_does_not_unfreeze() -> None:
    encoder = MolecularBasisEncoder(MolecularBasisConfig(latent_dim=16, layers=1, heads=2))
    encoder.train(True)
    assert not encoder.training


def test_partition_head_is_positive_and_attribution_is_ordered() -> None:
    head = PartitionHead(latent_dim=16)
    head.descriptor_weights.copy_(torch.linspace(-1.0, 1.0, len(DESCRIPTOR_NAMES)))
    embedding = torch.randn(4, 16)
    values = torch.rand(4, len(DESCRIPTOR_NAMES))
    partition = head(embedding, values)
    assert bool((partition > 0).all())
    ranked = head.ranked_attribution()
    assert ranked[0][0] == DESCRIPTOR_NAMES[0]
    assert ranked[-1][1] == pytest.approx(1.0)


def test_molecular_branch_forward_from_graphs(config) -> None:
    branch = MolecularBranch(config.basis)
    graphs = [molecule_fixture(9, 21), molecule_fixture(13, 22)]
    partition, embedding = branch.forward_graphs(graphs)
    assert partition.shape == (2,)
    assert embedding.shape == (2, config.basis.latent_dim)


def test_descriptor_scaling_updates_buffers(config) -> None:
    branch = MolecularBranch(config.basis)
    matrix = descriptor_matrix([molecule_fixture(10 + index, 900 + index) for index in range(6)])
    branch.calibrate_descriptor_scaling(matrix)
    assert torch.allclose(
        branch.head.descriptor_bias,
        torch.as_tensor(matrix.mean(axis=0), dtype=torch.float32),
        atol=1e-5,
    )


def test_molecule_graph_exposes_counts() -> None:
    graph = molecule_fixture(8, 31)
    assert isinstance(graph, MoleculeGraph)
    assert graph.n_atoms == 8
    assert graph.n_bonds == graph.bonds.shape[1]


def test_molecular_batch_moves_to_device() -> None:
    batch = collate_graphs([molecule_fixture(6, 41)])
    moved = batch.to(torch.device("cpu"))
    assert isinstance(moved, MolecularBatch)
    assert moved.batch_size == 1

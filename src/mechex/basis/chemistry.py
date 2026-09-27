"""Molecular and physicochemical descriptors read out by the retention attribution.

Ref: Sec. 4, third conclusion (efflux-substrate propensity, lipophilicity, molecular size).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mechex.basis.graph import ATOM_TYPES, MoleculeGraph, atom_feature_vector

DESCRIPTOR_NAMES = (
    "efflux_substrate_propensity",
    "lipophilicity",
    "molecular_size",
    "hydrogen_bond_donors",
    "hydrogen_bond_acceptors",
    "polar_surface_area",
    "aromatic_ring_count",
    "rotatable_bonds",
)

# Fragment contributions used to approximate a lipophilicity scale. The manuscript does
# not give a descriptor recipe, so these are the release's own calibration constants and
# live here rather than in a config because they belong to the descriptor definition.
_FRAGMENT_LOGP = {
    "C": 1.46,
    "N": -1.23,
    "O": -1.50,
    "S": 0.23,
    "F": 0.14,
    "Cl": 0.60,
    "Br": 0.88,
    "P": 0.40,
}
_HETERO_TPSA = {"N": 12.0, "O": 17.0, "S": 25.0, "P": 23.0}
_MASS = {
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "F": 18.998,
    "Cl": 35.45,
    "Br": 79.904,
    "P": 30.974,
}


@dataclass(frozen=True)
class DescriptorVector:
    """Named descriptors for one molecule."""

    values: np.ndarray
    names: tuple[str, ...] = DESCRIPTOR_NAMES

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(self.names, self.values, strict=True)}


def _elements(graph: MoleculeGraph) -> list[str]:
    offset = len(ATOM_TYPES)
    elements: list[str] = []
    for row in graph.atoms:
        index = int(np.argmax(row[:offset]))
        elements.append(ATOM_TYPES[index] if index < len(ATOM_TYPES) - 1 else "other")
    return elements


def _ring_atoms(graph: MoleculeGraph) -> np.ndarray:
    offset = len(ATOM_TYPES) + 4
    return graph.atoms[:, offset + 1] > 0.5


def _donors(graph: MoleculeGraph) -> np.ndarray:
    offset = len(ATOM_TYPES) + 4
    return graph.atoms[:, offset + 2] > 0.5


def _acceptors(graph: MoleculeGraph) -> np.ndarray:
    offset = len(ATOM_TYPES) + 4
    return graph.atoms[:, offset + 3] > 0.5


def efflux_substrate_propensity(
    graph: MoleculeGraph,
    efflux_affinity: float | None = None,
) -> float:
    """Composite propensity of a molecule to be actively transported out of a cell.

    Large, polar, multiply hydrogen-bonding molecules are the canonical efflux substrates;
    the composite weights those three axes and adds the caller-supplied transporter
    affinity when one is available. The manuscript reports the direction and the ranking
    of this quantity, not its formula.
    """
    elements = _elements(graph)
    n_atoms = max(graph.n_atoms, 1)
    polar = sum(1 for element in elements if element in _HETERO_TPSA)
    donors = int(_donors(graph).sum())
    acceptors = int(_acceptors(graph).sum())
    aromatic = int(_ring_atoms(graph).sum())
    polarity = polar / n_atoms
    size_term = np.log1p(n_atoms) / np.log1p(60.0)
    hbond_term = (donors + acceptors) / (2.0 * n_atoms)
    aromatic_term = aromatic / n_atoms
    base = 0.55 * polarity + 0.30 * size_term + 0.35 * hbond_term - 0.20 * aromatic_term
    if efflux_affinity is not None:
        base += 0.25 * float(efflux_affinity)
    return float(np.clip(base, 0.0, 1.0))


def lipophilicity(graph: MoleculeGraph) -> float:
    """Fragment-contribution lipophilicity estimate."""
    elements = _elements(graph)
    total = sum(_FRAGMENT_LOGP.get(element, 0.0) for element in elements)
    return float(total)


def molecular_size(graph: MoleculeGraph) -> float:
    """Heavy-atom count, the size axis the attribution reports."""
    return float(graph.n_atoms)


def polar_surface_area(graph: MoleculeGraph) -> float:
    """Polar surface area approximated from heteroatom contributions."""
    elements = _elements(graph)
    return float(sum(_HETERO_TPSA.get(element, 0.0) for element in elements))


def aromatic_ring_count(graph: MoleculeGraph) -> float:
    """Number of aromatic rings approximated by aromatic atoms over ring size six."""
    aromatic = int((graph.atoms[:, len(ATOM_TYPES) + 3] > 0.5).sum())
    return float(aromatic / 6.0)


def rotatable_bonds(graph: MoleculeGraph) -> float:
    """Non-ring single bonds, the conformational-flexibility axis."""
    in_ring = graph.bond_features[:, len(("single", "double", "triple", "aromatic")) + 1] > 0.5
    single = graph.bond_features[:, 0] > 0.5
    return float(int(np.sum(single & ~in_ring)))


def descriptors(
    graph: MoleculeGraph,
    efflux_affinity: float | None = None,
) -> DescriptorVector:
    """Full descriptor vector in :data:`DESCRIPTOR_NAMES` order."""
    values = np.array(
        [
            efflux_substrate_propensity(graph, efflux_affinity),
            lipophilicity(graph),
            molecular_size(graph),
            float(int(_donors(graph).sum())),
            float(int(_acceptors(graph).sum())),
            polar_surface_area(graph),
            aromatic_ring_count(graph),
            rotatable_bonds(graph),
        ],
        dtype=np.float64,
    )
    return DescriptorVector(values=values)


def descriptor_matrix(
    graphs: list[MoleculeGraph], efflux_affinities: list[float] | None = None
) -> np.ndarray:
    """Stack descriptors for a molecule set into a ``[n_molecules, n_descriptors]`` array."""
    rows: list[np.ndarray] = []
    for index, graph in enumerate(graphs):
        affinity = None
        if efflux_affinities is not None:
            affinity = efflux_affinities[index]
        rows.append(descriptors(graph, affinity).values)
    if not rows:
        return np.zeros((0, len(DESCRIPTOR_NAMES)), dtype=np.float64)
    return np.vstack(rows)


def molecule_fixture(
    n_atoms: int,
    seed: int,
    identifier: str = "",
    efflux_affinity: float | None = None,
) -> MoleculeGraph:
    """Build a deterministic schema-compatible atom and bond graph.

    Atom typing is drawn from a chemistry-flavoured prior so the descriptor reads differ
    across molecules; no source structure is represented.
    """
    generator = np.random.default_rng(seed)
    atom_rows: list[np.ndarray] = []
    elements: list[str] = []
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    position = np.zeros(3, dtype=np.float64)
    for index in range(n_atoms):
        element = str(
            generator.choice(
                list(_MASS.keys()), p=[0.62, 0.15, 0.13, 0.04, 0.03, 0.02, 0.005, 0.005]
            )
        )
        hybrid = str(generator.choice(["sp", "sp2", "sp3", "aromatic"], p=[0.05, 0.35, 0.45, 0.15]))
        ring = bool(generator.random() < 0.35)
        donor = bool(element in {"N", "O", "S"} and generator.random() < 0.4)
        acceptor = bool(element in {"N", "O", "S"} and generator.random() < 0.6)
        charge = 0.0
        atom_rows.append(
            atom_feature_vector(
                element,
                hybrid,
                ring,
                donor,
                acceptor,
                charge,
                _MASS[element],
            )
        )
        elements.append(element)
        step = generator.normal(0.0, 1.45, size=3)
        position = position + step
        coords[index] = position

    atoms = np.vstack(atom_rows)
    bond_pairs: list[tuple[int, int]] = []
    for index in range(1, n_atoms):
        parent = int(generator.integers(0, index))
        bond_pairs.append((parent, index))
    extra = int(generator.integers(0, max(1, n_atoms // 3)))
    seen = set(bond_pairs)
    for _ in range(extra):
        if n_atoms < 3:
            break
        a, b = sorted(generator.integers(0, n_atoms, size=2).tolist())
        if a != b and (a, b) not in seen:
            seen.add((a, b))
            bond_pairs.append((a, b))
    bond_array = (
        np.array(bond_pairs, dtype=np.int64).T if bond_pairs else np.zeros((2, 0), dtype=np.int64)
    )
    bond_rows: list[np.ndarray] = []
    for _ in range(len(bond_pairs)):
        bond_type = str(
            generator.choice(["single", "double", "triple", "aromatic"], p=[0.7, 0.15, 0.03, 0.12])
        )
        from mechex.basis.graph import bond_feature_vector

        bond_rows.append(
            bond_feature_vector(
                bond_type, bool(generator.random() < 0.3), bool(generator.random() < 0.35)
            )
        )
    bond_features = (
        np.vstack(bond_rows)
        if bond_rows
        else np.zeros((0, len(("single", "double", "triple", "aromatic")) + 2))
    )
    return MoleculeGraph(
        atoms=atoms,
        coords=coords,
        bonds=bond_array,
        bond_features=bond_features,
        identifier=identifier,
    )

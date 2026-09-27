"""Atom and bond graph representation of a repurposing candidate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ATOM_TYPES = ("C", "N", "O", "S", "F", "Cl", "Br", "P", "other")
HYBRIDISATIONS = ("sp", "sp2", "sp3", "aromatic")
BOND_TYPES = ("single", "double", "triple", "aromatic")

ATOM_FEATURE_NAMES = (
    *(f"atom_{name}" for name in ATOM_TYPES),
    *(f"hybrid_{name}" for name in HYBRIDISATIONS),
    "heavy",
    "ring",
    "donor",
    "acceptor",
    "charge",
    "mass_scaled",
)
BOND_FEATURE_NAMES = (
    *(f"bond_{name}" for name in BOND_TYPES),
    "conjugated",
    "in_ring",
)

ATOM_FEATURE_DIM = len(ATOM_FEATURE_NAMES)
BOND_FEATURE_DIM = len(BOND_FEATURE_NAMES)


@dataclass(frozen=True)
class MoleculeGraph:
    """A single molecule as atom and bond arrays.

    Coordinates are a 3D conformer stand-in; they carry geometry for the distance
    bias of the molecular transformer, never a redistributed structure.
    """

    atoms: np.ndarray
    coords: np.ndarray
    bonds: np.ndarray
    bond_features: np.ndarray
    identifier: str = ""

    @property
    def n_atoms(self) -> int:
        return int(self.atoms.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.bonds.shape[1])


def atom_feature_vector(
    element: str,
    hybridisation: str,
    ring: bool,
    donor: bool,
    acceptor: bool,
    charge: float,
    atomic_mass: float,
) -> np.ndarray:
    """Assemble the fixed-width atom descriptor used by the encoder."""
    vector = np.zeros(ATOM_FEATURE_DIM, dtype=np.float64)
    index = ATOM_TYPES.index(element) if element in ATOM_TYPES else len(ATOM_TYPES) - 1
    vector[index] = 1.0
    offset = len(ATOM_TYPES)
    hybrid_index = HYBRIDISATIONS.index(hybridisation) if hybridisation in HYBRIDISATIONS else 2
    vector[offset + hybrid_index] = 1.0
    tail = offset + len(HYBRIDISATIONS)
    vector[tail + 0] = 1.0
    vector[tail + 1] = 1.0 if ring else 0.0
    vector[tail + 2] = 1.0 if donor else 0.0
    vector[tail + 3] = 1.0 if acceptor else 0.0
    vector[tail + 4] = charge
    vector[tail + 5] = atomic_mass / 100.0
    return vector


def bond_feature_vector(bond_type: str, conjugated: bool, in_ring: bool) -> np.ndarray:
    """Assemble the fixed-width bond descriptor."""
    vector = np.zeros(BOND_FEATURE_DIM, dtype=np.float64)
    index = BOND_TYPES.index(bond_type) if bond_type in BOND_TYPES else 0
    vector[index] = 1.0
    vector[len(BOND_TYPES)] = 1.0 if conjugated else 0.0
    vector[len(BOND_TYPES) + 1] = 1.0 if in_ring else 0.0
    return vector


def to_torch(graph: MoleculeGraph, device: str = "cpu") -> dict[str, object]:
    """Convert a graph to torch tensors keyed for the encoder's forward signature."""
    import torch

    return {
        "atoms": torch.as_tensor(graph.atoms, dtype=torch.float32, device=device),
        "coords": torch.as_tensor(graph.coords, dtype=torch.float32, device=device),
        "bonds": torch.as_tensor(graph.bonds, dtype=torch.long, device=device),
        "bond_features": torch.as_tensor(graph.bond_features, dtype=torch.float32, device=device),
    }

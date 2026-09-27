"""Frozen 3D molecular basis transformer.

Ref: Sec. 3.6, molecular derivation; the encoder is held frozen and only its projection
head is adapted.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from mechex.basis.graph import BOND_FEATURE_DIM, MoleculeGraph, to_torch
from mechex.settings import MolecularBasisConfig

NO_BOND = 0
MASKED_SCORE = -1e9


class MolecularBatch:
    """Padded tensors for a set of molecules."""

    def __init__(
        self,
        atoms: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
        bond_index: torch.Tensor,
    ) -> None:
        self.atoms = atoms
        self.coords = coords
        self.mask = mask
        self.bond_index = bond_index

    def to(self, device: torch.device) -> MolecularBatch:
        return MolecularBatch(
            atoms=self.atoms.to(device),
            coords=self.coords.to(device),
            mask=self.mask.to(device),
            bond_index=self.bond_index.to(device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.atoms.shape[0])


def collate_graphs(graphs: list[MoleculeGraph]) -> MolecularBatch:
    """Pad a list of graphs into one dense batch.

    Bond type lands in a dense ``[batch, atoms, atoms]`` index map with zero reserved for
    "no bond", which lets the attention bias read an edge without a scatter.
    """
    if not graphs:
        raise ValueError("collate_graphs received no graphs")
    max_atoms = max(graph.n_atoms for graph in graphs)
    batch = len(graphs)
    feature_dim = int(graphs[0].atoms.shape[1])
    atoms = torch.zeros(batch, max_atoms, feature_dim, dtype=torch.float32)
    coords = torch.zeros(batch, max_atoms, 3, dtype=torch.float32)
    mask = torch.zeros(batch, max_atoms, dtype=torch.bool)
    bond_index = torch.zeros(batch, max_atoms, max_atoms, dtype=torch.long)
    for row, graph in enumerate(graphs):
        entry = to_torch(graph)
        atom_tensor = entry["atoms"]
        coord_tensor = entry["coords"]
        bond_tensor = entry["bonds"]
        bond_features = entry["bond_features"]
        assert isinstance(atom_tensor, torch.Tensor)
        assert isinstance(coord_tensor, torch.Tensor)
        assert isinstance(bond_tensor, torch.Tensor)
        assert isinstance(bond_features, torch.Tensor)
        count = graph.n_atoms
        atoms[row, :count] = atom_tensor
        coords[row, :count] = coord_tensor
        mask[row, :count] = True
        if bond_tensor.numel():
            types = torch.argmax(bond_features, dim=1) + 1 if bond_features.numel() else None
            for edge in range(int(bond_tensor.shape[1])):
                src = int(bond_tensor[0, edge])
                dst = int(bond_tensor[1, edge])
                kind = int(types[edge]) if types is not None else 1
                bond_index[row, src, dst] = kind
                bond_index[row, dst, src] = kind
    return MolecularBatch(atoms=atoms, coords=coords, mask=mask, bond_index=bond_index)


class RadialDistanceBias(nn.Module):
    """Head-wise bias from a radial basis expansion of pairwise interatomic distance."""

    centers: torch.Tensor

    def __init__(self, num_heads: int, num_rbf: int = 16, max_distance: float = 12.0) -> None:
        super().__init__()
        self.register_buffer("centers", torch.linspace(0.0, max_distance, num_rbf), persistent=True)
        self.width = float(max_distance / max(num_rbf - 1, 1)) * 2.0
        self.projection = nn.Linear(num_rbf, num_heads)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        delta = coords.unsqueeze(2) - coords.unsqueeze(1)
        distance = torch.sqrt(delta.pow(2).sum(dim=-1) + 1e-8)
        rbf = torch.exp(-((distance.unsqueeze(-1) - self.centers) / self.width).pow(2))
        bias: torch.Tensor = self.projection(rbf)
        return bias.permute(0, 3, 1, 2).contiguous()


class BondTypeBias(nn.Module):
    """Per-head additive bias for a bonded atom pair."""

    def __init__(self, num_heads: int, num_bond_types: int = 5) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_bond_types, num_heads)

    def forward(self, bond_index: torch.Tensor) -> torch.Tensor:
        bias: torch.Tensor = self.embedding(bond_index)
        return bias.permute(0, 3, 1, 2).contiguous()


class MolecularAttention(nn.Module):
    """Multi-head self-attention over atoms with distance and bond bias."""

    def __init__(self, width: int, heads: int, dropout: float, num_rbf: int = 16) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.distance_bias = RadialDistanceBias(heads, num_rbf=num_rbf)
        self.bond_bias = BondTypeBias(heads)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(width, width)

    def forward(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        bond_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.query(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = self.key(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(hidden).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        distance_bias: torch.Tensor = self.distance_bias(coords)
        bond_bias: torch.Tensor = self.bond_bias(bond_index)
        scores = scores + distance_bias + bond_bias
        pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        # A finite sentinel rather than -inf: a padded query row has every key masked, and
        # softmax over an all -inf row is undefined, which would put NaN into the pooled
        # embedding because the padding mask is applied by multiplication.
        scores = scores.masked_fill(~pair_mask.unsqueeze(1), MASKED_SCORE)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        context = context.transpose(1, 2).reshape(batch, length, self.width)
        context = context * mask.unsqueeze(-1).to(context.dtype)
        projected: torch.Tensor = self.output(context)
        return projected


class TransformerBlock(nn.Module):
    """Pre-norm transformer block over atom tokens."""

    def __init__(self, width: int, heads: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(width)
        self.attention = MolecularAttention(width, heads, dropout)
        self.norm_ffn = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * width, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        coords: torch.Tensor,
        bond_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        normalised = self.norm_attention(hidden)
        attended: torch.Tensor = self.attention(normalised, coords, bond_index, mask)
        hidden = hidden + self.dropout(attended)
        fed: torch.Tensor = self.ffn(self.norm_ffn(hidden))
        fed = self.dropout(fed)
        return hidden + fed


class MolecularBasisEncoder(nn.Module):
    """Frozen encoder mapping a molecule to a partition-relevant embedding."""

    def __init__(self, config: MolecularBasisConfig) -> None:
        super().__init__()
        self.config = config
        self.atom_projection = nn.Linear(config.atom_feature_dim, config.latent_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(config.latent_dim, config.heads, config.dropout)
            for _ in range(config.layers)
        )
        self.norm = nn.LayerNorm(config.latent_dim)
        if config.frozen:
            self.freeze()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> MolecularBasisEncoder:
        """Keep a frozen encoder in eval mode regardless of the outer training flag."""
        super().train(mode and not self.config.frozen)
        return self

    def forward(self, batch: MolecularBatch) -> torch.Tensor:
        hidden: torch.Tensor = self.atom_projection(batch.atoms)
        mask = batch.mask
        for block in self.blocks:
            hidden = block(hidden, batch.coords, batch.bond_index, mask)
        hidden = self.norm(hidden)
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled

    def embed_graphs(self, graphs: list[MoleculeGraph]) -> torch.Tensor:
        """Embed a python-side list of graphs at inference."""
        return self.forward(collate_graphs(graphs))


def bond_type_vocabulary() -> int:
    """Number of reservable bond type indices, including the no-bond sentinel."""
    return BOND_FEATURE_DIM - 1

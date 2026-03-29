"""
Molecular graph encoder using a Message Passing Neural Network.
Encodes SMILES -> graph -> fixed-dim embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool
from torch_geometric.nn import AttentionalAggregation
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


# --- Atom and bond featurization ---

ATOM_FEATURES = {
    'atomic_num': list(range(1, 119)),
    'degree': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
    'num_hs': [0, 1, 2, 3, 4],
    'chiral': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ],
}

BOND_FEATURES = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ],
}


def one_hot(val, choices):
    enc = [0] * (len(choices) + 1)  # +1 for "other"
    idx = choices.index(val) if val in choices else len(choices)
    enc[idx] = 1
    return enc


def atom_features(atom):
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
    feats += one_hot(atom.GetDegree(), ATOM_FEATURES['degree'])
    feats += one_hot(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge'])
    feats += one_hot(atom.GetHybridization(), ATOM_FEATURES['hybridization'])
    feats += one_hot(atom.GetTotalNumHs(), ATOM_FEATURES['num_hs'])
    feats += one_hot(atom.GetChiralTag(), ATOM_FEATURES['chiral'])
    feats += [int(atom.GetIsAromatic())]
    feats += [int(atom.IsInRing())]
    return feats


def bond_features(bond):
    feats = []
    feats += one_hot(bond.GetBondType(), BOND_FEATURES['bond_type'])
    feats += one_hot(bond.GetStereo(), BOND_FEATURES['stereo'])
    feats += [int(bond.GetIsConjugated())]
    feats += [int(bond.IsInRing())]
    return feats


ATOM_DIM = (
    len(ATOM_FEATURES['atomic_num']) + 1 +
    len(ATOM_FEATURES['degree']) + 1 +
    len(ATOM_FEATURES['formal_charge']) + 1 +
    len(ATOM_FEATURES['hybridization']) + 1 +
    len(ATOM_FEATURES['num_hs']) + 1 +
    len(ATOM_FEATURES['chiral']) + 1 +
    2   # aromatic, in_ring
)

BOND_DIM = (
    len(BOND_FEATURES['bond_type']) + 1 +
    len(BOND_FEATURES['stereo']) + 1 +
    2   # conjugated, in_ring
)


def smiles_to_graph(smiles: str):
    """Convert a SMILES string to a PyG Data object. Returns None if invalid."""
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = bond_features(bond)
        edge_index += [[i, j], [j, i]]
        edge_attr  += [bf, bf]

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, BOND_DIM), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr,  dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# --- Model ---

class MolEncoder(nn.Module):
    """
    GATv2-based molecular encoder.
    Outputs a unit-norm embedding of shape (B, embedding_dim).
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4,
                 dropout: float = 0.1, readout: str = "mean",
                 embedding_dim: int = 256):
        super().__init__()

        self.input_proj = nn.Linear(ATOM_DIM, hidden_dim)

        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim // 4,
                heads=4,
                edge_dim=BOND_DIM,
                concat=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        if readout == "attention":
            self.pool = AttentionalAggregation(gate_nn=nn.Linear(hidden_dim, 1))
        elif readout == "sum":
            self.pool = global_add_pool
        else:
            self.pool = global_mean_pool

        self.readout_type = readout

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new   # residual connection

        if self.readout_type == "attention":
            h = self.pool(h, batch)
        else:
            h = self.pool(h, batch)

        h = self.projection(h)
        return F.normalize(h, dim=-1)
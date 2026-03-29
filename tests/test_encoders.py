"""Smoke tests for the encoder forward passes."""

import torch
import pytest
from torch_geometric.data import Batch

from src.models.mol_encoder import MolEncoder, smiles_to_graph, ATOM_DIM, BOND_DIM
from src.models.morpho_encoder import MorphoEncoder
from src.models.morphoclip import MorphoCLIP

EMB_DIM = 64

MOCK_CFG = {
    'clip': {'embedding_dim': EMB_DIM, 'temperature_init': 0.07, 'learnable_temperature': True},
    'mol_encoder': {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.0, 'readout': 'mean'},
    'morpho_encoder': {'input_dim': 812, 'hidden_dims': [128], 'output_dim': EMB_DIM, 'dropout': 0.0, 'batch_norm': False},
}

SMILES = ["CC(=O)O", "c1ccccc1", "CN1CCC[C@H]1c2cccnc2"]


def test_smiles_to_graph():
    g = smiles_to_graph(SMILES[0])
    assert g is not None
    assert g.x.shape[1] == ATOM_DIM
    assert g.edge_attr.shape[1] == BOND_DIM


def test_mol_encoder_forward():
    graphs = [smiles_to_graph(s) for s in SMILES]
    batch = Batch.from_data_list(graphs)
    model = MolEncoder(hidden_dim=64, num_layers=2, embedding_dim=EMB_DIM)
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert out.shape == (len(SMILES), EMB_DIM)
    # Check unit norm
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(len(SMILES)), atol=1e-5)


def test_morpho_encoder_forward():
    model = MorphoEncoder(input_dim=812, hidden_dims=[128], output_dim=EMB_DIM, batch_norm=False)
    x = torch.randn(4, 812)
    out = model(x)
    assert out.shape == (4, EMB_DIM)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_morphoclip_forward():
    model = MorphoCLIP(MOCK_CFG)
    graphs = [smiles_to_graph(s) for s in SMILES]
    batch = Batch.from_data_list(graphs)
    morpho = torch.randn(len(SMILES), 812)
    mol_emb, morpho_emb, logits_mol, logits_morpho = model(batch, morpho)
    B = len(SMILES)
    assert mol_emb.shape    == (B, EMB_DIM)
    assert morpho_emb.shape == (B, EMB_DIM)
    assert logits_mol.shape == (B, B)
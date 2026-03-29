"""
MorphoCLIP: the full dual-encoder contrastive model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mol_encoder import MolEncoder
from .morpho_encoder import MorphoEncoder


class MorphoCLIP(nn.Module):
    """
    CLIP-style model that aligns molecular graph embeddings with
    Cell Painting morphological profile embeddings.

    Both encoders project to a shared embedding_dim space.
    Temperature is learnable (log-parameterised for stability).
    """

    def __init__(self, cfg: dict):
        super().__init__()

        emb_dim = cfg['clip']['embedding_dim']

        self.mol_encoder = MolEncoder(
            hidden_dim=cfg['mol_encoder']['hidden_dim'],
            num_layers=cfg['mol_encoder']['num_layers'],
            dropout=cfg['mol_encoder']['dropout'],
            readout=cfg['mol_encoder']['readout'],
            embedding_dim=emb_dim,
        )

        self.morpho_encoder = MorphoEncoder(
            input_dim=cfg['morpho_encoder']['input_dim'],
            hidden_dims=cfg['morpho_encoder']['hidden_dims'],
            output_dim=emb_dim,
            dropout=cfg['morpho_encoder']['dropout'],
            batch_norm=cfg['morpho_encoder']['batch_norm'],
        )

        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(cfg['clip']['temperature_init']), dtype=torch.float32),
            requires_grad=cfg['clip']['learnable_temperature'],
        )

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(min=0.01, max=10.0)

    def encode_mol(self, batch):
        """Encode a batch of molecular graphs. Returns (B, D) unit-norm."""
        return self.mol_encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

    def encode_morpho(self, profiles: torch.Tensor):
        """Encode a batch of morphological profiles. Returns (B, D) unit-norm."""
        return self.morpho_encoder(profiles)

    def forward(self, mol_batch, morpho_profiles: torch.Tensor):
        """
        Returns:
            mol_emb:    (B, D)
            morpho_emb: (B, D)
            logits_mol: (B, B) similarity matrix, mol -> morpho direction
            logits_morpho: (B, B) similarity matrix, morpho -> mol direction
        """
        mol_emb    = self.encode_mol(mol_batch)
        morpho_emb = self.encode_morpho(morpho_profiles)

        # Cosine similarity (both embeddings are already L2-normalised)
        sim = torch.matmul(mol_emb, morpho_emb.t())   # (B, B)

        logits_mol    = sim / self.temperature
        logits_morpho = sim.t() / self.temperature

        return mol_emb, morpho_emb, logits_mol, logits_morpho

    def encode_library(self, mol_loader, device):
        """
        Encode an entire molecular library to a numpy matrix.
        Used during zero-shot retrieval.
        """
        self.eval()
        embeddings = []
        with torch.no_grad():
            for batch in mol_loader:
                batch = batch.to(device)
                emb = self.encode_mol(batch)
                embeddings.append(emb.cpu().numpy())
        return np.vstack(embeddings)
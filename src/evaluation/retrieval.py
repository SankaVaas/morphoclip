"""
Zero-shot retrieval: given a molecular query, retrieve most similar
Cell Painting profiles and return their MoA annotations.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict


class ZeroShotRetriever:
    """
    After training, use this to retrieve phenotypically similar
    compounds given a query molecule — without any MoA label.
    """

    def __init__(self, model, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device
        self.model.eval()

        self._morpho_embs   = None
        self._morpho_meta   = None   # DataFrame with smiles, moa, etc.

    def index_library(self, morpho_profiles: torch.Tensor, metadata: pd.DataFrame):
        """
        Pre-encode the reference morphological library.
        Args:
            morpho_profiles: (N, 812) CellProfiler features
            metadata:         DataFrame with 'smiles', 'moa', 'compound_name' columns
        """
        with torch.no_grad():
            embs = self.model.encode_morpho(morpho_profiles.to(self.device))
        self._morpho_embs = embs.cpu()
        self._morpho_meta = metadata.reset_index(drop=True)

    def query(self, smiles: str, top_k: int = 10) -> pd.DataFrame:
        """
        Retrieve top-k morphologically similar compounds for a query SMILES.
        Returns a DataFrame with retrieved compounds and similarity scores.
        """
        from ..data.dataset import smiles_to_graph
        from torch_geometric.data import Batch

        graph = smiles_to_graph(smiles)
        if graph is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        graph = graph.to(self.device)
        batch_obj = Batch.from_data_list([graph])

        with torch.no_grad():
            mol_emb = self.model.encode_mol(batch_obj)   # (1, D)

        sims = torch.matmul(mol_emb.cpu(), self._morpho_embs.t()).squeeze(0)  # (N,)
        top_indices = sims.topk(top_k).indices.tolist()

        results = self._morpho_meta.iloc[top_indices].copy()
        results['cosine_similarity'] = sims[top_indices].numpy()
        results = results.sort_values('cosine_similarity', ascending=False)

        return results
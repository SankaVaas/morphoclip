"""
PyTorch Dataset: pairs (molecular graph, morphological profile) with MoA label.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
import pandas as pd
import numpy as np
from typing import List, Tuple

from .preprocessing import load_chembl_moa
from ..models.mol_encoder import smiles_to_graph


class MorphoCLIPDataset(Dataset):
    """
    Each item: (mol_graph, morpho_profile, moa_label)

    The dataset is constructed by matching compounds in the JUMP-CP
    metadata (via Metadata_broad_sample or Metadata_pert_iname)
    with MoA annotations from ChEMBL.
    """

    def __init__(self, profiles: pd.DataFrame, metadata: pd.DataFrame,
                 chembl_df: pd.DataFrame, profile_cols: List[str]):
        super().__init__()

        # Match on InChIKey or compound name — here we use a name-based join
        # In the real pipeline, replace with InChIKey matching for robustness
        merged = metadata.merge(
            chembl_df[['smiles', 'moa', 'compound_name']],
            left_on='Metadata_broad_sample',
            right_on='compound_name',
            how='inner',
        )

        self.graphs   = []
        self.profiles = []
        self.moa_labels = []

        for _, row in merged.iterrows():
            g = smiles_to_graph(row['smiles'])
            if g is None:
                continue
            self.graphs.append(g)
            # Fetch the corresponding profile row
            idx = merged.index.get_loc(row.name)
            self.profiles.append(
                torch.tensor(profiles.iloc[idx][profile_cols].values, dtype=torch.float32)
            )
            self.moa_labels.append(row['moa'])

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.profiles[idx], self.moa_labels[idx]


def morphoclip_collate(batch):
    """Custom collate: batch graphs with PyG, stack profiles as tensor."""
    graphs, profiles, moas = zip(*batch)
    mol_batch = Batch.from_data_list(list(graphs))
    morpho_tensor = torch.stack(profiles)
    return mol_batch, morpho_tensor, list(moas)


def get_dataloaders(profiles, metadata, chembl_df, profile_cols, cfg):
    from torch.utils.data import DataLoader, random_split

    dataset = MorphoCLIPDataset(profiles, metadata, chembl_df, profile_cols)

    n = len(dataset)
    n_train = int(n * cfg['data']['train_split'])
    n_val   = int(n * cfg['data']['val_split'])
    n_test  = n - n_train - n_val

    torch.manual_seed(cfg['data']['random_seed'])
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    loader_kwargs = dict(
        batch_size=cfg['training']['batch_size'],
        collate_fn=morphoclip_collate,
        num_workers=0,
    )

    return (
        DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    )
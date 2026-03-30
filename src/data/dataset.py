"""
PyTorch Dataset: loads pre-matched (molecular graph, morphological profile, MoA) triples
from data/processed/matched_pairs.csv and data/processed/jump_profiles.csv.
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Batch
import pandas as pd
import numpy as np

from ..models.mol_encoder import smiles_to_graph


class MorphoCLIPDataset(Dataset):
    """
    Loads matched pairs from the preprocessed CSV files.
    Each item: (mol_graph, morpho_profile_tensor, moa_label)
    """

    def __init__(self, matched_pairs: pd.DataFrame,
                 profiles: pd.DataFrame, profile_cols: list):
        super().__init__()

        self.graphs      = []
        self.profiles    = []
        self.moa_labels  = []

        skipped = 0
        for _, row in matched_pairs.iterrows():
            g = smiles_to_graph(row["smiles"])
            if g is None:
                skipped += 1
                continue

            idx = int(row["profile_idx"])
            if idx >= len(profiles):
                skipped += 1
                continue

            prof_vec = profiles.iloc[idx][profile_cols].values.astype(np.float32)
            if np.isnan(prof_vec).any():
                skipped += 1
                continue

            self.graphs.append(g)
            self.profiles.append(torch.tensor(prof_vec, dtype=torch.float32))
            self.moa_labels.append(str(row["moa"]))

        print(f"  Dataset: {len(self.graphs)} valid pairs "
              f"({skipped} skipped due to invalid SMILES or NaN profiles)")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.profiles[idx], self.moa_labels[idx]


def morphoclip_collate(batch):
    """Custom collate: batch graphs with PyG, stack profiles as tensor."""
    graphs, profiles, moas = zip(*batch)
    mol_batch     = Batch.from_data_list(list(graphs))
    morpho_tensor = torch.stack(profiles)
    return mol_batch, morpho_tensor, list(moas)


def get_dataloaders(profiles, metadata, chembl_df, profile_cols, cfg):
    """
    Load matched pairs from disk and build train/val/test DataLoaders.
    Ignores metadata and chembl_df — matching was done in preprocess.py.
    """
    matched_path = cfg['data'].get('matched_pairs_path',
                                   'data/processed/matched_pairs.csv')
    matched = pd.read_csv(matched_path)
    print(f"  Loaded {len(matched)} matched pairs from {matched_path}")

    if len(matched) == 0:
        raise RuntimeError(
            "matched_pairs.csv is empty. "
            "Re-run: python scripts/preprocess.py"
        )

    dataset = MorphoCLIPDataset(matched, profiles, profile_cols)

    if len(dataset) < 3:
        raise RuntimeError(
            f"Dataset too small ({len(dataset)} valid pairs). "
            "Check preprocess output and matched_pairs.csv."
        )

    n       = len(dataset)
    n_train = max(1, int(n * cfg['data']['train_split']))
    n_val   = max(1, int(n * cfg['data']['val_split']))
    n_test  = max(1, n - n_train - n_val)

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
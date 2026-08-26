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

    Consumes the already-matched (profile_idx, smiles, moa) table produced by
    scripts/preprocess.py (data/processed/matched_pairs.csv), and looks up each
    profile row by its explicit `profile_idx` — the positional index into
    `profiles` at the time the match was made. This is the ONLY safe way to
    align a molecule with its morphological profile: doing a fresh merge here
    (the previous approach) silently desynchronises the two, because merging
    drops/reorders rows and post-merge positional indices no longer point at
    the same rows in the original `profiles` frame.
    """

    def __init__(self, profiles: pd.DataFrame, matched: pd.DataFrame,
                 profile_cols: List[str]):
        super().__init__()

        if "split" not in matched.columns:
            raise ValueError(
                "matched_pairs.csv has no 'split' column. This means it was "
                "produced by an older version of scripts/preprocess.py — "
                "rerun preprocessing so the compound-level train/val/test "
                "split is assigned (and normalisation is fit) before "
                "profiles are written out."
            )

        self.graphs   = []
        self.profiles = []
        self.moa_labels = []
        self.compound_ids = []   
        self.splits = []        

        profiles_values = profiles[profile_cols].values

        for _, row in matched.iterrows():
            g = smiles_to_graph(row['smiles'])
            if g is None:
                continue
            self.graphs.append(g)
            profile_row = profiles_values[int(row['profile_idx'])]
            self.profiles.append(torch.tensor(profile_row, dtype=torch.float32))
            self.moa_labels.append(row['moa'])
            self.compound_ids.append(row.get('compound_name', row['smiles']))
            self.splits.append(row['split'])

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


def get_dataloaders(profiles, matched, profile_cols, cfg):
    """
    Builds train/val/test DataLoaders using the compound-level split that
    scripts/preprocess.py already assigned (matched['split']) — this is
    intentionally NOT re-split here. The split has to be decided before
    normalisation (so normalisation stats can be fit on train-only wells),
    so recomputing a different split at this stage would desynchronise the
    two and reintroduce leakage.
    """
    from torch.utils.data import DataLoader, Subset

    dataset = MorphoCLIPDataset(profiles, matched, profile_cols)
    splits  = np.array(dataset.splits)
    groups  = np.array(dataset.compound_ids)

    train_idx = np.where(splits == "train")[0]
    val_idx   = np.where(splits == "val")[0]
    test_idx  = np.where(splits == "test")[0]

    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        if len(idx) == 0:
            raise ValueError(f"Split '{name}' is empty after loading dataset items.")

    # Safety net: re-verify the pre-assigned split really is leakage-free at the compound level
    train_groups = set(groups[train_idx])
    val_groups   = set(groups[val_idx])
    test_groups  = set(groups[test_idx])
    assert not (train_groups & val_groups),  "compound leakage: train/val overlap"
    assert not (train_groups & test_groups), "compound leakage: train/test overlap"
    assert not (val_groups & test_groups),   "compound leakage: val/test overlap"

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds   = Subset(dataset, val_idx.tolist())
    test_ds  = Subset(dataset, test_idx.tolist())

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
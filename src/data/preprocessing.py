"""
Download and preprocess JUMP-CP and ChEMBL MoA data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
import os


# JUMP-CP well-level profiles are hosted by Broad Institute
JUMP_CP_URL = (
    "https://cellpainting-gallery.s3.amazonaws.com/"
    "cpg0016-jump/source_4/workspace/profiles/"
    "2020_11_04_CPJUMP1/BR00117006/BR00117006_normalized_feature_select_negcon.csv.gz"
)


def download_jump_cp(save_path: str = "data/raw/jump_profiles.csv.gz"):
    """Download a JUMP-CP normalized feature file."""
    import urllib.request
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading JUMP-CP profiles to {save_path}...")
    urllib.request.urlretrieve(JUMP_CP_URL, save_path)
    print("Done.")


def load_and_clean_jump_cp(raw_path: str) -> pd.DataFrame:
    """
    Load JUMP-CP profiles, keep only numeric CellProfiler features,
    drop NaN rows, and median-normalise per feature.
    """
    df = pd.read_csv(raw_path)

    # Metadata columns in JUMP-CP follow the 'Metadata_' prefix convention
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feat_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    profiles = df[feat_cols].select_dtypes(include=[np.number])
    metadata = df[meta_cols]

    # Drop rows with any NaN
    valid_mask = profiles.notna().all(axis=1)
    profiles = profiles[valid_mask].reset_index(drop=True)
    metadata = metadata[valid_mask].reset_index(drop=True)

    # Robust normalisation: median-center and MAD-scale each feature
    median = profiles.median()
    mad    = (profiles - median).abs().median()
    mad[mad == 0] = 1.0   # prevent division by zero on constant features
    profiles = (profiles - median) / mad

    return profiles, metadata


def load_chembl_moa(path: str = "data/raw/chembl_moa.csv") -> pd.DataFrame:
    """
    Load a ChEMBL-derived MoA annotation file.
    Expected columns: smiles, moa, target_name
    If not present, a minimal synthetic example is created for testing.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        print(f"[Warning] {path} not found — generating a tiny mock dataset for testing.")
        df = _make_mock_chembl()
    # Validate SMILES
    df = df[df['smiles'].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    return df.reset_index(drop=True)


def _make_mock_chembl():
    """A tiny mock MoA dataset for smoke-testing the pipeline."""
    data = {
        'smiles': [
            'CC(=O)Oc1ccccc1C(=O)O',       # Aspirin
            'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # Testosterone
            'c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34',      # Pyrene
            'CN1CCC[C@H]1c2cccnc2',                  # Nicotine
            'OC(=O)c1ccccc1O',                       # Salicylic acid
        ],
        'moa': ['COX inhibitor', 'androgen receptor agonist',
                'AhR ligand', 'nicotinic receptor agonist', 'COX inhibitor'],
        'compound_name': ['Aspirin', 'Testosterone', 'Pyrene', 'Nicotine', 'Salicylic acid'],
    }
    return pd.DataFrame(data)
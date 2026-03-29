"""
Preprocess raw JUMP-CP profiles and ChEMBL MoA data into model-ready files.
Usage: python scripts/preprocess.py

Outputs:
  data/processed/jump_profiles.csv      — cleaned, normalised CellProfiler features
  data/processed/jump_metadata.csv      — compound metadata aligned to profile rows
  data/processed/chembl_moa.csv         — validated SMILES + MoA labels
  data/processed/matched_pairs.csv      — inner join: profile rows matched to SMILES+MoA
  data/processed/dataset_stats.txt      — summary statistics
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


# ── helpers ──────────────────────────────────────────────────────────────────

def robust_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median-center and MAD-scale each feature column.
    Constant features (MAD == 0) are dropped entirely.
    """
    median = df.median()
    mad = (df - median).abs().median()

    constant_cols = mad[mad == 0].index.tolist()
    if constant_cols:
        print(f"  Dropping {len(constant_cols)} constant features")
        df = df.drop(columns=constant_cols)
        mad = mad.drop(index=constant_cols)
        median = median.drop(index=constant_cols)

    return (df - median) / mad


def standardise_smiles(smi: str) -> str | None:
    """
    Canonicalise and sanitise a SMILES string.
    Returns None if the molecule cannot be parsed or standardised.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # Remove salts, neutralise charges
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_jump_profiles(raw_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and concatenate all JUMP-CP plate CSVs.
    Returns (features_df, metadata_df).
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "jump_BR*.csv.gz")))
    if not files:
        raise FileNotFoundError(
            "No JUMP-CP profile files found in data/raw/. "
            "Run: python scripts/download_data.py"
        )

    print(f"  Found {len(files)} plate file(s): {[Path(f).name for f in files]}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total wells loaded: {len(df)}")

    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feat_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    metadata = df[meta_cols].copy()
    features = df[feat_cols].select_dtypes(include=[np.number]).copy()

    # Drop rows with any NaN in features
    valid = features.notna().all(axis=1)
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} rows with NaN features")
    features = features[valid].reset_index(drop=True)
    metadata = metadata[valid].reset_index(drop=True)

    return features, metadata


def load_jump_compound_meta(path: str = "data/raw/jump_compound_metadata.csv.gz") -> pd.DataFrame:
    """Load JUMP-CP compound metadata mapping broad_sample -> InChIKey -> SMILES."""
    if not os.path.exists(path):
        print("  [warn] JUMP compound metadata not found — skipping InChIKey join")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"  JUMP compound metadata: {len(df)} compounds")
    return df


def load_chembl_moa(path: str = "data/raw/chembl_moa_raw.csv") -> pd.DataFrame:
    """
    Load ChEMBL MoA CSV. Expected columns (flexible matching):
      canonical_smiles / smiles, moa / mechanism_of_action, compound_name / pref_name
    Falls back to mock dataset if file not found.
    """
    if not os.path.exists(path):
        print("  [warn] ChEMBL MoA file not found — using mock dataset")
        return _mock_chembl()

    df = pd.read_csv(path)
    print(f"  Raw ChEMBL MoA rows: {len(df)}")

    # Normalise column names
    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if "smiles" in lc:
            col_map[col] = "smiles"
        elif "mechanism" in lc or lc == "moa":
            col_map[col] = "moa"
        elif "pref_name" in lc or "compound_name" in lc or "name" in lc:
            col_map[col] = "compound_name"
    df = df.rename(columns=col_map)

    required = {"smiles", "moa"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [warn] Missing columns {missing} — using mock dataset")
        return _mock_chembl()

    if "compound_name" not in df.columns:
        df["compound_name"] = df["smiles"].str[:20]

    return df[["smiles", "moa", "compound_name"]].dropna(subset=["smiles", "moa"])


def _mock_chembl() -> pd.DataFrame:
    """Minimal labelled dataset for smoke-testing when real data is absent."""
    rows = [
        ("CC(=O)Oc1ccccc1C(=O)O",                         "COX inhibitor",                "Aspirin"),
        ("OC(=O)c1ccccc1O",                                "COX inhibitor",                "Salicylic acid"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",                    "COX inhibitor",                "Ibuprofen"),
        ("CN1CCC[C@H]1c2cccnc2",                           "nAChR agonist",                "Nicotine"),
        ("c1ccc2c(c1)[nH]c1ccccc12",                       "nAChR agonist",                "Carbazole"),
        ("C[C@H](N)Cc1ccccc1",                             "nAChR agonist",                "Amphetamine"),
        ("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",        "SGLT2 inhibitor",              "Glucose"),
        ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",           "androgen receptor agonist",    "Testosterone"),
        ("C[C@@]12CC[C@H]3[C@@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@H]34",
                                                            "estrogen receptor agonist",    "Estradiol"),
        ("CN(C)CCCN1c2ccccc2Sc3ccc(Cl)cc13",              "dopamine antagonist",          "Chlorpromazine"),
        ("O=C(O)c1ccc(cc1)N",                              "PABA",                         "PABA"),
        ("Nc1ccc(cc1)S(=O)(=O)N",                          "sulfonamide antibiotic",       "Sulfanilamide"),
    ]
    return pd.DataFrame(rows, columns=["smiles", "moa", "compound_name"])


# ── matching ──────────────────────────────────────────────────────────────────

def match_profiles_to_moa(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    chembl_df: pd.DataFrame,
    jump_meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join strategy (in priority order):
      1. If JUMP compound metadata available: join via InChIKey
      2. Fallback: fuzzy name match on Metadata_broad_sample / Metadata_pert_iname

    Returns a DataFrame with columns:
      profile_idx, smiles, moa, compound_name  + all Metadata_ columns
    """
    # Standardise ChEMBL SMILES
    print("  Standardising ChEMBL SMILES...")
    chembl_df = chembl_df.copy()
    chembl_df["smiles"] = chembl_df["smiles"].apply(standardise_smiles)
    chembl_df = chembl_df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
    print(f"  Valid ChEMBL SMILES after standardisation: {len(chembl_df)}")

    # Try InChIKey join if compound metadata available
    if not jump_meta.empty and "Metadata_InChIKey" in metadata.columns:
        print("  Joining via InChIKey...")
        from rdkit.Chem.inchi import MolToInchiKey
        chembl_df["inchikey"] = chembl_df["smiles"].apply(
            lambda s: MolToInchiKey(Chem.MolFromSmiles(s)) if s else None
        )
        meta_with_idx = metadata.copy()
        meta_with_idx["profile_idx"] = meta_with_idx.index
        merged = meta_with_idx.merge(
            chembl_df[["inchikey", "smiles", "moa", "compound_name"]],
            left_on="Metadata_InChIKey",
            right_on="inchikey",
            how="inner",
        )
    else:
        # Name-based fallback
        print("  Falling back to name-based matching...")
        name_col = next(
            (c for c in metadata.columns
             if "pert_iname" in c.lower() or "broad_sample" in c.lower()),
            None,
        )
        if name_col is None:
            print("  [warn] No usable join key found — returning empty matches")
            return pd.DataFrame()

        meta_with_idx = metadata.copy()
        meta_with_idx["profile_idx"] = meta_with_idx.index
        meta_with_idx["_join_key"] = meta_with_idx[name_col].str.lower().str.strip()
        chembl_df["_join_key"] = chembl_df["compound_name"].str.lower().str.strip()

        merged = meta_with_idx.merge(
            chembl_df[["_join_key", "smiles", "moa", "compound_name"]],
            on="_join_key",
            how="inner",
        ).drop(columns=["_join_key"])

    print(f"  Matched pairs: {len(merged)}")
    return merged


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs("data/processed", exist_ok=True)
    print("=" * 55)
    print("  MorphoCLIP — preprocessing")
    print("=" * 55)

    # 1. Load profiles
    print("\n[1/5] Loading JUMP-CP profiles...")
    features, metadata = load_jump_profiles()

    # 2. Normalise
    print("\n[2/5] Normalising features (robust MAD scaling)...")
    features_norm = robust_normalise(features)
    print(f"  Feature matrix: {features_norm.shape[0]} wells × {features_norm.shape[1]} features")

    # 3. Load ChEMBL MoA
    print("\n[3/5] Loading ChEMBL MoA annotations...")
    chembl_df = load_chembl_moa()
    moa_counts = chembl_df["moa"].value_counts()
    print(f"  Unique MoA classes: {len(moa_counts)}")
    print(f"  Top 5 classes:\n{moa_counts.head().to_string()}")

    # 4. Load JUMP compound metadata
    print("\n[4/5] Loading JUMP compound metadata...")
    jump_meta = load_jump_compound_meta()

    # 5. Match
    print("\n[5/5] Matching profiles to MoA annotations...")
    matched = match_profiles_to_moa(features_norm, metadata, chembl_df, jump_meta)

    if matched.empty:
        print("\n[warn] No matches found — saving mock-matched dataset for pipeline testing")
        matched = _build_mock_matched(features_norm)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nSaving processed files...")

    features_norm.to_csv("data/processed/jump_profiles.csv", index=False)
    print(f"  data/processed/jump_profiles.csv  ({features_norm.shape})")

    metadata.to_csv("data/processed/jump_metadata.csv", index=False)
    print(f"  data/processed/jump_metadata.csv  ({metadata.shape})")

    chembl_df[["smiles", "moa", "compound_name"]].to_csv(
        "data/processed/chembl_moa.csv", index=False
    )
    print(f"  data/processed/chembl_moa.csv  ({chembl_df.shape})")

    matched.to_csv("data/processed/matched_pairs.csv", index=False)
    print(f"  data/processed/matched_pairs.csv  ({matched.shape})")

    # Stats
    stats = [
        f"Wells (post-QC):        {features_norm.shape[0]}",
        f"CellProfiler features:  {features_norm.shape[1]}",
        f"Unique MoA classes:     {chembl_df['moa'].nunique()}",
        f"Matched mol-morpho pairs: {len(matched)}",
        f"MoA distribution:\n{matched['moa'].value_counts().to_string()}",
    ]
    stats_txt = "\n".join(stats)
    with open("data/processed/dataset_stats.txt", "w") as f:
        f.write(stats_txt)
    print(f"\n{stats_txt}")
    print("\nPreprocessing complete. Run next:")
    print("  python scripts/train.py --config configs/default.yaml")


def _build_mock_matched(features_norm: pd.DataFrame) -> pd.DataFrame:
    """
    When real matching fails, build a synthetic matched DataFrame
    by replicating mock compounds across the first N profile rows.
    Useful for validating the full training pipeline on CPU.
    """
    mock = _mock_chembl()
    n = min(len(features_norm), 120)
    rows = []
    for i in range(n):
        compound = mock.iloc[i % len(mock)]
        rows.append({
            "profile_idx": i,
            "smiles": compound["smiles"],
            "moa": compound["moa"],
            "compound_name": compound["compound_name"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw")
    args = parser.parse_args()
    main(args)
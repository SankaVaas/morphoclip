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
    Load and concatenate JUMP-CP plate CSVs.
    - Real plates (jump_BR*.csv.gz) and mock (jump_mock.csv.gz) are loaded separately
      then reconciled to a common feature set before concatenation.
    - Drops COLUMNS with >20% NaN, then drops any remaining NaN rows.
    """
    real_files = sorted(glob.glob(os.path.join(raw_dir, "jump_BR*.csv.gz")))
    mock_files = sorted(glob.glob(os.path.join(raw_dir, "jump_mock.csv.gz")))

    # Prefer real files; only use mock if no real files exist
    files = real_files if real_files else mock_files

    if not files:
        raise FileNotFoundError(
            "No JUMP-CP profile files found in data/raw/. "
            "Run: python scripts/download_data.py"
        )

    print(f"  Found {len(files)} plate file(s): {[Path(f).name for f in files]}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"    {Path(f).name}: {df.shape[0]} wells, {df.shape[1]} cols")
        dfs.append(df)

    # Find common columns across all plates
    if len(dfs) > 1:
        common_cols = set(dfs[0].columns)
        for df in dfs[1:]:
            common_cols &= set(df.columns)
        common_cols = sorted(common_cols)
        print(f"  Common columns across plates: {len(common_cols)}")
        dfs = [df[common_cols] for df in dfs]

    df = pd.concat(dfs, ignore_index=True)
    print(f"  Total wells loaded: {len(df)}")

    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    feat_cols = [c for c in df.columns if not c.startswith("Metadata_")]

    metadata = df[meta_cols].copy()
    features = df[feat_cols].select_dtypes(include=[np.number]).copy()

    # Drop columns with >20% NaN first
    nan_frac = features.isna().mean()
    bad_cols = nan_frac[nan_frac > 0.2].index.tolist()
    if bad_cols:
        print(f"  Dropping {len(bad_cols)} high-NaN columns (>20% missing)")
        features = features.drop(columns=bad_cols)

    # Now drop remaining rows with any NaN
    valid = features.notna().all(axis=1)
    n_dropped = (~valid).sum()
    if n_dropped:
        print(f"  Dropping {n_dropped} rows with remaining NaN values")
    features = features[valid].reset_index(drop=True)
    metadata = metadata[valid].reset_index(drop=True)

    print(f"  Clean feature matrix: {features.shape[0]} wells × {features.shape[1]} features")
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
    Join strategy:
      1. Use Metadata_smiles (already in JUMP profiles) — no external join needed
      2. Match to ChEMBL MoA via standardised InChIKey
      3. Fallback: pert_iname name match
    """
    from rdkit.Chem.inchi import MolToInchiKey

    # Standardise ChEMBL SMILES and compute InChIKeys
    print("  Standardising ChEMBL SMILES and computing InChIKeys...")
    chembl_df = chembl_df.copy()
    chembl_df["std_smiles"] = chembl_df["smiles"].apply(standardise_smiles)
    chembl_df = chembl_df.dropna(subset=["std_smiles"])

    def smiles_to_inchikey(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return MolToInchiKey(mol) if mol else None
        except Exception:
            return None

    chembl_df["inchikey14"] = chembl_df["std_smiles"].apply(
        lambda s: smiles_to_inchikey(s)[:14] if smiles_to_inchikey(s) else None
    )
    chembl_df = chembl_df.dropna(subset=["inchikey14"]).drop_duplicates(subset=["inchikey14"])
    print(f"  ChEMBL compounds with valid InChIKey: {len(chembl_df)}")

    # Use SMILES already present in the JUMP metadata
    meta_with_idx = metadata.copy()
    meta_with_idx["profile_idx"] = meta_with_idx.index

    if "Metadata_InChIKey" in meta_with_idx.columns:
        print("  Joining via Metadata_InChIKey (first 14 chars)...")
        meta_with_idx["inchikey14"] = meta_with_idx["Metadata_InChIKey"].str[:14]
        merged = meta_with_idx.merge(
            chembl_df[["inchikey14", "std_smiles", "moa", "compound_name"]],
            on="inchikey14",
            how="inner",
        ).rename(columns={"std_smiles": "smiles"})

    elif "Metadata_smiles" in meta_with_idx.columns:
        print("  Computing InChIKeys from Metadata_smiles...")
        meta_with_idx["inchikey14"] = meta_with_idx["Metadata_smiles"].apply(
            lambda s: smiles_to_inchikey(s)[:14]
            if isinstance(s, str) and smiles_to_inchikey(s) else None
        )
        merged = meta_with_idx.dropna(subset=["inchikey14"]).merge(
            chembl_df[["inchikey14", "std_smiles", "moa", "compound_name"]],
            on="inchikey14",
            how="inner",
        ).rename(columns={"std_smiles": "smiles"})

    else:
        print("  Falling back to pert_iname name match...")
        meta_with_idx["_key"] = meta_with_idx.get(
            "Metadata_pert_iname", pd.Series(dtype=str)
        ).str.lower().str.strip()
        chembl_df["_key"] = chembl_df["compound_name"].str.lower().str.strip()
        merged = meta_with_idx.merge(
            chembl_df[["_key", "std_smiles", "moa", "compound_name"]],
            on="_key", how="inner",
        ).rename(columns={"std_smiles": "smiles"}).drop(columns=["_key"])

    # For unmatched wells, use Metadata_smiles directly with pert_iname as label
    if len(merged) < 100 and "Metadata_smiles" in meta_with_idx.columns:
        print("  Few ChEMBL matches — enriching with JUMP-native SMILES+pert_iname pairs...")
        unmatched = meta_with_idx[
            ~meta_with_idx["profile_idx"].isin(merged.get("profile_idx", pd.Series()))
        ].copy()
        unmatched = unmatched[
            unmatched["Metadata_smiles"].notna() &
            unmatched["Metadata_pert_iname"].notna() &
            (unmatched["Metadata_pert_type"] == "trt")   # compounds only, not controls
        ].copy()
        unmatched["smiles"]         = unmatched["Metadata_smiles"].apply(standardise_smiles)
        unmatched["moa"]            = unmatched["Metadata_pert_iname"]  # use name as proxy label
        unmatched["compound_name"]  = unmatched["Metadata_pert_iname"]
        unmatched = unmatched.dropna(subset=["smiles"])
        merged = pd.concat([merged, unmatched], ignore_index=True)
        print(f"  After enrichment: {len(merged)} pairs")

    print(f"  Final matched pairs: {len(merged)}")
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

    # 4. Skip external compound metadata — SMILES are in the profiles directly
    print("\n[4/5] Skipping external compound metadata (SMILES in profiles already)...")
    jump_meta = pd.DataFrame()

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
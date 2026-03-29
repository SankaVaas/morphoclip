"""
Download JUMP-CP profiles and ChEMBL MoA annotations.

Strategy:
  - Primary:  cpg0000-jump-pilot profiles (HTTP, no AWS CLI needed)
              These are the published benchmark plates from the 2022 paper.
  - Fallback: synthetic mock dataset so the full pipeline can be validated
              locally on CPU without any download at all (--mock flag).

Usage:
    python scripts/download_data.py           # full download
    python scripts/download_data.py --mock    # skip downloads, use mock data
"""

import os
import urllib.request
import argparse
from pathlib import Path


# ── cpg0000-jump-pilot: these HTTP URLs are stable and confirmed working ──────
# Well-level, feature-selected, normalized-to-negcon profiles
# Plate BR00117010 from batch 2020_11_04_CPJUMP1 (~8 MB compressed)
JUMP_PILOT_FILES = [
    (
        "https://cellpainting-gallery.s3.amazonaws.com/"
        "cpg0000-jump-pilot/source_4/workspace/profiles/"
        "2020_11_04_CPJUMP1/BR00117010/"
        "BR00117010_normalized_feature_select_negcon.csv.gz",
        "data/raw/jump_BR00117010.csv.gz",
    ),
    (
        "https://cellpainting-gallery.s3.amazonaws.com/"
        "cpg0000-jump-pilot/source_4/workspace/profiles/"
        "2020_11_04_CPJUMP1/BR00117012/"
        "BR00117012_normalized_feature_select_negcon.csv.gz",
        "data/raw/jump_BR00117012.csv.gz",
    ),
]

# JUMP compound metadata: broad_sample -> InChIKey -> SMILES
# Hosted on GitHub (confirmed stable)
JUMP_COMPOUND_META_URL = (
    "https://raw.githubusercontent.com/jump-cellpainting/JUMP-Target/"
    "main/JUMP-Target-2_compound_metadata.tsv"
)

# ChEMBL-derived MoA table with SMILES (Pat Walters' curated teaching dataset)
CHEMBL_MOA_URL = (
    "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/"
    "main/data/chembl_mechanism.csv"
)


def download_file(url: str, dest: str, desc: str = "") -> bool:
    """Download url -> dest. Returns True on success, False on failure."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  [skip] {dest} already exists ({size_mb:.1f} MB)")
        return True
    print(f"  Downloading {desc or Path(url).name} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, \
             open(dest, "wb") as f:
            f.write(resp.read())
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Saved -> {dest} ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"  [WARN] Could not download {Path(url).name}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False


def download_jump_profiles() -> int:
    """Returns number of successfully downloaded profile files."""
    print("\n[1/3] Downloading JUMP-CP pilot profiles (cpg0000)...")
    ok = 0
    for url, dest in JUMP_PILOT_FILES:
        if download_file(url, dest):
            ok += 1
    print(f"  {ok}/{len(JUMP_PILOT_FILES)} profile files downloaded")
    return ok


def download_jump_metadata() -> bool:
    print("\n[2/3] Downloading JUMP compound metadata (SMILES + InChIKey)...")
    return download_file(
        JUMP_COMPOUND_META_URL,
        "data/raw/jump_compound_metadata.tsv",
        desc="JUMP-Target-2_compound_metadata.tsv",
    )


def download_chembl_moa() -> bool:
    print("\n[3/3] Downloading ChEMBL MoA annotations...")
    return download_file(
        CHEMBL_MOA_URL,
        "data/raw/chembl_moa_raw.csv",
        desc="chembl_mechanism.csv",
    )


def write_mock_profiles():
    """
    Write a tiny synthetic profiles CSV so the pipeline works immediately
    without any real data. Useful for CPU smoke-testing on Windows/local.
    """
    import pandas as pd
    import numpy as np

    print("\n  Generating synthetic mock profiles (500 wells × 200 features)...")
    os.makedirs("data/raw", exist_ok=True)
    rng = np.random.default_rng(42)

    n_wells = 500
    n_feats = 200

    # Mock compound names drawn from our known MoA set
    compounds = [
        "Aspirin", "Salicylic acid", "Ibuprofen",
        "Nicotine", "Carbazole",
        "Testosterone", "Estradiol",
        "Chlorpromazine", "Sulfanilamide",
    ]
    pert_names = [compounds[i % len(compounds)] for i in range(n_wells)]

    meta = pd.DataFrame({
        "Metadata_broad_sample": pert_names,
        "Metadata_Well": [f"{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[( i//12) % 26]}{(i%12)+1:02d}" for i in range(n_wells)],
        "Metadata_Plate": ["BR00000000"] * n_wells,
    })

    # Features: compound class signal + noise
    class_idx = [compounds.index(p) for p in pert_names]
    signal = rng.standard_normal((len(compounds), n_feats)) * 2
    noise  = rng.standard_normal((n_wells, n_feats)) * 0.5
    feats  = signal[class_idx] + noise

    feat_cols = [f"CP_feature_{i:03d}" for i in range(n_feats)]
    feat_df   = pd.DataFrame(feats, columns=feat_cols)

    df = pd.concat([meta, feat_df], axis=1)
    dest = "data/raw/jump_mock.csv.gz"
    df.to_csv(dest, index=False)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  Saved -> {dest} ({size_mb:.2f} MB, {n_wells} wells)")


def write_mock_chembl():
    """Write the mock ChEMBL MoA CSV used when real download fails."""
    import pandas as pd
    rows = [
        ("CC(=O)Oc1ccccc1C(=O)O",                      "COX inhibitor",             "Aspirin"),
        ("OC(=O)c1ccccc1O",                             "COX inhibitor",             "Salicylic acid"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",                 "COX inhibitor",             "Ibuprofen"),
        ("CN1CCC[C@H]1c2cccnc2",                        "nAChR agonist",             "Nicotine"),
        ("c1ccc2c(c1)[nH]c1ccccc12",                    "nAChR agonist",             "Carbazole"),
        ("CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",        "androgen receptor agonist", "Testosterone"),
        ("C[C@@]12CC[C@H]3[C@@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@H]34",
                                                         "estrogen receptor agonist", "Estradiol"),
        ("CN(C)CCCN1c2ccccc2Sc3ccc(Cl)cc13",           "dopamine antagonist",       "Chlorpromazine"),
        ("Nc1ccc(cc1)S(=O)(=O)N",                       "sulfonamide antibiotic",    "Sulfanilamide"),
    ]
    os.makedirs("data/raw", exist_ok=True)
    dest = "data/raw/chembl_moa_raw.csv"
    pd.DataFrame(rows, columns=["canonical_smiles", "mechanism_of_action", "pref_name"])\
      .to_csv(dest, index=False)
    print(f"  Saved mock ChEMBL MoA -> {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock", action="store_true",
        help="Skip all HTTP downloads and write synthetic mock files instead. "
             "Use this to test the full pipeline locally without real data."
    )
    parser.add_argument("--skip-jump",   action="store_true")
    parser.add_argument("--skip-chembl", action="store_true")
    args = parser.parse_args()

    print("=" * 55)
    print("  MorphoCLIP — data download")
    print("=" * 55)

    if args.mock:
        print("\n  --mock flag set: generating synthetic data only")
        write_mock_profiles()
        write_mock_chembl()
        print("\nMock data ready. Run next:")
        print("  python scripts/preprocess.py")
        return

    jump_ok = 0
    if not args.skip_jump:
        jump_ok += download_jump_profiles()
        download_jump_metadata()

    chembl_ok = True
    if not args.skip_chembl:
        chembl_ok = download_chembl_moa()

    # Fallbacks
    if jump_ok == 0:
        print("\n  [fallback] No JUMP profiles downloaded — generating mock profiles")
        write_mock_profiles()

    if not chembl_ok:
        print("\n  [fallback] ChEMBL download failed — writing mock MoA file")
        write_mock_chembl()

    print("\nDownloads complete. Run next:")
    print("  python scripts/preprocess.py")


if __name__ == "__main__":
    main()
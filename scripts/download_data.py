"""
Download JUMP-CP profiles and ChEMBL MoA annotations.
Usage: python scripts/download_data.py
"""

import os
import urllib.request
import argparse
from pathlib import Path


# --- JUMP-CP: we pull 3 plates from CPJUMP1 for a meaningful dataset size
# These are normalized, feature-selected profiles from Broad Institute S3
JUMP_CP_FILES = [
    (
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/"
        "source_4/workspace/profiles/2020_11_04_CPJUMP1/"
        "BR00117006/BR00117006_normalized_feature_select_negcon.csv.gz",
        "data/raw/jump_BR00117006.csv.gz",
    ),
    (
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/"
        "source_4/workspace/profiles/2020_11_04_CPJUMP1/"
        "BR00117010/BR00117010_normalized_feature_select_negcon.csv.gz",
        "data/raw/jump_BR00117010.csv.gz",
    ),
    (
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/"
        "source_4/workspace/profiles/2020_11_04_CPJUMP1/"
        "BR00117012/BR00117012_normalized_feature_select_negcon.csv.gz",
        "data/raw/jump_BR00117012.csv.gz",
    ),
]

# JUMP-CP compound metadata: maps broad_sample IDs to InChIKeys and SMILES
JUMP_COMPOUND_META_URL = (
    "https://raw.githubusercontent.com/jump-cellpainting/datasets/"
    "main/metadata/compound.csv.gz"
)

# ChEMBL-derived MoA table (curated subset with SMILES + MoA + target)
# Sourced from ChEMBL mechanism-of-action annotations via Open Targets
CHEMBL_MOA_URL = (
    "https://raw.githubusercontent.com/chembl/chembl_webresource_client/"
    "master/chembl_webresource_client/test_data/molecule.json"
)


def download_file(url: str, dest: str, desc: str = ""):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"  [skip] {dest} already exists")
        return
    print(f"  Downloading {desc or url.split('/')[-1]} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"  Saved {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        raise


def download_jump_cp():
    print("\n[1/3] Downloading JUMP-CP normalized profiles...")
    for url, dest in JUMP_CP_FILES:
        download_file(url, dest, desc=Path(dest).name)


def download_jump_metadata():
    print("\n[2/3] Downloading JUMP-CP compound metadata...")
    download_file(
        JUMP_COMPOUND_META_URL,
        "data/raw/jump_compound_metadata.csv.gz",
        desc="jump_compound_metadata.csv.gz",
    )


def download_chembl_moa():
    """
    The cleanest public source for SMILES+MoA is the Open Targets
    ChEMBL mechanism table. We download a pre-curated CSV that contains:
      - canonical_smiles
      - moa  (e.g. 'EGFR inhibitor', 'COX inhibitor')
      - target_chembl_id
      - compound_name
    If the remote file is unavailable the preprocessing step will fall
    back to the mock dataset automatically.
    """
    print("\n[3/3] Downloading ChEMBL MoA annotations...")
    url = (
        "https://raw.githubusercontent.com/PatWalters/practical_cheminformatics_tutorials/"
        "main/data/chembl_mechanism.csv"
    )
    download_file(url, "data/raw/chembl_moa_raw.csv", desc="chembl_moa_raw.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-jump",   action="store_true", help="Skip JUMP-CP download")
    parser.add_argument("--skip-chembl", action="store_true", help="Skip ChEMBL download")
    args = parser.parse_args()

    print("=" * 55)
    print("  MorphoCLIP — data download")
    print("=" * 55)

    if not args.skip_jump:
        download_jump_cp()
        download_jump_metadata()
    if not args.skip_chembl:
        download_chembl_moa()

    print("\nAll downloads complete. Run next:")
    print("  python scripts/preprocess.py")


if __name__ == "__main__":
    main()
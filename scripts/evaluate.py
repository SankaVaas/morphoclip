"""
Zero-shot MoA retrieval evaluation on the held-out test set.
Usage: python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --config configs/default.yaml
"""

import argparse
import yaml
import torch
import pandas as pd

from src.data.preprocessing import load_and_clean_jump_cp, load_chembl_moa
from src.data.dataset import get_dataloaders
from src.models.morphoclip import MorphoCLIP
from src.evaluation.metrics import mean_average_precision, recall_at_k
from src.evaluation.retrieval import ZeroShotRetriever


def main(cfg_path: str, ckpt_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    profiles, metadata = load_and_clean_jump_cp(cfg['data']['jump_cp_path'])
    profile_cols = profiles.columns.tolist()
    cfg['morpho_encoder']['input_dim'] = len(profile_cols)
    chembl_df = load_chembl_moa(cfg['data']['chembl_path'])

    _, _, test_loader = get_dataloaders(
        profiles, metadata, chembl_df, profile_cols, cfg
    )

    model = MorphoCLIP(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

    # --- Quantitative metrics ---
    all_mol_emb, all_morpho_emb, all_moa = [], [], []
    with torch.no_grad():
        for mol_batch, morpho_profiles, moa_labels in test_loader:
            mol_batch       = mol_batch.to(device)
            morpho_profiles = morpho_profiles.to(device)
            all_mol_emb.append(model.encode_mol(mol_batch).cpu())
            all_morpho_emb.append(model.encode_morpho(morpho_profiles).cpu())
            all_moa.extend(moa_labels)

    mol_emb    = torch.cat(all_mol_emb)
    morpho_emb = torch.cat(all_morpho_emb)

    k_vals  = cfg['evaluation']['k_values']
    recalls = recall_at_k(mol_emb, morpho_emb, all_moa, k_vals)
    mAP     = mean_average_precision(mol_emb, morpho_emb, all_moa)

    print("\nZero-shot MoA retrieval results:")
    print(f"  mAP: {mAP:.4f}")
    for k, r in zip(k_vals, recalls):
        print(f"  R@{k}: {r:.4f}")

    # --- Example qualitative retrieval ---
    retriever = ZeroShotRetriever(model, device)
    morpho_tensor = torch.tensor(profiles[profile_cols].values, dtype=torch.float32)
    retriever.index_library(morpho_tensor, chembl_df)

    query = "CC(=O)Oc1ccccc1C(=O)O"   # Aspirin
    print(f"\nTop-5 morphologically similar compounds to Aspirin ({query}):")
    results = retriever.query(query, top_k=5)
    print(results[['compound_name', 'moa', 'cosine_similarity']].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
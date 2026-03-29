"""
Entry point for training MorphoCLIP.
Usage: python scripts/train.py --config configs/default.yaml
"""

import argparse
import yaml
import torch

from src.data.preprocessing import load_and_clean_jump_cp, load_chembl_moa
from src.data.dataset import get_dataloaders
from src.models.morphoclip import MorphoCLIP
from src.training.trainer import Trainer


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Data ---
    print("Loading JUMP-CP profiles...")
    profiles, metadata = load_and_clean_jump_cp(cfg['data']['jump_cp_path'])
    profile_cols = profiles.columns.tolist()

    # Update input_dim from actual data
    cfg['morpho_encoder']['input_dim'] = len(profile_cols)

    print("Loading ChEMBL MoA annotations...")
    chembl_df = load_chembl_moa(cfg['data']['chembl_path'])

    print("Building dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        profiles, metadata, chembl_df, profile_cols, cfg
    )
    print(f"  Train: {len(train_loader.dataset)} pairs")
    print(f"  Val:   {len(val_loader.dataset)} pairs")
    print(f"  Test:  {len(test_loader.dataset)} pairs")

    # --- Model ---
    model = MorphoCLIP(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MorphoCLIP parameters: {n_params:,}")

    # --- Training ---
    trainer = Trainer(model, train_loader, val_loader, cfg_path, device)
    trainer.fit(save_dir="checkpoints")

    # --- Test evaluation ---
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print("Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
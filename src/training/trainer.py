"""
Training loop for MorphoCLIP.
"""

import torch
import yaml
import os
from tqdm import tqdm

from .losses import symmetric_infonce, hard_negative_infonce
from ..evaluation.metrics import mean_average_precision, recall_at_k


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg_path: str,
                 device: str = "cpu"):
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model      = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device     = device
        self.cfg_train  = self.cfg['training']

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg_train['lr'],
            weight_decay=self.cfg_train['weight_decay'],
        )

        total_steps = self.cfg_train['epochs'] * len(train_loader)
        warmup_steps = self.cfg_train['warmup_epochs'] * len(train_loader)

        self.scheduler = self._build_scheduler(total_steps, warmup_steps)
        self.negative_mining = self.cfg_train['negative_mining']
        self.best_val_map = 0.0

    def _build_scheduler(self, total_steps, warmup_steps):
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(3.14159 * progress)).item())
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _compute_loss(self, mol_batch, morpho_profiles):
        mol_emb, morpho_emb, logits_mol, logits_morpho = self.model(
            mol_batch, morpho_profiles
        )

        if self.negative_mining == "hard":
            loss = hard_negative_infonce(
                mol_emb, morpho_emb,
                self.model.temperature,
                num_hard=self.cfg_train['num_hard_negatives'],
            )
        else:
            loss = symmetric_infonce(logits_mol, logits_morpho)

        return loss

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for mol_batch, morpho_profiles, _ in tqdm(self.train_loader, leave=False):
            mol_batch = mol_batch.to(self.device)
            morpho_profiles = morpho_profiles.to(self.device)

            self.optimizer.zero_grad()
            loss = self._compute_loss(mol_batch, morpho_profiles)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_mol_emb, all_morpho_emb, all_moa = [], [], []

        for mol_batch, morpho_profiles, moa_labels in loader:
            mol_batch       = mol_batch.to(self.device)
            morpho_profiles = morpho_profiles.to(self.device)

            mol_emb    = self.model.encode_mol(mol_batch)
            morpho_emb = self.model.encode_morpho(morpho_profiles)

            all_mol_emb.append(mol_emb.cpu())
            all_morpho_emb.append(morpho_emb.cpu())
            all_moa.extend(moa_labels)

        mol_emb    = torch.cat(all_mol_emb)
        morpho_emb = torch.cat(all_morpho_emb)

        k_vals = self.cfg['evaluation']['k_values']
        recalls = recall_at_k(mol_emb, morpho_emb, all_moa, k_vals)
        mAP     = mean_average_precision(mol_emb, morpho_emb, all_moa)

        return {'mAP': mAP, **{f'R@{k}': v for k, v in zip(k_vals, recalls)}}

    def fit(self, save_dir: str = "checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        epochs = self.cfg_train['epochs']

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Loss: {train_loss:.4f} | "
                f"Val mAP: {val_metrics['mAP']:.4f} | "
                f"R@1: {val_metrics['R@1']:.4f} | "
                f"Temp: {self.model.temperature.item():.4f}"
            )

            if val_metrics['mAP'] > self.best_val_map:
                self.best_val_map = val_metrics['mAP']
                torch.save(
                    {'epoch': epoch,
                     'model_state': self.model.state_dict(),
                     'val_metrics': val_metrics},
                    os.path.join(save_dir, "best_model.pt")
                )
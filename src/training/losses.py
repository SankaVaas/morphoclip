"""
Contrastive losses for MorphoCLIP.
Includes symmetric InfoNCE and hard-negative mining utilities.
"""

import torch
import torch.nn.functional as F


def symmetric_infonce(logits_mol, logits_morpho):
    """
    Symmetric cross-entropy loss (CLIP-style).
    Diagonal entries are positives; all off-diagonal are negatives.

    Args:
        logits_mol:    (B, B) — mol-to-morpho similarity scaled by temperature
        logits_morpho: (B, B) — morpho-to-mol similarity scaled by temperature
    Returns:
        scalar loss
    """
    B = logits_mol.shape[0]
    labels = torch.arange(B, device=logits_mol.device)

    loss_mol    = F.cross_entropy(logits_mol,    labels)
    loss_morpho = F.cross_entropy(logits_morpho, labels)

    return (loss_mol + loss_morpho) / 2.0


def hard_negative_infonce(mol_emb, morpho_emb, temperature,
                           num_hard: int = 32):
    """
    InfoNCE with semi-hard negative mining.

    For each anchor, we mine the `num_hard` most similar negatives
    in the batch (by cosine similarity) and use only those plus the
    true positive in the denominator. This prevents the model from
    only learning to separate easy negatives.

    Args:
        mol_emb:    (B, D) unit-norm molecular embeddings
        morpho_emb: (B, D) unit-norm morphological embeddings
        temperature: scalar
        num_hard:   how many hard negatives to retain per anchor
    Returns:
        scalar loss
    """
    B = mol_emb.shape[0]
    device = mol_emb.device

    # Full similarity matrix
    sim = torch.matmul(mol_emb, morpho_emb.t())   # (B, B)

    loss = torch.tensor(0.0, device=device)
    for i in range(B):
        pos_sim = sim[i, i]  # positive pair

        # All negatives for anchor i
        neg_sims = torch.cat([sim[i, :i], sim[i, i+1:]])

        # Pick top-num_hard negatives (hardest)
        k = min(num_hard, neg_sims.shape[0])
        hard_neg_sims, _ = neg_sims.topk(k)

        # Numerator: positive; denominator: positive + hard negatives
        all_sims = torch.cat([pos_sim.unsqueeze(0), hard_neg_sims])
        log_prob = all_sims[0] / temperature - torch.logsumexp(all_sims / temperature, dim=0)
        loss -= log_prob

    return loss / B
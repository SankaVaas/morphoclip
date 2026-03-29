"""
Retrieval evaluation metrics for zero-shot MoA retrieval.
"""

import torch
import numpy as np
from typing import List


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Both inputs must be L2-normalised. Returns (N, M) similarity matrix."""
    return torch.matmul(a, b.t())


def recall_at_k(mol_emb: torch.Tensor, morpho_emb: torch.Tensor,
                moa_labels: List[str], k_values: List[int]) -> List[float]:
    """
    For each molecule, retrieve the top-k morphological profiles by cosine sim.
    A retrieval is correct if the MoA label matches.
    Returns recall@k for each k in k_values.
    """
    sim = cosine_similarity_matrix(mol_emb, morpho_emb)
    N = sim.shape[0]
    recalls = []

    for k in k_values:
        hits = 0
        topk_indices = sim.topk(k, dim=1).indices  # (N, k)
        for i in range(N):
            retrieved_moas = {moa_labels[j] for j in topk_indices[i].tolist()}
            if moa_labels[i] in retrieved_moas:
                hits += 1
        recalls.append(hits / N)

    return recalls


def mean_average_precision(mol_emb: torch.Tensor, morpho_emb: torch.Tensor,
                            moa_labels: List[str]) -> float:
    """
    Compute mAP for MoA retrieval.
    Query: molecular embedding. Corpus: morphological profile embeddings.
    Relevance: same MoA label.
    """
    sim = cosine_similarity_matrix(mol_emb, morpho_emb)
    N = sim.shape[0]
    ap_scores = []

    for i in range(N):
        ranked = sim[i].argsort(descending=True).tolist()
        query_moa = moa_labels[i]

        num_relevant = sum(1 for l in moa_labels if l == query_moa) - 1
        if num_relevant == 0:
            continue

        ap, hits = 0.0, 0
        for rank, j in enumerate(ranked):
            if j == i:
                continue  # skip self
            if moa_labels[j] == query_moa:
                hits += 1
                ap += hits / (rank + 1)

        ap_scores.append(ap / num_relevant)

    return float(np.mean(ap_scores)) if ap_scores else 0.0
"""
Morphological profile encoder.
Maps 812-dim CellProfiler/CellProfiler features -> unit-norm embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphoEncoder(nn.Module):
    """
    MLP encoder for Cell Painting CellProfiler feature vectors.
    Uses residual connections and batch norm for training stability on
    small-to-medium datasets.
    """

    def __init__(self, input_dim: int = 812, hidden_dims: list = None,
                 output_dim: int = 256, dropout: float = 0.1,
                 batch_norm: bool = True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 384]

        dims = [input_dim] + hidden_dims

        self.blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            block = ResidualMLPBlock(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                dropout=dropout,
                batch_norm=batch_norm,
            )
            self.blocks.append(block)

        self.projection = nn.Sequential(
            nn.Linear(dims[-1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) morphological feature vectors (pre-normalised)
        Returns:
            (B, output_dim) unit-norm embeddings
        """
        h = x
        for block in self.blocks:
            h = block(h)
        h = self.projection(h)
        return F.normalize(h, dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,
                 dropout: float = 0.1, batch_norm: bool = True):
        super().__init__()

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        self.bn1 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()

        # Projection shortcut when dims differ
        self.shortcut = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        h = F.gelu(self.bn1(self.linear1(x)))
        h = self.dropout(h)
        h = self.bn2(self.linear2(h))
        h = F.gelu(h + residual)
        return h
"""
src/models/dgt.py

Implementation of the Differential Graph Transformer (DGT) adapted from 
Li & Pourbabaee (2025) and Ye et al. (2024).
"""

import math
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadDiffAttn(nn.Module):
    """
    Multi-Head Differential Graph Attention Layer.
    Consumes spatial node embeddings of shape (B, N, d) and head-specific adjacency priors.
    Supports simultaneous multi-relational conditioning:
      - Head 0: Macro Pearson price correlation prior (A_global)
      - Head 1: Institutional Form 13F ownership overlap prior (A_own)
      - Heads 2..H-1: Unconstrained data-driven differential attention
    """

    def __init__(self, embed_dim: int, num_heads: int, lambda_init: float = 0.8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Linear projections for query, key (split into 2 for differential attention) and value
        self.q_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Learnable lambda parameters for noise cancellation
        self.lambda_init = lambda_init
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim))

    def _normalize_adj_prior(
        self,
        adj_prior: Union[torch.Tensor, List[Optional[torch.Tensor]], Dict[int, torch.Tensor], None],
        x: torch.Tensor,
        num_nodes: int,
    ) -> Optional[torch.Tensor]:
        """Constructs or formats a (1, num_heads, N, N) prior modulation tensor."""
        if adj_prior is None:
            return None

        # Case 1: Sequence / list of priors per head, e.g. [A_global, A_own]
        if isinstance(adj_prior, (list, tuple)):
            prior = x.new_ones(1, self.num_heads, num_nodes, num_nodes)
            for h_idx, p in enumerate(adj_prior):
                if h_idx < self.num_heads and p is not None:
                    prior[0, h_idx] = p.to(device=x.device, dtype=x.dtype)
            return prior

        # Case 2: Dictionary mapping head index to prior tensor, e.g. {0: A_global, 1: A_own}
        if isinstance(adj_prior, dict):
            prior = x.new_ones(1, self.num_heads, num_nodes, num_nodes)
            for h_idx, p in adj_prior.items():
                if h_idx < self.num_heads and p is not None:
                    prior[0, h_idx] = p.to(device=x.device, dtype=x.dtype)
            return prior

        # Case 3: PyTorch Tensor
        if isinstance(adj_prior, torch.Tensor):
            p = adj_prior.to(device=x.device, dtype=x.dtype)
            if p.dim() == 2:
                # (N, N) -> broadcast to (1, 1, N, N) across all heads
                return p.unsqueeze(0).unsqueeze(0)
            if p.dim() == 3 and p.shape[0] == self.num_heads:
                # (num_heads, N, N) -> (1, num_heads, N, N)
                return p.unsqueeze(0)
            if p.dim() == 3:
                # (Batch, N, N) -> (Batch, 1, N, N)
                return p.unsqueeze(1)
            return p

        return None

    def forward(
        self,
        x: torch.Tensor,
        adj_prior: Union[torch.Tensor, List[Optional[torch.Tensor]], Dict[int, torch.Tensor], None] = None,
    ) -> torch.Tensor:
        """
        x: (B, N, d)
        adj_prior: Prior tensor(s) modulating spatial attention.
        """
        B, N, d = x.shape

        # Project Q1, Q2, K1, K2, and V
        q = self.q_proj(x).view(B, N, 2 * self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, 2 * self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Transpose for multi-head dot product: (B, num_heads, N, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        q1, q2 = q[:, : self.num_heads], q[:, self.num_heads :]
        k1, k2 = k[:, : self.num_heads], k[:, self.num_heads :]

        scaling = 1.0 / math.sqrt(self.head_dim)
        attn_scores1 = torch.matmul(q1, k1.transpose(-2, -1)) * scaling
        attn_scores2 = torch.matmul(q2, k2.transpose(-2, -1)) * scaling

        attn_map1 = F.softmax(attn_scores1, dim=-1)
        attn_map2 = F.softmax(attn_scores2, dim=-1)

        # Modulate attention maps by topological graph prior(s)
        prior_tensor = self._normalize_adj_prior(adj_prior, x, N)
        if prior_tensor is not None:
            attn_map1 = attn_map1 * prior_tensor

        # Compute dynamic noise-cancellation coefficient lambda
        lambda_val = (
            torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
            - torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
            + self.lambda_init
        )

        # Differential attention weight matrix: (B, num_heads, N, N)
        diff_attn = attn_map1 - lambda_val * attn_map2

        # Weighted sum over values
        out = torch.matmul(diff_attn, v)  # (B, num_heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, d)  # (B, N, d)

        return self.out_proj(out)


class DifferentialGraphTransformer(nn.Module):
    """
    End-to-end DGT:
    1. Input projection: raw price scalar -> d-dimensional embedding.
    2. Causal temporal self-attention over the 64-day lookback window.
    3. Multi-relational spatial differential graph attention conditioned on graph priors.
    4. Prediction head: hidden state at t=64 -> next-day price forecast.
    """

    def __init__(
        self,
        num_stocks: int,
        window_size: int = 64,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        lambda_init: float = 0.8,
    ):
        super().__init__()
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.lambda_init = lambda_init

        # Project scalar normalized price at each day into hidden embedding
        self.input_proj = nn.Linear(1, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, 1, embed_dim))

        # Temporal Transformer Encoder (processes time dimension across lookback)
        temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_encoder_layer, num_layers=2
        )

        # Spatial Differential Graph Attention layer
        self.spatial_diff_attn = MultiheadDiffAttn(
            embed_dim, num_heads=num_heads, lambda_init=lambda_init
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Prediction Head: projects updated stock features to next-day scalar price
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        adj_prior: Union[torch.Tensor, List[Optional[torch.Tensor]], Dict[int, torch.Tensor], None] = None,
        use_spatial: bool = True,
    ) -> torch.Tensor:
        """
        x: (B, T, N) = (Batch, 64, num_stocks)
        adj_prior: Single prior (N, N), or list/dict of priors per head, e.g. {0: A_global, 1: A_own}
        use_spatial: If False, bypasses spatial_diff_attn for the Temporal-Only Control arm.
        Returns:
          y_pred: (B, N)
        """
        B, T, N = x.shape

        # 1. Input embedding projection: (B, T, N, 1) -> (B, T, N, d)
        h = self.input_proj(x.unsqueeze(-1)) + self.pos_embedding

        # 2. Causal temporal attention: reshape to (B * N, T, d) so each stock attends across past 64 days
        h_temp = h.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.embed_dim)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h_temp = self.temporal_encoder(h_temp, mask=causal_mask, is_causal=True)  # (B * N, T, d)

        # Extract the representation at the final day T (t = 64)
        h_final = h_temp[:, -1, :].view(B, N, self.embed_dim)  # (B, N, d)

        # 3. Spatial Differential Graph Attention across all N stocks conditioned on adj_prior
        if use_spatial:
            h_spatial = self.spatial_diff_attn(h_final, adj_prior=adj_prior)
            h_out = self.norm(h_final + h_spatial)  # Residual connection & LayerNorm
        else:
            # Temporal-Only Control: no spatial cross-stock mixing
            h_out = self.norm(h_final)

        # 4. Project to predicted next-day price: (B, N, d) -> (B, N, 1) -> (B, N)
        y_pred = self.head(h_out).squeeze(-1)

        return y_pred
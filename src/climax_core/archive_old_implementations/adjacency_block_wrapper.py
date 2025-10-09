"""
Block wrapper that adds learnable adjacency bias to attention.

Unlike physics mask wrapper (which was complex), this is SIMPLE:
- Take standard transformer block
- Add adjacency bias to attention scores
- That's it!
"""

import torch
import torch.nn as nn


class AdjacencyBiasedBlock(nn.Module):
    """
    Wraps a standard transformer block to add learnable adjacency bias.

    Args:
        block: Original transformer block (timm.Block)
        adjacency_module: LearnableSparseAdjacency module
    """

    def __init__(self, block, adjacency_module):
        super().__init__()
        self._block = block  # Original block
        self.adjacency = adjacency_module

    def forward(self, x):
        """
        x: [B, N, D] - sequence of patch embeddings

        Modifies attention to include adjacency bias.
        """
        B, N, D = x.shape

        # Get adjacency bias [N, N]
        adj_bias = self.adjacency()  # [N, N]

        # Forward through block WITH adjacency bias
        # We need to monkey-patch the attention forward temporarily
        original_attn_forward = self._block.attn.forward

        def attn_forward_with_bias(x):
            """Modified attention that adds adjacency bias."""
            B_attn, N_attn, C_attn = x.shape

            # Standard QKV projection
            qkv = self._block.attn.qkv(x).reshape(B_attn, N_attn, 3, self._block.attn.num_heads, C_attn // self._block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # [B, num_heads, N, head_dim]

            # Attention scores
            attn_scores = (q @ k.transpose(-2, -1)) * self._block.attn.scale  # [B, num_heads, N, N]

            # Add adjacency bias (broadcast over batch and heads)
            attn_scores = attn_scores + adj_bias.unsqueeze(0).unsqueeze(0)  # [B, num_heads, N, N]

            # Softmax
            attn_weights = attn_scores.softmax(dim=-1)
            attn_weights = self._block.attn.attn_drop(attn_weights)

            # Apply attention to values
            x_attn = (attn_weights @ v).transpose(1, 2).reshape(B_attn, N_attn, C_attn)

            # Output projection
            x_attn = self._block.attn.proj(x_attn)
            x_attn = self._block.attn.proj_drop(x_attn)

            return x_attn

        # Temporarily replace attention forward
        self._block.attn.forward = attn_forward_with_bias

        # Forward through block (will use modified attention)
        out = self._block(x)

        # Restore original attention forward
        self._block.attn.forward = original_attn_forward

        return out

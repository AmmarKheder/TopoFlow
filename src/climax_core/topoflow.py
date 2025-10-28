"""
Physics-Guided Attention with Elevation Bias
============================================

Standard masked attention approach for atmospheric modeling:
- Computes elevation bias based on topographic barriers
- Adds bias BEFORE softmax (not multiplication after)
- Uses real-valued bias (can be negative)
- Automatic normalization via softmax

Author: Ammar Kheder
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.
    Matches the structure from timm to ensure checkpoint compatibility.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PhysicsGuidedAttention(nn.Module):
    """
    Multi-head attention with physics-guided elevation bias.

    The elevation bias penalizes uphill transport (harder for pollutants to flow upward)
    while allowing normal attention for downhill transport.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        elevation_scale=1000.0,
    ):
        """
        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in Q, K, V projections
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
            elevation_scale: Characteristic elevation scale for normalization (meters)
        """
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.elevation_scale = elevation_scale

        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Learnable parameter for elevation bias strength
        self.alpha = nn.Parameter(torch.ones(1))

    def compute_elevation_bias(self, elevation_patches):
        """
        Compute elevation-based attention bias.

        Physics principle: Air pollutants flow more easily downhill than uphill.
        - Uphill transport (elev_j > elev_i): negative bias → reduced attention
        - Downhill transport (elev_j < elev_i): no bias → normal attention

        Args:
            elevation_patches: [B, N] elevation values per patch (in meters)

        Returns:
            elevation_bias: [B, N, N] attention bias matrix with values in ℝ
        """
        B, N = elevation_patches.shape

        # Compute pairwise elevation differences
        # elev_diff[i,j] = elevation_j - elevation_i
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1]
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N]
        elev_diff = elev_j - elev_i  # [B, N, N]

        # Normalize by characteristic scale (typically 1000m for regional modeling)
        elev_diff_norm = elev_diff / self.elevation_scale

        # Apply physics-based penalty:
        # - Positive differences (uphill): negative bias via -alpha * ReLU(diff)
        # - Negative differences (downhill): zero bias (ReLU zeros out negatives)
        elevation_bias = -self.alpha * F.relu(elev_diff_norm)

        # Clamp for numerical stability
        # Prevents extreme negative values that could cause softmax saturation
        elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)

        return elevation_bias

    def forward(self, x, elevation_patches):
        """
        Forward pass with physics-guided attention.

        Args:
            x: [B, N, D] patch embeddings
            elevation_patches: [B, N] elevation per patch in meters

        Returns:
            out: [B, N, D] attention output
        """
        B, N, D = x.shape

        # Linear projections and reshape for multi-head attention
        # qkv: [B, N, 3, num_heads, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # STEP 1: Compute raw attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]

        # STEP 2: Compute elevation bias
        elevation_bias = self.compute_elevation_bias(elevation_patches)  # [B, N, N]

        # Expand bias for all attention heads
        elevation_bias = elevation_bias.unsqueeze(1)  # [B, 1, N, N]
        elevation_bias = elevation_bias.expand(-1, self.num_heads, -1, -1)  # [B, num_heads, N, N]

        # STEP 3: ADD bias BEFORE softmax (CRITICAL!)
        # This is the standard masked attention approach
        attn_scores = attn_scores + elevation_bias

        # STEP 4: Apply softmax (automatic normalization)
        # Softmax ensures sum of attention weights = 1, regardless of bias values
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N, N]
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention to values
        out = (attn_weights @ v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class PhysicsGuidedBlock(nn.Module):
    """
    Transformer block with physics-guided attention.

    Standard architecture: LayerNorm → Attention → Residual → LayerNorm → MLP → Residual
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        """
        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            qkv_bias: Whether to use bias in attention projections
            drop: Dropout rate
            attn_drop: Attention dropout rate
            drop_path: Stochastic depth rate
            norm_layer: Normalization layer
        """
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = PhysicsGuidedAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, elevation_patches):
        """
        Args:
            x: [B, N, D] patch embeddings
            elevation_patches: [B, N] elevation per patch

        Returns:
            x: [B, N, D] block output
        """
        # Attention block with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x), elevation_patches))

        # MLP block with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    Used for regularization in deep networks.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor

        return output


def compute_patch_elevations(elevation_field, patch_size=2):
    """
    Compute average elevation for each patch.

    Args:
        elevation_field: [B, H, W] elevation in meters
        patch_size: Size of patches (default: 2x2)

    Returns:
        elevation_patches: [B, N] average elevation per patch
    """
    B, H, W = elevation_field.shape

    # Average pooling to compute mean elevation per patch
    elevation_patches = F.avg_pool2d(
        elevation_field.unsqueeze(1),  # Add channel dimension
        kernel_size=patch_size,
        stride=patch_size
    ).squeeze(1)  # [B, H_patches, W_patches]

    # Flatten spatial dimensions
    elevation_patches = elevation_patches.reshape(B, -1)  # [B, N]

    return elevation_patches


# Example usage
if __name__ == "__main__":
    # Configuration
    batch_size = 2
    num_patches = 8192  # (128/2) × (256/2) for 128×256 image with 2×2 patches
    embed_dim = 768
    num_heads = 8

    # Create sample data
    x = torch.randn(batch_size, num_patches, embed_dim)  # Patch embeddings
    elevation_field = torch.rand(batch_size, 128, 256) * 2000  # 0-2000m elevation

    # Compute patch-level elevations
    elevation_patches = compute_patch_elevations(elevation_field, patch_size=2)

    # Create physics-guided attention block
    block = PhysicsGuidedBlock(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop=0.1,
        attn_drop=0.1,
    )

    # Forward pass
    output = block(x, elevation_patches)

    print(f"Input shape: {x.shape}")
    print(f"Elevation patches shape: {elevation_patches.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Learned alpha parameter: {block.attn.alpha.item():.4f}")

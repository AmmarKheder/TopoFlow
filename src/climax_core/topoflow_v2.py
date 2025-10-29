"""
TopoFlow v2: Relative 2D Position + Alpha Elevation Bias
========================================================

OPTION 3 - Best of both worlds:
1. Relative positional bias for spatial (x, y) coordinates
   - Compatible with wind scanning
   - Learnable MLP
2. Alpha elevation mask for topographic barriers
   - Physics-informed inductive bias
   - Learnable α parameter (YOUR INNOVATION!)

Author: Ammar Kheder
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, DropPath


class RelativePositionBias2D(nn.Module):
    """
    Memory-efficient 2D relative positional bias using bucketing.

    Instead of computing N×N pairwise distances, we:
    1. Discretize relative positions into buckets
    2. Learn bias for each bucket
    3. Index the bias table (O(N²) indexing but NO MLP forward on N²!)

    Based on T5's bucketed relative position encoding.
    Compatible with wind scanning and any patch ordering.
    """

    def __init__(self, num_heads: int, num_buckets: int = 32, max_distance: int = 128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance

        # Learnable bias table: (num_buckets_x * num_buckets_y, num_heads)
        # num_buckets per dimension for 2D
        self.bias_table = nn.Parameter(
            torch.zeros(num_buckets * num_buckets, num_heads)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

    def _relative_position_bucket(self, relative_position, max_distance):
        """
        Bucketize relative positions (from T5).

        Args:
            relative_position: tensor of relative positions
            max_distance: maximum distance to consider

        Returns:
            bucket indices
        """
        num_buckets = self.num_buckets
        ret = 0
        n = -relative_position

        # Half buckets for exact distances, half for log-spaced
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        # Exact distances for small offsets
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # Log-spaced for large distances
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) /
            torch.log(torch.tensor(max_distance / max_exact)) *
            (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(
            val_if_large,
            torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, coords_2d: torch.Tensor):
        """
        Args:
            coords_2d: (B, N, 2) tensor of 2D coordinates (x, y) in [0, 1]

        Returns:
            rel_bias: (B, num_heads, N, N) relative positional bias
        """
        B, N, _ = coords_2d.shape
        device = coords_2d.device

        # Convert normalized coords to grid indices (0 to max_distance)
        coords_int = (coords_2d * self.max_distance).long()  # [B, N, 2]

        # Compute relative positions in grid space
        rel_x = coords_int[:, :, None, 0] - coords_int[:, None, :, 0]  # [B, N, N]
        rel_y = coords_int[:, :, None, 1] - coords_int[:, None, :, 1]  # [B, N, N]

        # Bucketize x and y separately
        bucket_x = self._relative_position_bucket(rel_x, self.max_distance)  # [B, N, N]
        bucket_y = self._relative_position_bucket(rel_y, self.max_distance)  # [B, N, N]

        # Combine buckets: bucket_index = bucket_x * num_buckets + bucket_y
        bucket_idx = bucket_x * self.num_buckets + bucket_y  # [B, N, N]

        # Index bias table: (B, N, N, num_heads)
        rel_bias = self.bias_table[bucket_idx]  # [B, N, N, num_heads]

        # Transpose to (B, num_heads, N, N)
        rel_bias = rel_bias.permute(0, 3, 1, 2).contiguous()

        return rel_bias


class TopoFlowAttentionV2(nn.Module):
    """
    Multi-head attention with:
    1. Relative 2D positional bias (learnable, compatible with wind scanning)
    2. Alpha elevation mask (physics-informed, YOUR INNOVATION)

    Both biases are added BEFORE softmax.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        elevation_scale=1000.0,
        rel_pos_hidden_dim=64,
    ):
        """
        Args:
            dim: Embedding dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in Q, K, V projections
            attn_drop: Dropout rate for attention weights
            proj_drop: Dropout rate for output projection
            elevation_scale: Characteristic elevation scale (meters)
            rel_pos_hidden_dim: Hidden dim for relative position MLP
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

        # 1. Relative 2D positional bias (bucketed, memory-efficient)
        self.rel_pos_bias_2d = RelativePositionBias2D(
            num_heads=num_heads,
            num_buckets=32,  # T5-style bucketing
            max_distance=128  # Grid size (64x128 patches)
        )

        # 2. Alpha elevation mask (YOUR INNOVATION - learnable)
        self.alpha = nn.Parameter(torch.ones(1))

        print(f"✅ TopoFlowAttentionV2 initialized:")
        print(f"   - Relative 2D position bias: {sum(p.numel() for p in self.rel_pos_bias_2d.parameters())} params")
        print(f"   - Alpha elevation: 1 learnable param")

    def compute_elevation_bias(self, elevation_patches):
        """
        Compute physics-based elevation bias.

        Principle: Uphill transport is harder (negative bias)

        Args:
            elevation_patches: [B, N] elevation in meters

        Returns:
            elevation_bias: [B, N, N] bias matrix
        """
        B, N = elevation_patches.shape

        # Pairwise elevation differences
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1]
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N]
        elev_diff = elev_j - elev_i  # [B, N, N]

        # Normalize by elevation scale
        elev_diff_norm = elev_diff / self.elevation_scale

        # Physics: penalize uphill (positive diff)
        elevation_bias = -self.alpha * F.relu(elev_diff_norm)

        # Clamp for numerical stability
        elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)

        return elevation_bias

    def forward(self, x, coords_2d, elevation_patches):
        """
        Forward pass with relative 2D position + alpha elevation.

        Args:
            x: [B, N, D] patch embeddings
            coords_2d: [B, N, 2] spatial coordinates (x, y) normalized to [0, 1]
            elevation_patches: [B, N] elevation per patch (meters)

        Returns:
            out: [B, N, D] attention output
        """
        B, N, D = x.shape

        # Linear projections for Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # STEP 1: Compute raw attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]

        # STEP 2: Add relative 2D positional bias (BUCKETED - memory efficient!)
        rel_pos_bias = self.rel_pos_bias_2d(coords_2d)  # [B, num_heads, N, N]
        attn_scores = attn_scores + rel_pos_bias

        # STEP 3: Add alpha elevation bias (YOUR INNOVATION)
        elevation_bias = self.compute_elevation_bias(elevation_patches)  # [B, N, N]
        elevation_bias = elevation_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        attn_scores = attn_scores + elevation_bias

        # STEP 4: Softmax (automatic normalization)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention to values
        out = (attn_weights @ v).transpose(1, 2).reshape(B, N, D)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class TopoFlowBlockV2(nn.Module):
    """
    Transformer block with TopoFlow v2 attention.

    Combines:
    - Relative 2D positional bias (spatial)
    - Alpha elevation mask (physics)
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
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        elevation_scale=1000.0,
        rel_pos_hidden_dim=64,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # TopoFlow v2 attention
        self.attn = TopoFlowAttentionV2(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            elevation_scale=elevation_scale,
            rel_pos_hidden_dim=rel_pos_hidden_dim,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, coords_2d, elevation_patches):
        """
        Forward pass.

        Args:
            x: [B, N, D] patch embeddings
            coords_2d: [B, N, 2] spatial coords (x, y) in [0, 1]
            elevation_patches: [B, N] elevation (meters)

        Returns:
            x: [B, N, D] output
        """
        # Attention with relative pos + elevation
        x = x + self.drop_path(
            self.attn(self.norm1(x), coords_2d, elevation_patches)
        )

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


def compute_patch_coords_2d(img_size=(128, 256), patch_size=2, device='cuda'):
    """
    Compute 2D spatial coordinates (x, y) for each patch.

    These are FIXED spatial positions, compatible with wind scanning.

    Args:
        img_size: (H, W) image size
        patch_size: size of each patch
        device: torch device

    Returns:
        coords_2d: (1, N, 2) where N = num_patches
                   coords_2d[0, :, 0] = x coordinate in [0, 1]
                   coords_2d[0, :, 1] = y coordinate in [0, 1]
    """
    H, W = img_size
    patches_h = H // patch_size
    patches_w = W // patch_size

    # Create normalized 2D grid
    y_coords = torch.arange(patches_h, device=device, dtype=torch.float32) / patches_h
    x_coords = torch.arange(patches_w, device=device, dtype=torch.float32) / patches_w

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coords_2d = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (N, 2)

    # Add batch dimension
    coords_2d = coords_2d.unsqueeze(0)  # (1, N, 2)

    return coords_2d


def compute_patch_elevations(elevation_field, patch_size):
    """
    Compute average elevation per patch (from topoflow.py).

    Args:
        elevation_field: [B, H, W] elevation in meters
        patch_size: int

    Returns:
        elevation_patches: [B, N] average elevation per patch
    """
    B, H, W = elevation_field.shape

    # Average pool to get patch-level elevations
    elev_field_4d = elevation_field.unsqueeze(1)  # [B, 1, H, W]
    elev_patches = F.avg_pool2d(
        elev_field_4d,
        kernel_size=patch_size,
        stride=patch_size
    )  # [B, 1, patches_h, patches_w]

    # Flatten to [B, N]
    elev_patches = elev_patches.squeeze(1).reshape(B, -1)

    return elev_patches


# =============================================================================
# TEST CODE
# =============================================================================

def test_topoflow_v2():
    """Test TopoFlow v2 implementation."""
    print("=" * 80)
    print("Testing TopoFlow v2: Relative 2D Position + Alpha Elevation")
    print("=" * 80)

    batch_size = 2
    num_patches = 100
    embed_dim = 768
    num_heads = 8

    # Create random inputs
    x = torch.randn(batch_size, num_patches, embed_dim)
    coords_2d = torch.rand(batch_size, num_patches, 2)  # x, y in [0, 1]
    elevation_patches = torch.rand(batch_size, num_patches) * 2000  # 0-2000m

    # Create block
    block = TopoFlowBlockV2(
        dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
        elevation_scale=1000.0,
        rel_pos_hidden_dim=64,
    )

    print(f"\n✓ Input shape: {x.shape}")
    print(f"✓ Coords 2D shape: {coords_2d.shape}")
    print(f"✓ Elevation shape: {elevation_patches.shape}")

    # Forward pass
    output = block(x, coords_2d, elevation_patches)

    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Alpha value: {block.attn.alpha.item():.4f}")

    # Test gradient flow
    loss = output.abs().mean()
    loss.backward()

    print(f"✓ Gradient flow works!")
    print(f"✓ Alpha gradient: {block.attn.alpha.grad.item():.6f}")

    # Test compute_patch_coords_2d
    print("\n" + "=" * 80)
    print("Testing compute_patch_coords_2d")
    print("=" * 80)

    coords = compute_patch_coords_2d(img_size=(128, 256), patch_size=2, device='cpu')
    print(f"✓ Coords shape: {coords.shape}")
    print(f"✓ X range: [{coords[0, :, 0].min():.3f}, {coords[0, :, 0].max():.3f}]")
    print(f"✓ Y range: [{coords[0, :, 1].min():.3f}, {coords[0, :, 1].max():.3f}]")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_topoflow_v2()

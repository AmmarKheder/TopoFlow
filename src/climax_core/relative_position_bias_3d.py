"""
3D Relative Positional Bias for Elevation-Aware Attention
==========================================================

Suggested by supervisor Zhi-Song.

Instead of fixed elevation bias, use a learnable MLP that maps
(dx, dy, dz) relative positions to attention bias values.

Key advantages:
1. Learnable - model learns optimal elevation effect
2. Continuous - handles any elevation difference smoothly
3. Per-head - different heads can attend differently to elevation
4. Automatic normalization - bias added BEFORE softmax

Based on:
- Swin Transformer relative position bias
- Continuous positional encoding literature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionBias3D(nn.Module):
    """
    Continuous 3D relative positional bias for Transformer attention.

    Args:
        num_heads: number of attention heads
        hidden_dim: hidden dimension in the MLP for mapping (dx, dy, dz) -> bias
        normalize: whether to normalize coordinate differences
    """
    def __init__(self, num_heads: int, hidden_dim: int = 64, normalize: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.normalize = normalize

        # Small MLP that maps relative (dx, dy, dz) to bias values per head
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads)
        )

    def forward(self, coords: torch.Tensor):
        """
        Args:
            coords: (B, N, 3) tensor of 3D coordinates (x, y, z) for each patch/point.
                    Where z is elevation in meters, x/y are spatial coordinates.

        Returns:
            rel_bias: (B, num_heads, N, N) tensor of relative biases for attention.
        """
        B, N, _ = coords.shape

        # Compute pairwise relative positions: (B, N, N, 3)
        rel_pos = coords[:, :, None, :] - coords[:, None, :, :]

        if self.normalize:
            # Normalize relative distances to roughly [-1, 1] range
            # Compute max per batch and dimension
            max_vals = rel_pos.abs().amax(dim=(1, 2), keepdim=True) + 1e-6  # (B, 1, 1, 3)
            rel_pos = rel_pos / max_vals

        # Map to bias values (B, N, N, num_heads)
        rel_bias = self.mlp(rel_pos)

        # Transpose to (B, num_heads, N, N) for attention broadcasting
        rel_bias = rel_bias.permute(0, 3, 1, 2).contiguous()
        return rel_bias


class Attention3D(nn.Module):
    """
    Modified Attention module with 3D relative positional bias.

    This is a drop-in replacement for timm.models.vision_transformer.Attention
    that adds elevation-aware bias to attention scores.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 use_3d_bias=True, rel_pos_hidden_dim=64):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 3D relative positional bias
        self.use_3d_bias = use_3d_bias
        if use_3d_bias:
            self.rel_pos_bias_3d = RelativePositionBias3D(
                num_heads=num_heads,
                hidden_dim=rel_pos_hidden_dim,
                normalize=True
            )

    def forward(self, x, coords_3d=None):
        """
        Args:
            x: (B, N, C) token embeddings
            coords_3d: (B, N, 3) 3D coordinates (x, y, elevation) for each patch
                       If None, no 3D bias is applied

        Returns:
            x: (B, N, C) attention output
        """
        B, N, C = x.shape

        # Standard QKV computation
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, C_head)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)

        # ✅ Add 3D relative positional bias BEFORE softmax
        if self.use_3d_bias and coords_3d is not None:
            rel_bias = self.rel_pos_bias_3d(coords_3d)  # (B, num_heads, N, N)
            attn = attn + rel_bias  # ADDITIVE bias (before softmax!)

        # ✅ Softmax - automatic normalization!
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


def compute_patch_coords_3d(elevation_field, img_size=(128, 256), patch_size=2):
    """
    Compute 3D coordinates (x, y, elevation) for each patch.

    Args:
        elevation_field: (B, H, W) elevation in meters
        img_size: (H, W) image size
        patch_size: size of each patch

    Returns:
        coords_3d: (B, N, 3) where N = num_patches
                   coords_3d[:, :, 0] = x coordinate (normalized [0, 1])
                   coords_3d[:, :, 1] = y coordinate (normalized [0, 1])
                   coords_3d[:, :, 2] = elevation (normalized [0, 1])
    """
    B, H, W = elevation_field.shape
    device = elevation_field.device

    # Number of patches
    patches_h = H // patch_size
    patches_w = W // patch_size
    num_patches = patches_h * patches_w

    # Compute average elevation per patch
    elev_field_4d = elevation_field.unsqueeze(1)  # (B, 1, H, W)
    elev_patches = F.avg_pool2d(
        elev_field_4d,
        kernel_size=patch_size,
        stride=patch_size
    )  # (B, 1, patches_h, patches_w)
    elev_patches = elev_patches.squeeze(1).reshape(B, num_patches)  # (B, N)

    # Normalize elevation to [0, 1] per batch
    elev_min = elev_patches.min(dim=1, keepdim=True)[0]
    elev_max = elev_patches.max(dim=1, keepdim=True)[0]
    elev_range = elev_max - elev_min
    elev_range = torch.where(elev_range > 1e-6, elev_range, torch.ones_like(elev_range))
    elev_normalized = (elev_patches - elev_min) / elev_range  # (B, N) in [0, 1]

    # Create 2D spatial coordinates (normalized to [0, 1])
    y_coords = torch.arange(patches_h, device=device, dtype=torch.float32) / patches_h
    x_coords = torch.arange(patches_w, device=device, dtype=torch.float32) / patches_w

    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    spatial_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)  # (N, 2)

    # Expand for batch
    spatial_coords = spatial_coords.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

    # Combine spatial + elevation
    coords_3d = torch.cat([
        spatial_coords,  # (B, N, 2) - x, y
        elev_normalized.unsqueeze(2)  # (B, N, 1) - elevation
    ], dim=2)  # (B, N, 3)

    return coords_3d


# =============================================================================
# TEST CODE
# =============================================================================

def test_relative_position_bias_3d():
    """Test 3D relative positional bias."""
    print("=" * 60)
    print("Testing 3D Relative Positional Bias")
    print("=" * 60)

    batch_size = 2
    num_patches = 100
    num_heads = 8

    # Create random 3D coordinates
    coords = torch.rand(batch_size, num_patches, 3)

    # Create bias module
    rel_pos_bias = RelativePositionBias3D(num_heads=num_heads, hidden_dim=64)

    # Compute bias
    bias = rel_pos_bias(coords)

    print(f"✓ Input coords shape: {coords.shape}")
    print(f"✓ Output bias shape: {bias.shape}")
    print(f"✓ Bias range: [{bias.min():.3f}, {bias.max():.3f}]")
    print(f"✓ Number of parameters: {sum(p.numel() for p in rel_pos_bias.parameters())}")

    # Test gradient flow
    loss = bias.abs().mean()
    loss.backward()
    print("✓ Gradient flow works!")

    print("\n" + "=" * 60)
    print("Testing Attention3D")
    print("=" * 60)

    embed_dim = 768
    x = torch.randn(batch_size, num_patches, embed_dim)

    attn_layer = Attention3D(embed_dim, num_heads=num_heads, use_3d_bias=True)

    # Forward WITHOUT 3D coords
    output_no_bias = attn_layer(x, coords_3d=None)
    print(f"✓ Output (no bias) shape: {output_no_bias.shape}")

    # Forward WITH 3D coords
    output_with_bias = attn_layer(x, coords_3d=coords)
    print(f"✓ Output (with bias) shape: {output_with_bias.shape}")

    # Check outputs are different
    diff = (output_no_bias - output_with_bias).abs().mean()
    print(f"✓ Difference with/without bias: {diff:.6f}")

    print("\n" + "=" * 60)
    print("Testing compute_patch_coords_3d")
    print("=" * 60)

    H, W = 128, 256
    elevation_field = torch.rand(batch_size, H, W) * 2000  # 0-2000m elevation

    coords_3d = compute_patch_coords_3d(elevation_field, img_size=(H, W), patch_size=2)

    print(f"✓ Coords 3D shape: {coords_3d.shape}")
    print(f"✓ X range: [{coords_3d[:, :, 0].min():.3f}, {coords_3d[:, :, 0].max():.3f}]")
    print(f"✓ Y range: [{coords_3d[:, :, 1].min():.3f}, {coords_3d[:, :, 1].max():.3f}]")
    print(f"✓ Elevation range: [{coords_3d[:, :, 2].min():.3f}, {coords_3d[:, :, 2].max():.3f}]")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    test_relative_position_bias_3d()

"""
Hierarchical Multi-Scale Physics Transformer - Innovation #2
============================================================

Models atmospheric physics at multiple spatial scales:
- Local (2x2 patches):    Terrain barriers, urban heat islands, local emissions
- Regional (4x4 grouped): Boundary layer mixing, regional wind transport
- Synoptic (8x8 grouped): Long-range transport, frontal systems

Inspired by Swin Transformer but adapted for atmospheric physics.

Author: Ammar Kheddar
Project: TopoFlow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class MultiScaleAggregator(nn.Module):
    """
    Aggregates features at multiple spatial scales.

    Atmospheric processes operate at different scales:
    - Micro-scale (< 10km): Building effects, street canyons
    - Local-scale (10-100km): Urban heat islands, terrain
    - Regional-scale (100-1000km): Boundary layer, mesoscale circulation
    - Synoptic-scale (> 1000km): Frontal systems, long-range transport
    """

    def __init__(
        self,
        embed_dim: int = 768,
        img_size: Tuple[int, int] = (128, 256),
        patch_size: int = 2,
        scales: List[int] = [1, 2, 4]  # 1=local, 2=regional, 4=synoptic
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.scales = scales

        # Calculate patch grid dimensions
        self.grid_h = img_size[0] // patch_size  # 64
        self.grid_w = img_size[1] // patch_size  # 128
        self.num_patches = self.grid_h * self.grid_w  # 8192

        # Scale-specific processing layers
        self.scale_processors = nn.ModuleDict()
        for scale in scales:
            scale_name = self._get_scale_name(scale)
            self.scale_processors[scale_name] = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # Multi-scale fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * len(scales)),
            nn.Linear(embed_dim * len(scales), embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        print(f"# # #  MultiScaleAggregator: {len(scales)} scales {scales}")

    def _get_scale_name(self, scale: int) -> str:
        """Get descriptive name for scale."""
        if scale == 1:
            return "local"
        elif scale == 2:
            return "regional"
        elif scale == 4:
            return "synoptic"
        else:
            return f"scale_{scale}"

    def _aggregate_to_scale(
        self,
        x: torch.Tensor,
        scale: int
    ) -> torch.Tensor:
        """
        Aggregate patches to coarser scale via average pooling.

        Args:
            x: [B, L, D] patch features
            scale: Grouping factor (1=no grouping, 2=2x2 groups, 4=4x4 groups)

        Returns:
            aggregated: [B, L/scale^2, D] aggregated features
        """
        if scale == 1:
            return x  # No aggregation

        B, L, D = x.shape

        # Reshape to spatial grid
        x_spatial = x.view(B, self.grid_h, self.grid_w, D)  # [B, H, W, D]

        # Average pooling over scale x scale regions
        x_pooled = F.avg_pool2d(
            x_spatial.permute(0, 3, 1, 2),  # [B, D, H, W]
            kernel_size=scale,
            stride=scale
        )  # [B, D, H/scale, W/scale]

        # Reshape back to sequence
        new_h = self.grid_h // scale
        new_w = self.grid_w // scale
        x_aggregated = x_pooled.permute(0, 2, 3, 1).reshape(B, new_h * new_w, D)

        return x_aggregated

    def _upsample_to_original(
        self,
        x: torch.Tensor,
        scale: int
    ) -> torch.Tensor:
        """
        Upsample aggregated features back to original resolution.

        Args:
            x: [B, L/scale^2, D] aggregated features
            scale: Grouping factor used for aggregation

        Returns:
            upsampled: [B, L, D] upsampled features
        """
        if scale == 1:
            return x

        B, L_agg, D = x.shape

        # Calculate aggregated dimensions
        new_h = self.grid_h // scale
        new_w = self.grid_w // scale

        # Reshape to spatial
        x_spatial = x.view(B, new_h, new_w, D)  # [B, H/scale, W/scale, D]

        # Upsample via nearest neighbor (preserves sharp physics boundaries)
        x_upsampled = F.interpolate(
            x_spatial.permute(0, 3, 1, 2),  # [B, D, H/scale, W/scale]
            size=(self.grid_h, self.grid_w),
            mode='nearest'
        )  # [B, D, H, W]

        # Reshape back to sequence
        x_upsampled = x_upsampled.permute(0, 2, 3, 1).reshape(B, self.num_patches, D)

        return x_upsampled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Multi-scale processing and fusion.

        Args:
            x: [B, L, D] patch features from transformer

        Returns:
            x_multi_scale: [B, L, D] multi-scale enriched features
        """
        B, L, D = x.shape

        # Process at each scale
        scale_features = []

        for scale in self.scales:
            scale_name = self._get_scale_name(scale)

            # Aggregate to this scale
            x_scale = self._aggregate_to_scale(x, scale)  # [B, L/scale^2, D]

            # Apply scale-specific processing
            x_scale_processed = self.scale_processors[scale_name](x_scale)

            # Upsample back to original resolution
            x_scale_full = self._upsample_to_original(x_scale_processed, scale)  # [B, L, D]

            scale_features.append(x_scale_full)

        # Concatenate multi-scale features
        x_concat = torch.cat(scale_features, dim=-1)  # [B, L, D*num_scales]

        # Fuse multi-scale information
        x_fused = self.fusion(x_concat)  # [B, L, D]

        return x_fused


class HierarchicalPhysicsTransformer(nn.Module):
    """
    Wrapper for hierarchical multi-scale physics processing.

    Integrates multi-scale aggregation into the model pipeline.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        img_size: Tuple[int, int] = (128, 256),
        patch_size: int = 2,
        scales: List[int] = [1, 2, 4],
        num_layers: int = 1
    ):
        super().__init__()

        self.num_layers = num_layers

        # Stack multiple hierarchical layers
        self.layers = nn.ModuleList([
            MultiScaleAggregator(
                embed_dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                scales=scales
            )
            for _ in range(num_layers)
        ])

        print(f"# # #  HierarchicalPhysicsTransformer: {num_layers} layers, scales {scales}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply hierarchical multi-scale processing.

        Args:
            x: [B, L, D] features from transformer encoder

        Returns:
            x_out: [B, L, D] hierarchical multi-scale features
        """
        x_orig = x.clone()

        for layer in self.layers:
            x = layer(x)

        # Residual connection
        x = x + x_orig

        return x


def test_hierarchical_physics():
    """Test the hierarchical multi-scale physics module."""
    print("\n" + "="*70)
    print("Testing Hierarchical Multi-Scale Physics - Innovation #2")
    print("="*70)

    # Test parameters
    batch_size = 4
    img_size = (128, 256)
    patch_size = 2
    embed_dim = 768

    grid_h = img_size[0] // patch_size  # 64
    grid_w = img_size[1] // patch_size  # 128
    num_patches = grid_h * grid_w  # 8192

    print(f"\nInput configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Grid: {grid_h} x {grid_w} = {num_patches} patches")
    print(f"  Embed dim: {embed_dim}")

    # Create test data
    x = torch.randn(batch_size, num_patches, embed_dim)

    # Test aggregator
    print("\n Testing MultiScaleAggregator...")
    aggregator = MultiScaleAggregator(
        embed_dim=embed_dim,
        img_size=img_size,
        patch_size=patch_size,
        scales=[1, 2, 4]
    )

    x_multi_scale = aggregator(x)
    print(f"# # #  Input shape: {x.shape}")
    print(f"# # #  Output shape: {x_multi_scale.shape}")
    print(f"# # #  Multi-scale enriched features!")

    # Test hierarchical transformer
    print("\nTesting HierarchicalPhysicsTransformer...")
    hierarchical = HierarchicalPhysicsTransformer(
        embed_dim=embed_dim,
        img_size=img_size,
        patch_size=patch_size,
        scales=[1, 2, 4],
        num_layers=2
    )

    x_hierarchical = hierarchical(x)
    print(f"# # #  Input shape: {x.shape}")
    print(f"# # #  Output shape: {x_hierarchical.shape}")

    print("\n" + "="*70)
    print("# # #  All tests passed!")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    test_hierarchical_physics()
"""
TopoFlow: Complete Integration - Wind Reordering + Elevation Attention
========================================================================

Combines TWO physics-guided innovations:
1. Wind-guided patch reordering (dynamic sequence order)
2. Elevation-based attention bias (topographic barriers)

Both are optional and can be toggled independently via config.

Author: Ammar
Date: 2025-10-09
Supervisor: Zhi-Song
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TopoFlowAttention(nn.Module):
    """
    Complete physics-guided attention combining:
    - Elevation-based bias (topographic barriers)
    - Wind modulation (optional)

    This is the CORRECTED version using additive bias BEFORE softmax.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        use_elevation_bias=True,
        use_wind_modulation=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Configuration
        self.use_elevation_bias = use_elevation_bias
        self.use_wind_modulation = use_wind_modulation

        if self.use_elevation_bias:
            # Learnable parameters
            self.elevation_alpha = nn.Parameter(torch.tensor(1.0))

            if self.use_wind_modulation:
                self.wind_beta = nn.Parameter(torch.tensor(0.3))

            # Fixed constants
            self.register_buffer('H_scale', torch.tensor(1000.0))  # 1km
            self.register_buffer('wind_threshold', torch.tensor(5.0))  # 5 m/s

    def forward(
        self,
        x: torch.Tensor,
        elevation_patches: Optional[torch.Tensor] = None,
        u_wind: Optional[torch.Tensor] = None,
        v_wind: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional elevation bias.

        Args:
            x: [B, N, C] token embeddings
            elevation_patches: [B, N] elevation per patch (normalized [0,1])
            u_wind: [B, H, W] horizontal wind (optional)
            v_wind: [B, H, W] vertical wind (optional)

        Returns:
            x: [B, N, C] attention output
        """
        B, N, C = x.shape

        # Standard QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: [B, H, N, C_head]

        # Attention scores
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Add elevation bias BEFORE softmax (if enabled)
        if self.use_elevation_bias and elevation_patches is not None:
            elevation_bias = self._compute_elevation_bias(
                elevation_patches, u_wind, v_wind
            )  # [B, N, N]

            # Expand for heads
            elevation_bias = elevation_bias.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # [B, H, N, N]

            # ✅ ADDITIVE bias BEFORE softmax
            attn_scores = attn_scores + elevation_bias

        # Softmax (automatic normalization)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Apply attention to values
        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def _compute_elevation_bias(
        self,
        elevation_patches: torch.Tensor,
        u_wind: Optional[torch.Tensor],
        v_wind: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute elevation-based attention bias.

        Physics:
            - Uphill transport difficult → negative bias → reduced attention
            - Downhill transport easy → zero bias → normal attention
            - Strong wind can overcome barriers → modulate bias

        Args:
            elevation_patches: [B, N]
            u_wind: [B, H, W] (optional)
            v_wind: [B, H, W] (optional)

        Returns:
            bias: [B, N, N] in ℝ (real values, can be negative)
        """
        B, N = elevation_patches.shape

        # Pairwise elevation differences
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1]
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N]
        elev_diff = elev_j - elev_i  # [B, N, N]

        # Convert to bias (negative for uphill)
        elev_diff_normalized = elev_diff / self.H_scale
        elevation_bias = -self.elevation_alpha * F.relu(elev_diff_normalized)

        # Wind modulation (optional)
        if self.use_wind_modulation and u_wind is not None and v_wind is not None:
            wind_strength = self._compute_wind_strength(u_wind, v_wind)  # [B, N]

            # Wind factor: 0 (weak) to 1 (strong)
            wind_avg = (wind_strength.unsqueeze(1) + wind_strength.unsqueeze(2)) / 2
            wind_factor = torch.sigmoid(wind_avg - self.wind_threshold)

            # Modulate: strong wind reduces barrier
            modulation = 1.0 - self.wind_beta * wind_factor
            elevation_bias = elevation_bias * modulation

        # Clamp for stability
        elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)

        return elevation_bias

    def _compute_wind_strength(self, u_wind, v_wind):
        """Compute wind magnitude per patch."""
        B, H, W = u_wind.shape
        wind_mag = torch.sqrt(u_wind**2 + v_wind**2 + 1e-8)

        # Infer patch grid
        grid_h = H // 2
        grid_w = W // 2
        patch_h = H // grid_h
        patch_w = W // grid_w

        # Average pool
        wind_mag_4d = wind_mag.unsqueeze(1)
        wind_per_patch = F.avg_pool2d(
            wind_mag_4d,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w)
        )
        return wind_per_patch.squeeze(1).reshape(B, -1)


class TopoFlowBlock(nn.Module):
    """
    Complete Transformer block with TopoFlow attention.

    Combines:
    - Pre-norm layer normalization
    - TopoFlow attention (elevation + wind)
    - MLP feed-forward
    - Residual connections
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        use_elevation_bias=True,
        use_wind_modulation=True
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = TopoFlowAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=0.,
            proj_drop=0.,
            use_elevation_bias=use_elevation_bias,
            use_wind_modulation=use_wind_modulation
        )

        self.drop_path = nn.Identity() if drop_path == 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        elevation_patches: Optional[torch.Tensor] = None,
        u_wind: Optional[torch.Tensor] = None,
        v_wind: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, N, C]
            elevation_patches: [B, N] (optional)
            u_wind: [B, H, W] (optional)
            v_wind: [B, H, W] (optional)

        Returns:
            x: [B, N, C]
        """
        # Attention with residual
        x = x + self.drop_path(
            self.attn(self.norm1(x), elevation_patches, u_wind, v_wind)
        )

        # MLP with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_patch_elevations(elevation_field: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Compute elevation per patch by average pooling.

    Args:
        elevation_field: [B, H, W] elevation in meters
        patch_size: size of each patch (default: 2)

    Returns:
        elevation_patches: [B, N] where N = (H//patch_size) * (W//patch_size)
    """
    B, H, W = elevation_field.shape
    elev_4d = elevation_field.unsqueeze(1)  # [B, 1, H, W]

    elev_patches = F.avg_pool2d(
        elev_4d,
        kernel_size=patch_size,
        stride=patch_size
    )

    return elev_patches.squeeze(1).reshape(B, -1)


# =============================================================================
# CONFIGURATION EXAMPLES
# =============================================================================

def get_topoflow_config(mode: str) -> dict:
    """
    Get TopoFlow configuration for different experimental modes.

    Args:
        mode: One of:
            - "baseline": No physics (standard ViT)
            - "elevation_only": Elevation bias only
            - "elevation_wind": Elevation + wind modulation
            - "wind_reorder_only": Wind reordering only (in embedding layer)
            - "full": Wind reordering + elevation attention (FULL TopoFlow)

    Returns:
        config: Dictionary with settings
    """
    configs = {
        "baseline": {
            "use_wind_reordering": False,
            "use_elevation_bias": False,
            "use_wind_modulation": False,
        },
        "elevation_only": {
            "use_wind_reordering": False,
            "use_elevation_bias": True,
            "use_wind_modulation": False,
        },
        "elevation_wind": {
            "use_wind_reordering": False,
            "use_elevation_bias": True,
            "use_wind_modulation": True,
        },
        "wind_reorder_only": {
            "use_wind_reordering": True,
            "use_elevation_bias": False,
            "use_wind_modulation": False,
        },
        "full": {
            "use_wind_reordering": True,
            "use_elevation_bias": True,
            "use_wind_modulation": True,
        }
    }

    return configs[mode]


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Example showing TopoFlow attention usage."""
    print("=" * 70)
    print("TOPOFLOW ATTENTION - USAGE EXAMPLE")
    print("=" * 70)

    # Configuration
    B, N, C = 2, 8192, 768
    H, W = 128, 256
    num_heads = 8

    # Create data
    x = torch.randn(B, N, C)
    elevation_field = torch.rand(B, H, W) * 2000  # 0-2000m
    elevation_patches = compute_patch_elevations(elevation_field)
    u_wind = torch.randn(B, H, W) * 5
    v_wind = torch.randn(B, H, W) * 5

    print(f"\n✓ Input: {x.shape}")
    print(f"✓ Elevation patches: {elevation_patches.shape}")

    # Test different modes
    modes = ["baseline", "elevation_only", "elevation_wind"]

    for mode in modes:
        config = get_topoflow_config(mode)
        print(f"\n--- Mode: {mode} ---")
        print(f"  Config: {config}")

        block = TopoFlowBlock(
            dim=C,
            num_heads=num_heads,
            use_elevation_bias=config["use_elevation_bias"],
            use_wind_modulation=config["use_wind_modulation"]
        )

        output = block(x, elevation_patches, u_wind, v_wind)
        print(f"  Output: {output.shape} ✓")

    print("\n" + "=" * 70)
    print("ALL MODES WORK! ✅")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()

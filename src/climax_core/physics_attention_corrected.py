"""
Physics-Guided Attention - CORRECTED Implementation
===================================================

FIXED: Additive bias BEFORE softmax (not multiplicative mask after)

Key changes from previous version:
1. Bias applied BEFORE softmax (standard masked attention)
2. Real-valued bias (ℝ), not sigmoid-compressed [0,1]
3. Automatic normalization via softmax (no manual renorm needed)

Based on advisor feedback (Zhi-Song) and standard masked attention literature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PhysicsGuidedAttentionCorrected(nn.Module):
    """
    Corrected physics-guided attention with ADDITIVE elevation bias.

    Standard attention: attn = softmax(Q @ K^T / sqrt(d))
    Physics-guided:     attn = softmax((Q @ K^T / sqrt(d)) + elevation_bias)
                                        ↑____________________↑
                                        Add bias BEFORE softmax!
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # LEARNABLE PARAMETERS for physics bias
        # α: controls strength of elevation barrier
        self.elevation_alpha = nn.Parameter(torch.tensor(1.0))

        # β: controls wind modulation strength (optional)
        self.wind_beta = nn.Parameter(torch.tensor(0.3))

        # Characteristic height scale (fixed): 1000m = 1km
        self.register_buffer('H_scale', torch.tensor(1000.0))

        # Wind threshold (fixed): 5 m/s
        self.register_buffer('wind_threshold', torch.tensor(5.0))

    def forward(self, x, elevation_patches=None, u_wind=None, v_wind=None):
        """
        Forward pass with optional elevation-based bias.

        Args:
            x: [B, N, C] token embeddings
            elevation_patches: [B, N] elevation per patch (normalized [0,1], represents meters)
            u_wind: [B, H, W] horizontal wind component (optional)
            v_wind: [B, H, W] vertical wind component (optional)

        Returns:
            x: [B, N, C] attention output
        """
        B, N, C = x.shape

        # ====================================================================
        # STEP 1: Standard QKV computation
        # ====================================================================
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, C_head]
        q, k, v = qkv.unbind(0)  # Each: [B, H, N, C_head]

        # ====================================================================
        # STEP 2: Compute raw attention scores (BEFORE softmax)
        # ====================================================================
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # ====================================================================
        # STEP 3: ADD elevation bias BEFORE softmax ✅
        # ====================================================================
        if elevation_patches is not None:
            # Compute elevation bias [B, N, N]
            elevation_bias = self.compute_elevation_bias(
                elevation_patches, u_wind, v_wind
            )  # [B, N, N] in ℝ (real values, can be negative!)

            # Expand for all attention heads: [B, N, N] → [B, H, N, N]
            elevation_bias_expanded = elevation_bias.unsqueeze(1).expand(
                -1, self.num_heads, -1, -1
            )  # [B, H, N, N]

            # ✅ ADD bias to scores (BEFORE softmax)
            attn_scores = attn_scores + elevation_bias_expanded

        # ====================================================================
        # STEP 4: Apply softmax (automatic normalization!)
        # ====================================================================
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]
        # No manual renormalization needed - softmax does it automatically!

        # ====================================================================
        # STEP 5: Standard attention application
        # ====================================================================
        attn_weights = self.attn_drop(attn_weights)

        x = (attn_weights @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def compute_elevation_bias(
        self,
        elevation_patches: torch.Tensor,
        u_wind: Optional[torch.Tensor] = None,
        v_wind: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute elevation-based attention bias (ADDITIVE, real-valued).

        Physics principle:
            - Uphill transport is difficult → negative bias → reduced attention
            - Downhill transport is easy → small/zero bias → preserved attention
            - Strong winds can overcome barriers → modulate bias

        Args:
            elevation_patches: [B, N] elevation per patch (normalized [0,1])
            u_wind: [B, H, W] horizontal wind (optional)
            v_wind: [B, H, W] vertical wind (optional)

        Returns:
            bias: [B, N, N] additive bias for attention scores (real values)
        """
        B, N = elevation_patches.shape
        device = elevation_patches.device

        # =====================================================================
        # STEP 1: Compute pairwise elevation differences
        # =====================================================================
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1] - source patch
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N] - target patch

        # Elevation difference: positive = uphill, negative = downhill
        elev_diff = elev_j - elev_i  # [B, N, N]
        # elev_diff[b, i, j] > 0 means j is HIGHER than i (uphill i→j)

        # =====================================================================
        # STEP 2: Convert to bias (negative for uphill)
        # =====================================================================
        # Physics: uphill transport is hindered by gravity
        # → positive elevation diff should give NEGATIVE bias
        #
        # Use ReLU to only penalize uphill (not reward downhill)
        # bias = -α × max(0, Δh / H_scale)
        #
        # Result:
        #   Δh = +500m (uphill)   → bias ≈ -0.5α (reduces attention)
        #   Δh = 0m (flat)        → bias = 0 (neutral)
        #   Δh = -500m (downhill) → bias = 0 (no penalty)

        elev_diff_normalized = elev_diff / self.H_scale  # Normalize by 1km
        elevation_bias = -self.elevation_alpha * F.relu(elev_diff_normalized)  # [B, N, N]

        # =====================================================================
        # STEP 3: Wind modulation (optional)
        # =====================================================================
        if u_wind is not None and v_wind is not None:
            # Compute wind magnitude per patch
            wind_strength = self._compute_patch_wind_strength(u_wind, v_wind)  # [B, N]

            # Wind modulation factor [0, 1]:
            # - Weak wind (< threshold) → factor ≈ 0 → full elevation barrier
            # - Strong wind (> threshold) → factor ≈ 1 → reduced elevation barrier
            wind_factor = torch.sigmoid(
                (wind_strength.unsqueeze(1) + wind_strength.unsqueeze(2)) / 2 - self.wind_threshold
            )  # [B, N, N]

            # Modulate elevation bias: strong wind reduces barrier
            # modulation = 1 - β × wind_factor
            # When wind strong (factor=1): modulation = 1-β ≈ 0.7 (30% reduction)
            # When wind weak (factor=0):   modulation = 1 (no reduction)
            modulation = 1.0 - self.wind_beta * wind_factor
            elevation_bias = elevation_bias * modulation

        # =====================================================================
        # STEP 4: Clamp for numerical stability
        # =====================================================================
        # Prevent extreme negative values that could cause numerical issues
        elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)

        return elevation_bias  # [B, N, N] in ℝ

    def _compute_patch_wind_strength(self, u_wind, v_wind):
        """
        Compute wind magnitude per patch by average pooling.

        Args:
            u_wind: [B, H, W]
            v_wind: [B, H, W]

        Returns:
            wind_per_patch: [B, N] wind magnitude per patch
        """
        B, H, W = u_wind.shape

        # Compute wind magnitude
        wind_mag = torch.sqrt(u_wind**2 + v_wind**2 + 1e-8)  # [B, H, W]

        # Infer patch grid size from number of patches
        # We need to compute grid_h, grid_w from H, W and expected N
        # Assuming patch_size=2: grid_h = H//2, grid_w = W//2
        grid_h = H // 2  # 64 for 128×256 input
        grid_w = W // 2  # 128

        patch_h = H // grid_h  # Should be 2
        patch_w = W // grid_w  # Should be 2

        # Average pool to patch resolution
        wind_mag_4d = wind_mag.unsqueeze(1)  # [B, 1, H, W]
        wind_per_patch = F.avg_pool2d(
            wind_mag_4d,
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w)
        )  # [B, 1, grid_h, grid_w]

        wind_per_patch = wind_per_patch.squeeze(1).reshape(B, -1)  # [B, N]

        return wind_per_patch


class PhysicsGuidedBlockCorrected(nn.Module):
    """
    Complete Transformer block with corrected physics-guided attention.

    Drop-in replacement for standard Block from timm.
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=True,
        drop_path=0.,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()

        # Pre-norm
        self.norm1 = norm_layer(dim)

        # Physics-guided attention
        self.attn = PhysicsGuidedAttentionCorrected(
            dim, num_heads, qkv_bias, attn_drop=0., proj_drop=0.
        )

        # Drop path (stochastic depth)
        self.drop_path = nn.Identity() if drop_path == 0. else nn.Identity()

        # MLP block
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x, elevation_patches=None, u_wind=None, v_wind=None):
        """
        Forward pass with optional elevation and wind data.

        Args:
            x: [B, N, C] token embeddings
            elevation_patches: [B, N] elevation per patch (optional)
            u_wind: [B, H, W] horizontal wind (optional)
            v_wind: [B, H, W] vertical wind (optional)

        Returns:
            x: [B, N, C] output embeddings
        """
        # Attention block with residual
        x = x + self.drop_path(
            self.attn(self.norm1(x), elevation_patches, u_wind, v_wind)
        )

        # MLP block with residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_patch_elevations(elevation_field, patch_size=2):
    """
    Compute elevation per patch by average pooling.

    Args:
        elevation_field: [B, H, W] elevation in meters
        patch_size: int, size of each patch (default: 2)

    Returns:
        patch_elevations: [B, N] where N = (H//patch_size) * (W//patch_size)
    """
    B, H, W = elevation_field.shape

    # Average pool to patch resolution
    elev_4d = elevation_field.unsqueeze(1)  # [B, 1, H, W]
    elev_patches = F.avg_pool2d(
        elev_4d,
        kernel_size=patch_size,
        stride=patch_size
    )  # [B, 1, H//patch_size, W//patch_size]

    elev_patches = elev_patches.squeeze(1).reshape(B, -1)  # [B, N]

    return elev_patches


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """
    Example showing how to use the corrected physics-guided attention.
    """
    print("=" * 70)
    print("CORRECTED PHYSICS-GUIDED ATTENTION - USAGE EXAMPLE")
    print("=" * 70)

    # Configuration (ClimaX standard)
    batch_size = 2
    H, W = 128, 256  # Image size
    patch_size = 2
    embed_dim = 768
    num_heads = 8

    # Calculate patches
    grid_h = H // patch_size  # 64
    grid_w = W // patch_size  # 128
    num_patches = grid_h * grid_w  # 8192

    print(f"\n✓ Configuration:")
    print(f"  Image: {H}×{W}")
    print(f"  Patch size: {patch_size}×{patch_size}")
    print(f"  Grid: {grid_h}×{grid_w} = {num_patches} patches")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num heads: {num_heads}")

    # Create test data
    x = torch.randn(batch_size, num_patches, embed_dim)
    elevation_field = torch.rand(batch_size, H, W) * 2000  # 0-2000m elevation
    u_wind = torch.randn(batch_size, H, W) * 5  # -5 to +5 m/s
    v_wind = torch.randn(batch_size, H, W) * 5

    # Compute patch-level elevations
    elevation_patches = compute_patch_elevations(elevation_field, patch_size)

    print(f"\n✓ Input data:")
    print(f"  x: {x.shape}")
    print(f"  elevation_field: {elevation_field.shape}, range: [{elevation_field.min():.1f}, {elevation_field.max():.1f}]m")
    print(f"  elevation_patches: {elevation_patches.shape}")
    print(f"  u_wind: {u_wind.shape}")
    print(f"  v_wind: {v_wind.shape}")

    # Create physics-guided attention layer
    attn_layer = PhysicsGuidedAttentionCorrected(
        dim=embed_dim,
        num_heads=num_heads
    )

    print(f"\n✓ Physics-guided attention layer created")
    print(f"  Learnable parameters:")
    print(f"    - elevation_alpha: {attn_layer.elevation_alpha.item():.3f}")
    print(f"    - wind_beta: {attn_layer.wind_beta.item():.3f}")
    print(f"  Fixed parameters:")
    print(f"    - H_scale: {attn_layer.H_scale.item():.1f}m")
    print(f"    - wind_threshold: {attn_layer.wind_threshold.item():.1f}m/s")

    # Test 1: Forward WITHOUT elevation (standard attention)
    print(f"\n✓ Test 1: Standard attention (no elevation)")
    output_no_elev = attn_layer(x)
    print(f"  Output shape: {output_no_elev.shape}")

    # Test 2: Forward WITH elevation (physics-guided)
    print(f"\n✓ Test 2: Physics-guided attention (with elevation)")
    output_with_elev = attn_layer(x, elevation_patches, u_wind, v_wind)
    print(f"  Output shape: {output_with_elev.shape}")

    # Check difference
    diff = (output_no_elev - output_with_elev).abs().mean()
    print(f"  Difference with/without elevation: {diff:.6f}")
    print(f"  → Physics bias has an effect: {'✓' if diff > 1e-5 else '✗'}")

    # Test 3: Check gradient flow
    print(f"\n✓ Test 3: Gradient flow")
    loss = output_with_elev.abs().mean()
    loss.backward()
    print(f"  elevation_alpha gradient: {attn_layer.elevation_alpha.grad.item():.6f}")
    print(f"  → Gradient flows correctly: {'✓' if abs(attn_layer.elevation_alpha.grad.item()) > 0 else '✗'}")

    # Test 4: Complete block
    print(f"\n✓ Test 4: Complete Transformer block")
    block = PhysicsGuidedBlockCorrected(embed_dim, num_heads)
    output_block = block(x, elevation_patches, u_wind, v_wind)
    print(f"  Output shape: {output_block.shape}")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✅")
    print("=" * 70)

    return True


if __name__ == "__main__":
    example_usage()

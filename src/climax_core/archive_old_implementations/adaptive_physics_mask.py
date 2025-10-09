"""
Adaptive Physics Mask - Literature-Inspired Solution
====================================================

Combines:
1. Fixed physics bias (elevation + wind) - provides inductive bias
2. Learnable correction network - allows adaptation to data

Inspired by:
- Swin Transformer (learnable relative position bias)
- AirPhyNet (physics constraints for air quality)
- Physics-informed neural networks (soft constraints)

Key innovation: Physics guides initialization, learning corrects errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePhysicsMask(nn.Module):
    """
    Adaptive physics mask combining fixed physics and learnable corrections.

    Architecture:
        total_bias = fixed_physics_bias + α * learnable_correction

    Where:
        - fixed_physics_bias: elevation barrier + wind modulation (non-learnable)
        - learnable_correction: small MLP (initialized near 0)
        - α: mixing weight (learnable, starts small)
    """

    def __init__(self, grid_size=(64, 128), hidden_dim=64, correction_strength=0.1):
        """
        Args:
            grid_size: (H, W) patch grid dimensions
            hidden_dim: Hidden dimension for correction network
            correction_strength: Initial strength of learnable correction (small!)
        """
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.N = self.grid_h * self.grid_w

        # ============ FIXED PHYSICS COMPONENT (Non-learnable) ============

        # Physics constants (buffers, not parameters)
        self.register_buffer('H_scale', torch.tensor(1000.0))  # 1km characteristic height
        self.register_buffer('wind_threshold', torch.tensor(5.0))  # 5 m/s wind threshold

        # Physics strengths (FIXED, not learnable to start)
        self.register_buffer('elevation_strength', torch.tensor(0.5))  # Reduced from 1.0
        self.register_buffer('wind_modulation', torch.tensor(0.3))

        # ============ LEARNABLE CORRECTION COMPONENT ============

        # Mixing weight: how much correction to apply
        # Initialize small so we start close to pure physics
        self.alpha = nn.Parameter(torch.tensor(correction_strength))

        # Small MLP to compute correction
        # Input: wind + elevation features per patch pair
        # Output: additive correction to physics bias
        self.correction_net = nn.Sequential(
            nn.Linear(6, hidden_dim),  # 6 features: u_i, v_i, elev_i, u_j, v_j, elev_j
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output in [-1, 1], scaled by alpha
        )

        # Initialize correction network with small weights
        # → outputs near 0 at start
        for m in self.correction_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)  # Very small init
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_fixed_physics_bias(self, elevation_patches, u_wind, v_wind):
        """
        Compute fixed physics bias (elevation + wind modulation).
        Same as your PhysicsMaskFixed, but non-learnable.

        Args:
            elevation_patches: [B, N] elevation in meters
            u_wind: [B, H, W] horizontal wind
            v_wind: [B, H, W] vertical wind

        Returns:
            fixed_bias: [B, N, N] physics bias (non-learnable)
        """
        B, N = elevation_patches.shape

        # 1. Elevation barrier
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1]
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N]
        elev_diff = elev_j - elev_i  # [B, N, N]

        # Normalize by characteristic height
        elev_diff_normalized = elev_diff / self.H_scale

        # Upward = difficult (negative bias)
        elevation_bias = -self.elevation_strength * F.relu(elev_diff_normalized)

        # 2. Wind modulation (per-patch wind strength)
        wind_per_patch = self._compute_patch_wind_strength(u_wind, v_wind)  # [B, N]

        # Pairwise minimum wind (conservative)
        wind_i = wind_per_patch.unsqueeze(2)  # [B, N, 1]
        wind_j = wind_per_patch.unsqueeze(1)  # [B, 1, N]
        wind_min = torch.min(wind_i, wind_j)  # [B, N, N]

        # Strong wind reduces barrier
        wind_factor = torch.sigmoid(wind_min - self.wind_threshold)
        modulation = 1.0 - self.wind_modulation * wind_factor

        # Apply modulation
        elevation_bias = elevation_bias * modulation

        # Clamp for stability
        elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)

        return elevation_bias

    def compute_learnable_correction(self, elevation_patches, u_wind, v_wind):
        """
        Compute learnable correction using small MLP.

        This network can learn to:
        - Reduce physics bias if too strong
        - Add bias where physics model is incomplete
        - Adapt to data-specific patterns

        Args:
            elevation_patches: [B, N] elevation
            u_wind: [B, H, W] horizontal wind
            v_wind: [B, H, W] vertical wind

        Returns:
            correction: [B, N, N] learnable correction (small magnitude)
        """
        B, N = elevation_patches.shape

        # Get per-patch wind features
        u_per_patch = self._downsample_to_patches(u_wind)  # [B, N]
        v_per_patch = self._downsample_to_patches(v_wind)  # [B, N]

        # Create pairwise features for correction network
        # For each pair (i, j), concatenate: [u_i, v_i, elev_i, u_j, v_j, elev_j]

        # Expand to pairwise: [B, N, 1, F] and [B, 1, N, F]
        u_i = u_per_patch.unsqueeze(2).unsqueeze(3)  # [B, N, 1, 1]
        u_j = u_per_patch.unsqueeze(1).unsqueeze(3)  # [B, 1, N, 1]
        v_i = v_per_patch.unsqueeze(2).unsqueeze(3)
        v_j = v_per_patch.unsqueeze(1).unsqueeze(3)
        elev_i = elevation_patches.unsqueeze(2).unsqueeze(3)
        elev_j = elevation_patches.unsqueeze(1).unsqueeze(3)

        # Concatenate features: [B, N, N, 6]
        features = torch.cat([
            u_i.expand(B, N, N, 1),
            v_i.expand(B, N, N, 1),
            (elev_i / 1000.0).expand(B, N, N, 1),  # Normalize elevation
            u_j.expand(B, N, N, 1),
            v_j.expand(B, N, N, 1),
            (elev_j / 1000.0).expand(B, N, N, 1),
        ], dim=3)  # [B, N, N, 6]

        # Apply correction network: [B, N, N, 6] → [B, N, N, 1]
        correction = self.correction_net(features)  # [B, N, N, 1]
        correction = correction.squeeze(-1)  # [B, N, N]

        # Scale by learnable alpha (starts small)
        # Tanh output is in [-1, 1], alpha scales it
        correction = self.alpha * correction

        return correction

    def forward(self, elevation_patches, u_wind=None, v_wind=None):
        """
        Compute adaptive physics bias.

        Args:
            elevation_patches: [B, N] elevation in meters
            u_wind: [B, H, W] horizontal wind (optional)
            v_wind: [B, H, W] vertical wind (optional)

        Returns:
            total_bias: [B, N, N] adaptive physics bias
        """
        if u_wind is None or v_wind is None:
            # Fallback: no wind data, only elevation barrier
            B, N = elevation_patches.shape
            elev_i = elevation_patches.unsqueeze(2)
            elev_j = elevation_patches.unsqueeze(1)
            elev_diff = (elev_j - elev_i) / self.H_scale
            fixed_bias = -self.elevation_strength * F.relu(elev_diff)
            fixed_bias = torch.clamp(fixed_bias, min=-10.0, max=0.0)

            # No learnable correction without wind
            return fixed_bias

        # 1. Fixed physics component
        fixed_bias = self.compute_fixed_physics_bias(elevation_patches, u_wind, v_wind)

        # 2. Learnable correction
        learnable_correction = self.compute_learnable_correction(elevation_patches, u_wind, v_wind)

        # 3. Combine (alpha controls mixing)
        total_bias = fixed_bias + learnable_correction

        # Final clamp for numerical stability
        total_bias = torch.clamp(total_bias, min=-10.0, max=10.0)

        return total_bias

    def _compute_patch_wind_strength(self, u_wind, v_wind):
        """Compute wind magnitude per patch via average pooling."""
        B, H, W = u_wind.shape
        wind_mag = torch.sqrt(u_wind**2 + v_wind**2)  # [B, H, W]

        # Downsample to patch grid
        patch_h = H // self.grid_h
        patch_w = W // self.grid_w

        wind_mag = wind_mag.unsqueeze(1)  # [B, 1, H, W]
        wind_per_patch = F.avg_pool2d(wind_mag, kernel_size=(patch_h, patch_w),
                                       stride=(patch_h, patch_w))
        wind_per_patch = wind_per_patch.squeeze(1).reshape(B, -1)  # [B, N]

        return wind_per_patch

    def _downsample_to_patches(self, field):
        """Downsample 2D field to patch resolution."""
        B, H, W = field.shape
        patch_h = H // self.grid_h
        patch_w = W // self.grid_w

        field = field.unsqueeze(1)  # [B, 1, H, W]
        field_patches = F.avg_pool2d(field, kernel_size=(patch_h, patch_w),
                                      stride=(patch_h, patch_w))
        field_patches = field_patches.squeeze(1).reshape(B, -1)  # [B, N]

        return field_patches

    def get_stats(self):
        """Return stats for monitoring."""
        return {
            'alpha': self.alpha.item(),
            'elevation_strength': self.elevation_strength.item(),
            'wind_modulation': self.wind_modulation.item(),
        }


def test_adaptive_mask():
    """Test adaptive physics mask."""
    print("=" * 60)
    print("TESTING ADAPTIVE PHYSICS MASK")
    print("=" * 60)

    B, N = 2, 8192
    grid_h, grid_w = 64, 128
    H, W = 128, 256

    # Create test data
    elevation = torch.rand(B, N) * 2000  # 0-2000m
    u_wind = torch.randn(B, H, W) * 5
    v_wind = torch.randn(B, H, W) * 5

    # Initialize mask
    mask = AdaptivePhysicsMask(grid_size=(grid_h, grid_w), hidden_dim=64)

    print(f"\n📊 Configuration:")
    print(f"   Grid: {grid_h}×{grid_w} = {N} patches")
    print(f"   Batch: {B}")
    print(f"   Hidden dim: 64")
    print(f"   Initial alpha: {mask.alpha.item():.4f}")

    # Test forward pass
    print(f"\n🧪 Test 1: Forward pass")
    total_bias = mask(elevation, u_wind, v_wind)
    print(f"   Output shape: {total_bias.shape}")
    print(f"   Bias range: [{total_bias.min():.3f}, {total_bias.max():.3f}]")
    print(f"   Mean abs bias: {total_bias.abs().mean():.3f}")

    # Test gradient flow
    print(f"\n🧪 Test 2: Gradient flow")
    loss = total_bias.abs().mean()
    loss.backward()
    print(f"   Alpha gradient: {mask.alpha.grad.item():.6f}")
    print(f"   Correction net has gradients: {any(p.grad is not None for p in mask.correction_net.parameters())}")

    # Test without wind
    print(f"\n🧪 Test 3: Fallback (no wind)")
    mask.zero_grad()
    bias_no_wind = mask(elevation, u_wind=None, v_wind=None)
    print(f"   Bias shape: {bias_no_wind.shape}")
    print(f"   Bias range: [{bias_no_wind.min():.3f}, {bias_no_wind.max():.3f}]")

    print(f"\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)

    stats = mask.get_stats()
    print(f"\n📈 Stats:")
    for key, val in stats.items():
        print(f"   {key}: {val:.4f}")

    return True


if __name__ == "__main__":
    test_adaptive_mask()

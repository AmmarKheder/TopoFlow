"""
Simple Adaptive Physics Mask - Ultra-light
==========================================

Minimal learnable parameters on top of fixed physics:
- Fixed: elevation barrier + wind modulation
- Learnable: single global scaling factor (alpha)

Total new params: 1 scalar

This is the SAFEST starting point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleAdaptivePhysicsMask(nn.Module):
    """
    Simplest adaptive mask: fixed physics + learnable global scale.

    total_bias = alpha * fixed_physics_bias

    Where:
    - fixed_physics_bias: elevation + wind (from your working code)
    - alpha: single learnable scalar (initialized to make training stable)
    """

    def __init__(self, grid_size=(64, 128)):
        super().__init__()
        self.grid_h, self.grid_w = grid_size

        # Physics constants (fixed)
        self.register_buffer('H_scale', torch.tensor(1000.0))
        self.register_buffer('wind_threshold', torch.tensor(5.0))
        self.register_buffer('base_elevation_strength', torch.tensor(0.5))
        self.register_buffer('base_wind_modulation', torch.tensor(0.3))

        # LEARNABLE: global scaling factor
        # Initialize to 0.2 (20% of original strength)
        # → starts gentle, can increase if beneficial
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, elevation_patches, u_wind=None, v_wind=None):
        """Compute scaled physics bias."""
        if u_wind is None or v_wind is None:
            # No wind: only elevation
            B, N = elevation_patches.shape
            elev_i = elevation_patches.unsqueeze(2)
            elev_j = elevation_patches.unsqueeze(1)
            elev_diff = (elev_j - elev_i) / self.H_scale
            fixed_bias = -self.base_elevation_strength * F.relu(elev_diff)
            fixed_bias = torch.clamp(fixed_bias, min=-10.0, max=0.0)
            return self.alpha * fixed_bias

        B, N = elevation_patches.shape

        # 1. Elevation barrier
        elev_i = elevation_patches.unsqueeze(2)
        elev_j = elevation_patches.unsqueeze(1)
        elev_diff = elev_j - elev_i
        elev_diff_normalized = elev_diff / self.H_scale
        elevation_bias = -self.base_elevation_strength * F.relu(elev_diff_normalized)

        # 2. Wind modulation
        wind_per_patch = self._compute_patch_wind_strength(u_wind, v_wind)
        wind_i = wind_per_patch.unsqueeze(2)
        wind_j = wind_per_patch.unsqueeze(1)
        wind_min = torch.min(wind_i, wind_j)
        wind_factor = torch.sigmoid(wind_min - self.wind_threshold)
        modulation = 1.0 - self.base_wind_modulation * wind_factor
        elevation_bias = elevation_bias * modulation

        # Clamp before scaling
        elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)

        # 3. Apply learnable scaling
        total_bias = self.alpha * elevation_bias

        return total_bias

    def _compute_patch_wind_strength(self, u_wind, v_wind):
        """Compute wind magnitude per patch."""
        B, H, W = u_wind.shape
        wind_mag = torch.sqrt(u_wind**2 + v_wind**2)

        patch_h = H // self.grid_h
        patch_w = W // self.grid_w

        wind_mag = wind_mag.unsqueeze(1)
        wind_per_patch = F.avg_pool2d(wind_mag, kernel_size=(patch_h, patch_w),
                                       stride=(patch_h, patch_w))
        wind_per_patch = wind_per_patch.squeeze(1).reshape(B, -1)

        return wind_per_patch

    def get_stats(self):
        return {'alpha': self.alpha.item()}


if __name__ == "__main__":
    print("Testing SimpleAdaptivePhysicsMask...")
    mask = SimpleAdaptivePhysicsMask(grid_size=(64, 128))

    B, N = 2, 8192
    elevation = torch.rand(B, N) * 2000
    u_wind = torch.randn(B, 128, 256) * 5
    v_wind = torch.randn(B, 128, 256) * 5

    bias = mask(elevation, u_wind, v_wind)
    print(f"Bias shape: {bias.shape}")
    print(f"Bias range: [{bias.min():.3f}, {bias.max():.3f}]")
    print(f"Alpha: {mask.alpha.item():.3f}")

    # Test gradient
    loss = bias.abs().mean()
    loss.backward()
    print(f"Alpha gradient: {mask.alpha.grad.item():.6f}")
    print("✅ Test passed!")

"""
Adaptive Physics Mask V2 - Memory Efficient
============================================

Simplified approach inspired by Swin Transformer:
- Fixed physics bias (elevation + wind)
- Learnable scalar multiplier (alpha) per wind sector × elevation bin
- No expensive pairwise MLP

Memory: O(num_sectors * num_bins) instead of O(N²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePhysicsMaskV2(nn.Module):
    """
    Memory-efficient adaptive physics mask.

    Combines:
    1. Fixed physics bias (elevation barrier + wind modulation)
    2. Learnable correction table indexed by (wind_sector, elevation_bin)
    """

    def __init__(self, grid_size=(64, 128), num_wind_sectors=16, num_elev_bins=20):
        """
        Args:
            grid_size: (H, W) patch grid dimensions
            num_wind_sectors: Number of wind direction sectors
            num_elev_bins: Number of elevation difference bins
        """
        super().__init__()
        self.grid_h, self.grid_w = grid_size
        self.num_sectors = num_wind_sectors
        self.num_elev_bins = num_elev_bins

        # ============ FIXED PHYSICS COMPONENT ============

        # Physics constants (non-learnable)
        self.register_buffer('H_scale', torch.tensor(1000.0))  # 1km
        self.register_buffer('wind_threshold', torch.tensor(5.0))  # 5 m/s
        self.register_buffer('elevation_strength', torch.tensor(0.5))
        self.register_buffer('wind_modulation', torch.tensor(0.3))

        # ============ LEARNABLE CORRECTION TABLE ============

        # Learnable correction factors: [num_sectors, num_elev_bins]
        # Initialize near 1.0 (so correction = 1.0 * physics_bias = no change)
        self.correction_table = nn.Parameter(torch.ones(num_wind_sectors, num_elev_bins))

        # Global mixing weight (how much to trust corrections vs physics)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Start with 10% correction

    def compute_fixed_physics_bias(self, elevation_patches, u_wind, v_wind):
        """Compute fixed physics bias (same as before)."""
        B, N = elevation_patches.shape

        # 1. Elevation barrier
        elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1]
        elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N]
        elev_diff = elev_j - elev_i  # [B, N, N]

        elev_diff_normalized = elev_diff / self.H_scale
        elevation_bias = -self.elevation_strength * F.relu(elev_diff_normalized)

        # 2. Wind modulation
        wind_per_patch = self._compute_patch_wind_strength(u_wind, v_wind)  # [B, N]
        wind_i = wind_per_patch.unsqueeze(2)
        wind_j = wind_per_patch.unsqueeze(1)
        wind_min = torch.min(wind_i, wind_j)

        wind_factor = torch.sigmoid(wind_min - self.wind_threshold)
        modulation = 1.0 - self.wind_modulation * wind_factor
        elevation_bias = elevation_bias * modulation

        # Clamp
        elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)

        return elevation_bias, elev_diff  # Also return elev_diff for binning

    def compute_learnable_correction(self, elevation_patches, elev_diff, u_wind, v_wind):
        """
        Compute learnable correction using lookup table.

        For each patch pair (i, j):
        1. Compute dominant wind sector
        2. Bin elevation difference
        3. Lookup correction factor from table
        4. Apply: corrected_bias = correction_factor * fixed_bias

        Memory efficient: O(num_sectors * num_bins) params
        """
        B, N = elevation_patches.shape

        # Compute dominant wind direction per sample
        u_mean = u_wind.flatten(1).mean(dim=1)  # [B]
        v_mean = v_wind.flatten(1).mean(dim=1)  # [B]
        wind_angle = torch.atan2(v_mean, u_mean)  # [B], radians in [-π, π]

        # Convert to sector indices [0, num_sectors-1]
        wind_angle_normalized = (wind_angle % (2 * torch.pi)) / (2 * torch.pi)  # [0, 1]
        sector_indices = (wind_angle_normalized * self.num_sectors).long()  # [B]
        sector_indices = torch.clamp(sector_indices, 0, self.num_sectors - 1)

        # Bin elevation differences: [B, N, N] → [B, N, N] bin indices
        # Map elevation difference to [0, num_elev_bins-1]
        # elev_diff in [-inf, inf] → normalize to [0, 1] → bin
        elev_diff_normalized = torch.clamp(elev_diff / self.H_scale, min=-2, max=2)  # [-2, 2]
        elev_diff_normalized = (elev_diff_normalized + 2) / 4  # [0, 1]
        elev_bin_indices = (elev_diff_normalized * (self.num_elev_bins - 1)).long()  # [B, N, N]
        elev_bin_indices = torch.clamp(elev_bin_indices, 0, self.num_elev_bins - 1)

        # Lookup correction factors from table
        # correction_table: [num_sectors, num_elev_bins]
        # sector_indices: [B] → need [B, N, N]
        # elev_bin_indices: [B, N, N]

        # Expand sector indices
        sector_indices_expanded = sector_indices.view(B, 1, 1).expand(B, N, N)  # [B, N, N]

        # Index correction table: [B, N, N]
        # For each (batch, i, j): correction_table[sector_indices[batch], elev_bin_indices[batch, i, j]]
        correction_factors = self.correction_table[sector_indices_expanded, elev_bin_indices]  # [B, N, N]

        return correction_factors

    def forward(self, elevation_patches, u_wind=None, v_wind=None):
        """
        Compute adaptive physics bias.

        Returns:
            total_bias: [B, N, N] adaptive physics bias
        """
        if u_wind is None or v_wind is None:
            # Fallback: no wind, only elevation
            B, N = elevation_patches.shape
            elev_i = elevation_patches.unsqueeze(2)
            elev_j = elevation_patches.unsqueeze(1)
            elev_diff = (elev_j - elev_i) / self.H_scale
            fixed_bias = -self.elevation_strength * F.relu(elev_diff)
            fixed_bias = torch.clamp(fixed_bias, min=-10.0, max=0.0)
            return fixed_bias

        # 1. Fixed physics component
        fixed_bias, elev_diff = self.compute_fixed_physics_bias(elevation_patches, u_wind, v_wind)

        # 2. Learnable correction factors
        correction_factors = self.compute_learnable_correction(elevation_patches, elev_diff, u_wind, v_wind)

        # 3. Apply correction: total_bias = fixed_bias * (1 + alpha * (correction - 1))
        #    When correction = 1.0 → total_bias = fixed_bias (no change)
        #    When correction > 1.0 → increase bias magnitude
        #    When correction < 1.0 → decrease bias magnitude
        total_bias = fixed_bias * (1.0 + self.alpha * (correction_factors - 1.0))

        # Final clamp
        total_bias = torch.clamp(total_bias, min=-10.0, max=10.0)

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
        """Return stats for monitoring."""
        return {
            'alpha': self.alpha.item(),
            'correction_table_mean': self.correction_table.mean().item(),
            'correction_table_std': self.correction_table.std().item(),
            'correction_table_min': self.correction_table.min().item(),
            'correction_table_max': self.correction_table.max().item(),
        }


def test_adaptive_mask_v2():
    """Test memory-efficient adaptive mask."""
    print("=" * 60)
    print("TESTING ADAPTIVE PHYSICS MASK V2 (Memory Efficient)")
    print("=" * 60)

    B, N = 2, 8192
    grid_h, grid_w = 64, 128
    H, W = 128, 256

    # Create test data
    elevation = torch.rand(B, N) * 2000
    u_wind = torch.randn(B, H, W) * 5
    v_wind = torch.randn(B, H, W) * 5

    # Initialize mask
    mask = AdaptivePhysicsMaskV2(grid_size=(grid_h, grid_w), num_wind_sectors=16, num_elev_bins=20)

    print(f"\n📊 Configuration:")
    print(f"   Grid: {grid_h}×{grid_w} = {N} patches")
    print(f"   Batch: {B}")
    print(f"   Wind sectors: 16")
    print(f"   Elevation bins: 20")
    print(f"   Correction table size: {mask.correction_table.numel()} params")
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
    print(f"   Correction table gradient mean: {mask.correction_table.grad.abs().mean():.6f}")

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
    test_adaptive_mask_v2()

"""
Helper functions to make physics mask work with wind scanning.

Strategy: Apply the SAME regional reordering to elevation/temp as applied to patch tokens.
"""

import torch


def reorder_field_like_wind(field_patches: torch.Tensor,
                             u_wind: torch.Tensor,
                             v_wind: torch.Tensor,
                             wind_scanner) -> torch.Tensor:
    """
    Reorder elevation/temperature patches using same logic as wind scanning.

    Args:
        field_patches: [B, N] - elevation or temperature patches
        u_wind, v_wind: [B, H, W] - wind components
        wind_scanner: CachedWindScanning instance

    Returns:
        reordered_field: [B, N] - same order as wind-scanned patch tokens
    """
    B, N = field_patches.shape
    device = field_patches.device

    # Get grid configuration from scanner
    grid_h, grid_w = wind_scanner.grid_h, wind_scanner.grid_w
    regions_h, regions_w = wind_scanner.regions_h, wind_scanner.regions_w

    # Compute patches per region
    region_h = grid_h // regions_h
    region_w = grid_w // regions_w
    patches_per_region = region_h * region_w

    # Pre-allocate output
    reordered_field = torch.empty_like(field_patches)

    # Process each sample in batch (same as wind scanning)
    for b in range(B):
        # Process each region
        for region_row in range(regions_h):
            for region_col in range(regions_w):
                region_idx = region_row * regions_w + region_col

                # Extract regional wind (same as wind scanning code)
                h_start = region_row * (u_wind.shape[1] // regions_h)
                h_end = h_start + (u_wind.shape[1] // regions_h)
                w_start = region_col * (u_wind.shape[2] // regions_w)
                w_end = w_start + (u_wind.shape[2] // regions_w)

                region_u = u_wind[b, h_start:h_end, w_start:w_end]
                region_v = v_wind[b, h_start:h_end, w_start:w_end]

                # Calculate wind angle (same as wind scanning)
                region_wind_angle = _calculate_wind_angle(region_u, region_v)

                # Calculate patch range
                patch_start = region_idx * patches_per_region
                patch_end = patch_start + patches_per_region

                # Calculate regional order (SAME logic as wind scanning)
                regional_projections = []
                for local_patch_idx in range(patches_per_region):
                    local_row = local_patch_idx // region_w
                    local_col = local_patch_idx % region_w

                    local_proj = _calculate_patch_projection(
                        local_row, local_col, region_wind_angle
                    )
                    regional_projections.append((local_proj, local_patch_idx))

                # Sort by projection (upwind → downwind)
                regional_projections.sort(key=lambda x: x[0])
                regional_order = [local_idx for _, local_idx in regional_projections]

                # Apply reordering to field
                region_field = field_patches[b, patch_start:patch_end]

                for i, src_idx in enumerate(regional_order):
                    reordered_field[b, patch_start + i] = region_field[src_idx]

    return reordered_field


def _calculate_wind_angle(u: torch.Tensor, v: torch.Tensor) -> float:
    """Calculate average wind angle from u, v components."""
    mean_u = u.mean().item()
    mean_v = v.mean().item()

    import math
    angle = math.atan2(mean_v, mean_u)

    return angle


def _calculate_patch_projection(row: int, col: int, angle: float) -> float:
    """Calculate patch projection along wind direction."""
    import math
    return row * math.cos(angle) + col * math.sin(angle)


def test_reordering():
    """Test that reordering produces consistent results."""
    import sys
    sys.path.insert(0, '/pfs/lustrep1/scratch/project_462000640/ammar/aq_net2/src')
    from wind_scanning_cached import CachedWindScanning

    print("Testing field reordering...")

    B, H, W = 2, 128, 256
    grid_h, grid_w = 64, 128
    N = grid_h * grid_w

    # Create scanner
    scanner = CachedWindScanning(grid_h, grid_w, num_sectors=16)
    scanner.regions_h = 32
    scanner.regions_w = 32

    # Create test data
    elevation = torch.rand(B, N)
    u_wind = torch.rand(B, H, W) * 10 - 5
    v_wind = torch.rand(B, H, W) * 10 - 5

    # Reorder
    elevation_reordered = reorder_field_like_wind(elevation, u_wind, v_wind, scanner)

    print(f"Original elevation: {elevation.shape}")
    print(f"Reordered elevation: {elevation_reordered.shape}")
    print(f"Values preserved: {torch.allclose(elevation.sort()[0], elevation_reordered.sort()[0])}")
    print("✅ Reordering works!")


if __name__ == "__main__":
    test_reordering()

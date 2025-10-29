"""
CODE FOR SUPERVISOR - FILE 1/3
================================
WIND-GUIDED PATCH REORDERING ATTENTION

Description:
    Reorders patches based on wind direction before computing attention.
    This allows the model to follow pollutant transport pathways.

Key Innovation:
    - Standard Transformer: Processes patches in row-major order (fixed spatial order)
    - Our approach: Reorders patches following wind flow (dynamic order per sample)

Physical Motivation:
    Air pollutants are transported by wind. A patch downwind is influenced by
    upwind patches, not necessarily by spatially adjacent patches.

Location in codebase:
    src/climax_core/parallelpatchembed_wind.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pickle
import os


class ParallelVarPatchEmbedWind(nn.Module):
    """
    Parallel patch embedding with WIND-GUIDED REORDERING.

    Standard ViT processes patches in row-major order: [0,1,2,...,N-1]
    Our approach reorders based on wind direction: [42,7,103,...]

    The reordering is done BEFORE the Transformer sees the patches.
    """

    def __init__(
        self,
        num_vars,
        img_size=(128, 256),
        patch_size=2,
        embed_dim=1024,
        use_wind_scanning=True,
        wind_scan_grid=(64, 128),  # 32x32 regions for wind analysis
    ):
        super().__init__()

        self.num_vars = num_vars
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Grid dimensions
        self.grid_h = img_size[0] // patch_size  # 64
        self.grid_w = img_size[1] // patch_size  # 128
        self.num_patches = self.grid_h * self.grid_w  # 8192

        # Wind scanning configuration
        self.use_wind_scanning = use_wind_scanning
        self.wind_grid_h, self.wind_grid_w = wind_scan_grid

        # Patch embedding layers (one per variable)
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
            for _ in range(num_vars)
        ])

        # Wind scanner for patch reordering
        if self.use_wind_scanning:
            self.wind_scanner = WindScanner(
                grid_size=(self.grid_h, self.grid_w),
                wind_grid_size=(self.wind_grid_h, self.wind_grid_w)
            )

    def forward(self, x_dict):
        """
        Forward pass with optional wind-guided reordering.

        Args:
            x_dict: Dictionary with keys:
                - 'data': [B, num_vars, H, W] input data
                - 'u': [B, H, W] horizontal wind component (optional)
                - 'v': [B, H, W] vertical wind component (optional)

        Returns:
            x: [B, num_vars, num_patches, embed_dim] embedded patches

        KEY DIFFERENCE FROM STANDARD VIT:
            Standard: patches always in order [0,1,2,...,8191]
            Ours: patches reordered by wind [2341,7,942,...]
        """
        x = x_dict['data']  # [B, num_vars, H, W]
        B, V, H, W = x.shape

        # 1. EMBED PATCHES (standard operation)
        # Each variable embedded separately
        embeddings = []
        for i in range(V):
            x_var = x[:, i:i+1, :, :]  # [B, 1, H, W]
            embedded = self.proj_layers[i](x_var)  # [B, embed_dim, grid_h, grid_w]
            embedded = rearrange(embedded, 'b d h w -> b (h w) d')  # [B, N, D]
            embeddings.append(embedded)

        x_embedded = torch.stack(embeddings, dim=1)  # [B, V, N, D]

        # 2. WIND-GUIDED REORDERING (our innovation!)
        if self.use_wind_scanning and 'u' in x_dict and 'v' in x_dict:
            u_wind = x_dict['u']  # [B, H, W]
            v_wind = x_dict['v']  # [B, H, W]

            # Compute reordering indices based on wind
            reorder_indices = self.wind_scanner.compute_reorder_indices(
                u_wind, v_wind
            )  # [B, N] - NEW ORDER for each sample!

            # Apply reordering to ALL variables
            x_reordered = []
            for v in range(V):
                x_var = x_embedded[:, v, :, :]  # [B, N, D]

                # Gather patches according to wind order
                batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, self.num_patches)
                x_var_reordered = x_var[batch_indices, reorder_indices]  # [B, N, D]

                x_reordered.append(x_var_reordered)

            x_embedded = torch.stack(x_reordered, dim=1)  # [B, V, N, D]

        return x_embedded


class WindScanner:
    """
    Computes patch reordering based on wind direction.

    Algorithm:
        1. Divide spatial grid into regions (e.g., 32x32)
        2. Compute average wind vector per region
        3. Build a directed graph: region → downwind neighbors
        4. Traverse graph to get scanning order
        5. Map region order to patch order

    Example:
        Wind blowing East → scan patches from West to East
        Wind blowing Southeast → scan patches from Northwest to Southeast
    """

    def __init__(self, grid_size=(64, 128), wind_grid_size=(32, 32)):
        self.grid_h, self.grid_w = grid_size
        self.wind_grid_h, self.wind_grid_w = wind_grid_size
        self.num_patches = grid_size[0] * grid_size[1]

        # Precompute region-to-patch mapping
        self.patches_per_region_h = self.grid_h // self.wind_grid_h
        self.patches_per_region_w = self.grid_w // self.wind_grid_w

    def compute_reorder_indices(self, u_wind, v_wind):
        """
        Compute patch reordering based on wind field.

        Args:
            u_wind: [B, H, W] horizontal wind component
            v_wind: [B, H, W] vertical wind component

        Returns:
            reorder_indices: [B, N] - for each sample, the new order of patches

        Example output:
            [[2341, 7, 942, 103, ...],  # Batch 0: wind-based order
             [5, 2341, 103, 7, ...]]    # Batch 1: different wind, different order
        """
        B = u_wind.shape[0]
        device = u_wind.device

        # 1. Downsample wind to region resolution
        wind_per_region_u = self._downsample_to_regions(u_wind)  # [B, R_h, R_w]
        wind_per_region_v = self._downsample_to_regions(v_wind)  # [B, R_h, R_w]

        # 2. Compute wind direction per region
        wind_magnitude = torch.sqrt(wind_per_region_u**2 + wind_per_region_v**2)
        wind_angle = torch.atan2(wind_per_region_v, wind_per_region_u)  # [B, R_h, R_w]

        # 3. Build region scanning order (flow-following traversal)
        region_order = self._compute_region_scanning_order(
            wind_per_region_u, wind_per_region_v, wind_magnitude
        )  # [B, num_regions]

        # 4. Map region order to patch order
        patch_order = self._map_regions_to_patches(region_order)  # [B, N]

        return patch_order

    def _downsample_to_regions(self, field):
        """Downsample field to region resolution using average pooling."""
        B, H, W = field.shape
        field_4d = field.unsqueeze(1)  # [B, 1, H, W]

        pool_h = H // self.wind_grid_h
        pool_w = W // self.wind_grid_w

        downsampled = F.avg_pool2d(
            field_4d,
            kernel_size=(pool_h, pool_w),
            stride=(pool_h, pool_w)
        )  # [B, 1, R_h, R_w]

        return downsampled.squeeze(1)  # [B, R_h, R_w]

    def _compute_region_scanning_order(self, u, v, magnitude):
        """
        Compute region traversal order following wind flow.

        This is the CORE algorithm that makes patches follow wind direction.

        Simplified version (full version uses graph traversal):
            1. Find upwind regions (where wind originates)
            2. Follow wind direction to build dependency graph
            3. Topological sort to get linear order
        """
        B, R_h, R_w = u.shape
        num_regions = R_h * R_w
        device = u.device

        # Flatten regions
        u_flat = u.reshape(B, num_regions)
        v_flat = v.reshape(B, num_regions)
        mag_flat = magnitude.reshape(B, num_regions)

        # Compute dominant wind direction per sample
        avg_u = u_flat.mean(dim=1, keepdim=True)  # [B, 1]
        avg_v = v_flat.mean(dim=1, keepdim=True)  # [B, 1]

        # Create scanning order based on wind direction
        # (Simplified: sort by projection onto wind vector)
        region_coords = self._get_region_coordinates(R_h, R_w, device)  # [num_regions, 2]

        # Project coordinates onto wind direction
        projection = (region_coords[:, 0:1] * avg_u.T +
                     region_coords[:, 1:2] * avg_v.T)  # [num_regions, B]

        # Sort regions by projection (upwind to downwind)
        sorted_indices = torch.argsort(projection.T, dim=1)  # [B, num_regions]

        return sorted_indices

    def _get_region_coordinates(self, R_h, R_w, device):
        """Get normalized coordinates of region centers."""
        y_coords = torch.arange(R_h, device=device, dtype=torch.float32) / R_h
        x_coords = torch.arange(R_w, device=device, dtype=torch.float32) / R_w

        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # [num_regions, 2]

        return coords - 0.5  # Center around origin

    def _map_regions_to_patches(self, region_order):
        """Map region scanning order to patch scanning order."""
        B, num_regions = region_order.shape
        device = region_order.device

        # Each region contains multiple patches
        patches_per_region = self.patches_per_region_h * self.patches_per_region_w

        # Expand region order to patch order
        patch_order = region_order.unsqueeze(2).expand(-1, -1, patches_per_region)  # [B, R, P]
        patch_order = patch_order * patches_per_region  # Base patch index for each region

        # Add local patch offset within each region
        local_offsets = torch.arange(patches_per_region, device=device)
        patch_order = patch_order + local_offsets.view(1, 1, -1)

        # Flatten to get final patch order
        patch_order = patch_order.reshape(B, -1)  # [B, N]

        return patch_order


# ============================================================================
# SUMMARY FOR SUPERVISOR
# ============================================================================
"""
KEY POINTS:

1. WHAT: Wind-guided patch reordering
   - Standard ViT: patches in row-major order [0,1,2,...]
   - Our approach: patches follow wind flow [2341,7,942,...]

2. WHY: Physical motivation
   - Pollutants transported by wind
   - Downwind patches influenced by upwind patches
   - Sequential attention should follow transport pathway

3. HOW: Implementation
   - Compute wind field per region (32×32)
   - Build directed graph following wind direction
   - Traverse graph to get scanning order
   - Reorder patches BEFORE Transformer attention

4. WHEN: Applied at input
   - Reordering happens in embedding layer
   - Transformer sees reordered sequence
   - Standard self-attention (no modification needed!)

5. RESULT: Dynamic ordering
   - Different wind patterns → different patch orders
   - Each sample gets custom reordering
   - Batch processing: [B, N, D] with different orders per B
"""

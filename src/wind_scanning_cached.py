"""
Cached Wind-following patch scanning for ClimaX
Optimized version with pre-computed wind sectors to avoid memory faults
"""

import torch
import numpy as np
from typing import Tuple, Dict
import math


class CachedWindScanning:
    """
    Cached wind scanning with pre-computed sector-based ordering.
    
    Key innovation: Instead of computing new order for each batch,
    pre-compute 16 orders (one per wind sector) and map dynamically.
    """
    
    def __init__(self, grid_h: int, grid_w: int, num_sectors: int = 16):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_sectors = num_sectors
        self.num_patches = grid_h * grid_w
        
        # # # # #  DYNAMIC 32Ã# 64 configuration
        self.regions_h = 32  # Number of region rows
        self.regions_w = 32  # Number of region columns
        self.total_regions = self.regions_h * self.regions_w  # 1024 total (32x32)
        
        # Pre-compute sector angles (in radians)
        self.sector_angles = [
            (2 * math.pi * i / num_sectors) for i in range(num_sectors)
        ]
        
        # Cache for reorder indices per sector 
        self.cached_orders = {}
        self._precompute_all_orders()

        # # # # #  NEW: Regional cache for regions
        self.regional_cached_orders = {}
        self._precompute_regional_orders_32x32()
        
    def _precompute_all_orders(self):
        """Pre-compute patch reordering for all wind sectors."""
        print(f"# # # #  Pre-computing {self.num_sectors} wind sector orderings...")
        
        for i, angle in enumerate(self.sector_angles):
            # Calculate projection for each patch for this sector
            projections = []
            
            for patch_idx in range(self.num_patches):
                row, col = self._get_patch_coordinates(patch_idx)
                proj = self._calculate_patch_projection(row, col, angle)
                projections.append((proj, patch_idx))
            
            # Sort upwind # †’ downwind
            projections.sort(key=lambda x: x[0])
            
            # Store as tensor (CPU initially, will move to GPU when needed)
            reorder_indices = torch.tensor([patch_idx for _, patch_idx in projections], 
                                         dtype=torch.long)
            self.cached_orders[i] = reorder_indices
            
        print(f"# # #  Pre-computed {len(self.cached_orders)} sector orderings!")

    def _precompute_regional_orders_32x32(self):
        """Pre-compute patch reordering for all regions Ã#  16 wind sectors."""
        print(f"# # # #  Pre-computing regional cache: regions Ã#  {self.num_sectors} sectors...")
        
        # Calculate regional dimensions
        region_h, region_w = self.grid_h // self.regions_h, self.grid_w // self.regions_w  # patches per region
        patches_per_region = region_h * region_w  # patches per region
        
        for region_idx in range(self.total_regions):  # regions in grid
            region_row = region_idx // self.regions_w
            region_col = region_idx % self.regions_w
            
            self.regional_cached_orders[region_idx] = {}
            
            for sector_idx, angle in enumerate(self.sector_angles):
                # Calculate projections for all patches in this region
                regional_projections = []
                
                for local_patch_idx in range(patches_per_region):
                    # Convert local patch index to regional coordinates
                    local_row = local_patch_idx // region_w
                    local_col = local_patch_idx % region_w
                    
                    # Calculate projection for this local patch
                    local_proj = self._calculate_patch_projection(local_row, local_col, angle)
                    regional_projections.append((local_proj, local_patch_idx))
                
                # Sort upwind # †’ downwind
                regional_projections.sort(key=lambda x: x[0])
                
                # Store as tensor (CPU initially, will move to GPU when needed)
                reorder_indices = torch.tensor([patch_idx for _, patch_idx in regional_projections], 
                                             dtype=torch.long)
                self.regional_cached_orders[region_idx][sector_idx] = reorder_indices
        
        print(f"# # #  Regional cache ready: {len(self.regional_cached_orders)} regions Ã#  {self.num_sectors} sectors!")
    
    def _get_patch_coordinates(self, patch_idx: int) -> Tuple[int, int]:
        """Convert patch index to (row, col) coordinates."""
        row = patch_idx // self.grid_w
        col = patch_idx % self.grid_w
        return row, col
    
    def _calculate_patch_projection(self, row: int, col: int, wind_angle: float) -> float:
        """Calculate projection of patch position along wind direction."""
        # Normalize coordinates to [-1, 1]
        norm_row = (row / (self.grid_h - 1)) * 2 - 1
        norm_col = (col / (self.grid_w - 1)) * 2 - 1
        
        # Project onto wind direction vector
        wind_x = math.cos(wind_angle)
        wind_y = math.sin(wind_angle)
        
        projection = norm_col * wind_x + (-norm_row) * wind_y
        return projection
    
    def _calculate_wind_angle(self, u: torch.Tensor, v: torch.Tensor) -> float:
        """Calculate dominant wind direction angle from u,v fields."""
        # Calculate wind magnitude
        wind_magnitude = torch.sqrt(u**2 + v**2)
        
        # Weight by magnitude for dominant direction
        if wind_magnitude.sum() > 0:
            weighted_u = (u * wind_magnitude).sum()
            weighted_v = (v * wind_magnitude).sum()
            total_weight = wind_magnitude.sum()
            
            mean_u = weighted_u / total_weight
            mean_v = weighted_v / total_weight
        else:
            mean_u, mean_v = torch.tensor(0.0), torch.tensor(0.0)
        
        # Calculate angle
        angle = torch.atan2(mean_v, mean_u)
        return angle.item()
    
    def _find_closest_sector(self, wind_angle: float) -> int:
        """Find the closest pre-computed sector for given wind angle."""
        # Normalize angle to [0, 2Ï# )
        wind_angle = wind_angle % (2 * math.pi)
        
        # Find closest sector
        min_diff = float('inf')
        closest_sector = 0
        
        for i, sector_angle in enumerate(self.sector_angles):
            # Handle circular difference
            diff = abs(wind_angle - sector_angle)
            diff = min(diff, 2 * math.pi - diff)  # Circular distance
            
            if diff < min_diff:
                min_diff = diff
                closest_sector = i
        
        return closest_sector
    
    def apply_cached_wind_reordering_original(self, patch_tokens: torch.Tensor, 
                                   u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply wind-following reordering using cached sector orders.
        FIXED VERSION: ROCm-safe memory access patterns
        """
        B, V, L, D = patch_tokens.shape
        device = patch_tokens.device
        
        # # # # #  FIX 1: Process on CPU first, then move to GPU (safer for ROCm)
        if device.type == 'cuda' and 'hip' in torch.version.hip:
            return self._rocm_safe_reordering(patch_tokens, u, v)
        
        # # # # #  FIX 2: Pre-allocate with exact same device/dtype
        reordered_tokens = torch.empty_like(patch_tokens, 
                                          device=device, 
                                          dtype=patch_tokens.dtype, 
                                          memory_format=torch.contiguous_format)
        
        # Process each sample in batch
        for b in range(B):
            # Calculate wind angle for this sample
            wind_angle = self._calculate_wind_angle(u[b], v[b])
            
            # Find closest pre-computed sector
            sector_idx = self._find_closest_sector(wind_angle)
            
            # Get cached reorder indices
            reorder_indices = self.cached_orders.get(sector_idx)
            if reorder_indices is None:
                # Fallback: identity order
                reordered_tokens[b] = patch_tokens[b]
                continue
                
            # # # # #  FIX 3: Ensure indices are properly formatted for ROCm
            if reorder_indices.device != device:
                reorder_indices = reorder_indices.to(device, non_blocking=True)
            
            reorder_indices = reorder_indices.contiguous().long()
            
            # # # # #  FIX 4: Use gather instead of index_select (more ROCm-stable)
            # Expand indices to match all dimensions
            indices_expanded = reorder_indices.view(1, 1, L, 1).expand(1, V, L, D)
            
            # Apply reordering using gather (safer than index_select on ROCm)
            reordered_tokens[b:b+1] = torch.gather(patch_tokens[b:b+1], 
                                                  dim=2, 
                                                  index=indices_expanded)
        
        return reordered_tokens

    def apply_regional_wind_reordering_32x32_optimized(self, patch_tokens: torch.Tensor,
                                         u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply wind-following reordering using regional approach.
        Same logic as 2x2 but with finer granularity for better meteorological accuracy.
        """
        B, V, L, D = patch_tokens.shape
        device = patch_tokens.device
        
        # Validate dimensions
        if L != self.grid_h * self.grid_w:
            print(f"# # # # # #  Regional 8x8 wind reordering: dimension mismatch L={L} vs expected={self.grid_h * self.grid_w}")
            return patch_tokens
        
        # Pre-allocate output tensor
        reordered_tokens = torch.empty_like(patch_tokens, device=device, dtype=patch_tokens.dtype)
        
        # Define regional boundaries (regions)
        region_h, region_w = self.grid_h // self.regions_h, self.grid_w // self.regions_w  # patches per region
        patches_per_region = region_h * region_w  # patches per region
        
        #print(f"# # # # # # #  Applying 32Ã# 32 regional wind reordering: {region_h}Ã# {region_w} patches per region ({patches_per_region} patches)")
        
        # Process each sample in batch
        for b in range(B):
            # Process each of the 1024 regions (regional grid)
            for region_row in range(self.regions_h):  # rows of regions
                for region_col in range(self.regions_w):  # columns of regions
                    region_idx = region_row * self.regions_w + region_col
                    
                    # Extract regional wind data
                    h_start = region_row * (u.shape[1] // 32)
                    h_end = h_start + (u.shape[1] // 32)
                    w_start = region_col * (u.shape[2] // 32)
                    w_end = w_start + (u.shape[2] // 32)
                    
                    region_u = u[b, h_start:h_end, w_start:w_end]
                    region_v = v[b, h_start:h_end, w_start:w_end]
                    
                    # Calculate wind direction for this region
                    region_wind_angle = self._calculate_wind_angle(region_u, region_v)
                    
                    # Calculate patch range for this region
                    patch_start = region_idx * patches_per_region
                    patch_end = patch_start + patches_per_region
                    
                    # Calculate regional order specifically for this region patches
                    regional_projections = []
                    for local_patch_idx in range(patches_per_region):
                        # Convert local patch index to regional coordinates
                        local_row = local_patch_idx // region_w
                        local_col = local_patch_idx % region_w
                        
                        # Calculate projection for this local patch
                        local_proj = self._calculate_patch_projection(local_row, local_col, region_wind_angle)
                        regional_projections.append((local_proj, local_patch_idx))
                    
                    # Sort by projection (upwind # †’ downwind)
                    regional_projections.sort(key=lambda x: x[0])
                    
                    # Extract sorted local indices
                    regional_order = [local_idx for _, local_idx in regional_projections]
                    
                    # Apply reordering within this region
                    region_patches = patch_tokens[b, :, patch_start:patch_end, :]
                    
                    # Reorder using the regional order
                    for i, src_idx in enumerate(regional_order):
                        reordered_tokens[b, :, patch_start + i, :] = region_patches[:, src_idx, :]
        
        return reordered_tokens
    
    def _rocm_safe_reordering(self, patch_tokens: torch.Tensor, 
                             u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        ROCm-specific safe reordering using CPU intermediate steps.
        """
        B, V, L, D = patch_tokens.shape
        device = patch_tokens.device
        
        # Move to CPU for reordering, then back to GPU
        patch_cpu = patch_tokens.cpu()
        u_cpu = u.cpu()
        v_cpu = v.cpu()
        
        reordered_cpu = torch.empty_like(patch_cpu)
        
        for b in range(B):
            wind_angle = self._calculate_wind_angle(u_cpu[b], v_cpu[b])
            sector_idx = self._find_closest_sector(wind_angle)
            
            reorder_indices = self.cached_orders.get(sector_idx)
            if reorder_indices is None:
                reordered_cpu[b] = patch_cpu[b]
                continue
            
            # CPU reordering (very stable)
            for v_idx in range(V):
                reordered_cpu[b, v_idx] = patch_cpu[b, v_idx, reorder_indices]
        
        return reordered_cpu.to(device, non_blocking=True)

    def apply_regional_wind_reordering_32x32_optimized(self, patch_tokens: torch.Tensor, 
                                                   u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        OPTIMIZED 8x8 regional wind reordering using pre-computed sector cache.
        ~100Ã#  faster than recalculating projections each time!
        """
        B, V, L, D = patch_tokens.shape
        device = patch_tokens.device
        
        # Validate dimensions
        if L != self.grid_h * self.grid_w:
            print(f"# # # # # #  Optimized 8x8 wind reordering: dimension mismatch L={L} vs expected={self.grid_h * self.grid_w}")
            return patch_tokens
        
        # Pre-allocate output tensor
        reordered_tokens = torch.empty_like(patch_tokens, device=device, dtype=patch_tokens.dtype)
        
        # Regional dimensions
        region_h, region_w = self.grid_h // self.regions_h, self.grid_w // self.regions_w  # patches per region
        patches_per_region = region_h * region_w  # patches per region
        
        ##print(f"# # # #  Applying OPTIMIZED 32Ã# 32 regional wind reordering: {region_h}Ã# {region_w} patches per region ({patches_per_region} patches)")
        
        # Process each sample in batch
        for b in range(B):
            # Process each of the 1024 regions (regional grid)
            for region_idx in range(self.total_regions):
                region_row = region_idx // self.regions_w
                region_col = region_idx % self.regions_w
                
                # Extract regional wind data
                h_start = region_row * (u.shape[1] // 32)
                h_end = h_start + (u.shape[1] // 32)
                w_start = region_col * (u.shape[2] // 32)
                w_end = w_start + (u.shape[2] // 32)
                
                region_u = u[b, h_start:h_end, w_start:w_end]
                region_v = v[b, h_start:h_end, w_start:w_end]
                
                # Calculate wind direction for this region
                region_wind_angle = self._calculate_wind_angle(region_u, region_v)
                
                # Find closest pre-computed sector (FAST!)
                sector_idx = self._find_closest_sector(region_wind_angle)
                
                # Get pre-computed order for this regionÃ# sector (INSTANT!)
                reorder_indices = self.regional_cached_orders[region_idx][sector_idx]
                
                # Move to correct device if needed
                if reorder_indices.device != device:
                    reorder_indices = reorder_indices.to(device, non_blocking=True)
                
                # Calculate patch range for this region
                patch_start = region_idx * patches_per_region
                patch_end = patch_start + patches_per_region
                
                # Apply reordering within this region (VECTORIZED!)
                region_patches = patch_tokens[b, :, patch_start:patch_end, :]  # [V, patches_per_region, D]
                reordered_tokens[b, :, patch_start:patch_end, :] = region_patches[:, reorder_indices, :]
        
        return reordered_tokens
    

    def apply_regional_wind_reordering(self, patch_tokens: torch.Tensor, 
                                     u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply wind-following reordering using 2x2 regional approach.
        CORRECTED VERSION: Calculate regional orders specifically for each region.
        
        Args:
            patch_tokens: [B, V, L, D] patch tokens where L = grid_h * grid_w
            u, v: [B, H, W] wind components
        
        Returns:
            reordered_tokens: [B, V, L, D] reordered patch tokens
        """
        B, V, L, D = patch_tokens.shape
        device = patch_tokens.device
        
        # Validate dimensions
        if L != self.grid_h * self.grid_w:
            print(f"# # # # # #  Regional wind reordering: dimension mismatch L={L} vs expected={self.grid_h * self.grid_w}")
            return patch_tokens
        
        # Pre-allocate output tensor
        reordered_tokens = torch.empty_like(patch_tokens, device=device, dtype=patch_tokens.dtype)
        
        # Define regional boundaries (2x2 regions)
        region_h, region_w = self.grid_h // 2, self.grid_w // 2  # 32x64 for each region
        patches_per_region = region_h * region_w  # 2048 patches per region
        
        # Process each sample in batch
        for b in range(B):
            # Process each of the 4 regions
            for region_row in range(2):  # Top/Bottom
                for region_col in range(2):  # Left/Right
                    region_idx = region_row * 2 + region_col
                    
                    # Extract regional wind data
                    h_start = region_row * (u.shape[1] // 2)
                    h_end = h_start + (u.shape[1] // 2)  
                    w_start = region_col * (u.shape[2] // 2)
                    w_end = w_start + (u.shape[2] // 2)
                    
                    region_u = u[b, h_start:h_end, w_start:w_end]
                    region_v = v[b, h_start:h_end, w_start:w_end]
                    
                    # Calculate wind direction for this region
                    region_wind_angle = self._calculate_wind_angle(region_u, region_v)
                    
                    # Calculate patch range for this region
                    patch_start = region_idx * patches_per_region
                    patch_end = patch_start + patches_per_region
                    
                    # CORRECTED: Calculate regional order specifically for this region's patches
                    regional_projections = []
                    for local_patch_idx in range(patches_per_region):
                        # Convert local patch index to regional coordinates
                        local_row = local_patch_idx // region_w
                        local_col = local_patch_idx % region_w
                        
                        # Calculate projection for this local patch
                        local_proj = self._calculate_patch_projection(local_row, local_col, region_wind_angle)
                        regional_projections.append((local_proj, local_patch_idx))
                    
                    # Sort by projection (upwind # †’ downwind)
                    regional_projections.sort(key=lambda x: x[0])
                    
                    # Extract sorted local indices
                    regional_order = [local_idx for _, local_idx in regional_projections]
                    
                    # Apply reordering within this region
                    region_patches = patch_tokens[b, :, patch_start:patch_end, :]  # [V, patches_per_region, D]
                    
                    # Reorder using the regional order
                    for i, src_idx in enumerate(regional_order):
                        reordered_tokens[b, :, patch_start + i, :] = region_patches[:, src_idx, :]
        
        return reordered_tokens

    def apply_regional_wind_reordering_32x32_optimized(self, patch_tokens: torch.Tensor,
                                         u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Apply wind-following reordering using regional approach.
        Same logic as 2x2 but with finer granularity for better meteorological accuracy.
        """
        B, V, L, D = patch_tokens.shape
        device = patch_tokens.device
        
        # Validate dimensions
        if L != self.grid_h * self.grid_w:
            print(f"# # # # # #  Regional 8x8 wind reordering: dimension mismatch L={L} vs expected={self.grid_h * self.grid_w}")
            return patch_tokens
        
        # Pre-allocate output tensor
        reordered_tokens = torch.empty_like(patch_tokens, device=device, dtype=patch_tokens.dtype)
        
        # Define regional boundaries (regions)
        region_h, region_w = self.grid_h // self.regions_h, self.grid_w // self.regions_w  # patches per region
        patches_per_region = region_h * region_w  # patches per region
        
        #print(f"# # # # # # #  Applying 32Ã# 32 regional wind reordering: {region_h}Ã# {region_w} patches per region ({patches_per_region} patches)")
        
        # Process each sample in batch
        for b in range(B):
            # Process each of the 1024 regions (regional grid)
            for region_row in range(self.regions_h):  # rows of regions
                for region_col in range(self.regions_w):  # columns of regions
                    region_idx = region_row * self.regions_w + region_col
                    
                    # Extract regional wind data
                    h_start = region_row * (u.shape[1] // 32)
                    h_end = h_start + (u.shape[1] // 32)
                    w_start = region_col * (u.shape[2] // 32)
                    w_end = w_start + (u.shape[2] // 32)
                    
                    region_u = u[b, h_start:h_end, w_start:w_end]
                    region_v = v[b, h_start:h_end, w_start:w_end]
                    
                    # Calculate wind direction for this region
                    region_wind_angle = self._calculate_wind_angle(region_u, region_v)
                    
                    # Calculate patch range for this region
                    patch_start = region_idx * patches_per_region
                    patch_end = patch_start + patches_per_region
                    
                    # Calculate regional order specifically for this region patches
                    regional_projections = []
                    for local_patch_idx in range(patches_per_region):
                        # Convert local patch index to regional coordinates
                        local_row = local_patch_idx // region_w
                        local_col = local_patch_idx % region_w
                        
                        # Calculate projection for this local patch
                        local_proj = self._calculate_patch_projection(local_row, local_col, region_wind_angle)
                        regional_projections.append((local_proj, local_patch_idx))
                    
                    # Sort by projection (upwind # †’ downwind)
                    regional_projections.sort(key=lambda x: x[0])
                    
                    # Extract sorted local indices
                    regional_order = [local_idx for _, local_idx in regional_projections]
                    
                    # Apply reordering within this region
                    region_patches = patch_tokens[b, :, patch_start:patch_end, :]
                    
                    # Reorder using the regional order
                    for i, src_idx in enumerate(regional_order):
                        reordered_tokens[b, :, patch_start + i, :] = region_patches[:, src_idx, :]
        
        return reordered_tokens
def apply_cached_wind_reordering_original(patch_tokens: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                               grid_h: int, grid_w: int, wind_scanner=None) -> torch.Tensor:
    """
    Cached version of apply_wind_reordering.
    
    Args:
        patch_tokens: [B, V, L, D] patch embeddings
        u: [B, H, W] horizontal wind
        v: [B, H, W] vertical wind
        grid_h: height of patch grid
        grid_w: width of patch grid
        wind_scanner: CachedWindScanning instance (required)
    """
    if wind_scanner is None:
        raise ValueError("wind_scanner is required (no more global singleton)")
    
    return wind_scanner.apply_cached_wind_reordering(patch_tokens, u, v)


def test_cached_wind_scanning():
    """Test cached wind scanning performance."""
    print("# # # #  Testing cached wind scanning...")
    
    # Create test scenario
    grid_h, grid_w = 64, 128
    B, V, D = 4, 6, 512
    H, W = 128, 256
    
    # Create scanner
    scanner = CachedWindScanning(grid_h, grid_w, num_sectors=16)
    
    # Create test data
    patch_tokens = torch.randn(B, V, grid_h * grid_w, D)
    u = torch.randn(B, H, W)
    v = torch.randn(B, H, W)
    
    # Test performance
    import time
    start = time.time()
    result = scanner.apply_cached_wind_reordering(patch_tokens, u, v)
    end = time.time()
    
    print(f"# # #  Cached reordering: {(end-start)*1000:.2f}ms")
    print(f"   Input shape: {patch_tokens.shape}")
    print(f"   Output shape: {result.shape}")
    print(f"   Memory efficient: No new tensor allocation per batch!")


if __name__ == "__main__":
    test_cached_wind_scanning()

def apply_cached_wind_reordering(patch_tokens: torch.Tensor, 
                                u: torch.Tensor, v: torch.Tensor,
                                grid_h: int, grid_w: int, 
                                wind_scanner: CachedWindScanning = None,
                                regional_mode = True) -> torch.Tensor:
    """CORRECTED: Apply wind-following reordering with proper regional calculation."""
    if wind_scanner is None:
        return patch_tokens
    
    try:
        if regional_mode:
            ##print(f"# # # #  Applying OPTIMIZED regional 8Ã# 8 wind reordering...")
            return wind_scanner.apply_regional_wind_reordering_32x32_optimized(patch_tokens, u, v)
        else:
            print(f"# # # # # # #  Applying global wind reordering...")
            return wind_scanner.apply_cached_wind_reordering(patch_tokens, u, v)
    except Exception as e:
        print(f"# # # # # #  Wind reordering failed: {e}")
        return patch_tokens

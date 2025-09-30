#!/usr/bin/env python3
"""
Wind Band-Hilbert Scanning Orders
=================================

Physics-aware scanning that follows atmospheric wind patterns combined with 
space-filling curves for optimal locality.

Key Concepts:
- Tiles: Divide grid into 16x16 tiles
- Wind Analysis: Compute dominant wind direction per tile
- Band Sorting: 3 bands per tile (upstream # †’ downstream)
- Hilbert Internal: Space-filling curve within each band
- Fallback: Pure Hilbert when wind is calm (< 0.5 m/s)
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
import math

# Import existing Hilbert implementation
try:
    from .scan_orders import generalize_hilbert_curve_scan_order
except ImportError:
    # Fallback if scan_orders not available
    def generalize_hilbert_curve_scan_order(w, h):
        return list(range(w * h))


def compute_wind_stats(u_field: np.ndarray, v_field: np.ndarray, 
                      tile_i: int, tile_j: int, tile_size: int = 16) -> Tuple[float, float, float]:
    """
    Compute wind statistics for a specific tile.
    
    Args:
        u_field: U-component wind field [H, W]
        v_field: V-component wind field [H, W]  
        tile_i, tile_j: Tile indices
        tile_size: Size of each tile (default 16x16)
        
    Returns:
        (mean_u, mean_v, wind_magnitude)
    """
    H, W = u_field.shape
    
    # Tile boundaries
    start_i = tile_i * tile_size
    end_i = min(start_i + tile_size, H)
    start_j = tile_j * tile_size  
    end_j = min(start_j + tile_size, W)
    
    # Extract tile data
    u_tile = u_field[start_i:end_i, start_j:end_j]
    v_tile = v_field[start_i:end_i, start_j:end_j]
    
    # Compute mean wind components
    mean_u = np.mean(u_tile)
    mean_v = np.mean(v_tile)
    wind_mag = np.sqrt(mean_u**2 + mean_v**2)
    
    return mean_u, mean_v, wind_mag


def compute_upstream_downstream_projection(x: int, y: int, mean_u: float, mean_v: float, 
                                         tile_size: int = 16) -> float:
    """
    Compute upstream# †’downstream projection for a point within a tile.
    
    Args:
        x, y: Point coordinates within tile [0, tile_size)
        mean_u, mean_v: Mean wind components for the tile
        tile_size: Size of tile
        
    Returns:
        Projection value (higher = more downstream)
    """
    # Normalize coordinates to [-0.5, 0.5]
    norm_x = (x - tile_size/2) / tile_size
    norm_y = (y - tile_size/2) / tile_size
    
    # Project onto wind direction
    wind_mag = np.sqrt(mean_u**2 + mean_v**2)
    if wind_mag < 1e-6:
        return 0.0
        
    # Unit wind vector
    wind_u_norm = mean_u / wind_mag
    wind_v_norm = mean_v / wind_mag
    
    # Dot product gives projection
    projection = norm_x * wind_u_norm + norm_y * wind_v_norm
    return projection


def create_wind_bands(tile_indices: List[int], mean_u: float, mean_v: float, 
                     tile_size: int = 16, n_bands: int = 3) -> List[List[int]]:
    """
    Sort tile patches into wind-based bands (upstream # †’ downstream).
    
    Args:
        tile_indices: List of patch indices within this tile
        mean_u, mean_v: Mean wind components  
        tile_size: Size of tile
        n_bands: Number of bands to create
        
    Returns:
        List of bands, each containing sorted patch indices
    """
    if not tile_indices:
        return []
        
    # Compute projections for all patches in tile
    projections = []
    for idx in tile_indices:
        # Convert flat index to (row, col) within tile
        tile_row = idx // tile_size
        tile_col = idx % tile_size
        
        proj = compute_upstream_downstream_projection(tile_col, tile_row, mean_u, mean_v, tile_size)
        projections.append((idx, proj))
    
    # Sort by projection (upstream # †’ downstream)
    projections.sort(key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in projections]
    
    # Divide into bands
    band_size = len(sorted_indices) // n_bands
    bands = []
    
    for b in range(n_bands):
        start = b * band_size
        if b == n_bands - 1:  # Last band gets remaining indices
            end = len(sorted_indices)
        else:
            end = (b + 1) * band_size
            
        band_indices = sorted_indices[start:end]
        bands.append(band_indices)
    
    return bands


def hilbert_within_band(band_indices: List[int], tile_size: int = 16) -> List[int]:
    """
    Apply Hilbert curve ordering within a band.
    
    Args:
        band_indices: Patch indices within this band
        tile_size: Size of tile
        
    Returns:
        Hilbert-ordered indices
    """
    if len(band_indices) <= 1:
        return band_indices
        
    # Convert to coordinates
    coords = []
    for idx in band_indices:
        row = idx // tile_size
        col = idx % tile_size
        coords.append((row, col, idx))
    
    # Simple distance-based ordering as Hilbert approximation
    # Start from first point and build nearest-neighbor chain
    ordered = [coords[0]]
    remaining = coords[1:]
    
    while remaining:
        last_row, last_col, _ = ordered[-1]
        
        # Find nearest remaining point
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, (row, col, idx) in enumerate(remaining):
            dist = (row - last_row)**2 + (col - last_col)**2
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
                
        ordered.append(remaining.pop(nearest_idx))
    
    return [idx for _, _, idx in ordered]


def wind_band_hilbert_scan_order(u_field: np.ndarray, v_field: np.ndarray,
                                grid_shape: Tuple[int, int] = (128, 256),
                                patch_size: int = 2,
                                tile_size: int = 16,
                                wind_threshold: float = 0.5,
                                n_bands: int = 3) -> List[int]:
    """
    Main Wind Band-Hilbert scanning algorithm.
    
    Algorithm:
    1. Divide grid into tiles
    2. For each tile:
       - Compute wind statistics
       - If wind > threshold: create wind-based bands + Hilbert within bands
       - If wind # # ¤ threshold: pure Hilbert on tile
    3. Order tiles by global wind direction
    4. Concatenate tile orders
    
    Args:
        u_field: U-component wind field [H, W]
        v_field: V-component wind field [H, W]
        grid_shape: (Height, Width) of grid
        patch_size: Size of patches for patchification
        tile_size: Size of tiles  
        wind_threshold: Minimum wind speed for wind-based ordering (m/s)
        n_bands: Number of bands per tile
        
    Returns:
        List of patch indices in wind-optimized order
    """
    H, W = grid_shape
    
    # Calculate patch grid dimensions
    patch_H = H // patch_size
    patch_W = W // patch_size
    total_patches = patch_H * patch_W
    
    # Calculate tile grid dimensions  
    tile_rows = (patch_H + tile_size - 1) // tile_size
    tile_cols = (patch_W + tile_size - 1) // tile_size
    
    print(f"# # # # # # #   Wind Band-Hilbert: {tile_rows}Ã# {tile_cols} tiles, wind_threshold={wind_threshold}")
    
    # Compute global wind for tile ordering
    global_u = np.mean(u_field)
    global_v = np.mean(v_field)
    global_wind_mag = np.sqrt(global_u**2 + global_v**2)
    
    # Create tile processing order (upstream # †’ downstream globally)
    tile_order = []
    for ti in range(tile_rows):
        for tj in range(tile_cols):
            # Project tile center onto global wind direction
            tile_center_i = (ti + 0.5) * tile_size  
            tile_center_j = (tj + 0.5) * tile_size
            
            if global_wind_mag > 1e-6:
                projection = (tile_center_j * global_u + tile_center_i * global_v) / global_wind_mag
            else:
                projection = ti * tile_cols + tj  # fallback to row-major
                
            tile_order.append((ti, tj, projection))
    
    # Sort tiles by global projection (upstream # †’ downstream)
    tile_order.sort(key=lambda x: x[2])
    
    # Process each tile and build final scan order
    final_scan_order = []
    
    for ti, tj, _ in tile_order:
        # Get patch indices for this tile
        tile_patch_indices = []
        
        for local_i in range(tile_size):
            for local_j in range(tile_size):
                global_patch_i = ti * tile_size + local_i
                global_patch_j = tj * tile_size + local_j
                
                if global_patch_i < patch_H and global_patch_j < patch_W:
                    patch_idx = global_patch_i * patch_W + global_patch_j
                    tile_patch_indices.append(local_i * tile_size + local_j)
        
        if not tile_patch_indices:
            continue
            
        # Compute local wind statistics for this tile
        wind_start_i = ti * tile_size * patch_size
        wind_end_i = min(wind_start_i + tile_size * patch_size, H)
        wind_start_j = tj * tile_size * patch_size  
        wind_end_j = min(wind_start_j + tile_size * patch_size, W)
        
        u_tile_field = u_field[wind_start_i:wind_end_i, wind_start_j:wind_end_j]
        v_tile_field = v_field[wind_start_i:wind_end_i, wind_start_j:wind_end_j]
        
        mean_u = np.mean(u_tile_field)
        mean_v = np.mean(v_tile_field)
        wind_mag = np.sqrt(mean_u**2 + mean_v**2)
        
        # Apply wind-based or Hilbert ordering
        if wind_mag > wind_threshold:
            # Wind-based band ordering
            bands = create_wind_bands(tile_patch_indices, mean_u, mean_v, tile_size, n_bands)
            tile_ordered = []
            
            for band in bands:
                hilbert_band = hilbert_within_band(band, tile_size)
                tile_ordered.extend(hilbert_band)
        else:
            # Pure Hilbert fallback
            tile_ordered = generalize_hilbert_curve_scan_order(
                min(tile_size, patch_W - tj * tile_size),
                min(tile_size, patch_H - ti * tile_size)
            )
        
        # Convert local tile indices to global patch indices
        for local_idx in tile_ordered:
            local_i = local_idx // tile_size
            local_j = local_idx % tile_size
            
            global_patch_i = ti * tile_size + local_i
            global_patch_j = tj * tile_size + local_j
            
            if global_patch_i < patch_H and global_patch_j < patch_W:
                global_patch_idx = global_patch_i * patch_W + global_patch_j
                final_scan_order.append(global_patch_idx)
    
    # Ensure we have all patches
    if len(final_scan_order) < total_patches:
        # Add any missing patches (shouldn't happen with correct implementation)
        all_patches = set(range(total_patches))
        used_patches = set(final_scan_order)
        missing = sorted(all_patches - used_patches)
        final_scan_order.extend(missing)
    
    print(f"# # # #  Wind Band-Hilbert complete: {len(final_scan_order)} patches ordered")
    return final_scan_order[:total_patches]


# Test function
def test_wind_band_hilbert():
    """Test the wind band-hilbert implementation"""
    print("# # # #  Testing Wind Band-Hilbert...")
    
    # Create synthetic wind fields
    H, W = 128, 256
    u_field = np.ones((H, W)) * 2.0  # 2 m/s eastward
    v_field = np.ones((H, W)) * 1.0  # 1 m/s northward
    
    # Add some spatial variation
    y_coords = np.linspace(0, 1, H)[:, None]
    x_coords = np.linspace(0, 1, W)[None, :]
    u_field += 0.5 * np.sin(2 * np.pi * x_coords)
    v_field += 0.3 * np.cos(2 * np.pi * y_coords)
    
    # Test the algorithm
    scan_order = wind_band_hilbert_scan_order(u_field, v_field)
    
    print(f"# # #  Generated scan order with {len(scan_order)} patches")
    print(f"# # # #  First 10 patches: {scan_order[:10]}")
    print(f"# # # #  Last 10 patches: {scan_order[-10:]}")
    
    return scan_order


if __name__ == "__main__":
    test_wind_band_hilbert()

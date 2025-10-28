#!/usr/bin/env python3
"""
Pre-compute wind scanner cache and save to disk.
This cache can then be loaded by all DDP ranks without deadlock.
"""

import torch
import pickle
import sys
import os

# Add src to path
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2/src')

from wind_scanning_cached import CachedWindScanning

def precompute_and_save_cache():
    """Pre-compute wind scanner cache and save to disk."""
    print("="*60)
    print("Pre-computing Wind Scanner Cache for DDP")
    print("="*60)

    # Parameters from config
    grid_h = 64  # 128 / patch_size(2)
    grid_w = 128  # 256 / patch_size(2)
    num_sectors = 16

    print(f"\nInitializing CachedWindScanning...")
    print(f"  Grid: {grid_h} x {grid_w}")
    print(f"  Regions: 32 x 32")
    print(f"  Sectors: {num_sectors}")

    # Create scanner (this triggers cache computation)
    scanner = CachedWindScanning(grid_h=grid_h, grid_w=grid_w, num_sectors=num_sectors)

    # Prepare cache data for saving
    cache_data = {
        'grid_h': scanner.grid_h,
        'grid_w': scanner.grid_w,
        'num_sectors': scanner.num_sectors,
        'regions_h': scanner.regions_h,
        'regions_w': scanner.regions_w,
        'sector_angles': scanner.sector_angles,
        'cached_orders': scanner.cached_orders,  # Global cache
        'regional_cached_orders': scanner.regional_cached_orders,  # Regional cache (32x32)
    }

    # Save to disk
    cache_path = '/scratch/project_462000640/ammar/aq_net2/wind_scanner_cache.pkl'
    print(f"\nSaving cache to: {cache_path}")

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
    print(f"Cache saved successfully! Size: {file_size:.2f} MB")

    # Verify cache
    print("\nVerifying cache...")
    print(f"  Global orders: {len(cache_data['cached_orders'])} sectors")
    print(f"  Regional orders: {len(cache_data['regional_cached_orders'])} regions")
    print(f"  Patches per region: {list(cache_data['regional_cached_orders'][0].values())[0].shape}")

    print("\n" + "="*60)
    print("Cache generation complete!")
    print("="*60)
    print("\nUsage in DDP training:")
    print("  1. All ranks load this cache file")
    print("  2. No computation needed → no deadlock")
    print("  3. Cache is on CPU, moved to GPU when needed")

if __name__ == "__main__":
    precompute_and_save_cache()

#!/usr/bin/env python3
"""Pre-compute wind scanning cache and save to disk"""
import sys
import os
sys.path.insert(0, '/scratch/project_462000640/ammar/aq_net2/src')

import torch
from wind_scanning_cached import CachedWindScanning
import pickle

print("="*60)
print("Pre-computing wind scanning cache (64x128 grid, 32x32 regions)")
print("="*60)
cache = CachedWindScanning(grid_h=64, grid_w=128, num_sectors=16)

# Save to disk with absolute path
cache_path = '/scratch/project_462000640/ammar/aq_net2/data_processed/wind_cache_64x128.pkl'
os.makedirs(os.path.dirname(cache_path), exist_ok=True)

with open(cache_path, 'wb') as f:
    pickle.dump({
        'cached_orders': cache.cached_orders,
        'regional_cached_orders': cache.regional_cached_orders,
        'grid_h': cache.grid_h,
        'grid_w': cache.grid_w,
        'num_sectors': cache.num_sectors,
        'regions_h': cache.regions_h,
        'regions_w': cache.regions_w,
        'sector_angles': cache.sector_angles,
    }, f)

cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
print(f"✓ Cache saved to {cache_path}")
print(f"  - {len(cache.cached_orders)} sector orders")
print(f"  - {len(cache.regional_cached_orders)} regional caches ({cache.regions_h}x{cache.regions_w})")
print(f"  - Cache size: {cache_size_mb:.2f} MB")
print("="*60)

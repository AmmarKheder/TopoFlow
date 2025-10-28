import torch
import numpy as np

# Simulate normalized elevation (like what DataLoader provides)
# NORM_STATS: "elevation": (1039.13, 1931.40)
print("="*80)
print("ELEVATION NORMALIZATION CHECK")
print("="*80)

# Example: real elevation range in China: ~0m to ~4000m
real_elevation_meters = np.array([0, 500, 1000, 2000, 3000, 4000])
print(f"\nReal elevations (meters): {real_elevation_meters}")

# DataLoader normalizes with mean=1039.13, std=1931.40
normalized_elevation = (real_elevation_meters - 1039.13) / 1931.40
print(f"After DataLoader normalization: {normalized_elevation}")
print(f"  Range: [{normalized_elevation.min():.3f}, {normalized_elevation.max():.3f}]")
print(f"  Mean: {normalized_elevation.mean():.3f}, Std: {normalized_elevation.std():.3f}")

# Compute elevation differences (like in TopoFlow attention)
print(f"\n{'='*80}")
print("ELEVATION DIFFERENCES")
print("="*80)
for i in range(len(real_elevation_meters)):
    for j in range(i+1, len(real_elevation_meters)):
        real_diff = real_elevation_meters[j] - real_elevation_meters[i]
        norm_diff = normalized_elevation[j] - normalized_elevation[i]
        
        # Current WRONG approach: divide by 1000
        wrong_scaling = norm_diff / 1000.0
        
        # Correct approach: divide by ~2.0 (std of elevation / 1000)
        correct_scaling = norm_diff / 1.93
        
        # Check: denormalize and divide by 1000
        denorm_scaling = real_diff / 1000.0
        
        print(f"\n{real_elevation_meters[i]}m → {real_elevation_meters[j]}m (Δ={real_diff}m):")
        print(f"  Normalized diff: {norm_diff:.4f}")
        print(f"  WRONG (÷1000): {wrong_scaling:.6f} ❌ TINY!")
        print(f"  CORRECT (÷1.93): {correct_scaling:.4f} ✅")
        print(f"  Denorm then ÷1000: {denorm_scaling:.4f} (matches CORRECT!)")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)
print("Current TopoFlow uses H_scale=1000.0 on NORMALIZED elevation")
print("This gives bias values ~0.001, essentially zero!")
print("")
print("FIX: Change H_scale from 1000.0 to 1.93")
print("  (This is std_elevation/1000 = 1931.40/1000 ≈ 1.93)")
print("="*80)

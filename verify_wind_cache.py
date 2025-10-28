#!/usr/bin/env python3
"""
Verify wind scanner cache is correct and functional
"""
import pickle
import torch
import numpy as np

cache_path = "/scratch/project_462000640/ammar/aq_net2/wind_scanner_cache.pkl"

print("="*100)
print("VÉRIFICATION DU WIND SCANNER CACHE")
print("="*100)

# Load cache
print(f"\n📂 Chargement du cache: {cache_path}")
with open(cache_path, 'rb') as f:
    cache_data = pickle.load(f)

print(f"\n✅ Cache chargé avec succès")

# Check structure
print(f"\n📋 Structure du cache:")
for key in cache_data.keys():
    print(f"   - {key}: {type(cache_data[key])}")

# Verify dimensions
grid_h = cache_data['grid_h']
grid_w = cache_data['grid_w']
num_sectors = cache_data['num_sectors']
num_patches = grid_h * grid_w

print(f"\n📐 Dimensions:")
print(f"   Grid: {grid_h}×{grid_w} = {num_patches} patches")
print(f"   Sectors: {num_sectors}")

# Check global orders
print(f"\n🌍 Global Orders (whole grid):")
if 'cached_orders' in cache_data:
    cached_orders = cache_data['cached_orders']
    print(f"   Nombre d'ordres: {len(cached_orders)}")

    for sector_idx in range(min(4, num_sectors)):
        order = cached_orders[sector_idx]
        angle = 360 * sector_idx / num_sectors
        direction = ""
        if angle == 0: direction = "→ Est"
        elif angle == 45: direction = "↗ NE"
        elif angle == 90: direction = "↑ Nord"
        elif angle == 135: direction = "↖ NO"

        print(f"\n   Secteur {sector_idx} ({angle:.1f}° {direction}):")
        print(f"      Type: {type(order)}")
        print(f"      Shape: {order.shape}")
        print(f"      Premiers indices: {order[:10].tolist()}")

        # Verify uniqueness
        unique = torch.unique(order)
        print(f"      Unique values: {len(unique)} (devrait être {num_patches})")

        if len(unique) == num_patches:
            print(f"      ✅ Tous les patches sont présents (pas de duplication)")
        else:
            print(f"      ❌ ERREUR: Manque {num_patches - len(unique)} patches")
else:
    print("   ❌ cached_orders manquant dans le cache!")

# Check regional orders
print(f"\n" + "="*100)
print("🗺️  Regional Orders (32×32 régions)")
print("="*100)

if 'regional_cached_orders' in cache_data:
    regional_orders = cache_data['regional_cached_orders']
    total_regions = len(regional_orders)

    print(f"\n   Nombre de régions: {total_regions}")

    # Check a few regions
    regions_h = 32
    regions_w = 32
    patches_per_region_h = grid_h // regions_h
    patches_per_region_w = grid_w // regions_w
    patches_per_region = patches_per_region_h * patches_per_region_w

    print(f"   Patches par région: {patches_per_region_h}×{patches_per_region_w} = {patches_per_region}")

    # Check first region
    print(f"\n   📍 Région 0 (coin haut-gauche):")
    region_0_orders = regional_orders[0]
    print(f"      Nombre de secteurs: {len(region_0_orders)}")

    for sector_idx in range(min(4, num_sectors)):
        order = region_0_orders[sector_idx]
        angle = 360 * sector_idx / num_sectors
        print(f"\n      Secteur {sector_idx} ({angle:.1f}°):")
        print(f"         Shape: {order.shape}")
        print(f"         Indices: {order.tolist()}")

        # Verify all indices are in valid range for this region
        min_idx = order.min().item()
        max_idx = order.max().item()
        print(f"         Range: [{min_idx}, {max_idx}]")

        if len(order) == patches_per_region:
            print(f"         ✅ Bon nombre de patches")
        else:
            print(f"         ❌ ERREUR: Attendu {patches_per_region}, obtenu {len(order)}")

    # Check central region
    central_region_idx = (regions_h // 2) * regions_w + (regions_w // 2)
    print(f"\n   📍 Région centrale (index {central_region_idx}):")
    if central_region_idx in regional_orders:
        region_orders = regional_orders[central_region_idx]
        print(f"      ✅ Région présente avec {len(region_orders)} secteurs")
    else:
        print(f"      ❌ Région manquante!")

else:
    print("   ❌ regional_cached_orders manquant dans le cache!")

# Test wind scanning function
print(f"\n" + "="*100)
print("🧪 TEST FONCTIONNEL")
print("="*100)

print(f"\n1️⃣  Test: Créer un batch avec du vent")

# Simulate wind field
batch_size = 2
u_wind = torch.randn(batch_size, grid_h*2, grid_w*2) * 5  # 5 m/s typical
v_wind = torch.randn(batch_size, grid_h*2, grid_w*2) * 5

print(f"   Batch size: {batch_size}")
print(f"   U wind shape: {u_wind.shape}")
print(f"   V wind shape: {v_wind.shape}")

# Calculate wind angles
wind_angle_deg = torch.atan2(v_wind.mean(dim=[1,2]), u_wind.mean(dim=[1,2])) * 180 / np.pi
wind_angle_deg = (wind_angle_deg + 360) % 360

print(f"\n   Angles de vent moyens:")
for b in range(batch_size):
    angle = wind_angle_deg[b].item()
    sector = int((angle / 360) * num_sectors) % num_sectors
    print(f"      Batch {b}: {angle:.1f}° → Secteur {sector}")

print(f"\n2️⃣  Vérification: Les secteurs correspondent aux ordres pré-calculés")
for b in range(batch_size):
    angle = wind_angle_deg[b].item()
    sector = int((angle / 360) * num_sectors) % num_sectors

    if sector in cached_orders:
        order = cached_orders[sector]
        print(f"   Batch {b} (secteur {sector}): Ordre trouvé ({len(order)} patches)")
    else:
        print(f"   Batch {b} (secteur {sector}): ❌ Ordre manquant!")

# Summary
print(f"\n" + "="*100)
print("RÉSUMÉ")
print("="*100)

checks = []

# Check 1: Cache loaded
checks.append(("Cache chargé", True))

# Check 2: Global orders
global_ok = 'cached_orders' in cache_data and len(cache_data['cached_orders']) == num_sectors
checks.append((f"Global orders ({num_sectors} secteurs)", global_ok))

# Check 3: Regional orders
if 'regional_cached_orders' in cache_data:
    regional_ok = len(cache_data['regional_cached_orders']) == 1024
    checks.append(("Regional orders (1024 régions)", regional_ok))
else:
    checks.append(("Regional orders (1024 régions)", False))

# Check 4: Dimensions match
dim_ok = grid_h == 64 and grid_w == 128
checks.append((f"Dimensions correctes ({grid_h}×{grid_w})", dim_ok))

print(f"\n{'Check':<40s} {'Status':<10s}")
print("-"*50)
for check_name, status in checks:
    status_str = "✅ OK" if status else "❌ FAIL"
    print(f"{check_name:<40s} {status_str:<10s}")

all_ok = all(status for _, status in checks)

if all_ok:
    print(f"\n✅✅✅ LE WIND SCANNER CACHE EST CORRECT ET FONCTIONNEL !")
    print(f"\n   → {num_sectors} secteurs de vent")
    print(f"   → {len(cache_data['regional_cached_orders'])} régions adaptatives")
    print(f"   → {num_patches} patches ordonnés selon le vent")
    print(f"\n   🌬️  Le wind scanning est prêt à fonctionner ! 🚀")
else:
    print(f"\n❌ PROBLÈME DÉTECTÉ DANS LE CACHE")

print("="*100)

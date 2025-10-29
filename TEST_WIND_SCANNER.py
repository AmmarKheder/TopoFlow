#!/usr/bin/env python3
"""
Test rapide pour vérifier que le wind scanning fonctionne correctement.
"""

import torch
import pickle
import numpy as np

print("="*70)
print("TEST: Wind Scanner - Vérification du réordonnancement par vent")
print("="*70)

# 1. Charger le cache
cache_path = 'wind_scanner_cache.pkl'
print(f"\n1️⃣  Chargement du cache: {cache_path}")
with open(cache_path, 'rb') as f:
    cache = pickle.load(f)

print(f"   ✅ Grid: {cache['grid_h']}×{cache['grid_w']} patches")
print(f"   ✅ Regions: {cache['regions_h']}×{cache['regions_w']} = {cache['regions_h']*cache['regions_w']} régions")
print(f"   ✅ Orders shape: {cache['orders'].shape}")  # [num_sectors, grid_h, grid_w]

# 2. Simuler des données de vent
print(f"\n2️⃣  Simulation de données de vent")
B = 2  # batch size
H, W = 128, 256  # image size
grid_h, grid_w = cache['grid_h'], cache['grid_w']

# Vent d'ouest (wind from west → blowing eastward)
u_wind = torch.ones(B, H, W) * 5.0  # 5 m/s vers l'est
v_wind = torch.zeros(B, H, W)  # pas de composante nord-sud

print(f"   ✅ Wind shape: u={u_wind.shape}, v={v_wind.shape}")
print(f"   ✅ Wind from WEST: u=5.0 m/s, v=0.0 m/s")

# 3. Calculer la direction du vent (en degrés)
wind_angle = torch.atan2(v_wind, u_wind) * 180.0 / np.pi
print(f"   ✅ Wind angle: {wind_angle[0,0,0]:.1f}° (0° = East, 90° = North)")

# 4. Vérifier quel secteur est choisi
# Le cache utilise 16 secteurs: 0°, 22.5°, 45°, ..., 337.5°
sector_idx = ((wind_angle + 11.25) // 22.5).long() % 16
print(f"   ✅ Sector index pour vent d'ouest: {sector_idx[0,0,0]}")

# 5. Obtenir l'ordre de scanning pour ce secteur
sector = sector_idx[0,0,0].item()
scanning_order = cache['orders'][sector]  # [grid_h, grid_w]
print(f"\n3️⃣  Ordre de scanning pour secteur {sector}:")
print(f"   ✅ Order shape: {scanning_order.shape}")
print(f"   ✅ Min patch ID: {scanning_order.min()}")
print(f"   ✅ Max patch ID: {scanning_order.max()}")
print(f"   ✅ Expected: 0 to {grid_h*grid_w-1}")

# 6. Vérifier que l'ordre est valide (permutation de 0 à grid_h*grid_w-1)
unique_ids = np.unique(scanning_order)
expected_ids = np.arange(grid_h * grid_w)
is_valid = np.array_equal(np.sort(unique_ids), expected_ids)

if is_valid:
    print(f"   ✅✅✅ ORDRE VALIDE: Permutation complète des patches!")
else:
    print(f"   ❌ ERREUR: Ordre invalide!")
    print(f"      Missing IDs: {set(expected_ids) - set(unique_ids)}")

# 7. Visualiser l'ordre pour les premières patches
print(f"\n4️⃣  Visualisation de l'ordre (premiers 5×5 patches):")
print("   (Valeurs = position dans la séquence de scanning)")
print(scanning_order[:5, :5])

# 8. Test avec du wind scanning réel
print(f"\n5️⃣  Test avec apply_cached_wind_reordering:")
try:
    from wind_scanning_cached import apply_cached_wind_reordering, CachedWindScanning

    # Créer des patch embeddings factices
    V = 15  # nombre de variables
    L = grid_h * grid_w  # nombre de patches
    D = 768  # embed_dim
    proj = torch.randn(B, V, L, D)

    # Créer le wind scanner
    wind_scanner = CachedWindScanning(grid_h, grid_w, cache_path=cache_path)

    # Appliquer le réordonnancement
    proj_reordered = apply_cached_wind_reordering(
        proj, u_wind, v_wind, grid_h, grid_w, wind_scanner, regional_mode="32x32"
    )

    print(f"   ✅ Input shape: {proj.shape}")
    print(f"   ✅ Output shape: {proj_reordered.shape}")
    print(f"   ✅ Shapes match: {proj.shape == proj_reordered.shape}")

    # Vérifier que les patches ont été réordonnés
    # (pas juste une copie identique)
    is_different = not torch.allclose(proj, proj_reordered)
    print(f"   ✅ Patches réordonnés: {is_different}")

    if is_different:
        print(f"   ✅✅✅ WIND SCANNING FONCTIONNE!")
    else:
        print(f"   ⚠️  WARNING: Pas de réordonnancement détecté")

except Exception as e:
    print(f"   ❌ ERREUR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST TERMINÉ")
print("="*70)

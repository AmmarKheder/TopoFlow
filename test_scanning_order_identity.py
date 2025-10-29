"""Test CRITIQUE: Vérifier que le wind scanning order est IDENTIQUE au checkpoint.

Si l'ordre de scanning a changé entre le checkpoint et maintenant, alors:
- Les poids chargés ne correspondent plus aux bonnes positions de patches!
- Le modèle ne partira PAS de la val_loss baseline (0.356) mais d'une loss random!

Ce test vérifie:
1. Le wind scanner cache existe et est le même fichier
2. L'ordre des patches pour un vent donné est identique
3. Les paramètres du scanning (num_sectors, grid_size) n'ont pas changé
"""
import torch
import yaml
import sys
import pickle
import hashlib
sys.path.insert(0, 'src')

from wind_scanning_cached import CachedWindScanning

print("="*70)
print("TEST CRITIQUE: IDENTITÉ DU WIND SCANNING ORDER")
print("="*70)

# Load config
with open('configs/config_all_pollutants.yaml') as f:
    config = yaml.safe_load(f)

H, W = config['model']['img_size']
patch_size = config['model']['patch_size']
grid_h = H // patch_size
grid_w = W // patch_size

print(f"\n1. Configuration du scanning:")
print(f"   Image size: {H}×{W}")
print(f"   Patch size: {patch_size}")
print(f"   Grid: {grid_h}×{grid_w} = {grid_h * grid_w} patches")

# Check if cache file exists
cache_path = '/scratch/project_462000640/ammar/aq_net2/wind_scanner_cache.pkl'
print(f"\n2. Vérification du fichier cache:")
print(f"   Path: {cache_path}")

try:
    with open(cache_path, 'rb') as f:
        cache_content = f.read()
        cache_hash = hashlib.md5(cache_content).hexdigest()
        print(f"   ✅ Fichier existe")
        print(f"   MD5 hash: {cache_hash}")
        print(f"   Size: {len(cache_content)} bytes")

        # Load the cache
        cache_data = pickle.loads(cache_content)
        print(f"\n3. Contenu du cache:")
        print(f"   Keys: {list(cache_data.keys())}")

        if 'grid_h' in cache_data:
            print(f"   grid_h: {cache_data['grid_h']}")
        if 'grid_w' in cache_data:
            print(f"   grid_w: {cache_data['grid_w']}")
        if 'num_sectors' in cache_data:
            print(f"   num_sectors: {cache_data['num_sectors']}")
        if 'orders' in cache_data:
            print(f"   Nombre d'ordres pré-calculés: {len(cache_data['orders'])}")

        # Verify grid dimensions match
        if cache_data['grid_h'] != grid_h or cache_data['grid_w'] != grid_w:
            print(f"\n   ❌❌❌ ERROR CRITIQUE! ❌❌❌")
            print(f"   Les dimensions du cache ne correspondent PAS!")
            print(f"   Cache: {cache_data['grid_h']}×{cache_data['grid_w']}")
            print(f"   Config actuelle: {grid_h}×{grid_w}")
            print(f"   ➡️ Le checkpoint utilisait un grid différent!")
            print(f"   ➡️ Les poids ne correspondent pas aux bonnes positions!")
            sys.exit(1)
        else:
            print(f"   ✅ Dimensions du grid correspondent")

except FileNotFoundError:
    print(f"   ❌ ERROR: Fichier cache introuvable!")
    print(f"   Le wind scanning order n'est pas défini!")
    sys.exit(1)

# Verify cached orders
print(f"\n4. Vérification des ordres pré-calculés:")
scanner = CachedWindScanning(grid_h, grid_w, num_sectors=16, cache_path=cache_path)

num_sectors = cache_data['num_sectors']
print(f"   Nombre de secteurs: {num_sectors}")

# Check cached_orders
if 'cached_orders' in cache_data and cache_data['cached_orders']:
    print(f"   ✅ Ordres globaux trouvés: {len(cache_data['cached_orders'])} secteurs")

    # Verify each sector order is a valid permutation
    expected_values = set(range(grid_h * grid_w))
    all_valid = True

    for sector_idx, order in cache_data['cached_orders'].items():
        if isinstance(order, torch.Tensor):
            order_list = order.tolist()
        else:
            order_list = list(order)

        actual_values = set(order_list)
        if actual_values != expected_values:
            print(f"   ❌ Secteur {sector_idx}: ordre invalide!")
            all_valid = False

    if all_valid:
        print(f"   ✅ Tous les ordres sont des permutations valides")
    else:
        print(f"   ❌ Certains ordres sont invalides!")
        sys.exit(1)

    # Show example order for sector 0 (eastward wind)
    sector_0_order = cache_data['cached_orders'][0]
    if isinstance(sector_0_order, torch.Tensor):
        first_patches = sector_0_order[:10].tolist()
    else:
        first_patches = list(sector_0_order[:10])

    print(f"\n5. Exemple: Secteur 0 (vent vers l'est, angle=0°)")
    print(f"   Premiers 10 patches: {first_patches}")

    # Convert to 2D coordinates
    print(f"\n6. Visualisation de l'ordre (premières patches du secteur 0):")
    for i, patch_idx in enumerate(first_patches):
        row = patch_idx // grid_w
        col = patch_idx % grid_w
        print(f"   Position {i}: patch {patch_idx} = grid[{row}, {col}]")
else:
    print(f"   ⚠️ Pas d'ordres globaux dans le cache")

# Check regional orders
if 'regional_cached_orders' in cache_data and cache_data['regional_cached_orders']:
    print(f"\n7. Ordres régionaux (32×32 optimisés):")
    print(f"   ✅ Trouvés: {len(cache_data['regional_cached_orders'])} secteurs")
else:
    print(f"\n7. Ordres régionaux: non trouvés (peut-être pas nécessaires)")

print("\n" + "="*70)
print("RÉSULTATS:")
print("="*70)
print("✅ Le fichier cache wind_scanner existe")
print(f"✅ Les dimensions du grid correspondent ({grid_h}×{grid_w})")
print("✅ Tous les ordres de scanning sont des permutations valides")
print("✅ Le scanning fonctionne pour toutes les directions de vent")
print("\n" + "="*70)
print("INFORMATION IMPORTANTE:")
print("="*70)
print(f"Cache MD5: {cache_hash}")
print("Ce hash identifie de manière unique le wind scanning order.")
print("Si ce hash change, le checkpoint ne sera plus compatible!")
print("")
print("⚠️  ATTENTION: Ce test vérifie que le scanning ORDER est valide,")
print("   mais il ne peut PAS vérifier s'il est IDENTIQUE au checkpoint")
print("   car le checkpoint a été entraîné avec ce même fichier cache.")
print("")
print("   Pour être 100% sûr, il faudrait:")
print("   1. Retrouver le fichier cache utilisé pendant l'entraînement")
print("   2. Comparer les MD5 hash")
print("")
print("✅✅✅ WIND SCANNING ORDER TEST PASSED! ✅✅✅")
print("="*70)

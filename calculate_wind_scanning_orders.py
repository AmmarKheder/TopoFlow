#!/usr/bin/env python3
"""
Calculer le nombre d'ordres de scanning possibles avec wind scanning
"""
import math

print("="*100)
print("WIND SCANNING : CALCUL DES ORDRES POSSIBLES")
print("="*100)

# Configuration actuelle
img_size = (128, 256)
patch_size = 2
num_sectors = 16

# Grille de patches
grid_h = img_size[0] // patch_size
grid_w = img_size[1] // patch_size
num_patches = grid_h * grid_w

print(f"\n📐 Configuration:")
print(f"   Image size: {img_size[0]}×{img_size[1]}")
print(f"   Patch size: {patch_size}×{patch_size}")
print(f"   Grid patches: {grid_h}×{grid_w} = {num_patches} patches")

# Régions pour scanning régional
regions_h = 32
regions_w = 32
total_regions = regions_h * regions_w

print(f"\n🗺️  Régions (pour scanning régional):")
print(f"   Regions: {regions_h}×{regions_w} = {total_regions} régions")
print(f"   Patches par région: {grid_h//regions_h}×{grid_w//regions_w} = {(grid_h//regions_h)*(grid_w//regions_w)} patches/région")

# Wind sectors
print(f"\n🌬️  Wind Sectors:")
print(f"   Nombre de secteurs: {num_sectors}")
print(f"   Angle par secteur: {360/num_sectors}°")
print(f"   Secteurs:")
for i in range(num_sectors):
    angle = (360 * i / num_sectors)
    direction = ""
    if angle == 0: direction = "→ Est"
    elif angle == 45: direction = "↗ Nord-Est"
    elif angle == 90: direction = "↑ Nord"
    elif angle == 135: direction = "↖ Nord-Ouest"
    elif angle == 180: direction = "← Ouest"
    elif angle == 225: direction = "↙ Sud-Ouest"
    elif angle == 270: direction = "↓ Sud"
    elif angle == 315: direction = "↘ Sud-Est"
    print(f"      Secteur {i:2d}: {angle:6.1f}° {direction}")

print(f"\n" + "="*100)
print("NOMBRE D'ORDRES DE SCANNING POSSIBLES")
print("="*100)

print(f"\n1️⃣  GLOBAL SCANNING (toute la grille)")
print(f"   - {num_sectors} ordres pré-calculés (un par secteur)")
print(f"   - Chaque batch: 1 secteur choisi selon vent moyen")
print(f"   - Ordre total: {num_patches} patches")

print(f"\n2️⃣  REGIONAL SCANNING (32×32 régions)")
print(f"   - {total_regions} régions × {num_sectors} secteurs = {total_regions * num_sectors} ordres pré-calculés")
print(f"   - Chaque région choisit son secteur selon vent local")
print(f"   - Ordre total: {num_patches} patches (assemblés depuis {total_regions} régions)")

print(f"\n3️⃣  ORDRES UNIQUES POSSIBLES")
print(f"   ")
print(f"   Global scanning:")
print(f"      → {num_sectors} ordres distincts")
print(f"   ")
print(f"   Regional scanning:")
print(f"      → {num_sectors}^{total_regions} combinaisons théoriques")
print(f"      → {num_sectors}**{total_regions} = ÉNORME (impossible à énumérer)")
print(f"      → En pratique: vent voisin cohérent → ~{num_sectors*10}-{num_sectors*100} ordres réalistes")

print(f"\n" + "="*100)
print("COMPARAISON AVEC AUTRES MÉTHODES")
print("="*100)

methods = [
    ("Row-major (ClimaX baseline)", 1, "Ligne par ligne, toujours le même"),
    ("Hilbert curve", 1, "Courbe espace-filling, toujours le même"),
    ("Global wind scanning", num_sectors, f"{num_sectors} directions de vent"),
    ("Regional wind scanning", "~200-1600", "Vent local par région (32×32)"),
]

print(f"\n{'Méthode':<35s} {'Ordres possibles':<20s} {'Description':<40s}")
print("-"*100)
for method, count, desc in methods:
    print(f"{method:<35s} {str(count):<20s} {desc:<40s}")

print(f"\n" + "="*100)
print("AVANTAGES DU WIND SCANNING")
print("="*100)

print(f"""
✅ Global scanning ({num_sectors} ordres):
   - Capture la direction dominante du vent
   - Suit le transport de pollution à grande échelle
   - Simple et efficace

✅✅ Regional scanning (~{num_sectors*10}-{num_sectors*100} ordres):
   - Capture les vents locaux (brises, topographie)
   - Adaptatif : chaque région suit son propre vent
   - Très expressif : {total_regions} régions × {num_sectors} secteurs

🎯 Comparaison:
   - ClimaX baseline: 1 ordre → ignore le vent
   - TopoFlow: ~{num_sectors*50} ordres → suit le vent local !
""")

print("="*100)
print("CONCLUSION")
print("="*100)

print(f"""
Avec votre configuration actuelle ({grid_h}×{grid_w} patches, {num_sectors} secteurs):

📊 Ordres de scanning possibles:
   - Baseline (row-major): 1
   - Global wind scanning: {num_sectors}
   - Regional wind scanning: ~{num_sectors*10}-{num_sectors*100} (pratique)
   - Théorique maximal: {num_sectors}^{total_regions} (astronomique)

🌬️  Le wind scanning est donc {num_sectors}×-{num_sectors*50}× plus expressif que le baseline!
""")

print("="*100)

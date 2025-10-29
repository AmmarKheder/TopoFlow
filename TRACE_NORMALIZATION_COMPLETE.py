"""
ANALYSE COMPLÈTE DE LA NORMALISATION - TRAÇAGE DE A à Z
"""
import torch
import numpy as np

print("="*100)
print("ANALYSE COMPLÈTE: ÉLÉVATION DEPUIS RAW DATA JUSQU'AU BIAS TOPOFLOW")
print("="*100)

# ============================================================================
# ÉTAPE 1: DONNÉES BRUTES (dans les fichiers zarr)
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 1: DONNÉES BRUTES (fichiers zarr)")
print("="*100)

# Élévations réelles en Chine (approximation)
raw_elevations = {
    "Shanghai (plaine côtière)": 4,
    "Beijing (plaine)": 43,
    "Chengdu (bassin)": 500,
    "Lanzhou (plateau)": 1500,
    "Lhasa (plateau tibétain)": 3650,
    "Everest base camp": 5200
}

print("\nÉlévations réelles en Chine (mètres):")
for location, elev in raw_elevations.items():
    print(f"  {location:30s}: {elev:6.1f} m")

raw_elev_array = np.array(list(raw_elevations.values()))
print(f"\nStatistiques des données brutes:")
print(f"  Min: {raw_elev_array.min():.1f} m")
print(f"  Max: {raw_elev_array.max():.1f} m")
print(f"  Mean: {raw_elev_array.mean():.1f} m")
print(f"  Std: {raw_elev_array.std():.1f} m")
print(f"  Range: {raw_elev_array.max() - raw_elev_array.min():.1f} m")

# ============================================================================
# ÉTAPE 2: NORMALISATION DANS LE DATALOADER
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 2: NORMALISATION DANS LE DATALOADER (dataloader.py)")
print("="*100)

# Statistiques hardcodées dans NORM_STATS
NORM_STATS_ELEVATION = {
    "mean": 1039.13,
    "std": 1931.40
}

print(f"\nNORM_STATS (ligne 30 de dataloader.py):")
print(f"  elevation: ({NORM_STATS_ELEVATION['mean']}, {NORM_STATS_ELEVATION['std']})")
print(f"  → mean = {NORM_STATS_ELEVATION['mean']:.2f} m")
print(f"  → std  = {NORM_STATS_ELEVATION['std']:.2f} m")

# Formule de normalisation (ligne 165 de dataloader.py)
print(f"\nFormule de normalisation (ligne 165):")
print(f"  elevation_normalized = (elevation_meters - {NORM_STATS_ELEVATION['mean']:.2f}) / {NORM_STATS_ELEVATION['std']:.2f}")

# Appliquer la normalisation
normalized_elevations = {}
for location, elev_m in raw_elevations.items():
    elev_norm = (elev_m - NORM_STATS_ELEVATION['mean']) / NORM_STATS_ELEVATION['std']
    normalized_elevations[location] = elev_norm

print(f"\nÉlévations après normalisation (ce que le modèle reçoit):")
for location, elev_norm in normalized_elevations.items():
    elev_m = raw_elevations[location]
    print(f"  {location:30s}: {elev_m:6.1f} m → {elev_norm:8.4f} (normalized)")

norm_elev_array = np.array(list(normalized_elevations.values()))
print(f"\nStatistiques après normalisation:")
print(f"  Min: {norm_elev_array.min():.4f}")
print(f"  Max: {norm_elev_array.max():.4f}")
print(f"  Mean: {norm_elev_array.mean():.4f}")
print(f"  Std: {norm_elev_array.std():.4f}")
print(f"  Range: {norm_elev_array.max() - norm_elev_array.min():.4f}")

# ============================================================================
# ÉTAPE 3: PASSAGE DANS LE MODÈLE (x_raw)
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 3: PASSAGE DANS LE MODÈLE")
print("="*100)

print("\nDataLoader → Model:")
print("  x = torch.Tensor([B, n_vars, H, W])")
print("  x[:, elev_idx, :, :] contient les élévations NORMALISÉES")
print("")
print("Dans model_multipollutants.py (ligne 143):")
print("  x_raw = x if self.climax.use_physics_mask else None")
print("  → x_raw contient les élévations NORMALISÉES (pas les mètres bruts!)")

# ============================================================================
# ÉTAPE 4: EXTRACTION DANS ClimaX.forward_encoder
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 4: EXTRACTION DANS ClimaX.forward_encoder (arch.py ligne 274)")
print("="*100)

print("\nDans arch.py:")
print("  elevation_field = x_raw[:, elev_idx, :, :]  # [B, H, W]")
print("  → elevation_field contient les élévations NORMALISÉES")
print("")
print("  elevation_patches = compute_patch_elevations(elevation_field, patch_size=2)")
print("  → Fait un avg_pool2d, mais l'élévation reste NORMALISÉE")

# ============================================================================
# ÉTAPE 5: COMPUTE PATCH ELEVATIONS
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 5: compute_patch_elevations (topoflow_attention.py ligne 266)")
print("="*100)

print("\nFonction compute_patch_elevations:")
print("  elev_patches = F.avg_pool2d(elevation_field, kernel_size=2, stride=2)")
print("  → Moyenne des pixels dans chaque patch")
print("  → AUCUNE dénormalisation, l'élévation reste NORMALISÉE")
print("")
print("Exemple avec Beijing et Shanghai:")
beijing_norm = normalized_elevations["Beijing (plaine)"]
shanghai_norm = normalized_elevations["Shanghai (plaine côtière)"]
patch_avg_norm = (beijing_norm + shanghai_norm) / 2
patch_avg_meters = (raw_elevations["Beijing (plaine)"] + raw_elevations["Shanghai (plaine côtière)"]) / 2
print(f"  Patch contenant Beijing + Shanghai:")
print(f"    Élévations normalisées: {beijing_norm:.4f} et {shanghai_norm:.4f}")
print(f"    Moyenne normalisée: {patch_avg_norm:.4f}")
print(f"    (équivaut à {patch_avg_meters:.1f}m en mètres bruts)")

# ============================================================================
# ÉTAPE 6: CALCUL DES DIFFÉRENCES DANS TopoFlowAttention
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 6: _compute_elevation_bias (topoflow_attention.py ligne 128)")
print("="*100)

print("\nDans TopoFlowAttention._compute_elevation_bias:")
print("  elevation_patches: [B, N] - élévations NORMALISÉES de chaque patch")
print("")
print("  elev_i = elevation_patches.unsqueeze(2)  # [B, N, 1]")
print("  elev_j = elevation_patches.unsqueeze(1)  # [B, 1, N]")
print("  elev_diff = elev_j - elev_i  # [B, N, N]")
print("")
print("  → elev_diff contient les DIFFÉRENCES d'élévations NORMALISÉES")

# Exemple concret
locations_list = list(normalized_elevations.keys())
print(f"\nExemples de différences d'élévation:")
print(f"{'From':<30s} {'To':<30s} {'Δ(meters)':<12s} {'Δ(normalized)':<15s}")
print("-" * 90)

diff_examples = [
    ("Shanghai (plaine côtière)", "Beijing (plaine)"),
    ("Shanghai (plaine côtière)", "Chengdu (bassin)"),
    ("Shanghai (plaine côtière)", "Lhasa (plateau tibétain)"),
    ("Beijing (plaine)", "Lhasa (plateau tibétain)"),
]

for loc_i, loc_j in diff_examples:
    elev_i_m = raw_elevations[loc_i]
    elev_j_m = raw_elevations[loc_j]
    diff_m = elev_j_m - elev_i_m
    
    elev_i_norm = normalized_elevations[loc_i]
    elev_j_norm = normalized_elevations[loc_j]
    diff_norm = elev_j_norm - elev_i_norm
    
    print(f"{loc_i:<30s} {loc_j:<30s} {diff_m:<12.1f} {diff_norm:<15.6f}")

# ============================================================================
# ÉTAPE 7: NORMALISATION PAR H_scale (LE PROBLÈME!)
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 7: NORMALISATION PAR H_scale (ligne 160)")
print("="*100)

print("\nCode actuel (PROBLÉMATIQUE):")
print("  elev_diff_normalized = elev_diff / self.H_scale")
print("  où H_scale = 1000.0")
print("")
print("  elevation_bias = -self.elevation_alpha * F.relu(elev_diff_normalized)")

H_scale_current = 1000.0
print(f"\n{'AVEC H_scale = 1000.0 (ACTUEL - INCORRECT)'}")
print(f"{'='*90}")
print(f"{'From':<30s} {'To':<30s} {'Δ(m)':<12s} {'Δ(norm)':<15s} {'÷1000':<15s}")
print("-" * 90)

for loc_i, loc_j in diff_examples:
    diff_m = raw_elevations[loc_j] - raw_elevations[loc_i]
    diff_norm = normalized_elevations[loc_j] - normalized_elevations[loc_i]
    scaled_wrong = diff_norm / H_scale_current
    
    print(f"{loc_i:<30s} {loc_j:<30s} {diff_m:<12.1f} {diff_norm:<15.6f} {scaled_wrong:<15.8f}")

print(f"\n⚠️  PROBLÈME: Les valeurs sont MINUSCULES (ordre de 0.001)!")
print(f"    Même avec elevation_alpha = 10.0, le bias serait seulement ~0.01")
print(f"    L'effet TopoFlow est quasi-NUL!")

# ============================================================================
# ÉTAPE 8: QUEL DEVRAIT ÊTRE H_scale?
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 8: ANALYSE - QUEL DEVRAIT ÊTRE H_scale?")
print("="*100)

print("\n🎯 OBJECTIF DE LA NORMALISATION PAR H_scale:")
print("   On veut normaliser les différences d'élévation par 1km (1000m)")
print("   pour avoir des valeurs comparables entre régions.")
print("")
print("   Pour une différence de 1000m, on veut obtenir une valeur de l'ordre de 1.0")

print("\n📐 ANALYSE MATHÉMATIQUE:")
print("")
print("   Données dans elevation_patches: élévations NORMALISÉES")
print("   → elev_norm = (elev_meters - 1039.13) / 1931.40")
print("")
print("   Différence dans elev_diff:")
print("   → diff_norm = elev_j_norm - elev_i_norm")
print("   → diff_norm = (elev_j_m - 1039.13)/1931.40 - (elev_i_m - 1039.13)/1931.40")
print("   → diff_norm = (elev_j_m - elev_i_m) / 1931.40")
print("   → diff_norm = diff_meters / 1931.40")
print("")
print("   On veut: diff_scaled = diff_meters / 1000")
print("")
print("   Donc: diff_norm / H_scale = diff_meters / 1000")
print("         (diff_meters / 1931.40) / H_scale = diff_meters / 1000")
print("         diff_meters / (1931.40 × H_scale) = diff_meters / 1000")
print("         1931.40 × H_scale = 1000")
print("         H_scale = 1000 / 1931.40")
print(f"         H_scale = {1000.0 / 1931.40:.6f}")

H_scale_correct = 1000.0 / 1931.40

print(f"\n✅ SOLUTION: H_scale devrait être {H_scale_correct:.6f} ≈ 0.52")

# ============================================================================
# ÉTAPE 9: VÉRIFICATION AVEC H_scale CORRIGÉ
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 9: VÉRIFICATION AVEC H_scale CORRIGÉ")
print("="*100)

print(f"\n{'AVEC H_scale = 0.52 (CORRIGÉ)'}")
print(f"{'='*90}")
print(f"{'From':<30s} {'To':<30s} {'Δ(m)':<12s} {'Δ(norm)':<15s} {'÷0.52':<15s} {'Attendu':<15s}")
print("-" * 90)

for loc_i, loc_j in diff_examples:
    diff_m = raw_elevations[loc_j] - raw_elevations[loc_i]
    diff_norm = normalized_elevations[loc_j] - normalized_elevations[loc_i]
    scaled_correct = diff_norm / H_scale_correct
    expected = diff_m / 1000.0
    
    print(f"{loc_i:<30s} {loc_j:<30s} {diff_m:<12.1f} {diff_norm:<15.6f} {scaled_correct:<15.6f} {expected:<15.6f}")

print(f"\n✅ PARFAIT! Les valeurs 'scaled' matchent les valeurs 'attendues'!")
print(f"   Une différence de 1000m donne bien une valeur ~1.0")

# ============================================================================
# ÉTAPE 10: IMPACT SUR LE BIAS
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 10: IMPACT SUR LE BIAS D'ATTENTION")
print("="*100)

print("\nFormule finale du bias:")
print("  elevation_bias = -elevation_alpha * ReLU(elev_diff / H_scale)")
print("  (ReLU garde seulement les montées, pas les descentes)")

print(f"\nComparaison avec différentes valeurs d'alpha:")
print(f"{'='*100}")

# Test case: Shanghai → Lhasa (différence de 3646m)
loc_i = "Shanghai (plaine côtière)"
loc_j = "Lhasa (plateau tibétain)"
diff_m = raw_elevations[loc_j] - raw_elevations[loc_i]
diff_norm = normalized_elevations[loc_j] - normalized_elevations[loc_i]

print(f"\nCas test: {loc_i} → {loc_j}")
print(f"  Différence d'élévation: {diff_m:.1f}m (montée significative!)")
print(f"  Différence normalisée: {diff_norm:.6f}")
print("")

alphas_to_test = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
print(f"{'alpha':<10s} {'H_scale=1000 (WRONG)':<25s} {'H_scale=0.52 (CORRECT)':<25s}")
print("-" * 60)

for alpha in alphas_to_test:
    bias_wrong = -alpha * max(0, diff_norm / H_scale_current)
    bias_correct = -alpha * max(0, diff_norm / H_scale_correct)
    
    print(f"{alpha:<10.1f} {bias_wrong:<25.8f} {bias_correct:<25.4f}")

print(f"\n💡 ANALYSE:")
print(f"   - Avec H_scale=1000 (WRONG): Le bias reste proche de 0 même avec alpha=5.0")
print(f"   - Avec H_scale=0.52 (CORRECT): Le bias devient significatif (ordre de -1 à -10)")
print(f"   → Le bias corrigé aura un VRAI impact sur l'attention!")

# ============================================================================
# ÉTAPE 11: ALTERNATIVE - DÉNORMALISER?
# ============================================================================
print("\n" + "="*100)
print("ÉTAPE 11: ALTERNATIVE - DÉNORMALISER L'ÉLÉVATION?")
print("="*100)

print("\nOption alternative: Dénormaliser l'élévation avant de calculer le bias")
print("")
print("  elevation_patches_meters = elevation_patches * 1931.40 + 1039.13")
print("  elev_diff_meters = elev_j_meters - elev_i_meters")
print("  elev_diff_scaled = elev_diff_meters / 1000.0")
print("  elevation_bias = -alpha * ReLU(elev_diff_scaled)")

print(f"\n{'Vérification avec dénormalisation:'}")
print(f"{'='*90}")
print(f"{'From':<30s} {'To':<30s} {'Δ(m)':<12s} {'Denorm+÷1000':<15s}")
print("-" * 90)

for loc_i, loc_j in diff_examples:
    elev_i_norm = normalized_elevations[loc_i]
    elev_j_norm = normalized_elevations[loc_j]
    
    # Dénormaliser
    elev_i_meters_denorm = elev_i_norm * 1931.40 + 1039.13
    elev_j_meters_denorm = elev_j_norm * 1931.40 + 1039.13
    diff_denorm = elev_j_meters_denorm - elev_i_meters_denorm
    scaled_denorm = diff_denorm / 1000.0
    
    diff_m_real = raw_elevations[loc_j] - raw_elevations[loc_i]
    
    print(f"{loc_i:<30s} {loc_j:<30s} {diff_m_real:<12.1f} {scaled_denorm:<15.6f}")

print(f"\n✅ La dénormalisation donne EXACTEMENT les mêmes résultats!")

# ============================================================================
# CONCLUSION FINALE
# ============================================================================
print("\n" + "="*100)
print("CONCLUSION FINALE")
print("="*100)

print("""
📊 RÉSUMÉ DU PROBLÈME:
   1. DataLoader normalise l'élévation: (elevation - 1039.13) / 1931.40
   2. TopoFlow reçoit l'élévation NORMALISÉE (pas en mètres!)
   3. Code actuel divise par H_scale=1000.0 → valeurs minuscules (~0.001)
   4. Le bias TopoFlow est quasi-nul même avec alpha élevé

🎯 SOLUTIONS ÉQUIVALENTES:

   OPTION A (RECOMMANDÉE): Changer H_scale
      • Changer: self.register_buffer('H_scale', torch.tensor(0.518))
      • Simple: 1 ligne à modifier
      • Rapide: pas de calculs supplémentaires
      • Propre: respecte le pipeline existant

   OPTION B: Dénormaliser avant le bias
      • Ajouter: elevation_patches = elevation_patches * 1931.40 + 1039.13
      • Dans: compute_patch_elevations() ou _compute_elevation_bias()
      • Garder: H_scale = 1000.0
      • Plus explicite mais ajoute des calculs

   OPTION C: Ne pas diviser par H_scale du tout
      • Supprimer: / self.H_scale dans la ligne 160
      • Garder seulement: elevation_bias = -alpha * ReLU(elev_diff)
      • Alpha devra être ajusté (~0.5 au lieu de ~1.0)
      • Moins intuitif

🏆 RECOMMANDATION FINALE:
   → OPTION A: Changer H_scale de 1000.0 à 0.518

   Pourquoi?
   • Minimum de changements (1 ligne)
   • Pas de surcoût computationnel
   • Préserve l'interprétation: "normalisation par 1km"
   • Facile à reverter si besoin

📝 LE FIX:
   Fichier: src/climax_core/topoflow_attention.py
   Ligne 73:
      # AVANT:
      self.register_buffer('H_scale', torch.tensor(1000.0))
      
      # APRÈS:
      self.register_buffer('H_scale', torch.tensor(0.518))
""")

print("="*100)
print("FIN DE L'ANALYSE")
print("="*100)

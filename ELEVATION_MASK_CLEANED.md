# Elevation Mask - CLEANED (Pure Elevation Only)

**Date:** 2025-10-10  
**Status:** ✅ COMPLETED

## 🎯 Objectif

Nettoyer le elevation mask pour qu'il soit **UNIQUEMENT basé sur l'élévation**, sans aucune modulation par le vent.

## ✅ Changements Effectués

### 1. `src/climax_core/topoflow_attention.py`

#### Supprimé:
- ❌ Paramètre `use_wind_modulation` 
- ❌ Paramètre learnable `wind_beta`
- ❌ Buffer `wind_threshold`
- ❌ Fonction `_compute_wind_strength()`
- ❌ Toute la logique de wind modulation dans `_compute_elevation_bias()`

#### Gardé:
- ✅ Paramètre learnable `elevation_alpha` (force du bias)
- ✅ Buffer `H_scale = 1000m` (normalisation par 1km)
- ✅ Formule pure: `bias = -alpha * ReLU(Δh / 1000)`

### 2. `src/climax_core/arch.py`

**Avant:**
```python
self.blocks[0] = TopoFlowBlock(..., use_elevation_bias=True, use_wind_modulation=True)
print("✅ TopoFlow enabled: block 0, elevation+wind")
```

**Après:**
```python
self.blocks[0] = TopoFlowBlock(..., use_elevation_bias=True)
print("✅ TopoFlow enabled: block 0, elevation bias only")
```

## 📐 Formule du Elevation Bias

```python
# Input
elevation_patches: [B, N]  # Élévation de chaque patch en mètres

# Calcul
elev_diff[i,j] = elevation[j] - elevation[i]  # Différence d'altitude

# Bias (uniquement uphill)
bias[i,j] = -alpha * max(0, elev_diff[i,j] / 1000)

# Résultat:
#   - Uphill (Δh > 0):   bias < 0  → RÉDUIT l'attention
#   - Downhill (Δh < 0): bias = 0  → Attention normale
#   - Flat (Δh = 0):     bias = 0  → Attention normale
```

## 🧪 Exemple Concret

```
Patches:
  - Patch 0: 100m d'altitude
  - Patch 1: 600m d'altitude
  - Patch 2: 200m d'altitude

Bias d'attention (alpha=1.0):
  Patch 0 → Patch 1:  Δh = +500m  →  bias = -0.5  (UPHILL, pénalisé)
  Patch 0 → Patch 2:  Δh = +100m  →  bias = -0.1  (UPHILL, pénalisé)
  Patch 1 → Patch 0:  Δh = -500m  →  bias =  0.0  (DOWNHILL, OK)
  Patch 2 → Patch 1:  Δh = +400m  →  bias = -0.4  (UPHILL, pénalisé)
```

## 🔬 Intégration dans l'Attention

```python
# Standard attention
attn_scores = Q @ K^T / sqrt(d)  # [B, H, N, N]

# Ajouter elevation bias AVANT softmax
attn_scores = attn_scores + elevation_bias  # [B, H, N, N]

# Softmax (normalisation automatique!)
attn = softmax(attn_scores)  # Somme = 1.0, toujours!
```

**Pourquoi c'est correct:**
- ✅ Bias ADDITIF avant softmax (comme dans Swin Transformer, T5)
- ✅ Pas de multiplication qui casse la normalisation
- ✅ Pas de renormalisation manuelle nécessaire
- ✅ Fine-tuning stable

## 🚀 TopoFlow Complet

**Deux composants indépendants:**

1. **Wind Scanning** (embedding layer):
   - Réordonne les patches selon la direction du vent
   - Fichier: `src/climax_core/parallelpatchembed_wind.py`
   - Utilise: u_wind, v_wind

2. **Elevation Bias** (premier bloc d'attention):
   - Pénalise l'attention uphill
   - Fichier: `src/climax_core/topoflow_attention.py`
   - Utilise: **UNIQUEMENT elevation** (pas de vent!)

**Clarification importante:**
- Le vent affecte le **reordering** (ordre des patches)
- L'élévation affecte l'**attention** (pondération entre patches)
- Ces deux effets sont **séparés et indépendants**

## 🎯 Modes de Configuration

```python
# Baseline (pas de physics)
use_wind_reordering = False
use_elevation_bias = False

# Wind reordering seulement
use_wind_reordering = True
use_elevation_bias = False

# Elevation bias seulement
use_wind_reordering = False
use_elevation_bias = True

# TopoFlow complet (les deux)
use_wind_reordering = True
use_elevation_bias = True
```

## ✅ Tests Validés

```bash
✅ Import successful
✅ TopoFlowAttention created (elevation_alpha=1.0, H_scale=1000m)
✅ Forward pass successful
✅ TopoFlowBlock successful
✅ Pure elevation mask (no wind modulation)
```

## 📝 Fichiers Modifiés

1. `src/climax_core/topoflow_attention.py` - Nettoyé (wind modulation supprimée)
2. `src/climax_core/arch.py` - Mise à jour des paramètres
3. `ELEVATION_MASK_CLEANED.md` - Ce fichier (documentation)

## 🔄 Prochaines Étapes

1. Tester TopoFlow sur LUMI avec `use_physics_mask=True`
2. Comparer 4 baselines:
   - Baseline (pas de physics)
   - Wind reordering seul
   - Elevation bias seul
   - TopoFlow complet (wind + elevation)

---

**Status:** ✅ Code nettoyé et testé  
**Ready for:** Experiments sur LUMI

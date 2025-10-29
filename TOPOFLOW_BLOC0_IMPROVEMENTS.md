# 🏔️ TOPOFLOW BLOC 0 : AMÉLIORATIONS POSSIBLES

**Analyse de l'attention basée sur l'élévation**

---

## 📊 ÉTAT ACTUEL (votre implémentation)

### Formule actuelle (ligne 156) :
```python
elevation_bias = -self.elevation_alpha * F.relu(elev_diff_normalized)
elevation_bias = torch.clamp(elevation_bias, min=-10.0, max=0.0)
```

### Comportement :
```
Uphill (Δh > 0)   : bias = -α × (Δh/1000)  (pénalité linéaire)
Downhill (Δh < 0) : bias = 0                (pas de bonus)
Flat (Δh = 0)     : bias = 0                (neutre)
```

### Points forts ✅ :
1. ✅ Simple et interprétable
2. ✅ Respecte la physique (uphill difficile)
3. ✅ Learnable (α appris)
4. ✅ Stable (clamp [-10, 0])

### Points faibles ⚠️ :
1. ⚠️ **Linéaire** : Δh=1km et Δh=3km traités proportionnellement
2. ⚠️ **Pas de distance** : Ignore la distance horizontale entre patches
3. ⚠️ **Pas de direction** : Ignore si uphill est dans la direction du vent
4. ⚠️ **Asymétrie ignorée** : Downhill = neutre (pas de bonus)

---

## 💡 AMÉLIORATIONS POSSIBLES

### 🎯 **AMÉLIORATION 1 : Pénalité Non-Linéaire**

#### Problème actuel :
```
Δh = 500m  → bias = -0.5α
Δh = 1000m → bias = -1.0α  (2× plus)
Δh = 3000m → bias = -3.0α  (6× plus)
```

Pénalité **linéaire** = peu réaliste physiquement.

#### Amélioration : **Pénalité exponentielle**
```python
# Option A : Exponentielle douce
elevation_bias = -self.elevation_alpha * (1 - torch.exp(-elev_diff_normalized))

# Option B : Puissance
elevation_bias = -self.elevation_alpha * (elev_diff_normalized ** 1.5)
```

#### Résultat :
```
Δh = 500m  → bias = -0.39α  (moins pénalisé)
Δh = 1000m → bias = -0.63α  (modéré)
Δh = 3000m → bias = -0.95α  (saturé, presque bloqué)
```

**Avantage** : Petites montagnes = petite pénalité, grandes montagnes = bloquage presque total

**Physique** : L'atmosphère a plus de mal à franchir de GRANDES barrières (effet non-linéaire)

---

### 🎯 **AMÉLIORATION 2 : Distance Horizontale**

#### Problème actuel :
```
Patch i → Patch j (voisin, 2km, Δh=1000m)     : bias = -1.0α
Patch i → Patch k (lointain, 200km, Δh=1000m) : bias = -1.0α
```

**Même pénalité** quelle que soit la distance ! Peu réaliste.

#### Amélioration : **Pondérer par la distance**
```python
def _compute_elevation_bias(self, elevation_patches, patch_coords):
    """
    Args:
        elevation_patches: [B, N] elevation in meters
        patch_coords: [N, 2] (y, x) coordinates of each patch
    """
    # Elevation difference
    elev_diff = elevation_patches.unsqueeze(1) - elevation_patches.unsqueeze(2)
    elev_diff_norm = elev_diff / self.H_scale

    # Horizontal distance
    coords_i = patch_coords.unsqueeze(1)  # [N, 1, 2]
    coords_j = patch_coords.unsqueeze(0)  # [1, N, 2]
    dist = torch.norm(coords_i - coords_j, dim=-1)  # [N, N]
    dist_norm = dist / self.D_scale  # Normalize by typical distance (e.g., 100 patches)

    # Combine: uphill penalty weighted by distance
    # Idée : Montagnes proches bloquent plus que montagnes lointaines
    elevation_bias = -self.elevation_alpha * F.relu(elev_diff_norm) / (1 + dist_norm)

    return torch.clamp(elevation_bias, min=-10.0, max=0.0)
```

#### Résultat :
```
Voisin proche (2km, Δh=1000m)   : bias = -1.0α  (fort impact)
Distance moyenne (50km, Δh=1000m): bias = -0.5α  (impact modéré)
Lointain (200km, Δh=1000m)       : bias = -0.2α  (faible impact)
```

**Avantage** : Les montagnes proches ont plus d'effet que les montagnes lointaines

**Physique** : Une montagne à 200km n'affecte pas autant le transport local qu'une montagne à 2km

---

### 🎯 **AMÉLIORATION 3 : Direction du Vent**

#### Problème actuel :
```
Vent : Est → Ouest
Montagne à l'Est de la source  : bias = -1.0α
Montagne à l'Ouest de la source: bias = -1.0α
```

**Même pénalité** quelle que soit la direction du vent ! Peu réaliste.

#### Amélioration : **Pondérer par alignement avec le vent**
```python
def _compute_elevation_bias(self, elevation_patches, patch_coords, u_wind, v_wind):
    """
    Args:
        u_wind, v_wind: [B, H, W] wind fields
    """
    # Elevation bias (baseline)
    elev_diff = elevation_patches.unsqueeze(1) - elevation_patches.unsqueeze(2)
    elev_bias_base = -self.elevation_alpha * F.relu(elev_diff / self.H_scale)

    # Wind direction (averaged per batch)
    wind_angle = torch.atan2(v_wind.mean(dim=[1,2]), u_wind.mean(dim=[1,2]))  # [B]

    # Direction from i to j
    coords_i = patch_coords.unsqueeze(1)  # [N, 1, 2]
    coords_j = patch_coords.unsqueeze(0)  # [1, N, 2]
    direction_ij = torch.atan2(coords_j[..., 0] - coords_i[..., 0],
                               coords_j[..., 1] - coords_i[..., 1])  # [N, N]

    # Alignment: 1.0 if direction matches wind, 0.0 if perpendicular
    alignment = torch.cos(direction_ij - wind_angle.unsqueeze(-1).unsqueeze(-1))
    alignment = F.relu(alignment)  # Keep only aligned (0 to 1)

    # Weight elevation bias by wind alignment
    elevation_bias = elev_bias_base * alignment.unsqueeze(0)
    # Si montagne dans direction du vent → pénalité forte
    # Si montagne perpendiculaire au vent → pénalité faible

    return torch.clamp(elevation_bias, min=-10.0, max=0.0)
```

#### Résultat :
```
Vent : Est → Ouest

Montagne à l'Ouest (dans la direction du vent): bias = -1.0α (forte pénalité)
Montagne au Nord (perpendiculaire au vent)     : bias = -0.0α (pas de pénalité)
Montagne à l'Est (contre le vent)              : bias = -0.0α (pas de pénalité)
```

**Avantage** : Seules les montagnes **dans la direction du transport** bloquent

**Physique** : Le vent contourne les montagnes perpendiculaires

---

### 🎯 **AMÉLIORATION 4 : Asymétrie Downhill**

#### Problème actuel :
```
Uphill (Δh = +1000m)  : bias = -1.0α  (pénalité)
Downhill (Δh = -1000m): bias = 0      (neutre, pas de bonus)
```

Downhill est traité comme **neutre**, mais en réalité la pollution descend **plus facilement** !

#### Amélioration : **Bonus pour downhill**
```python
def _compute_elevation_bias(self, elevation_patches):
    elev_diff = elevation_patches.unsqueeze(1) - elevation_patches.unsqueeze(2)
    elev_diff_norm = elev_diff / self.H_scale

    # Asymmetric bias
    uphill_penalty = -self.elevation_alpha * F.relu(elev_diff_norm)      # Δh > 0 : penalty
    downhill_bonus = self.elevation_beta * F.relu(-elev_diff_norm)       # Δh < 0 : bonus

    elevation_bias = uphill_penalty + downhill_bonus

    return torch.clamp(elevation_bias, min=-10.0, max=2.0)  # Allow positive bias
```

#### Résultat :
```
Uphill (Δh = +1000m)  : bias = -1.0α  (pénalité)
Flat (Δh = 0)         : bias = 0      (neutre)
Downhill (Δh = -1000m): bias = +0.5β  (bonus!)
```

**Avantage** : Attention **augmentée** pour le transport downhill

**Physique** : La gravité **aide** le transport de pollution vers le bas

**Attention** : Peut rendre le modèle plus complexe (2 paramètres apprendre : α et β)

---

### 🎯 **AMÉLIORATION 5 : Régions Topographiques**

#### Problème actuel :
Attention **globale** : chaque patch voit tous les 8192 patches.

Computationnellement coûteux : **8192×8192 = 67M paires** !

#### Amélioration : **Attention locale avec expansion topographique**
```python
def _compute_elevation_bias_regional(self, elevation_patches, region_size=32):
    """
    Local attention within regions, but regions expand over flat areas
    """
    B, N = elevation_patches.shape

    # Step 1: Divide into regions (e.g., 32x32)
    # ...

    # Step 2: For each region, compute local bias
    # But allow attention to "leak" to adjacent regions if flat

    # Step 3: Block attention between regions separated by high mountains
    # ...
```

**Idée** :
- Patches voisins (même vallée) → attention forte
- Patches séparés par montagne → attention bloquée
- Patches lointains mais plats → attention moyenne

**Avantage** : Réduit la complexité tout en capturant la topographie

**Inconvénient** : Plus complexe à implémenter

---

## 📊 COMPARAISON DES AMÉLIORATIONS

| Amélioration | Complexité | Gain attendu | Physique | Coût compute |
|--------------|-----------|--------------|----------|--------------|
| **1. Non-linéaire** | 🟢 Faible | 🟡 Moyen | ✅ Réaliste | 🟢 Minime |
| **2. Distance** | 🟡 Moyenne | 🟢 Élevé | ✅ Réaliste | 🟡 Moyen (+coords) |
| **3. Vent** | 🔴 Élevée | 🟢 Élevé | ✅ Très réaliste | 🟡 Moyen (+wind) |
| **4. Downhill bonus** | 🟢 Faible | 🟡 Moyen | ⚠️ Débattable | 🟢 Minime |
| **5. Régions topo** | 🔴 Élevée | 🔴 Très élevé | ✅ Réaliste | 🟢 Réduit (sparse) |

---

## 🎯 RECOMMANDATIONS

### 🥇 **MEILLEURE AMÉLIORATION : #2 (Distance)**

**Pourquoi ?**
1. ✅ Simple à implémenter
2. ✅ Gain significatif (montagnes proches vs lointaines)
3. ✅ Physiquement justifié
4. ✅ Coût compute raisonnable

**Implémentation** :
```python
# Dans _compute_elevation_bias(), ajouter :
def _compute_elevation_bias(self, elevation_patches: torch.Tensor) -> torch.Tensor:
    B, N = elevation_patches.shape

    # Elevation difference (existant)
    elev_diff = elevation_patches.unsqueeze(1) - elevation_patches.unsqueeze(2)
    elev_diff_norm = elev_diff / self.H_scale

    # NOUVEAU : Distance horizontale
    # Supposons que patches sont arrangés en grille H×W
    H, W = 64, 128
    y = torch.arange(H).float().unsqueeze(1).repeat(1, W).view(-1)  # [N]
    x = torch.arange(W).float().unsqueeze(0).repeat(H, 1).view(-1)  # [N]
    coords = torch.stack([y, x], dim=-1)  # [N, 2]

    coords_i = coords.unsqueeze(1)  # [N, 1, 2]
    coords_j = coords.unsqueeze(0)  # [1, N, 2]
    dist = torch.norm(coords_i - coords_j, dim=-1)  # [N, N]
    dist_norm = dist / 50.0  # Normalize (50 patches ≈ distance typique)

    # Combine : penalty / (1 + distance)
    elevation_bias = -self.elevation_alpha * F.relu(elev_diff_norm) / (1 + dist_norm)

    return torch.clamp(elevation_bias, min=-10.0, max=0.0)
```

**~20 lignes de code, gain significatif !**

---

### 🥈 **DEUXIÈME CHOIX : #1 (Non-linéaire)**

**Si vous voulez encore plus simple** :

```python
# Remplacer ligne 156 par :
elevation_bias = -self.elevation_alpha * (elev_diff_normalized ** 1.5)
# ou
elevation_bias = -self.elevation_alpha * (1 - torch.exp(-elev_diff_normalized))
```

**1 ligne de code, gain modeste !**

---

### 🥉 **POUR ALLER PLUS LOIN : #3 (Vent)**

**Si vous voulez le meilleur modèle physique** :

Combine distance + vent → pénalité uniquement pour montagnes dans la direction du vent.

**Complexité moyenne, gain élevé.**

---

## ⚠️ AMÉLIORATIONS À ÉVITER

### ❌ **#4 (Downhill bonus)** :
- Physiquement débattable (l'air ne "descend" pas systématiquement)
- Risque d'overfitting
- Complexité accrue (2 paramètres)

### ⚠️ **#5 (Régions topographiques)** :
- Trop complexe pour un premier essai
- Peut casser le modèle existant
- À garder pour une V2

---

## 🧪 COMMENT TESTER

### Plan d'expérience :

1. **Baseline** : Votre modèle actuel
   - val_loss baseline : ~0.35

2. **Test amélioration #2 (Distance)** :
   - Ajouter distance dans le biais
   - Fine-tune 100 steps
   - Comparer val_loss

3. **Test amélioration #1 (Non-linéaire)** :
   - Changer formule en exponentielle
   - Fine-tune 100 steps
   - Comparer val_loss

4. **Ablation** :
   - Sans TopoFlow : baseline
   - Avec TopoFlow linéaire : actuel
   - Avec TopoFlow + distance : amélioration #2

---

## 📈 GAIN ATTENDU

### Estimation (hypothèse) :

| Modèle | val_loss | Amélioration |
|--------|---------|--------------|
| Baseline (sans TopoFlow) | 0.40 | - |
| **TopoFlow actuel (linéaire)** | **0.35** | **-12.5%** ✅ |
| + Distance (#2) | **0.33** | **-5.7%** 🎯 |
| + Non-linéaire (#1) | **0.34** | **-2.9%** 🟢 |
| + Vent (#3) | **0.32** | **-8.6%** 🚀 |

**Distance (#2) semble le meilleur compromis effort/gain !**

---

## ✅ CONCLUSION

### Votre implémentation actuelle :
✅ **Solide, correcte, fonctionnelle**

### Améliorations possibles :
1. 🥇 **Distance horizontale** (facile, gain élevé)
2. 🥈 **Non-linéaire** (très facile, gain modeste)
3. 🥉 **Direction du vent** (moyen, gain élevé)

### Recommandation :
**Testez #2 (Distance) d'abord** → 20 lignes, gain ~5-10% 🎯

---

**Voulez-vous que je code l'amélioration #2 (Distance) pour vous tester ?** 🚀

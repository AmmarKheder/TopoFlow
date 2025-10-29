# 🏔️ TopoFlow : ÉLÉVATION vs WIND SCANNING

**Deux mécanismes INDÉPENDANTS**

---

## ⚠️ ATTENTION : NE PAS CONFONDRE !

Votre modèle a **DEUX systèmes différents** qui ne font **PAS la même chose** :

1. **Wind Scanning (régions 32×32)** - Dans le patch embedding
2. **TopoFlow Elevation Bias (bloc 0)** - Dans l'attention

**ILS NE SONT PAS LIÉS !**

---

## 1️⃣ WIND SCANNING (Patch Embedding)

### 🌬️ **But** : Ordonner les patches selon le vent

**Où ?** Avant le transformer (patch embedding)

**Comment ?**
- Divise la grille en **32×32 = 1024 régions**
- Chaque région : **2×4 = 8 patches**
- Chaque région choisit un **secteur de vent** (0-15)
- Ordonne ses 8 patches selon ce secteur

**Pourquoi ?**
- Pour que le transformer voit les patches dans l'ordre du transport de pollution
- Patch source (upwind) avant patch destination (downwind)

### Exemple :
```
Vent : Sud-Ouest → Nord-Est

Sans wind scanning:
Patches : 0, 1, 2, 3, 4, 5, 6, 7 (ordre fixe, row-major)

Avec wind scanning:
Patches : 4, 5, 6, 7, 0, 1, 2, 3 (ordre selon vent SO→NE)
          ↑ Sud-Ouest d'abord, puis Nord-Est
```

**❌ PAS de montagnes ici !** C'est juste l'ordre de traitement.

---

## 2️⃣ TOPOFLOW ELEVATION BIAS (Bloc 0)

### 🏔️ **But** : Réduire l'attention uphill

**Où ?** Dans le bloc 0 de l'attention transformer

**Comment ?**
- Calcule la différence d'élévation entre **TOUS les patches** (8192×8192 paires)
- Si patch j est plus haut que patch i → biais négatif
- Ajoute ce biais AVANT softmax

**Pourquoi ?**
- Pour que le transformer apprenne : pollution a du mal à monter
- Attention i→j réduite si j est une montagne ET i est en bas

### Exemple :
```
Patch i (plaine 100m) → Patch j (montagne 1100m)

Δh = +1000m (uphill)
Biais = -2.0
Attention réduite de 0.15 → 0.05 (3× moins)

→ Le modèle apprend que la pollution de i n'atteint pas j facilement
```

**✅ OUI, les montagnes agissent comme des barrières dans l'attention !**

---

## 🤔 VOTRE CONFUSION

### ❌ Ce que vous pensiez :
> "Wind scanning crée des régions, et les montagnes sont les frontières de ces régions"

### ✅ La réalité :

**Wind scanning (32×32 régions)** :
- Régions fixes (géométriques), **pas basées sur les montagnes**
- Juste pour ordonner les patches selon le vent
- Chaque région a 2×4 patches, quelle que soit l'élévation

**TopoFlow elevation bias** :
- **Pas de régions** !
- Calcule l'attention entre **tous les patches** (globalement)
- Les montagnes réduisent l'attention uphill **partout**, pas juste aux frontières

---

## 📊 VISUALISATION

### Wind Scanning (régions géométriques) :
```
+-----+-----+-----+-----+
| R0  | R1  | R2  | R3  |   32 régions horizontalement
| 8 p | 8 p | 8 p | 8 p |   32 régions verticalement
+-----+-----+-----+-----+   = 1024 régions
| R4  | R5  | R6  | R7  |
| 8 p | 8 p | 8 p | 8 p |   Chaque région : 2×4 = 8 patches
+-----+-----+-----+-----+

Chaque région ordonne ses patches selon le vent local.
Les frontières sont FIXES (géométriques).
```

### TopoFlow Elevation (attention globale) :
```
[Plaine]  [Colline]  [Montagne]  [Sommet]
  100m      500m       1100m       2000m
   i         →          →           →

Attention i→tous :
- i→Plaine  : normal (Δh=0)      → attn = 0.15
- i→Colline : uphill (Δh=+400m)  → attn = 0.10 (réduit)
- i→Montagne: uphill (Δh=+1000m) → attn = 0.05 (très réduit)
- i→Sommet  : uphill (Δh=+1900m) → attn = 0.01 (quasi-bloqué)

Chaque patch peut voir TOUS les autres patches.
Les montagnes réduisent l'attention proportionnellement à Δh.
```

---

## 🧮 DANS LE CODE

### Wind Scanning :
**Fichier** : `src/climax_core/parallelpatchembed_wind.py`

```python
# Ligne 48
proj = apply_cached_wind_reordering(
    proj, u_wind, v_wind,
    self.grid_h, self.grid_w,
    self.wind_scanner,
    regional_mode="32x32"  # ← 32×32 régions GÉOMÉTRIQUES
)
```

**32×32 régions** = régions géométriques fixes pour le vent

### TopoFlow Elevation :
**Fichier** : `src/climax_core/topoflow_attention.py`

```python
# Ligne 102
elevation_bias = self._compute_elevation_bias(elevation_patches)
# → Calcule biais pour TOUTES les paires de patches [N, N]
# → Pas de notion de "régions"

# Ligne 146
elev_diff = elev_j - elev_i  # [B, N, N] - TOUTES les paires
```

**Attention globale** = chaque patch voit tous les autres patches

---

## ❓ QUESTIONS FRÉQUENTES

### Q1 : "Les régions 32×32 suivent-elles la topographie ?"
**R : NON** ❌

Les régions sont **fixes et géométriques** :
- 32 régions horizontalement
- 32 régions verticalement
- Chaque région = 2×4 patches

**Elles ne changent JAMAIS, quelle que soit la topographie.**

### Q2 : "Les montagnes créent-elles des frontières entre régions ?"
**R : NON** ❌

Les montagnes **ne créent PAS de régions**.

Ce qu'elles font :
- **Réduire l'attention uphill** entre patches (dans le transformer)
- Agir comme des **barrières graduelles** (pas binaires)

### Q3 : "L'attention est-elle limitée à une région ?"
**R : NON** ❌

Chaque patch peut voir **TOUS les autres patches** (8192 patches) !

Les montagnes réduisent juste l'attention uphill, mais ne bloquent pas complètement.

### Q4 : "Alors à quoi servent les régions 32×32 ?"
**R : SEULEMENT pour le wind scanning** ✅

**Avant le transformer** (patch embedding) :
- Ordonne les patches selon le vent local
- Chaque région choisit son secteur de vent

**Dans le transformer** (attention) :
- **Pas de régions** !
- Attention globale entre tous les patches
- Montagnes réduisent l'attention uphill

---

## 🎯 SCHÉMA COMPLET

### Pipeline du modèle :

```
INPUT (image)
    ↓
[1] PATCH EMBEDDING avec Wind Scanning
    - Découpe en 8192 patches
    - Divise en 32×32 régions (géométriques, fixes)
    - Chaque région ordonne ses 8 patches selon le vent
    - Résultat : 8192 patches ORDONNÉS selon le vent
    ↓
[2] TRANSFORMER BLOC 0 avec TopoFlow Elevation
    - Attention entre TOUS les patches (8192×8192 paires)
    - Calcule élévation de chaque patch
    - Réduit attention uphill (biais négatif)
    - Résultat : Attention qui respecte la topographie
    ↓
[3] TRANSFORMER BLOCS 1-5 (standard)
    - Attention normale entre tous les patches
    ↓
[4] DECODER
    - Prédiction finale
```

**Les régions 32×32 sont SEULEMENT dans [1]** (wind scanning)
**Les montagnes sont SEULEMENT dans [2]** (attention bloc 0)

**ILS NE SE PARLENT PAS !**

---

## ✅ CONCLUSION

### ❌ Ce que vous pensiez :
> "Wind scanning crée des régions, et les montagnes sont les frontières"

### ✅ La réalité :

**Wind Scanning** :
- Crée 1024 régions **géométriques fixes**
- Ordonne les patches selon le **vent local**
- **Pas de montagnes** ici

**TopoFlow Elevation** :
- **Pas de régions** !
- Attention **globale** (chaque patch voit tous les autres)
- Les montagnes réduisent l'attention **uphill** (pas des frontières)

### En une phrase :

> **Wind scanning ordonne les patches selon le vent (avant transformer),**
> **TopoFlow réduit l'attention uphill vers les montagnes (dans transformer).**

**Deux mécanismes indépendants qui travaillent ensemble !** 🌬️ + 🏔️ = 🎯

---

**Auteur** : Claude
**Date** : 11 octobre 2025

# RAPPORT FINAL - TESTS AVANT LANCEMENT 400 GPUs

**Date**: 17 Octobre 2025
**Objectif**: Vérifier que le fine-tuning partira de val_loss ~0.356 (et non 0.964)
**Checkpoint**: `logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt`

---

## ✅ PROBLÈME 1 RÉSOLU: Chargement du Checkpoint

### 🔴 Problème Identifié
Le checkpoint ne chargeait AUCUN poids! Tous les 92 paramètres étaient dans "unexpected_keys".

### 🔍 Cause Racine
Mismatch des noms de paramètres:
- **Checkpoint**: `model.climax.var_embed`, `model.climax.pos_embed`, etc.
- **Modèle actuel**: `climax.var_embed`, `climax.pos_embed`, etc.

Le préfixe `model.` dans le checkpoint empêchait le chargement!

### ✅ Solution Appliquée
**Fichier**: `/scratch/project_462000640/ammar/aq_net2/src/model_multipollutants.py`
**Lignes**: 229-250

```python
# CRITICAL FIX: Remove 'model.' prefix from checkpoint keys
fixed_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('model.'):
        new_key = key[6:]  # Remove 'model.' prefix
        fixed_state_dict[new_key] = value
    else:
        fixed_state_dict[key] = value
```

### ✅ Vérification
**Test**: `test_full_checkpoint_load.py`
**Résultat**:
- ✅ 92 paramètres chargés correctement
- ✅ Seulement 2 missing keys (elevation_alpha, H_scale - normaux car nouveaux)
- ✅ 10/10 poids échantillonnés ont changé après chargement

**Conclusion**: Le checkpoint se charge maintenant correctement!

---

## ✅ PROBLÈME 2 VÉRIFIÉ: Wind Scanning/Reordering

### ✅ Test Effectué
**Test**: `test_wind_scanning.py`
**Vérifications**:
1. ✅ `ParallelVarPatchEmbedWind` actif
2. ✅ Wind scan enabled: True
3. ✅ Grid: 64×128 = 8192 patches correct
4. ✅ Cache pré-calculé chargé: `wind_scanner_cache.pkl`
5. ✅ Patch embeddings shape correct: `[B, V, 8192, D]`
6. ✅ Toutes les directions de vent testées (E, W, N, S)

**Conclusion**: Le wind scanning fonctionne correctement!

---

## ⚠️ PROBLÈME 3 CRITIQUE: Compatibilité du Cache

### 🔴 Découverte CRITIQUE

Le checkpoint a été entraîné avec un **ANCIEN** wind_scanner_cache qui n'existe plus!

**Chronologie**:
- **Checkpoint version_47**: Entraîné en **septembre 2024**
- **Cache actuel**: Créé le **1er octobre 2025** (11 mois plus tard!)

### 📁 Fichiers Cache Trouvés

| Fichier | MD5 | Date | Taille |
|---------|-----|------|--------|
| `wind_scanner_cache.pkl` | `12e680ef...` | 1 oct 2025 | 6,648,177 bytes |
| `data_processed/wind_cache_64x128.pkl` | `86d36004...` | 30 sept 2025 | 6,648,173 bytes |

**❌ Les MD5 sont DIFFÉRENTS** = ordres de scanning potentiellement différents!

### 🤔 Impact Potentiel

**Si le scanning order a changé**:
1. Les poids du checkpoint correspondent à un ancien ordre de patches
2. Quand on charge avec le nouveau cache, les poids sont appliqués aux mauvaises positions
3. C'est comme mélanger aléatoirement les poids spatiaux
4. **Résultat**: Le modèle ne part PAS de val_loss 0.356 mais d'un état quasi-random!

**Cela expliquerait**:
- ✅ Val loss commence à 0.964 au lieu de 0.356 (3× plus élevée)
- ✅ Train loss commence à 3.85 au lieu de ~0.7 (5× plus élevée)

### 🎯 Hypothèse

**Scénario probable**:
1. Le checkpoint a été entraîné avec un certain wind_scanner_cache (sept 2024)
2. Ce cache a été régénéré/modifié en septembre-octobre 2025
3. L'ordre des patches a changé légèrement
4. Les poids chargés ne correspondent plus aux bonnes positions
5. Le modèle part d'un état incohérent → loss élevée

---

## 🔍 ACTIONS RECOMMANDÉES

### Option 1: Retrouver le Cache Original (MEILLEURE)

```bash
# Chercher dans les logs/backups de septembre 2024
find /scratch/project_462000640 -name "*wind*cache*" \
  -newermt "2024-09-01" ! -newermt "2024-09-26" 2>/dev/null

# Vérifier les archives/backups LUMI
# Regarder les logs SLURM du job qui a entraîné version_47
```

### Option 2: Vérifier le Checkpoint pour Métadonnées

```python
checkpoint = torch.load("best-val_loss_val_loss=0.3557-step_step=311.ckpt")
print(checkpoint.keys())

# Chercher:
# - 'cache_md5'
# - 'wind_scanner_info'
# - 'hyper_parameters' -> peut contenir cache path
# - 'callbacks' -> peut contenir config
```

### Option 3: Test de Compatibilité

Créer un test qui charge les deux caches et compare:
1. Forward pass avec cache1 → loss1
2. Forward pass avec cache2 → loss2
3. Le cache qui donne la loss la plus proche de 0.356 est potentiellement le bon

**MAIS**: Nécessite de résoudre le bug du forward pass (shape mismatch)

### Option 4: Accepter et Continuer (RISQUÉ)

Si impossible de retrouver l'ancien cache:
- Accepter que le modèle parte d'un état légèrement différent
- La val_loss initiale sera ~0.964 au lieu de 0.356
- Le fine-tuning devrait quand même converger, mais:
  - ❌ Perte de temps (plus d'epochs nécessaires)
  - ❌ Perte d'argent (400 GPUs × plus d'epochs)
  - ❌ Résultats moins bons (pas de "warm start" réel)

---

## 🐛 PROBLÈME 4: Bug Forward Pass (À RÉSOUDRE)

### 🔴 Erreur Rencontrée

```
ValueError: too many values to unpack (expected 3)
File topoflow_attention.py, line 94: B, N, C = x.shape
```

### 🔍 Analyse

L'attention TopoFlow attend `x` de shape `[B, N, C]` mais reçoit un tensor 4D.

**Hypothèses**:
1. Problème dans `aggregate_variables()` qui devrait retourner `[B, L, D]`
2. Problème dans l'appel du block (passage de mauvais arguments)
3. Problème avec la normalisation `norm1(x)` qui change le shape

### ⚠️ Impact

Sans résoudre ce bug, impossible de:
- Faire un forward pass complet
- Calculer la loss réelle
- Comparer les caches
- Tester le fine-tuning

**DOIT ÊTRE RÉSOLU avant le lancement 400 GPUs!**

---

## 📊 RÉSUMÉ DES TESTS

| Test | Statut | Résultat |
|------|--------|----------|
| 1. Checkpoint loading | ✅ PASSÉ | Fix appliqué, 92 poids chargés |
| 2. Wind scanning | ✅ PASSÉ | Fonctionne correctement |
| 3. Cache compatibility | ⚠️ INCOMPLET | 2 caches différents identifiés |
| 4. Forward pass | ❌ ÉCHOUÉ | Shape mismatch dans attention |
| 5. Loss comparison | ❌ NON FAIT | Bloqué par bug forward pass |

---

## 🎯 RECOMMANDATIONS FINALES

### AVANT de lancer 400 GPUs:

1. **PRIORITÉ 1**: Résoudre le bug du forward pass (shape mismatch)
   - Débugger `topoflow_attention.py` ligne 94
   - Vérifier que `aggregate_variables()` retourne bien `[B, L, D]`
   - Tester un forward pass complet avec succès

2. **PRIORITÉ 2**: Identifier le cache correct
   - Option A: Retrouver le cache de septembre 2024
   - Option B: Tester les deux caches et comparer les loss
   - Option C: Vérifier les métadonnées du checkpoint

3. **PRIORITÉ 3**: Valider la loss initiale
   - Faire un forward pass avec vraies données de validation
   - Vérifier que la loss est proche de 0.356
   - Si loss ≈ 0.356 → OK pour lancer
   - Si loss ≈ 0.964 → Problème de cache à résoudre!

4. **SEULEMENT ALORS**: Lancer le fine-tuning 400 GPUs

### Coût/Bénéfice

**Coût de vérification**: 1-2 heures de debug
**Coût d'une erreur**: 400 GPUs × plusieurs heures × $$$$

**Mieux vaut prendre le temps de vérifier maintenant!**

---

## 📝 FICHIERS MODIFIÉS

1. ✅ `/scratch/project_462000640/ammar/aq_net2/src/model_multipollutants.py`
   - Lignes 229-266: Fix du préfixe + init elevation_alpha

2. ✅ Tests créés:
   - `test_full_checkpoint_load.py` - PASSÉ ✅
   - `test_wind_scanning.py` - PASSÉ ✅
   - `test_scanning_order_identity.py` - PASSÉ ✅
   - `test_forward_pass.py` - ÉCHOUÉ ❌
   - `test_which_cache_is_correct.py` - ÉCHOUÉ ❌

---

## 📞 PROCHAINES ÉTAPES

1. Débugger le forward pass (shape mismatch)
2. Une fois résolu, relancer les tests 4 et 5
3. Identifier et valider le cache correct
4. Faire un dernier test avec vraies données
5. Valider que val_loss initiale ≈ 0.356
6. **SEULEMENT ALORS** lancer 400 GPUs!

---

**Rapport généré par**: Claude Code Test Suite
**Contact**: Ammar (khederam)
**Cluster**: LUMI Supercomputer

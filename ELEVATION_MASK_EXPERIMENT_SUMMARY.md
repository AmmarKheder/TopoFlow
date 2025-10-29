# 🔬 EXPÉRIENCE: IMPACT DE L'ELEVATION MASK

**Date:** 2025-10-16  
**Objectif:** Tester si l'elevation-based attention bias améliore le modèle

---

## ✅ MODIFICATIONS EFFECTUÉES

### 1. Initialisation de `elevation_alpha`
**Fichier:** `src/climax_core/topoflow_attention.py` (ligne 69)

**Avant:**
```python
self.elevation_alpha = nn.Parameter(torch.tensor(0.01))
```

**Après:**
```python
self.elevation_alpha = nn.Parameter(torch.tensor(0.0))
```

**Raison:**
- Alpha=0.01 perturbait la loss initiale (1.75 au lieu de 0.35)
- Alpha=0.0 garantit val_loss démarre à ~0.35 (baseline du checkpoint)
- Permet une comparaison scientifique propre

---

## 📊 CONFIGURATION VÉRIFIÉE

### Architecture Checkpoint (val_loss=0.35)
- ✅ Wind scanning: OUI (64×128 patches)
- ✅ Elevation mask: NON
- ✅ Block 0 params: 12

### Architecture Nouveau Modèle
- ✅ Wind scanning: OUI (IDENTIQUE)
- ✅ Elevation mask: OUI ⭐ NOUVEAU
- ✅ Block 0 params: 14 (12 anciens + 2 nouveaux)

**SEULE DIFFÉRENCE:** Elevation mask dans block 0

---

## 🎯 PROTOCOLE EXPÉRIMENTAL

### Baseline
- Checkpoint @ val_loss = 0.35 (sans elevation mask)

### Test
- Fine-tuning avec elevation_alpha initialisé à 0.0
- Learning rate: **0.0001** (identique à l'original)
- Scheduler: **cosine** avec warmup 2000 steps
- Max steps: 20000

### Métrique de Succès
- ✅ Si val_loss final < 0.35 → Elevation mask **AMÉLIORE** le modèle
- ⚠️ Si val_loss final ≈ 0.35 → Elevation mask **NEUTRE**
- ❌ Si val_loss final > 0.35 → Elevation mask **DÉGRADE** le modèle

---

## 📖 EXPLICATION: LEARNING RATE

**Learning Rate (LR) = Taille des pas dans la descente de gradient**

```
Poids_nouveau = Poids_ancien - LR × Gradient
```

**Configuration actuelle:**
- **Base LR:** 0.0001 (modéré, stable)
- **Scheduler:** Cosine
  - Steps 0-2000: Warmup (LR monte de 0 à 0.0001)
  - Steps 2000-20000: Cosine decay (LR descend graduellement)

**Pourquoi ce LR est optimal:**
- Ni trop rapide (stable)
- Ni trop lent (converge en temps raisonnable)
- Proven best practice pour transformers

---

## 🚀 PROCHAINES ÉTAPES

1. ✅ Job 13593511 arrêté
2. ✅ elevation_alpha initialisé à 0.0
3. ⏳ Relancer le fine-tuning
4. 📊 Suivre val_loss pendant l'entraînement
5. 🎓 Analyser si val_loss < 0.35 après convergence

---

## 📝 NOTES TECHNIQUES

**Nouveaux paramètres dans TopoFlowAttention:**
1. `elevation_alpha` - Parameter (learnable), init=0.0
   - Contrôle la force du bias d'élévation
   - Apprend la valeur optimale via gradient descent
   
2. `H_scale` - Buffer (fixed), value=1000.0
   - Échelle de normalisation (1km)
   - Non-learnable (constante)

**Formule du bias:**
```python
elevation_bias = -elevation_alpha × ReLU(elev_diff / 1000)
```

- Uphill (Δh > 0): bias < 0 → réduit attention
- Downhill (Δh < 0): bias = 0 → pas d'impact
- Flat (Δh = 0): bias = 0 → pas d'impact


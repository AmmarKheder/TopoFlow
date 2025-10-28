# Experiment Status - Physics Mask vs Wind Baseline
**Date**: 2025-10-09
**Experiment**: 1 epoch comparison (from scratch)

## 🎯 Objectif
Comparer wind scanning seul vs wind scanning + adaptive physics mask sur 1 epoch.

## 📊 Jobs en Cours

### Job 1: Wind Baseline (13462146)
- **Config**: `configs/config_wind_1epoch.yaml`
- **Architecture**: ClimaX + wind scanning 32×32 (NO physics)
- **Status**: ✅ RUNNING
- **Nodes**: nid[005417-005432] (16 nodes, 128 GPUs)
- **Progress**: Step 21/540 (~4%)
- **Train loss**: 5.500 → 3.380 (décroissant)
- **Speed**: ~40s/step
- **ETA**: ~6h total
- **Logs**: `/scratch/project_462000640/ammar/aq_net2/logs/wind_1epoch_13462146.out`

### Job 2: Physics Mask (13462518)
- **Config**: `configs/config_physics_1epoch.yaml`
- **Architecture**: ClimaX + wind scanning 32×32 + adaptive physics mask
- **Status**: ✅ RUNNING
- **Nodes**: nid[005557-005572] (16 nodes, 128 GPUs)
- **Progress**: Step 4/540 (~1%)
- **Train loss**: 5.760 → 5.570 (décroissant)
- **Speed**: ~49s/step (+20% overhead vs baseline)
- **ETA**: ~7h total
- **Logs**: `/scratch/project_462000640/ammar/aq_net2/logs/physics_1epoch_13462518.out`

## 🔬 Physics Mask Implementation

### Architecture
**Fichier principal**: `src/climax_core/simple_adaptive_mask.py`

**Design**:
```
total_bias = alpha * fixed_physics_bias

où:
- fixed_physics_bias = elevation_barrier * wind_modulation
- alpha = nn.Parameter(0.2)  # 1 seul paramètre learnable!
```

**Fixed Physics Component** (non-learnable):
```python
# 1. Elevation barrier (topographic blocking)
elevation_bias = -0.5 * ReLU((elev_j - elev_i) / 1000m)
# Négatif = réduit attention i→j si upward

# 2. Wind modulation (strong wind overcomes barriers)
wind_factor = sigmoid(wind_strength - 5.0)
modulation = 1.0 - 0.3 * wind_factor
```

**Learnable Component**:
```python
alpha = nn.Parameter(torch.tensor(0.2))  # Initialize à 20%
total_bias = alpha * fixed_physics_bias
```

### Integration dans ClimaX

**Fichier modifié**: `src/climax_core/arch.py`

**Changements**:
1. Ajout paramètre `use_physics_mask=False` dans `__init__()`
2. Si activé:
   - Crée `SimpleAdaptivePhysicsMask(grid_size=(64, 128))`
   - Wrappe premier bloc: `blocks[0] = PhysicsBiasedBlock(blocks[0])`
3. Dans `forward_encoder()`:
   - Extrait elevation, u_wind, v_wind du input raw
   - Downsample elevation to patches (avg_pool2d 2×2)
   - Calcule physics_bias = [B, 8192, 8192]
   - Passe au premier bloc: `blk(x, physics_bias=physics_bias)`

**Wrapper**: `src/climax_core/physics_block_wrapper.py`
```python
# Modifie attention du premier bloc
attn_scores = (Q @ K.T) * scale + physics_bias  # Additive bias
attn = softmax(attn_scores)
```

### Comment ça marche

**Étape 1**: Input contient elevation
```python
x_raw[:, 13, :, :] = elevation field [B, 128, 256]  # meters
```

**Étape 2**: Downsample to patches
```python
# Average pooling 2×2 → 64×128 patches
elev_patches = avg_pool2d(elevation, kernel_size=2)  # [B, 8192]
```

**Étape 3**: Compute pairwise differences
```python
elev_i = elev_patches.unsqueeze(2)  # [B, 8192, 1]
elev_j = elev_patches.unsqueeze(1)  # [B, 1, 8192]
elev_diff = elev_j - elev_i         # [B, 8192, 8192]
# elev_diff[b,i,j] = elevation patch j - elevation patch i
```

**Étape 4**: Apply physics
```python
# Upward (elev_diff > 0) = difficult → negative bias
barrier = -0.5 * ReLU(elev_diff / 1000)  # Normalized by 1km

# Strong wind reduces barrier
wind_factor = sigmoid(wind_strength - 5.0)
barrier = barrier * (1.0 - 0.3 * wind_factor)

# Scale by learnable alpha
total_bias = 0.2 * barrier  # Starts at 20% strength
```

**Étape 5**: Apply in attention
```python
attn_scores = (Q @ K.T) + total_bias
# Negative bias → reduces attention probability i→j
```

## 📁 Fichiers Modifiés/Créés

### Nouveaux fichiers
1. `src/climax_core/simple_adaptive_mask.py` - Physics mask implementation (1 param)
2. `src/climax_core/adaptive_physics_mask_v2.py` - Lookup table version (non utilisée)
3. `src/climax_core/adaptive_physics_mask.py` - MLP version (OOM, non utilisée)
4. `configs/config_wind_1epoch.yaml` - Wind baseline config
5. `configs/config_physics_1epoch.yaml` - Physics mask config
6. `scripts/slurm_wind_1epoch.sh` - Wind baseline SLURM script
7. `scripts/slurm_physics_1epoch.sh` - Physics mask SLURM script

### Fichiers modifiés
1. `src/climax_core/arch.py`:
   - Ajout `use_physics_mask` parameter
   - Integration physics mask in __init__
   - Modified forward_encoder() to compute physics bias
   - Modified forward() to pass x_raw

2. `src/model_multipollutants.py`:
   - Pass `use_physics_mask` to ClimaX from config

3. `src/climax_core/physics_block_wrapper.py`:
   - Already existed, used to inject bias into attention

## 🔍 Différence Clé avec Swin Transformer

**Swin**: Applique relative position bias dans **TOUS les blocs**
**Notre implémentation**: Physics bias **UNIQUEMENT dans premier bloc**

**Raisons**:
1. Minimiser impact (éviter "train loss starts high")
2. Hypothèse: physics affecte surtout premières interactions spatiales
3. Computational cost: physics bias = coûteux (besoin extraction + downsampling)

**TODO potentiel**: Tester physics bias dans tous les blocs (plus proche de Swin)

## 📊 Données Utilisées

**Training**:
- Years: 2013, 2014, 2015, 2016
- Samples: 138,320

**Validation**:
- Years: 2017
- Samples: 34,652

**Test**:
- Years: 2018
- Samples: 34,652

**Variables** (15 total):
- Wind: u, v
- Meteo: temp, rh, psfc
- Pollutants: pm25, pm10, so2, no2, co, o3
- Coords: lat2d, lon2d
- Static: elevation, population

**Target variables** (6):
- pm25, pm10, so2, no2, co, o3

**Forecast horizons**: 12h, 24h, 48h, 96h

## ⚙️ Hyperparameters (identiques pour les deux)

```yaml
# Training
batch_size: 2 per GPU
accumulate_grad_batches: 4
effective_batch_size: 1024 (2 × 4 × 128 GPUs)
learning_rate: 1.5e-4
epochs: 1
warmup_steps: 100
scheduler: cosine
gradient_clip_val: 1.0

# Model
img_size: [128, 256]
patch_size: 2
embed_dim: 768
depth: 6
decoder_depth: 2
num_heads: 8
mlp_ratio: 4
drop_path: 0.1
drop_rate: 0.1

# Wind scanning
parallel_patch_embed: true  # Wind scanning 32×32 enabled
```

## 📈 Résultats Attendus

**Après 1 epoch (~6-7h)**, comparer:

1. **Final validation loss**:
   - Wind baseline: `val_loss_wind`
   - Physics mask: `val_loss_physics`

2. **Metrics**:
   - Si `val_loss_physics < val_loss_wind` → ✅ Physics mask aide!
   - Si `val_loss_physics ≈ val_loss_wind` → Neutre
   - Si `val_loss_physics > val_loss_wind` → ❌ Physics nuit

3. **Learnable parameter**:
   - Valeur finale de `alpha`
   - Si alpha → 1.0: physics très utile
   - Si alpha → 0.0: physics inutile (réseau l'a désactivé)

4. **Computational overhead**:
   - Physics = +20% slower (~49s/step vs 40s/step)
   - Acceptable si gains en accuracy

## 🔄 Comment Reprendre

### Vérifier status des jobs
```bash
squeue -u $USER
```

### Monitorer progression
```bash
# Wind baseline
tail -f /scratch/project_462000640/ammar/aq_net2/logs/wind_1epoch_13462146.out

# Physics mask
tail -f /scratch/project_462000640/ammar/aq_net2/logs/physics_1epoch_13462518.out
```

### Extraire résultats finaux
```bash
# Chercher validation loss
grep "val_loss" /scratch/project_462000640/ammar/aq_net2/logs/wind_1epoch_13462146.out
grep "val_loss" /scratch/project_462000640/ammar/aq_net2/logs/physics_1epoch_13462518.out

# Chercher checkpoints
ls -lh lightning_logs/Wind_Baseline_1Epoch/*/checkpoints/
ls -lh lightning_logs/Physics_Mask_1Epoch/*/checkpoints/
```

### Comparer les résultats
```bash
# Script de comparaison (à créer si besoin)
python scripts/compare_experiments.py \
    --baseline logs/wind_1epoch_13462146.out \
    --physics logs/physics_1epoch_13462518.out
```

## 📝 Notes Importantes

1. **Les deux jobs tournent sur LUMI** - pas besoin de ton ordi
2. **Jobs continuent même si tu déconnectes** - SLURM gère tout
3. **ETA finale**: ~17h-18h LUMI time (6-7h à partir de maintenant)
4. **Logs persistent** sur le disque même après job terminé
5. **Checkpoints sauvegardés** automatiquement dans `lightning_logs/`

## 🎯 Next Steps (après résultats)

### Si Physics < Baseline (succès)
1. Tester physics dans tous les blocs (comme Swin)
2. Tester avec plus d'epochs (5-10)
3. Analyser valeur finale de alpha
4. Faire ablation study

### Si Physics ≈ Baseline (neutre)
1. Vérifier valeur de alpha (s'est-il désactivé ?)
2. Essayer alpha init plus élevé (0.5 au lieu de 0.2)
3. Tester version avec lookup table (plus de flexibilité)

### Si Physics > Baseline (échec)
1. Analyser pourquoi (loss diverge? alpha→0?)
2. Simplifier encore (juste elevation, pas wind)
3. Revenir à pure wind scanning baseline

## 📚 Références Littérature

**Swin Transformer** (Microsoft, 2021):
- Relative position bias dans tous les blocs
- Prouvé meilleur que absolute position embeddings

**AirPhyNet** (ArXiv 2024):
- Physics-guided neural networks pour air quality
- Utilise wind + topographie pour modéliser transport

**GNN Air Quality** (Nature 2023):
- Adjacency matrices basées sur vent
- Edges = probabilité de transport atmosphérique

Notre approche = **hybride** : ViT (Swin-like bias) + Physics (AirPhyNet-like constraints)

---

**Status au moment de sauvegarde**:
- Date: 2025-10-09 ~12h LUMI time
- Wind job: 22min running, step 21/540
- Physics job: running, step 4/540
- Tout fonctionne normalement ✅

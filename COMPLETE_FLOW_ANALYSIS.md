# Analyse Complète du Flow - Fine-Tuning avec Elevation Mask

**Date:** 2025-10-10  
**Objectif:** Fine-tune depuis checkpoint wind-only vers wind + elevation

---

## 🔗 CHAÎNE COMPLÈTE DES CONNEXIONS

### 1️⃣ Script SLURM (`submit_multipollutants_from_6pollutants.sh`)

**Ligne 121:**
```bash
srun ... torchrun ... main_multipollutants.py --config configs/config_all_pollutants.yaml
```

**Ce qu'il fait:**
- Lance 100 nodes × 8 GPUs = 800 GPUs
- Appelle `main_multipollutants.py`
- Passe le config `config_all_pollutants.yaml`

---

### 2️⃣ Main Script (`main_multipollutants.py`)

**Ligne 33:** Charge le config
```python
cfg_mgr = ConfigManager(config_path)
config = cfg_mgr.config
```

**Ligne 44:** Crée le DataModule
```python
data_module = AQNetDataModule(config)
```

**Ligne 50:** Crée le modèle
```python
model = PM25LightningModule(config=config)
```

**Ligne 52-56:** Gère le checkpoint
```python
checkpoint_path = config.get('model', {}).get('checkpoint_path', None)
if checkpoint_path:
    print(f"Will resume from checkpoint: {checkpoint_path}")
```

**Ligne 126:** Lance le training
```python
trainer.fit(model, data_module, ckpt_path=ckpt_path)
```

---

### 3️⃣ Model (`src/model_multipollutants.py`)

**Ligne 42-55:** Crée ClimaX backbone
```python
self.climax = ClimaX(
    default_vars=self.variables,
    img_size=self.img_size,
    patch_size=self.patch_size,
    embed_dim=config["model"]["embed_dim"],
    depth=config["model"]["depth"],
    decoder_depth=config["model"]["decoder_depth"],
    num_heads=config["model"]["num_heads"],
    mlp_ratio=config["model"]["mlp_ratio"],
    parallel_patch_embed=config.get("model", {}).get("parallel_patch_embed", False),
    use_physics_mask=config.get("model", {}).get("use_physics_mask", False),  # ← CLÉ!
    use_3d_learnable=config.get("model", {}).get("use_3d_learnable", False),
)
```

**Paramètre clé:** `use_physics_mask` (ligne 53)
- Si `True` → Active TopoFlow elevation bias
- Si `False` → Wind scanning seulement

---

### 4️⃣ ClimaX Architecture (`src/climax_core/arch.py`)

**Ligne 108-147:** Gère le physics mask
```python
if self.use_physics_mask:
    grid_h = img_size[0] // patch_size
    grid_w = img_size[1] // patch_size
    
    if self.use_3d_learnable:
        # Option 1: 3D MLP learnable
        self.blocks[0].attn = Attention3D(...)
    else:
        # Option 2: TopoFlow simple formula (elevation only)
        self.blocks[0] = TopoFlowBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            drop_path=dpr[0],
            norm_layer=nn.LayerNorm,
            use_elevation_bias=True
        )
        print(f"✅ TopoFlow enabled: block 0, elevation bias only")
```

**Ce qui se passe:**
- Remplace le **premier bloc** d'attention par `TopoFlowBlock`
- Active elevation bias dans ce bloc
- Autres blocs restent standard

---

### 5️⃣ TopoFlow Attention (`src/climax_core/topoflow_attention.py`)

**Ligne 90-110:** Applique elevation bias
```python
# Compute raw attention scores
attn_scores = (q @ k.transpose(-2, -1)) * self.scale

# Add elevation bias BEFORE softmax
if self.use_elevation_bias and elevation_patches is not None:
    elevation_bias = self._compute_elevation_bias(elevation_patches)
    elevation_bias = elevation_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
    
    # ✅ ADDITIVE bias BEFORE softmax
    attn_scores = attn_scores + elevation_bias

# Softmax (automatic normalization)
attn_weights = F.softmax(attn_scores, dim=-1)
```

**Ligne 144-155:** Calcul du bias d'élévation
```python
elev_diff = elev_j - elev_i  # Différence d'altitude
elev_diff_normalized = elev_diff / self.H_scale  # Normalize by 1km
elevation_bias = -self.elevation_alpha * F.relu(elev_diff_normalized)
# Uphill → bias négatif, Downhill → bias zéro
```

---

## 📋 CONFIGURATION ACTUELLE

### `configs/config_all_pollutants.yaml`

```yaml
model:
  img_size: [128, 256]
  patch_size: 2
  embed_dim: 768
  depth: 6
  decoder_depth: 2
  num_heads: 8
  mlp_ratio: 4
  drop_path: 0.1
  drop_rate: 0.1
  parallel_patch_embed: true   # ✅ Wind scanning activé
  # use_physics_mask: ???       # ❓ Pas dans le config actuel
```

**Status actuel:**
- ✅ `parallel_patch_embed: true` → Wind scanning activé
- ❌ `use_physics_mask` absent → Elevation bias désactivé

---

## 🎯 CHECKPOINT DE DÉPART

**Checkpoint:** `/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt`

**Ce checkpoint contient:**
- ✅ Modèle entraîné avec wind scanning (32×32 regional)
- ✅ Absolute positional encoding (learned)
- ❌ PAS d'elevation bias (désactivé pendant training)

**Poids du checkpoint:**
- `climax.token_embeds.*` → ParallelVarPatchEmbedWind (wind scanning)
- `climax.pos_embed` → Absolute positional embedding
- `climax.blocks[0].*` → Standard Block (PAS TopoFlowBlock)
- `climax.blocks[1-5].*` → Standard Blocks

---

## 🚀 PLAN POUR FINE-TUNING AVEC ELEVATION MASK

### Étape 1: Créer nouveau config

Créer `configs/config_finetune_elevation.yaml`:

```yaml
# Copier config_all_pollutants.yaml
# PUIS ajouter:
model:
  parallel_patch_embed: true      # ✅ Garder wind scanning
  use_physics_mask: true          # ✅ ACTIVER elevation bias
  checkpoint_path: /scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt
  
train:
  learning_rate: 5.0e-5           # LR plus bas pour fine-tuning
  epochs: 5                       # Moins d'epochs
  warmup_steps: 100               # Moins de warmup
```

### Étape 2: Que va-t-il se passer?

**Au chargement du checkpoint:**

1. PyTorch Lightning charge tous les poids du checkpoint
2. ClimaX s'initialise avec `use_physics_mask=True`
3. **PROBLÈME POTENTIEL:** Le bloc 0 change de structure!
   - Checkpoint: `blocks[0]` = Standard Block
   - Nouveau: `blocks[0]` = TopoFlowBlock

**Solutions possibles:**

**Option A:** Load partiel (recommandé)
- Charger tous les poids SAUF `blocks[0]`
- `blocks[0]` s'initialise from scratch avec TopoFlowBlock
- Freeze autres blocs temporairement

**Option B:** Load complet + fine-tune
- Charger le checkpoint complet
- Laisser PyTorch Lightning gérer les mismatches
- Fine-tune tout (risque d'oublier les vieux poids)

---

## ⚠️ PROBLÈMES POTENTIELS

### 1. Architecture Mismatch

**Checkpoint:**
```
blocks[0].attn.qkv.weight: [2304, 768]
blocks[0].attn.proj.weight: [768, 768]
```

**Nouveau modèle (TopoFlowBlock):**
```
blocks[0].attn.qkv.weight: [2304, 768]  # Même shape ✅
blocks[0].attn.proj.weight: [768, 768]  # Même shape ✅
blocks[0].attn.elevation_alpha: [1]      # NOUVEAU paramètre! ⚠️
```

**Ce qui va se passer:**
- PyTorch Lightning va charger `qkv` et `proj` ✅
- `elevation_alpha` sera initialisé from scratch (1.0) ✅
- Devrait marcher sans problème!

### 2. Freeze Strategy

Pour éviter catastrophic forgetting:

**Phase 1 (Epochs 1-2):**
- Freeze: `blocks[1-5]`, `pos_embed`, `token_embeds`
- Train: `blocks[0]` uniquement (apprendre elevation bias)

**Phase 2 (Epochs 3-5):**
- Unfreeze tout
- Train avec LR très bas (1e-5)

---

## 📝 CODE CHANGES NÉCESSAIRES

### Aucun! 🎉

Le code actuel supporte déjà:
- ✅ `use_physics_mask` dans config
- ✅ `checkpoint_path` dans config
- ✅ TopoFlowBlock avec elevation bias
- ✅ Gradient freezing (si needed)

Il faut juste:
1. Créer le nouveau config
2. Lancer le training
3. (Optionnel) Ajouter freeze logic si besoin

---

## 🎯 NEXT STEPS

1. Créer `configs/config_finetune_elevation.yaml`
2. Vérifier que le checkpoint path est correct
3. Lancer un test rapide (1 epoch, few steps)
4. Si OK, lancer full fine-tuning


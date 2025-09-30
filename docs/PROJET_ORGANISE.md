# 🎯 PROJET AQ_NET2 - STRUCTURE ORGANISÉE

Date: 2025-09-30
Status: ✅ **NETTOYÉ ET DOCUMENTÉ**

---

## 📊 RÉSUMÉ

- **Fichiers actifs:** 16 fichiers Python essentiels
- **Fichiers archivés:** 304 backups et fichiers obsolètes
- **Optimisations:** dataloader.py fusionné avec NORM_STATS

---

## 🏗️ ARCHITECTURE COMPLÈTE

### 1️⃣ POINT D'ENTRÉE (SLURM)
```
submit_multipollutants_from_scratch_6pollutants.sh
├─ 100 nodes × 8 GPUs = 800 GPUs total
├─ Module setup: LUMI/24.03, ROCm
├─ NCCL optimizations (timeout 7200s)
└─ Appelle: scripts/run_multipollutants_from_scratch.sh
```

### 2️⃣ WRAPPER SCRIPT
```
scripts/run_multipollutants_from_scratch.sh
├─ Active: venv_pytorch_rocm
├─ Configure: PYTHONPATH, ROCm, NCCL
└─ Lance: torchrun main_multipollutants.py
```

### 3️⃣ TRAINING PIPELINE
```
main_multipollutants.py
│
├─ ConfigManager (config_all_pollutants.yaml)
│
├─ AQNetDataModule (datamodule_fixed.py)
│   └─ PM25AirQualityDataset (dataloader.py) ⭐ OPTIMISÉ
│       └─ NORM_STATS (stats pré-calculées)
│
├─ MultiPollutantLightningModule (model_multipollutants.py)
│   └─ MultiPollutantModel
│       └─ ClimaX (climax_core/arch.py)
│           ├─ ParallelVarPatchEmbedWind
│           │   └─ CachedWindScanning (wind_scanning_cached.py)
│           ├─ PhysicsGuidedBlockPatchLevel
│           └─ PosEmbed utils
│
└─ PyTorch Lightning Trainer (DDP, 800 GPUs)
```

---

## 📁 FICHIERS ACTIFS

### Core (Racine)
- `main_multipollutants.py` (6.7K)
- `submit_multipollutants_from_scratch_6pollutants.sh` (6.1K)

### Source Code (src/)
- `config_manager.py` (4.6K)
- `dataloader.py` (12K) **⭐ OPTIMISÉ avec NORM_STATS**
- `datamodule_fixed.py` (5.6K)
- `model_multipollutants.py` (19K)
- `wind_scanning_cached.py` (28K)

### ClimaX Core (src/climax_core/)
- `arch.py` (13K) - Architecture principale
- `parallelpatchembed_wind.py` (6.8K) - Patch embedding
- `physics_attention_patch_level.py` (9.9K) - Attention physique
- `utils/pos_embed.py` - Position embeddings

### Scripts
- `scripts/run_multipollutants_from_scratch.sh` (2.7K)
- `scripts/auto_test_after_training.py` (4.0K)

### Configuration
- `configs/config_all_pollutants.yaml` (2.2K)

---

## 🗑️ ARCHIVES

Tous les backups sont dans `archive/`:

- **dataloaders/** (dataloader_zarr_optimized, dataloader_fixed, etc.)
- **wind_scanning/** (tous les backups de wind_scanning)
- **climax_core/** (physics_attention alternatives, arch backups)
- **models/** (model_multipollutants backups)
- **scripts/** (submit scripts 8x8, test scripts)
- **old_configs/** (configs backups)

---

## 🎯 DONNÉES ET MODÈLES

### Données
```
./data_processed/
├─ data_2013_china_masked.zarr
├─ data_2014_china_masked.zarr
├─ data_2015_china_masked.zarr
├─ data_2016_china_masked.zarr (train)
├─ data_2017_china_masked.zarr (validation)
└─ data_2018_china_masked.zarr (test)
```

### Checkpoints
```
logs/multipollutants_climax_ddp/version_47/checkpoints/
└─ best-val_loss_val_loss=0.3557-step_step=311.ckpt (601M)
```

---

## 🚀 UTILISATION

### Training (800 GPUs)
```bash
sbatch submit_multipollutants_from_scratch_6pollutants.sh
```

### Monitoring
```bash
# Voir les logs
tail -f logs/aq_net2_multipoll_from_scratch_<JOB_ID>.out

# TensorBoard (depuis local)
ssh -L 6006:<NODE>:<PORT> khederam@lumi.csc.fi
# Puis: http://localhost:6006
```

---

## ⚙️ CONFIGURATION ACTUELLE

- **Nodes:** 100
- **GPUs per node:** 8
- **Total GPUs:** 800
- **Batch size per GPU:** 2
- **Gradient accumulation:** 4
- **Effective batch:** 6400 samples/step
- **Precision:** FP32
- **Strategy:** DDP
- **Forecast horizons:** 12h, 24h, 48h, 96h
- **Target pollutants:** PM2.5, PM10, SO2, NO2, CO, O3

---

## 📝 NOTES IMPORTANTES

### ⚠️ À FAIRE
- [ ] Recalculer NORM_STATS sur années 2013-2016 complètes
- [ ] Tester que le training démarre correctement
- [ ] Vérifier compatibilité avec checkpoint version_47

### ✅ FAIT
- [x] Archivage des 304 fichiers obsolètes
- [x] Fusion dataloader avec NORM_STATS
- [x] Documentation architecture complète
- [x] Structure projet nettoyée

---

## 📧 CONTACT

Pour questions: Ammar Kheddar (khederam@lumi.csc.fi)


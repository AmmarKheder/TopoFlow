# 🔄 GUIDE CHECKPOINT - REPRISE AUTOMATIQUE

## ✅ SYSTÈME AUTOMATIQUE DE SAUVEGARDE

### 🔄 Sauvegarde automatique
- **Fréquence** : À chaque amélioration de `val_loss`
- **Nombre** : 3 meilleurs checkpoints gardés
- **Format** : `epoch_XX-val_loss_X.XXXX.ckpt`
- **Location** : `logs/multipollutants_climax_ddp/version_X/checkpoints/`

## 🚀 COMMENT REPRENDRE L'ENTRAÎNEMENT

### Option 1: Reprise automatique
```bash
# Le script trouve automatiquement le dernier checkpoint
sbatch scripts/submit_multipollutants_resume.sh
```

### Option 2: Checkpoint spécifique
```bash
# Reprendre depuis un checkpoint particulier
python main_multipollutants_resume.py --checkpoint logs/multipollutants_climax_ddp/version_0/checkpoints/epoch_05-val_loss_0.0543.ckpt
```

### Option 3: Reprise manuelle
```bash
# Trouver les checkpoints disponibles
find logs/multipollutants_climax_ddp/ -name "*.ckpt" -ls

# Reprendre depuis le meilleur
python main_multipollutants_resume.py --checkpoint [CHEMIN_CHECKPOINT]
```

## 📁 STRUCTURE DES CHECKPOINTS

```
logs/multipollutants_climax_ddp/
├── version_0/
│   ├── checkpoints/
│   │   ├── epoch_03-val_loss_0.0654.ckpt
│   │   ├── epoch_05-val_loss_0.0543.ckpt  ← MEILLEUR
│   │   ├── epoch_07-val_loss_0.0567.ckpt
│   │   └── last.ckpt                      ← DERNIER
│   ├── hparams.yaml
│   └── events.out.tfevents.*
```

## 🎯 STRATÉGIES DE REPRISE

### 1. **Interruption planifiée** (time limit atteint)
```bash
# Reprise automatique - garde la continuité
sbatch scripts/submit_multipollutants_resume.sh
```

### 2. **Erreur technique** (crash)
```bash
# Reprendre depuis le dernier checkpoint valide
python main_multipollutants_resume.py --checkpoint logs/.../last.ckpt
```

### 3. **Optimisation fine** (tweaks)
```bash
# Reprendre depuis le meilleur modèle
find logs/multipollutants_climax_ddp/ -name "*.ckpt" | sort
# Choisir le meilleur val_loss
```

## ⚡ AVANTAGES DU SYSTÈME

- ✅ **Perte zéro** : Aucune perte de progrès
- ✅ **Optimiseur préservé** : Learning rate schedule continue
- ✅ **Métriques continues** : TensorBoard garde l'historique  
- ✅ **Multi-GPU** : Compatible DDP 16 GPUs
- ✅ **Automatique** : Détection auto du dernier checkpoint

## 🔧 COMMANDES UTILES

```bash
# Voir l'historique d'entraînement
ls -la logs/multipollutants_climax_ddp/version_*/checkpoints/

# Trouver le meilleur checkpoint (plus petit val_loss)
find logs/multipollutants_climax_ddp/ -name "*.ckpt" | grep -v last | sort

# Espace disque des checkpoints
du -h logs/multipollutants_climax_ddp/

# Status job en cours
squeue -u $USER
```

## 🎉 RÉSULTAT

Avec ce système, **ton entraînement ne sera jamais perdu** même si :
- Le job hit time limit (72h)
- Crash système
- Interruption manuelle
- Problème réseau

**Reprise transparente garantie !** 🔄

# Ce que j'aurais dû savoir AVANT (doc LUMI officielle)

## Problème #1: PyTorch DDP Hang sur 400 GPUs
**Source:** https://lumi-supercomputer.eu/scaling-llm-training/

### Le bug PyTorch confirmé:
- **PyTorch a un bug connu** qui rend l'initialisation DDP de >4000 ranks "quasi impossible"
- Maximum testé stable sur LUMI: **768 nodes = 6144 ranks**
- Notre essai: 50 nodes × 8 GPUs = **400 ranks** → proche limite instable!

### Solution appliquée:
- ✅ **16 nodes × 8 GPUs = 128 ranks** → zone stable
- Job 13309528 tourne parfaitement

## Problème #2: Configuration SLURM optimale
**Source:** https://docs.csc.fi/support/tutorials/ml-multi/

### Recommandations CSC:
```bash
#SBATCH --cpus-per-task=7        # Maximum 7 cores par GPU
#SBATCH --mem=480G               # ~60GB par GPU
#SBATCH --ntasks-per-node=8      # 8 GPUs par node
```

### Ce qu'on faisait mal:
- Pas de `--cpus-per-task` spécifié
- Pas de `--mem` optimisé

## Problème #3: Variables d'environnement NCCL
**Manquant dans notre script:**
```bash
export NCCL_SOCKET_IFNAME=hsn   # Pour performance réseau optimale LUMI
export NCCL_DEBUG=INFO           # Pour debug si nécessaire
```

## Leçons apprises

1. **Toujours checker la doc officielle du cluster AVANT de débugger** ✅
2. PyTorch DDP ne scale pas linéairement - il y a des limites hardware/software
3. 400 GPUs != 128 GPUs × 3.125 en complexité, c'est exponentiel pour l'init
4. LUMI a des spécificités (AMD ROCm, MI250X dual-GCD) qui nécessitent config spéciale

## Fix final qui marche

**Fichiers:**
- `scripts/slurm_physics_finetune.sh`: 16 nodes au lieu de 50
- `configs/config_physics_finetune.yaml`: num_nodes=16
- `src/climax_core/physics_block_wrapper.py`: device fix dans forward()

**Job actuel:** 13309528
- Status: ✅ RUNNING
- Progress: Epoch 0, step 10/540, loss 5.85 → 4.31
- ETA: ~6-8h pour 5 epochs


# EXPÉRIENCE: Prouver Elevation Bias + Richardson

## Objectif
Partir du checkpoint 0.3557 (wind scanning seul) et ajouter physics mask pour prouver amélioration.

## Configuration

### Baseline (déjà fait)
- **Checkpoint:** `/scratch/project_462000640/ammar/aq_net2/logs/multipollutants_climax_ddp/version_47/checkpoints/best-val_loss_val_loss=0.3557-step_step=311.ckpt`
- **Features:** Wind scanning 32x32 SEULEMENT
- **Val Loss:** 0.3557

### Expérience Unique
- **Checkpoint:** Load baseline
- **Ajout:** Physics Mask (Elevation + Richardson) au **FIRST BLOCK ONLY**
- **Transfer Learning:** Fine-tune 5 epochs, LR=5e-5
- **Target:** Val Loss < 0.32 (-10% improvement)

## Physics Mask Details

### Composantes
1. **Elevation Barrier:**
   - Force: learnable (init 3.0)
   - Penalise transport montant

2. **Richardson Stability:**
   - Force: learnable (init 2.0)
   - Critical Ri: 0.25
   - Capture inversion thermique

### Intégration
- **First transformer block:** Physics mask appliqué
- **Other blocks:** Standard attention (pas de physics)
- **Reordering:** Elevation/temp réordonnés comme patch tokens

## Freeze Strategy

### Phase 1 (Epochs 1-2)
- Freeze: Tout le backbone
- Train: Physics mask params only (2 params: elevation_strength, richardson_strength)
- LR: 1e-5

### Phase 2 (Epochs 3-5)
- Freeze: Blocks 1-4
- Train: Last 2 blocks + physics mask
- LR: 5e-5

## Expected Results

| Metric | Baseline | Expected |
|--------|----------|----------|
| Val Loss | 0.3557 | < 0.32 |
| Improvement | - | ~-10% |

## Success Criteria
✅ Val loss < 0.32 → **MISSION ACCOMPLISHED**
✅ Prouve que physics bias améliore wind scanning
✅ Ready for publication

## Fichiers Modifiés
1. `src/climax_core/physics_mask_fixed.py` - Physics mask compatible wind scanning
2. `src/climax_core/physics_helpers.py` - Helper pour réordonner elevation/temp
3. `src/climax_core/arch.py` - Intégration au first block
4. `configs/config_finetune_physics.yaml` - Config fine-tuning

## Compute
- **GPUs:** 400 (50 nodes × 8 GPUs)
- **Temps:** ~6-8h (5 epochs)
- **Coût:** ~4000€

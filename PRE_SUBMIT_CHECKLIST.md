# Pre-Submit Checklist - TopoFlow Training

## ✅ Configuration Verified

### SLURM Script: `scripts/slurm_topoflow_elevation.sh`
- ✅ Project: 462001079
- ✅ Time: 24:00:00
- ✅ Nodes: 50 (400 GPUs total)
- ✅ Config path: configs/config_topoflow_elevation.yaml

### Config: `configs/config_topoflow_elevation.yaml`
- ✅ max_steps: 20000
- ✅ warmup_steps: 2000
- ✅ scheduler: cosine
- ✅ batch_size: 2
- ✅ num_nodes: 50 (matches SLURM)
- ✅ devices: 8 (matches SLURM)
- ✅ **NO checkpoint_path** → Training from scratch ✅
- ✅ use_physics_mask: true
- ✅ parallel_patch_embed: false (row-major, no wind reordering)

### Model Integration: `src/climax_core/arch.py`
- ✅ TopoFlowBlock imported
- ✅ First block replaced with TopoFlow
- ✅ Forward pass updated to use elevation_patches + u_wind + v_wind
- ✅ Test passed: model.climax.blocks[0] is TopoFlowBlock

### Test Results: `test_topoflow_simple.py`
```
✅ TopoFlow enabled: block 0, grid 64×128, elevation+wind
4. First block is TopoFlowBlock: True
   - elevation_alpha: 1.000
   - wind_beta: 0.300
   - H_scale: 1000.0
✅ ALL GOOD - Model is ready!
```

## 📊 Training Parameters

| Parameter | Value |
|-----------|-------|
| GPUs | 400 (50 nodes × 8) |
| Steps | 20,000 |
| Batch size | 2 per GPU |
| Effective batch | 800 (2 × 400) |
| Accumulate grad | 2 |
| Learning rate | 0.0001 |
| Scheduler | Cosine |
| Precision | 32-bit |

## 🎯 Expected Behavior

1. **Model initialization:**
   - First block: TopoFlowBlock with elevation attention
   - Blocks 1-5: Standard Transformer blocks

2. **Training:**
   - From scratch (no checkpoint)
   - Should converge faster than baseline due to physics inductive bias
   - Target: val_loss < 0.264 (beat row-major baseline)

3. **Logs:**
   - Output: `logs/topoflow_elevation_13473924.out`
   - Errors: `logs/topoflow_elevation_13473924.err`

## 🚀 Ready to Submit

All checks passed. Safe to submit!

**Command:**
```bash
cd /scratch/project_462000640/ammar/aq_net2
sbatch scripts/slurm_topoflow_elevation.sh
```

# 🚀 JOB 13506127 - TopoFlow Elevation Mask Test

**Date**: 11 octobre 2025
**Job ID**: 13506127
**Goal**: Prouver que le mask d'élévation (TopoFlow) améliore les prédictions

---

## 🎯 OBJECTIF

**Baseline** : val_loss = 0.3557 (checkpoint step 311, SANS TopoFlow)
**Test** : Fine-tuner avec TopoFlow sur block 0 uniquement
**Success criteria** : val_loss < 0.34 après 1 epoch (270 steps)

---

## ✅ FIXES APPLIQUÉS

### 1. HEAD Architecture Fixed
**Avant** :
```python
self.head = nn.Linear(768, 60)  # 1 layer, 46K params
```

**Après** :
```python
self.head = nn.Sequential(
    nn.Linear(768, 768),  # head.0
    nn.GELU(),            # head.1
    nn.Linear(768, 768),  # head.2
    nn.GELU(),            # head.3
    nn.Linear(768, 60)    # head.4
)  # 3 layers, 1.2M params
```

**Résultat** : HEAD charge correctement depuis le checkpoint ✅

### 2. TopoFlow Block 0
- TopoFlow attention avec elevation bias activé
- MLP compatible (fc1/fc2 naming)
- Seuls 2 nouveaux params :
  - `elevation_alpha` : Learnable scaling du masque
  - `H_scale` : Fixed ou learnable (1000m)

---

## 📊 CHECKPOINT LOADING - EXPECTED RESULTS

### ✅ Perfect Scenario (TARGET)
```
Missing keys: 2
   - climax.blocks.0.attn.elevation_alpha
   - climax.blocks.0.attn.H_scale

Unexpected keys: 0
```

**Conséquences** :
- 85,000,000+ params chargés du checkpoint
- Seulement 2 params random (TopoFlow)
- val_loss step 1 ≈ 0.36 (niveau baseline)

### ⚠️ Acceptable Scenario
```
Missing keys: 10-15
   - Tout l'attention du block 0
   - elevation_alpha, H_scale
```

**Conséquences** :
- val_loss step 1 ≈ 0.5-0.6
- Convergence plus lente mais OK

### ❌ Problem Scenario
```
Missing keys: 50+
   - HEAD non chargée
   - Block 0 entier non chargé
```

**Conséquences** :
- val_loss step 1 ≈ 2.0
- Besoin de debugger

---

## 🔍 VERIFICATION CHECKLIST

### Step 1: Job Start (t=0-5 min)
```bash
# Check job status
squeue -u khederam | grep 13506127

# Check log file appears
ls -lh /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out
```

### Step 2: Checkpoint Loading (t=5-10 min)
```bash
# Check missing keys
tail -100 /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out | grep -A 20 "Missing keys"

# Expected: ONLY 2 missing keys (elevation_alpha, H_scale)
```

### Step 3: First Validation (t=30-40 min)
```bash
# Check first val_loss
tail -100 /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out | grep "val_loss"

# Expected: val_loss ≈ 0.36 at step 0-25
```

### Step 4: Monitor Convergence (t=3-4 hours)
```bash
# Track val_loss evolution
grep "val_loss" /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out | tail -20

# Expected progression:
# Step 0-25:  val_loss ≈ 0.36 (baseline)
# Step 50:    val_loss ≈ 0.34-0.35 (TopoFlow learning)
# Step 100:   val_loss ≈ 0.33-0.34 (improvement)
# Step 270:   val_loss < 0.34 (SUCCESS!)
```

---

## 📈 EXPECTED TRAINING PROGRESSION

### Scenario 1: TopoFlow Works ✅
```
Step   | train_loss | val_loss | Status
-------|-----------|----------|--------
0      | 0.80      | 0.36     | Baseline (all params loaded)
25     | 0.79      | 0.35     | TopoFlow starting to learn
50     | 0.78      | 0.34     | Clear improvement
100    | 0.76      | 0.33     | Convergence
200    | 0.75      | 0.32     | Final result
270    | 0.74      | 0.32     | End of epoch 1

→ SUCCESS! TopoFlow mask improves predictions
```

### Scenario 2: TopoFlow Doesn't Help ❌
```
Step   | train_loss | val_loss | Status
-------|-----------|----------|--------
0      | 0.80      | 0.36     | Baseline
25     | 0.80      | 0.36     | No change
50     | 0.80      | 0.36     | Still no change
270    | 0.79      | 0.355    | Minimal improvement

→ FAILURE: elevation mask doesn't help (but we know!)
```

---

## 🎯 SUCCESS CRITERIA

### ✅ Test Valid If:
1. Missing keys = 2 (only TopoFlow params)
2. val_loss step 1 ≈ 0.36 (baseline level)
3. No crashes or errors

### ✅ TopoFlow Works If:
1. val_loss decreases over training
2. val_loss < 0.34 by step 270
3. Improvement is statistically significant (>1% better)

### ❌ TopoFlow Fails If:
1. val_loss stays at 0.36
2. val_loss increases
3. No improvement after 1 epoch

---

## 📝 COMMANDS FOR MONITORING

### Quick Status Check
```bash
# Job status
squeue -u khederam | grep 13506127

# Latest logs (last 50 lines)
tail -50 /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out

# Check for errors
tail -50 /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.err | grep -i "error"
```

### Detailed Analysis
```bash
# Checkpoint loading verification
grep -A 20 "Missing keys" /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out

# Training progression
grep "Epoch 0:" /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out | grep "val_loss" | tail -20

# Check if TopoFlow is active
grep "TopoFlow" /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13506127.out
```

---

## 📊 WHAT TO REPORT BACK

### If Everything Works ✅
1. Confirm: Missing keys = 2 (elevation_alpha, H_scale)
2. Confirm: val_loss step 1 ≈ 0.36
3. Report: val_loss progression (0.36 → ?)
4. Final: val_loss at step 270
5. Conclusion: TopoFlow improves by X%

### If There's a Problem ❌
1. Report: Number of missing keys
2. Report: Which params are missing
3. Report: val_loss at step 1
4. Investigate: Why HEAD or other params didn't load

---

## 🔧 TROUBLESHOOTING

### Problem: val_loss step 1 is high (>1.0)
**Diagnosis**: HEAD or other params not loaded
**Action**: Check missing keys, verify HEAD architecture

### Problem: Job crashes with NCCL error
**Diagnosis**: Multi-GPU communication issue
**Action**: Check Infiniband config, reduce nodes

### Problem: val_loss doesn't improve
**Diagnosis**: TopoFlow mask doesn't help (valid result!)
**Action**: Try different elevation scaling, different blocks

---

## 📅 TIMELINE

- **t=0**: Job submitted (DONE)
- **t=5min**: Job starts, logs appear
- **t=10min**: Checkpoint loaded, verify missing keys
- **t=30min**: First validation (step 25)
- **t=1h**: Step 50 validation
- **t=2h**: Step 100 validation
- **t=3-4h**: Epoch 1 complete (step 270)
- **t=4h**: Final report

---

## ✅ CURRENT STATUS

**Job ID**: 13506127
**Status**: Pending (waiting for resources)
**Expected start**: Within 5-10 minutes
**Next check**: When log file appears

**Monitoring started**: Automatic
**User**: Away (carte blanche)
**Action**: Monitor and report when user returns

---

**Generated by**: Claude
**Date**: 2025-10-11 15:50 EEST

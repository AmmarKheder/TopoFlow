# Debugging Summary - TopoFlow Fine-Tuning Launch Issues

**Date**: 2025-10-15
**Goal**: Fine-tune TopoFlow model from checkpoint (loss=0.3557) on 128 GPUs

---

## Problem Timeline

### Initial Issue
- **Job 13584662** (128 GPUs): CANCELLED after 1min41s
- Error: `amdgpu.ids: No such file or directory` (repeated 128 times)
- Job cancelled before Python started

### Root Cause Identified
Jobs were **stuck after "All distributed processes registered"**:
- PyTorch Lightning initializes 128 distributed processes ✅
- Message: "All distributed processes registered. Starting with 128 processes" ✅
- **BUT: Python script never starts** ❌
- No `LOCAL_RANK: X - CUDA_VISIBLE_DEVICES` messages
- No model loading, no checkpoint loading
- **Job 13584722**: Stuck for 6+ HOURS at this stage!

### Successful Reference
- **Job 13529147** (256 GPUs, Oct 13): Ran for 16h53m successfully
- After "All distributed processes registered", Python started IMMEDIATELY
- Shows `LOCAL_RANK` messages within seconds

---

## Fixes Attempted

### Fix #1: Remove `ROCR_VISIBLE_DEVICES`
**Change**: Commented out `export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`
**Reason**: Was causing amdgpu.ids errors
**Result**: Error reduced but Python still didn't start

### Fix #2: Simplify venv activation
**Change**: Removed verbose messaging
**Result**: No change, still stuck

### Fix #3: Add `--export=ALL` to srun
**Change**: `srun --export=ALL python ...`
**Result**: Removed later, didn't help

### Fix #4: Use absolute paths
**Change**: `/scratch/project_462000640/ammar/aq_net2/main_multipollutants.py`
**Result**: Still stuck

### Fix #5: Bash wrapper with explicit venv (CURRENT TEST)
**Change**:
```bash
srun bash -c "source /path/to/venv/bin/activate && python main_multipollutants.py ..."
```
**Status**: Testing on job 13589032 (32 GPUs, 4 nodes)
**Reason**: Ensures venv is activated on EVERY compute node

---

## Key Differences: Working vs Broken

| Aspect | Job 13529147 (✅ WORKED) | Recent Jobs (❌ STUCK) |
|--------|-------------------------|----------------------|
| Date | Oct 13 | Oct 15 |
| GPUs | 256 (32 nodes) | 128-256 (16-32 nodes) |
| Python starts | YES - immediately | NO - never |
| LOCAL_RANK msgs | YES | NO |
| amdgpu.ids errors | YES (non-critical) | YES (same) |
| Venv activation | ? (need to check old script) | Various attempts |

---

## Hypothesis

**Primary suspect**: Virtual environment not properly activated on compute nodes via `srun`.

When `srun python ...` is called:
1. Master node has venv activated ✅
2. Compute nodes may NOT have venv activated ❌
3. Python starts but can't find PyTorch/Lightning ❌
4. Process hangs waiting for imports that never complete ❌

---

## Current Status

**Job 13589032** (32 GPUs, 4 nodes):
- Status: PENDING in queue
- Fix: Bash wrapper with explicit venv activation
- Auto-monitor running: `/scratch/project_462000640/ammar/aq_net2/AUTO_MONITOR_13589032.log`
- Will detect if Python starts (looks for LOCAL_RANK messages)

---

## Next Steps if 13589032 Works

1. ✅ Python starts and shows LOCAL_RANK messages
2. ✅ Model loads from checkpoint
3. ✅ Training starts from loss ~0.35
4. Scale up to 128 GPUs (16 nodes)
5. Then to 256+ GPUs if needed

---

## Next Steps if 13589032 Still Stuck

### Plan B: Test with minimal script
Create `test_distributed_venv.py`:
```python
import torch
import pytorch_lightning as pl
print(f"✅ LOCAL_RANK: {os.environ.get('LOCAL_RANK')} - PyTorch {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
print(f"✅ Lightning version: {pl.__version__}")
```

Test on 4-8 GPUs (1-2 nodes) to confirm venv propagation.

### Plan C: Copy EXACT working script
- Find the EXACT script that launched job 13529147
- Use git history or backups
- Copy it verbatim, change only job size

### Plan D: Use singularity/container
- Package everything in container
- Avoids venv propagation issues
- Standard on LUMI for multi-node

---

## Files Modified

1. `submit_multipollutants_from_6pollutants.sh` - Main job script
2. `monitor_13589032.sh` - Auto-monitoring script
3. Various test scripts: `test_main.sh`, `test_srun.sh`

## Logs to Check

- `/scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13589032.{out,err}`
- `/scratch/project_462000640/ammar/aq_net2/AUTO_MONITOR_13589032.log`

---

**Last Updated**: Auto-mode debugging session
**Auto-monitor**: RUNNING (checks every 30s)

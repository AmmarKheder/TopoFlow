# Critical Fixes Applied - October 11, 2025

## Problem Analysis

Job `13494914` (topoflow_full_finetune) crashed with:
- **Segmentation fault** on node `nid007365`, task 352/400
- **pthread_mutex_unlock: No such file or directory**
- Crash occurred right after initializing all 400 distributed processes (NCCL)

### Root Causes Identified

1. **Infiniband disabled for 400 GPUs** - Forces unstable TCP/IP communication
2. **`find_unused_parameters` not actually used** - Config had it but code ignored it
3. **Incomplete DDPStrategy configuration** - Using string instead of DDPStrategy object

---

## Fixes Applied

### 1. Re-enabled Slingshot/Infiniband (CRITICAL)

**File:** `submit_multipollutants_from_6pollutants.sh`

**Changes:**
- ✅ Re-enabled Slingshot (LUMI's Infiniband) for optimal 50-node communication
- ✅ Added proper NCCL socket interface configuration
- ✅ Commented out problematic `NCCL_IB_DISABLE=1`
- ✅ Changed `NCCL_DEBUG=WARN` to `INFO` for better diagnostics
- ✅ Added AMD-specific RCCL optimizations

**Before:**
```bash
export NCCL_IB_DISABLE=1
export RCCL_IB_DISABLE=1
export NCCL_NET_PLUGIN=none
```

**After:**
```bash
# Enable Slingshot for optimal multi-node communication
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export RCCL_MSCCL_ENABLE=0
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=8

# Commented out (these were causing pthread/segfault issues):
# export NCCL_IB_DISABLE=1
# export RCCL_IB_DISABLE=1
# export NCCL_NET_PLUGIN=none
```

### 2. Fixed DDPStrategy Configuration (CRITICAL)

**File:** `main_multipollutants.py`

**Changes:**
- ✅ Imported `DDPStrategy` from pytorch_lightning.strategies
- ✅ Created proper DDPStrategy object with `find_unused_parameters`
- ✅ Added 2-hour timeout to DDPStrategy (matching NCCL timeout)
- ✅ Added `max_steps` support to Trainer
- ✅ Increased `num_sanity_val_steps` from 1 to 2

**Before:**
```python
trainer = pl.Trainer(
    ...
    strategy=config['train']['strategy'],  # Just string "ddp"
    ...
)
```

**After:**
```python
from datetime import timedelta
from pytorch_lightning.strategies import DDPStrategy

# Create DDPStrategy with find_unused_parameters from config
find_unused = config['train'].get('find_unused_parameters', False)

if config['train']['strategy'] == 'ddp' and config["train"]["num_nodes"] > 1:
    strategy = DDPStrategy(
        find_unused_parameters=find_unused,
        timeout=timedelta(seconds=7200)  # 2 hours for 400 GPUs (MUST be timedelta!)
    )
else:
    strategy = config['train']['strategy']

trainer = pl.Trainer(
    ...
    strategy=strategy,  # Proper DDPStrategy object
    max_steps=config['train'].get('max_steps', -1),
    ...
)
```

**Note:** The timeout parameter MUST be a `timedelta` object, not an integer. This was causing a TypeError in the first submission attempt (job 13503025).

### 3. Updated Configuration

**File:** `configs/config_all_pollutants.yaml`

**Changes:**
- ✅ Clarified that `find_unused_parameters` is now properly used
- ✅ Increased `epochs: 10` → `epochs: 100` to let `max_steps` control stopping
- ✅ Added comments explaining precedence

**Before:**
```yaml
  epochs: 10
  max_steps: 20000
```

**After:**
```yaml
  max_steps: 20000  # Primary stopping criterion (20k steps)
  epochs: 100  # Backup: increased to avoid early stopping (steps take precedence)
```

---

## Expected Impact

### Why These Fixes Should Work

1. **Infiniband Re-enabled:**
   - LUMI's Slingshot interconnect is designed for 50+ node jobs
   - Provides low-latency, high-bandwidth communication
   - Solves the pthread mutex errors caused by TCP/IP saturation

2. **Proper DDP Configuration:**
   - `find_unused_parameters=true` now actually active
   - Prevents "unused parameter" errors with TopoFlow elevation mask
   - 2-hour timeout prevents premature termination during initialization

3. **Better Diagnostics:**
   - `NCCL_DEBUG=INFO` will show detailed communication patterns
   - Can identify bottlenecks if issues persist

### Remaining Risks

- 400 GPUs is still massive - initialization overhead ~5-10 minutes expected
- First few steps will be slow as NCCL builds communication topology
- Watch for memory issues during data loading (consolidated zarr on 400 ranks)

---

## Verification Steps

After launching the job, check:

1. **Initialization completes:**
   ```bash
   tail -f logs/topoflow_full_finetune_<JOBID>.err
   # Should see: "All distributed processes registered. Starting with 400 processes"
   # WITHOUT segfault
   ```

2. **NCCL communication works:**
   ```bash
   grep "NCCL INFO" logs/topoflow_full_finetune_<JOBID>.err | head -20
   # Should show network topology and ring configuration
   ```

3. **Training starts:**
   ```bash
   tail -f logs/topoflow_full_finetune_<JOBID>.out
   # Should see training steps progressing
   ```

---

## Rollback Plan

If job still crashes:

1. **Reduce to 25 nodes (200 GPUs):**
   ```bash
   #SBATCH --nodes=25
   ```

2. **Or try intermediate 128 GPUs (proven stable):**
   ```bash
   #SBATCH --nodes=16
   ```

3. **Disable find_unused_parameters as last resort:**
   ```yaml
   find_unused_parameters: false
   ```

---

## Testing Commands

**Check queue:**
```bash
squeue -u $USER
```

**Monitor job:**
```bash
watch -n 5 'squeue -u $USER'
```

**View logs:**
```bash
tail -f aq_net2/logs/topoflow_full_finetune_*.err
tail -f aq_net2/logs/topoflow_full_finetune_*.out
```

**Launch job:**
```bash
cd /scratch/project_462000640/ammar/aq_net2
sbatch submit_multipollutants_from_6pollutants.sh
```

---

## Files Modified

1. ✅ `submit_multipollutants_from_6pollutants.sh` - NCCL/Infiniband fixes
2. ✅ `main_multipollutants.py` - DDPStrategy configuration
3. ✅ `configs/config_all_pollutants.yaml` - Epochs adjustment
4. ✅ `FIXES_APPLIED_OCT11.md` (this file) - Documentation

---

## References

- LUMI docs: https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/
- PyTorch Lightning DDP: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.DDPStrategy.html
- NCCL env variables: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Previous issues: `TRAINING_ISSUES.md`

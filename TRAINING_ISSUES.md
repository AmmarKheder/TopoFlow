# TopoFlow Training Issues & Solutions (LUMI Supercomputer)

## Summary

This document chronicles the debugging process for training TopoFlow on LUMI's AMD MI250X GPUs with PyTorch Lightning DDP. Multiple critical issues were encountered and resolved, with one remaining incompatibility between wind scanning and multi-GPU training.

---

## Issue 1: Git Push Blocked by Large Core File

### Error
```
File core is 1102.61 MB; this exceeds GitHub's file size limit of 100.00 MB
```

### Root Cause
A 1.1GB core dump file was committed to git history, blocking all pushes to GitHub.

### Solution
```bash
# Remove core file from entire git history
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch core' --prune-empty --tag-name-filter cat -- --all

# Add to .gitignore
echo "core" >> .gitignore

# Force push to clean history
git push --force
```

### Prevention
Always add core dumps to `.gitignore` before committing.

---

## Issue 2: DataLoader prefetch_factor with num_workers=0

### Error
```
ValueError: prefetch_factor option could only be specified in multiprocessing.
let num_workers > 0
```

### Root Cause
PyTorch DataLoader cannot use `prefetch_factor` when `num_workers=0` (single-process mode), but our datamodule unconditionally set this parameter.

### Solution
Modified `src/datamodule_fixed.py` to conditionally include multiprocessing parameters:

```python
def train_dataloader(self):
    loader_kwargs = {
        'batch_size': self.batch_size,
        'shuffle': True,
        'num_workers': self.num_workers,
        'pin_memory': True,
        'persistent_workers': False,
        'drop_last': True,
    }

    # Only add multiprocessing params when using workers
    if self.num_workers > 0:
        loader_kwargs['multiprocessing_context'] = "spawn"
        loader_kwargs['prefetch_factor'] = self.data_config.get('prefetch_factor', 2)

    return DataLoader(self.train_dataset, **loader_kwargs)
```

### Impact
Critical for both single-GPU debugging and multi-GPU training with `num_workers=0`.

---

## Issue 3: MIOpen Convolution Algorithm Failures

### Error
```
MIOpen Error: No suitable algorithm was found to execute the required convolution
RuntimeError: miopenStatusUnknownError
```

### Root Causes
1. **Initial mistake:** Set `MIOPEN_DISABLE_CACHE=1`, preventing kernel caching entirely
2. **SQLite locking:** Multiple nodes writing to same MIOpen cache database caused DB locks
3. **Missing kernel compilation:** MIOpen needs to compile kernels on first run

### Failed Approaches
- Shared cache path: `/scratch/project_462000640/ammar/miopen_cache` → SQLite DB locking across nodes
- Disabling cache entirely → Kernel compilation failures

### Working Solution
Use **per-node temporary cache** in SLURM script:

```bash
# MIOpen cache - per-node cache to avoid DB locking
export MIOPEN_USER_DB_PATH=/tmp/miopen_cache_$SLURM_JOB_ID
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH

# Let MIOpen find/compile algorithms on first run
export MIOPEN_FIND_MODE=1
export MIOPEN_LOG_LEVEL=3
```

### Key Points
- Each node gets isolated SQLite database: `/tmp/miopen_cache_<JOB_ID>/`
- `MIOPEN_FIND_MODE=1` enables kernel search/compilation
- Kernels compile during first epoch (expected ~10-30min overhead)
- Cache persists for job duration, cleaned up automatically after job ends

---

## Issue 4: DDP Deadlock with Wind Scanning ⚠️ **UNRESOLVED**

### Error
Jobs hang indefinitely after:
```
Fine-tuned optimizer configuration:
  - AdamW optimizer...
```

Or crash with `SIGTERM` after 1+ hour of initialization without starting training.

### Root Cause
**Wind scanner cache initialization happens at first forward pass**, causing DDP synchronization deadlock:

1. All DDP ranks wait for first batch
2. Rank 0 initializes wind scanner cache (loads 1024 region masks)
3. Other ranks wait indefinitely for rank 0
4. **Deadlock** → job hangs or times out

### Failed Solutions

#### Attempt 1: Load cache before DDP
```python
# In model __init__ before lightning setup
if self.parallel_patch_embed:
    _ = self.patch_embed(torch.zeros(1, in_chans, *img_size))
```
**Result:** Still hangs - DDP already initialized by Lightning

#### Attempt 2: Broadcast approach
```python
# Rank 0 initializes, then broadcast to all ranks
if dist.get_rank() == 0:
    self.wind_scanner.initialize_cache()
dist.barrier()
# Broadcast cache to other ranks...
```
**Result:** Still hangs - implementation issues with cache broadcasting

#### Attempt 3: Reduce GPU count
Tried 400 → 80 → 16 GPUs
**Result:** Still hangs on all multi-GPU configurations

### Current Workaround
**Disable wind scanning for multi-GPU training:**

```yaml
# configs/config_wind_baseline.yaml
model:
  parallel_patch_embed: false  # DISABLE wind scanning for DDP
```

### Verified Behavior
- ✅ **1 GPU + wind scanning enabled:** Works perfectly
- ✅ **128 GPUs + wind scanning disabled:** Works, training stable
- ❌ **Multi-GPU + wind scanning enabled:** Deadlock

### Impact
~~**Critical limitation:** Cannot test TopoFlow's main innovation (wind-following scan order) on multi-GPU setup.~~

**✅ RESOLVED!** See Issue 6 below for the final solution.

### Attempted Solutions (Failed)
1. ❌ Load cache before DDP → Still hangs
2. ❌ Broadcast approach → Implementation issues
3. ❌ Reduce GPU count → Still hangs on all multi-GPU configs

---

## Issue 5: Job Initialization Delays

### Error
400 GPU and 80 GPU jobs took 1+ hour to initialize without starting training.

### Root Causes
1. DDP initialization overhead scales with GPU count
2. Dataset loading on all ranks (especially with consolidated zarr)
3. Wind cache deadlock (see Issue 4)

### Solution
1. Reduced to manageable GPU counts (16 nodes = 128 GPUs)
2. Fixed underlying wind scanner issue (disabled for now)
3. Set `num_workers=0` to avoid multiprocessing overhead during debugging

### Optimal Configuration
For 128 GPUs (16 nodes × 8 GPUs):
- `batch_size: 2` per GPU
- `accumulate_grad_batches: 4` → effective batch size = 1024
- `num_workers: 0` (for stability, can increase after debugging)
- `strategy: ddp`
- `find_unused_parameters: false`

---

## Working Configurations

### 1 GPU (Wind Scanning Enabled)
```yaml
train:
  devices: 1
  num_nodes: 1
  batch_size: 2
  num_workers: 0

model:
  parallel_patch_embed: true  # Wind scanning works on 1 GPU
```

### 128 GPUs (Wind Scanning Disabled)
```yaml
train:
  devices: 8
  num_nodes: 16  # 128 GPUs
  batch_size: 2
  accumulate_grad_batches: 4
  num_workers: 0
  strategy: ddp
  find_unused_parameters: false

model:
  parallel_patch_embed: false  # MUST disable for DDP
```

---

## Lessons Learned

1. **MIOpen cache must be per-node** on multi-node systems to avoid SQLite locking
2. **DataLoader parameters must be conditional** on num_workers value
3. **Custom initialization in models breaks DDP** - must happen before DDP setup
4. **Test on 1 GPU first** to isolate model issues from distributed issues
5. **Wind scanning + DDP is fundamentally incompatible** in current implementation
6. **Core dumps should always be in .gitignore**
7. **Large-scale jobs (400 GPUs) have extreme initialization overhead** - start smaller

---

## Open Questions

1. How to make wind scanner DDP-compatible without complete refactor?
2. Should we pre-compute all wind caches offline as static files?
3. Is FSDP a better strategy than DDP for this architecture?
4. Can we use Lightning's `setup()` hook to initialize wind cache before DDP?

---

## Issue 6: DDP Wind Scanning Fixed! ✅

### Solution: Pre-computed Wind Scanner Cache

**The breakthrough:** Instead of computing wind scanner cache at runtime (causing DDP deadlock), pre-compute it once and save to disk. All DDP ranks load the same cache file → no synchronization needed!

### Implementation Steps

#### 1. Pre-compute cache offline
```python
# scripts/precompute_wind_cache_ddp.py
scanner = CachedWindScanning(grid_h=64, grid_w=128, num_sectors=16)
cache_data = {
    'grid_h': scanner.grid_h,
    'grid_w': scanner.grid_w,
    'sector_angles': scanner.sector_angles,
    'cached_orders': scanner.cached_orders,
    'regional_cached_orders': scanner.regional_cached_orders,
}
with open('wind_scanner_cache.pkl', 'wb') as f:
    pickle.dump(cache_data, f)
```

#### 2. Modify wind scanner to load from disk
```python
# src/wind_scanning_cached.py
def __init__(self, grid_h, grid_w, num_sectors=16, cache_path=None):
    if cache_path and self._load_cache_from_disk(cache_path):
        print(f"✅ Loaded pre-computed wind scanner cache")
        return
    # Fallback: compute on-the-fly (single-GPU only)
    self._precompute_all_orders()
```

#### 3. Update model to use cached version
```python
# src/climax_core/parallelpatchembed_wind.py
def _ensure_wind_scanner(self, L_expected, num_sectors=16):
    cache_path = '/scratch/.../wind_scanner_cache.pkl'
    self.wind_scanner = CachedWindScanning(
        self.grid_h, self.grid_w,
        num_sectors=num_sectors,
        cache_path=cache_path  # DDP-safe!
    )
```

### Results

**✅ 400 GPUs (50 nodes × 8 GPUs) - WORKING!**
- Wind scanning: **ENABLED**
- Elevation bias: **ENABLED**
- Sanity check: 39s (passed)
- Training: Step 1-10 completed, loss decreasing (5.26 → 3.5)
- **No deadlock!**

Job ID: 13268747
Config: `configs/config_wind_400gpu.yaml`

### Cache Details
- File size: 6.34 MB
- Contains: 16 sectors × 1024 regions pre-computed orders
- Load time: < 1 second per rank
- No network synchronization needed!

---

## Timeline Summary

- **Initial attempts:** 400 GPUs, 80 GPUs → all failed/hung
- **Debugging phase:** 16 GPUs, 8 GPUs, 1 GPU
- **Root cause identified:** Wind scanner + DDP deadlock
- **Solution implemented:** Pre-computed wind scanner cache (DDP-safe)
- **✅ FINAL STATUS:** 400 GPUs training WITH wind scanning + elevation bias

---

## Final Working Configuration

### 400 GPUs (PRODUCTION)
```yaml
# configs/config_wind_400gpu.yaml
train:
  devices: 8
  num_nodes: 50  # 400 GPUs
  batch_size: 2
  accumulate_grad_batches: 2  # Effective batch = 1600
  learning_rate: 0.000200
  strategy: ddp

model:
  parallel_patch_embed: true  # ✅ Wind scanning ENABLED with cache!
```

**Pre-requisite:** Run `python scripts/precompute_wind_cache_ddp.py` once to generate `wind_scanner_cache.pkl`

---

**Date:** 2025-10-01
**System:** LUMI-G (AMD MI250X, ROCm 5.x, PyTorch 2.x)
**Framework:** PyTorch Lightning 2.x with DDP strategy
**Final Job:** 13268747 (400 GPUs, wind scanning + elevation bias active)

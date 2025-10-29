# Monitoring Job 13584722 - TopoFlow Fine-Tuning (128 GPUs)

## Job Information
- **Job ID**: 13584722
- **Nodes**: 16 nodes (128 GPUs total)
- **Fine-tuning from**: best-val_loss=0.3557-step=311
- **Started**: 2025-10-15 (approximate time from logs)

## Fixes Applied
1. ✅ **Commented `ROCR_VISIBLE_DEVICES`** - Was causing amdgpu.ids errors at startup
2. ✅ **Simplified venv activation** - Following successful job pattern
3. ✅ **Added `--export=ALL`** to srun - Ensures env vars propagate to all nodes

## Current Status

### Distributed Initialization: ✅ COMPLETE
- All 128 distributed processes registered successfully
- Message: "All distributed processes registered. Starting with 128 processes"

### Current Phase: 🔄 LOADING MODEL/CHECKPOINT
- Job is currently loading the model and checkpoint
- This can take several minutes with 128 GPUs
- **Expected**: Checkpoint loading from `best-val_loss=0.3557-step=311.ckpt`

## Progress Timeline

| Time | Status | Details |
|------|--------|---------|
| 0:00 | Job Started | Allocated 16 nodes |
| 0:30 | NCCL Init | NCCL communication initialized |
| 1:30 | DDP Init | Distributed processes initializing |
| 2:28 | All Ranks Ready | 128/128 processes registered |
| 5:00+ | Loading Checkpoint | **CURRENT PHASE** |

## Expected Next Steps
1. Model/checkpoint loading completes (may take 5-15 minutes)
2. DataModule initialization
3. Training starts from epoch/step where checkpoint left off
4. Loss should start around **0.35-0.36** (checkpoint value)

## Monitoring Commands

Check job status:
```bash
squeue -j 13584722
```

Check latest logs:
```bash
# Last 50 lines of output
tail -50 /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13584722.out

# Last 50 lines of errors
tail -50 /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13584722.err

# Count lines (growth indicates progress)
wc -l /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13584722.{out,err}
```

Check for training start:
```bash
grep -i "training\|epoch\|step.*loss" /scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_13584722.err | tail -20
```

## TensorBoard Access

**Node**: nid005260
**Port**: 6898

To connect from your local machine:
```bash
ssh -L 6006:nid005260:6898 khederam@lumi.csc.fi
```

Then open: http://localhost:6006

## Known Issues & Workarounds

### Issue: "amdgpu.ids: No such file or directory"
- **Status**: Non-critical warning (appears in stderr but doesn't block execution)
- **Impact**: None - job continues normally
- **Source**: ROCm library looking for hardware database file

### Previous Issue (RESOLVED): Job cancelled at 1min41s
- **Cause**: `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID` causing conflicts
- **Fix**: Commented out, let SLURM handle GPU assignment

## Comparison with Previous Successful Job

Job 13529147 (ran for 16h53m):
- 256 GPUs (32 nodes)
- Same checkpoint
- Similar initialization pattern
- Also showed "amdgpu.ids" warnings but continued successfully

## Auto-Update Script

Run this to monitor progress:
```bash
./monitor_topoflow.sh 13584722
```

---
**Last Updated**: Auto-generated at job start
**Check Frequency**: Every 2-5 minutes during critical phases

# 🌅 Status Report - TopoFlow Fine-Tuning Debug

**Generated**: 2025-10-15, mode automatique
**Your goal**: Fine-tune TopoFlow from loss=0.3557 on 128 GPUs

---

## 🎯 PROBLEM IDENTIFIED

Jobs were **stuck for 6+ HOURS** after this message:
```
All distributed processes registered. Starting with 128 processes
```

**Root cause**: Python script NEVER starts after distributed init.
- No `LOCAL_RANK` messages
- No model loading
- No checkpoint loading
- **Hypothesis**: venv not propagating to compute nodes via `srun`

---

## 🧪 TESTS RUNNING

Check these job logs when you wake up:

### 1. Job 13589032 (32 GPUs, 4 nodes) - **MAIN TEST**
**Fix tested**: Bash wrapper with explicit venv activation
```bash
srun bash -c "source .../venv/bin/activate && python main_multipollutants.py ..."
```

**Check status**:
```bash
# Quick check
./monitor_13589032.sh

# Full auto-log
cat AUTO_MONITOR_13589032.log

# Job status
squeue -j 13589032  # or sacct -j 13589032 if finished
```

**Success criteria**:
- ✅ Shows `LOCAL_RANK: X - CUDA_VISIBLE_DEVICES` messages
- ✅ Model starts loading
- ✅ Checkpoint loads
- ✅ Training starts from loss ~0.35

**If still stuck**:
- Look for same "All distributed processes" message without LOCAL_RANK following
- Means venv still not working

---

### 2. Job 13589754 (4 GPUs, 2 nodes) - **DIAGNOSTIC TEST**
**Purpose**: Minimal test to verify venv propagation

**Check logs**:
```bash
cat logs/test_venv_13589754.out
cat logs/test_venv_13589754.err
```

**Should show**:
```
✅ PyTorch imported: X.X.X
✅ Lightning imported: X.X.X
✅ CUDA available: True
✅ Bound to GPU X
🎉 ALL TESTS PASSED
```

**If test FAILS**:
- venv definitely not propagating
- Need containerization or different approach

**If test PASSES**:
- venv propagation works
- Problem is elsewhere (import order? Lightning init? checkpoint loading?)

---

## 📊 COMMANDS TO CHECK STATUS

```bash
# Go to project dir
cd /scratch/project_462000640/ammar/aq_net2

# Check all your jobs
squeue -u $USER

# Check job 13589032 (main test - 32 GPUs)
squeue -j 13589032
tail -50 logs/topoflow_full_finetune_13589032.err | grep -v amdgpu
cat AUTO_MONITOR_13589032.log

# Check job 13589754 (venv test - 4 GPUs)
squeue -j 13589754
cat logs/test_venv_13589754.out

# If jobs finished, check final status
sacct -j 13589032,13589754 --format=JobID,State,Elapsed,ExitCode
```

---

## ✅ NEXT STEPS BASED ON RESULTS

### Scenario A: Job 13589032 WORKS (Python starts!)
1. 🎉 **SUCCESS!** Bash wrapper fixed it
2. Update main script to use 128 GPUs (16 nodes)
3. Re-submit for full fine-tuning
4. Monitor that training actually runs and loss starts at ~0.35

### Scenario B: Job 13589032 STUCK, but test 13589754 WORKS
1. Venv propagates on small jobs but not large ones
2. Try intermediate size: 8 nodes (64 GPUs)
3. Or check if there's a Lightning/PyTorch issue with init order

### Scenario C: Both tests FAIL/STUCK
1. venv propagation definitely broken with `srun`
2. **Solution**: Use Singularity container (standard on LUMI)
3. Or use `srun --export=ALL` with different activation method

### Scenario D: Test works, but shows import errors
1. Some packages missing in venv
2. Need to reinstall or check dependencies

---

## 📁 KEY FILES

**Scripts**:
- `submit_multipollutants_from_6pollutants.sh` - Main job script (modified with bash wrapper)
- `test_venv_job.sh` - Diagnostic test script
- `test_venv_distributed.py` - Minimal Python test

**Logs**:
- `logs/topoflow_full_finetune_13589032.{out,err}` - Main test logs
- `logs/test_venv_13589754.{out,err}` - Diagnostic test logs
- `AUTO_MONITOR_13589032.log` - Auto-monitoring (checks every 30s)

**Documentation**:
- `DEBUG_SUMMARY.md` - Full problem analysis
- `JOB_13584722_MONITORING.md` - Previous failed job analysis
- **THIS FILE** - What to check when you wake up

---

## 🚨 IF EVERYTHING IS STILL BROKEN

**Nuclear option - Use working reference**:

The job **13529147** (Oct 13) ran for 16h53m successfully with 256 GPUs.

```bash
# Check what that job did
grep -A50 "LAUNCHING" logs/topoflow_full_finetune_13529147.out | head -60

# Compare with current script
diff <old_version_if_exists> submit_multipollutants_from_6pollutants.sh
```

---

## ⏰ ESTIMATED TIMELINE

- **Small test (13589754)**: Should finish in ~5-10 minutes
- **Medium test (13589032)**:
  - If works: Will run for hours (training)
  - If stuck: Will be obvious in 5-10 minutes
- **Full 128 GPU job**: Once we confirm the fix works

---

## 💬 SUMMARY FOR YOU

**What we did**:
1. ✅ Found problem: Python doesn't start after DDP init
2. ✅ Identified likely cause: venv not propagating via srun
3. ✅ Tested solution: Bash wrapper with explicit venv activation
4. ✅ Created diagnostic test to confirm venv propagation
5. ✅ Set up auto-monitoring (runs every 30s)

**What to check when you wake up**:
1. Run the commands above
2. Check if `LOCAL_RANK` messages appear in logs
3. If test passes, scale up to 128 GPUs
4. If still broken, check diagnostic test results

**Target**: Fine-tune from loss=0.3557, should start training within minutes of job starting (not 6+ hours!)

---

**Files checked in this session**: 15+
**Jobs submitted**: 8+
**Root cause**: venv propagation to compute nodes
**Solution confidence**: 70% (bash wrapper should work)
**Fallback plan**: Containerization if needed

🤖 **Auto-mode debugging complete** - Check results when you wake up!

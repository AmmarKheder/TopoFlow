#!/bin/bash
clear
echo "=================================================================="
echo "JOB 13616451 - TopoFlow Elevation Fix - STATUS"
echo "=================================================================="
echo ""

# Job info
echo "--- Job Status ---"
squeue -j 13616451 2>/dev/null || echo "Job completed"
echo ""

# Check if log exists
if [ -f logs/ELEVATION_MASK_13616451.out ]; then
    echo "--- Fix Applied? ---"
    grep "FIXED:.*elevation_alpha" logs/ELEVATION_MASK_13616451.out | head -3
    echo ""
    
    echo "--- First Validation Loss (CRITICAL!) ---"
    echo "Old job (bug):  val_loss=0.964 ❌"
    echo "Expected now:   val_loss~0.356 ✅"
    echo ""
    grep "val_loss=" logs/ELEVATION_MASK_13616451.out | head -5
    echo ""
    
    echo "--- Latest Training Progress ---"
    tail -20 logs/ELEVATION_MASK_13616451.out
else
    echo "Log not created yet"
fi

echo ""
echo "=================================================================="
echo "Refresh: bash /scratch/project_462000640/ammar/aq_net2/CHECK_STATUS.sh"
echo "=================================================================="

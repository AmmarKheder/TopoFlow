#!/bin/bash
echo "=========================================="
echo "MONITORING JOB 13785881 - MANUAL OPTIMIZER LOADING TEST"
echo "=========================================="
echo ""

# Check job status
echo "=== JOB STATUS ==="
squeue -j 13785881 -o "%.18i %.12P %.30j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "Job not found in queue"
echo ""

# Check if log exists
if [ -f logs/ELEVATION_MASK_13785881.out ]; then
    echo "=== LOG FILE SIZE ==="
    wc -l logs/ELEVATION_MASK_13785881.out
    echo ""
    
    echo "=== CHECKPOINT LOADING HOOK ==="
    grep -A 5 "on_load_checkpoint HOOK" logs/ELEVATION_MASK_13785881.out | head -10
    echo ""
    
    echo "=== MANUAL OPTIMIZER LOADING ==="
    grep -A 15 "MANUAL OPTIMIZER STATE LOADING" logs/ELEVATION_MASK_13785881.out
    echo ""
    
    echo "=== TRAIN LOSS VALUES ==="
    grep "train_loss=" logs/ELEVATION_MASK_13785881.out | head -15
    echo ""
    
    echo "=== LATEST OUTPUT ==="
    tail -20 logs/ELEVATION_MASK_13785881.out
else
    echo "❌ Log file not created yet"
    echo "Waiting for job to start..."
fi

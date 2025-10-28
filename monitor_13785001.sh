#!/bin/bash
echo "=========================================="
echo "Job 13785001 - TEST 50 NODES (2h)"
echo "=========================================="
squeue -j 13785001 2>/dev/null || echo "Job not in queue"
echo ""
LOG="/scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_13785001.out"
ERR="/scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_13785001.err"

if [ -f "$LOG" ]; then
    echo "on_load_checkpoint hook:"
    grep -A 10 "on_load_checkpoint HOOK" "$LOG" | head -15
    echo ""
    echo "Optimizer config:"
    grep -A 4 "Fine-tuned optimizer configuration" "$LOG" | head -6
    echo ""
    echo "🔥 TRAIN LOSS (devrait être ~0.7 si RESUME marche):"
    grep "train_loss" "$LOG" | head -15
    echo ""
    echo "Latest log:"
    tail -20 "$LOG"
else
    echo "Log not created yet"
fi

if [ -f "$ERR" ]; then
    echo ""
    echo "⚠️ ERRORS:"
    tail -30 "$ERR" | grep -i "error\|nccl\|abort" | head -10
fi

#!/bin/bash
echo "=========================================="
echo "Job 13783436 - CORRECT LEARNING RATE"
echo "=========================================="
squeue -j 13783436 2>/dev/null || echo "Job completed/not in queue"
echo ""
LOG="/scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_13783436.out"
if [ -f "$LOG" ]; then
    echo "on_load_checkpoint hook:"
    grep -A 10 "on_load_checkpoint HOOK" "$LOG" | head -15
    echo ""
    echo "Optimizer config (should show base_lr=0.0001):"
    grep -A 4 "Fine-tuned optimizer configuration" "$LOG" | head -6
    echo ""
    echo "Train loss (should start at ~0.35-0.50):"
    grep "train_loss" "$LOG" | tail -10
else
    echo "Log not created yet"
fi

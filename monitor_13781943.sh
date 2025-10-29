#!/bin/bash
echo "=================================="
echo "Monitoring Job 13781943"
echo "=================================="
echo ""
echo "Job Status:"
squeue -j 13781943 -o "%.10i %.9P %.50j %.8T %.10M %.6D %R" 2>/dev/null || echo "Job not in queue (completed or failed)"
echo ""

LOG_FILE="/scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_13781943.out"

if [ -f "$LOG_FILE" ]; then
    echo "Latest output from log:"
    echo "=================================="
    tail -50 "$LOG_FILE"
    echo ""
    echo "=================================="
    echo "Checking for on_load_checkpoint hook:"
    grep -A 5 "on_load_checkpoint HOOK CALLED" "$LOG_FILE" | head -20 || echo "Hook not called yet"
    echo ""
    echo "Latest train_loss values:"
    grep "train_loss" "$LOG_FILE" | tail -10 || echo "No train_loss yet"
else
    echo "Log file not created yet: $LOG_FILE"
fi

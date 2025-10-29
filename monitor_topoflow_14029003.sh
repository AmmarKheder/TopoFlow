#!/bin/bash
# Monitoring script for TopoFlow job 14029003

LOG_FILE="/scratch/project_462000640/ammar/aq_net2/logs/TOPOFLOW_BLOCK0_14029003.out"

echo "=============================================="
echo "📊 MONITORING TOPOFLOW JOB 14029003"
echo "=============================================="
echo ""

# Check job status
echo "🔍 Job Status:"
squeue -u khederam | grep -E "JOBID|14029003"
echo ""

# Show important log lines
echo "📝 Recent Activity:"
echo ""

if [ -f "$LOG_FILE" ]; then
    # Show model initialization
    echo "--- Model Info ---"
    grep -E "Target variables|Block 0|PhysicsGuidedBlock|alpha|use_physics_mask" "$LOG_FILE" | tail -10
    echo ""

    # Show checkpoint loading
    echo "--- Checkpoint Loading ---"
    grep -E "Loading checkpoint|checkpoint from epoch|val_loss=" "$LOG_FILE" | tail -5
    echo ""

    # Show training progress
    echo "--- Training Progress ---"
    grep -E "Epoch|train_loss|val_loss" "$LOG_FILE" | tail -20
    echo ""

    # Show any errors
    echo "--- Recent Errors (if any) ---"
    grep -i "error\|exception\|failed" "$LOG_FILE" | tail -5
    echo ""

    echo "--- Last 10 lines of log ---"
    tail -10 "$LOG_FILE"
else
    echo "⚠️  Log file not found yet"
fi

echo ""
echo "=============================================="
echo "To see live updates: tail -f $LOG_FILE"
echo "=============================================="

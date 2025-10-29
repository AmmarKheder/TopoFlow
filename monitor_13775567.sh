#!/bin/bash
# Monitor job 13775567 (RESUME mode job)

JOB_ID=13775567

echo "=============================================="
echo "📊 MONITORING JOB $JOB_ID (RESUME MODE)"
echo "=============================================="

# Job status
echo ""
echo "=== Job Status ==="
squeue -j $JOB_ID 2>/dev/null || echo "⚠️  Job not in queue (completed, failed, or cancelled)"

# Check if logs exist
if [ -f "/scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.out" ]; then
    echo ""
    echo "=== Last 30 lines of output ==="
    tail -30 /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.out

    echo ""
    echo "=== Training progress (if any) ==="
    grep -E "Epoch|train_loss|val_loss|step" /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.out | tail -10 || echo "No training progress yet"

    echo ""
    echo "=== Check for RESUME message ==="
    grep -A 3 "RESUME training" /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.out || echo "RESUME message not found yet"

    echo ""
    echo "=== Recent errors (excluding amdgpu warnings) ==="
    tail -50 /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.err 2>/dev/null | grep -v "amdgpu.ids" | grep -E "Error|Exception|Traceback|Failed" | tail -5 || echo "No recent errors"

    echo ""
    echo "=== File sizes ==="
    ls -lh /scratch/project_462000640/ammar/aq_net2/logs/ELEVATION_MASK_${JOB_ID}.{out,err} 2>/dev/null
else
    echo ""
    echo "⏳ Logs not created yet - job hasn't started"
    echo "   Estimated start time: Check with 'squeue --start -j $JOB_ID'"
fi

echo ""
echo "=============================================="
echo "To watch continuously: watch -n 30 $0"
echo "=============================================="

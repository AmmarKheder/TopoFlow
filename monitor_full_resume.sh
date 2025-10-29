#!/bin/bash
# Monitor full 256 GPU resume job 13968302

JOB_ID=13968302
LOG_OUT="logs/ELEVATION_MASK_${JOB_ID}.out"
LOG_ERR="logs/ELEVATION_MASK_${JOB_ID}.err"

echo "=========================================="
echo "🚀 MONITORING FULL RESUME - 256 GPUs"
echo "Job: $JOB_ID"
echo "=========================================="
echo ""

# Check job status
echo "📊 Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job not in queue (may have finished or not started)"
echo ""

# Check output log
if [ -f "$LOG_OUT" ]; then
    echo "📝 Output log (last 30 lines):"
    tail -30 "$LOG_OUT"
    echo ""

    # Check for training
    if grep -q "train_loss=" "$LOG_OUT" 2>/dev/null; then
        echo "=========================================="
        echo "✅✅✅ TRAINING IS RUNNING! ✅✅✅"
        echo "=========================================="
        echo ""
        echo "Recent losses:"
        grep "train_loss=" "$LOG_OUT" | tail -10
    else
        echo "⏳ Waiting for training to start..."
    fi
else
    echo "⏳ Output log not created yet"
fi

echo ""
echo "=========================================="

# Check for errors
if [ -f "$LOG_ERR" ]; then
    if grep -q "error\|Error\|ERROR\|Segmentation\|pthread" "$LOG_ERR" 2>/dev/null; then
        echo "⚠️  Errors detected in log:"
        tail -20 "$LOG_ERR"
    fi
fi

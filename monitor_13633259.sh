#!/bin/bash
# Monitoring script for job 13633742 (test_val_loss_gpu)

JOB_ID=13633742
LOG_OUT="logs/test_val_loss_gpu_${JOB_ID}.out"
LOG_ERR="logs/test_val_loss_gpu_${JOB_ID}.err"

echo "========================================"
echo "🔍 MONITORING TEST VAL LOSS GPU - JOB ${JOB_ID}"
echo "========================================"
echo ""

# Check job status
echo "📊 Job Status:"
squeue -j ${JOB_ID}
echo ""

# Check if job is running
JOB_STATE=$(squeue -j ${JOB_ID} -h -o "%T" 2>/dev/null)

if [ -z "$JOB_STATE" ]; then
    echo "✅ Job completed or not in queue"
    echo ""

    if [ -f "$LOG_OUT" ]; then
        echo "========================================"
        echo "📊 FULL OUTPUT:"
        echo "========================================"
        cat "$LOG_OUT"
        echo ""
    else
        echo "⚠️  Output log not found: $LOG_OUT"
    fi

    if [ -f "$LOG_ERR" ]; then
        echo "========================================"
        echo "❌ ERRORS (if any):"
        echo "========================================"
        cat "$LOG_ERR"
    fi

elif [ "$JOB_STATE" == "RUNNING" ]; then
    echo "✅ Job is RUNNING"
    echo ""

    if [ -f "$LOG_OUT" ]; then
        echo "📊 Current output:"
        cat "$LOG_OUT"
    else
        echo "⏳ Log file not created yet..."
    fi

elif [ "$JOB_STATE" == "PENDING" ]; then
    echo "⏳ Job is PENDING (waiting for GPU)"

else
    echo "ℹ️  Job state: $JOB_STATE"
fi

echo ""
echo "========================================"
echo "⏰ Checked at: $(date)"
echo "========================================"

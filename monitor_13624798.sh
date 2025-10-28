#!/bin/bash
# Monitoring script for job 13624798

JOB_ID=13624798
LOG_OUT="logs/ELEVATION_MASK_${JOB_ID}.out"
LOG_ERR="logs/ELEVATION_MASK_${JOB_ID}.err"

echo "========================================"
echo "🔍 MONITORING JOB ${JOB_ID}"
echo "========================================"
echo ""

# Check job status
echo "📊 Job Status:"
squeue -j ${JOB_ID}
echo ""

# Check if job is running
JOB_STATE=$(squeue -j ${JOB_ID} -h -o "%T" 2>/dev/null)

if [ -z "$JOB_STATE" ]; then
    echo "⚠️  Job not found in queue (may have completed or failed)"
    echo ""

    # Check if logs exist
    if [ -f "$LOG_OUT" ]; then
        echo "📄 Last 50 lines of output log:"
        tail -50 "$LOG_OUT"
    fi

    if [ -f "$LOG_ERR" ]; then
        echo ""
        echo "❌ Last 30 lines of error log:"
        tail -30 "$LOG_ERR"
    fi

elif [ "$JOB_STATE" == "RUNNING" ]; then
    echo "✅ Job is RUNNING"
    echo ""

    # Show recent output
    if [ -f "$LOG_OUT" ]; then
        echo "📊 Last 30 lines of training output:"
        tail -30 "$LOG_OUT"
        echo ""

        # Check for train_loss
        echo "🔥 Recent train_loss values:"
        grep "train_loss" "$LOG_OUT" | tail -10
        echo ""

        # Check for val_loss
        echo "📈 Recent val_loss values:"
        grep "val_loss" "$LOG_OUT" | tail -5
    else
        echo "⏳ Log file not created yet..."
    fi

elif [ "$JOB_STATE" == "PENDING" ]; then
    echo "⏳ Job is PENDING (waiting for resources)"
    echo ""
    squeue -j ${JOB_ID} -o "%.18i %.9P %.30j %.8u %.8T %.10M %.6D %R"

else
    echo "ℹ️  Job state: $JOB_STATE"
fi

echo ""
echo "========================================"
echo "⏰ Checked at: $(date)"
echo "========================================"

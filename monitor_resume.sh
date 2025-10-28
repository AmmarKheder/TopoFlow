#!/bin/bash
# Monitor resume training job 13966538

JOB_ID=13966538
LOG_FILE="logs/RESUME_WIND_${JOB_ID}.out"

echo "=========================================="
echo "🔍 MONITORING RESUME TRAINING - Job $JOB_ID"
echo "=========================================="
echo ""

# Check job status
echo "📊 Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job not found (may have finished or not started)"
echo ""

# Check if training has started
if [ -f "$LOG_FILE" ]; then
    echo "📝 Last 30 lines of log:"
    tail -30 "$LOG_FILE"
    echo ""
    echo "=========================================="
    echo "🔍 Training Progress:"
    echo "=========================================="

    # Check for checkpoint loading
    if grep -q "Loaded checkpoint" "$LOG_FILE" 2>/dev/null; then
        echo "✅ Checkpoint loaded!"
    else
        echo "⏳ Waiting for checkpoint to load..."
    fi

    # Check for training start
    if grep -q "train_loss=" "$LOG_FILE" 2>/dev/null; then
        echo "✅ Training started!"
        echo ""
        echo "Recent losses:"
        grep "train_loss=" "$LOG_FILE" | tail -5
    else
        echo "⏳ Training not started yet..."
    fi

else
    echo "⚠️  Log file not found: $LOG_FILE"
fi

echo ""
echo "=========================================="
echo "To monitor continuously: watch -n 10 ./monitor_resume.sh"
echo "=========================================="

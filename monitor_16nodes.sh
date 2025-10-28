#!/bin/bash
# Monitor 16 nodes test job 13968053

JOB_ID=13968053
LOG_FILE="logs/TEST_RESUME_16NODES_${JOB_ID}.out"

echo "=========================================="
echo "🔍 MONITORING 16 NODES TEST - Job $JOB_ID"
echo "=========================================="
echo ""

# Check job status
echo "📊 Job Status:"
squeue -j $JOB_ID 2>/dev/null || echo "Job not found (may have finished or not started)"
echo ""

# Check if training has started
if [ -f "$LOG_FILE" ]; then
    echo "📝 Last 40 lines of log:"
    tail -40 "$LOG_FILE"
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
echo "To monitor continuously: watch -n 10 ./monitor_16nodes.sh"
echo "=========================================="

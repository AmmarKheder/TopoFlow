#!/bin/bash
# Monitoring script for job 13633855 (test_val_loss_gpu - 4 nodes × 8 GPUs)

JOB_ID=13633855
LOG_OUT="logs/test_val_loss_gpu_${JOB_ID}.out"
LOG_ERR="logs/test_val_loss_gpu_${JOB_ID}.err"

echo "========================================"
echo "TEST VAL LOSS - JOB ${JOB_ID} (32 GPUs)"
echo "========================================"
echo ""

# Check job status
echo "Job Status:"
squeue -j ${JOB_ID}
echo ""

# Check if job is running
JOB_STATE=$(squeue -j ${JOB_ID} -h -o "%T" 2>/dev/null)

if [ -z "$JOB_STATE" ]; then
    echo "Job completed or not in queue"
    echo ""

    if [ -f "$LOG_OUT" ]; then
        echo "========================================"
        echo "FULL OUTPUT:"
        echo "========================================"
        cat "$LOG_OUT"
        echo ""
    else
        echo "Output log not found: $LOG_OUT"
    fi

    if [ -f "$LOG_ERR" ]; then
        echo "========================================"
        echo "ERRORS (if any):"
        echo "========================================"
        cat "$LOG_ERR"
    fi

elif [ "$JOB_STATE" == "RUNNING" ]; then
    echo "Job is RUNNING"
    echo ""

    if [ -f "$LOG_OUT" ]; then
        echo "Current output (last 50 lines):"
        tail -50 "$LOG_OUT"
    else
        echo "Log file not created yet..."
    fi

elif [ "$JOB_STATE" == "PENDING" ]; then
    echo "Job is PENDING (waiting for nodes)"

else
    echo "Job state: $JOB_STATE"
fi

echo ""
echo "========================================"
echo "Checked at: $(date)"
echo "========================================"

#!/bin/bash
# Wait for job 13633855 to complete and show results

JOB_ID=13633855
LOG_OUT="logs/test_val_loss_gpu_${JOB_ID}.out"

echo "Waiting for job ${JOB_ID} to complete..."
echo ""

while true; do
    # Check if job is still in queue
    JOB_STATE=$(squeue -j ${JOB_ID} -h -o "%T" 2>/dev/null)

    if [ -z "$JOB_STATE" ]; then
        # Job finished
        echo "Job completed at $(date)"
        echo ""

        if [ -f "$LOG_OUT" ]; then
            echo "========================================"
            echo "RESULTS:"
            echo "========================================"
            cat "$LOG_OUT"
        else
            echo "Log file not found: $LOG_OUT"
        fi

        break
    elif [ "$JOB_STATE" == "RUNNING" ]; then
        echo "[$(date +%H:%M:%S)] Job is RUNNING..."
    else
        echo "[$(date +%H:%M:%S)] Job is $JOB_STATE..."
    fi

    sleep 30
done

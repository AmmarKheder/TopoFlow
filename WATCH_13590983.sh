#!/bin/bash
# Active monitoring of job 13590983 (128 GPUs)

JOB_ID=13590983
ERR_LOG="logs/topoflow_full_finetune_${JOB_ID}.err"
OUT_LOG="logs/topoflow_full_finetune_${JOB_ID}.out"

echo "========================================"
echo "🔍 MONITORING JOB $JOB_ID (128 GPUs)"
echo "========================================"
echo ""

for i in {1..30}; do
    echo "=== Check $i/30 at $(date +%H:%M:%S) ==="

    STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)

    if [ -z "$STATUS" ]; then
        echo "❌ Job not in queue"
        sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed
        break
    fi

    echo "Status: $STATUS"

    if [ "$STATUS" = "RUNNING" ]; then
        echo "✅ Job is RUNNING!"

        if [ -f "$ERR_LOG" ]; then
            LINES=$(wc -l < "$ERR_LOG")
            echo "Error log: $LINES lines"

            # Check for Python startup
            if grep -q "Bound process to cuda.*LOCAL_RANK\|D.*MARRAGE AQ_NET2" "$ERR_LOG" 2>/dev/null; then
                echo ""
                echo "🎉🎉🎉 PYTHON STARTED SUCCESSFULLY!"
                echo ""
                echo "First few LOCAL_RANK messages:"
                grep "Bound process to cuda.*LOCAL_RANK" "$ERR_LOG" | head -5
                echo ""
                echo "Job is working! Training will start soon."
                echo "Monitor progress with: tail -f $ERR_LOG"
                break
            elif grep -q "All distributed processes registered" "$ERR_LOG" 2>/dev/null; then
                echo "⏳ DDP initialized, waiting for Python to start..."
            else
                echo "⏳ Initializing distributed processes..."
            fi
        else
            echo "⏳ Log file not created yet..."
        fi
    fi

    sleep 30
done

echo ""
echo "========================================"
echo "Monitoring ended. Check logs manually:"
echo "  tail -f $ERR_LOG"
echo "  tail -f $OUT_LOG"
echo "========================================"

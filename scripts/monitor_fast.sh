#!/bin/bash

# Fast monitoring every 20 seconds
# Job IDs from round 4: 13257206-13257209

JOBS=(13257206 13257207 13257208 13257209)
NAMES=("Baseline" "Innovation1" "Innovation2" "Full_Model")

echo "=== TopoFlow Fast Monitor (20s intervals) ==="
echo "Jobs: ${JOBS[@]}"
echo "Started: $(date)"
echo ""

while true; do
    clear
    echo "=== TopoFlow Status - $(date) ==="
    echo ""

    # Check job status
    for i in "${!JOBS[@]}"; do
        JOB_ID=${JOBS[$i]}
        NAME=${NAMES[$i]}

        STATUS=$(squeue -j $JOB_ID --format="%T" --noheader 2>/dev/null | head -1)

        if [ -z "$STATUS" ]; then
            # Job not in queue - check if completed or failed
            LOG_FILE="logs/topoflow_wind_*${JOB_ID}.out"
            ERR_FILE="logs/topoflow_wind_*${JOB_ID}.err"

            if ls logs/*${JOB_ID}.out 2>/dev/null | grep -q .; then
                LAST_LINE=$(ls logs/*${JOB_ID}.out | xargs tail -1 2>/dev/null)
                if [[ "$LAST_LINE" == *"completed"* ]]; then
                    echo "[$NAME] ✓ COMPLETED"
                else
                    echo "[$NAME] ✗ FAILED - checking logs..."

                    # Check for errors
                    if ls logs/*${JOB_ID}.err 2>/dev/null | grep -q .; then
                        ERROR=$(ls logs/*${JOB_ID}.err | xargs tail -5 2>/dev/null | grep -E "Error|Traceback" | head -1)
                        if [ -n "$ERROR" ]; then
                            echo "  Error: ${ERROR:0:100}"
                        fi
                    fi
                fi
            else
                echo "[$NAME] ⏳ PENDING/STARTING"
            fi
        else
            echo "[$NAME] ▶ $STATUS"

            # Show latest training progress if available
            if ls logs/*${JOB_ID}.out 2>/dev/null | grep -q .; then
                PROGRESS=$(ls logs/*${JOB_ID}.out | xargs grep -E "Epoch|val_loss" 2>/dev/null | tail -1)
                if [ -n "$PROGRESS" ]; then
                    echo "  ${PROGRESS:0:120}"
                fi
            fi
        fi
        echo ""
    done

    echo "---"
    echo "Next check in 20s... (Ctrl+C to stop)"
    sleep 20
done
#!/bin/bash
# Auto-monitor script for job 13589032

JOB_ID=13589032
LOG_FILE="/scratch/project_462000640/ammar/aq_net2/AUTO_MONITOR_${JOB_ID}.log"

echo "========================================" | tee -a $LOG_FILE
echo "Auto-monitoring Job $JOB_ID" | tee -a $LOG_FILE
echo "Started: $(date)" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

# Monitor for max 30 minutes (60 iterations x 30 seconds)
for i in {1..60}; do
    echo "" >> $LOG_FILE
    echo "=== Check $i/60 at $(date +%T) ===" >> $LOG_FILE

    # Check job status
    JOB_STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)

    if [ -z "$JOB_STATUS" ]; then
        echo "❌ Job $JOB_ID finished or cancelled" | tee -a $LOG_FILE
        sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed >> $LOG_FILE
        break
    fi

    echo "Status: $JOB_STATUS" >> $LOG_FILE

    if [ "$JOB_STATUS" = "RUNNING" ]; then
        # Check if logs exist and have new content
        ERR_FILE="/scratch/project_462000640/ammar/aq_net2/logs/topoflow_full_finetune_${JOB_ID}.err"

        if [ -f "$ERR_FILE" ]; then
            LINES=$(wc -l < "$ERR_FILE")
            echo "Error log lines: $LINES" >> $LOG_FILE

            # Check for key milestones
            if grep -q "LOCAL_RANK.*CUDA_VISIBLE_DEVICES" "$ERR_FILE" 2>/dev/null; then
                echo "✅✅✅ PYTHON STARTED! Checking logs..." | tee -a $LOG_FILE
                tail -30 "$ERR_FILE" >> $LOG_FILE
                echo "🎉 SUCCESS - Python is running!" | tee -a $LOG_FILE
                break
            elif grep -q "All distributed processes registered" "$ERR_FILE" 2>/dev/null; then
                echo "⏳ Distributed init done, waiting for Python to start..." >> $LOG_FILE
                # Check if stuck (same line count for 3 minutes)
                if [ ! -z "$PREV_LINES" ] && [ "$LINES" -eq "$PREV_LINES" ]; then
                    STUCK_COUNT=$((STUCK_COUNT + 1))
                    echo "⚠️  Stuck counter: $STUCK_COUNT/6" >> $LOG_FILE
                    if [ $STUCK_COUNT -ge 6 ]; then
                        echo "❌ JOB STUCK - No progress for 3 minutes after DDP init" | tee -a $LOG_FILE
                        tail -20 "$ERR_FILE" >> $LOG_FILE
                        break
                    fi
                else
                    STUCK_COUNT=0
                fi
            else
                echo "⏳ Still initializing distributed..." >> $LOG_FILE
            fi

            PREV_LINES=$LINES
        fi
    fi

    sleep 30
done

echo "" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE
echo "Monitoring ended: $(date)" | tee -a $LOG_FILE
echo "Check full log: $LOG_FILE" | tee -a $LOG_FILE
echo "========================================" | tee -a $LOG_FILE

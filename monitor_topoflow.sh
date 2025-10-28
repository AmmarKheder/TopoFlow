#!/bin/bash

# Get the latest topoflow job
JOB_ID=$(squeue -u $USER -n topoflow -h -o "%i" | head -1)

if [ -z "$JOB_ID" ]; then
    echo "No topoflow job found in queue"
    # Try to find the most recent log files
    LATEST_LOG=$(ls -t logs/topoflow_full_finetune_*.out 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        JOB_ID=$(basename $LATEST_LOG | sed 's/topoflow_full_finetune_//;s/.out//')
        echo "Found recent job: $JOB_ID (completed or failed)"
    fi
fi

if [ -z "$JOB_ID" ]; then
    echo "No logs found"
    exit 1
fi

LOG_OUT="logs/topoflow_full_finetune_${JOB_ID}.out"
LOG_ERR="logs/topoflow_full_finetune_${JOB_ID}.err"

clear
echo "============================================================"
echo "🔥 TOPOFLOW TRAINING MONITOR - Job $JOB_ID"
echo "============================================================"
date
echo ""

# Job status
STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
if [ -n "$STATUS" ]; then
    echo "📊 Status: $STATUS"
    squeue -j $JOB_ID
else
    echo "📊 Status: COMPLETED or FAILED"
    sacct -j $JOB_ID --format=JobID,State,ExitCode,Elapsed -n | head -1
fi
echo ""

# Log info
echo "📁 Logs:"
ls -lh $LOG_OUT $LOG_ERR 2>/dev/null | awk '{print "  " $9 ": " $5}'
echo ""

# Check progress
echo "🔍 Training Progress:"
echo "----------------------------------------"

# Check if Python started
if grep -q "# # # #" $LOG_ERR 2>/dev/null; then
    echo "✅ Python started"
    grep "# # # #" $LOG_ERR | head -5
else
    echo "⏳ Waiting for Python to start..."
fi
echo ""

# Check checkpoint loading
if grep -q "Loading checkpoint" $LOG_ERR 2>/dev/null; then
    echo "✅ Checkpoint loading:"
    grep "Loading checkpoint\|Checkpoint loaded\|Missing keys" $LOG_ERR | head -10
else
    echo "⏳ Waiting for checkpoint loading..."
fi
echo ""

# Check training steps
TRAIN_LINES=$(grep -E "Epoch|train_loss|val_loss|step" $LOG_ERR 2>/dev/null | wc -l)
if [ $TRAIN_LINES -gt 0 ]; then
    echo "✅ Training in progress ($TRAIN_LINES updates):"
    grep -E "Epoch|train_loss|val_loss" $LOG_ERR | tail -20
else
    echo "⏳ Waiting for training to start..."
fi

echo ""
echo "============================================================"
echo "Run: watch -n 10 ./monitor_topoflow.sh"
echo "============================================================"
